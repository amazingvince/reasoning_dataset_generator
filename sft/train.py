#!/usr/bin/env python3
"""
Chess Reasoning SFT Training Script

Fine-tunes Qwen3-4B on chess reasoning traces with custom <uci_move> tokens.

Usage:
    python train.py                    # Use default config.yaml
    python train.py --config my.yaml   # Use custom config

Features:
    - Custom chat template with <uci_move> special tokens
    - Liger kernels + padding-free training for efficiency
    - Streaming dataset with eval holdout
    - Stockfish centipawn evaluation metrics
    - W&B logging + HF Hub checkpoint uploads
"""

import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Training configuration with sensible defaults."""
    
    # Model
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    
    # Dataset
    dataset_name: str = "amazingvince/chess-traces"
    eval_holdout_size: int = 500
    max_length: int = 8192
    
    # Training
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    max_steps: int = 10000
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    bf16: bool = True
    tf32: bool = True
    
    # Optimizations
    gradient_checkpointing: bool = True
    use_liger_kernel: bool = True
    padding_free: bool = True
    
    # Evaluation
    eval_steps: int = 500
    stockfish_depth: int = 10
    stockfish_eval_samples: int = 50
    eval_batch_size: int = 16
    eval_max_new_tokens: int = 2048
    
    # Saving
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    output_dir: str = "./chess_qwen3_4b_reasoning"
    
    # Hub
    push_to_hub: bool = True
    hub_model_id: str = "amazingvince/chess_qwen3_4b_reasoning"
    hub_strategy: str = "checkpoint"
    hub_private_repo: bool = True
    
    # W&B
    wandb_project: str = "chess-reasoning-sft"
    wandb_run_name: str = "chess-qwen3-4b-reasoning"
    
    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Helper to update config from nested dict
        def update_from_section(section_name: str, mappings: dict):
            if section_name in data:
                section = data[section_name]
                for yaml_key, config_attr in mappings.items():
                    if yaml_key in section:
                        setattr(config, config_attr, section[yaml_key])
        
        # Model section
        update_from_section("model", {
            "name": "model_name",
            "torch_dtype": "torch_dtype", 
            "attn_implementation": "attn_implementation",
        })
        
        # Dataset section
        update_from_section("dataset", {
            "name": "dataset_name",
            "eval_holdout_size": "eval_holdout_size",
            "max_length": "max_length",
        })
        
        # Training section
        update_from_section("training", {
            "per_device_train_batch_size": "per_device_train_batch_size",
            "per_device_eval_batch_size": "per_device_eval_batch_size",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "lr_scheduler_type": "lr_scheduler_type",
            "warmup_ratio": "warmup_ratio",
            "num_train_epochs": "num_train_epochs",
            "max_steps": "max_steps",
            "optim": "optim",
            "max_grad_norm": "max_grad_norm",
            "weight_decay": "weight_decay",
            "bf16": "bf16",
            "tf32": "tf32",
        })
        
        # Optimization section
        update_from_section("optimization", {
            "gradient_checkpointing": "gradient_checkpointing",
            "use_liger_kernel": "use_liger_kernel",
            "padding_free": "padding_free",
        })
        
        # Evaluation section
        if "evaluation" in data:
            e = data["evaluation"]
            if "eval_steps" in e:
                config.eval_steps = e["eval_steps"]
            if "batch_size" in e:
                config.eval_batch_size = e["batch_size"]
            if "max_new_tokens" in e:
                config.eval_max_new_tokens = e["max_new_tokens"]
            if "stockfish" in e:
                sf = e["stockfish"]
                if "depth" in sf:
                    config.stockfish_depth = sf["depth"]
                if "eval_samples" in sf:
                    config.stockfish_eval_samples = sf["eval_samples"]
        
        # Saving section
        update_from_section("saving", {
            "save_strategy": "save_strategy",
            "save_steps": "save_steps",
            "save_total_limit": "save_total_limit",
            "output_dir": "output_dir",
        })
        
        # Hub section
        update_from_section("hub", {
            "push_to_hub": "push_to_hub",
            "hub_model_id": "hub_model_id",
            "hub_strategy": "hub_strategy",
            "hub_private_repo": "hub_private_repo",
        })
        
        # W&B section
        if "wandb" in data:
            w = data["wandb"]
            if "project" in w:
                config.wandb_project = w["project"]
            if "run_name" in w:
                config.wandb_run_name = w["run_name"]
        
        # Logging section
        update_from_section("logging", {
            "logging_steps": "logging_steps",
            "logging_first_step": "logging_first_step",
        })
        
        return config


# =============================================================================
# Chat Template (simple prompt/completion format)
# =============================================================================

CHAT_TEMPLATE = """\
{%- for message in messages %}
{%- if message['role'] == 'user' %}
{{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
{{ message['content'] }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<think>
{%- endif %}
"""


# =============================================================================
# Setup
# =============================================================================

def find_stockfish() -> Optional[str]:
    """Locate a Stockfish binary on common paths."""
    candidates = [
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def setup_auth():
    """Setup HuggingFace and W&B authentication."""
    import wandb
    from huggingface_hub import login
    
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")
    
    if hf_token:
        login(token=hf_token)
        print("  ✓ HuggingFace authenticated")
    else:
        print("  ⚠ HF_TOKEN not set")
    
    if wandb_key:
        wandb.login(key=wandb_key)
        print("  ✓ W&B authenticated")
    else:
        print("  ⚠ WANDB_API_KEY not set")


def setup_tokenizer(model_name: str):
    """Load tokenizer with custom special tokens."""
    from transformers import AutoTokenizer
    
    print(f"\n🔤 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add UCI move tokens
    special_tokens = {"additional_special_tokens": ["<uci_move>", "</uci_move>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"  ✓ Added {num_added} special tokens")
    
    # Set chat template
    tokenizer.chat_template = CHAT_TEMPLATE
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def setup_model(model_name: str, tokenizer, config: Config):
    """Load model and resize embeddings."""
    from transformers import AutoModelForCausalLM
    
    print(f"\n🧠 Loading model: {model_name}")
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map.get(config.torch_dtype, torch.bfloat16),
        attn_implementation=config.attn_implementation,
        trust_remote_code=True,
    )
    
    # Resize for new tokens
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_size = model.get_input_embeddings().weight.shape[0]
    print(f"  ✓ Embeddings: {old_size} → {new_size}")
    
    return model


# =============================================================================
# Dataset
# =============================================================================

def format_example(example: dict) -> dict:
    """Format a chess example into prompt-completion format."""
    legal_moves = example["valid_moves"]
    if isinstance(legal_moves, list):
        legal_moves = ", ".join(legal_moves)
    
    prompt = f"""You are an expert chess player. Choose the best move.
FEN: {example['fen']}
Legal moves (UCI): {legal_moves}

Rules:
- Put all reasoning inside <think>...</think>.
- Output exactly one <uci_move>...</uci_move> tag with a single move copied from the legal moves list (no spaces).
- Do not output anything after the closing </uci_move>.
- Do not output "resign".

Output format:
<think>...</think>
<uci_move>...</uci_move>
"""
    
    completion = f"""<think>{example['reasoning_trace']}</think>
<uci_move>{example['chosen_move']}</uci_move>"""
    
    return {"prompt": prompt, "completion": completion}


def prepare_datasets(config: Config):
    """Load streaming dataset and create train/eval splits."""
    from datasets import Dataset, IterableDataset, load_dataset
    
    print(f"\n📊 Loading dataset: {config.dataset_name}")
    
    # Stream and collect samples for splitting
    dataset = load_dataset(config.dataset_name, split="train", streaming=True)
    buffer_size = config.eval_holdout_size + 5000
    
    samples = []
    for i, sample in enumerate(dataset):
        samples.append(sample)
        if i >= buffer_size - 1:
            break
        if (i + 1) % 1000 == 0:
            print(f"  Collected {i + 1} samples...")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    
    eval_samples = samples[:config.eval_holdout_size]
    train_samples = samples[config.eval_holdout_size:]
    
    print(f"  ✓ Eval: {len(eval_samples)}, Train buffer: {len(train_samples)}")
    
    # Format
    eval_formatted = [format_example(s) for s in eval_samples]
    train_formatted = [format_example(s) for s in train_samples]
    
    # Create datasets
    eval_dataset = Dataset.from_list(eval_formatted)
    
    def train_generator():
        yield from train_formatted
        stream = load_dataset(config.dataset_name, split="train", streaming=True)
        for i, sample in enumerate(stream):
            if i < buffer_size:
                continue
            yield format_example(sample)
    
    train_dataset = IterableDataset.from_generator(train_generator)
    
    return train_dataset, eval_dataset


# =============================================================================
# Stockfish Evaluation
# =============================================================================

class StockfishEngine:
    """Wrapper for Stockfish chess engine."""
    
    def __init__(self, path: Optional[str] = None, depth: int = 15):
        self.path = path
        self.depth = depth
        self.engine = None
        self.available = path is not None
    
    def start(self) -> bool:
        """Start the engine. Returns True if successful."""
        if not self.available:
            return False
        
        try:
            import chess.engine
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=30.0)
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to start Stockfish: {e}")
            self.available = False
            return False
    
    def stop(self):
        """Stop the engine."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None
    
    def analyze(self, fen: str, move_uci: Optional[str] = None) -> Optional[dict]:
        """
        Analyze a position. If move_uci is provided, analyze after that move.
        Returns dict with 'best_move' and 'score' (centipawns from side to move).
        """
        if not self.engine:
            return None
        
        try:
            import chess
            import chess.engine
            
            board = chess.Board(fen)
            
            # If move provided, check legality and apply
            if move_uci:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    return None
                board.push(move)
            
            # Get best move and score in one call
            result = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth),
                info=chess.engine.INFO_ALL
            )
            
            score = result["score"].relative
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
            
            best_move = None
            if "pv" in result and result["pv"]:
                best_move = result["pv"][0].uci()
            
            # If we pushed a move, return negated score (opponent's perspective)
            return {"best_move": best_move, "score": -cp if move_uci else cp}
            
        except Exception:
            return None
    
    def is_legal(self, fen: str, move_uci: str) -> bool:
        """Check if a move is legal."""
        try:
            import chess
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            return move in board.legal_moves
        except Exception:
            return False


def extract_uci_move(text: str) -> Optional[str]:
    """Extract UCI move from model output (must be in proper tags)."""
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', text)
    return match.group(1) if match else None


def run_chess_eval(model, tokenizer, eval_dataset, config: Config) -> dict:
    """
    Run chess evaluation with Stockfish.
    
    Uses proper stop tokens (EOS + </uci_move>) for efficient generation.
    """
    from tqdm import tqdm
    
    print("\n♟️  Running chess evaluation...")
    model.eval()
    
    # Get stop token IDs - use list for multiple stop tokens
    eos_token_id = tokenizer.eos_token_id
    uci_end_token = tokenizer.convert_tokens_to_ids("</uci_move>")
    
    stop_token_ids = [eos_token_id]
    if uci_end_token != tokenizer.unk_token_id:
        stop_token_ids.append(uci_end_token)
    
    print(f"  Stop tokens: EOS={eos_token_id}, </uci_move>={uci_end_token}")
    
    # Setup Stockfish
    sf_path = find_stockfish()
    stockfish = StockfishEngine(sf_path, depth=config.stockfish_depth)
    sf_available = stockfish.start()
    
    if sf_available:
        print(f"  ✓ Stockfish ready (depth={config.stockfish_depth})")
    else:
        print("  ⚠️ Stockfish unavailable - limited eval")
    
    # Prepare samples
    num_samples = min(config.stockfish_eval_samples, len(eval_dataset))
    samples = list(eval_dataset.select(range(num_samples)))
    
    prompts, fens, gt_moves = [], [], []
    for sample in samples:
        prompt = sample["prompt"]
        try:
            fen = prompt.split("FEN: ")[1].split("\n")[0].strip()
        except (IndexError, AttributeError):
            continue
        
        gt_move = extract_uci_move(sample["completion"])
        if not gt_move:
            continue
        
        prompts.append(prompt)
        fens.append(fen)
        gt_moves.append(gt_move)
    
    if not prompts:
        print("  ⚠️ No valid samples")
        stockfish.stop()
        return {"chess/num_evaluated": 0}
    
    # Set left padding for batched generation
    orig_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Generate predictions
    all_generated = []
    batch_size = config.eval_batch_size
    
    print(f"  Generating {len(prompts)} predictions (batch_size={batch_size})...")
    with tqdm(total=len(prompts), desc="  Gen", unit="pos") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Apply chat template
            messages = [[{"role": "user", "content": p}] for p in batch]
            texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) 
                for m in messages
            ]
            
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.eval_max_new_tokens,
                    do_sample=False,  # Greedy for eval (faster + deterministic)
                    eos_token_id=stop_token_ids,  # Stop on EOS or </uci_move>
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            # Decode outputs
            for j, output in enumerate(outputs):
                input_len = int(inputs.attention_mask[j].sum().item())
                generated = tokenizer.decode(output[input_len:], skip_special_tokens=False)
                all_generated.append(generated)
            
            pbar.update(len(batch))
    
    tokenizer.padding_side = orig_padding
    
    # Evaluate predictions
    parsed, legal, exact, total_cp_loss, comparisons = 0, 0, 0, 0, 0
    examples = []
    
    print("  Evaluating with Stockfish...")
    with tqdm(total=len(prompts), desc="  Eval", unit="pos") as pbar:
        for fen, gt_move, generated in zip(fens, gt_moves, all_generated):
            pred_move = extract_uci_move(generated)
            
            # Collect examples
            if len(examples) < 3:
                examples.append({
                    "fen": fen,
                    "gt": gt_move,
                    "pred": pred_move,
                    "text": generated,  # Truncate for display
                })
            
            if pred_move:
                parsed += 1
                
                if stockfish.is_legal(fen, pred_move):
                    legal += 1
                    
                    if sf_available:
                        # Get score for predicted move
                        pred_analysis = stockfish.analyze(fen, pred_move)
                        # Get best move score (position before any move)
                        best_analysis = stockfish.analyze(fen)
                        
                        if pred_analysis and best_analysis:
                            if best_analysis["best_move"] == pred_move:
                                exact += 1
                            
                            # CP loss = best_score - pred_score
                            cp_loss = abs(best_analysis["score"] - pred_analysis["score"])
                            total_cp_loss += cp_loss
                            comparisons += 1
            
            pbar.update(1)
    
    stockfish.stop()
    total = len(prompts)
    
    # Print examples
    print("\n" + "=" * 60)
    print("📝 Example Responses")
    print("=" * 60)
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"FEN: {ex['fen']}")
        print(f"Ground Truth: {ex['gt']}")
        print(f"Prediction:   {ex['pred'] or '(none)'}")
        print(f"Generated:\n{ex['text']}")
    print("=" * 60 + "\n")
    
    # Build metrics
    metrics = {
        "chess/tag_parse_rate": parsed / total if total else 0,
        "chess/legal_move_rate": legal / total if total else 0,
        "chess/num_evaluated": total,
    }
    
    if sf_available and comparisons > 0:
        metrics["chess/exact_match_rate"] = exact / total if total else 0
        metrics["chess/avg_centipawn_loss"] = total_cp_loss / comparisons
    
    print(f"  ✓ Tag parse rate: {parsed}/{total} ({metrics['chess/tag_parse_rate']:.1%})")
    print(f"  ✓ Legal moves: {legal}/{total} ({metrics['chess/legal_move_rate']:.1%})")
    if sf_available and comparisons > 0:
        print(f"  ✓ Exact matches: {exact}/{total} ({metrics['chess/exact_match_rate']:.1%})")
        print(f"  ✓ Avg CP loss: {metrics['chess/avg_centipawn_loss']:.1f}")
    
    return metrics


# =============================================================================
# Training Callback
# =============================================================================

from transformers import TrainerCallback


class ChessEvalCallback(TrainerCallback):
    """Run chess evaluation at regular intervals."""
    
    def __init__(self, eval_dataset, tokenizer, config: Config):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config
    
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step > 0 and state.global_step % self.config.eval_steps == 0:
            print(f"\n📊 Chess evaluation @ step {state.global_step}")
            metrics = run_chess_eval(model, self.tokenizer, self.eval_dataset, self.config)
            
            # Log to W&B
            try:
                import wandb
                if wandb.run:
                    wandb.log(metrics, step=state.global_step)
            except Exception:
                pass
            
            # Add to trainer history
            for key, value in metrics.items():
                state.log_history.append({key: value, "step": state.global_step})
        
        return control


# =============================================================================
# Main Training
# =============================================================================

def train(config: Config):
    """Main training function."""
    from trl import SFTConfig, SFTTrainer
    
    print("\n" + "=" * 60)
    print("♟️  Chess Reasoning SFT Training")
    print("=" * 60)
    
    # Setup
    print("\n🔎 Checking for Stockfish...")
    sf_path = find_stockfish()
    print(f"  ✓ Stockfish: {sf_path or 'not found'}")
    
    print("\n🔑 Setting up auth...")
    setup_auth()
    
    # Load model/tokenizer
    tokenizer = setup_tokenizer(config.model_name)
    model = setup_model(config.model_name, tokenizer, config)
    
    # Prepare data
    train_dataset, eval_dataset = prepare_datasets(config)
    
    # Training config
    print("\n⚙️  Configuring trainer...")
    training_args = SFTConfig(
        output_dir=config.output_dir,
        run_name=config.wandb_run_name,
        report_to="wandb",
        
        # Training
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        
        # Precision
        bf16=config.bf16,
        tf32=config.tf32,
        
        # Optimizations
        gradient_checkpointing=config.gradient_checkpointing,
        use_liger_kernel=config.use_liger_kernel,
        padding_free=config.padding_free,
        max_length=config.max_length,
        
        # No standard eval (using custom callback)
        eval_strategy="no",
        
        # Saving
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        # Hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_strategy=config.hub_strategy,
        hub_private_repo=config.hub_private_repo,
        
        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=config.logging_first_step,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        
        dataset_num_proc=4,
        seed=42069,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        callbacks=[ChessEvalCallback(eval_dataset, tokenizer, config)],
    )
    
    # Summary
    eff_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    print(f"\n📋 Training Summary:")
    print(f"  Model: {config.model_name}")
    print(f"  Effective batch: {eff_batch}")
    print(f"  LR: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Liger: {config.use_liger_kernel}, Padding-free: {config.padding_free}")
    print(f"  Chess eval every {config.eval_steps} steps")
    
    # Initial eval
    print("\n📊 Baseline evaluation...")
    initial_metrics = run_chess_eval(model, tokenizer, eval_dataset, config)
    try:
        import wandb
        if wandb.run:
            wandb.log({f"baseline/{k.split('/')[-1]}": v for k, v in initial_metrics.items()}, step=0)
    except Exception:
        pass
    
    # Train
    print("\n" + "=" * 60)
    print("🏃 Starting training...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save
    print("\n💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    if config.push_to_hub:
        print("\n☁️  Pushing to Hub...")
        trainer.push_to_hub()
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"  Saved to: {config.output_dir}")
    if config.push_to_hub:
        print(f"  Hub: https://huggingface.co/{config.hub_model_id}")
    print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Train chess reasoning model")
    parser.add_argument("--config", "-c", default=str(script_dir / "config.yaml"), help="Config YAML file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        candidate = (script_dir / config_path).resolve()
        if candidate.exists():
            config_path = candidate

    if config_path.exists():
        print(f"📄 Loading config: {config_path}")
        config = Config.from_yaml(str(config_path))
    else:
        print(f"⚠️ Config {config_path} not found, using defaults")
        config = Config()
    
    train(config)


if __name__ == "__main__":
    main()
