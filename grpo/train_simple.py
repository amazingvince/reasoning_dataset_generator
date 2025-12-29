#!/usr/bin/env python3
"""
Simplified Chess GRPO Training with Unsloth

This is a minimal, easy-to-understand version optimized for:
- Single H100 80GB GPU
- Quick experimentation
- Clear code structure

Usage:
    python train_simple.py --model your-chess-model
"""

import os
import re
import random

# Enable Unsloth's memory optimization BEFORE any imports
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import chess
import chess.engine
import wandb
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer


# ============================================================================
# Configuration - Edit these for your setup
# ============================================================================

MODEL_NAME = "your-chess-model-4b"  # Your 4B chess model
STOCKFISH_PATH = "/usr/games/stockfish"
STOCKFISH_DEPTH = 20
OUTPUT_DIR = "./chess_grpo_output"

# Training settings
NUM_SAMPLES = 10000  # Start small for testing
NUM_GENERATIONS = 8
MAX_COMPLETION_LENGTH = 2048
LEARNING_RATE = 5e-6

# Wandb settings
USE_WANDB = True
WANDB_PROJECT = "chess-grpo"


# ============================================================================
# Reward Function
# ============================================================================

# Global Stockfish engine (initialized once)
STOCKFISH_ENGINE = None
STOCKFISH_RESTARTS = 0


def init_stockfish():
    """Initialize or reinitialize Stockfish engine with retry logic."""
    global STOCKFISH_ENGINE, STOCKFISH_RESTARTS

    # Clean up existing engine if it exists but is unresponsive
    if STOCKFISH_ENGINE is not None:
        try:
            STOCKFISH_ENGINE.ping()
            return  # Engine is alive
        except Exception:
            # Engine is dead, clean up
            try:
                STOCKFISH_ENGINE.quit()
            except Exception:
                pass
            STOCKFISH_ENGINE = None
            STOCKFISH_RESTARTS += 1

    # Initialize new engine
    STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    STOCKFISH_ENGINE.configure({"Threads": 4, "Hash": 256})


def extract_move(text: str) -> str | None:
    """Extract UCI move from model output."""
    match = re.search(r'<uci_move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</uci_move>', text, re.I)
    return match.group(1).lower() if match else None


def _extract_prompt_text(prompt) -> str:
    """Extract text content from prompt (handles both string and chat format)."""
    if isinstance(prompt, list):
        # Chat format: list of message dicts
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return " ".join(msg.get("content", "") for msg in prompt if isinstance(msg, dict))
    return str(prompt)


def stockfish_reward(completions, prompts, **kwargs) -> list[float]:
    """
    Main reward function using Stockfish.

    Rewards (rebalanced per 2025 best practices):
    - 1.5: Matches Stockfish top move
    - 1.0: In top 3
    - 0.6: In top 5
    - 0.1: Legal but not top 5
    - -0.5: Illegal move
    - -1.0: No move found
    """
    init_stockfish()
    rewards = []

    for completion, prompt in zip(completions, prompts):
        # Handle TRL completion format (string or chat format)
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)

        # Extract FEN from prompt (handles both string and chat format)
        prompt_text = _extract_prompt_text(prompt)
        fen_match = re.search(r'FEN:\s*([^\n]+)', prompt_text)
        if not fen_match:
            rewards.append(-1.0)
            continue

        fen = fen_match.group(1).strip()
        predicted_move = extract_move(text)

        # No move extracted
        if not predicted_move:
            rewards.append(-1.0)
            continue

        # Check if legal
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(predicted_move)
            if move not in board.legal_moves:
                rewards.append(-0.5)
                continue
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
            rewards.append(-0.5)
            continue

        # Get Stockfish top moves with retry logic
        try:
            init_stockfish()  # Ensure engine is alive
            result = STOCKFISH_ENGINE.analyse(
                board,
                chess.engine.Limit(depth=STOCKFISH_DEPTH),
                multipv=5
            )
            top_moves = [pv["pv"][0].uci() for pv in result if "pv" in pv and pv["pv"]]
        except chess.engine.EngineTerminatedError:
            global STOCKFISH_ENGINE
            STOCKFISH_ENGINE = None  # Force restart on next call
            rewards.append(0.1)  # Fallback: legal but couldn't analyze
            continue
        except Exception as e:
            rewards.append(0.1)  # Fallback: legal but couldn't analyze
            continue

        # Score based on ranking (rebalanced rewards)
        if predicted_move == top_moves[0]:
            rewards.append(1.5)
        elif predicted_move in top_moves[:3]:
            rewards.append(1.0)
        elif predicted_move in top_moves:
            rewards.append(0.6)
        else:
            rewards.append(0.1)

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Reward for proper XML format (reduced to avoid masking move quality)."""
    rewards = []
    pattern = r'<think>[\s\S]*?</think>\s*<uci_move>[a-h][1-8][a-h][1-8][qrbn]?</uci_move>'

    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        # Reduced format bonus per 2025 best practices
        rewards.append(0.1 if re.search(pattern, text, re.I) else -0.1)

    return rewards


# ============================================================================
# Dataset Creation (Simplified) - Chat format per TRL 2025 recommendations
# ============================================================================

SYSTEM_CONTENT = """You are an expert chess player. Analyze the position and choose the best move.
Rules:
- Put all reasoning inside <think>...</think>.
- Output exactly one <uci_move>...</uci_move> tag with a single move.
- Do not output anything after </uci_move>.
Output format:
<think>...</think>
<uci_move>...</uci_move>"""


def create_simple_dataset(num_samples: int = 1000) -> Dataset:
    """Create a simple dataset from Lichess puzzles using chat format."""
    from datasets import load_dataset

    print(f"Loading {num_samples} puzzle positions...")

    puzzles = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    positions = []

    for puzzle in puzzles:
        if len(positions) >= num_samples:
            break

        fen = puzzle.get("FEN")
        if not fen:
            continue

        try:
            board = chess.Board(fen)
            # Skip positions with no legal moves
            if not list(board.legal_moves):
                continue
            legal_moves = " ".join(sorted([m.uci() for m in board.legal_moves]))
        except ValueError:
            # Invalid FEN
            continue

        # Chat format per TRL 2025 recommendations
        prompt = [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": f"FEN: {fen}\nLegal moves (UCI): {legal_moves}\n\nChoose the best move."},
        ]
        positions.append({"prompt": prompt, "fen": fen})

    print(f"Created dataset with {len(positions)} positions")
    return Dataset.from_list(positions)


# ============================================================================
# Main Training
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--use_fp8", action="store_true", help="Use FP8 for H100")
    parser.add_argument("--wandb", action="store_true", default=USE_WANDB, help="Enable wandb logging")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable wandb logging")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT, help="Wandb project name")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chess GRPO Training with Unsloth")
    print("=" * 60)
    
    # Load model with Unsloth
    print(f"\nLoading model: {args.model}")
    # FP8 KV cache requires FP8 model (FlashInfer limitation)
    use_fp8_kv = args.use_fp8  # Only enable if model is FP8
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        load_in_4bit=False,  # 16-bit for H100
        load_in_fp8=args.use_fp8,
        fast_inference=True,  # Enable vLLM
        max_lora_rank=32,
        gpu_memory_utilization=0.85,
        float8_kv_cache=use_fp8_kv,
    )
    
    # Add LoRA
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Create dataset
    dataset = create_simple_dataset(args.samples)
    
    # GRPO config with 2025 best practices (DAPO loss)
    print("\nConfiguring GRPO trainer with DAPO loss...")
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.7,
        # 2025 best practices: beta=0 (no KL penalty with verifiable rewards)
        beta=0.0,
        # Batch-level reward scaling is more robust
        scale_rewards="batch",
        # Ignore truncated outputs in loss
        mask_truncated_completions=True,
        # DAPO loss configuration (2025 best practice for reasoning tasks)
        # DAPO achieved 50% on AIME 2024 vs 30% baseline
        loss_type="dapo",
        # Asymmetric clipping (DAPO Clip-Higher strategy)
        epsilon=0.2,       # Lower bound for suppression
        epsilon_high=0.28,  # Upper bound for encouragement
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        logging_steps=1,
        save_steps=500,
        max_grad_norm=0.1,
        bf16=is_bfloat16_supported(),
        use_vllm=True,
        report_to=["wandb", "tensorboard"] if args.wandb else ["tensorboard"],
    )

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"chess-grpo-{args.model.split('/')[-1]}",
            config={
                "model_name": args.model,
                "num_samples": args.samples,
                "num_generations": NUM_GENERATIONS,
                "max_completion_length": MAX_COMPLETION_LENGTH,
                "learning_rate": LEARNING_RATE,
                "use_fp8": args.use_fp8,
                "loss_type": "dapo",
                "beta": 0.0,
            },
            reinit=True,
        )
        print(f"Wandb initialized: {wandb.run.name}")
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[stockfish_reward, format_reward],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("Starting GRPO training...")
    print("=" * 60)
    print(f"  Samples: {len(dataset)}")
    print(f"  Generations per prompt: {NUM_GENERATIONS}")
    print(f"  Max completion length: {MAX_COMPLETION_LENGTH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()
    print("TIP: Wait 300+ steps before expecting reward improvement")
    print("NOTE: Using DAPO loss with 2025 best practices")
    print("      - Asymmetric clipping (epsilon=0.2, epsilon_high=0.28)")
    print("      - No KL penalty (beta=0), batch-level reward scaling")
    print("=" * 60 + "\n")
    
    try:
        trainer.train()

        # Save
        print("\nSaving model...")
        model.save_lora(f"{OUTPUT_DIR}/lora_weights")
        print(f"Done! Model saved to {OUTPUT_DIR}")

    finally:
        if STOCKFISH_ENGINE:
            STOCKFISH_ENGINE.quit()
        if args.wandb and wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
