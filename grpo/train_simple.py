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


# ============================================================================
# Reward Function
# ============================================================================

# Global Stockfish engine (initialized once)
STOCKFISH_ENGINE = None


def init_stockfish():
    global STOCKFISH_ENGINE
    if STOCKFISH_ENGINE is None:
        STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        STOCKFISH_ENGINE.configure({"Threads": 4, "Hash": 256})


def extract_move(text: str) -> str | None:
    """Extract UCI move from model output."""
    match = re.search(r'<uci_move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</uci_move>', text, re.I)
    return match.group(1).lower() if match else None


def stockfish_reward(completions, prompts, **kwargs) -> list[float]:
    """
    Main reward function using Stockfish.
    
    Rewards:
    - 1.0: Matches Stockfish top move
    - 0.7: In top 3
    - 0.4: In top 5
    - 0.1: Legal but not top 5
    - -0.5: Illegal move
    - -1.0: No move found
    """
    init_stockfish()
    rewards = []
    
    for completion, prompt in zip(completions, prompts):
        # Handle TRL completion format
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        
        # Extract FEN from prompt
        fen_match = re.search(r'FEN:\s*([^\n]+)', prompt)
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
        except:
            rewards.append(-0.5)
            continue
        
        # Get Stockfish top moves
        try:
            result = STOCKFISH_ENGINE.analyse(
                board,
                chess.engine.Limit(depth=STOCKFISH_DEPTH),
                multipv=5
            )
            top_moves = [pv["pv"][0].uci() for pv in result if "pv" in pv and pv["pv"]]
        except:
            rewards.append(0.1)  # Fallback: legal but couldn't analyze
            continue
        
        # Score based on ranking
        if predicted_move == top_moves[0]:
            rewards.append(1.0)
        elif predicted_move in top_moves[:3]:
            rewards.append(0.7)
        elif predicted_move in top_moves:
            rewards.append(0.4)
        else:
            rewards.append(0.1)
    
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """Reward for proper XML format."""
    rewards = []
    pattern = r'<think>[\s\S]*?</think>\s*<uci_move>[a-h][1-8][a-h][1-8][qrbn]?</uci_move>'
    
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        rewards.append(0.2 if re.search(pattern, text, re.I) else -0.1)
    
    return rewards


# ============================================================================
# Dataset Creation (Simplified)
# ============================================================================

def create_simple_dataset(num_samples: int = 1000) -> Dataset:
    """Create a simple dataset from Lichess puzzles."""
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
            legal_moves = " ".join(sorted([m.uci() for m in board.legal_moves]))
        except:
            continue
        
        prompt = f"""You are an expert chess player. Choose the best move.
FEN: {fen}
Legal moves (UCI): {legal_moves}
Rules:
- Put all reasoning inside <think>...</think>.
- Output exactly one <uci_move>...</uci_move> tag with a single move.
- Do not output anything after </uci_move>.
Output format:
<think>...</think>
<uci_move>...</uci_move>
"""
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
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chess GRPO Training with Unsloth")
    print("=" * 60)
    
    # Load model with Unsloth
    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        load_in_4bit=False,  # 16-bit for H100
        load_in_fp8=args.use_fp8,
        fast_inference=True,  # Enable vLLM
        max_lora_rank=32,
        gpu_memory_utilization=0.85,
        float8_kv_cache=True,  # 2x less KV cache on H100
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
    
    # GRPO config
    print("\nConfiguring GRPO trainer...")
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=512,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.7,
        beta=0.04,
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
        report_to=["tensorboard"],
    )
    
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


if __name__ == "__main__":
    main()
