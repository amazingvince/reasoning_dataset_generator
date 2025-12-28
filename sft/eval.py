#!/usr/bin/env python3
"""
Chess Model Evaluation Script

Evaluates a trained chess reasoning model on random positions with full Stockfish analysis.

Usage:
    python eval.py --model amazingvince/chess_qwen3_4b_reasoning --num_samples 1000
    python eval.py --model ./chess_qwen3_4b_reasoning --num_samples 500
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    model_path: str = "./chess_qwen3_4b_reasoning"
    dataset_name: str = "amazingvince/chess-traces"
    num_samples: int = 1000
    batch_size: int = 16
    max_new_tokens: int = 2048
    stockfish_depth: int = 12
    output_file: str = "eval_results.json"
    
    # Qwen3 recommended generation settings
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    do_sample: bool = True


# ============================================================================
# Stockfish Evaluator
# ============================================================================

class StockfishEvaluator:
    """Evaluates chess moves using Stockfish."""
    
    def __init__(self, depth: int = 12):
        self.depth = depth
        self.engine = None
        self.stockfish_path = None
        self.available = False
        self._find_stockfish()
    
    def _find_stockfish(self):
        """Find Stockfish executable."""
        import shutil
        
        paths_to_try = [
            shutil.which("stockfish"),
            "/usr/games/stockfish",
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
        ]
        
        for path in paths_to_try:
            if path and os.path.isfile(path):
                self.stockfish_path = path
                self.available = True
                print(f"✓ Found Stockfish at: {path}")
                return
        
        print("⚠️ Stockfish not found!")
        self.available = False
    
    def start(self):
        """Start the Stockfish engine."""
        import chess.engine
        
        if not self.available:
            return False
        
        if self.engine is None:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(
                    self.stockfish_path,
                    timeout=30.0
                )
                return True
            except Exception as e:
                print(f"⚠️ Failed to start Stockfish: {e}")
                self.available = False
                return False
        return True
    
    def stop(self):
        """Stop the Stockfish engine."""
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None
    
    def get_move_score(self, fen: str, move_uci: str) -> Optional[int]:
        """Get centipawn score after making a move."""
        import chess
        import chess.engine
        
        if not self.available or self.engine is None:
            return None
        
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            if move not in board.legal_moves:
                return None
            
            board.push(move)
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].relative
            
            if score.is_mate():
                mate_in = score.mate()
                return 10000 if mate_in > 0 else -10000
            else:
                return -score.score()
                
        except Exception:
            return None
    
    def get_best_move_score(self, fen: str) -> Optional[tuple]:
        """Get best move and its score."""
        import chess
        import chess.engine
        
        if not self.available or self.engine is None:
            return None
        
        try:
            board = chess.Board(fen)
            result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            best_move = result.move.uci()
            
            board.push(result.move)
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].relative
            
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = -score.score()
            
            return best_move, cp
        except:
            return None
    
    def is_legal_move(self, fen: str, move_uci: str) -> bool:
        """Check if a move is legal."""
        import chess
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            return move in board.legal_moves
        except:
            return False
    
    def get_top_moves(self, fen: str, num_moves: int = 5) -> list:
        """Get top N moves with their scores."""
        import chess
        import chess.engine
        
        if not self.available or self.engine is None:
            return []
        
        try:
            board = chess.Board(fen)
            info = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth),
                multipv=num_moves
            )
            
            results = []
            for pv_info in info if isinstance(info, list) else [info]:
                if "pv" in pv_info and pv_info["pv"]:
                    move = pv_info["pv"][0].uci()
                    score = pv_info["score"].relative
                    if score.is_mate():
                        cp = 10000 if score.mate() > 0 else -10000
                    else:
                        cp = score.score()
                    results.append({"move": move, "score": cp})
            
            return results
        except:
            return []


def extract_uci_move(text: str) -> Optional[str]:
    """Extract UCI move from model output. Only counts if between proper tags."""
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', text)
    return match.group(1) if match else None


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(model_path: str):
    """Load model with optimizations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n🔄 Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )
    
    model.eval()
    
    # Compile model for faster inference
    print("🔧 Compiling model with torch.compile...")
    model = torch.compile(model, mode="reduce-overhead")
    
    print(f"✓ Model loaded on {model.device}")
    
    return model, tokenizer


# ============================================================================
# Dataset Loading
# ============================================================================

def load_eval_samples(dataset_name: str, num_samples: int):
    """Load random samples from dataset."""
    from datasets import load_dataset
    
    print(f"\n📊 Loading {num_samples} random samples from {dataset_name}...")
    
    # Load streaming and collect samples
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Collect more than needed for random selection
    buffer_size = min(num_samples * 3, 50000)
    all_samples = []
    
    print(f"  Buffering up to {buffer_size} samples...")
    for i, sample in enumerate(tqdm(dataset, total=buffer_size, desc="  Loading")):
        all_samples.append(sample)
        if i >= buffer_size - 1:
            break
    
    # Random sample
    random.seed(42)
    if len(all_samples) > num_samples:
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples
    
    print(f"✓ Selected {len(samples)} samples")
    
    return samples


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(config: EvalConfig):
    """Run full evaluation."""
    import chess
    from transformers import StoppingCriteria, StoppingCriteriaList
    
    print("\n" + "=" * 70)
    print("♟️  Chess Model Evaluation")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model_path)
    
    # Set left padding for generation
    tokenizer.padding_side = 'left'
    
    # Create stopping criteria
    stop_token_ids = tokenizer.encode("</uci_move>", add_special_tokens=False)
    
    class StopOnUciMove(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            for ids in input_ids:
                if len(ids) >= len(stop_token_ids):
                    if ids[-len(stop_token_ids):].tolist() == stop_token_ids:
                        return True
            return False
    
    stopping_criteria = StoppingCriteriaList([StopOnUciMove()])
    
    # Load samples
    samples = load_eval_samples(config.dataset_name, config.num_samples)
    
    # Prepare prompts
    print("\n📝 Preparing prompts...")
    eval_data = []
    for sample in samples:
        prompt = sample.get("prompt") or f"""You are an expert chess player. Choose the best move.
FEN: {sample['fen']}
Legal moves (UCI): {', '.join(sample['valid_moves']) if isinstance(sample['valid_moves'], list) else sample['valid_moves']}"""
        
        try:
            fen = sample['fen']
            gt_move = sample['chosen_move']
        except:
            continue
        
        eval_data.append({
            "fen": fen,
            "prompt": prompt,
            "ground_truth_move": gt_move,
            "reasoning_trace": sample.get('reasoning_trace', ''),
        })
    
    print(f"✓ Prepared {len(eval_data)} evaluation samples")
    
    # Start Stockfish
    stockfish = StockfishEvaluator(depth=config.stockfish_depth)
    stockfish.start()
    
    # Warmup generation (for torch.compile)
    print("\n🔥 Warming up model...")
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": eval_data[0]["prompt"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    warmup_inputs = tokenizer(warmup_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    print("✓ Warmup complete")
    
    # Run generation in batches
    print(f"\n🚀 Generating predictions (batch_size={config.batch_size})...")
    all_results = []
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(eval_data), config.batch_size), desc="Generating"):
        batch = eval_data[i:i + config.batch_size]
        
        # Tokenize
        messages_batch = [[{"role": "user", "content": d["prompt"]}] for d in batch]
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
        
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
                use_cache=True,
            )
        
        # Decode and store
        for j, output in enumerate(outputs):
            input_len = inputs.attention_mask[j].sum()
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=False)
            
            batch[j]["generated_text"] = generated
            batch[j]["predicted_move"] = extract_uci_move(generated)
        
        all_results.extend(batch)
    
    generation_time = time.time() - start_time
    print(f"✓ Generation complete in {generation_time:.1f}s ({len(all_results)/generation_time:.1f} samples/sec)")
    
    # Evaluate with Stockfish
    print(f"\n🔍 Evaluating with Stockfish (depth={config.stockfish_depth})...")
    
    stats = {
        "total": 0,
        "parsed": 0,
        "legal": 0,
        "exact_match": 0,
        "top_3_match": 0,
        "top_5_match": 0,
        "total_cp_loss": 0,
        "valid_cp_comparisons": 0,
        "cp_loss_buckets": {"0-10": 0, "10-25": 0, "25-50": 0, "50-100": 0, "100-200": 0, "200+": 0},
    }
    
    for result in tqdm(all_results, desc="Evaluating"):
        fen = result["fen"]
        gt_move = result["ground_truth_move"]
        pred_move = result["predicted_move"]
        
        stats["total"] += 1
        
        result["is_parsed"] = pred_move is not None
        result["is_legal"] = False
        result["is_exact_match"] = False
        result["centipawn_loss"] = None
        result["stockfish_best_move"] = None
        result["stockfish_top_moves"] = []
        
        if not pred_move:
            continue
        
        stats["parsed"] += 1
        
        # Check legality
        is_legal = stockfish.is_legal_move(fen, pred_move)
        result["is_legal"] = is_legal
        
        if not is_legal:
            continue
        
        stats["legal"] += 1
        
        # Get Stockfish analysis
        best_result = stockfish.get_best_move_score(fen)
        top_moves = stockfish.get_top_moves(fen, num_moves=5)
        result["stockfish_top_moves"] = top_moves
        
        if best_result:
            best_move, best_score = best_result
            result["stockfish_best_move"] = best_move
            
            # Check exact match
            if pred_move == best_move:
                stats["exact_match"] += 1
                result["is_exact_match"] = True
            
            # Check top-N match
            top_move_list = [m["move"] for m in top_moves]
            if pred_move in top_move_list[:3]:
                stats["top_3_match"] += 1
            if pred_move in top_move_list[:5]:
                stats["top_5_match"] += 1
            
            # Calculate centipawn loss
            pred_score = stockfish.get_move_score(fen, pred_move)
            if pred_score is not None:
                cp_loss = abs(best_score - pred_score)
                result["centipawn_loss"] = cp_loss
                stats["total_cp_loss"] += cp_loss
                stats["valid_cp_comparisons"] += 1
                
                # Bucket
                if cp_loss <= 10:
                    stats["cp_loss_buckets"]["0-10"] += 1
                elif cp_loss <= 25:
                    stats["cp_loss_buckets"]["10-25"] += 1
                elif cp_loss <= 50:
                    stats["cp_loss_buckets"]["25-50"] += 1
                elif cp_loss <= 100:
                    stats["cp_loss_buckets"]["50-100"] += 1
                elif cp_loss <= 200:
                    stats["cp_loss_buckets"]["100-200"] += 1
                else:
                    stats["cp_loss_buckets"]["200+"] += 1
    
    stockfish.stop()
    
    # Calculate final metrics
    total = stats["total"]
    metrics = {
        "num_samples": total,
        "tag_parse_rate": stats["parsed"] / total if total > 0 else 0,
        "legal_move_rate": stats["legal"] / total if total > 0 else 0,
        "exact_match_rate": stats["exact_match"] / total if total > 0 else 0,
        "top_3_match_rate": stats["top_3_match"] / total if total > 0 else 0,
        "top_5_match_rate": stats["top_5_match"] / total if total > 0 else 0,
        "avg_centipawn_loss": stats["total_cp_loss"] / stats["valid_cp_comparisons"] if stats["valid_cp_comparisons"] > 0 else 0,
        "cp_loss_distribution": stats["cp_loss_buckets"],
        "generation_time_sec": generation_time,
        "samples_per_sec": len(all_results) / generation_time,
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nModel: {config.model_path}")
    print(f"Samples: {total}")
    print(f"Stockfish depth: {config.stockfish_depth}")
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    print(f"{'Tag Parse Rate':<30} {metrics['tag_parse_rate']:>14.1%}")
    print(f"{'Legal Move Rate':<30} {metrics['legal_move_rate']:>14.1%}")
    print(f"{'Exact Match Rate':<30} {metrics['exact_match_rate']:>14.1%}")
    print(f"{'Top-3 Match Rate':<30} {metrics['top_3_match_rate']:>14.1%}")
    print(f"{'Top-5 Match Rate':<30} {metrics['top_5_match_rate']:>14.1%}")
    print(f"{'Avg Centipawn Loss':<30} {metrics['avg_centipawn_loss']:>14.1f}")
    
    print(f"\n📈 Centipawn Loss Distribution:")
    for bucket, count in stats["cp_loss_buckets"].items():
        pct = count / stats["valid_cp_comparisons"] * 100 if stats["valid_cp_comparisons"] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>8}: {count:>5} ({pct:>5.1f}%) {bar}")
    
    print(f"\n⏱️  Performance:")
    print(f"  Generation time: {generation_time:.1f}s")
    print(f"  Throughput: {metrics['samples_per_sec']:.1f} samples/sec")
    
    # Print some example results
    print("\n" + "=" * 70)
    print("📝 SAMPLE RESULTS (10 examples)")
    print("=" * 70)
    
    # Show mix of good and bad results
    exact_matches = [r for r in all_results if r.get("is_exact_match")]
    non_matches = [r for r in all_results if r.get("is_legal") and not r.get("is_exact_match")]
    failures = [r for r in all_results if not r.get("is_parsed") or not r.get("is_legal")]
    
    examples = []
    if exact_matches:
        examples.extend(random.sample(exact_matches, min(4, len(exact_matches))))
    if non_matches:
        examples.extend(random.sample(non_matches, min(3, len(non_matches))))
    if failures:
        examples.extend(random.sample(failures, min(3, len(failures))))
    
    for i, ex in enumerate(examples[:10], 1):
        status = "✅ EXACT" if ex.get("is_exact_match") else "⚠️ LEGAL" if ex.get("is_legal") else "❌ FAIL"
        print(f"\n--- Example {i} [{status}] ---")
        print(f"FEN: {ex['fen']}")
        print(f"Ground Truth: {ex['ground_truth_move']}")
        print(f"Prediction:   {ex['predicted_move'] or '(none)'}")
        print(f"SF Best Move: {ex.get('stockfish_best_move', 'N/A')}")
        if ex.get("centipawn_loss") is not None:
            print(f"CP Loss:      {ex['centipawn_loss']}")
        print(f"\nGenerated:\n{ex['generated_text']}")
    
    # Save full results
    output_data = {
        "config": {
            "model_path": config.model_path,
            "dataset_name": config.dataset_name,
            "num_samples": config.num_samples,
            "stockfish_depth": config.stockfish_depth,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    
    with open(config.output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to: {config.output_file}")
    print("=" * 70)
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate chess reasoning model")
    parser.add_argument("--model", "-m", type=str, default="./chess_qwen3_4b_reasoning",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--dataset", "-d", type=str, default="amazingvince/chess-traces",
                        help="Dataset to evaluate on")
    parser.add_argument("--num_samples", "-n", type=int, default=1000,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--stockfish_depth", type=int, default=12,
                        help="Stockfish search depth")
    parser.add_argument("--output", "-o", type=str, default="eval_results.json",
                        help="Output file for results")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_path=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        stockfish_depth=args.stockfish_depth,
        output_file=args.output,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    
    # Install dependencies if needed
    try:
        import chess
    except ImportError:
        print("Installing python-chess...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess"])
    
    run_evaluation(config)


if __name__ == "__main__":
    main()
