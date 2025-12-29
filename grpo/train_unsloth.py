#!/usr/bin/env python3
"""
Chess GRPO Training with Unsloth on Single H100

Optimized for:
- Single NVIDIA H100 80GB
- 4B parameter model
- 2048 token reasoning traces
- Stockfish verification rewards

Key optimizations:
- Unsloth for 2x faster training, 70% less VRAM
- vLLM integration for fast generation during GRPO
- FP8 KV cache for 2x less KV cache memory
- Memory-efficient GRPO with Unsloth Standby
- Gradient checkpointing with Unsloth's smart offloading
"""

import os
import re
import random
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, field

import yaml
import wandb

# Enable Unsloth's memory-efficient vLLM standby mode BEFORE imports
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import chess
import chess.engine
from datasets import load_dataset, Dataset

# Unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration for H100
# ============================================================================

@dataclass
class ChessGRPOConfig:
    """Configuration optimized for single H100 80GB."""
    
    # Model - adjust to your 4B model
    model_name: str = "your-chess-model-4b"  # Replace with your model
    max_seq_length: int = 4096  # Prompt + completion
    
    # Unsloth settings
    load_in_4bit: bool = False  # Use 16-bit for H100 (plenty of VRAM)
    use_fp8: bool = False  # Set True for FP8 training (1.4x faster on H100)

    # LoRA vs Full Fine-tuning
    use_lora: bool = True  # Set False for full fine-tuning
    use_gradient_checkpointing: bool = True  # Required for full fine-tuning

    # LoRA settings (ignored if use_lora=False)
    lora_r: int = 32  # Larger rank for better learning
    lora_alpha: int = 32
    lora_dropout: float = 0.0  # 0 is optimized in Unsloth
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # vLLM settings for H100
    gpu_memory_utilization: float = 0.85  # High utilization with Unsloth Standby
    use_fp8_kv_cache: bool = True  # 2x less KV cache memory on H100
    use_vllm: bool = True

    # Stockfish
    stockfish_path: Optional[str] = None
    stockfish_depth: int = 20
    stockfish_threads: int = 8  # H100 has good CPU too
    stockfish_hash_mb: int = 512
    
    # Data settings
    games_dataset: str = "Lichess/standard-chess-games"
    puzzles_dataset: str = "Lichess/chess-puzzles"
    min_elo: int = 800
    sample_rate: float = 0.3
    skip_first_moves: int = 4
    skip_last_moves: int = 2
    min_puzzle_rating: int = 800
    max_puzzle_rating: int = 2500
    puzzle_ratio: float = 0.3
    num_train_samples: int = 50000
    
    # ELO weights including lower levels
    elo_weights: Dict[int, float] = field(default_factory=lambda: {
        800: 0.08,
        1000: 0.10,
        1200: 0.12,
        1400: 0.15,
        1600: 0.18,
        1800: 0.20,
        2000: 0.18,
        2200: 0.12,
        2400: 0.08,
    })
    
    # GRPO Training settings optimized for H100
    per_device_train_batch_size: int = 1  # GRPO uses 1, gradient accumulation handles rest
    gradient_accumulation_steps: int = 4
    num_generations: int = 8  # More generations = better advantage estimation
    max_prompt_length: int = 1024  # Input tokens (FEN + legal moves + instructions)
    max_completion_length: int = 2048  # Output tokens (thinking + move)
    
    # Optimizer
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    # adamw_torch_fused: PyTorch fused AdamW kernel, fastest on H100
    # paged_adamw_8bit: memory efficient for LoRA, but slower
    optim: str = "adamw_torch_fused"
    
    # Training duration
    num_train_epochs: int = 1
    max_steps: int = -1  # Use epochs
    
    # GRPO hyperparameters
    # beta=0.0 per 2025 research (Open-Reasoner-Zero, DAPO, Understanding R1-Zero)
    # With verifiable rewards like Stockfish, KL penalty is unnecessary
    beta: float = 0.0
    temperature: float = 0.7
    # scale_rewards="batch" is more robust than per-group normalization
    scale_rewards: str = "batch"
    # Ignore truncated outputs in loss computation
    mask_truncated_completions: bool = True

    # DAPO loss configuration (2025 best practice for reasoning tasks)
    # loss_type options: "grpo" (default TRL), "dapo", "sapo", "dr_grpo", "bnpo"
    loss_type: str = "dapo"
    # Asymmetric clipping (DAPO Clip-Higher strategy)
    # epsilon: lower bound for suppression (standard)
    # epsilon_high: upper bound for encouragement (allows more exploration)
    epsilon: float = 0.2
    epsilon_high: float = 0.28

    # Logging and saving
    logging_steps: int = 1
    save_steps: int = 500
    max_grad_norm: float = 0.1

    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "chess-grpo"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Output
    output_dir: str = "./chess_grpo_h100_output"
    
    # Reward weights (rebalanced to emphasize move quality)
    reward_top1: float = 1.5
    reward_top3: float = 1.0
    reward_top5: float = 0.6
    reward_legal: float = 0.1
    reward_illegal: float = -0.5
    reward_no_move: float = -1.0
    # Format rewards reduced to avoid masking move quality signal
    reward_format_bonus: float = 0.1
    reward_format_penalty: float = -0.1
    reward_xml_bonus: float = 0.05
    reward_xml_penalty: float = -0.15

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChessGRPOConfig":
        config = cls()

        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        def section(name: str) -> Dict[str, Any]:
            value = data.get(name) or {}
            return value if isinstance(value, dict) else {}

        def as_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def as_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            return bool(value)

        model = section("model")
        if "name_or_path" in model:
            config.model_name = str(model["name_or_path"])
        if "max_seq_length" in model:
            config.max_seq_length = as_int(model["max_seq_length"], config.max_seq_length)

        unsloth_cfg = section("unsloth")
        if "load_in_4bit" in unsloth_cfg:
            config.load_in_4bit = as_bool(unsloth_cfg["load_in_4bit"], config.load_in_4bit)
        if "use_fp8" in unsloth_cfg:
            config.use_fp8 = as_bool(unsloth_cfg["use_fp8"], config.use_fp8)
        if "use_lora" in unsloth_cfg:
            config.use_lora = as_bool(unsloth_cfg["use_lora"], config.use_lora)
        if "use_gradient_checkpointing" in unsloth_cfg:
            config.use_gradient_checkpointing = as_bool(unsloth_cfg["use_gradient_checkpointing"], config.use_gradient_checkpointing)
        if "gpu_memory_utilization" in unsloth_cfg:
            config.gpu_memory_utilization = as_float(unsloth_cfg["gpu_memory_utilization"], config.gpu_memory_utilization)
        if "use_fp8_kv_cache" in unsloth_cfg:
            config.use_fp8_kv_cache = as_bool(unsloth_cfg["use_fp8_kv_cache"], config.use_fp8_kv_cache)

        lora = section("lora")
        if "r" in lora:
            config.lora_r = as_int(lora["r"], config.lora_r)
        if "alpha" in lora:
            config.lora_alpha = as_int(lora["alpha"], config.lora_alpha)
        if "dropout" in lora:
            config.lora_dropout = as_float(lora["dropout"], config.lora_dropout)
        if "target_modules" in lora and isinstance(lora["target_modules"], list):
            config.lora_target_modules = [str(x) for x in lora["target_modules"] if str(x).strip()]

        stockfish = section("stockfish")
        if "path" in stockfish and stockfish["path"]:
            config.stockfish_path = str(stockfish["path"])
        if "depth" in stockfish:
            config.stockfish_depth = as_int(stockfish["depth"], config.stockfish_depth)
        if "threads" in stockfish:
            config.stockfish_threads = as_int(stockfish["threads"], config.stockfish_threads)
        if "hash_mb" in stockfish:
            config.stockfish_hash_mb = as_int(stockfish["hash_mb"], config.stockfish_hash_mb)

        reward = section("reward")
        if "top1_reward" in reward:
            config.reward_top1 = as_float(reward["top1_reward"], config.reward_top1)
        if "top3_reward" in reward:
            config.reward_top3 = as_float(reward["top3_reward"], config.reward_top3)
        if "top5_reward" in reward:
            config.reward_top5 = as_float(reward["top5_reward"], config.reward_top5)
        if "legal_reward" in reward:
            config.reward_legal = as_float(reward["legal_reward"], config.reward_legal)
        if "illegal_penalty" in reward:
            config.reward_illegal = as_float(reward["illegal_penalty"], config.reward_illegal)
        if "no_move_penalty" in reward:
            config.reward_no_move = as_float(reward["no_move_penalty"], config.reward_no_move)
        if "format_bonus" in reward:
            config.reward_format_bonus = as_float(reward["format_bonus"], config.reward_format_bonus)
        if "format_penalty" in reward:
            config.reward_format_penalty = as_float(reward["format_penalty"], config.reward_format_penalty)
        if "xml_bonus" in reward:
            config.reward_xml_bonus = as_float(reward["xml_bonus"], config.reward_xml_bonus)
        if "xml_penalty" in reward:
            config.reward_xml_penalty = as_float(reward["xml_penalty"], config.reward_xml_penalty)

        data_cfg = section("data")
        if "games_dataset" in data_cfg:
            config.games_dataset = str(data_cfg["games_dataset"])
        if "puzzles_dataset" in data_cfg:
            config.puzzles_dataset = str(data_cfg["puzzles_dataset"])
        if "min_elo" in data_cfg:
            config.min_elo = as_int(data_cfg["min_elo"], config.min_elo)
        if "sample_rate" in data_cfg:
            config.sample_rate = as_float(data_cfg["sample_rate"], config.sample_rate)
        if "skip_first_moves" in data_cfg:
            config.skip_first_moves = as_int(data_cfg["skip_first_moves"], config.skip_first_moves)
        if "skip_last_moves" in data_cfg:
            config.skip_last_moves = as_int(data_cfg["skip_last_moves"], config.skip_last_moves)
        if "min_puzzle_rating" in data_cfg:
            config.min_puzzle_rating = as_int(data_cfg["min_puzzle_rating"], config.min_puzzle_rating)
        if "max_puzzle_rating" in data_cfg:
            config.max_puzzle_rating = as_int(data_cfg["max_puzzle_rating"], config.max_puzzle_rating)
        if "puzzle_ratio" in data_cfg:
            config.puzzle_ratio = as_float(data_cfg["puzzle_ratio"], config.puzzle_ratio)
        if "num_train_samples" in data_cfg:
            config.num_train_samples = as_int(data_cfg["num_train_samples"], config.num_train_samples)
        elo_weights = data_cfg.get("elo_weights")
        if isinstance(elo_weights, dict) and elo_weights:
            parsed: Dict[int, float] = {}
            for key, value in elo_weights.items():
                try:
                    parsed[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if parsed:
                config.elo_weights = parsed

        training = section("training")
        if "per_device_train_batch_size" in training:
            config.per_device_train_batch_size = as_int(training["per_device_train_batch_size"], config.per_device_train_batch_size)
        if "gradient_accumulation_steps" in training:
            config.gradient_accumulation_steps = as_int(training["gradient_accumulation_steps"], config.gradient_accumulation_steps)
        if "num_generations" in training:
            config.num_generations = as_int(training["num_generations"], config.num_generations)
        if "max_prompt_length" in training:
            config.max_prompt_length = as_int(training["max_prompt_length"], config.max_prompt_length)
        if "max_completion_length" in training:
            config.max_completion_length = as_int(training["max_completion_length"], config.max_completion_length)
        if "temperature" in training:
            config.temperature = as_float(training["temperature"], config.temperature)
        if "beta" in training:
            config.beta = as_float(training["beta"], config.beta)
        if "scale_rewards" in training:
            config.scale_rewards = str(training["scale_rewards"])
        if "mask_truncated_completions" in training:
            config.mask_truncated_completions = as_bool(training["mask_truncated_completions"], config.mask_truncated_completions)
        # DAPO loss configuration
        if "loss_type" in training:
            config.loss_type = str(training["loss_type"])
        if "epsilon" in training:
            config.epsilon = as_float(training["epsilon"], config.epsilon)
        if "epsilon_high" in training:
            config.epsilon_high = as_float(training["epsilon_high"], config.epsilon_high)
        if "learning_rate" in training:
            config.learning_rate = as_float(training["learning_rate"], config.learning_rate)
        if "weight_decay" in training:
            config.weight_decay = as_float(training["weight_decay"], config.weight_decay)
        if "warmup_ratio" in training:
            config.warmup_ratio = as_float(training["warmup_ratio"], config.warmup_ratio)
        if "lr_scheduler_type" in training:
            config.lr_scheduler_type = str(training["lr_scheduler_type"])
        if "optim" in training:
            config.optim = str(training["optim"])
        if "num_train_epochs" in training:
            config.num_train_epochs = as_int(training["num_train_epochs"], config.num_train_epochs)
        if "max_steps" in training:
            config.max_steps = as_int(training["max_steps"], config.max_steps)
        if "max_grad_norm" in training:
            config.max_grad_norm = as_float(training["max_grad_norm"], config.max_grad_norm)
        if "logging_steps" in training:
            config.logging_steps = as_int(training["logging_steps"], config.logging_steps)
        if "save_steps" in training:
            config.save_steps = as_int(training["save_steps"], config.save_steps)
        if "use_vllm" in training:
            config.use_vllm = as_bool(training["use_vllm"], config.use_vllm)
        if "output_dir" in training:
            config.output_dir = str(training["output_dir"])

        wandb_cfg = section("wandb")
        if "enabled" in wandb_cfg:
            config.use_wandb = as_bool(wandb_cfg["enabled"], config.use_wandb)
        if "project" in wandb_cfg:
            config.wandb_project = str(wandb_cfg["project"])
        if "run_name" in wandb_cfg:
            config.wandb_run_name = str(wandb_cfg["run_name"]) if wandb_cfg["run_name"] else None
        if "entity" in wandb_cfg:
            config.wandb_entity = str(wandb_cfg["entity"]) if wandb_cfg["entity"] else None

        return config


def find_stockfish() -> Optional[str]:
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


def resolve_stockfish_path(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return str(candidate)
        resolved = shutil.which(str(explicit_path))
        if resolved and Path(resolved).exists():
            return resolved
    return find_stockfish()


# ============================================================================
# Data Pipeline (same as before but as functions)
# ============================================================================

def _parse_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_movetext(movetext: str) -> List[str]:
    movetext = re.sub(r'\{[^}]*\}', '', movetext)
    movetext = re.sub(r'\d+\.+', '', movetext)
    movetext = re.sub(r'(1-0|0-1|1/2-1/2|\*)$', '', movetext)
    return [m.strip() for m in movetext.split() if m.strip()]


def iter_game_positions(config: ChessGRPOConfig, seed: int) -> Iterator[Dict[str, Any]]:
    """Iterate through positions from chess games."""
    rng = random.Random(seed)
    dataset = load_dataset(config.games_dataset, split="train", streaming=True)
    
    for game in dataset:
        if game.get("Termination") == "Time forfeit":
            continue
        
        result = game.get("Result", "")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue
        
        white_elo = _parse_int(game.get("WhiteElo", 1500), 1500)
        black_elo = _parse_int(game.get("BlackElo", 1500), 1500)
        avg_elo = (white_elo + black_elo) / 2
        
        if avg_elo < config.min_elo:
            continue
        
        weight = 0.01
        for threshold in sorted(config.elo_weights.keys(), reverse=True):
            if avg_elo >= threshold:
                weight = config.elo_weights[threshold]
                break
        
        if rng.random() > weight:
            continue
        
        movetext = game.get("movetext", "") or game.get("moves", "")
        if not movetext:
            continue
        
        moves = parse_movetext(movetext)
        if len(moves) < config.skip_first_moves + config.skip_last_moves + 1:
            continue
        
        board = chess.Board()
        for move_idx, move_san in enumerate(moves):
            if move_idx < config.skip_first_moves:
                try:
                    board.push_san(move_san)
                except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError) as e:
                    logger.debug(f"Invalid opening move {move_san}: {e}")
                    break
                continue
            
            if move_idx >= len(moves) - config.skip_last_moves:
                break
            
            try:
                if rng.random() <= config.sample_rate:
                    # Validate position has legal moves before yielding
                    if list(board.legal_moves):
                        yield {
                            "fen": board.fen(),
                            "source": "game",
                            "avg_elo": avg_elo,
                        }
                board.push_san(move_san)
            except (chess.IllegalMoveError, chess.InvalidMoveError, ValueError) as e:
                logger.debug(f"Invalid move {move_san} at position {move_idx}: {e}")
                break


def iter_puzzle_positions(config: ChessGRPOConfig) -> Iterator[Dict[str, Any]]:
    """Iterate through positions from chess puzzles."""
    dataset = load_dataset(config.puzzles_dataset, split="train", streaming=True)
    
    for puzzle in dataset:
        puzzle_id = puzzle.get("PuzzleId")
        rating = _parse_int(puzzle.get("Rating", 1500), 1500)
        
        if rating < config.min_puzzle_rating or rating > config.max_puzzle_rating:
            continue
        
        themes = puzzle.get("Themes") or []
        if isinstance(themes, str):
            themes = [t for t in themes.split() if t]
        
        fen = puzzle.get("FEN")
        moves_str = puzzle.get("Moves", "")
        if not fen or not moves_str:
            continue
        
        moves = moves_str.split()
        if len(moves) < 2:
            continue
        
        try:
            board = chess.Board(fen)
        except ValueError:
            # Invalid FEN string
            continue

        # Skip positions with no legal moves
        if not list(board.legal_moves):
            continue

        for idx, move_uci in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break

            if move not in board.legal_moves:
                break

            if idx % 2 == 1:
                # Validate position has legal moves before yielding
                if list(board.legal_moves):
                    yield {
                        "fen": board.fen(),
                        "source": "puzzle",
                        "puzzle_id": str(puzzle_id) if puzzle_id else None,
                        "puzzle_rating": rating,
                        "puzzle_themes": themes,
                    }

            board.push(move)


def create_dataset(config: ChessGRPOConfig, seed: int = 42) -> Dataset:
    """Create training dataset from games and puzzles."""
    rng = random.Random(seed)
    positions = []
    
    game_iter = iter_game_positions(config, seed)
    puzzle_iter = iter_puzzle_positions(config)
    
    game_buffer, puzzle_buffer = [], []
    
    logger.info(f"Collecting {config.num_train_samples} positions...")
    
    while len(positions) < config.num_train_samples:
        use_puzzle = rng.random() < config.puzzle_ratio
        
        if use_puzzle:
            if not puzzle_buffer:
                try:
                    for _ in range(100):
                        puzzle_buffer.append(next(puzzle_iter))
                except StopIteration:
                    puzzle_iter = iter_puzzle_positions(config)
                    continue
            if puzzle_buffer:
                positions.append(puzzle_buffer.pop(0))
        else:
            if not game_buffer:
                try:
                    for _ in range(100):
                        game_buffer.append(next(game_iter))
                except StopIteration:
                    game_iter = iter_game_positions(config, seed + len(positions))
                    continue
            if game_buffer:
                positions.append(game_buffer.pop(0))
        
        if len(positions) % 5000 == 0:
            logger.info(f"  Collected {len(positions)} positions...")
    
    # Add prompts
    for pos in positions:
        pos["prompt"] = format_chess_prompt(pos["fen"])
    
    logger.info(f"Created dataset with {len(positions)} positions")
    return Dataset.from_list(positions)


# ============================================================================
# Prompt Formatting (Must match SFT format exactly!)
# ============================================================================

# This prompt format MUST match what was used in SFT training (sft/train.py)
# Key differences from standard chat:
# - NO system message (everything in user content)
# - Custom chat template adds <think> at generation prompt

CHESS_PROMPT_TEMPLATE = """You are an expert chess player. Choose the best move.
FEN: {fen}
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


def format_chess_prompt(fen: str) -> list[dict[str, str]]:
    """Format a FEN into chat format for GRPO training.

    Returns a list of message dicts matching SFT format (user message only, no system).
    """
    try:
        board = chess.Board(fen)
        legal_moves = ", ".join(sorted([m.uci() for m in board.legal_moves]))
    except ValueError:
        # Invalid FEN - return empty legal moves
        legal_moves = ""

    # NO system message - must match SFT training format
    return [
        {"role": "user", "content": CHESS_PROMPT_TEMPLATE.format(fen=fen, legal_moves=legal_moves)},
    ]


def format_chess_prompt_string(fen: str) -> str:
    """String format for FEN extraction in rewards."""
    try:
        board = chess.Board(fen)
        legal_moves = ", ".join(sorted([m.uci() for m in board.legal_moves]))
    except ValueError:
        # Invalid FEN - return empty legal moves
        legal_moves = ""

    return CHESS_PROMPT_TEMPLATE.format(fen=fen, legal_moves=legal_moves)


def extract_uci_move(text: str) -> Optional[str]:
    """Extract UCI move from model output."""
    match = re.search(
        r'<uci_move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</uci_move>',
        text,
        re.IGNORECASE
    )
    return match.group(1).lower() if match else None


# ============================================================================
# Stockfish Reward Functions
# ============================================================================

class StockfishRewardEngine:
    """Stockfish-based reward computation with automatic recovery."""

    MAX_RETRIES = 3

    def __init__(self, config: ChessGRPOConfig):
        self.config = config
        self.engine = None
        self._init_engine()

        # Stats tracking
        self.stats = {
            "total": 0, "top1": 0, "top3": 0, "top5": 0,
            "legal": 0, "illegal": 0, "no_move": 0, "engine_restarts": 0
        }

    def _init_engine(self):
        """Initialize or reinitialize Stockfish engine."""
        # Clean up existing engine if any
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
            self.engine.configure({
                "Threads": self.config.stockfish_threads,
                "Hash": self.config.stockfish_hash_mb,
            })
            logger.info(f"Stockfish initialized: depth={self.config.stockfish_depth}")
        except Exception as e:
            logger.error(f"Failed to init Stockfish: {e}")
            raise

    def _ensure_engine_alive(self) -> bool:
        """Check if engine is alive and restart if necessary."""
        if self.engine is None:
            self._init_engine()
            self.stats["engine_restarts"] += 1
            return True

        try:
            # Ping the engine to check if it's responsive
            self.engine.ping()
            return True
        except Exception as e:
            logger.warning(f"Stockfish engine unresponsive: {e}. Restarting...")
            self._init_engine()
            self.stats["engine_restarts"] += 1
            return True
    
    def get_top_moves(self, fen: str, n: int = 5) -> List[tuple]:
        """Get top N moves with scores. Includes retry logic for engine failures."""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                self._ensure_engine_alive()
                board = chess.Board(fen)
                result = self.engine.analyse(
                    board,
                    chess.engine.Limit(depth=self.config.stockfish_depth),
                    multipv=n
                )
                if isinstance(result, dict):
                    result = [result]

                moves = []
                for pv in result:
                    if "pv" in pv and pv["pv"]:
                        move = pv["pv"][0].uci()
                        pov_score = pv.get("score")
                        cp = 0
                        if pov_score is not None:
                            relative = getattr(pov_score, "relative", pov_score)
                            if relative.is_mate():
                                mate_in = relative.mate()
                                cp = 30000 if mate_in and mate_in > 0 else -30000
                            else:
                                score_cp = relative.score()
                                cp = int(score_cp) if score_cp is not None else 0
                        moves.append((move, cp))
                return moves

            except chess.engine.EngineTerminatedError as e:
                last_error = e
                logger.warning(f"Engine terminated on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
                self.engine = None  # Force restart on next attempt

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Analysis attempt {attempt + 1}/{self.MAX_RETRIES} failed for {fen}: {e}")
                else:
                    logger.warning(f"Analysis failed after {self.MAX_RETRIES} attempts for {fen}: {e}")

        return []
    
    def compute_single_reward(self, fen: str, predicted_move: Optional[str]) -> float:
        """Compute reward for a single prediction."""
        self.stats["total"] += 1
        
        if predicted_move is None:
            self.stats["no_move"] += 1
            return self.config.reward_no_move
        
        try:
            board = chess.Board(fen)
        except ValueError:
            # Invalid FEN string
            self.stats["no_move"] += 1
            return self.config.reward_no_move
        
        # Check legality
        try:
            move = chess.Move.from_uci(predicted_move)
            if move not in board.legal_moves:
                self.stats["illegal"] += 1
                return self.config.reward_illegal
        except ValueError:
            self.stats["illegal"] += 1
            return self.config.reward_illegal
        
        # Get Stockfish analysis
        top_moves = self.get_top_moves(fen)
        if not top_moves:
            self.stats["legal"] += 1
            return self.config.reward_legal
        
        # Check ranking
        for rank, (sf_move, _) in enumerate(top_moves, 1):
            if sf_move == predicted_move:
                if rank == 1:
                    self.stats["top1"] += 1
                    return self.config.reward_top1
                elif rank <= 3:
                    self.stats["top3"] += 1
                    return self.config.reward_top3
                else:
                    self.stats["top5"] += 1
                    return self.config.reward_top5
        
        self.stats["legal"] += 1
        return self.config.reward_legal
    
    def log_stats(self):
        total = self.stats["total"]
        if total > 0:
            logger.info(
                f"Rewards - Top1: {self.stats['top1']/total:.1%}, "
                f"Top3: {self.stats['top3']/total:.1%}, "
                f"Top5: {self.stats['top5']/total:.1%}, "
                f"Legal: {self.stats['legal']/total:.1%}, "
                f"Illegal: {self.stats['illegal']/total:.1%}, "
                f"Engine restarts: {self.stats['engine_restarts']}"
            )
    
    def close(self):
        if self.engine:
            self.engine.quit()


def create_reward_functions(config: ChessGRPOConfig):
    """
    Create reward functions for GRPO.
    
    Returns multiple reward functions that are summed by GRPOTrainer.
    """
    stockfish = StockfishRewardEngine(config)
    
    def _extract_prompt_text(prompt) -> str:
        """Extract text content from prompt (handles both string and chat format)."""
        if isinstance(prompt, list):
            # Chat format: list of message dicts
            # Find the user message which contains the FEN
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            # Fallback: concatenate all content
            return " ".join(msg.get("content", "") for msg in prompt if isinstance(msg, dict))
        return str(prompt)

    # Debug counter for logging samples
    _debug_counter = {"count": 0}

    # Main Stockfish reward function
    def stockfish_reward_func(completions, prompts, **kwargs):
        """
        Stockfish-based move quality reward.

        Args:
            completions: List of model completions (string or chat format)
            prompts: List of prompts (string or chat format, contains FEN)

        Returns:
            List of float rewards
        """
        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Handle completion format (could be string or list of dicts)
            if isinstance(completion, list):
                text = completion[0].get("content", "") if completion else ""
            else:
                text = str(completion)

            # Debug: print first few completions to see what model is generating
            _debug_counter["count"] += 1
            if _debug_counter["count"] <= 5:
                print(f"\n{'='*60}", flush=True)
                print(f"[REWARD DEBUG {_debug_counter['count']}/5]", flush=True)
                print(f"Completion type: {type(completion)}", flush=True)
                print(f"Completion ({len(text)} chars):", flush=True)
                print(text[:800], flush=True)
                if len(text) > 800:
                    print(f"... [truncated] ...", flush=True)
                    print(text[-200:], flush=True)

            # Extract FEN from prompt (handles both string and chat format)
            prompt_text = _extract_prompt_text(prompt)
            fen_match = re.search(r'FEN:\s*([^\n]+)', prompt_text)
            if not fen_match:
                if _debug_counter["count"] <= 5:
                    print(f"[REWARD DEBUG] No FEN found in prompt!", flush=True)
                    print(f"Prompt: {prompt_text[:300]}...", flush=True)
                rewards.append(config.reward_no_move)
                continue

            fen = fen_match.group(1).strip()
            predicted_move = extract_uci_move(text)

            if _debug_counter["count"] <= 5:
                print(f"FEN: {fen}", flush=True)
                print(f"Extracted move: {predicted_move}", flush=True)
                print(f"{'='*60}\n", flush=True)

            reward = stockfish.compute_single_reward(fen, predicted_move)
            rewards.append(reward)

        return rewards
    
    # Format reward - encourages proper <think>...</think><uci_move>...</uci_move> format
    def format_reward_func(completions, **kwargs):
        """Reward for proper output format."""
        pattern = r'<think>[\s\S]*?</think>\s*<uci_move>[a-h][1-8][a-h][1-8][qrbn]?</uci_move>'
        
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                text = completion[0].get("content", "") if completion else ""
            else:
                text = str(completion)
            
            # Check format
            if re.search(pattern, text, re.IGNORECASE):
                rewards.append(config.reward_format_bonus)  # Small bonus for correct format
            else:
                rewards.append(config.reward_format_penalty)  # Small penalty for wrong format
        
        return rewards
    
    # XML tag count reward - ensures exactly one of each tag
    def xml_count_reward_func(completions, **kwargs):
        """Reward for having exactly one of each XML tag."""
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                text = completion[0].get("content", "") if completion else ""
            else:
                text = str(completion)
            
            think_count = len(re.findall(r'<think>', text, re.IGNORECASE))
            think_close = len(re.findall(r'</think>', text, re.IGNORECASE))
            move_count = len(re.findall(r'<uci_move>', text, re.IGNORECASE))
            move_close = len(re.findall(r'</uci_move>', text, re.IGNORECASE))
            
            if think_count == 1 and think_close == 1 and move_count == 1 and move_close == 1:
                rewards.append(config.reward_xml_bonus)
            else:
                rewards.append(config.reward_xml_penalty)
        
        return rewards
    
    # Attach cleanup
    stockfish_reward_func.stockfish = stockfish
    stockfish_reward_func.close = stockfish.close
    stockfish_reward_func.log_stats = stockfish.log_stats
    
    return [stockfish_reward_func, format_reward_func, xml_count_reward_func]


# ============================================================================
# Main Training Function
# ============================================================================

def main(config: Optional[ChessGRPOConfig] = None):
    """Main training function optimized for single H100."""
    
    if config is None:
        config = ChessGRPOConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)

    config.stockfish_path = resolve_stockfish_path(config.stockfish_path)
    if not config.stockfish_path:
        raise SystemExit("Stockfish not found. Install it or set `stockfish.path` / pass `--stockfish-path`.")
    
    # ========================================================================
    # Load Model with Unsloth
    # ========================================================================
    logger.info(f"Loading model: {config.model_name}")
    logger.info(f"Mode: {'LoRA' if config.use_lora else 'Full Fine-tuning'}")
    logger.info(f"Using Unsloth with vLLM fast inference")

    # FP8 KV cache requires FP8 model weights (FlashInfer limitation)
    # Mixed BF16 queries + FP8 KV cache causes compilation errors
    use_fp8_kv = config.use_fp8_kv_cache
    if use_fp8_kv and not config.use_fp8:
        logger.warning("FP8 KV cache requires FP8 model weights. Disabling FP8 KV cache.")
        use_fp8_kv = False

    # Build model loading kwargs
    model_kwargs = {
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
        "load_in_fp8": config.use_fp8,
        "fast_inference": config.use_vllm,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "float8_kv_cache": use_fp8_kv,
    }
    # Only set max_lora_rank when using LoRA
    if config.use_lora:
        model_kwargs["max_lora_rank"] = config.lora_r

    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)

    # ========================================================================
    # Setup Tokenizer (must match SFT training!)
    # ========================================================================
    # Custom chat template from SFT - adds <think> at generation prompt
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
    tokenizer.chat_template = CHAT_TEMPLATE
    logger.info("Set custom chat template (matches SFT)")

    # Add special tokens if not present
    special_tokens = ["<uci_move>", "</uci_move>"]
    tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added special tokens: {tokens_to_add}")

    # ========================================================================
    # CRITICAL: Patch tokenizer to preserve special tokens in completions
    # ========================================================================
    # TRL's GRPOTrainer hardcodes skip_special_tokens=True in batch_decode,
    # which strips our <uci_move> tokens before they reach the reward function.
    # See: https://github.com/huggingface/trl/issues/2897
    original_batch_decode = tokenizer.batch_decode
    def patched_batch_decode(*args, **kwargs):
        # Force skip_special_tokens=False to preserve <uci_move> tags
        kwargs["skip_special_tokens"] = False
        return original_batch_decode(*args, **kwargs)
    tokenizer.batch_decode = patched_batch_decode

    original_decode = tokenizer.decode
    def patched_decode(*args, **kwargs):
        kwargs["skip_special_tokens"] = False
        return original_decode(*args, **kwargs)
    tokenizer.decode = patched_decode

    logger.info("Patched tokenizer to preserve special tokens (fixes TRL issue #2897)")

    # ========================================================================
    # Apply LoRA or Enable Full Fine-tuning
    # ========================================================================
    if config.use_lora:
        logger.info("Applying LoRA configuration...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            use_gradient_checkpointing="unsloth" if config.use_gradient_checkpointing else False,
            random_state=42,
            max_seq_length=config.max_seq_length,
        )
    else:
        logger.info("Full fine-tuning mode (no LoRA)")
        # For full fine-tuning, we need gradient checkpointing to fit in memory
        if config.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for full fine-tuning...")
            model.gradient_checkpointing_enable()
        # Ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # Create Dataset
    # ========================================================================
    logger.info("Creating training dataset...")
    dataset = create_dataset(config, seed=42)
    
    # Split
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # ========================================================================
    # Create Reward Functions
    # ========================================================================
    logger.info("Creating Stockfish reward functions...")
    reward_funcs = create_reward_functions(config)
    
    # ========================================================================
    # GRPO Configuration for H100
    # ========================================================================
    logger.info("Configuring GRPO trainer...")
    
    training_args = GRPOConfig(
        # Output
        output_dir=config.output_dir,
        run_name="chess-grpo-h100",
        
        # Batch settings
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # GRPO specific
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        beta=config.beta,
        # 2025 best practices: batch-level reward scaling and mask truncated outputs
        scale_rewards=config.scale_rewards,
        mask_truncated_completions=config.mask_truncated_completions,

        # DAPO loss configuration (2025 best practice for reasoning tasks)
        loss_type=config.loss_type,
        epsilon=config.epsilon,
        epsilon_high=config.epsilon_high,

        # vLLM integration (Unsloth handles this)
        use_vllm=config.use_vllm,
        
        # Optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        
        # Training duration
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        
        # Precision
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        
        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        max_grad_norm=config.max_grad_norm,
        
        # Misc
        remove_unused_columns=False,
        report_to=["wandb", "tensorboard"] if config.use_wandb else ["tensorboard"],
    )

    # ========================================================================
    # Initialize Wandb
    # ========================================================================
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"chess-grpo-{config.model_name.split('/')[-1]}",
            entity=config.wandb_entity,
            config={
                "model_name": config.model_name,
                "max_seq_length": config.max_seq_length,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "learning_rate": config.learning_rate,
                "num_generations": config.num_generations,
                "max_completion_length": config.max_completion_length,
                "temperature": config.temperature,
                "beta": config.beta,
                "loss_type": config.loss_type,
                "epsilon": config.epsilon,
                "epsilon_high": config.epsilon_high,
                "num_train_samples": config.num_train_samples,
                "stockfish_depth": config.stockfish_depth,
                "reward_top1": config.reward_top1,
                "reward_top3": config.reward_top3,
                "reward_top5": config.reward_top5,
            },
            reinit=True,
        )
        logger.info(f"Wandb initialized: {wandb.run.name}")
    
    # ========================================================================
    # Initialize Trainer
    # ========================================================================
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # ========================================================================
    # Train
    # ========================================================================
    logger.info("Starting GRPO training on H100...")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Num generations: {config.num_generations}")
    logger.info(f"  Max completion length: {config.max_completion_length}")
    
    try:
        trainer.train()

        # Log final stats
        reward_funcs[0].log_stats()

        # Log final stats to wandb
        if config.use_wandb and wandb.run:
            stats = reward_funcs[0].stockfish.stats
            total = stats["total"] if stats["total"] > 0 else 1
            wandb.log({
                "final/top1_rate": stats["top1"] / total,
                "final/top3_rate": stats["top3"] / total,
                "final/top5_rate": stats["top5"] / total,
                "final/legal_rate": stats["legal"] / total,
                "final/illegal_rate": stats["illegal"] / total,
                "final/no_move_rate": stats["no_move"] / total,
                "final/engine_restarts": stats["engine_restarts"],
            })

        # Save model
        logger.info("Saving model...")

        if config.use_lora:
            # Save LoRA weights
            model.save_lora(os.path.join(config.output_dir, "lora_weights"))
            logger.info(f"LoRA weights saved to {config.output_dir}/lora_weights")

            # Optionally merge and save full model
            # model.save_pretrained_merged(
            #     os.path.join(config.output_dir, "merged_model"),
            #     tokenizer,
            #     save_method="merged_16bit",
            # )
        else:
            # Save full model for full fine-tuning
            save_path = os.path.join(config.output_dir, "model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Full model saved to {save_path}")

        logger.info(f"Training complete! Model saved to {config.output_dir}")

    except KeyboardInterrupt:
        logger.info("Training interrupted")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        reward_funcs[0].close()
        if config.use_wandb and wandb.run:
            wandb.finish()


if __name__ == "__main__":
    import argparse
    
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Chess GRPO Training with Unsloth on H100")
    parser.add_argument("--config", "-c", default=str(script_dir / "config_h100.yaml"), help="YAML config file")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--stockfish-path", "--stockfish_path", dest="stockfish_path", type=str, default=None)
    parser.add_argument("--stockfish-depth", "--stockfish_depth", dest="stockfish_depth", type=int, default=None)
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=None)
    parser.add_argument("--num-generations", "--num_generations", dest="num_generations", type=int, default=None)
    parser.add_argument("--use-fp8", "--use_fp8", dest="use_fp8", action="store_true", help="Use FP8 for faster training")
    parser.add_argument("--full-finetune", "--full_finetune", dest="full_finetune", action="store_true", help="Full fine-tuning (no LoRA)")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        candidate = (script_dir / config_path).resolve()
        if candidate.exists():
            config_path = candidate

    if config_path.exists():
        logger.info("Loading config: %s", config_path)
        config = ChessGRPOConfig.from_yaml(config_path)
    else:
        logger.warning("Config not found: %s (using defaults)", config_path)
        config = ChessGRPOConfig()

    if args.model:
        config.model_name = args.model
    if args.stockfish_path:
        config.stockfish_path = args.stockfish_path
    if args.stockfish_depth is not None:
        config.stockfish_depth = args.stockfish_depth
    if args.num_samples is not None:
        config.num_train_samples = args.num_samples
    if args.num_generations is not None:
        config.num_generations = args.num_generations
    if args.use_fp8:
        config.use_fp8 = True
    if args.full_finetune:
        config.use_lora = False
    if args.output_dir:
        config.output_dir = args.output_dir

    main(config)
