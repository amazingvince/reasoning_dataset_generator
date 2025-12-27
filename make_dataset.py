#!/usr/bin/env python3
"""
Standalone dataset-generation CLI (no training).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def find_stockfish() -> Optional[str]:
    candidates = [
        shutil.which("stockfish"),
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.warning("Config not found: %s (using defaults)", path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a streaming JSONL chess dataset with Stockfish traces.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "config_stockfish_traces_full.yaml"),
        help="Path to YAML config (relative paths are resolved from the current working directory).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(Path("data") / "chess_traces.jsonl"),
        help="Path to output JSONL file (one example per line).",
    )
    parser.add_argument("--size", type=int, default=None, help="Number of positions to generate")
    parser.add_argument("--depth", type=int, default=None, help="Stockfish search depth")
    parser.add_argument("--workers", type=int, default=None, help="Number of Stockfish workers (processes)")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish binary")
    parser.add_argument("--games-ratio", type=float, default=None, help="Ratio of game positions (vs puzzles)")
    parser.add_argument("--multipv", type=int, default=None, help="Cap MultiPV moves (0 = all moves)")
    parser.add_argument(
        "--jsonl-max-in-flight",
        type=int,
        default=0,
        help="Max concurrent positions being analyzed when writing JSONL (0 = auto).",
    )
    parser.add_argument(
        "--jsonl-flush-every",
        type=int,
        default=100,
        help="Flush JSONL output every N written lines.",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Optional HuggingFace *dataset* repo id for incremental uploads (e.g. 'user/chess-traces').",
    )
    parser.add_argument("--hub-private", action="store_true", help="Create/upload to a private Hub dataset repo.")
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="HuggingFace token (defaults to HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var).",
    )
    parser.add_argument(
        "--hub-upload-every-minutes",
        type=int,
        default=None,
        help="Upload sealed JSONL shards every N minutes (overrides config).",
    )
    parser.add_argument(
        "--hub-upload-every-seconds",
        type=int,
        default=None,
        help="Upload sealed JSONL shards every N seconds (overrides config).",
    )
    parser.add_argument(
        "--hub-shard-max-lines",
        type=int,
        default=None,
        help="Rotate/upload shard once it reaches N lines (0 = disable, overrides config).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-verify-stockfish", action="store_true", help="Skip Stockfish verification.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    from src.utils.stockfish_eval import StockfishEvaluator

    config = load_yaml_config(Path(args.config))

    stockfish_cfg = config.get("stockfish", {})
    data_cfg = config.get("data", {})
    hub_cfg = config.get("hub", {}) or {}

    target_size = args.size or data_cfg.get("target_size", 100_000)
    depth = args.depth or stockfish_cfg.get("depth", 12)
    workers = args.workers or stockfish_cfg.get("workers", 8)
    stockfish_path = args.stockfish_path or stockfish_cfg.get("path")
    games_ratio = args.games_ratio or data_cfg.get("games_ratio", 0.7)
    multipv = args.multipv if args.multipv is not None else stockfish_cfg.get("multipv")
    if multipv == 0:
        multipv = None
    if multipv is None:
        rt_cfg = config.get("reasoning_trace", {}) or {}
        multipv = rt_cfg.get("candidate_pool_size", 12)
        try:
            multipv = int(multipv)
        except (TypeError, ValueError):
            multipv = 12

    if stockfish_path is None:
        stockfish_path = find_stockfish()
    if stockfish_path is None:
        raise SystemExit(
            "Stockfish not found. Install it or pass --stockfish-path."
        )

    logger.info("JSONL path: %s", args.output_jsonl)
    logger.info("Target size: %s", f"{int(target_size):,}")
    logger.info("Games ratio: %.1f%%", float(games_ratio) * 100)
    logger.info("Stockfish: %s (depth=%s, workers=%s)", stockfish_path, depth, workers)
    logger.info("MultiPV: %s", multipv or "all moves")

    hub_repo_id = args.hub_repo_id
    if hub_repo_id is None:
        hub_repo_id = hub_cfg.get("repo_id") or None
    hub_private = bool(args.hub_private) or bool(hub_cfg.get("private", False))
    hub_token = args.hub_token

    hub_upload_interval_seconds = None
    if args.hub_upload_every_seconds is not None:
        hub_upload_interval_seconds = int(args.hub_upload_every_seconds)
    elif args.hub_upload_every_minutes is not None:
        hub_upload_interval_seconds = int(args.hub_upload_every_minutes) * 60
    else:
        try:
            hub_upload_interval_seconds = int(hub_cfg.get("upload_interval_seconds", 900))
        except (TypeError, ValueError):
            hub_upload_interval_seconds = 900

    hub_shard_max_lines = None
    if args.hub_shard_max_lines is not None:
        hub_shard_max_lines = int(args.hub_shard_max_lines)
    else:
        try:
            hub_shard_max_lines = int(hub_cfg.get("shard_max_lines", 0))
        except (TypeError, ValueError):
            hub_shard_max_lines = 0

    if not args.no_verify_stockfish:
        logger.info("Verifying Stockfish...")
        try:
            with StockfishEvaluator(stockfish_path=stockfish_path, depth=int(depth)) as evaluator:
                import chess

                analysis = evaluator.analyze_position(chess.Board(), multipv=5)
                logger.info("Stockfish OK (best move in start position: %s)", analysis.best_move_san)
        except Exception as exc:
            raise SystemExit(f"Stockfish verification failed: {exc}") from exc

    from src.utils.jsonl_export import write_jsonl_with_traces

    write_jsonl_with_traces(
        output_jsonl=args.output_jsonl,
        target_size=int(target_size),
        games_ratio=float(games_ratio),
        stockfish_path=stockfish_path,
        stockfish_depth=int(depth),
        stockfish_workers=int(workers),
        multipv=int(multipv) if multipv is not None else None,
        config=config,
        seed=int(args.seed),
        max_in_flight=int(args.jsonl_max_in_flight),
        flush_every=int(args.jsonl_flush_every),
        hub_repo_id=str(hub_repo_id).strip() if hub_repo_id else None,
        hub_private=bool(hub_private),
        hub_token=hub_token,
        hub_upload_interval_seconds=int(hub_upload_interval_seconds or 900),
        hub_shard_max_lines=int(hub_shard_max_lines or 0),
    )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
