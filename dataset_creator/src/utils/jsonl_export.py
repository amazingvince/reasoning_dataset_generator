"""
Streaming JSONL export for Stockfish-annotated chess datasets.

This module writes one JSON object per line as examples are produced, which is
useful for long-running dataset generation jobs where you want output on disk
continuously (instead of materializing a full HuggingFace Dataset in memory).

Each row contains only the requested training columns:
  - fen: FEN string
  - valid_moves: list[str] of legal UCI moves
  - reasoning_trace: natural-language trace (no token budget by default)
  - chosen_move: the Stockfish best move (UCI)

Additional metadata (e.g. Lichess puzzle themes/ratings) may be used internally
to diversify the trace text, but is not written as extra dataset columns.
"""

from __future__ import annotations

import json
import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple

import chess
from datasets import load_dataset
from tqdm import tqdm

from src.chess_dataset.reasoning_trace import ReasoningTraceGenerator
from src.utils.chess_utils import parse_movetext
from src.utils.hub_uploader import HubDatasetUploader, resolve_hf_token
from src.utils.stockfish_eval import StockfishEvaluator
from src.utils.trace_analysis import position_with_eval_to_analysis

logger = logging.getLogger(__name__)


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pv_settings_for_traces(cfg: Dict[str, Any]) -> Tuple[int, int]:
    include_best_pv = bool(cfg.get("include_pv", True))
    include_candidate_pv = bool(cfg.get("include_candidate_pv", True))
    include_trap_detection = bool(cfg.get("include_trap_detection", False))
    if not (include_best_pv or include_candidate_pv or include_trap_detection):
        return (0, 0)

    max_pv_length = _parse_int(cfg.get("max_pv_length", 8), 8)
    candidate_pv_max_len = _parse_int(cfg.get("candidate_pv_max_len", 4), 4)
    trap_refutation_max_len = _parse_int(cfg.get("trap_refutation_max_len", 4), 4)
    pv_max_len = max(0, max(max_pv_length, candidate_pv_max_len, trap_refutation_max_len))

    pv_store_top_k_raw = cfg.get("pv_store_top_k", cfg.get("candidate_pool_size", 12))
    pv_store_top_k = _parse_int(pv_store_top_k_raw, 12)

    max_candidates = _parse_int(cfg.get("max_candidates", 5), 5)
    pv_store_top_k = max(1, pv_store_top_k, max_candidates)
    return (pv_max_len, pv_store_top_k)


def _iter_game_positions(
    *,
    config: Dict[str, Any],
    seed: int,
) -> Iterator[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("games_dataset", "Lichess/standard-chess-games")
    min_elo = _parse_int(data_cfg.get("min_elo", 1200), 1200)
    sample_rate = _parse_float(data_cfg.get("sample_rate", 0.3), 0.3)
    skip_first = _parse_int(data_cfg.get("skip_first_moves", 4), 4)
    skip_last = _parse_int(data_cfg.get("skip_last_moves", 2), 2)

    elo_weights = config.get("elo_weights") or {
        1200: 0.05,
        1400: 0.10,
        1600: 0.15,
        1800: 0.25,
        2000: 0.30,
        2200: 0.20,
        2400: 0.10,
    }

    rng = random.Random(seed)

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    for game in dataset:
        if game.get("Termination") == "Time forfeit":
            continue
        result = game.get("Result", "")
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue

        white_elo = game.get("WhiteElo", 1500) or 1500
        black_elo = game.get("BlackElo", 1500) or 1500
        white_elo = _parse_int(white_elo, 1500)
        black_elo = _parse_int(black_elo, 1500)
        avg_elo = (white_elo + black_elo) / 2
        if avg_elo < min_elo:
            continue

        weight = 0.01
        for threshold in sorted(elo_weights.keys(), reverse=True):
            try:
                threshold_int = int(threshold)
            except (TypeError, ValueError):
                continue
            if avg_elo >= threshold_int:
                weight = float(elo_weights[threshold])
                break
        if rng.random() > float(weight):
            continue

        movetext = game.get("movetext", "") or game.get("moves", "")
        if not movetext:
            continue

        moves = parse_movetext(movetext)
        if len(moves) < skip_first + skip_last + 1:
            continue

        board = chess.Board()
        for move_idx, move_san in enumerate(moves):
            if move_idx < skip_first:
                try:
                    board.push_san(move_san)
                except Exception:
                    break
                continue

            if move_idx >= len(moves) - skip_last:
                break

            try:
                if rng.random() <= sample_rate:
                    yield {
                        "fen": board.fen(),
                        "source": "game",
                    }
                board.push_san(move_san)
            except Exception:
                break


def _iter_puzzle_positions(
    *,
    config: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("puzzles_dataset", "Lichess/chess-puzzles")
    min_rating = _parse_int(data_cfg.get("min_puzzle_rating", 1000), 1000)
    max_rating = _parse_int(data_cfg.get("max_puzzle_rating", 2500), 2500)

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    for puzzle in dataset:
        puzzle_id = puzzle.get("PuzzleId")
        rating = puzzle.get("Rating", 1500) or 1500
        rating = _parse_int(rating, 1500)
        if rating < min_rating or rating > max_rating:
            continue

        themes = puzzle.get("Themes") or []
        if isinstance(themes, str):
            themes = [t for t in themes.split() if t]
        themes = [str(t) for t in themes if t]

        opening_tags = puzzle.get("OpeningTags") or []
        if isinstance(opening_tags, str):
            opening_tags = [opening_tags]
        opening_tags = [str(t) for t in opening_tags if t]

        fen = puzzle.get("FEN")
        moves_str = puzzle.get("Moves", "")
        if not fen or not moves_str:
            continue

        moves = moves_str.split()
        if len(moves) < 2:
            continue

        try:
            board = chess.Board(fen)
        except Exception:
            continue

        for idx, move_uci in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break

            if idx % 2 == 1:
                yield {
                    "fen": board.fen(),
                    "source": "puzzle",
                    "puzzle_id": str(puzzle_id) if puzzle_id is not None else None,
                    "puzzle_rating": rating,
                    "puzzle_themes": themes,
                    "puzzle_opening_tags": opening_tags,
                    "puzzle_move_index": idx,
                    "puzzle_line_length": len(moves),
                }

            board.push(move)


def write_jsonl_with_traces(
    *,
    output_jsonl: str | Path,
    target_size: int,
    games_ratio: float,
    stockfish_path: str,
    stockfish_depth: int,
    stockfish_workers: int,
    multipv: Optional[int],
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    max_in_flight: int = 0,
    flush_every: int = 100,
    hub_repo_id: Optional[str] = None,
    hub_private: bool = False,
    hub_token: Optional[str] = None,
    hub_upload_interval_seconds: int = 900,
    hub_shard_max_lines: int = 0,
) -> None:
    """
    Stream positions from Lichess datasets, annotate with Stockfish + traces, and write JSONL.

    The JSONL file is written incrementally as examples complete, so you can
    monitor progress and resume downstream processing immediately.
    """

    cfg = config or {}
    games_ratio = float(games_ratio)
    games_target = int(target_size * games_ratio)
    puzzles_target = int(target_size) - games_target

    trace_cfg = dict(cfg.get("reasoning_trace", {}) or {})
    trace_cfg.setdefault("seed", seed)
    trace_cfg["max_trace_tokens"] = None  # requested: no cutoff
    source_overrides = trace_cfg.get("source_overrides")
    if isinstance(source_overrides, dict):
        patched: Dict[str, Any] = {}
        for key, override in source_overrides.items():
            override_dict = dict(override or {})
            override_dict["max_trace_tokens"] = None
            patched[str(key)] = override_dict
        trace_cfg["source_overrides"] = patched

    trace_generator = ReasoningTraceGenerator(trace_cfg, tokenizer=None)

    distill_cfg = cfg.get("distillation", {}) or {}
    min_probability = float(distill_cfg.get("min_probability", 0.001))
    stockfish_temperature = float(distill_cfg.get("stockfish_temperature", 100.0))

    stockfish_cfg = cfg.get("stockfish", {}) or {}
    threads_per_worker = _parse_int(stockfish_cfg.get("threads_per_worker", stockfish_cfg.get("threads", 1)), 1)
    hash_mb = _parse_int(stockfish_cfg.get("hash_mb", stockfish_cfg.get("hash_mb_per_worker", 128)), 128)

    pv_max_len, pv_store_top_k = _pv_settings_for_traces(trace_cfg)

    trap_detection_enabled = bool(trace_cfg.get("include_trap_detection", False))
    trap_probe_top_k_raw = trace_cfg.get("trap_probe_top_k", trace_cfg.get("candidate_pool_size", 12))
    trap_probe_top_k = max(0, _parse_int(trap_probe_top_k_raw, 12))
    shallow_depth_default = max(4, int(stockfish_depth) // 2)
    shallow_depth = _parse_int(stockfish_cfg.get("shallow_depth", shallow_depth_default), shallow_depth_default)
    confirm_depth_default = int(stockfish_depth) + 2
    confirm_depth = _parse_int(stockfish_cfg.get("confirm_depth", confirm_depth_default), confirm_depth_default)
    if not trap_detection_enabled:
        shallow_depth = 0
        confirm_depth = 0

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_in_flight <= 0:
        max_in_flight = max(8, int(stockfish_workers) * 4)
    flush_every = max(1, int(flush_every))

    hub_cfg = dict((cfg.get("hub") or {}) if isinstance(cfg.get("hub"), dict) else {})
    if hub_repo_id is None:
        hub_repo_id = hub_cfg.get("repo_id") or hub_cfg.get("dataset_repo_id")
    if hub_repo_id is not None:
        hub_repo_id = str(hub_repo_id).strip() or None
    if hub_private is False and bool(hub_cfg.get("private", False)):
        hub_private = True
    hub_upload_interval_seconds = _parse_int(
        hub_cfg.get("upload_interval_seconds", hub_upload_interval_seconds),
        int(hub_upload_interval_seconds),
    )
    hub_shard_max_lines = _parse_int(hub_cfg.get("shard_max_lines", hub_shard_max_lines), int(hub_shard_max_lines))

    uploader: Optional[HubDatasetUploader] = None
    upload_queue: Optional[Queue[Optional["_UploadJob"]]] = None
    upload_thread: Optional[threading.Thread] = None
    shards_dir: Optional[Path] = None
    split_name = str(hub_cfg.get("split", "train") or "train")

    @dataclass
    class _UploadJob:
        local_path: Path
        path_in_repo: str
        commit_message: str

    def _dataset_card_markdown(repo_id: str) -> str:
        return "\n".join(
            [
                "---",
                "license: other",
                "task_categories:",
                "- text-generation",
                "tags:",
                "- chess",
                "- stockfish",
                "- reasoning",
                "pretty_name: Chess reasoning traces (streaming)",
                "---",
                "",
                "# Chess Reasoning Traces (Streaming)",
                "",
                "This dataset is uploaded incrementally while it is being generated.",
                "",
                "## Columns",
                "",
                "- `fen`: FEN string",
                "- `valid_moves`: list of legal moves in UCI",
                "- `reasoning_trace`: natural-language trace text",
                "- `chosen_move`: Stockfish best move in UCI",
                "",
                "## Usage",
                "",
                "```python",
                "from datasets import load_dataset",
                "",
                f"ds = load_dataset(\"{repo_id}\", split=\"train\")",
                "print(ds[0])",
                "",
                "# Or stream while it's growing:",
                f"stream = load_dataset(\"{repo_id}\", split=\"train\", streaming=True)",
                "print(next(iter(stream)))",
                "```",
                "",
                "## Notes",
                "",
                "- Data is stored as sharded JSONL files under `data/`.",
                "- New shards are committed periodically; re-run `load_dataset` to pick up new shards.",
                "",
            ]
        )

    if hub_repo_id:
        token = resolve_hf_token(hub_token)
        if not token:
            raise RuntimeError(
                "Hub upload enabled but no token found. Set HF_TOKEN / HUGGINGFACE_HUB_TOKEN or pass --hub-token."
            )
        uploader = HubDatasetUploader(repo_id=str(hub_repo_id), token=token, private=bool(hub_private))
        uploader.ensure_repo()
        uploader.ensure_default_files(dataset_card_markdown=_dataset_card_markdown(str(hub_repo_id)))

        shards_dir = output_path.parent / f"{output_path.stem}_shards"
        shards_dir.mkdir(parents=True, exist_ok=True)

        upload_queue = Queue()

        def _upload_loop() -> None:
            assert upload_queue is not None
            assert uploader is not None
            while True:
                job = upload_queue.get()
                try:
                    if job is None:
                        return
                    max_retries = _parse_int(hub_cfg.get("max_retries", 10), 10)
                    backoff = float(hub_cfg.get("retry_backoff_seconds", 30.0))
                    attempt = 0
                    while True:
                        try:
                            uploader.upload_shard(
                                local_path=job.local_path,
                                path_in_repo=job.path_in_repo,
                                commit_message=job.commit_message,
                            )
                            logger.info("Uploaded shard to Hub: %s -> %s", job.local_path.name, job.path_in_repo)
                            break
                        except Exception as exc:
                            attempt += 1
                            logger.warning(
                                "Hub upload failed (attempt %s/%s) for %s: %s",
                                attempt,
                                max_retries,
                                job.local_path,
                                exc,
                            )
                            if max_retries > 0 and attempt >= max_retries:
                                logger.error("Giving up uploading %s after %s attempts.", job.local_path, attempt)
                                break
                            time.sleep(min(max(backoff, 0.0) * attempt, 300.0))
                finally:
                    upload_queue.task_done()

        upload_thread = threading.Thread(target=_upload_loop, name="hf-hub-uploader", daemon=True)
        upload_thread.start()

    evaluator_local: threading.local = threading.local()
    evaluators: List[StockfishEvaluator] = []
    evaluators_lock = threading.Lock()

    def get_evaluator() -> StockfishEvaluator:
        evaluator = getattr(evaluator_local, "evaluator", None)
        if evaluator is None:
            evaluator = StockfishEvaluator(
                stockfish_path=stockfish_path,
                depth=int(stockfish_depth),
                threads=int(threads_per_worker),
                hash_mb=int(hash_mb),
            )
            evaluator_local.evaluator = evaluator
            with evaluators_lock:
                evaluators.append(evaluator)
        return evaluator

    def process_one(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fen = raw.get("fen")
        if not fen:
            return None
        try:
            board = chess.Board(str(fen))
        except Exception:
            return None

        meta = {
            "puzzle_id": raw.get("puzzle_id"),
            "puzzle_rating": raw.get("puzzle_rating"),
            "puzzle_themes": raw.get("puzzle_themes"),
            "puzzle_opening_tags": raw.get("puzzle_opening_tags"),
            "puzzle_move_index": raw.get("puzzle_move_index"),
            "puzzle_line_length": raw.get("puzzle_line_length"),
        }

        valid_moves = [m.uci() for m in board.legal_moves]
        if not valid_moves:
            return None

        evaluator = get_evaluator()
        try:
            analysis = evaluator.analyze_position(
                board,
                multipv=multipv,
                pv_max_len=pv_max_len,
                pv_store_top_k=pv_store_top_k,
                include_position_eval=False,
            )
        except Exception:
            return None

        shallow_move_cps: Dict[str, int] = {}
        confirm_move_cps: Dict[str, int] = {}
        if trap_detection_enabled and trap_probe_top_k > 0:
            root_moves: List[chess.Move] = []
            for mv in analysis.move_evaluations[:trap_probe_top_k]:
                try:
                    root_moves.append(chess.Move.from_uci(mv.uci))
                except ValueError:
                    continue
            if shallow_depth > 0 and root_moves:
                try:
                    shallow_move_cps = evaluator.analyze_root_moves(board, root_moves, depth=shallow_depth)
                except Exception:
                    shallow_move_cps = {}
            if confirm_depth > 0 and root_moves:
                try:
                    confirm_move_cps = evaluator.analyze_root_moves(board, root_moves, depth=confirm_depth)
                except Exception:
                    confirm_move_cps = {}

        move_evaluations = [
            {
                "uci": mv.uci,
                "san": mv.san,
                "centipawn": mv.centipawn,
                "mate_in": mv.mate_in,
                "pv_uci": list(mv.pv_uci or []),
            }
            for mv in analysis.move_evaluations
        ]

        return {
            **meta,
            "fen": fen,
            "source": raw.get("source"),
            "valid_moves": valid_moves,
            "move_evaluations": move_evaluations,
            "best_move_uci": analysis.best_move_uci,
            "best_move_san": analysis.best_move_san,
            "best_score_cp": analysis.best_score_cp,
            "shallow_move_cps": shallow_move_cps,
            "confirm_move_cps": confirm_move_cps,
        }

    in_flight = {"game": 0, "puzzle": 0}
    written = {"game": 0, "puzzle": 0}

    def need_more(source: str) -> bool:
        if source == "game":
            return (written["game"] + in_flight["game"]) < games_target
        if source == "puzzle":
            return (written["puzzle"] + in_flight["puzzle"]) < puzzles_target
        return False

    pending: set[Future[Optional[Dict[str, Any]]]] = set()
    pending_meta: Dict[Future[Optional[Dict[str, Any]]], Tuple[str, str]] = {}

    def submit(raw: Dict[str, Any]) -> None:
        source = str(raw.get("source") or "")
        fut = executor.submit(process_one, raw)
        pending.add(fut)
        pending_meta[fut] = (source, str(raw.get("fen") or ""))
        if source in in_flight:
            in_flight[source] += 1

    def drain(block: bool) -> None:
        nonlocal next_upload_at, shard_lines
        if not pending:
            return
        done: set[Future[Optional[Dict[str, Any]]]] = set()
        if block:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
        else:
            done, _ = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
        for fut in done:
            pending.remove(fut)
            source, fen = pending_meta.pop(fut, ("", ""))
            if source in in_flight:
                in_flight[source] = max(0, in_flight[source] - 1)

            result = None
            try:
                result = fut.result()
            except Exception as exc:
                logger.warning("Worker failed (source=%s fen=%s): %s", source, fen, exc)
                continue

            if not result:
                continue

            src = str(result.get("source") or source or "")
            best = str(result.get("best_move_uci") or "")

            valid_moves = list(result.get("valid_moves") or [])
            chosen_move = best
            if not chosen_move:
                move_evaluations = list(result.get("move_evaluations") or [])
                chosen_move = str((move_evaluations[0] or {}).get("uci") or "") if move_evaluations else ""
            if chosen_move and chosen_move not in valid_moves:
                chosen_move = ""
            if not chosen_move and valid_moves:
                chosen_move = valid_moves[0]

            example = {
                "fen": result["fen"],
                "source": src,
                "legal_moves_uci": " ".join(valid_moves),
                "move_evaluations": result.get("move_evaluations") or [],
                "best_move_uci": result.get("best_move_uci") or "",
                "best_move_san": result.get("best_move_san") or "",
                "best_score_cp": result.get("best_score_cp") or 0,
            }
            for key in (
                "puzzle_id",
                "puzzle_rating",
                "puzzle_themes",
                "puzzle_opening_tags",
                "puzzle_move_index",
                "puzzle_line_length",
                "shallow_move_cps",
                "confirm_move_cps",
            ):
                if key in result:
                    example[key] = result.get(key)

            try:
                analysis = position_with_eval_to_analysis(
                    example,
                    min_probability=min_probability,
                    stockfish_temperature=stockfish_temperature,
                )
                digest = hashlib.sha256(f"{seed}|{example['fen']}".encode("utf-8")).digest()
                per_example_rng = random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
                reasoning_trace = trace_generator.generate(example, analysis, rng=per_example_rng)
            except Exception as exc:
                logger.warning("Analysis/trace pipeline failed (fen=%s): %s", example.get("fen"), exc)
                reasoning_trace = f"Play {chosen_move}."

            row = {
                "fen": example["fen"],
                "valid_moves": valid_moves,
                "reasoning_trace": reasoning_trace,
                "chosen_move": chosen_move,
            }

            # Enforce target counts without overshooting.
            if src == "game" and written["game"] >= games_target:
                continue
            if src == "puzzle" and written["puzzle"] >= puzzles_target:
                continue

            line = json.dumps(row, ensure_ascii=False) + "\n"
            handle.write(line)
            if shard_handle is not None:
                try:
                    shard_handle.write(line)
                    shard_lines += 1
                except Exception:
                    pass

            total_written = written["game"] + written["puzzle"] + 1
            if total_written % flush_every == 0:
                handle.flush()
                if shard_handle is not None:
                    try:
                        shard_handle.flush()
                    except Exception:
                        pass

            if src in written:
                written[src] += 1
            pbar.update(1)

            if hub_repo_id and shard_handle is not None:
                now = time.monotonic()
                if hub_shard_max_lines > 0 and shard_lines >= hub_shard_max_lines:
                    _seal_and_enqueue_shard(reason="max_lines", total_written=total_written)
                    if int(hub_upload_interval_seconds) > 0 and now >= next_upload_at:
                        next_upload_at = now + int(hub_upload_interval_seconds)
                elif int(hub_upload_interval_seconds) > 0 and now >= next_upload_at:
                    _seal_and_enqueue_shard(reason="interval", total_written=total_written)
                    next_upload_at = now + int(hub_upload_interval_seconds)

    logger.info(
        "Writing JSONL: %s examples (%s games, %s puzzles) -> %s",
        f"{int(target_size):,}",
        f"{int(games_target):,}",
        f"{int(puzzles_target):,}",
        str(output_path),
    )
    if hub_repo_id:
        logger.info(
            "Hub upload: repo=%s (interval=%ss, shard_max_lines=%s, split=%s)",
            hub_repo_id,
            int(hub_upload_interval_seconds),
            hub_shard_max_lines or "disabled",
            split_name,
        )
    logger.info(
        "Stockfish: %s (depth=%s, workers=%s, threads/worker=%s, hash_mb=%s, multipv=%s)",
        stockfish_path,
        stockfish_depth,
        stockfish_workers,
        threads_per_worker,
        hash_mb,
        multipv or "all moves",
    )

    game_iter = _iter_game_positions(config=cfg, seed=seed)
    puzzle_iter = _iter_puzzle_positions(config=cfg)

    shard_handle = None
    shard_path: Optional[Path] = None
    shard_index = 0
    shard_lines = 0
    next_upload_at = time.monotonic() + max(1, int(hub_upload_interval_seconds or 900))

    def _open_next_shard() -> None:
        nonlocal shard_handle, shard_path, shard_index, shard_lines
        if shards_dir is None:
            return
        if shard_handle is not None:
            try:
                shard_handle.flush()
            except Exception:
                pass
            try:
                shard_handle.close()
            except Exception:
                pass
        shard_path = shards_dir / f"{split_name}-{shard_index:06d}.jsonl"
        shard_index += 1
        shard_lines = 0
        shard_handle = shard_path.open("w", encoding="utf-8")

    def _seal_and_enqueue_shard(*, reason: str, total_written: int) -> None:
        nonlocal shard_handle, shard_path, shard_lines
        if uploader is None or upload_queue is None or shards_dir is None:
            return
        if shard_handle is None or shard_path is None or shard_lines <= 0:
            return
        try:
            shard_handle.flush()
        except Exception:
            pass
        try:
            shard_handle.close()
        except Exception:
            pass
        shard_handle = None
        path_in_repo = f"data/{shard_path.name}"
        message = f"Add {split_name} shard ({reason}) - {total_written:,} examples"
        upload_queue.put(
            _UploadJob(
                local_path=shard_path,
                path_in_repo=path_in_repo,
                commit_message=message,
            )
        )
        _open_next_shard()

    try:
        with output_path.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(
            max_workers=int(stockfish_workers)
        ) as executor, tqdm(total=int(target_size), desc="JSONL") as pbar:
            if hub_repo_id:
                _open_next_shard()
            # Games first, then puzzles (keeps the ratio exact while still streaming output).
            while written["game"] < games_target:
                while need_more("game") and len(pending) < max_in_flight:
                    try:
                        raw = next(game_iter)
                    except StopIteration:
                        raise RuntimeError("Ran out of game positions before reaching target.") from None
                    submit(raw)
                    drain(block=False)
                drain(block=True)

            while written["puzzle"] < puzzles_target:
                while need_more("puzzle") and len(pending) < max_in_flight:
                    try:
                        raw = next(puzzle_iter)
                    except StopIteration:
                        raise RuntimeError("Ran out of puzzle positions before reaching target.") from None
                    submit(raw)
                    drain(block=False)
                drain(block=True)

            while pending:
                drain(block=True)

            handle.flush()
    finally:
        if shard_handle is not None:
            try:
                shard_handle.flush()
            except Exception:
                pass
            try:
                shard_handle.close()
            except Exception:
                pass
            shard_handle = None

        # Enqueue the last shard if it has data.
        if hub_repo_id and shard_path is not None and shard_lines > 0:
            try:
                assert upload_queue is not None
                upload_queue.put(
                    _UploadJob(
                        local_path=shard_path,
                        path_in_repo=f"data/{shard_path.name}",
                        commit_message=f"Add {split_name} shard (final) - {written['game'] + written['puzzle']:,} examples",
                    )
                )
            except Exception:
                pass

        if upload_queue is not None:
            try:
                upload_queue.put(None)
                upload_queue.join()
            except Exception:
                pass
        if upload_thread is not None:
            try:
                upload_thread.join(timeout=5)
            except Exception:
                pass

        for evaluator in evaluators:
            try:
                evaluator.close()
            except Exception:
                pass
