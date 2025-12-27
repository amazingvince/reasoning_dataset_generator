# Chess Trace Dataset Generator (Streaming JSONL)

Generate a streaming JSONL dataset of chess positions annotated with Stockfish + a natural-language reasoning trace.

Each JSONL row contains:
- `fen`: FEN string
- `valid_moves`: list of legal UCI moves
- `reasoning_trace`: trace text (no token cutoff in JSONL mode)
- `chosen_move`: Stockfish best move (UCI)

## Requirements

- Python 3.10+
- A Stockfish binary available locally
- Network access (to stream Lichess datasets, and optionally download openings/tablebases)

Install Stockfish:
- Linux: `sudo apt install stockfish`
- macOS: `brew install stockfish`
- Windows: download a Stockfish release and pass `--stockfish-path`

## Install

```bash
python -m pip install -r requirements.txt
```

## Quickstart

Generate JSONL:

```bash
python make_dataset.py \
  --config configs/config_stockfish_traces_full.yaml \
  --output-jsonl ./data/chess_traces.jsonl
```

Useful overrides:
- `--size`: total number of positions (games + puzzles)
- `--depth`: Stockfish depth (higher = slower/stronger)
- `--workers`: parallel Stockfish processes (scale this with CPU cores)
- `--multipv`: MultiPV cap (default: `reasoning_trace.candidate_pool_size`, use `0` for all moves)
- `--stockfish-path`: explicit Stockfish binary
- `--jsonl-max-in-flight`: max queued positions (higher hides input/network jitter)
- `--jsonl-flush-every`: flush frequency (higher = faster)

Example:

```bash
python make_dataset.py \
  --config configs/config_stockfish_traces_full.yaml \
  --output-jsonl ./data/chess_traces_small.jsonl \
  --size 2000 \
  --depth 10 \
  --workers 8 \
  --multipv 12
```

## Stream uploads to HuggingFace Hub (every 15 minutes)

If you want the dataset to be usable on the Hub while it is still being generated, upload sealed JSONL shards to a Hub *dataset* repo:

```bash
export HF_TOKEN=...  # or HUGGINGFACE_HUB_TOKEN

python make_dataset.py \
  --config configs/config_stockfish_traces_full.yaml \
  --output-jsonl ./data/chess_traces.jsonl \
  --hub-repo-id USER/REPO \
  --hub-upload-every-minutes 15
```

Then load directly (no conversion step required):

```python
from datasets import load_dataset
ds = load_dataset("USER/REPO", split="train", streaming=True)
print(next(iter(ds)))
```

## Scaling notes (200+ cores)

This generator is designed to run “one Stockfish process per worker”.

- Prefer `stockfish.threads_per_worker: 1` when scaling `--workers` high.
- Set `--multipv` to something like `12`/`16` (evaluating *all* legal moves is very slow).
- If you want maximum throughput, set `stockfish.shallow_depth: 0` / `stockfish.confirm_depth: 0` (skips extra trap-detection passes).
- Increase `--jsonl-max-in-flight` if workers go idle due to input/network stalls.
- Increase `--jsonl-flush-every` to reduce IO overhead on large machines.

## Optional: openings + tablebase enrichments

These are only used to enrich the reasoning traces. If you skip them, generation still works.

```bash
python scripts/download_openings.py --output-dir ./data/openings
python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5
```

Trace variety knobs live under `reasoning_trace` in `configs/config_stockfish_traces_full.yaml` (e.g. `include_fen_walkthrough`, `include_context_hints`, `include_decision_profile`, `pv_prob`, `threat_scan_prob`, `candidate_pv_max_per_trace`, `candidate_motif_prob`).

Recent enrichments include `include_opening_plan_hints`, `include_endgame_technique`, `include_avoid_blunders` (with short “why it fails” explanations when PV is available), `include_pv_key_moments`, and `include_pv_explainer` (PV-to-English narration). A coherent Q/A trace format is available via the `dialogue` style. Trap detection can use extra Stockfish passes via `stockfish.shallow_depth` / `stockfish.confirm_depth` (slower, but richer traces).

## Convert JSONL to a HuggingFace dataset

```bash
python scripts/jsonl_to_hf.py --input ./data/chess_traces.jsonl --output ./data/chess_traces_hf
```

Then:

```python
from datasets import load_from_disk
ds = load_from_disk("./data/chess_traces_hf")
print(ds[0])
```
