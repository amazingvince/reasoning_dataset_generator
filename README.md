# Chess LLM Pipeline (Dataset → SFT → GRPO → UI)

This repo contains an end-to-end chess LLM pipeline:

1. `dataset_creator/` — stream positions from Lichess, annotate with Stockfish, emit JSONL traces
2. `sft/` — supervised fine-tuning on the trace dataset (Qwen3 `<think>` + `<uci_move>` format)
3. `grpo/` — GRPO / RL fine-tuning with Stockfish-based rewards (Unsloth + vLLM)
4. `UI/` — a Modal-deployed play UI (vLLM + GPU memory snapshots)

Each stage has its own README with details:

- Dataset creation: `dataset_creator/README.md`
- SFT: `sft/README.md`
- GRPO: `grpo/README.md`
- UI: `UI/README.md`

## Quick navigation

### Generate a dataset (JSONL)

```bash
cd dataset_creator
python -m pip install -r requirements.txt
python make_dataset.py --config configs/config_stockfish_traces_full.yaml --output-jsonl ./data/chess_traces.jsonl
```

### Run SFT

```bash
cd sft
python -m pip install -r requirements.txt
python train.py --config config.yaml
```

### Run GRPO (Unsloth)

```bash
cd grpo
python -m pip install -r requirements.txt
python train_unsloth.py --config config_h100.yaml --model YOUR_SFT_MODEL
```

### Deploy UI (Modal)

```bash
cd UI
python -m pip install -r requirements.txt
modal setup
modal deploy chess_app.py
```

