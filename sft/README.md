# Chess Reasoning SFT Training

Fine-tune Qwen3-4B-Thinking on chess reasoning traces with Stockfish evaluation.

## Features

- **Custom special tokens**: `<uci_move>` / `</uci_move>` for structured move output
- **Efficient generation**: Stops on EOS or `</uci_move>` (native `eos_token_id` support)
- **Optimized training**: Liger kernels + padding-free collation
- **Streaming dataset**: `amazingvince/chess-traces` with eval holdout
- **Stockfish eval**: Legal move rate, exact match rate, centipawn loss
- **Logging**: W&B metrics + HuggingFace Hub checkpoints

## Quick Start

### 1. Set environment variables

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_api_key"
```

### 2. Run training

```bash
# Using default config
python train.py

# Or from the repo root:
python sft/train.py

# Or with custom config
python train.py --config my_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Key settings
model:
  name: "Qwen/Qwen3-4B-Thinking-2507"

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 32
  learning_rate: 2.0e-5
  num_train_epochs: 3

optimization:
  use_liger_kernel: true
  padding_free: true

hub:
  hub_model_id: "amazingvince/chess_qwen3_4b_reasoning"
  hub_private_repo: true
```

## Dataset Format

The training uses `amazingvince/chess-traces` with:

- **Input**: FEN position + legal moves
- **Output**: `<think>reasoning</think><uci_move>move</uci_move>`

Example formatted sample:

```
Prompt:
You are an expert chess player. Choose the best move.
FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
Legal moves (UCI): a6, b6, c6, d6, e6, f6, g6, h6, ...

Completion:
<think>White has played 1.e4. I should consider the most solid responses.
The Sicilian Defense (c5) is aggressive, the French (e6) is solid...</think>
<uci_move>e7e5</uci_move>
```

## Evaluation Metrics

During training, the following metrics are logged:

| Metric | Description |
|--------|-------------|
| `chess/legal_move_rate` | % of predictions that are legal moves |
| `chess/exact_match_rate` | % that match Stockfish's top choice |
| `chess/avg_centipawn_loss` | Average centipawn difference from best move |

## Hardware Requirements

Tested on **1x H100 80GB**:

- Full fine-tune of 4B model
- Batch size 4, gradient accumulation 8
- ~8GB VRAM with Liger kernels + padding-free

## Output

After training:

- Local checkpoint: `./chess_qwen3_4b_reasoning/`
- HuggingFace Hub: `https://huggingface.co/amazingvince/chess_qwen3_4b_reasoning`
- W&B dashboard: `https://wandb.ai/<your-entity>/chess-reasoning-sft`

## Files

```
├── train.py           # Main training script
├── config.yaml        # Training configuration
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Troubleshooting

### Flash Attention not available
```bash
pip install flash-attn --no-build-isolation
```

### Stockfish not found
```bash
sudo apt-get install stockfish
```

### Out of memory
Reduce `per_device_train_batch_size` or `max_length` in config.

## License

Apache 2.0 (following Qwen3 license)
