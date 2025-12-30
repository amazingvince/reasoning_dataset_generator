# Chess GRPO Training - Unsloth + Single H100

Train your chess LLM using **GRPO (Group Relative Policy Optimization)** with **DAPO loss** and **Stockfish rewards** on a **single H100 80GB GPU**.

This implementation uses **2025 best practices** from recent research including DAPO, Open-Reasoner-Zero, and Understanding R1-Zero-Like Training.

## 2025 Best Practices

This implementation incorporates findings from recent RL research:

### DAPO Loss (Decoupled Clip and Dynamic Sampling)

We use **DAPO loss** instead of standard GRPO, based on [ByteDance's research](https://arxiv.org/abs/2503.14476) showing **50% on AIME 2024** vs 30% baseline.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `loss_type` | `"dapo"` | Asymmetric clipping prevents entropy collapse |
| `epsilon` | `0.2` | Lower bound for suppressing bad actions |
| `epsilon_high` | `0.28` | Higher bound encourages exploration of good actions |
| `beta` | `0.0` | No KL penalty - unnecessary with verifiable rewards |
| `scale_rewards` | `"batch"` | Batch-level normalization avoids difficulty bias |

### Why These Settings?

**No KL Penalty (beta=0.0)**: Multiple 2025 papers (Open-Reasoner-Zero, DAPO, Understanding R1-Zero-Like Training) show that with verifiable rewards like Stockfish, KL penalty suppresses exploration without benefit. We have ground-truth verification, not a reward model that could be over-optimized.

**Asymmetric Clipping**: Standard PPO/GRPO uses symmetric clipping (epsilon=0.2 both ways). DAPO's "Clip-Higher" strategy uses epsilon_high=0.28 for the upper bound, allowing the model to more aggressively reinforce good moves while conservatively suppressing bad ones.

**Batch-Level Reward Scaling**: Per-group normalization can cause difficulty bias (easy puzzles get same weight as hard ones). Batch-level scaling is more robust.

**Mask Truncated Completions**: Outputs that hit the length limit shouldn't contribute to the loss - they may be mid-thought.

## Why Unsloth?

Based on my research, Unsloth provides significant advantages for GRPO training:

| Feature | Standard TRL + FA2 | Unsloth |
|---------|-------------------|---------|
| VRAM for Llama 8B @ 20K context | 510 GB | **54 GB (90% less)** |
| Training speed | 1x | **2x faster** |
| vLLM integration | Manual | **Built-in** |
| Memory-efficient GRPO | No | **Yes (Standby mode)** |
| FP8 support on H100 | Limited | **Full support (1.4x faster)** |

## Quick Start

```bash
# Install
pip install -r requirements.txt
sudo apt-get install stockfish

# Train
python train_unsloth.py --config config_h100.yaml --model your-chess-model-4b
```

## Key Optimizations for H100

### 1. Unsloth Standby Mode
Automatically enabled via `UNSLOTH_VLLM_STANDBY=1`. This allows:
- Setting `gpu_memory_utilization` to 0.85-0.95
- Dynamic memory management between training and vLLM inference
- No manual tuning needed

### 2. FP8 Training (Optional)
H100 supports FP8 for 1.4x faster training:
```bash
python train_unsloth.py --model your-model --use_fp8
```

### 3. FP8 KV Cache
Enabled by default - halves KV cache memory usage on H100.

### 4. vLLM Fast Inference
Built into Unsloth with `fast_inference=True`:
- ~2000-4000 tokens/sec generation
- Shared memory space with training (saves ~16GB)

### 5. Smart Gradient Checkpointing
Unsloth's `use_gradient_checkpointing="unsloth"`:
- 30% less VRAM than standard checkpointing
- Async offloading to CPU RAM
- Only 1% slower

## Memory Budget (4B Model)

| Component | Memory |
|-----------|--------|
| Model (16-bit LoRA) | ~8 GB |
| LoRA adapters | ~0.5 GB |
| Optimizer (paged_adamw_8bit) | ~2 GB |
| Gradients | ~8 GB |
| vLLM KV cache (FP8, 2K ctx) | ~4 GB |
| GRPO logits (8 gens, Unsloth optimized) | ~10 GB |
| Buffer | ~10 GB |
| **Total** | **~45-50 GB** |

This leaves ~30GB headroom on an 80GB H100.

## Configuration

`train_unsloth.py` loads `config_h100.yaml` by default; CLI flags override the file (e.g. `--model`, `--stockfish-depth`, `--num-generations`, `--output-dir`).

Edit `config_h100.yaml`:

```yaml
model:
  name_or_path: "your-chess-model-4b"

unsloth:
  load_in_4bit: false  # 16-bit for H100
  use_fp8: false       # Set true for 1.4x speed
  gpu_memory_utilization: 0.85

training:
  # DAPO loss (2025 best practice)
  loss_type: "dapo"
  epsilon: 0.2         # Lower bound (suppression)
  epsilon_high: 0.28   # Upper bound (encouragement)
  beta: 0.0            # No KL penalty
  scale_rewards: "batch"
  mask_truncated_completions: true

  # Generation
  num_generations: 8
  max_completion_length: 2048
  temperature: 0.7
  learning_rate: 5.0e-6
```

## Reward Function

Two reward strategies are available, controlled by `reward_type` in config:

### Win Probability Delta (WPD) - Recommended

Set `reward_type: "wpd"` (default). WPD measures how much win probability a move loses compared to the best move:

| Move Quality | WPD Range | Reward |
|-------------|-----------|--------|
| Excellent   | < 0.02    | +1.50  |
| Good        | < 0.05    | +1.08  |
| Acceptable  | < 0.10    | +0.68  |
| Inaccuracy  | < 0.20    | +0.40  |
| Mistake     | < 0.35    | +0.00  |
| Blunder     | ≥ 0.35    | -0.60  |

**Why WPD over Top-N?**
- **Quiet positions**: Moves 1-5 may all be within 5cp → WPD rewards all highly
- **Tactical positions**: "Top 3" move might lose 200cp → WPD correctly penalizes as blunder
- Top-N would reward both scenarios the same, despite vastly different quality

**Key WPD config options:**
```yaml
reward:
  reward_type: "wpd"
  wpd_penalty_scale: 4.0  # Higher = stricter (reward = 1.0 - wpd * scale)
  excellent_threshold: 0.02
  excellent_bonus: 0.5
```

### Legacy Top-N Ranking

Set `reward_type: "topn"` for ranking-based rewards:

```python
Top 1 match:  +1.5   # Matches Stockfish best move
Top 3 match:  +1.0   # Strong alternative
Top 5 match:  +0.6   # Reasonable move
Legal move:   +0.1   # Legal but not in top 5
Illegal:      -0.5
No move:      -1.0
```

### Format Rewards (both modes)

```python
Correct format: +0.1   # <think>...</think><uci_move>...</uci_move>
Wrong format:   -0.1
```

## Training Tips

1. **Wait for 300+ steps** before expecting reward improvement
2. **12+ hours** training recommended for good results
3. **Monitor rewards** - they should trend upward
4. **Log every step** (`logging_steps: 1`) to track progress
5. **DAPO helps exploration** - asymmetric clipping prevents the model from collapsing to a single response pattern

## Troubleshooting

### Out of Memory
```python
# Option 1: Reduce generations
num_generations: 6  # or 4

# Option 2: Enable 4-bit
load_in_4bit: true

# Option 3: Reduce completion length
max_completion_length: 1024

# Option 4: Lower GPU utilization
gpu_memory_utilization: 0.7
```

### Slow Training
```python
# Enable FP8 (H100 only)
use_fp8: true

# Reduce Stockfish depth for faster rewards
stockfish_depth: 15
```

### vLLM Issues
```bash
# Make sure vLLM is up to date
pip install --upgrade vllm

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

## Files

```
grpo/
├── train_unsloth.py     # Main training script
├── train_simple.py      # Minimal variant (puzzles-only)
├── config_h100.yaml     # Configuration
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Expected Results

After training:
- **Legal move rate**: 95%+

### WPD Metrics (when using `reward_type: wpd`)
- **Avg WPD**: < 0.05 (lower is better)
- **Excellent rate**: > 30% (moves with wpd < 0.02)
- **Blunder rate**: < 5% (moves with wpd >= 0.35)

### Top-N Metrics (when using `reward_type: topn`)
- **Top-1 accuracy**: 15-25% (matching Stockfish exactly)
- **Top-5 accuracy**: 50-70%
- **Avg centipawn loss**: < 50

## References

### Implementation
- [Unsloth GRPO Documentation](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
- [Unsloth Memory-Efficient RL](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)

### 2025 Research (Configuration Choices)
- [DAPO: Decoupled Clip and Dynamic Sampling](https://arxiv.org/abs/2503.14476) - Asymmetric clipping, 50% AIME 2024
- [Open-Reasoner-Zero](https://huggingface.co/papers/2503.24290) - Beta=0 for verifiable rewards
- [Understanding R1-Zero-Like Training](https://huggingface.co/papers/2503.20783) - Batch-level reward scaling

### Original Papers
- [DeepSeekMath GRPO Paper](https://arxiv.org/abs/2402.03300) - Original GRPO algorithm
