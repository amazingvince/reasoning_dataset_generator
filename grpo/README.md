# Chess GRPO Training - Unsloth + Single H100

Train your chess LLM using **GRPO (Group Relative Policy Optimization)** with **Stockfish rewards** on a **single H100 80GB GPU**.

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
  num_generations: 8
  max_completion_length: 2048
  learning_rate: 5.0e-6
```

## Reward Function

Stockfish-based reward with format bonuses:

```python
# Move quality (main reward)
Top 1 match:  +1.0
Top 3 match:  +0.7
Top 5 match:  +0.4
Legal move:   +0.1
Illegal:      -0.5
No move:      -1.0

# Format (stacked)
Correct format: +0.2
Wrong format:   -0.1
```

## Training Tips

1. **Wait for 300+ steps** before expecting reward improvement
2. **12+ hours** training recommended for good results
3. **Monitor rewards** - they should trend upward
4. **Log every step** (`logging_steps: 1`) to track progress

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
- **Top-1 accuracy**: 15-25% (matching Stockfish exactly)
- **Top-5 accuracy**: 50-70% 
- **Legal move rate**: 95%+
- **Avg centipawn loss**: <50

## References

- [Unsloth GRPO Documentation](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)
- [Unsloth Memory-Efficient RL](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [DeepSeekMath GRPO Paper](https://arxiv.org/abs/2402.03300)
