# Chess LLM Game 🎯♟️

Play chess against a fine-tuned chess reasoning LLM (`amazingvince/chess_qwen3_4b_reasoning_v2`) deployed on Modal with **GPU memory snapshots** for blazing-fast cold starts.

## ✨ Features

- **⚡ GPU Memory Snapshots** - ~10x faster cold starts (45s → ~5s)
- **🔄 Streaming Reasoning** - Watch the LLM think in real-time
- **⚪⚫ Play as White or Black** - Choose your side
- **📖 Opening Book** - 12 chess openings to choose from
- **🎨 Modern UI** - Dark theme with drag-and-drop board
- **💰 Serverless** - Scales to zero when idle

## 🚀 Quick Start

### Prerequisites

```bash
pip install modal
modal setup  # Authenticate with Modal
```

### Deploy

```bash
# Optional overrides
export CHESS_MODEL_NAME="amazingvince/chess_qwen3_4b_reasoning_v2"
export CHESS_GPU="T4"
export CHESS_SNAPSHOT_VERSION="v1"

# Deploy to Modal (creates permanent URL)
modal deploy chess_app.py
```

After deployment, you'll see a URL like:
```
https://your-workspace--chess-llm-game-web.modal.run
```

Visit that URL to play chess!

### Development Mode

```bash
# Run locally with hot reload
modal serve chess_app.py
```

## 🏎️ GPU Memory Snapshots

This app uses Modal's GPU memory snapshots to dramatically reduce cold start times.

### How It Works

1. **First Cold Start (~45-60s)**: vLLM loads the model, warms up, then sleeps
2. **Snapshot Creation**: Modal captures GPU + CPU memory state
3. **Subsequent Starts (~5s)**: Restore from snapshot, skip model loading

### Key Configuration

```python
@app.cls(
    gpu="T4",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class ChessLLMServer:
    @modal.enter(snap=True)  # Runs before snapshot
    def start_and_snapshot(self):
        # Start vLLM, warmup, then sleep
        ...
    
    @modal.enter(snap=False)  # Runs after restore
    def wake_after_restore(self):
        # Wake up the server
        ...
```

### vLLM Sleep Mode

The app uses vLLM's sleep mode to prepare for snapshotting:

```bash
vllm serve ... --enable-sleep-mode
```

When sleeping:
- Model weights offloaded to CPU memory
- KV cache emptied
- Ready for memory snapshot

### Invalidating Snapshots

To force new snapshots (e.g., after model update), change:

```python
SNAPSHOT_VERSION = "v2"  # Increment this
```

## 📖 Opening Book

| Opening | Description |
|---------|-------------|
| Starting Position | Standard start |
| Italian Game | 1.e4 e5 2.Nf3 Nc6 3.Bc4 |
| Sicilian Defense | 1.e4 c5 |
| Sicilian Najdorf | Bobby Fischer's favorite |
| Queen's Gambit | 1.d4 d5 2.c4 |
| King's Indian | Dynamic counterattack |
| French Defense | 1.e4 e6 2.d4 d5 3.e5 |
| Caro-Kann | 1.e4 c6 2.d4 d5 3.e5 |
| Ruy Lopez | The Spanish Game |
| London System | Solid for White |
| Scandinavian | 1.e4 d5 2.exd5 |
| English Opening | 1.c4 |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Modal Cloud                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌──────────────────────────────┐  │
│  │   Web Server    │────▶│    ChessLLMServer (GPU)      │  │
│  │   (FastAPI)     │     │                              │  │
│  │                 │     │  • vLLM subprocess           │  │
│  │  • Game state   │◀────│  • GPU memory snapshots      │  │
│  │  • REST API     │     │  • Streaming generation      │  │
│  │  • HTML UI      │     │                              │  │
│  └─────────────────┘     └──────────────────────────────┘  │
│                                     │                       │
│                          ┌──────────┴──────────┐           │
│                          │   Modal Volumes      │           │
│                          │  • HuggingFace cache │           │
│                          │  • vLLM cache        │           │
│                          └─────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ Configuration

### GPU Selection

```python
# T4 - Good for 4B models, cost-effective (~$0.59/hr)
gpu="T4"

# A10G - Faster inference (~$1.10/hr)
gpu="A10G"

# L4 - Good balance (~$0.80/hr)
gpu="L4"
```

### Sampling Parameters (Qwen3 Defaults)

```python
temperature=0.7
top_p=0.8
top_k=20
repetition_penalty=1.05
```

### Scaledown Window

```python
scaledown_window=5 * MINUTES  # Stay warm for 5 mins after last request
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the chess UI |
| `/api/new-game` | POST | Start a new game |
| `/api/move` | POST | Make a player move |
| `/api/llm-move/{id}` | GET | Stream LLM's move (SSE) |
| `/api/game/{id}` | GET | Get game state |
| `/api/openings` | GET | List available openings |

## 💰 Cost Estimation

- **T4 GPU**: ~$0.59/hour
- **Idle**: $0 (scales to zero)
- **Typical game**: ~5-10 minutes = ~$0.05-0.10

With GPU snapshots, cold starts are fast enough that you can scale to zero aggressively without sacrificing user experience.

## 🐛 Troubleshooting

### Slow First Request

The first few requests after deployment create GPU snapshots. This is expected. Subsequent cold starts will be ~10x faster.

### "Model not found" Error

Ensure the model is accessible:
```bash
# Test model access
python -c "from huggingface_hub import snapshot_download; snapshot_download('amazingvince/chess_qwen3_4b_reasoning_v2')"
```

### Snapshot Not Working

1. Make sure you're using `modal deploy` (not `modal run`)
2. Check that `enable_memory_snapshot=True` is set
3. Try changing `SNAPSHOT_VERSION` to force new snapshot

### vLLM Startup Timeout

If the model takes too long to load:
```python
timeout=15 * MINUTES  # Increase timeout
```

## 📚 References

- [Modal GPU Snapshots Documentation](https://modal.com/docs/examples/gpu_snapshot)
- [Modal vLLM Example](https://modal.com/docs/examples/vllm_inference)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/stable/features/sleep_mode/)
- [Ministral 3 Snapshot Example](https://modal.com/docs/examples/ministral3_inference)

## 📄 License

MIT
