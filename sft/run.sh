#!/bin/bash
# Chess Reasoning SFT Training Runner
# Usage: ./run.sh [config_file]

set -e

CONFIG="${1:-config.yaml}"

echo "=================================================="
echo "♟️  Chess Reasoning SFT Training"
echo "=================================================="

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set - Hub push will fail"
    echo "   Set it with: export HF_TOKEN=your_token"
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not set - W&B logging disabled"
    echo "   Set it with: export WANDB_API_KEY=your_key"
fi

# Install requirements if needed
if ! python -c "import trl" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Run training
echo ""
echo "🚀 Starting training with config: $CONFIG"
echo ""

python train.py --config "$CONFIG"
