#!/usr/bin/env python3
"""
Convert a JSONL dataset into a HuggingFace Dataset.save_to_disk() directory.

Example:
  python scripts/jsonl_to_hf.py --input ./data/chess.jsonl --output ./data/chess_hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSONL -> HuggingFace dataset (save_to_disk).")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for Dataset.save_to_disk()")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input JSONL not found: {args.input}")

    ds = load_dataset("json", data_files=str(args.input), split="train")
    args.output.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.output))
    print(f"Saved dataset with {len(ds):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

