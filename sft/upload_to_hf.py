#!/usr/bin/env python3
"""
Upload a model to Hugging Face Hub.

Usage:
    python upload_to_hf.py --model-path ./my_model --repo-name username/model-name
    python upload_to_hf.py --model-path ./my_model --repo-name username/model-name --private
    python upload_to_hf.py --model-path ./chess_qwen3_4b_reasoning --repo-name amazingvince/chess_qwen3_4b_reasoning_v2 --private
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login, create_repo


def upload_model(
    model_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload model",
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
):
    """
    Upload a model directory to Hugging Face Hub.

    Args:
        model_path: Local path to the model directory
        repo_name: Repository name (format: username/repo-name)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        token: Hugging Face API token (uses cached token if not provided)
        ignore_patterns: File patterns to ignore during upload
    """
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Login if token provided, otherwise use cached credentials
    if token:
        login(token=token)

    api = HfApi()

    # Create the repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"✓ Repository '{repo_name}' ready")
    except Exception as e:
        print(f"Note: {e}")

    # Default ignore patterns
    if ignore_patterns is None:
        ignore_patterns = [
            "*.pyc",
            "__pycache__",
            ".git",
            ".gitignore",
            "*.log",
            ".DS_Store",
            "runs/*",
            "wandb/*",
        ]

    print(f"Uploading from: {model_path}")
    print(f"Uploading to: https://huggingface.co/{repo_name}")

    # Upload the entire folder
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_name,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )

    print(f"✓ Upload complete!")
    print(f"  View at: https://huggingface.co/{repo_name}")


def upload_single_file(
    file_path: str,
    repo_name: str,
    path_in_repo: str | None = None,
    private: bool = False,
    commit_message: str = "Upload file",
    token: str | None = None,
):
    """
    Upload a single file to Hugging Face Hub.

    Args:
        file_path: Local path to the file
        repo_name: Repository name (format: username/repo-name)
        path_in_repo: Path where the file will be stored in the repo
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        token: Hugging Face API token
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if token:
        login(token=token)

    api = HfApi()

    # Create the repository if it doesn't exist
    create_repo(
        repo_id=repo_name,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    # Use the filename if path_in_repo not specified
    if path_in_repo is None:
        path_in_repo = file_path.name

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=path_in_repo,
        repo_id=repo_name,
        repo_type="model",
        commit_message=commit_message,
    )

    print(f"✓ Uploaded {file_path.name} to {repo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a model directory
  python upload_to_hf.py --model-path ./my_model --repo-name myuser/my-model

  # Upload as private repo
  python upload_to_hf.py --model-path ./my_model --repo-name myuser/my-model --private

  # Upload a single file
  python upload_to_hf.py --file ./model.safetensors --repo-name myuser/my-model

  # Use a specific token
  python upload_to_hf.py --model-path ./my_model --repo-name myuser/my-model --token hf_xxx
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory to upload",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single file to upload (alternative to --model-path)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository name in format 'username/repo-name'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=None,
        help="Path in the repo for single file upload",
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")

    if args.file:
        upload_single_file(
            file_path=args.file,
            repo_name=args.repo_name,
            path_in_repo=args.path_in_repo,
            private=args.private,
            commit_message=args.commit_message,
            token=token,
        )
    elif args.model_path:
        upload_model(
            model_path=args.model_path,
            repo_name=args.repo_name,
            private=args.private,
            commit_message=args.commit_message,
            token=token,
        )
    else:
        parser.error("Either --model-path or --file must be provided")


if __name__ == "__main__":
    main()
