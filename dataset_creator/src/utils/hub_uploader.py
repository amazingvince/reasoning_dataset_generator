"""
HuggingFace Hub helpers for incremental dataset uploads.

This repo generates JSONL rows continuously; to make the dataset usable on the
Hub during generation, we upload sealed JSONL shard files (e.g.
`data/train-000000.jsonl`) to a dataset repository.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolve a HuggingFace token from an explicit value or common env vars."""

    return explicit_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


@dataclass(frozen=True)
class HubDatasetUploader:
    """Small wrapper around `huggingface_hub.HfApi` for dataset uploads."""

    repo_id: str
    token: str
    private: bool = False

    def _api(self) -> HfApi:
        return HfApi(token=self.token)

    def ensure_repo(self) -> None:
        api = self._api()
        api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=bool(self.private), exist_ok=True)

    def ensure_default_files(
        self,
        *,
        dataset_card_markdown: str,
        overwrite_readme: bool = False,
        overwrite_gitattributes: bool = False,
    ) -> None:
        """
        Upload a dataset card and basic `.gitattributes` (for large JSONL shards).

        This is best-effort; failures are non-fatal for the actual shard uploads.
        """
        api = self._api()

        existing: set[str] = set()
        try:
            existing = set(api.list_repo_files(repo_id=self.repo_id, repo_type="dataset"))
        except Exception:
            existing = set()

        readme_bytes = io.BytesIO(dataset_card_markdown.encode("utf-8"))
        if overwrite_readme or "README.md" not in existing:
            try:
                api.upload_file(
                    path_or_fileobj=readme_bytes,
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message="Add dataset card",
                )
            except Exception:
                pass

        gitattributes = "\n".join(
            [
                "*.jsonl filter=lfs diff=lfs merge=lfs -text",
                "*.parquet filter=lfs diff=lfs merge=lfs -text",
                "*.arrow filter=lfs diff=lfs merge=lfs -text",
                "",
            ]
        )
        git_bytes = io.BytesIO(gitattributes.encode("utf-8"))
        if overwrite_gitattributes or ".gitattributes" not in existing:
            try:
                api.upload_file(
                    path_or_fileobj=git_bytes,
                    path_in_repo=".gitattributes",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message="Add gitattributes",
                )
            except Exception:
                pass

    def upload_shard(
        self,
        *,
        local_path: Path,
        path_in_repo: str,
        commit_message: str,
    ) -> None:
        api = self._api()
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=str(path_in_repo),
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=str(commit_message),
        )
