#!/usr/bin/env python3
"""
Upload a trained checkpoint directory to a private Hugging Face model repo.

    HF_WRITE_TOKEN=... python scripts/upload_checkpoint.py \
        --repo user/llama3-8b-sft --path outputs/sft/final --message "run abc123 sft"

The repo is created private when it does not exist. An existing repo must
already be private: the base model is gated (Llama 3 license), so a checkpoint
is never pushed to a public repo by accident. Optimizer and RNG state files are
left out; the uploaded folder is what from_pretrained and vLLM load.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Training-state files a final model directory should not carry to the Hub.
EXCLUDE = ("optimizer*", "scheduler*", "rng_state*", "trainer_state.json", "training_args.bin", "*.distcp")


def upload_checkpoint(repo: str, path: Path, message: str, token: str, api: Optional[Any] = None) -> str:
    """Create `repo` private if needed, check it is private, upload `path`; returns the commit URL."""
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"{path} is not a Hugging Face model directory (no config.json)")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo).private:
        raise PermissionError(f"{repo} exists and is public; refusing to upload a Llama 3 derivative to it")
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(path),
        commit_message=message,
        ignore_patterns=list(EXCLUDE),
    )
    return str(getattr(commit, "commit_url", commit))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="user/name of the private model repo")
    parser.add_argument("--path", required=True, help="Checkpoint directory (config.json, safetensors, tokenizer)")
    parser.add_argument("--message", default="Upload checkpoint", help="Commit message")
    args = parser.parse_args()

    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        print("HF_WRITE_TOKEN is not set (a Hugging Face token with write access)", file=sys.stderr)
        return 2
    print(upload_checkpoint(args.repo, Path(args.path), args.message, token))
    return 0


if __name__ == "__main__":
    sys.exit(main())
