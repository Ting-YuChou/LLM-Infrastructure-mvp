"""scripts/upload_checkpoint.py: private-only uploads of a final checkpoint."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import upload_checkpoint  # noqa: E402


class FakeApi:
    def __init__(self, private=True):
        self.private = private
        self.calls = []

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo, private, exist_ok))

    def model_info(self, repo):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return SimpleNamespace(commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/abc")


def _checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("weights")
    return tmp_path


def test_upload_creates_a_private_repo_and_leaves_training_state_out(tmp_path):
    api = FakeApi()

    url = upload_checkpoint.upload_checkpoint("me/llama3-sft", _checkpoint(tmp_path), "run sft", "token", api=api)

    assert url == "https://huggingface.co/me/llama3-sft/commit/abc"
    assert api.calls[0] == ("create_repo", "me/llama3-sft", True, True)
    upload = api.calls[1][1]
    assert upload["folder_path"] == str(tmp_path) and upload["commit_message"] == "run sft"
    assert "optimizer*" in upload["ignore_patterns"]


def test_upload_refuses_an_existing_public_repo(tmp_path):
    api = FakeApi(private=False)

    with pytest.raises(PermissionError, match="public"):
        upload_checkpoint.upload_checkpoint("me/llama3-sft", _checkpoint(tmp_path), "m", "token", api=api)
    assert not any(call[0] == "upload_folder" for call in api.calls)


def test_upload_needs_a_model_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        upload_checkpoint.upload_checkpoint("me/x", tmp_path, "m", "token", api=FakeApi())
