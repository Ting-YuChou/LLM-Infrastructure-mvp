import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.check_serving_runtime_matrix import validate_runtime_matrix  # noqa: E402


def test_current_serving_runtime_matrix_is_aligned():
    errors = validate_runtime_matrix(
        Path("requirements.txt"),
        Path("docker/requirements.serving.txt"),
        Path("docker/Dockerfile.serving"),
    )

    assert errors == []


def test_serving_runtime_matrix_detects_version_drift(tmp_path):
    root_requirements = tmp_path / "requirements.txt"
    serving_requirements = tmp_path / "requirements.serving.txt"
    dockerfile = tmp_path / "Dockerfile.serving"

    root_requirements.write_text(
        "\n".join(
            [
                "torch==2.10.0",
                "torchvision==0.25.0",
                "torchaudio==2.10.0",
                "transformers==4.57.6",
                "tokenizers==0.22.2",
                "accelerate==1.13.0",
                "sentencepiece==0.2.1",
                "protobuf==5.29.6",
                "fastapi==0.136.1",
                "uvicorn[standard]==0.46.0",
                "pydantic==2.13.3",
                "vllm==0.19.1",
            ]
        ),
        encoding="utf-8",
    )
    serving_requirements.write_text(
        root_requirements.read_text(encoding="utf-8").replace("vllm==0.19.1", "vllm==0.2.7"),
        encoding="utf-8",
    )
    dockerfile.write_text(
        "FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04\n"
        "RUN apt-get update && apt-get install -y python3.10\n"
        "RUN pip install -r /tmp/requirements.serving.txt\n",
        encoding="utf-8",
    )

    errors = validate_runtime_matrix(root_requirements, serving_requirements, dockerfile)

    assert any("vllm mismatch" in error for error in errors)
    assert any("CUDA 12.9.1" in error for error in errors)
