"""scripts/vllm_requirements.py: vllm's dependencies minus the numpy-2-only opencv."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("packaging")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from vllm_requirements import runtime_requirements  # noqa: E402


def test_opencv_optional_extras_and_other_platforms_are_dropped():
    specs = [
        "numpy",
        "opencv-python-headless>=4.13.0",
        "mistral_common[image]>=1.10.0",
        "torch==2.10.0",
        "pandas; extra == 'bench'",
        "never-installed; sys_platform == 'nonexistent'",
    ]
    assert runtime_requirements(specs) == ["numpy", "mistral_common>=1.10.0", "torch==2.10.0"]
