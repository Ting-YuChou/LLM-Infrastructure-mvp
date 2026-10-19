import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api.model_routing import ModelRouter  # noqa: E402
from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402


ROOT = Path(__file__).parent.parent.parent


def test_nano_routing_manifest_targets_nano_backend():
    manifest_path = ROOT / "config" / "model_routing.nano.json"
    router = ModelRouter.from_file(str(manifest_path))

    route = router.resolve(None, user_id="tester", endpoint="/v1/completions")

    assert route.logical_model == "nano-llm-local"
    assert route.target.backend_url == "http://nano-vllm-server:8000"
    assert route.target.model == "/workspace/models/nano/checkpoint-final"


def test_nano_config_loads_yaml_defaults_and_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "nano.yaml"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "name": "/models/original",
                    "served_model_name": "nano-original",
                    "dtype": "float32",
                },
                "engine": {"device": "cpu", "use_triton": False},
                "server": {"port": 9000},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_NAME", "/models/env")
    monkeypatch.setenv("USE_TRITON", "true")

    config = NanoVLLMConfig.from_yaml(str(config_path))

    assert config.model_name == "/models/env"
    assert config.served_model_name == "nano-original"
    assert config.dtype == "float32"
    assert config.device == "cpu"
    assert config.use_triton is True
    assert config.port == 9000
