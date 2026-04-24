import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from api.model_routing import ModelRouter, UnknownModelRoute


def test_static_router_preserves_legacy_passthrough():
    router = ModelRouter.from_static(
        backend_url="http://backend:8000",
        default_model="default-model",
    )

    route = router.resolve(
        requested_model="explicit-model",
        user_id="user",
        endpoint="/v1/completions",
    )

    assert route.target.backend_url == "http://backend:8000"
    assert route.target.model == "explicit-model"
    assert route.target.stage == "Passthrough"


def test_manifest_router_routes_to_weighted_canary(tmp_path):
    manifest = {
        "default_model": "logical",
        "routes": {
            "logical": {
                "targets": [
                    {
                        "name": "stable",
                        "backend_url": "http://stable:8000",
                        "model": "stable-model",
                        "weight": 0,
                    },
                    {
                        "name": "canary",
                        "backend_url": "http://canary:8000",
                        "model": "canary-model",
                        "weight": 100,
                        "stage": "Staging",
                        "version": "2",
                    },
                ]
            }
        },
    }
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    router = ModelRouter.from_file(str(path))
    route = router.resolve(None, user_id="user", endpoint="/v1/completions")

    assert route.logical_model == "logical"
    assert route.target.name == "canary"
    assert route.target.backend_url == "http://canary:8000"
    assert route.target.model == "canary-model"


def test_manifest_router_rejects_unknown_model():
    router = ModelRouter.from_dict(
        {
            "allow_passthrough": False,
            "routes": {
                "known": {
                    "targets": [
                        {
                            "name": "stable",
                            "backend_url": "http://backend:8000",
                            "model": "known",
                            "weight": 100,
                        }
                    ]
                }
            },
        }
    )

    with pytest.raises(UnknownModelRoute):
        router.resolve("unknown", user_id="user", endpoint="/v1/completions")
