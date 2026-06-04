import json
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import jwt  # noqa: F401
except ModuleNotFoundError:
    _tokens = {}
    jwt_stub = types.ModuleType("jwt")

    class ExpiredSignatureError(Exception):
        pass

    class InvalidTokenError(Exception):
        pass

    def encode(payload, secret, algorithm):
        token = f"stub-token-{len(_tokens)}"
        _tokens[token] = payload
        return token

    def decode(token, secret, algorithms):
        if token not in _tokens:
            raise InvalidTokenError("unknown token")
        return _tokens[token]

    jwt_stub.ExpiredSignatureError = ExpiredSignatureError
    jwt_stub.InvalidTokenError = InvalidTokenError
    jwt_stub.encode = encode
    jwt_stub.decode = decode
    sys.modules["jwt"] = jwt_stub

from api.gateway import APIGateway, GatewayConfig  # noqa: E402


class FakeBackendResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200
        self.text = str(payload)

    def json(self):
        return self.payload


class FakeBackendClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    async def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        return FakeBackendResponse(self.payload)


def _config(manifest_path: Path) -> GatewayConfig:
    config = GatewayConfig()
    config.JWT_SECRET = "test-secret"
    config.AUTH_USERS = {"testuser": "testpass"}
    config.DEV_AUTH_ENABLED = False
    config.RATE_LIMIT_ENABLED = False
    config.DEFAULT_COMPLETION_MODEL = "nano-llm-local"
    config.MODEL_ROUTING_CONFIG = str(manifest_path)
    config.ROUTING_CANARY_SALT = "test"
    return config


def _auth_headers(client: TestClient) -> dict:
    token = client.post(
        "/auth/token",
        json={"username": "testuser", "password": "testpass"},
    ).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_gateway_routes_default_model_to_nano_backend(tmp_path):
    manifest = {
        "default_model": "nano-llm-local",
        "allow_passthrough": False,
        "routes": {
            "nano-llm-local": {
                "targets": [
                    {
                        "name": "stable",
                        "backend_url": "http://nano-backend:8000",
                        "model": "/workspace/models/nano/checkpoint-final",
                        "weight": 100,
                    }
                ]
            }
        },
    }
    manifest_path = tmp_path / "routing.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gateway = APIGateway(config=_config(manifest_path))
    gateway.http_client = FakeBackendClient(
        {
            "id": "nano-cmpl-test",
            "object": "text_completion",
            "created": 123,
            "model": "/workspace/models/nano/checkpoint-final",
            "choices": [
                {
                    "text": "hello from nano",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )
    client = TestClient(gateway.app)

    response = client.post(
        "/v1/completions",
        headers=_auth_headers(client),
        json={"prompt": "say hello", "max_tokens": 8},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "hello from nano"
    assert gateway.http_client.requests[0]["url"] == (
        "http://nano-backend:8000/v1/completions"
    )
    assert gateway.http_client.requests[0]["json"]["model"] == (
        "/workspace/models/nano/checkpoint-final"
    )
