import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import jwt  # noqa: F401
except ModuleNotFoundError:
    jwt_stub = types.ModuleType("jwt")
    jwt_stub.encode = lambda *args, **kwargs: "token"
    jwt_stub.decode = lambda *args, **kwargs: {}
    jwt_stub.ExpiredSignatureError = Exception
    jwt_stub.InvalidTokenError = Exception
    sys.modules["jwt"] = jwt_stub

from api.gateway import APIGateway, GatewayConfig, RateLimiter  # noqa: E402


def make_config(environment="production"):
    config = GatewayConfig()
    config.ENVIRONMENT = environment
    config.JWT_SECRET = "test-secret"
    config.AUTH_USERS = {"testuser": "testpass"}
    config.CORS_ORIGINS = ["https://llm.example.com"]
    config.ALLOWED_HOSTS = ["llm.example.com"]
    config.RATE_LIMIT_ENABLED = False
    config.DEFAULT_COMPLETION_MODEL = "test-model"
    config.MODEL_ROUTING_CONFIG = None
    return config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("JWT_SECRET", "change-me", "JWT_SECRET"),
        ("AUTH_USERS", {}, "AUTH_USERS"),
        ("CORS_ORIGINS", ["*"], "CORS_ORIGINS"),
        ("ALLOWED_HOSTS", ["*"], "ALLOWED_HOSTS"),
    ],
)
def test_gateway_production_config_fails_fast(field, value, message):
    config = make_config(environment="production")
    setattr(config, field, value)

    with pytest.raises(RuntimeError, match=message):
        APIGateway(config=config)


def test_gateway_development_allows_wildcard_defaults():
    config = make_config(environment="development")
    config.AUTH_USERS = {}
    config.DEV_AUTH_ENABLED = True
    config.CORS_ORIGINS = ["*"]
    config.ALLOWED_HOSTS = ["*"]

    gateway = APIGateway(config=config)

    assert gateway.config.ENVIRONMENT == "development"


class FakeRedis:
    def __init__(self):
        self.store = {}

    def eval(self, script, numkeys, *args):
        keys = list(args[:numkeys])
        token_cost, rpm, rph, tpm, tph = [int(value) for value in args[numkeys:]]
        current = [int(self.store.get(key, 0)) for key in keys]
        attempted = [
            current[0] + 1,
            current[1] + 1,
            current[2] + token_cost,
            current[3] + token_cost,
        ]

        def under(value, limit):
            return limit <= 0 or value <= limit

        allowed = (
            under(attempted[0], rpm)
            and under(attempted[1], rph)
            and under(attempted[2], tpm)
            and under(attempted[3], tph)
        )
        if allowed:
            for key, value in zip(keys, attempted):
                self.store[key] = value
            return [1, *attempted]
        return [0, *attempted]


def test_redis_rate_limit_rejection_does_not_consume_quota():
    redis = FakeRedis()
    limiter = RateLimiter(
        redis_client=redis,
        requests_per_minute=1,
        requests_per_hour=100,
        tokens_per_minute=100,
        tokens_per_hour=1000,
    )

    first_allowed, first_remaining = limiter.check_rate_limit("user", token_cost=3)
    second_allowed, second_remaining = limiter.check_rate_limit("user", token_cost=3)

    request_counts = [
        value
        for key, value in redis.store.items()
        if key.startswith("ratelimit:user:") and not key.endswith(":tokens")
    ]
    token_counts = [value for key, value in redis.store.items() if key.endswith(":tokens")]

    assert first_allowed is True
    assert first_remaining["requests_per_minute"] == 0
    assert second_allowed is False
    assert second_remaining["requests_per_minute"] == 0
    assert sorted(request_counts) == [1, 1]
    assert sorted(token_counts) == [3, 3]
