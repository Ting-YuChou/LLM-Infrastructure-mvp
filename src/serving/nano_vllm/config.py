"""Configuration for the nano-vLLM serving backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


def _env(name: str, default: Any = None) -> Any:
    return os.getenv(name, default)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class NanoVLLMConfig:
    """Runtime settings for the nano-vLLM backend."""

    model_name: str = "/workspace/models/nano/checkpoint-final"
    served_model_name: str = "nano-llm-local"
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "float16"
    device: str = "auto"
    max_model_len: int = 2048
    max_new_tokens: int = 256
    use_triton: bool = True
    eager_load: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 16
    enable_metrics: bool = True

    @classmethod
    def from_yaml(cls, config_path: str | None) -> "NanoVLLMConfig":
        data: dict[str, Any] = {}
        if config_path:
            path = Path(config_path)
            if path.is_file():
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        model = data.get("model", {})
        server = data.get("server", {})
        engine = data.get("engine", {})
        monitoring = data.get("monitoring", {})

        return cls(
            model_name=str(_env("MODEL_NAME", model.get("name", cls.model_name))),
            served_model_name=str(
                _env(
                    "SERVED_MODEL_NAME",
                    model.get("served_model_name", cls.served_model_name),
                )
            ),
            tokenizer=_optional_str(_env("TOKENIZER", model.get("tokenizer"))),
            trust_remote_code=_bool_value(
                _env("TRUST_REMOTE_CODE", model.get("trust_remote_code")),
                default=False,
            ),
            dtype=str(_env("DTYPE", model.get("dtype", cls.dtype))),
            device=str(_env("DEVICE", engine.get("device", cls.device))),
            max_model_len=int(
                _env("MAX_MODEL_LEN", model.get("max_model_len", cls.max_model_len))
            ),
            max_new_tokens=int(
                _env("MAX_NEW_TOKENS", engine.get("max_new_tokens", cls.max_new_tokens))
            ),
            use_triton=_bool_value(
                _env("USE_TRITON", engine.get("use_triton")),
                default=cls.use_triton,
            ),
            eager_load=_bool_value(
                _env("EAGER_LOAD", engine.get("eager_load")),
                default=cls.eager_load,
            ),
            host=str(_env("HOST", server.get("host", cls.host))),
            port=int(_env("PORT", server.get("port", cls.port))),
            max_concurrent_requests=int(
                _env(
                    "MAX_CONCURRENT_REQUESTS",
                    server.get("max_concurrent_requests", cls.max_concurrent_requests),
                )
            ),
            enable_metrics=_bool_value(
                _env("ENABLE_METRICS", monitoring.get("enable_metrics")),
                default=cls.enable_metrics,
            ),
        )
