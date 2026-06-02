"""Minimal Llama-family decoder used by the nano-vLLM backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .kernels import apply_rope, causal_attention, rms_norm, silu_mul
from .kv_cache import StaticKVCache


def torch_dtype_from_string(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"auto", "float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for nano-vLLM: {value}")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@dataclass(frozen=True)
class NanoLlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "NanoLlamaConfig":
        path = Path(model_path) / "config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        model_type = str(data.get("model_type", "llama")).lower()
        if model_type not in {"llama", "mistral"}:
            raise ValueError(
                "nano-vLLM v0.1 supports Llama-family decoder configs only, "
                f"got model_type={model_type!r}"
            )
        heads = int(data["num_attention_heads"])
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=int(data.get("num_key_value_heads", heads)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            pad_token_id=data.get("pad_token_id"),
        )


def _load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        state: Dict[str, torch.Tensor] = {}
        for file_path in safetensor_files:
            state.update(load_file(str(file_path), device="cpu"))
        return state

    bin_path = model_path / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"No .safetensors or pytorch_model.bin weights found in {model_path}"
    )


class NanoLlamaForCausalLM:
    """Small, explicit Llama forward path for learning serving internals."""

    def __init__(
        self,
        config: NanoLlamaConfig,
        weights: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        use_triton: bool = True,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype if device.type == "cuda" else torch.float32
        self.use_triton = use_triton
        self.weights = {
            name: self._move_weight(tensor) for name, tensor in weights.items()
        }
        self._validate_required_weights()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        dtype: str = "float16",
        device: str = "auto",
        use_triton: bool = True,
    ) -> "NanoLlamaForCausalLM":
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"nano-vLLM model path does not exist: {path}")
        config = NanoLlamaConfig.from_pretrained(path)
        resolved_device = resolve_device(device)
        return cls(
            config=config,
            weights=_load_state_dict(path),
            device=resolved_device,
            dtype=torch_dtype_from_string(dtype),
            use_triton=use_triton,
        )

    def new_cache(self, batch_size: int, max_seq_len: int) -> StaticKVCache:
        return StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            num_key_value_heads=self.config.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: StaticKVCache,
        start_pos: int = 0,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        batch_size, seq_len = input_ids.shape
        if batch_size != cache.batch_size:
            raise ValueError("input batch size must match cache batch size")

        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            dtype=torch.long,
            device=self.device,
        )
        x = self.weights["model.embed_tokens.weight"][input_ids.to(self.device)]

        for layer_idx in range(self.config.num_hidden_layers):
            residual = x
            x_norm = rms_norm(
                x,
                self._weight(f"model.layers.{layer_idx}.input_layernorm.weight"),
                eps=self.config.rms_norm_eps,
                use_triton=self.use_triton,
            )
            query = self._linear(x_norm, f"model.layers.{layer_idx}.self_attn.q_proj")
            key = self._linear(x_norm, f"model.layers.{layer_idx}.self_attn.k_proj")
            value = self._linear(x_norm, f"model.layers.{layer_idx}.self_attn.v_proj")
            query = self._shape_heads(query, self.config.num_attention_heads)
            key = self._shape_heads(key, self.config.num_key_value_heads)
            value = self._shape_heads(value, self.config.num_key_value_heads)
            query, key = apply_rope(
                query,
                key,
                positions=positions,
                theta=self.config.rope_theta,
                use_triton=self.use_triton,
            )

            cache.update(layer_idx, positions, key, value)
            full_key, full_value = cache.read(
                layer_idx, end_position=start_pos + seq_len
            )
            full_key = self._repeat_kv(full_key)
            full_value = self._repeat_kv(full_value)
            attn = causal_attention(
                query,
                full_key,
                full_value,
                query_positions=positions,
                use_triton=self.use_triton,
            )
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            x = residual + self._linear(
                attn, f"model.layers.{layer_idx}.self_attn.o_proj"
            )

            residual = x
            x_norm = rms_norm(
                x,
                self._weight(
                    f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                ),
                eps=self.config.rms_norm_eps,
                use_triton=self.use_triton,
            )
            gate = self._linear(x_norm, f"model.layers.{layer_idx}.mlp.gate_proj")
            up = self._linear(x_norm, f"model.layers.{layer_idx}.mlp.up_proj")
            x = residual + self._linear(
                silu_mul(gate, up, use_triton=self.use_triton),
                f"model.layers.{layer_idx}.mlp.down_proj",
            )

        x = rms_norm(
            x,
            self._weight("model.norm.weight"),
            eps=self.config.rms_norm_eps,
            use_triton=self.use_triton,
        )
        lm_head = self.weights.get(
            "lm_head.weight", self.weights["model.embed_tokens.weight"]
        )
        return F.linear(x, lm_head)

    def _weight(self, name: str) -> torch.Tensor:
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(f"Missing nano-vLLM weight: {name}") from exc

    def _move_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(device=self.device, dtype=self.dtype)
        return tensor.to(device=self.device)

    def _linear(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        weight = self._weight(f"{prefix}.weight")
        bias = self.weights.get(f"{prefix}.bias")
        return F.linear(x, weight, bias)

    def _shape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, num_heads, self.config.head_dim).transpose(
            1, 2
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeats == 1:
            return x
        return x.repeat_interleave(repeats, dim=1)

    def _validate_required_weights(self) -> None:
        required = [
            "model.embed_tokens.weight",
            "model.norm.weight",
        ]
        for layer_idx in range(self.config.num_hidden_layers):
            required.extend(
                [
                    f"model.layers.{layer_idx}.input_layernorm.weight",
                    f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                    f"model.layers.{layer_idx}.mlp.up_proj.weight",
                    f"model.layers.{layer_idx}.mlp.down_proj.weight",
                ]
            )
        missing = [name for name in required if name not in self.weights]
        if missing:
            raise KeyError(
                f"Model checkpoint is missing required Llama weights: {missing[:5]}"
            )
