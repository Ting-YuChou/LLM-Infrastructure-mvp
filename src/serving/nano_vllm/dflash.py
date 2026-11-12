"""DFlash block-diffusion drafter for speculative decoding.

A DFlash drafter (https://arxiv.org/abs/2602.06036; the reference forward is
``dflash.py`` in the checkpoint's Hugging Face repo) proposes a whole block
of tokens in one forward pass. It is a few Qwen3-style decoder layers with no
embedding or output head of its own: it reads the target model through

- context features: the target's residual stream after the layers in
  ``target_layer_ids`` (transformers' ``hidden_states[layer + 1]``),
  concatenated per token, projected by ``fc`` and normalized by
  ``hidden_norm``. One feature per context token, the same for every layer;
- the target's token embedding, for the block [anchor, MASK x K], where the
  anchor is the last accepted token (its target features do not exist yet);
- the target's lm_head, on the drafter's own final ``norm`` of the K mask
  positions.

Each layer attends from the block tokens only. Its keys and values are
``k_proj``/``v_proj`` of [context features, block hidden states]: the
features go in as they are, the block states through ``input_layernorm``.
Keys get ``k_norm`` and queries ``q_norm`` (RMSNorm over head_dim) before
RoPE at each token's own position, and there is no mask: a block token sees
all of the context and all of the block.

A context token's K/V depend only on its feature, so they are computed once
(:meth:`DFlashDrafter.write_context`) into a paged KV cache with the
drafter's five layers that shares block ids with the target's cache. A
draft (:meth:`DFlashDrafter.draft`) writes the block's own K/V into the
slots after the context, the lookahead slots, and reads context and block
back with one block attention; the next ``write_context`` overwrites them.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .kernels import (
    W4A16Weight,
    linear,
    load_cuda_extension,
    norm_rope_and_cache_write,
    paged_block_attention,
    quantize_w4a16,
    rms_norm,
    rope_cos_sin,
    silu_and_mul,
)
from .kernels.rotary import compute_inv_freq
from .kernels.w4a16 import W4A16_CONFIG, w4a16_tiled_supported, w4a16_workspace
from .kv_cache import PagedKVCache
from .model import Linear, NanoLlamaForCausalLM, _build_on_stream, _concat_projections, _load_state_dict

# "w4a16" RTN-quantizes the layer projections at load. It changes which
# tokens are proposed, never what the target accepts.
DFLASH_WEIGHT_QUANT_MODES = ("none", "w4a16")
# fc (read once per context token) and the norms stay in the model dtype.
DFLASH_QUANTIZED_LINEARS = (
    "self_attn.q_proj.weight",
    "self_attn.kv_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
)


def default_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """The reference's ``build_target_layer_ids``: evenly spaced target layers
    from 1 to ``num_target_layers - 3``."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    return [
        int(round(start + i * (end - start) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


@dataclass(frozen=True)
class DFlashConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    block_size: int
    mask_token_id: int
    # Target layers whose output residual feeds ``fc``, in concatenation order.
    target_layer_ids: tuple[int, ...]
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    # The target's layer count the checkpoint was trained against, when given.
    num_target_layers: Optional[int] = None
    torch_dtype: Optional[str] = None

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError("a DFlash block is the anchor plus at least one mask token")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if not self.target_layer_ids:
            raise ValueError("a DFlash drafter needs at least one target layer")

    @property
    def num_draft_tokens(self) -> int:
        """K: the mask positions of a full block."""
        return self.block_size - 1

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "DFlashConfig":
        data = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        dflash = data.get("dflash_config")
        if dflash is None or "block_size" not in data:
            raise ValueError(
                f"{model_path}: not a DFlash drafter (config.json needs dflash_config and block_size)"
            )
        if dflash.get("mask_token_id") is None:
            raise ValueError(f"{model_path}: dflash_config has no mask_token_id")
        # What the forward below does not model.
        unsupported = {
            "attention_bias": bool(data.get("attention_bias", False)),
            "rope_scaling": bool(data.get("rope_scaling")),
            "use_sliding_window": bool(data.get("use_sliding_window", False)),
            "hidden_act": str(data.get("hidden_act", "silu")) != "silu",
        }
        if any(unsupported.values()):
            found = {name: data.get(name) for name, bad in unsupported.items() if bad}
            raise ValueError(f"{model_path}: unsupported DFlash drafter config: {found}")
        heads = int(data["num_attention_heads"])
        layers = int(data["num_hidden_layers"])
        num_target_layers = data.get("num_target_layers")
        target_layer_ids = dflash.get("target_layer_ids")
        if target_layer_ids is None:
            if num_target_layers is None:
                raise ValueError(
                    f"{model_path}: needs dflash_config.target_layer_ids or num_target_layers"
                )
            target_layer_ids = default_target_layer_ids(int(num_target_layers), layers)
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=int(data.get("num_key_value_heads", heads)),
            head_dim=int(data.get("head_dim") or int(data["hidden_size"]) // heads),
            block_size=int(data["block_size"]),
            mask_token_id=int(dflash["mask_token_id"]),
            target_layer_ids=tuple(int(layer) for layer in target_layer_ids),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            num_target_layers=None if num_target_layers is None else int(num_target_layers),
            torch_dtype=data.get("torch_dtype") or data.get("dtype"),
        )


@dataclass(frozen=True)
class _DraftLayer:
    input_norm: torch.Tensor
    q: Linear
    kv: Linear  # k_proj and v_proj as one projection
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o_proj: Linear
    post_norm: torch.Tensor
    gate_up: Linear
    down: Linear


class DFlashDrafter:
    """A DFlash drafter on the paged KV cache, next to its target model.

    ``weights`` is the checkpoint's state dict (``layers.<i>.…``, ``fc``,
    ``hidden_norm``, ``norm``). From ``target`` it takes the device, dtype
    and kernel backends, the embedding (``weights["model.embed_tokens.weight"]``)
    and ``lm_head_weight``, both shared and not copied, and the length of its
    RoPE table.
    """

    def __init__(
        self,
        config: DFlashConfig,
        weights: Dict[str, torch.Tensor],
        target: NanoLlamaForCausalLM,
        weight_quant: str = "none",
        max_positions: Optional[int] = None,
    ) -> None:
        if weight_quant not in DFLASH_WEIGHT_QUANT_MODES:
            raise ValueError(
                f"the drafter's weight_quant must be one of: {', '.join(DFLASH_WEIGHT_QUANT_MODES)}"
            )
        self._check_target(config, target)
        self.config = config
        self.target = target
        self.device = target.device
        self.dtype = target.dtype
        self.use_triton = target.use_triton
        self.attention_backend = target.attention_backend
        self.linear_backend = target.linear_backend
        self.kv_cache_dtype = target.kv_cache_dtype
        self.weight_quant = weight_quant
        if weight_quant == "w4a16" and self.device.type == "cuda":
            load_cuda_extension("w4a16")  # never inside a CUDA-graph capture

        self._validate_required_weights(weights)
        self.weights = self._load_weights(self._fuse(weights))
        quantized = [w for w in self.weights.values() if isinstance(w, W4A16Weight)]
        if quantized and self.device.type == "cuda":
            w4a16_workspace(self.device, max(w.shape[0] for w in quantized))
        self.fc = self.weights["fc.weight"]
        self.hidden_norm = self.weights["hidden_norm.weight"]
        self.norm = self.weights["norm.weight"]
        self.layers = [self._layer_weights(i) for i in range(config.num_hidden_layers)]
        self.embed_tokens = target.weights["model.embed_tokens.weight"]
        self.scale = config.head_dim**-0.5

        # The drafter's own frequencies; a block reaches block_size - 1
        # positions past the last one the target computes.
        inv_freq = compute_inv_freq(config.head_dim, theta=config.rope_theta, device=self.device)
        self.rope_cos, self.rope_sin = rope_cos_sin(
            inv_freq, max_positions or target.rope_cos.shape[0] + config.block_size, self.dtype
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        target: NanoLlamaForCausalLM,
        weight_quant: str = "none",
        max_positions: Optional[int] = None,
    ) -> "DFlashDrafter":
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"DFlash drafter path does not exist: {path}")
        return cls(
            DFlashConfig.from_pretrained(path), _load_state_dict(path), target,
            weight_quant=weight_quant, max_positions=max_positions,
        )

    def new_paged_cache(self, num_blocks: int, block_size: int) -> PagedKVCache:
        """The drafter's KV pool: its own layers over the target cache's block
        ids (same ``num_blocks`` and ``block_size``), so one block table and
        one slot mapping address both."""
        return PagedKVCache(
            num_layers=self.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_key_value_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    @torch.inference_mode()
    def write_context(
        self,
        taps: torch.Tensor,
        positions: torch.Tensor,
        dcache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Every layer's K/V of context tokens, into ``dcache`` at ``slot_mapping``.

        ``taps`` is [tokens, len(target_layer_ids) * hidden]: per token, the
        target's residual after each layer in ``target_layer_ids``, in that
        order. ``positions`` [tokens] are the tokens' positions in their
        sequences. Tokens are independent, so any packing and chunking of a
        context gives the same cache.
        """
        features = rms_norm(
            self._linear(taps, self.fc), self.hidden_norm,
            eps=self.config.rms_norm_eps, use_triton=self.use_triton,
        )
        rope = (self.rope_cos[positions], self.rope_sin[positions])
        for layer_idx, layer in enumerate(self.layers):
            self._rope_and_write(
                layer_idx, None, self._linear(features, layer.kv), rope, dcache, slot_mapping
            )

    @torch.inference_mode()
    def draft(
        self,
        anchor_ids: torch.Tensor,
        positions: torch.Tensor,
        dcache: PagedKVCache,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising pass over a block per sequence: logits [B * K, vocab].

        Sequence ``b`` has ``context_lens[b]`` tokens in ``dcache``
        (:meth:`write_context`) and the block [``anchor_ids[b]``, MASK x K];
        row ``b * K + j`` holds the logits of the token ``j + 1`` positions
        after the anchor. ``positions`` and ``slot_mapping`` are [B * Q],
        Q = K + 1, row ``b * Q + t`` for block token ``t``: its position
        (``context_lens[b] + t``) and its lookahead slot, the slot of that
        position in ``block_tables[b]``. The block's K/V are written there;
        nothing else in the cache changes. K is taken from the shapes, at
        most ``block_size - 1``.
        """
        config = self.config
        batch = anchor_ids.shape[0]
        q_max = positions.numel() // max(batch, 1)
        if positions.numel() != batch * q_max or slot_mapping.shape != positions.shape:
            raise ValueError("positions and slot_mapping must hold batch * (K + 1) entries")
        if not 2 <= q_max <= config.block_size:
            raise ValueError(
                f"a block is the anchor plus 1 to {config.block_size - 1} mask tokens, got {q_max} rows"
            )
        rows = batch * q_max
        input_ids = torch.full((batch, q_max), config.mask_token_id, dtype=torch.long, device=self.device)
        input_ids[:, 0] = anchor_ids
        x = self.embed_tokens[input_ids.view(-1)]
        rope = (self.rope_cos[positions], self.rope_sin[positions])
        # The block is in the cache by the time each layer attends.
        total_lens = context_lens + q_max
        for layer_idx, layer in enumerate(self.layers):
            x_norm = self._norm(x, layer.input_norm)
            query = self._rope_and_write(
                layer_idx, self._linear(x_norm, layer.q), self._linear(x_norm, layer.kv),
                rope, dcache, slot_mapping,
            )
            attn = paged_block_attention(
                query, dcache.keys[layer_idx], dcache.values[layer_idx], block_tables,
                total_lens, None, q_max, self.scale, causal=False,
                use_triton=self.attention_backend == "triton",
                key_scales=dcache.key_scales[layer_idx], value_scales=dcache.value_scales[layer_idx],
            )
            x = x + self._linear(attn.reshape(rows, -1), layer.o_proj)
            gate_up = self._linear(self._norm(x, layer.post_norm), layer.gate_up)
            x = x + self._linear(silu_and_mul(gate_up, use_triton=self.use_triton), layer.down)
        # The anchor row is context for the block, not a prediction.
        x = x.view(batch, q_max, -1)[:, 1:].reshape(batch * (q_max - 1), -1)
        return self._linear(self._norm(x, self.norm), self.target.lm_head_weight)

    def _rope_and_write(
        self,
        layer_idx: int,
        query: Optional[torch.Tensor],
        kv: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        dcache: PagedKVCache,
        slot_mapping: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """q_norm/k_norm, RoPE, and K/V into the cache: the rotated query
        [tokens, heads, head_dim] (None for context tokens, which have none)."""
        config, layer = self.config, self.layers[layer_idx]
        return norm_rope_and_cache_write(
            query, kv, layer.q_norm, layer.k_norm, config.rms_norm_eps, rope[0], rope[1],
            dcache.keys[layer_idx], dcache.values[layer_idx], slot_mapping,
            config.num_attention_heads, config.num_key_value_heads,
            use_triton=self.attention_backend == "triton",
            key_scales=dcache.key_scales[layer_idx], value_scales=dcache.value_scales[layer_idx],
        )

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, weight, eps=self.config.rms_norm_eps, use_triton=self.use_triton)

    def _linear(self, x: torch.Tensor, weight: Linear) -> torch.Tensor:
        return linear(x, weight, backend=self.linear_backend)

    @staticmethod
    def _check_target(config: DFlashConfig, target: NanoLlamaForCausalLM) -> None:
        tc = target.config
        if (config.hidden_size, config.vocab_size) != (tc.hidden_size, tc.vocab_size):
            raise ValueError(
                "the drafter shares the target's embedding and lm_head: hidden_size and "
                f"vocab_size must match (drafter {config.hidden_size}/{config.vocab_size}, "
                f"target {tc.hidden_size}/{tc.vocab_size})"
            )
        if config.num_target_layers not in (None, tc.num_hidden_layers):
            raise ValueError(
                f"the drafter was trained for a {config.num_target_layers}-layer target, "
                f"this one has {tc.num_hidden_layers}"
            )
        if not all(0 <= layer < tc.num_hidden_layers for layer in config.target_layer_ids):
            raise ValueError(
                f"target_layer_ids {list(config.target_layer_ids)} are outside the target's "
                f"{tc.num_hidden_layers} layers"
            )
        if not 0 <= config.mask_token_id < tc.vocab_size:
            raise ValueError(f"mask_token_id {config.mask_token_id} is outside the vocabulary")

    def _validate_required_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        config = self.config
        attn_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        hidden, inter = config.hidden_size, config.intermediate_size
        required: Dict[str, tuple[int, ...]] = {
            "fc.weight": (hidden, len(config.target_layer_ids) * hidden),
            "hidden_norm.weight": (hidden,),
            "norm.weight": (hidden,),
        }
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layers.{layer_idx}"
            required.update({
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (attn_dim, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_dim, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_dim, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, attn_dim),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (inter, hidden),
                f"{prefix}.mlp.up_proj.weight": (inter, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, inter),
            })
        missing = [name for name in required if name not in weights]
        if missing:
            raise KeyError(f"DFlash checkpoint is missing required weights: {missing[:5]}")
        wrong = {
            name: tuple(weights[name].shape)
            for name, shape in required.items() if tuple(weights[name].shape) != shape
        }
        if wrong:
            name, shape = next(iter(wrong.items()))
            raise ValueError(
                f"DFlash checkpoint does not match its config: {name} is {shape}, "
                f"expected {required[name]}"
            )
        biases = [name for name in weights if name.endswith(".bias")]
        if biases:
            raise ValueError(f"DFlash projections have no bias; the checkpoint holds {biases[:3]}")

    def _fuse(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """k/v -> ``self_attn.kv_proj`` (context tokens need K and V but no
        query, so q_proj stays apart) and gate/up -> ``mlp.gate_up_proj``."""
        weights = dict(weights)
        for layer_idx in range(self.config.num_hidden_layers):
            attn, mlp = f"layers.{layer_idx}.self_attn", f"layers.{layer_idx}.mlp"
            _concat_projections(weights, [f"{attn}.k_proj", f"{attn}.v_proj"], f"{attn}.kv_proj")
            _concat_projections(weights, [f"{mlp}.gate_proj", f"{mlp}.up_proj"], f"{mlp}.gate_up_proj")
        return weights

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """As the target's: on CUDA each weight is built on a side stream, so
        quantization temporaries stay out of the weights' segments."""
        if self.device.type != "cuda":
            return {name: self._load_weight(name, tensor) for name, tensor in weights.items()}
        scratch = torch.cuda.Stream(self.device)
        loaded = {
            name: _build_on_stream(scratch, functools.partial(self._load_weight, name, tensor))
            for name, tensor in weights.items()
        }
        torch.cuda.empty_cache()
        return loaded

    def _load_weight(self, name: str, tensor: torch.Tensor) -> Linear:
        moved = (
            tensor.to(device=self.device, dtype=self.dtype)
            if tensor.is_floating_point() else tensor.to(device=self.device)
        )
        if self.weight_quant == "w4a16" and name.endswith(DFLASH_QUANTIZED_LINEARS):
            tiled = W4A16_CONFIG["layout"] == "tiled" and w4a16_tiled_supported(*moved.shape)
            return quantize_w4a16(moved, layout="tiled" if tiled else "rows")
        return moved

    def _layer_weights(self, layer_idx: int) -> _DraftLayer:
        prefix = f"layers.{layer_idx}"
        return _DraftLayer(
            input_norm=self.weights[f"{prefix}.input_layernorm.weight"],
            q=self.weights[f"{prefix}.self_attn.q_proj.weight"],
            kv=self.weights[f"{prefix}.self_attn.kv_proj.weight"],
            q_norm=self.weights[f"{prefix}.self_attn.q_norm.weight"],
            k_norm=self.weights[f"{prefix}.self_attn.k_norm.weight"],
            o_proj=self.weights[f"{prefix}.self_attn.o_proj.weight"],
            post_norm=self.weights[f"{prefix}.post_attention_layernorm.weight"],
            gate_up=self.weights[f"{prefix}.mlp.gate_up_proj.weight"],
            down=self.weights[f"{prefix}.mlp.down_proj.weight"],
        )
