"""Generation engine wrapper for the nano-vLLM server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .config import NanoVLLMConfig
from .model import NanoLlamaForCausalLM
from .sampling import SamplingOptions, sample_next_token


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    token_ids: list[int]


class NanoLlamaEngine:
    """Single-model, single-GPU decoder loop."""

    def __init__(self, config: NanoVLLMConfig):
        self.config = config
        self.model: Optional[NanoLlamaForCausalLM] = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_path = self.config.tokenizer or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = NanoLlamaForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=self.config.dtype,
            device=self.config.device,
            use_triton=self.config.use_triton,
        )
        self.initialized = True

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling: SamplingOptions,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        if not self.initialized or self.model is None or self.tokenizer is None:
            raise RuntimeError("nano-vLLM engine is not initialized")

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.model.device)
        if input_ids.shape[0] != 1:
            raise ValueError("nano-vLLM v0.1 supports batch size 1 inside the engine")
        input_ids = input_ids[:, -self.config.max_model_len :]
        prompt_tokens = int(input_ids.shape[1])
        max_new_tokens = max(1, min(sampling.max_tokens, self.config.max_new_tokens))
        max_seq_len = min(self.config.max_model_len, prompt_tokens + max_new_tokens)
        cache = self.model.new_cache(batch_size=1, max_seq_len=max_seq_len)

        logits = self.model.forward(input_ids, cache=cache, start_pos=0)
        generated: list[int] = []
        current_pos = prompt_tokens
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(seed)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits[0, -1],
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                generator=generator,
            )
            if self._is_eos(next_token):
                break
            generated.append(next_token)
            if current_pos >= max_seq_len:
                break
            token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.model.device
            )
            logits = self.model.forward(
                token_tensor, cache=cache, start_pos=current_pos
            )
            current_pos += 1

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = self._apply_stop_strings(text, sampling.stop or [])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated),
            token_ids=generated,
        )

    def _is_eos(self, token_id: int) -> bool:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return False
        if isinstance(eos, list):
            return token_id in eos
        return token_id == eos

    def _apply_stop_strings(self, text: str, stop: list[str]) -> str:
        cutoff = None
        for marker in stop:
            index = text.find(marker)
            if index >= 0:
                cutoff = index if cutoff is None else min(cutoff, index)
        return text if cutoff is None else text[:cutoff]
