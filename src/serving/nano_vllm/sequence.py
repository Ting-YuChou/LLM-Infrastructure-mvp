"""Request sequence state for nano-vLLM continuous batching."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Optional

from .sampling import SamplingOptions


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class GenerationEvent:
    token_id: Optional[int] = None
    finished: bool = False
    error: Optional[BaseException] = None


@dataclass
class Sequence:
    """One generation request tracked by the scheduler."""

    prompt_token_ids: list[int]
    sampling: SamplingOptions
    eos_token_id: Optional[int] = None
    ignore_eos: bool = False
    seq_id: int = field(default_factory=lambda: next(Sequence._counter))
    status: SequenceStatus = SequenceStatus.WAITING
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    is_prefill: bool = True
    block_table: list[int] = field(default_factory=list)
    token_ids: list[int] = field(init=False)
    events: Optional[asyncio.Queue[GenerationEvent]] = None

    _counter = count()
    block_size = 16

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("Sequence requires at least one prompt token")
        self.token_ids = list(self.prompt_token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, index: int) -> list[int]:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)
        start = index * self.block_size
        return self.token_ids[start : start + self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def should_finish(self, token_id: int) -> bool:
        reached_eos = (
            self.eos_token_id is not None
            and not self.ignore_eos
            and token_id == self.eos_token_id
        )
        reached_max = self.num_completion_tokens >= self.sampling.max_tokens
        return reached_eos or reached_max
