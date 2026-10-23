"""NVTX ranges that show the engine's host-side phases in nsys timelines.

Off unless NANO_NVTX=1 and CUDA is available, so a disabled range costs one
attribute lookup and returns a shared no-op context manager.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch

_NULL = nullcontext()
ENABLED = os.environ.get("NANO_NVTX", "").strip() == "1" and torch.cuda.is_available()


class _Range:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *exc):
        torch.cuda.nvtx.range_pop()


def nvtx_range(name: str):
    """``with nvtx_range("schedule"):`` marks a range when NVTX is enabled."""
    return _Range(name) if ENABLED else _NULL
