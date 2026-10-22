"""JIT build of the CUDA kernels in csrc/ with torch.utils.cpp_extension."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def load_cuda_extension(source: str = "gemv_bf16"):
    """Build csrc/<source>.cu once per process; None if it cannot be built.

    Needs nvcc matching torch's CUDA version (CUDA_HOME) and ninja. Call it
    before capturing CUDA graphs: compiling inside a capture is not allowed.
    """
    try:
        from torch.utils.cpp_extension import load

        path = Path(__file__).with_name("csrc") / f"{source}.cu"
        return load(
            name=f"nano_vllm_{source}",
            sources=[str(path)],
            extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"],
            verbose=False,
        )
    except Exception:  # pragma: no cover - depends on the CUDA toolchain
        logger.exception("nano-vLLM: building csrc/%s.cu failed; using the PyTorch path", source)
        return None
