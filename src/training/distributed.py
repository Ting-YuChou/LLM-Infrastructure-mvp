"""
Process-group helpers shared by the trainers when launched with torchrun.

Every helper degrades to a single-process no-op when WORLD_SIZE is unset, so the
same code path runs on a laptop, a single GPU and an 8-GPU node. torch is
imported lazily so the pipeline module still imports on hosts without torch.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from datetime import timedelta
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _dist():
    try:
        import torch.distributed as dist
    except ImportError:  # host checks run without torch
        return None
    if dist.is_available() and dist.is_initialized():
        return dist
    return None


def rank() -> int:
    dist = _dist()
    return dist.get_rank() if dist else int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    dist = _dist()
    return dist.get_world_size() if dist else int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return rank() == 0


def is_distributed() -> bool:
    return world_size() > 1


def distributed_timeout_minutes(raw_config: Optional[Mapping[str, Any]]) -> float:
    """distributed.timeout_minutes from a training YAML (default 60)."""
    distributed = (raw_config or {}).get("distributed") or {}
    return float(distributed.get("timeout_minutes", 60))


def init_distributed(timeout_minutes: float = 60):
    """
    Join the torchrun process group (once) and return this rank's device.

    Without WORLD_SIZE>1 this only picks cuda:0 or cpu. Non-main ranks log at
    WARNING so an 8-GPU run does not print every line eight times.
    """
    import torch
    import torch.distributed as dist

    use_cuda = torch.cuda.is_available()
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        if use_cuda:
            torch.cuda.set_device(local_rank())
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if use_cuda else "gloo",
                timeout=timedelta(minutes=timeout_minutes),
            )
        if not is_main_process():
            logging.getLogger().setLevel(logging.WARNING)
            try:
                from transformers.utils import logging as hf_logging

                hf_logging.set_verbosity_warning()
            except ImportError:
                pass
    if use_cuda:
        return torch.device("cuda", local_rank())
    return torch.device("cpu")


def barrier() -> None:
    dist = _dist()
    if dist is None or dist.get_world_size() == 1:
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[local_rank()])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    dist = _dist()
    if dist is not None:
        dist.destroy_process_group()


@contextlib.contextmanager
def main_process_first() -> Iterator[None]:
    """Run the block on rank 0 first (e.g. to fill the datasets cache), then the rest."""
    if not is_main_process():
        barrier()
    try:
        yield
    finally:
        if is_main_process():
            barrier()


def main_process_only(fn: Callable[..., T]) -> Callable[..., Optional[T]]:
    """Run fn on rank 0 only; other ranks get None. No barrier is implied."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        if is_main_process():
            return fn(*args, **kwargs)
        return None

    return wrapper


def all_reduce_mean(tensor):
    """Mean of a tensor across ranks (returned unchanged without a process group)."""
    dist = _dist()
    if dist is None:
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / dist.get_world_size()


def masked_global_mean(values, mask):
    """Mean of values where mask is set, counted over every rank's tokens."""
    import torch

    mask = mask.to(values.dtype)
    totals = torch.stack([(values * mask).sum(), mask.sum()]).detach()
    dist = _dist()
    if dist is not None:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals[0] / totals[1].clamp(min=1.0)


def masked_global_whiten(values, mask, eps: float = 1e-8):
    """Whiten masked values with mean/std pooled across ranks, so every rank agrees."""
    import torch

    float_mask = mask.to(values.dtype)
    stats = torch.stack(
        [(values * float_mask).sum(), (values.pow(2) * float_mask).sum(), float_mask.sum()]
    ).detach()
    dist = _dist()
    if dist is not None:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    count = stats[2].clamp(min=1.0)
    mean = stats[0] / count
    variance = (stats[1] / count - mean.pow(2)).clamp(min=0.0)
    return (values - mean) * torch.rsqrt(variance + eps) * float_mask


def reduce_scalar_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    """Average a dict of python floats across ranks in one all-reduce."""
    dist = _dist()
    if dist is None or not metrics:
        return dict(metrics)
    import torch

    keys = sorted(metrics)
    device = torch.device("cuda", local_rank()) if dist.get_backend() == "nccl" else torch.device("cpu")
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return dict(zip(keys, values.tolist()))


def save_trainer_model(trainer, output_dir: str, dtype=None) -> None:
    """
    Save the final model from every rank's call.

    Under FSDP, Trainer.save_model writes nothing with a sharded state dict and
    fp32 with a full one, so gather the full state dict here (a collective) and
    let rank 0 write it in the training dtype. Every rank must call this.
    """
    if getattr(trainer, "is_fsdp_enabled", False):
        import torch

        state_dict = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.args.should_save:
            target = dtype or torch.bfloat16
            state_dict = {
                name: tensor.to(target) if tensor.is_floating_point() else tensor
                for name, tensor in state_dict.items()
            }
            trainer.accelerator.unwrap_model(trainer.model).save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size="5GB",
            )
    else:
        trainer.save_model(output_dir)
    barrier()
