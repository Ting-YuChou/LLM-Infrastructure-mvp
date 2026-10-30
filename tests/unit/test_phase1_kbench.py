"""kbench's timing setup: how many input copies a shape rotates through."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "phase1"))

import kbench  # noqa: E402

L2 = 40 * 1024**2  # A100 40GB
DOWN = {"layer": "down_proj", "M": 1, "N": 4096, "K": 14336}
GATE_UP = {"layer": "gate_up_proj", "M": 1, "N": 28672, "K": 4096}


def _input_bytes(kernel, shape):
    return sum(t.nbytes for t in kbench.tensors(kernel.inputs(shape, "meta")))


@pytest.mark.parametrize("name", ["w4a16_linear", "w4a16_dequant"])
def test_w4a16_rotates_by_the_4bit_weight(name):
    kernel = kbench.KERNELS[name]
    shape = DOWN if name == "w4a16_linear" else {k: v for k, v in DOWN.items() if k != "M"}
    # The inputs hold the codes a byte each, twice the 4-bit weight: rotating by
    # them would time down_proj's 30 MB weight with half the copies it needs.
    assert _input_bytes(kernel, shape) > 1.9 * kbench._w4a16_weight_bytes(shape)
    assert kbench.copies_for(_input_bytes(kernel, shape), L2) < 18
    assert kbench.timing_copies(kernel, shape, _input_bytes(kernel, shape), L2) == 18
    gate_up = GATE_UP if name == "w4a16_linear" else {k: v for k, v in GATE_UP.items() if k != "M"}
    assert kbench.timing_copies(kernel, gate_up, _input_bytes(kernel, gate_up), L2) == 9


def test_other_kernels_rotate_by_their_inputs():
    for name, kernel in kbench.KERNELS.items():
        if kernel.rotate_bytes is not None:
            continue
        shape = kernel.shapes[0]
        size = _input_bytes(kernel, shape)
        assert kbench.timing_copies(kernel, shape, size, L2) == kbench.copies_for(size, L2), name
