"""gemm_breakdown.py: labelling decode GEMMs by their place in the layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "phase1"))

import gemm_breakdown  # noqa: E402


def _layer(gemm, activation, qkv, o, gate_up, down):
    """One decoder layer's kernels as either engine launches them."""
    return [
        ("rms_norm", 2), (gemm, qkv), ("rope", 3), ("attention", 10), (gemm, o),
        ("fused_add_rms_norm", 3), (gemm, gate_up), (activation, 2), (gemm, down), ("fused_add_rms_norm", 3),
    ]


def test_labels_follow_the_activation_kernel():
    gemm = "void (anonymous namespace)::gemm_w4a16_tiled_kernel<__nv_bfloat16, 1, 256, 128, false>"
    # A capture that starts mid-layer: its first two GEMMs (gate_up, down of a
    # layer whose qkv and o were not captured) must be left out.
    kernels = [(gemm, 50.0), ("_silu_and_mul_kernel", 2), (gemm, 30.0)]
    for _ in range(3):
        kernels += _layer(gemm, "_silu_and_mul_kernel", 17.0, 15.0, 53.0, 31.0)
    names, durations = zip(*kernels)
    layers = gemm_breakdown.label_layers(list(names), list(durations), "gemm_w4a16")
    assert layers == {"qkv": [17.0] * 3, "o_proj": [15.0] * 3, "gate_up": [53.0] * 3, "down": [31.0] * 3}
    assert gemm_breakdown.summary(layers)["sum_us"] == 116.0


def test_vllm_kernel_names():
    kernels = []
    for _ in range(2):
        kernels += _layer("void marlin::Marlin<...>", "void vllm::act_and_mul_kernel<...>", 18.0, 16.0, 60.0, 33.0)
    names, durations = zip(*kernels)
    layers = gemm_breakdown.label_layers(list(names), list(durations), "Marlin")
    assert [v[0] for v in layers.values()] == [18.0, 16.0, 60.0, 33.0] and len(layers["down"]) == 2
