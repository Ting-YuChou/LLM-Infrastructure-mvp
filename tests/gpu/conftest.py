"""Shared setup of the GPU tests."""

import gc

import pytest


@pytest.fixture(autouse=True)
def _collect_finished_tests_graphs():
    """Destroy a finished test's CUDA graphs before the next test runs.

    PyTorch keeps the default generator's graph state in tensors made when
    the first live graph is captured. An engine left in a reference cycle
    keeps its graphs alive, and if they were captured in inference mode (as
    the model runner's are), a later test that captures outside it fails with
    "Inplace update to inference tensor outside InferenceMode".
    """
    yield
    gc.collect()
