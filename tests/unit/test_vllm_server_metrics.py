import asyncio
import importlib
import sys
import types

import pytest


def _install_fake_vllm(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.LLM = object

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm_module.SamplingParams = SamplingParams

    engine_module = types.ModuleType("vllm.engine")
    arg_utils_module = types.ModuleType("vllm.engine.arg_utils")
    async_engine_module = types.ModuleType("vllm.engine.async_llm_engine")

    class AsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            return cls()

    arg_utils_module.AsyncEngineArgs = AsyncEngineArgs
    async_engine_module.AsyncLLMEngine = AsyncLLMEngine

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_module)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.engine.async_llm_engine",
        async_engine_module,
    )


@pytest.mark.asyncio
async def test_request_admission_controller_exports_active_and_queue(monkeypatch):
    """The serving admission queue backs the HPA queue-depth signal."""
    _install_fake_vllm(monkeypatch)
    module = importlib.import_module("src.serving.vllm_server")
    metrics = module.monitoring_metrics
    controller = module.RequestAdmissionController(
        max_concurrent_requests=1,
        model_name="admission-test-model",
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_slot():
        async with controller.admitted():
            entered.set()
            await release.wait()

    async def wait_for_slot():
        async with controller.admitted():
            return

    first = asyncio.create_task(hold_slot())
    await entered.wait()

    second = asyncio.create_task(wait_for_slot())
    await asyncio.sleep(0)

    assert metrics.active_requests.labels(model="admission-test-model")._value.get() == 1
    assert metrics.request_queue_size.labels(model="admission-test-model")._value.get() == 1

    release.set()
    await asyncio.gather(first, second)

    assert metrics.active_requests.labels(model="admission-test-model")._value.get() == 0
    assert metrics.request_queue_size.labels(model="admission-test-model")._value.get() == 0
