"""engine.multiprocess: the engine core in a spawned process, end to end on CPU.

A tiny real Llama and tokenizer are saved to disk; the core process loads the
model, the test process keeps the tokenizer, detokenization and HTTP.
"""

import asyncio
import json
import sys
import time
from contextlib import aclosing
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from tiny_llama import build_tiny_llama, build_tiny_tokenizer, real_transformers_available  # noqa: E402

if not real_transformers_available():  # pragma: no cover - stubbed transformers
    pytest.skip("needs the real transformers stack", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from serving.nano_vllm.config import NanoVLLMConfig  # noqa: E402
from serving.nano_vllm.engine import NanoLlamaEngine  # noqa: E402
from serving.nano_vllm.sampling import SamplingOptions  # noqa: E402
from serving.nano_vllm.server import NanoVLLMServer  # noqa: E402

GREEDY = SamplingOptions(max_tokens=6, temperature=0.0, ignore_eos=True)
PROMPTS = ["explain tests they catch regressions", "write good code", "hi"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_llama")
    tokenizer = build_tiny_tokenizer()
    build_tiny_llama(len(tokenizer)).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def _config(model_dir, multiprocess, **overrides):
    values = dict(
        model_name=str(model_dir),
        device="cpu",
        attention_backend="torch",
        linear_backend="torch",
        use_triton=False,
        max_model_len=64,
        max_new_tokens=16,
        kvcache_block_size=4,
        num_kvcache_blocks=64,
        max_num_seqs=4,
        max_num_batched_tokens=8,
        multiprocess=multiprocess,
        eager_load=True,
        enable_metrics=True,
    )
    values.update(overrides)
    return NanoVLLMConfig(**values)


async def _generate_all(engine):
    try:
        return await asyncio.gather(*(engine.generate_async(p, GREEDY) for p in PROMPTS))
    finally:
        await engine.shutdown()


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_core_process_matches_the_in_process_engine(model_dir, async_scheduling):
    expected = []
    for multiprocess in (False, True):
        engine = NanoLlamaEngine(
            _config(model_dir, multiprocess, async_scheduling=async_scheduling and multiprocess)
        )
        engine.initialize()
        results = asyncio.run(_generate_all(engine))
        expected.append([(r.token_ids, r.text, r.finish_reason) for r in results])
        if multiprocess:
            assert engine.core is None and engine.model is None  # all in the child
            assert not engine._core_process.process.is_alive()  # shut down
    assert expected[0] == expected[1]
    assert all(len(ids) == 6 for ids, _, _ in expected[1])


def test_core_process_aborts_closed_streams_and_reports_stats(model_dir):
    engine = NanoLlamaEngine(_config(model_dir, True, max_new_tokens=60))
    engine.initialize()

    async def scenario():
        try:
            seq = engine.prepare_request(PROMPTS[0], SamplingOptions(max_tokens=50, temperature=0.0, ignore_eos=True))
            async with aclosing(engine.stream_request(seq)) as stream:
                await stream.__anext__()
                await stream.__anext__()
            assert seq.finish_reason == "abort"
            # The core sends its stats when it goes idle again.
            deadline = time.monotonic() + 10
            while engine.stats().get("num_running", 1) or engine.stats().get("kv_blocks_used", 1):
                assert time.monotonic() < deadline, engine.stats()
                await asyncio.sleep(0.02)
            return await engine.generate_async(PROMPTS[1], GREEDY)
        finally:
            await engine.shutdown()

    result = asyncio.run(scenario())
    assert len(result.token_ids) == 6
    assert engine.stats()["prefix_query_tokens"] > 0


def test_server_streams_through_the_core_process(model_dir):
    server = NanoVLLMServer(_config(model_dir, True))
    body = {"prompt": PROMPTS[0], "max_tokens": 5, "temperature": 0.0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    with TestClient(server.app) as client:
        assert client.get("/ready").json()["status"] == "ready"
        response = client.post("/v1/completions", json=body)
        payloads = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert client.post("/start_profile").status_code == 200  # forwarded to the core
        assert client.post("/stop_profile").status_code == 200
    assert payloads[-1]["usage"]["completion_tokens"] == 5
    assert payloads[-2]["choices"][0]["finish_reason"] == "length"
    assert not server.engine._core_process.process.is_alive()
