"""Chat template, pad/stop tokens and PPO rollout masking, on a real tokenizer."""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from tiny_llama import (  # noqa: E402
    LLAMA3_PAD_TOKEN,
    build_tiny_llama,
    build_tiny_tokenizer,
    real_transformers_available,
)
from training import chat_format  # noqa: E402

pytestmark = pytest.mark.skipif(
    not real_transformers_available(),
    reason="needs real transformers and tokenizers",
)


@pytest.fixture
def chat_tokenizer():
    return chat_format.ensure_chat_tokenizer(build_tiny_tokenizer())


def _token_id(tokenizer, token):
    return tokenizer.convert_tokens_to_ids(token)


def test_ensure_chat_tokenizer_sets_template_pad_and_eot(chat_tokenizer):
    assert "<|start_header_id|>" in chat_tokenizer.chat_template
    assert chat_tokenizer.eos_token == chat_format.END_OF_TURN
    assert chat_tokenizer.pad_token not in {chat_format.END_OF_TURN, "</s>"}
    assert chat_format.stop_token_ids(chat_tokenizer) == [_token_id(chat_tokenizer, "<|eot_id|>")]


def test_ensure_chat_tokenizer_prefers_llama3_reserved_pad_and_is_idempotent():
    tokenizer = build_tiny_tokenizer(
        extra_special_tokens=list(chat_format.CHAT_SPECIAL_TOKENS) + [LLAMA3_PAD_TOKEN]
    )
    size = len(tokenizer)

    chat_format.ensure_chat_tokenizer(tokenizer)
    chat_format.ensure_chat_tokenizer(tokenizer)

    assert tokenizer.pad_token == LLAMA3_PAD_TOKEN
    assert len(tokenizer) == size


def test_prompt_and_completion_match_the_full_chat_template(chat_tokenizer):
    full = chat_tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "explain tests"},
            {"role": "assistant", "content": "they catch regressions"},
        ],
        tokenize=False,
    )

    rendered = chat_format.format_prompt(chat_tokenizer, "explain tests") + chat_format.format_completion(
        " they catch regressions "
    )

    assert rendered == full
    prompt_ids, completion_ids = chat_format.encode_prompt_and_completion(
        chat_tokenizer, "explain tests", "they catch regressions"
    )
    # BOS comes from the template exactly once.
    assert prompt_ids.count(chat_tokenizer.bos_token_id) == 1
    assert prompt_ids[0] == chat_tokenizer.bos_token_id
    assert completion_ids[-1] == _token_id(chat_tokenizer, "<|eot_id|>")


def test_sft_labels_cover_completion_and_eot_and_drop_truncated_prompts():
    pytest.importorskip("datasets")
    from training import sft_trainer

    preprocessor = sft_trainer.SFTDataPreprocessor(build_tiny_tokenizer(), max_length=24)
    tokenizer = preprocessor.tokenizer
    prompt_ids, completion_ids = chat_format.encode_prompt_and_completion(
        tokenizer, "explain tests", "they catch regressions"
    )

    batch = preprocessor.tokenize_function(
        {
            "prompt": ["explain tests", " ".join(["hi"] * 40)],
            "completion": ["they catch regressions", "good"],
        }
    )

    assert len(batch["input_ids"]) == 1  # the second prompt fills max_length
    input_ids, labels, mask = batch["input_ids"][0], batch["labels"][0], batch["attention_mask"][0]
    length = len(prompt_ids) + len(completion_ids)
    assert input_ids[:length] == prompt_ids + completion_ids
    assert labels[:len(prompt_ids)] == [-100] * len(prompt_ids)
    assert labels[len(prompt_ids):length] == completion_ids
    assert labels[length - 1] == _token_id(tokenizer, "<|eot_id|>")
    assert input_ids[length:] == [tokenizer.pad_token_id] * (24 - length)
    assert labels[length:] == [-100] * (24 - length)
    assert mask == [1] * length + [0] * (24 - length)


def test_reward_pairs_end_on_eot_and_drop_overlong():
    pytest.importorskip("datasets")
    from training import reward_trainer

    preprocessor = reward_trainer.RewardDataPreprocessor(build_tiny_tokenizer(), max_length=24)
    tokenizer = preprocessor.tokenizer

    batch = preprocessor.create_comparison_pairs(
        {
            "prompt": ["answer question", "answer question"],
            "chosen": ["good answer", "good"],
            "rejected": ["bad answer", " ".join(["bad"] * 40)],
        }
    )

    assert len(batch["chosen_input_ids"]) == 1
    eot = _token_id(tokenizer, "<|eot_id|>")
    for side in ("chosen", "rejected"):
        ids, mask = batch[f"{side}_input_ids"][0], batch[f"{side}_attention_mask"][0]
        last = sum(mask) - 1
        assert ids[last] == eot
        assert all(token == tokenizer.pad_token_id for token in ids[last + 1:])


def test_collate_prompts_left_pads():
    from training.rlhf_trainer import collate_prompts

    batch = collate_prompts(
        [{"input_ids": [5, 6, 7], "prompt_text": "a"}, {"input_ids": [8], "prompt_text": "b"}],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[5, 6, 7], [0, 0, 8]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]


def test_response_mask_keeps_first_stop_token_and_drops_the_rest():
    from training.rlhf_trainer import position_ids_from_mask, response_mask_from_stops

    response_ids = torch.tensor([[11, 12, 99, 0, 0], [11, 12, 13, 14, 15], [99, 12, 99, 0, 0]])

    mask = response_mask_from_stops(response_ids, stop_ids=[99])

    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]
    assert position_ids_from_mask(torch.tensor([[0, 0, 1, 1, 1]])).tolist() == [[0, 0, 0, 1, 2]]


def _ppo_trainer(tokenizer, temperature=0.7):
    from training import rlhf_trainer

    trainer = rlhf_trainer.RLHFTrainer.__new__(rlhf_trainer.RLHFTrainer)
    trainer.config = rlhf_trainer.RLHFConfig(
        policy_model_path="p",
        reward_model_path="r",
        output_dir="o",
        temperature=temperature,
        max_response_length=4,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = tokenizer
    trainer.stop_token_ids = chat_format.stop_token_ids(tokenizer)
    trainer.policy_model = build_tiny_llama(len(tokenizer), seed=0)
    trainer.ref_model = build_tiny_llama(len(tokenizer), seed=1)
    trainer.value_head = rlhf_trainer.ValueHead(trainer.policy_model.config.hidden_size)
    return trainer


def test_ppo_log_probs_do_not_depend_on_left_padding(chat_tokenizer):
    trainer = _ppo_trainer(chat_tokenizer)
    pad = chat_tokenizer.pad_token_id
    response_ids = torch.tensor([[5, 6, 7]])
    unpadded = torch.tensor([[1, 10, 11, 12] + [5, 6, 7]])
    padded = torch.tensor([[pad, pad, 1, 10, 11, 12] + [5, 6, 7]])
    padded_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1]])

    with torch.no_grad():
        expected, _, expected_values = trainer.compute_log_probs_and_entropy_per_token(
            unpadded, response_ids, torch.ones_like(unpadded)
        )
        actual, _, actual_values = trainer.compute_log_probs_and_entropy_per_token(
            padded, response_ids, padded_mask
        )
        ref_actual = trainer.compute_ref_log_probs(padded, response_ids, padded_mask)
        ref_expected = trainer.compute_ref_log_probs(unpadded, response_ids, torch.ones_like(unpadded))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(ref_actual, ref_expected)


def test_ppo_log_probs_use_sampling_temperature(chat_tokenizer):
    trainer = _ppo_trainer(chat_tokenizer, temperature=0.5)
    input_ids = torch.tensor([[1, 10, 11, 5, 6]])
    response_ids = input_ids[:, 3:]

    with torch.no_grad():
        log_probs, _, _ = trainer.compute_log_probs_and_entropy_per_token(input_ids, response_ids)
        logits = trainer.policy_model(input_ids).logits[:, 2:4, :].float() / 0.5
    expected = torch.log_softmax(logits, dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(log_probs, expected)


def test_ppo_generate_masks_after_first_stop_token(chat_tokenizer):
    trainer = _ppo_trainer(chat_tokenizer)
    pad = chat_tokenizer.pad_token_id
    prompt_ids = torch.tensor([[pad, 1, 10], [1, 10, 11]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

    response_ids, response_mask = trainer.generate_response(prompt_ids, attention_mask)

    expected = chat_format_mask(response_ids, trainer.stop_token_ids)
    assert response_mask.tolist() == expected
    assert response_ids.shape[1] <= trainer.config.max_response_length


def chat_format_mask(response_ids, stop_ids):
    rows = []
    for row in response_ids.tolist():
        mask, stopped = [], False
        for token in row:
            mask.append(0 if stopped else 1)
            stopped = stopped or token in stop_ids
        rows.append(mask)
    return rows


def _install_fake_vllm(monkeypatch):
    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    arg_utils_module = types.ModuleType("vllm.engine.arg_utils")
    arg_utils_module.AsyncEngineArgs = lambda **kwargs: kwargs
    async_engine_module = types.ModuleType("vllm.engine.async_llm_engine")
    async_engine_module.AsyncLLMEngine = object
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", types.ModuleType("vllm.engine"))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_module)
    monkeypatch.setitem(sys.modules, "vllm.engine.async_llm_engine", async_engine_module)


def test_vllm_server_uses_checkpoint_chat_template_and_eot_stop(monkeypatch, tmp_path, chat_tokenizer):
    _install_fake_vllm(monkeypatch)
    module = importlib.import_module("src.serving.vllm_server")
    chat_tokenizer.save_pretrained(tmp_path)
    server = module.VLLMServer(module.VLLMServerConfig(model_name=str(tmp_path), enable_metrics=False))

    server._load_chat_tokenizer()
    messages = [module.ChatMessage(role="user", content="explain tests")]
    prompt, prompt_tokens = server._build_chat_prompt(messages)

    expected = chat_format.encode(chat_tokenizer, chat_format.format_prompt(chat_tokenizer, "explain tests"))
    assert prompt == {"prompt_token_ids": expected}
    assert prompt_tokens == len(expected)
    assert server.stop_token_ids == [_token_id(chat_tokenizer, "<|eot_id|>")]

    captured = {}

    async def fake_generate(prompt, sampling_params, request_id=None):
        captured["prompt"] = prompt
        captured["sampling_params"] = sampling_params
        return "they catch regressions"

    server.engine.generate = fake_generate
    route = next(r for r in server.app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    request = module.ChatCompletionRequest(messages=[{"role": "user", "content": "explain tests"}])
    response = asyncio.run(route.endpoint(request))

    assert captured["prompt"] == {"prompt_token_ids": expected}
    assert captured["sampling_params"].stop_token_ids == server.stop_token_ids
    assert response.status_code == 200
