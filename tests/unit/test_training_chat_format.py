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
    # Rows are unpadded; PaddingCollator pads each batch.
    assert input_ids == prompt_ids + completion_ids
    assert labels == [-100] * len(prompt_ids) + completion_ids
    assert labels[-1] == _token_id(tokenizer, "<|eot_id|>")
    assert mask == [1] * len(input_ids)


CONVERSATION = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "explain tests"},
    {"role": "assistant", "content": " they catch regressions "},
    {"role": "user", "content": "and more"},
    {"role": "assistant", "content": "they document code"},
]


def test_conversation_ids_match_the_full_template_and_label_every_reply(chat_tokenizer):
    encoded = chat_format.encode_conversation(chat_tokenizer, CONVERSATION)

    full = chat_tokenizer.apply_chat_template(CONVERSATION, tokenize=False)
    assert encoded.input_ids == chat_format.encode(chat_tokenizer, full)
    assert encoded.input_ids.count(chat_tokenizer.bos_token_id) == 1
    # Exactly the replies are labelled, as the same ids RM and PPO use for a response.
    replies = [
        chat_format.encode(chat_tokenizer, chat_format.format_completion(text))
        for text in (" they catch regressions ", "they document code")
    ]
    assert [label for label in encoded.labels if label != -100] == replies[0] + replies[1]
    assert encoded.turn_ends[-1] == len(encoded.input_ids)
    first_reply_end = encoded.turn_ends[0]
    assert encoded.labels[first_reply_end - len(replies[0]):first_reply_end] == replies[0]


def test_single_turn_conversation_matches_prompt_and_completion(chat_tokenizer):
    prompt_ids, completion_ids = chat_format.encode_prompt_and_completion(chat_tokenizer, "explain tests", "ok")

    encoded = chat_format.encode_conversation(
        chat_tokenizer,
        [{"role": "user", "content": "explain tests"}, {"role": "assistant", "content": "ok"}],
    )

    assert encoded.input_ids == prompt_ids + completion_ids
    assert encoded.labels == [-100] * len(prompt_ids) + completion_ids


@pytest.mark.parametrize(
    "roles",
    [
        ["assistant", "user"],
        ["user", "user", "assistant"],
        ["user", "system", "assistant"],
        ["user"],
        [],
    ],
)
def test_conversations_with_bad_role_order_are_rejected(chat_tokenizer, roles):
    messages = [{"role": role, "content": "x"} for role in roles]

    with pytest.raises(ValueError, match="roles"):
        chat_format.encode_conversation(chat_tokenizer, messages)


def test_prompt_can_be_a_conversation_ending_with_user(chat_tokenizer):
    context = CONVERSATION[:4]

    assert chat_format.format_prompt(chat_tokenizer, context) == chat_tokenizer.apply_chat_template(
        context, tokenize=False, add_generation_prompt=True
    )
    with pytest.raises(ValueError, match="roles"):
        chat_format.format_prompt(chat_tokenizer, CONVERSATION)


def test_sft_cuts_long_conversations_after_the_last_reply_that_fits():
    pytest.importorskip("datasets")
    from training import sft_trainer

    tokenizer = chat_format.ensure_chat_tokenizer(build_tiny_tokenizer())
    encoded = chat_format.encode_conversation(tokenizer, CONVERSATION)
    first_end = encoded.turn_ends[0]
    preprocessor = sft_trainer.SFTDataPreprocessor(tokenizer, max_length=len(encoded.input_ids) - 1)

    batch = preprocessor.tokenize_function({"messages": [CONVERSATION]})

    assert batch["input_ids"] == [encoded.input_ids[:first_end]]
    assert batch["labels"] == [encoded.labels[:first_end]]


def test_padding_collator_pads_every_sequence_to_one_length():
    collate = chat_format.PaddingCollator(pad_token_id=7, pad_to_multiple_of=4)

    batch = collate(
        [
            {"chosen_input_ids": [1, 2, 3], "chosen_attention_mask": [1, 1, 1], "chosen_labels": [-100, 2, 3],
             "rejected_input_ids": [1, 2], "rejected_attention_mask": [1, 1], "rejected_labels": [-100, 2],
             "ref_chosen_logps": -1.5},
            {"chosen_input_ids": [1, 2, 3, 4, 5], "chosen_attention_mask": [1] * 5, "chosen_labels": [-100] + [2] * 4,
             "rejected_input_ids": [1], "rejected_attention_mask": [1], "rejected_labels": [-100],
             "ref_chosen_logps": -2.0},
        ]
    )

    assert {tuple(value.shape) for key, value in batch.items() if key != "ref_chosen_logps"} == {(2, 8)}
    assert batch["chosen_input_ids"][0].tolist() == [1, 2, 3] + [7] * 5
    assert batch["chosen_attention_mask"][0].tolist() == [1, 1, 1] + [0] * 5
    assert batch["rejected_labels"][1].tolist() == [-100] * 8
    assert batch["ref_chosen_logps"].tolist() == [-1.5, -2.0]


def test_verl_reward_model_input_matches_reward_training_tokens(chat_tokenizer):
    """verl_ppo scores rollouts with our RM through verl's reward loop; it must see training's tokens."""
    prompt, response = "answer question", "good answer"
    prompt_ids, response_ids = chat_format.encode_prompt_and_completion(chat_tokenizer, prompt, response)

    # verl v0.9.1 reward_loop._preprocess_reward_inputs: decode the rollout, drop
    # the eos text, re-apply the RM tokenizer's template, strip the leading BOS
    # text; the vLLM classify server then tokenizes and adds BOS back.
    rollout = chat_tokenizer.decode(response_ids).replace(chat_tokenizer.eos_token, "")
    rm_prompt = chat_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": rollout}],
        add_generation_prompt=False,
        tokenize=False,
    )
    bos = chat_tokenizer.bos_token
    assert rm_prompt.startswith(bos)
    served = [chat_tokenizer.bos_token_id] + chat_format.encode(chat_tokenizer, rm_prompt[len(bos):])

    assert served == prompt_ids + response_ids


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
        # Unpadded: the pair ends on <|eot_id|>; PaddingCollator pads each batch.
        assert ids[-1] == eot
        assert mask == [1] * len(ids)


def test_reward_pairs_accept_a_conversation_prompt():
    pytest.importorskip("datasets")
    from training import reward_trainer

    preprocessor = reward_trainer.RewardDataPreprocessor(build_tiny_tokenizer(), max_length=128)
    tokenizer = preprocessor.tokenizer
    context = CONVERSATION[:4]

    batch = preprocessor.create_comparison_pairs(
        {"prompt": [context], "chosen": ["good answer"], "rejected": ["bad"]}
    )

    prompt_ids = chat_format.encode(tokenizer, chat_format.format_prompt(tokenizer, context))
    assert batch["chosen_input_ids"][0] == prompt_ids + chat_format.encode(
        tokenizer, chat_format.format_completion("good answer")
    )
    assert batch["rejected_input_ids"][0][:len(prompt_ids)] == prompt_ids


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
