"""scripts/prepare_data.py --source hf: UltraChat / UltraFeedback conversion and length stats."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import prepare_data  # noqa: E402


def _turn(role, content):
    return {"role": role, "content": content}


def _ultrachat(index, turns=1, trailing_user=False):
    messages = []
    for turn in range(turns):
        messages += [_turn("user", f"question {index}.{turn}"), _turn("assistant", f"answer {index}.{turn}")]
    if trailing_user:
        messages.append(_turn("user", "unanswered"))
    return {"prompt": f"question {index}.0", "prompt_id": f"c{index}", "messages": messages}


def _ultrafeedback(index, chosen="good", rejected="bad", score_chosen=8.0, score_rejected=4.0):
    prompt = f"prompt {index}"
    return {
        "prompt": prompt,
        "prompt_id": f"p{index}",
        "chosen": [_turn("user", prompt), _turn("assistant", f"{chosen} {index}")],
        "rejected": [_turn("user", prompt), _turn("assistant", f"{rejected} {index}")],
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
    }


def test_sft_conversation_keeps_complete_turns_and_rejects_bad_rows():
    row = _ultrachat(0, turns=2, trailing_user=True)

    assert prepare_data.sft_conversation(row) == row["messages"][:4]
    assert prepare_data.sft_conversation({"messages": [_turn("assistant", "hi")]}) is None
    assert prepare_data.sft_conversation({"messages": [_turn("user", "q"), _turn("assistant", " ")]}) is None
    assert prepare_data.sft_conversation({"messages": []}) is None


def test_preference_pair_skips_ties_identical_replies_and_multi_turn_rows():
    assert prepare_data.preference_pair(_ultrafeedback(0)) == {
        "prompt": "prompt 0", "chosen": "good 0", "rejected": "bad 0",
    }
    assert prepare_data.preference_pair(_ultrafeedback(1, score_chosen=5.0, score_rejected=5.0)) is None
    assert prepare_data.preference_pair(_ultrafeedback(2, chosen="same", rejected="same")) is None
    multi_turn = _ultrafeedback(3)
    multi_turn["chosen"] = [_turn("user", "a"), _turn("assistant", "b"), _turn("user", "c"), _turn("assistant", "d")]
    assert prepare_data.preference_pair(multi_turn) is None


def _fake_hub(rows_by_split):
    calls = []

    def load_split(name, split, seed):
        calls.append((name, split, seed))
        return list(rows_by_split[split])

    return load_split, calls


def test_build_hf_datasets_writes_trainer_files_and_keeps_ppo_prompts_out_of_rm_training(tmp_path):
    train_prefs = [_ultrafeedback(i) for i in range(30)]
    train_prefs.insert(3, _ultrafeedback(1))  # duplicate prompt
    train_prefs.insert(5, _ultrafeedback(99, score_chosen=3.0, score_rejected=3.0))  # tie
    load_split, calls = _fake_hub({
        "train_sft": [_ultrachat(i, turns=1 + i % 3) for i in range(30)],
        "test_sft": [_ultrachat(100 + i) for i in range(20)],
        "train_prefs": train_prefs,
        "test_prefs": [_ultrafeedback(200 + i) for i in range(20)],
    })

    summary = prepare_data.build_hf_datasets(
        str(tmp_path), sft_examples=12, preference_examples=10, rl_prompts=8, seed=3, load_split=load_split
    )

    assert {seed for _, _, seed in calls} == {3}
    assert summary["counts"] == {
        "sft_train": 12, "sft_val": 10, "preference_train": 10, "preference_val": 10, "rl_prompts": 8,
    }
    sft = json.loads((tmp_path / "sft" / "train.json").read_text())
    assert all(row["messages"][-1]["role"] == "assistant" for row in sft)
    assert max(len(row["messages"]) for row in sft) == 6
    pairs = json.loads((tmp_path / "preferences" / "train.json").read_text())
    assert set(pairs[0]) == {"id", "prompt", "chosen", "rejected"}
    assert len({row["prompt"] for row in pairs}) == 10
    prompts = json.loads((tmp_path / "prompts" / "rl_prompts.json").read_text())
    # The tie was skipped as a pair, so PPO may use its prompt; RM training prompts never.
    assert not {row["prompt"] for row in prompts} & {row["prompt"] for row in pairs}
    assert "prompt 99" in {row["prompt"] for row in prompts}
    val = json.loads((tmp_path / "preferences" / "val.json").read_text())
    assert all(row["prompt"].startswith("prompt 2") for row in val)


def test_length_summary_reports_percentiles_and_fraction_over_each_limit():
    summary = prepare_data.length_summary(list(range(1, 101)), (50, 90))

    assert (summary["p50"], summary["p90"], summary["max"]) == (51, 91, 100)
    assert summary["over"] == {"50": 0.5, "90": 0.1}


def test_data_stats_and_trainers_read_the_written_files(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from tiny_llama import build_tiny_tokenizer, real_transformers_available

    if not real_transformers_available():
        pytest.skip("needs real transformers and tokenizers")
    sys.path.insert(0, str(ROOT / "src"))
    from training.reward_trainer import RewardDataPreprocessor
    from training.sft_trainer import SFTDataPreprocessor

    load_split, _ = _fake_hub({
        "train_sft": [_ultrachat(i, turns=1 + i % 3) for i in range(12)],
        "test_sft": [_ultrachat(100 + i) for i in range(12)],
        "train_prefs": [_ultrafeedback(i) for i in range(24)],
        "test_prefs": [_ultrafeedback(200 + i) for i in range(12)],
    })
    prepare_data.build_hf_datasets(str(tmp_path), 12, 12, 10, load_split=load_split)
    limits = {"sft_max_length": 40, "reward_max_length": 1024, "dpo_max_length": 1024, "rl_max_prompt_length": 512}

    stats = prepare_data.data_stats(str(tmp_path), build_tiny_tokenizer(), limits)

    sft = stats["sft_conversation_tokens"]
    assert sft["count"] == 12 and sft["turns"]["max"] == 3
    assert 0 < sft["over"]["40"] < 1
    assert stats["preference_pair_tokens"]["over"]["1024"] == 0.0
    assert stats["rl_prompt_tokens"]["count"] == 10

    # The trainers' own preprocessors take the written rows, and agree with the stats.
    def columns(path):
        rows = json.loads((tmp_path / path).read_text())
        return {key: [row[key] for row in rows] for key in rows[0]}

    tokenizer = build_tiny_tokenizer()
    sft_rows = columns("sft/train.json")
    batch = SFTDataPreprocessor(tokenizer, max_length=40).tokenize_function(sft_rows)
    assert len(batch["input_ids"]) == 12
    lengths = [len(ids) for ids in batch["input_ids"]]
    assert max(lengths) <= 40
    assert sum(length < full for length, full in zip(lengths, _full_lengths(tokenizer, sft_rows))) == round(
        sft["over"]["40"] * 12
    )
    pairs = RewardDataPreprocessor(tokenizer, max_length=1024).create_comparison_pairs(columns("preferences/train.json"))
    assert len(pairs["chosen_input_ids"]) == 12


def _full_lengths(tokenizer, rows):
    from training.chat_format import encode_conversation, ensure_chat_tokenizer

    tokenizer = ensure_chat_tokenizer(tokenizer)
    return [len(encode_conversation(tokenizer, messages).input_ids) for messages in rows["messages"]]
