import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training.pipeline import TrainingPipeline, validate_checkpoint_artifact  # noqa: E402


def test_pipeline_dry_run_builds_deterministic_plan():
    pipeline = TrainingPipeline(
        config_path="config/training_pipeline.yaml",
        stages="all",
        dry_run=True,
    )

    result = pipeline.run()
    plan = result["plan"]

    assert result["status"] == "dry_run"
    assert plan["stages"] == [
        "sft",
        "reward",
        "rlhf",
        "training_eval",
        "mlflow_promotion",
        "serving_handoff",
    ]
    assert plan["configs"]["sft"] == "config/sft_config.yaml"
    assert plan["path_checks"]["sft_config_exists"] is True
    assert plan["path_checks"]["reward_config_exists"] is True
    assert plan["path_checks"]["rlhf_config_exists"] is True
    assert plan["path_checks"]["gate_config_exists"] is True
    assert plan["outputs"]["metrics"] == "outputs/training_pipeline/eval/training_candidate.json"
    assert plan["injections"]["reward.model.name"] == "${sft.checkpoint_path}"
    assert plan["serving_handoff"]["quantization"] == "awq"


def test_pipeline_output_dir_override_rebases_stage_outputs(tmp_path):
    output_dir = tmp_path / "pipeline"

    pipeline = TrainingPipeline(
        config_path="config/training_pipeline.yaml",
        stages="sft,handoff",
        output_dir=str(output_dir),
        dry_run=True,
    )
    plan = pipeline.build_execution_plan()

    assert plan["outputs"]["root"] == str(output_dir)
    assert plan["outputs"]["sft"] == str(output_dir / "models" / "sft")
    assert plan["outputs"]["reward"] == str(output_dir / "models" / "reward")
    assert plan["outputs"]["rlhf"] == str(output_dir / "models" / "aligned")
    assert plan["outputs"]["serving_handoff"] == str(output_dir / "serving_handoff.json")


def test_checkpoint_validation_reports_missing_and_ready_artifacts(tmp_path):
    missing = tmp_path / "missing"
    missing_validation = validate_checkpoint_artifact(missing)

    assert missing_validation["exists"] is False
    assert missing_validation["ready_for_awq_conversion"] is False

    incomplete = tmp_path / "checkpoint-with-value-head-only"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    (incomplete / "tokenizer.json").write_text("{}", encoding="utf-8")
    (incomplete / "value_head.pt").write_text("value", encoding="utf-8")

    incomplete_validation = validate_checkpoint_artifact(incomplete)

    assert incomplete_validation["weights"] is False
    assert incomplete_validation["ready_for_awq_conversion"] is False

    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_text("weights", encoding="utf-8")

    ready_validation = validate_checkpoint_artifact(checkpoint)

    assert ready_validation["exists"] is True
    assert ready_validation["config_json"] is True
    assert ready_validation["tokenizer"] is True
    assert ready_validation["weights"] is True
    assert ready_validation["ready_for_awq_conversion"] is True


def test_serving_handoff_manifest_contains_awq_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint-final"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_text("weights", encoding="utf-8")

    pipeline = TrainingPipeline(
        config_path="config/training_pipeline.yaml",
        output_dir=str(tmp_path / "pipeline"),
        dry_run=True,
    )
    handoff = pipeline.write_serving_handoff(str(checkpoint))
    handoff_path = tmp_path / "pipeline" / "serving_handoff.json"

    assert handoff_path.exists()
    written = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert written == handoff
    assert handoff["final_aligned_checkpoint"] == str(checkpoint)
    assert handoff["expected_awq_target_path"] == "/workspace/models/awq/checkpoint-final"
    assert handoff["quantization"] == "awq"
    assert handoff["artifact_validation"]["ready_for_awq_conversion"] is True
