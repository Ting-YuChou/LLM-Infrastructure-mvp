import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from training.mlflow_lifecycle import (  # noqa: E402
    MLflowTrainingLifecycle,
    PromotionConfig,
)


class FakeRunContext:
    def __init__(self, run_id="run-1"):
        self.run = SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def __enter__(self):
        return self.run

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeArtifacts:
    def __init__(self):
        self.downloads = []

    def download_artifacts(self, artifact_uri, dst_path=None):
        self.downloads.append((artifact_uri, dst_path))
        raise FileNotFoundError(artifact_uri)


class FakeMLflow:
    def __init__(self):
        self.artifacts = FakeArtifacts()
        self.metrics = []
        self.params = {}
        self.tags = {}
        self.logged_artifacts = []
        self.logged_dicts = {}
        self.logged_text = {}
        self.tracking_uri = None

    def set_tracking_uri(self, tracking_uri):
        self.tracking_uri = tracking_uri

    def get_experiment_by_name(self, name):
        return None

    def create_experiment(self, name, tags=None):
        return "exp-1"

    def start_run(self, experiment_id=None, run_name=None):
        return FakeRunContext()

    def set_tags(self, tags):
        self.tags.update(tags)

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_params(self, params):
        self.params.update(params)

    def log_metric(self, key, value, step=None):
        self.metrics.append((key, value, step))

    def log_artifact(self, path, artifact_path=None):
        self.logged_artifacts.append(("file", path, artifact_path))

    def log_artifacts(self, path, artifact_path=None):
        self.logged_artifacts.append(("dir", path, artifact_path))

    def log_dict(self, data, artifact_file):
        self.logged_dicts[artifact_file] = data

    def log_text(self, text, artifact_file):
        self.logged_text[artifact_file] = text


class FakeClient:
    def __init__(self):
        self.registered_models = set()
        self.aliases = {}
        self.version_tags = {}
        self.model_versions = {}
        self.created_versions = []

    def get_registered_model(self, name):
        if name not in self.registered_models:
            raise KeyError(name)
        return SimpleNamespace(name=name)

    def create_registered_model(self, name, tags=None):
        self.registered_models.add(name)
        return SimpleNamespace(name=name)

    def create_model_version(self, name, source, run_id, tags=None):
        version = str(len(self.created_versions) + 1)
        mv = SimpleNamespace(name=name, version=version, source=source, run_id=run_id)
        self.created_versions.append(mv)
        self.model_versions[(name, version)] = mv
        return mv

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[(name, alias)] = str(version)

    def get_model_version_by_alias(self, name, alias):
        version = self.aliases.get((name, alias))
        if version is None:
            raise KeyError(alias)
        return self.model_versions[(name, version)]

    def set_model_version_tag(self, name, version, key, value):
        self.version_tags[(name, str(version), key)] = value

    def update_model_version(self, name, version, description):
        self.version_tags[(name, str(version), "description")] = description


def raw_config(enabled=True):
    return {
        "tracking": {
            "mlflow_enabled": enabled,
            "mlflow_tracking_uri": "http://mlflow:5000",
            "mlflow_experiment_name": "sft-training",
            "wandb_run_name": "demo-run",
        },
        "registry": {
            "auto_register": True,
            "model_name": "demo-model",
            "tags": {"task": "instruction-following"},
        },
        "model": {"name": "base-model"},
        "training": {"learning_rate": 1e-5},
    }


def make_checkpoint(tmp_path):
    checkpoint = tmp_path / "final"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    return checkpoint


def make_lifecycle(raw, promotion=None):
    return MLflowTrainingLifecycle.from_training_config(
        raw,
        model_type="sft",
        promotion=promotion,
        mlflow_module=FakeMLflow(),
        client=FakeClient(),
    )


def test_lifecycle_config_parsing_uses_tracking_and_registry_blocks():
    lifecycle = make_lifecycle(raw_config())

    assert lifecycle.config.enabled
    assert lifecycle.config.tracking_uri == "http://mlflow:5000"
    assert lifecycle.config.experiment_name == "sft-training"
    assert lifecycle.config.model_name == "demo-model"
    assert lifecycle.config.tags["task"] == "instruction-following"


def test_disabled_lifecycle_is_noop_without_checkpoint_or_mlflow():
    lifecycle = make_lifecycle(raw_config(enabled=False))

    with lifecycle:
        lifecycle.log_metrics({"loss": 1.2}, step=1)
        result = lifecycle.register_checkpoint("/path/does/not/exist")

    assert result is None


def test_register_checkpoint_sets_candidate_pending_without_promotion_metrics(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    lifecycle = make_lifecycle(raw_config())

    with lifecycle:
        result = lifecycle.register_checkpoint(str(checkpoint), final_metrics={"loss": 0.4})

    assert result.validation_status == "pending"
    assert result.promoted is False
    assert lifecycle.client.aliases[("demo-model", "candidate")] == "1"
    assert ("final.loss", 0.4, None) in lifecycle._mlflow.metrics


def test_passing_gate_promotes_candidate_to_champion(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"win_rate": 0.7}), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"thresholds": ["win_rate>=0.6"]}), encoding="utf-8")
    lifecycle = make_lifecycle(
        raw_config(),
        PromotionConfig(metrics_path=str(metrics), gate_config_path=str(gate)),
    )

    with lifecycle:
        result = lifecycle.register_checkpoint(str(checkpoint))

    assert result.validation_status == "passed"
    assert result.promoted is True
    assert lifecycle.client.aliases[("demo-model", "candidate")] == "1"
    assert lifecycle.client.aliases[("demo-model", "champion")] == "1"
    assert lifecycle.client.version_tags[("demo-model", "1", "validation_status")] == "passed"


def test_failing_gate_keeps_previous_champion_unchanged(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"win_rate": 0.4}), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"thresholds": ["win_rate>=0.6"]}), encoding="utf-8")
    lifecycle = make_lifecycle(
        raw_config(),
        PromotionConfig(metrics_path=str(metrics), gate_config_path=str(gate)),
    )

    with lifecycle:
        result = lifecycle.register_checkpoint(str(checkpoint))

    assert result.validation_status == "failed"
    assert result.promoted is False
    assert lifecycle.client.aliases[("demo-model", "candidate")] == "1"
    assert ("demo-model", "champion") not in lifecycle.client.aliases


def test_delta_gate_without_baseline_fails_closed(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"win_rate": 0.7}), encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"min_delta": {"win_rate": 0.0}}), encoding="utf-8")
    lifecycle = make_lifecycle(
        raw_config(),
        PromotionConfig(metrics_path=str(metrics), gate_config_path=str(gate)),
    )

    with lifecycle:
        result = lifecycle.register_checkpoint(str(checkpoint))

    assert result.validation_status == "failed"
    assert result.promoted is False
    assert "baseline metrics are required" in result.message
