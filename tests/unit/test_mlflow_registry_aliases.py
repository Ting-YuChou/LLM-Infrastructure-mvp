import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from registry.mlflow_client import LLMModelRegistry  # noqa: E402


class FakeArtifacts:
    def __init__(self):
        self.calls = []

    def download_artifacts(self, artifact_uri, dst_path=None):
        self.calls.append((artifact_uri, dst_path))
        return "/tmp/downloaded-model"


class FakeMLflow:
    def __init__(self):
        self.artifacts = FakeArtifacts()


class FakeClient:
    def __init__(self):
        self.aliases = {}
        self.tags = {}
        self.versions = {
            ("demo-model", "1"): SimpleNamespace(
                name="demo-model",
                version="1",
                source="runs:/old/model",
                run_id="run-old",
                current_stage="None",
            ),
            ("demo-model", "2"): SimpleNamespace(
                name="demo-model",
                version="2",
                source="runs:/new/model",
                run_id="run-new",
                current_stage="None",
            ),
        }

    def set_registered_model_alias(self, model_name, alias, version):
        self.aliases[(model_name, alias)] = str(version)

    def get_model_version_by_alias(self, model_name, alias):
        version = self.aliases[(model_name, alias)]
        return self.versions[(model_name, version)]

    def set_model_version_tag(self, model_name, version, key, value):
        self.tags[(model_name, str(version), key)] = str(value)


def make_registry():
    registry = LLMModelRegistry.__new__(LLMModelRegistry)
    registry.client = FakeClient()
    registry.mlflow = FakeMLflow()
    return registry


def test_set_and_get_model_alias():
    registry = make_registry()

    registry.set_model_alias("demo-model", "candidate", "2")
    version = registry.get_model_version_by_alias("demo-model", "candidate")

    assert version.version == "2"


def test_promote_to_champion_archives_previous_alias_target():
    registry = make_registry()
    registry.set_model_alias("demo-model", "champion", "1")

    registry.promote_to_champion("demo-model", "2")

    assert registry.client.aliases[("demo-model", "champion")] == "2"
    assert registry.client.tags[("demo-model", "1", "lifecycle_alias")] == "archived"
    assert registry.client.tags[("demo-model", "1", "replaced_by_version")] == "2"
    assert registry.client.tags[("demo-model", "2", "lifecycle_alias")] == "champion"
    assert registry.client.tags[("demo-model", "2", "validation_status")] == "passed"


def test_load_model_defaults_to_champion_alias_and_downloads_source():
    registry = make_registry()
    registry.set_model_alias("demo-model", "champion", "2")

    local_path = registry.load_model("demo-model")

    assert local_path == "/tmp/downloaded-model"
    assert registry.mlflow.artifacts.calls == [
        ("runs:/new/model", "/tmp/mlflow_models/demo-model/2")
    ]
