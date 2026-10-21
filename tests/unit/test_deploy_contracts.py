import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.deploy import DeploymentManager  # noqa: E402


def test_raw_vllm_manifest_init_container_reads_configmap_env():
    docs = [
        doc
        for doc in yaml.safe_load_all(Path("k8s/deployments/vllm-deployment.yaml").read_text())
        if doc
    ]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    init_container = deployment["spec"]["template"]["spec"]["initContainers"][0]

    assert init_container["envFrom"] == [{"configMapRef": {"name": "vllm-config"}}]
    script = "\n".join(init_container["command"])
    assert "PVC must be preloaded with AWQ weights at MODEL_NAME" in script


def test_helm_values_do_not_use_production_wildcards():
    for values_path in (
        Path("charts/vllm-inference/values.yaml"),
        Path("charts/vllm-inference/values-staging.yaml"),
        Path("charts/vllm-inference/values-production.yaml"),
    ):
        values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
        gateway_config = values["gateway"]["config"]
        assert gateway_config["environment"] == "production"
        assert gateway_config["corsOrigins"] != "*"
        assert gateway_config["allowedHosts"] != "*"


def test_helm_deploy_command_uses_atomic_wait_and_set_values():
    manager = DeploymentManager(dry_run=True)

    assert manager.deploy_helm_release(
        release="test-release",
        chart="charts/vllm-inference",
        values_files=["values.yaml"],
        set_values={"image.tag": "sha"},
    )

    command = manager.commands_executed[0]
    assert command[:4] == ["helm", "upgrade", "--install", "test-release"]
    assert "--atomic" in command
    assert "--wait" in command
    assert "--set-string" in command
    assert "image.tag=sha" in command


def test_blue_green_is_explicitly_unsupported():
    manager = DeploymentManager(dry_run=True)

    assert not manager.deploy_with_strategy(strategy="blue-green")


def test_canary_dry_run_generates_weight_override(tmp_path):
    values_file = tmp_path / "values-canary.yaml"
    values_file.write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "routing": {
                        "routes": {
                            "model-a": {
                                "targets": [
                                    {"name": "stable", "backend_url": "http://stable", "weight": 100},
                                    {"name": "canary", "backend_url": "http://canary", "weight": 0},
                                ]
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    manager = DeploymentManager(dry_run=True)

    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        assert manager.deploy_with_strategy(
            strategy="canary",
            deploy_mode="helm",
            release="vllm-inference",
            chart="charts/vllm-inference",
            values_files=[str(values_file)],
            canary_percentage=25,
        )

    output = json.loads(output_buffer.getvalue())
    assert output["canary_release"] == "vllm-inference-canary"
    assert output["canary_release_plan"]["deployment"] == "vllm-server-canary"
    assert output["canary_release_plan"]["service"] == "vllm-service-canary"
    targets = output["routing_override"]["gateway"]["routing"]["routes"]["model-a"]["targets"]
    weights = {target["name"]: target["weight"] for target in targets}
    assert weights == {"stable": 75, "canary": 25}


def test_canary_values_define_backend_release_only():
    values = yaml.safe_load(Path("charts/vllm-inference/values-canary.yaml").read_text())

    assert values["resourceNames"]["deployment"] == "vllm-server-canary"
    assert values["resourceNames"]["service"] == "vllm-service-canary"
    assert values["gateway"]["enabled"] is False


def test_production_values_enable_secret_tls_and_managed_redis():
    values = yaml.safe_load(Path("charts/vllm-inference/values-production.yaml").read_text())

    assert values["gateway"]["auth"]["externalSecret"]["enabled"] is True
    assert values["gateway"]["ingress"]["certManager"]["enabled"] is True
    assert values["gateway"]["ingress"]["tls"]
    assert values["redis"]["enabled"] is False
    assert values["redis"]["externalHost"]
    assert values["redis"]["auth"]["existingSecret"] == "gateway-auth"


def test_training_pipeline_job_values_are_disabled_by_default():
    values = yaml.safe_load(Path("charts/vllm-inference/values.yaml").read_text())
    training = values["trainingPipeline"]

    assert training["enabled"] is False
    assert training["resume"] is True
    assert training["stages"] == "all"
    assert training["outputPersistence"]["enabled"] is True
