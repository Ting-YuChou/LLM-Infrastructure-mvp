#!/usr/bin/env python3
"""
Deployment Automation Script
Handles complete deployment workflow with safety checks and rollback
"""

import os
import sys
import time
import subprocess
import argparse
import logging
import tempfile
from typing import List, Dict, Optional
from datetime import datetime
import yaml
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """
    Manages deployment workflow
    
    Features:
    - Pre-deployment validation
    - Rolling deployment
    - Helm routing-weight canary deployment
    - Health checks
    - Automatic rollback
    - Deployment history tracking
    """
    
    def __init__(
        self,
        namespace: str = "llm-inference",
        kubeconfig: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Initialize deployment manager
        
        Args:
            namespace: Kubernetes namespace
            kubeconfig: Path to kubeconfig file
        """
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.dry_run = dry_run
        self.commands_executed: List[List[str]] = []
        
        # Set kubeconfig if provided
        if kubeconfig:
            os.environ['KUBECONFIG'] = kubeconfig
        
        logger.info(f"Initialized deployment manager for namespace: {namespace}")
    
    def run_command(
        self,
        cmd: List[str],
        check: bool = True,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run shell command
        
        Args:
            cmd: Command and arguments
            check: Raise exception on failure
            capture_output: Capture stdout/stderr
            
        Returns:
            Completed process
        """
        logger.debug(f"Running: {' '.join(cmd)}")
        self.commands_executed.append(cmd)

        if self.dry_run:
            logger.info("[dry-run] %s", " ".join(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True
        )
        
        return result
    
    def validate_prerequisites(self, deploy_mode: str = "helm") -> bool:
        """
        Validate deployment prerequisites
        
        Returns:
            True if all checks pass
        """
        logger.info("Validating prerequisites...")
        
        checks = [
            ("kubectl", ["kubectl", "version", "--client"]),
            ("docker", ["docker", "--version"]),
            ("cluster access", ["kubectl", "cluster-info"]),
        ]

        if deploy_mode == "helm":
            checks.append(("helm", ["helm", "version", "--short"]))
        
        for name, cmd in checks:
            try:
                self.run_command(cmd)
                logger.info(f"  ✓ {name}")
            except subprocess.CalledProcessError:
                logger.error(f"  ✗ {name} - FAILED")
                return False
        
        return True
    
    def build_docker_images(self, tag: str = "latest") -> bool:
        """
        Build Docker images
        
        Args:
            tag: Image tag
            
        Returns:
            True if successful
        """
        logger.info(f"Building Docker images with tag: {tag}")
        
        images = ["base", "serving", "gateway"]
        
        for image in images:
            logger.info(f"Building {image}...")
            
            try:
                self.run_command([
                    "docker", "build",
                    "-f", f"docker/Dockerfile.{image}",
                    "-t", f"llm-{image}:{tag}",
                    "."
                ])
                logger.info(f"  ✓ Built llm-{image}:{tag}")
            
            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ Failed to build {image}: {e}")
                return False
        
        return True
    
    def push_docker_images(self, registry: str, tag: str = "latest") -> bool:
        """
        Push Docker images to registry
        
        Args:
            registry: Container registry URL
            tag: Image tag
            
        Returns:
            True if successful
        """
        logger.info(f"Pushing images to {registry}...")
        
        images = ["base", "serving", "gateway"]
        
        for image in images:
            local_tag = f"llm-{image}:{tag}"
            remote_tag = f"{registry}/llm-{image}:{tag}"
            
            logger.info(f"Tagging {local_tag} -> {remote_tag}")
            
            try:
                # Tag for registry
                self.run_command(["docker", "tag", local_tag, remote_tag])
                
                # Push to registry
                logger.info(f"Pushing {remote_tag}...")
                self.run_command(["docker", "push", remote_tag])
                
                logger.info(f"  ✓ Pushed {remote_tag}")
            
            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ Failed to push {image}: {e}")
                return False
        
        return True
    
    def deploy_manifest(self, manifest_path: str) -> bool:
        """
        Deploy Kubernetes manifest
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            True if successful
        """
        logger.info(f"Deploying manifest: {manifest_path}")
        
        try:
            self.run_command([
                "kubectl", "apply",
                "-f", manifest_path,
                "-n", self.namespace
            ])
            
            logger.info(f"  ✓ Deployed {manifest_path}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Deployment failed: {e}")
            return False

    def deploy_helm_release(
        self,
        release: str,
        chart: str,
        values_files: Optional[List[str]] = None,
        set_values: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Deploy the vLLM inference stack with Helm.

        Args:
            release: Helm release name
            chart: Path to the Helm chart
            values_files: Optional values files to pass with -f

        Returns:
            True if successful
        """
        logger.info(f"Deploying Helm release: {release} ({chart})")

        cmd = [
            "helm", "upgrade", "--install", release, chart,
            "--namespace", self.namespace,
            "--create-namespace",
            "--atomic",
            "--wait",
        ]

        for values_file in values_files or []:
            cmd.extend(["-f", values_file])
        for key, value in (set_values or {}).items():
            cmd.extend(["--set-string", f"{key}={value}"])

        try:
            self.run_command(cmd)
            logger.info(f"  ✓ Deployed Helm release {release}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Helm deployment failed: {e}")
            return False
    
    def wait_for_rollout(
        self,
        deployment: str,
        timeout: int = 300
    ) -> bool:
        """
        Wait for deployment rollout to complete
        
        Args:
            deployment: Deployment name
            timeout: Timeout in seconds
            
        Returns:
            True if successful
        """
        logger.info(f"Waiting for rollout: {deployment}")
        
        try:
            self.run_command([
                "kubectl", "rollout", "status",
                f"deployment/{deployment}",
                "-n", self.namespace,
                f"--timeout={timeout}s"
            ])
            
            logger.info(f"  ✓ Rollout complete")
            return True
        
        except subprocess.CalledProcessError:
            logger.error(f"  ✗ Rollout failed or timed out")
            return False
    
    def health_check(
        self,
        service: str,
        path: str = "/health",
        local_port: int = 18080,
        service_port: int = 80,
    ) -> bool:
        """
        Perform health check on service
        
        Args:
            service: Service name
            path: Health check path
            local_port: Local port for kubectl port-forward
            service_port: Service port to forward
            
        Returns:
            True if healthy
        """
        logger.info(f"Health check: {service}{path}")
        if self.dry_run:
            logger.info("[dry-run] health check skipped")
            return True
        
        # Port forward to service
        port_forward_proc = subprocess.Popen([
            "kubectl", "port-forward",
            f"svc/{service}",
            f"{local_port}:{service_port}",
            "-n", self.namespace
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for port forward to establish
        time.sleep(3)
        
        try:
            # Make health check request
            import requests
            response = requests.get(f"http://localhost:{local_port}{path}", timeout=10)
            
            if response.status_code == 200:
                logger.info("  ✓ Service is healthy")
                return True
            else:
                logger.error(f"  ✗ Unhealthy: HTTP {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"  ✗ Health check failed: {e}")
            return False
        
        finally:
            # Clean up port forward
            port_forward_proc.terminate()
            port_forward_proc.wait()
    
    def get_deployment_status(self, deployment: str) -> Dict:
        """
        Get deployment status
        
        Args:
            deployment: Deployment name
            
        Returns:
            Status information
        """
        try:
            result = self.run_command([
                "kubectl", "get", "deployment", deployment,
                "-n", self.namespace,
                "-o", "json"
            ])
            
            return json.loads(result.stdout)
        
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    def rollback_deployment(self, deployment: str) -> bool:
        """
        Rollback deployment to previous version
        
        Args:
            deployment: Deployment name
            
        Returns:
            True if successful
        """
        logger.warning(f"Rolling back deployment: {deployment}")
        
        try:
            self.run_command([
                "kubectl", "rollout", "undo",
                f"deployment/{deployment}",
                "-n", self.namespace
            ])
            
            # Wait for rollback to complete
            return self.wait_for_rollout(deployment)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def rollback_helm_release(self, release: str) -> bool:
        """
        Roll back a Helm release to the previous revision.

        Args:
            release: Helm release name

        Returns:
            True if successful
        """
        logger.warning(f"Rolling back Helm release: {release}")

        try:
            self.run_command([
                "helm", "rollback", release,
                "--namespace", self.namespace,
                "--wait",
            ])
            return self.wait_for_rollouts(["vllm-server", "api-gateway"])

        except subprocess.CalledProcessError as e:
            logger.error(f"Helm rollback failed: {e}")
            return False

    def wait_for_rollouts(self, deployments: List[str], timeout: int = 300) -> bool:
        """Wait for each named Deployment rollout to complete."""
        for deployment in deployments:
            if not self.wait_for_rollout(deployment, timeout=timeout):
                return False
        return True
    
    def deploy_with_strategy(
        self,
        strategy: str = "rolling",
        deploy_mode: str = "helm",
        manifest_dir: str = "k8s/deployments",
        release: str = "vllm-inference",
        chart: str = "charts/vllm-inference",
        values_files: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """
        Deploy with specified strategy
        
        Args:
            strategy: Deployment strategy (rolling, blue-green, canary)
            deploy_mode: Deployment backend (helm or manifest)
            manifest_dir: Directory with manifests
            release: Helm release name
            chart: Path to Helm chart
            values_files: Helm values files
            **kwargs: Strategy-specific parameters
            
        Returns:
            True if successful
        """
        logger.info(f"Deploying with {strategy} strategy via {deploy_mode}")
        
        if strategy == "rolling":
            return self._deploy_rolling(
                deploy_mode=deploy_mode,
                manifest_dir=manifest_dir,
                release=release,
                chart=chart,
                values_files=values_files,
            )
        elif strategy == "blue-green":
            return self._deploy_blue_green(manifest_dir)
        elif strategy == "canary":
            return self._deploy_canary(
                deploy_mode=deploy_mode,
                release=release,
                chart=chart,
                values_files=values_files,
                **kwargs,
            )
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return False
    
    def _deploy_rolling(
        self,
        deploy_mode: str,
        manifest_dir: str,
        release: str,
        chart: str,
        values_files: Optional[List[str]]
    ) -> bool:
        """Standard rolling deployment"""
        if deploy_mode == "helm":
            deployed = self.deploy_helm_release(release, chart, values_files)
        elif deploy_mode == "manifest":
            deployed = self.deploy_manifest(manifest_dir)
        else:
            logger.error(f"Unknown deploy mode: {deploy_mode}")
            return False

        if not deployed:
            return False
        
        # Wait for rollout
        deployments = ["vllm-server", "api-gateway"]
        if not self.wait_for_rollouts(deployments):
            logger.error("Rollout failed, initiating rollback...")
            self._rollback_for_mode(deploy_mode, release, deployments)
            return False
        
        # Health check the public serving entrypoint.
        if not self.health_check(
            "api-gateway",
            path="/ready",
            local_port=18080,
            service_port=8080,
        ):
            logger.error("Health check failed, initiating rollback...")
            self._rollback_for_mode(deploy_mode, release, deployments)
            return False
        
        logger.info("✓ Rolling deployment successful")
        return True
    
    def _rollback_for_mode(
        self,
        deploy_mode: str,
        release: str,
        deployments: List[str],
    ) -> None:
        """Roll back with the same backend used for deployment."""
        if deploy_mode == "helm":
            self.rollback_helm_release(release)
            return
        for deployment in deployments:
            self.rollback_deployment(deployment)

    def _deploy_blue_green(self, manifest_dir: str) -> bool:
        """Blue-green deployment"""
        logger.error(
            "Blue-green deployment is not supported. Use --strategy rolling or "
            "--strategy canary with --deploy-mode helm."
        )
        return False
    
    def _deploy_canary(
        self,
        deploy_mode: str,
        release: str,
        chart: str,
        values_files: Optional[List[str]],
        canary_percentage: int = 10,
        canary_release: str = "vllm-inference-canary",
        canary_values_files: Optional[List[str]] = None,
    ) -> bool:
        """Deploy a canary backend release, then shift gateway routing weights."""
        if deploy_mode != "helm":
            logger.error("Canary deployment is only supported with --deploy-mode helm")
            return False
        if canary_percentage < 0 or canary_percentage > 100:
            logger.error("Canary percentage must be between 0 and 100")
            return False

        logger.info(f"Canary deployment ({canary_percentage}% traffic)")
        try:
            override = self.build_canary_values_override(values_files or [], canary_percentage)
            canary_values = canary_values_files or [f"{chart}/values-canary.yaml"]
            canary_plan = self.build_canary_release_plan(canary_values)
        except ValueError as exc:
            logger.error(str(exc))
            return False

        if self.dry_run:
            print(json.dumps({
                "strategy": "canary",
                "deploy_mode": "helm",
                "release": release,
                "canary_release": canary_release,
                "chart": chart,
                "canary_percentage": canary_percentage,
                "canary_values_files": canary_values,
                "canary_release_plan": canary_plan,
                "routing_override": override,
            }, indent=2, sort_keys=True))
            return True

        deployed_canary = self.deploy_helm_release(
            canary_release,
            chart,
            values_files=[*(values_files or []), *canary_values],
        )
        if not deployed_canary:
            return False

        if not self.wait_for_rollout(canary_plan["deployment"]):
            logger.error("Canary backend rollout failed, initiating Helm rollback...")
            self.rollback_helm_release(canary_release)
            return False
        if not self.health_check(
            canary_plan["service"],
            path="/ready",
            local_port=18081,
            service_port=canary_plan["service_port"],
        ):
            logger.error("Canary backend health check failed, initiating Helm rollback...")
            self.rollback_helm_release(canary_release)
            return False

        with tempfile.NamedTemporaryFile("w", suffix="-canary-values.yaml", delete=False) as handle:
            yaml.safe_dump(override, handle, sort_keys=False)
            override_path = handle.name

        deployed = self.deploy_helm_release(
            release,
            chart,
            values_files=[*(values_files or []), override_path],
        )
        if not deployed:
            return False

        deployments = ["vllm-server", "api-gateway"]
        if not self.wait_for_rollouts(deployments):
            logger.error("Canary rollout failed, initiating Helm rollback...")
            self.rollback_helm_release(release)
            self.rollback_helm_release(canary_release)
            return False
        if not self.health_check("api-gateway", path="/ready", local_port=18080, service_port=8080):
            logger.error("Canary health check failed, initiating Helm rollback...")
            self.rollback_helm_release(release)
            self.rollback_helm_release(canary_release)
            return False
        logger.info("✓ Canary deployment successful")
        return True

    def build_canary_values_override(
        self,
        values_files: List[str],
        canary_percentage: int,
    ) -> Dict:
        """Build a Helm values override that adjusts stable/canary route weights."""
        merged: Dict = {}
        for values_file in values_files:
            with open(values_file, "r") as handle:
                merged = deep_merge(merged, yaml.safe_load(handle) or {})

        routes = (
            merged.get("gateway", {})
            .get("routing", {})
            .get("routes", {})
        )
        if not routes:
            raise ValueError("Canary deployment requires gateway.routing.routes in Helm values")

        updated_routes = json.loads(json.dumps(routes))
        found_canary = False
        for route in updated_routes.values():
            targets = route.get("targets") or []
            stable_target = next((target for target in targets if target.get("name") == "stable"), None)
            canary_target = next((target for target in targets if target.get("name") == "canary"), None)
            if not stable_target or not canary_target:
                continue
            stable_target["weight"] = 100 - canary_percentage
            canary_target["weight"] = canary_percentage
            found_canary = True

        if not found_canary:
            raise ValueError(
                "Canary deployment requires each canary route to define targets named "
                "'stable' and 'canary'"
            )

        return {"gateway": {"routing": {"routes": updated_routes}}}

    def build_canary_release_plan(self, values_files: List[str]) -> Dict[str, object]:
        """Return the expected resource names for the canary backend release."""
        merged: Dict = {}
        for values_file in values_files:
            with open(values_file, "r") as handle:
                merged = deep_merge(merged, yaml.safe_load(handle) or {})

        resource_names = merged.get("resourceNames", {})
        if not resource_names.get("deployment") or not resource_names.get("service"):
            raise ValueError("Canary values must override resourceNames.deployment and service")
        if merged.get("gateway", {}).get("enabled", True):
            raise ValueError("Canary backend release values must set gateway.enabled=false")

        service_port = (
            merged.get("service", {}).get("port")
            or 80
        )
        return {
            "deployment": resource_names["deployment"],
            "service": resource_names["service"],
            "service_port": int(service_port),
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LLM Infrastructure Deployment")
    
    parser.add_argument(
        "--action",
        choices=["validate", "build", "push", "deploy", "rollback"],
        required=True,
        help="Deployment action"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["rolling", "blue-green", "canary"],
        default="rolling",
        help="Deployment strategy"
    )

    parser.add_argument(
        "--deploy-mode",
        choices=["helm", "manifest"],
        default="helm",
        help="Deployment backend"
    )
    
    parser.add_argument(
        "--namespace",
        default="llm-inference",
        help="Kubernetes namespace"
    )
    
    parser.add_argument(
        "--registry",
        help="Container registry URL"
    )
    
    parser.add_argument(
        "--tag",
        default="latest",
        help="Image tag"
    )
    
    parser.add_argument(
        "--kubeconfig",
        help="Path to kubeconfig file"
    )

    parser.add_argument(
        "--release",
        default="vllm-inference",
        help="Helm release name"
    )

    parser.add_argument(
        "--chart",
        default="charts/vllm-inference",
        help="Path to Helm chart"
    )

    parser.add_argument(
        "--values",
        action="append",
        help="Helm values file; may be specified multiple times"
    )

    parser.add_argument(
        "--manifest-dir",
        default="k8s/deployments",
        help="Path to raw Kubernetes manifest file or directory"
    )

    parser.add_argument(
        "--canary-percentage",
        type=int,
        default=10,
        help="Canary traffic percentage for --strategy canary"
    )

    parser.add_argument(
        "--canary-release",
        default="vllm-inference-canary",
        help="Helm release name for the canary backend"
    )

    parser.add_argument(
        "--canary-values",
        action="append",
        help="Helm values file for the canary backend release"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned deployment commands/overrides without mutating the cluster"
    )
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager(
        namespace=args.namespace,
        kubeconfig=args.kubeconfig,
        dry_run=args.dry_run,
    )
    
    # Execute action
    success = False

    values_files = args.values or [f"{args.chart}/values.yaml"]
    
    if args.action == "validate":
        success = manager.validate_prerequisites(deploy_mode=args.deploy_mode)
    
    elif args.action == "build":
        success = manager.build_docker_images(tag=args.tag)
    
    elif args.action == "push":
        if not args.registry:
            logger.error("--registry required for push action")
            sys.exit(1)
        success = manager.push_docker_images(args.registry, args.tag)
    
    elif args.action == "deploy":
        success = manager.deploy_with_strategy(
            strategy=args.strategy,
            deploy_mode=args.deploy_mode,
            manifest_dir=args.manifest_dir,
            release=args.release,
            chart=args.chart,
            values_files=values_files,
            canary_percentage=args.canary_percentage,
            canary_release=args.canary_release,
            canary_values_files=args.canary_values,
        )
    
    elif args.action == "rollback":
        if args.deploy_mode == "helm":
            success = manager.rollback_helm_release(args.release)
        else:
            success = manager.rollback_deployment("vllm-server")
    
    # Exit with appropriate code
    if success:
        logger.info("✓ Action completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Action failed")
        sys.exit(1)

def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge two dictionaries without mutating inputs."""
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    main()
