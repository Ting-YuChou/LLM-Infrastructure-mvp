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
    - Blue-green deployment
    - Canary deployment
    - Health checks
    - Automatic rollback
    - Deployment history tracking
    """
    
    def __init__(
        self,
        namespace: str = "llm-inference",
        kubeconfig: Optional[str] = None
    ):
        """
        Initialize deployment manager
        
        Args:
            namespace: Kubernetes namespace
            kubeconfig: Path to kubeconfig file
        """
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        
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
        values_files: Optional[List[str]] = None
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
        ]

        for values_file in values_files or []:
            cmd.extend(["-f", values_file])

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
    
    def health_check(self, service: str, path: str = "/health") -> bool:
        """
        Perform health check on service
        
        Args:
            service: Service name
            path: Health check path
            
        Returns:
            True if healthy
        """
        logger.info(f"Health check: {service}{path}")
        
        # Port forward to service
        port_forward_proc = subprocess.Popen([
            "kubectl", "port-forward",
            f"svc/{service}",
            "8080:80",
            "-n", self.namespace
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for port forward to establish
        time.sleep(3)
        
        try:
            # Make health check request
            import requests
            response = requests.get(f"http://localhost:8080{path}", timeout=10)
            
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
            return self.wait_for_rollout("vllm-server")

        except subprocess.CalledProcessError as e:
            logger.error(f"Helm rollback failed: {e}")
            return False
    
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
            return self._deploy_canary(manifest_dir, **kwargs)
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
        if not self.wait_for_rollout("vllm-server"):
            logger.error("Rollout failed, initiating rollback...")
            self.rollback_deployment("vllm-server")
            return False
        
        # Health check
        if not self.health_check("vllm-service"):
            logger.error("Health check failed, initiating rollback...")
            self.rollback_deployment("vllm-server")
            return False
        
        logger.info("✓ Rolling deployment successful")
        return True
    
    def _deploy_blue_green(self, manifest_dir: str) -> bool:
        """Blue-green deployment"""
        logger.info("Blue-green deployment not yet implemented")
        return False
    
    def _deploy_canary(
        self,
        manifest_dir: str,
        canary_percentage: int = 10
    ) -> bool:
        """Canary deployment"""
        logger.info(f"Canary deployment ({canary_percentage}% traffic)")
        logger.info("Canary deployment not yet implemented")
        return False


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
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    manager = DeploymentManager(
        namespace=args.namespace,
        kubeconfig=args.kubeconfig
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


if __name__ == "__main__":
    main()
