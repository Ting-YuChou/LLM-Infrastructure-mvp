"""
MLflow Model Registry Client
Handles model versioning, lineage tracking, and deployment workflows
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:  # Allow non-MLflow tests and tools to import this module.
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _require_mlflow():
    """Return MLflow module/client class or raise a clear runtime error."""
    if mlflow is None or MlflowClient is None:
        raise RuntimeError(
            "MLflow is required for registry operations. Install the 'mlflow' package "
            "or disable tracking.mlflow_enabled."
        )
    return mlflow, MlflowClient


class LLMModelRegistry:
    """
    MLflow-based model registry for LLM infrastructure
    
    Features:
    - Model versioning and lineage tracking
    - Stage transitions (Staging -> Production)
    - Metadata and tags management
    - Deployment tracking
    - Model comparison and selection
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "llm-training"
    ):
        """
        Initialize registry client
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.mlflow, client_cls = _require_mlflow()
        
        # Set tracking URI
        self.mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize client
        self.client = client_cls(tracking_uri=tracking_uri)
        
        # Set or create experiment
        self.experiment_id = self._get_or_create_experiment(experiment_name)
        
        logger.info(f"Initialized MLflow registry")
        logger.info(f"Tracking URI: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """Get or create experiment"""
        try:
            experiment = self.mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass
        
        # Create new experiment
        experiment_id = self.mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "llm-infrastructure",
                "created_at": datetime.now().isoformat()
            }
        )
        logger.info(f"Created new experiment: {experiment_name}")
        return experiment_id
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        model_type: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Register a new model version to the registry
        
        Args:
            model_path: Path to model checkpoint
            model_name: Name to register model under
            model_type: Type of model (sft, reward, rlhf)
            metrics: Training/evaluation metrics
            params: Model parameters and config
            tags: Additional tags for filtering
            description: Model description
            
        Returns:
            Model version string
        """
        logger.info(f"Registering model: {model_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model type: {model_type}")
        
        # Start MLflow run
        with self.mlflow.start_run(experiment_id=self.experiment_id) as run:
            
            # Log parameters
            if params:
                self.mlflow.log_params(params)
            
            # Log metrics
            if metrics:
                self.mlflow.log_metrics(metrics)
            
            # Log tags
            run_tags = {
                "model_type": model_type,
                "model_name": model_name,
                "registered_at": datetime.now().isoformat(),
            }
            if tags:
                run_tags.update(tags)
            self.mlflow.set_tags(run_tags)
            
            # Log model artifacts
            logger.info("Logging model artifacts...")
            
            # For HuggingFace models, log the entire checkpoint directory
            if os.path.isdir(model_path):
                # Log as artifact
                self.mlflow.log_artifacts(model_path, artifact_path="model")
            else:
                # Log single file
                self.mlflow.log_artifact(model_path)
            
            # Register model
            logger.info(f"Registering model to registry: {model_name}")
            model_uri = f"runs:/{run.info.run_id}/model"
            
            mv = self.mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=run_tags
            )
            
            # Update model version description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=mv.version,
                    description=description
                )
            
            logger.info(f"✓ Model registered: {model_name} v{mv.version}")
            logger.info(f"  Run ID: {run.info.run_id}")
            
            return mv.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> None:
        """
        Transition model to a new stage
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in target stage
        """
        logger.info(f"Transitioning {model_name} v{version} to {stage}")
        
        # Archive existing models in target stage if requested
        if archive_existing and stage in ["Staging", "Production"]:
            existing_versions = self.client.get_latest_versions(
                model_name,
                stages=[stage]
            )
            
            for mv in existing_versions:
                logger.info(f"Archiving existing {stage} version: {mv.version}")
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
        
        # Transition to new stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"✓ Transitioned to {stage}")
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Any:
        """
        Get specific model version
        
        Args:
            model_name: Model name
            version: Specific version (overrides stage)
            stage: Stage to get (e.g., "Production")
            alias: Alias to get (e.g., "champion")
            
        Returns:
            Model version object
        """
        if version:
            return self.client.get_model_version(model_name, version)

        if alias:
            return self.get_model_version_by_alias(model_name, alias)
        
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0]
            raise ValueError(f"No model found in stage: {stage}")
        
        # Get latest version
        versions = self.client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(versions, key=lambda v: int(v.version))
        
        raise ValueError(f"No versions found for model: {model_name}")
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        alias: Optional[str] = "champion",
        dst_path: Optional[str] = None,
    ) -> str:
        """
        Get model path for loading
        
        Args:
            model_name: Model name
            version: Specific version
            stage: Legacy stage to load from
            alias: Alias to load from when version is not provided
            dst_path: Optional download destination
            
        Returns:
            Path to model artifacts
        """
        mv = self.get_model_version(model_name, version, stage, alias)
        
        logger.info(f"Loading model: {model_name} v{mv.version}")
        logger.info(f"Stage: {getattr(mv, 'current_stage', 'None')}")
        logger.info(f"Alias: {alias}")
        
        # Download artifacts to local path
        safe_model_name = model_name.replace("/", "_")
        local_path = self.mlflow.artifacts.download_artifacts(
            artifact_uri=mv.source,
            dst_path=dst_path or f"/tmp/mlflow_models/{safe_model_name}/{mv.version}"
        )
        
        logger.info(f"✓ Model downloaded to: {local_path}")
        return local_path

    def set_model_alias(
        self,
        model_name: str,
        alias: str,
        version: str,
    ) -> None:
        """Set or update a registered model alias."""
        self.client.set_registered_model_alias(model_name, alias, version)
        logger.info(f"✓ Set alias {model_name}@{alias} -> v{version}")

    def get_model_version_by_alias(
        self,
        model_name: str,
        alias: str,
    ) -> Any:
        """Get a model version by alias."""
        return self.client.get_model_version_by_alias(model_name, alias)

    def promote_to_champion(
        self,
        model_name: str,
        version: str,
        archive_previous: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Promote a model version by assigning the champion alias.

        This alias-based API is preferred for new code because MLflow model
        stages are deprecated.
        """
        if archive_previous:
            try:
                previous = self.get_model_version_by_alias(model_name, "champion")
            except Exception:
                previous = None

            if previous is not None and str(previous.version) != str(version):
                self.client.set_model_version_tag(
                    model_name,
                    str(previous.version),
                    "lifecycle_alias",
                    "archived",
                )
                self.client.set_model_version_tag(
                    model_name,
                    str(previous.version),
                    "replaced_by_version",
                    str(version),
                )

        version_tags = {
            "lifecycle_alias": "champion",
            "validation_status": "passed",
            "promoted_at": datetime.now().isoformat(),
        }
        if tags:
            version_tags.update(tags)

        for key, value in version_tags.items():
            self.client.set_model_version_tag(model_name, version, key, str(value))
        self.set_model_alias(model_name, "champion", version)
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered models
        
        Args:
            model_type: Filter by model type (sft, reward, rlhf)
            stage: Filter by stage
            
        Returns:
            List of model information dicts
        """
        # Search for models
        filter_string = ""
        if model_type:
            filter_string = f"tags.model_type='{model_type}'"
        
        models = []
        
        # Get all registered models
        for rm in self.client.search_registered_models(filter_string):
            # Get versions for this model
            versions = self.client.search_model_versions(f"name='{rm.name}'")
            
            # Filter by stage if specified
            if stage:
                versions = [v for v in versions if v.current_stage == stage]
            
            for version in versions:
                models.append({
                    "name": rm.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "created_at": datetime.fromtimestamp(version.creation_timestamp / 1000),
                    "tags": version.tags,
                    "run_id": version.run_id,
                })
        
        return models
    
    def compare_models(
        self,
        model_name: str,
        version1: str,
        version2: str,
        metric_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics between two model versions
        
        Args:
            model_name: Model name
            version1: First version
            version2: Second version
            metric_names: Metrics to compare
            
        Returns:
            Comparison results
        """
        # Get model versions
        mv1 = self.get_model_version(model_name, version1)
        mv2 = self.get_model_version(model_name, version2)
        
        # Get runs
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)
        
        # Extract metrics
        comparison = {}
        
        for metric_name in metric_names:
            comparison[metric_name] = {
                f"v{version1}": run1.data.metrics.get(metric_name),
                f"v{version2}": run2.data.metrics.get(metric_name),
            }
        
        return comparison
    
    def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> None:
        """
        Delete a specific model version
        
        Args:
            model_name: Model name
            version: Version to delete
        """
        logger.warning(f"Deleting model version: {model_name} v{version}")
        self.client.delete_model_version(model_name, version)
        logger.info("✓ Model version deleted")
    
    def add_model_tags(
        self,
        model_name: str,
        version: str,
        tags: Dict[str, str]
    ) -> None:
        """Add tags to model version"""
        for key, value in tags.items():
            self.client.set_model_version_tag(model_name, version, key, value)
        
        logger.info(f"✓ Added {len(tags)} tags to {model_name} v{version}")
    
    def get_deployment_history(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> List[Dict[str, Any]]:
        """
        Get deployment history for a model stage
        
        Args:
            model_name: Model name
            stage: Stage to track
            
        Returns:
            List of deployment events
        """
        versions = self.client.search_model_versions(
            f"name='{model_name}'"
        )
        
        history = []
        for v in versions:
            if v.current_stage == stage or stage == "All":
                # Get run details
                run = self.client.get_run(v.run_id)
                
                history.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "deployed_at": datetime.fromtimestamp(v.last_updated_timestamp / 1000),
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                })
        
        # Sort by deployment time
        history.sort(key=lambda x: x["deployed_at"], reverse=True)
        
        return history


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLflow Model Registry CLI")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--action", choices=["list", "register", "promote"])
    parser.add_argument("--model-name")
    parser.add_argument("--model-path")
    parser.add_argument("--version")
    parser.add_argument("--stage")
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = LLMModelRegistry(tracking_uri=args.tracking_uri)
    
    # Execute action
    if args.action == "list":
        models = registry.list_models()
        print(f"\nFound {len(models)} models:\n")
        for model in models:
            print(f"  {model['name']} v{model['version']} [{model['stage']}]")
    
    elif args.action == "register":
        if not args.model_name or not args.model_path:
            print("Error: --model-name and --model-path required")
            return
        
        version = registry.register_model(
            model_path=args.model_path,
            model_name=args.model_name,
            model_type="sft",
        )
        print(f"✓ Registered {args.model_name} v{version}")
    
    elif args.action == "promote":
        if not args.model_name or not args.version or not args.stage:
            print("Error: --model-name, --version, and --stage required")
            return
        
        registry.transition_model_stage(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage
        )
        print(f"✓ Promoted to {args.stage}")


if __name__ == "__main__":
    main()
