"""
Model routing primitives for the API gateway.

The router is intentionally file-backed and dependency-light so it can run in
the slim gateway image. A JSON manifest can describe production and canary
targets for each logical model without requiring MLflow to be reachable on the
hot path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class UnknownModelRoute(ValueError):
    """Raised when a requested model cannot be routed."""


@dataclass(frozen=True)
class RouteTarget:
    """One routable backend target for a logical model."""

    name: str
    backend_url: str
    model: str
    weight: int = 100
    stage: str = "Production"
    version: Optional[str] = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteTarget":
        missing = [key for key in ("name", "backend_url", "model") if not data.get(key)]
        if missing:
            raise ValueError(f"Route target missing required fields: {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            backend_url=str(data["backend_url"]).rstrip("/"),
            model=str(data["model"]),
            weight=int(data.get("weight", 100)),
            stage=str(data.get("stage", "Production")),
            version=str(data["version"]) if data.get("version") is not None else None,
            enabled=bool(data.get("enabled", True)),
        )

    def as_public_dict(self) -> Dict[str, Any]:
        """Return safe routing metadata for the /models endpoint."""
        return {
            "name": self.name,
            "backend_url": self.backend_url,
            "model": self.model,
            "weight": self.weight,
            "stage": self.stage,
            "version": self.version,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ResolvedRoute:
    """Selected backend target for one request."""

    logical_model: str
    requested_model: Optional[str]
    target: RouteTarget


class ModelRouter:
    """Deterministic weighted router for logical model names."""

    def __init__(
        self,
        routes: Dict[str, List[RouteTarget]],
        default_model: Optional[str] = None,
        allow_passthrough: bool = False,
        fallback_backend_url: Optional[str] = None,
        canary_salt: str = "default",
    ):
        self.routes = routes
        self.default_model = default_model
        self.allow_passthrough = allow_passthrough
        self.fallback_backend_url = (
            fallback_backend_url.rstrip("/") if fallback_backend_url else None
        )
        self.canary_salt = canary_salt

    @classmethod
    def from_file(
        cls,
        path: str,
        fallback_backend_url: Optional[str] = None,
        canary_salt: str = "default",
    ) -> "ModelRouter":
        """Load a router from a JSON manifest."""
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(
            manifest,
            fallback_backend_url=fallback_backend_url,
            canary_salt=canary_salt,
        )

    @classmethod
    def from_dict(
        cls,
        manifest: Dict[str, Any],
        fallback_backend_url: Optional[str] = None,
        canary_salt: str = "default",
    ) -> "ModelRouter":
        routes: Dict[str, List[RouteTarget]] = {}
        for logical_model, route_config in (manifest.get("routes") or {}).items():
            if isinstance(route_config, dict):
                targets_data = route_config.get("targets", [])
            else:
                targets_data = route_config
            routes[str(logical_model)] = [
                RouteTarget.from_dict(target)
                for target in targets_data
            ]

        return cls(
            routes=routes,
            default_model=manifest.get("default_model"),
            allow_passthrough=bool(manifest.get("allow_passthrough", False)),
            fallback_backend_url=fallback_backend_url,
            canary_salt=canary_salt,
        )

    @classmethod
    def from_static(
        cls,
        backend_url: str,
        default_model: Optional[str] = None,
        canary_salt: str = "default",
    ) -> "ModelRouter":
        """Create a passthrough router matching the legacy single-backend behavior."""
        logical_model = default_model or "default"
        routes: Dict[str, List[RouteTarget]] = {
            logical_model: [
                RouteTarget(
                    name="default",
                    backend_url=backend_url.rstrip("/"),
                    model=default_model or "",
                    weight=100,
                    stage="Production",
                )
            ]
        }

        return cls(
            routes=routes,
            default_model=default_model,
            allow_passthrough=True,
            fallback_backend_url=backend_url,
            canary_salt=canary_salt,
        )

    def resolve(
        self,
        requested_model: Optional[str],
        user_id: str,
        endpoint: str,
    ) -> ResolvedRoute:
        """Resolve a request into a concrete backend target."""
        logical_model = requested_model or self.default_model
        if not logical_model:
            logical_model = "default"

        active_targets = self._active_targets(logical_model)
        if not active_targets and not requested_model and self.default_model:
            logical_model = self.default_model
            active_targets = self._active_targets(logical_model)

        if active_targets:
            target = self._choose_target(logical_model, user_id, endpoint, active_targets)
            return ResolvedRoute(
                logical_model=logical_model,
                requested_model=requested_model,
                target=target,
            )

        if self.allow_passthrough and requested_model and self.fallback_backend_url:
            return ResolvedRoute(
                logical_model=requested_model,
                requested_model=requested_model,
                target=RouteTarget(
                    name="passthrough",
                    backend_url=self.fallback_backend_url,
                    model=requested_model,
                    weight=100,
                    stage="Passthrough",
                ),
            )

        raise UnknownModelRoute(f"No active route configured for model '{logical_model}'")

    def list_models(self) -> List[Dict[str, Any]]:
        """Return logical models and their configured targets."""
        models = []
        for logical_model in sorted(self.routes):
            targets = self.routes[logical_model]
            active_targets = [target for target in targets if target.enabled and target.weight > 0]
            models.append({
                "id": logical_model,
                "type": "completion",
                "active": bool(active_targets),
                "targets": [target.as_public_dict() for target in targets],
            })
        return models

    def _active_targets(self, logical_model: str) -> List[RouteTarget]:
        return [
            target for target in self.routes.get(logical_model, [])
            if target.enabled and target.weight > 0
        ]

    def _choose_target(
        self,
        logical_model: str,
        user_id: str,
        endpoint: str,
        targets: List[RouteTarget],
    ) -> RouteTarget:
        total_weight = sum(max(0, target.weight) for target in targets)
        if total_weight <= 0:
            raise UnknownModelRoute(f"No weighted route configured for model '{logical_model}'")

        bucket = self._stable_bucket(logical_model, user_id, endpoint) % total_weight
        cumulative = 0
        for target in targets:
            cumulative += max(0, target.weight)
            if bucket < cumulative:
                return target
        return targets[-1]

    def _stable_bucket(self, logical_model: str, user_id: str, endpoint: str) -> int:
        route_key = f"{self.canary_salt}:{logical_model}:{user_id}:{endpoint}"
        digest = hashlib.sha256(route_key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
