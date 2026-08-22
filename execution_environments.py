"""Provider-neutral remote execution environment models and scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class ExecutionProvider(str, Enum):
    """Identify one implementation of remote workflow execution."""

    MODAL = "modal"
    SSH_DOCKER = "ssh_docker"


class ExecutionPolicy(str, Enum):
    """Describe how eligible workflow components select a provider."""

    MODAL = "modal"
    SELF_HOSTED = "self_hosted"
    AUTOMATIC = "automatic"


class EnvironmentHealth(str, Enum):
    """Describe whether a remote environment may receive new work."""

    UNKNOWN = "unknown"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DRAINING = "draining"


@dataclass(frozen=True)
class GpuCapability:
    """Describe one GPU visible to a remote execution environment."""

    uuid: str
    name: str
    total_vram_bytes: int
    free_vram_bytes: int | None = None
    compute_capability: str | None = None
    driver_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this GPU."""
        return {
            "uuid": self.uuid,
            "name": self.name,
            "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GpuCapability":
        """Build one validated GPU capability from persisted data."""
        return cls(
            uuid=str(payload.get("uuid") or "").strip(),
            name=str(payload.get("name") or "Unknown GPU").strip(),
            total_vram_bytes=_non_negative_int(
                payload.get("total_vram_bytes"), "total_vram_bytes"
            ),
            free_vram_bytes=_optional_non_negative_int(
                payload.get("free_vram_bytes"), "free_vram_bytes"
            ),
            compute_capability=_optional_string(payload.get("compute_capability")),
            driver_version=_optional_string(payload.get("driver_version")),
        )


@dataclass(frozen=True)
class EnvironmentCapabilities:
    """Describe resources and runtime features discovered on one environment."""

    architecture: str
    operating_system: str
    cpu_count: int
    total_ram_bytes: int
    available_ram_bytes: int | None
    available_disk_bytes: int | None
    docker_version: str
    docker_rootless: bool
    nvidia_container_runtime: bool
    gpus: tuple[GpuCapability, ...] = ()
    probed_at_epoch: float | None = None

    @property
    def maximum_vram_bytes(self) -> int:
        """Return the largest total VRAM value exposed by one GPU."""
        return max((gpu.total_vram_bytes for gpu in self.gpus), default=0)

    @property
    def maximum_free_vram_bytes(self) -> int:
        """Return the largest currently free VRAM value when reported."""
        return max(
            (
                gpu.free_vram_bytes
                for gpu in self.gpus
                if gpu.free_vram_bytes is not None
            ),
            default=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of these capabilities."""
        return {
            "architecture": self.architecture,
            "operating_system": self.operating_system,
            "cpu_count": self.cpu_count,
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "available_disk_bytes": self.available_disk_bytes,
            "docker_version": self.docker_version,
            "docker_rootless": self.docker_rootless,
            "nvidia_container_runtime": self.nvidia_container_runtime,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "probed_at_epoch": self.probed_at_epoch,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnvironmentCapabilities":
        """Build validated capabilities from persisted data."""
        raw_gpus = payload.get("gpus")
        gpu_payloads = raw_gpus if isinstance(raw_gpus, list) else []
        return cls(
            architecture=str(payload.get("architecture") or "unknown").strip(),
            operating_system=str(payload.get("operating_system") or "unknown").strip(),
            cpu_count=_non_negative_int(payload.get("cpu_count"), "cpu_count"),
            total_ram_bytes=_non_negative_int(
                payload.get("total_ram_bytes"), "total_ram_bytes"
            ),
            available_ram_bytes=_optional_non_negative_int(
                payload.get("available_ram_bytes"), "available_ram_bytes"
            ),
            available_disk_bytes=_optional_non_negative_int(
                payload.get("available_disk_bytes"), "available_disk_bytes"
            ),
            docker_version=str(payload.get("docker_version") or "unknown").strip(),
            docker_rootless=bool(payload.get("docker_rootless", False)),
            nvidia_container_runtime=bool(
                payload.get("nvidia_container_runtime", False)
            ),
            gpus=tuple(
                GpuCapability.from_dict(gpu_payload)
                for gpu_payload in gpu_payloads
                if isinstance(gpu_payload, Mapping)
            ),
            probed_at_epoch=_optional_non_negative_float(
                payload.get("probed_at_epoch"), "probed_at_epoch"
            ),
        )


@dataclass(frozen=True)
class ComponentResourceRequirements:
    """Describe hard requirements and cost inputs for one remote component."""

    minimum_vram_bytes: int = 0
    minimum_ram_bytes: int = 0
    gpu_required: bool = True
    architecture: str | None = None
    estimated_execution_seconds: float = 0.0
    estimated_transfer_seconds: float = 0.0
    required_tags: frozenset[str] = frozenset()
    preferred_environment_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentSchedulingState:
    """Combine configured and transient state used for scheduling."""

    environment_id: str
    provider: ExecutionProvider
    enabled: bool
    health: EnvironmentHealth
    cost_usd_per_second: float | None
    capabilities: EnvironmentCapabilities | None
    tags: frozenset[str] = frozenset()
    active_workers: int = 0
    maximum_workers: int = 1
    queue_delay_seconds: float = 0.0
    cold_start_seconds: float = 0.0
    cached_artifact_keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ExecutionAssignment:
    """Record the selected environment and its predicted scheduling cost."""

    environment_id: str
    provider: ExecutionProvider
    predicted_cost_usd: float | None
    predicted_completion_seconds: float
    reasons: tuple[str, ...] = ()


class ExecutionEnvironmentClient(Protocol):
    """Provider-neutral client used to prepare and invoke remote environments."""

    def execute_payload(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one component payload and return deserialized outputs."""

    async def execute_payload_async(
        self,
        payload: Mapping[str, Any],
        kwargs: Mapping[str, Any],
    ) -> Sequence[Any]:
        """Execute one component payload asynchronously."""


class NoCompatibleExecutionEnvironmentError(RuntimeError):
    """Raised when no configured environment satisfies component requirements."""


class CostAwareEnvironmentScheduler:
    """Select the least-cost compatible environment with deterministic tie-breaking."""

    def choose(
        self,
        environments: Sequence[EnvironmentSchedulingState],
        requirements: ComponentResourceRequirements,
    ) -> ExecutionAssignment:
        """Return the best currently compatible execution environment."""
        candidates: list[tuple[tuple[Any, ...], ExecutionAssignment]] = []
        rejection_reasons: list[str] = []
        for environment in environments:
            rejection = self._incompatibility_reason(environment, requirements)
            if rejection is not None:
                rejection_reasons.append(f"{environment.environment_id}: {rejection}")
                continue

            completion_seconds = max(
                0.0,
                environment.queue_delay_seconds
                + environment.cold_start_seconds
                + requirements.estimated_transfer_seconds
                + requirements.estimated_execution_seconds,
            )
            predicted_cost = (
                completion_seconds * environment.cost_usd_per_second
                if environment.cost_usd_per_second is not None
                else None
            )
            preferred_rank = self._preferred_rank(
                environment.environment_id,
                requirements.preferred_environment_ids,
            )
            unknown_cost_rank = 1 if predicted_cost is None else 0
            cost_rank = predicted_cost if predicted_cost is not None else math.inf
            assignment = ExecutionAssignment(
                environment_id=environment.environment_id,
                provider=environment.provider,
                predicted_cost_usd=predicted_cost,
                predicted_completion_seconds=completion_seconds,
                reasons=(
                    f"compatible with {len(environment.capabilities.gpus) if environment.capabilities else 0} GPU(s)",
                    (
                        f"estimated cost ${predicted_cost:.6f}"
                        if predicted_cost is not None
                        else "cost unknown"
                    ),
                ),
            )
            candidates.append(
                (
                    (
                        preferred_rank,
                        unknown_cost_rank,
                        cost_rank,
                        completion_seconds,
                        environment.environment_id,
                    ),
                    assignment,
                )
            )

        if not candidates:
            details = "; ".join(rejection_reasons) or "no environments are configured"
            raise NoCompatibleExecutionEnvironmentError(
                f"No compatible remote execution environment is available: {details}."
            )
        return min(candidates, key=lambda candidate: candidate[0])[1]

    def _incompatibility_reason(
        self,
        environment: EnvironmentSchedulingState,
        requirements: ComponentResourceRequirements,
    ) -> str | None:
        """Return a human-readable hard-constraint failure when incompatible."""
        if not environment.enabled:
            return "disabled"
        if environment.health not in {EnvironmentHealth.READY, EnvironmentHealth.DEGRADED}:
            return f"health is {environment.health.value}"
        if environment.maximum_workers <= 0:
            return "worker limit is zero"
        capabilities = environment.capabilities
        if capabilities is None:
            return "capabilities have not been probed"
        if requirements.architecture and capabilities.architecture != requirements.architecture:
            return (
                f"architecture {capabilities.architecture!r} does not satisfy "
                f"{requirements.architecture!r}"
            )
        if capabilities.total_ram_bytes < requirements.minimum_ram_bytes:
            return "insufficient RAM"
        if not requirements.required_tags.issubset(environment.tags):
            return "required tags are missing"
        if requirements.gpu_required:
            if not capabilities.nvidia_container_runtime:
                return "NVIDIA container runtime is unavailable"
            if capabilities.maximum_vram_bytes < requirements.minimum_vram_bytes:
                return "insufficient GPU VRAM"
        return None

    def _preferred_rank(
        self,
        environment_id: str,
        preferred_environment_ids: tuple[str, ...],
    ) -> int:
        """Return a stable rank for explicit environment preferences."""
        if not preferred_environment_ids:
            return 0
        try:
            return preferred_environment_ids.index(environment_id)
        except ValueError:
            return len(preferred_environment_ids) + 1


def _non_negative_int(value: Any, field_name: str) -> int:
    """Return one non-negative integer or raise a validation error."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


def _optional_non_negative_int(value: Any, field_name: str) -> int | None:
    """Return an optional non-negative integer."""
    if value is None:
        return None
    return _non_negative_int(value, field_name)


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    """Return an optional finite non-negative floating-point number."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative number.") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative number.")
    return normalized


def _optional_string(value: Any) -> str | None:
    """Return stripped optional text."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
