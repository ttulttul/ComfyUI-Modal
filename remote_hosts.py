"""Persistent configuration for SSH-accessible Docker execution hosts."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from .execution_environments import (
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionProvider,
    )
else:  # pragma: no cover - stable remote entrypoints may import modules top-level.
    from execution_environments import (
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionProvider,
    )

logger = logging.getLogger(__name__)

REMOTE_HOSTS_CONFIG_VERSION = 1
REMOTE_HOSTS_CONFIG_FILENAME = "remote-execution-environments.json"
_ENVIRONMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


@dataclass(frozen=True)
class SshHostConfig:
    """Describe one user-managed Docker host reachable through SSH."""

    environment_id: str
    display_name: str
    ssh_target: str
    enabled: bool = True
    draining: bool = False
    cost_usd_per_second: float | None = None
    maximum_workers: int = 1
    reserve_vram_bytes: int = 0
    tags: frozenset[str] = frozenset()
    storage_volume_name: str | None = None
    capabilities: EnvironmentCapabilities | None = None
    health: EnvironmentHealth = EnvironmentHealth.UNKNOWN
    last_error: str | None = None

    def __post_init__(self) -> None:
        """Validate security-sensitive and scheduler-visible host fields."""
        if not _ENVIRONMENT_ID_PATTERN.fullmatch(self.environment_id):
            raise ValueError(
                "environment_id must contain lowercase letters, digits, underscores, or "
                "hyphens and be at most 63 characters."
            )
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty.")
        if not self.ssh_target.strip() or any(
            character in self.ssh_target for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("ssh_target must be a non-empty single-line SSH destination.")
        if self.ssh_target.startswith("-"):
            raise ValueError("ssh_target must not begin with an option prefix.")
        if self.maximum_workers < 0:
            raise ValueError("maximum_workers must not be negative.")
        if self.reserve_vram_bytes < 0:
            raise ValueError("reserve_vram_bytes must not be negative.")
        if self.cost_usd_per_second is not None and (
            not math.isfinite(self.cost_usd_per_second)
            or self.cost_usd_per_second < 0
        ):
            raise ValueError("cost_usd_per_second must be finite and non-negative.")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("tags must not contain empty values.")

    @property
    def resolved_storage_volume_name(self) -> str:
        """Return the Docker volume used for durable content-addressed storage."""
        if self.storage_volume_name:
            return self.storage_volume_name
        return f"comfy-remote-{self.environment_id}"

    def scheduling_state(self) -> EnvironmentSchedulingState:
        """Return the scheduler-facing view of this host."""
        health = EnvironmentHealth.DRAINING if self.draining else self.health
        return EnvironmentSchedulingState(
            environment_id=self.environment_id,
            provider=ExecutionProvider.SSH_DOCKER,
            enabled=self.enabled,
            health=health,
            cost_usd_per_second=self.cost_usd_per_second,
            capabilities=self.capabilities,
            tags=self.tags,
            maximum_workers=self.maximum_workers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible host record without credentials."""
        return {
            "environment_id": self.environment_id,
            "display_name": self.display_name,
            "ssh_target": self.ssh_target,
            "enabled": self.enabled,
            "draining": self.draining,
            "cost_usd_per_second": self.cost_usd_per_second,
            "maximum_workers": self.maximum_workers,
            "reserve_vram_bytes": self.reserve_vram_bytes,
            "tags": sorted(self.tags),
            "storage_volume_name": self.storage_volume_name,
            "capabilities": (
                self.capabilities.to_dict() if self.capabilities is not None else None
            ),
            "health": self.health.value,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SshHostConfig":
        """Build one validated host record from configuration data."""
        raw_capabilities = payload.get("capabilities")
        raw_tags = payload.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        raw_cost = payload.get("cost_usd_per_second")
        return cls(
            environment_id=str(payload.get("environment_id") or "").strip(),
            display_name=str(payload.get("display_name") or "").strip(),
            ssh_target=str(payload.get("ssh_target") or "").strip(),
            enabled=bool(payload.get("enabled", True)),
            draining=bool(payload.get("draining", False)),
            cost_usd_per_second=(float(raw_cost) if raw_cost is not None else None),
            maximum_workers=int(payload.get("maximum_workers", 1)),
            reserve_vram_bytes=int(payload.get("reserve_vram_bytes", 0)),
            tags=frozenset(str(tag).strip() for tag in tags),
            storage_volume_name=(
                str(payload["storage_volume_name"]).strip()
                if payload.get("storage_volume_name")
                else None
            ),
            capabilities=(
                EnvironmentCapabilities.from_dict(raw_capabilities)
                if isinstance(raw_capabilities, Mapping)
                else None
            ),
            health=EnvironmentHealth(str(payload.get("health") or "unknown")),
            last_error=(
                str(payload["last_error"]).strip()
                if payload.get("last_error")
                else None
            ),
        )


@dataclass(frozen=True)
class RemoteExecutionConfig:
    """Persist node-pack-wide generic remote execution configuration."""

    hosts: tuple[SshHostConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON document stored on disk."""
        return {
            "version": REMOTE_HOSTS_CONFIG_VERSION,
            "hosts": [host.to_dict() for host in self.hosts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RemoteExecutionConfig":
        """Build and validate one versioned configuration document."""
        version = int(payload.get("version", 0))
        if version != REMOTE_HOSTS_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported remote execution configuration version {version}."
            )
        raw_hosts = payload.get("hosts")
        host_payloads = raw_hosts if isinstance(raw_hosts, list) else []
        hosts = tuple(
            SshHostConfig.from_dict(host_payload)
            for host_payload in host_payloads
            if isinstance(host_payload, Mapping)
        )
        environment_ids = [host.environment_id for host in hosts]
        if len(environment_ids) != len(set(environment_ids)):
            raise ValueError("Remote execution environment IDs must be unique.")
        return cls(hosts=hosts)


@dataclass
class RemoteHostRegistry:
    """Read and atomically update SSH host configuration."""

    config_path: Path
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def for_user_directory(cls, user_directory: Path) -> "RemoteHostRegistry":
        """Create a registry beneath a ComfyUI user directory."""
        return cls(
            config_path=(
                user_directory.expanduser().resolve()
                / "comfyui-modal"
                / REMOTE_HOSTS_CONFIG_FILENAME
            )
        )

    def load(self) -> RemoteExecutionConfig:
        """Load the current registry or return an empty initial configuration."""
        with self._lock:
            if not self.config_path.exists():
                return RemoteExecutionConfig()
            try:
                payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Remote execution configuration {self.config_path} is unreadable."
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError("Remote execution configuration must be a JSON object.")
            return RemoteExecutionConfig.from_dict(payload)

    def save(self, config: RemoteExecutionConfig) -> None:
        """Atomically persist one complete registry document."""
        serialized = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        with self._lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.config_path.name}.",
                suffix=".tmp",
                dir=self.config_path.parent,
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
                    output_file.write(serialized)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.replace(temporary_path, self.config_path)
            finally:
                temporary_path.unlink(missing_ok=True)

    def replace_hosts(self, hosts: Sequence[SshHostConfig]) -> RemoteExecutionConfig:
        """Validate and replace the complete host list."""
        config = RemoteExecutionConfig(hosts=tuple(hosts))
        RemoteExecutionConfig.from_dict(config.to_dict())
        self.save(config)
        return config

    def get_host(self, environment_id: str) -> SshHostConfig:
        """Return one configured host or raise a descriptive lookup error."""
        for host in self.load().hosts:
            if host.environment_id == environment_id:
                return host
        raise KeyError(f"Unknown remote execution environment {environment_id!r}.")

    def update_probe_result(
        self,
        environment_id: str,
        *,
        capabilities: EnvironmentCapabilities | None,
        health: EnvironmentHealth,
        last_error: str | None,
    ) -> SshHostConfig:
        """Persist one host's latest capability probe outcome."""
        config = self.load()
        updated_hosts: list[SshHostConfig] = []
        updated_host: SshHostConfig | None = None
        for host in config.hosts:
            if host.environment_id != environment_id:
                updated_hosts.append(host)
                continue
            updated_host = replace(
                host,
                capabilities=capabilities,
                health=health,
                last_error=last_error,
            )
            updated_hosts.append(updated_host)
        if updated_host is None:
            raise KeyError(f"Unknown remote execution environment {environment_id!r}.")
        self.save(RemoteExecutionConfig(hosts=tuple(updated_hosts)))
        logger.info(
            "Updated SSH environment probe state environment=%s health=%s.",
            environment_id,
            health.value,
        )
        return updated_host
