"""Workflow-declared remote execution configuration models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

if __package__:
    from .execution_environments import ExecutionProvider
    from .remote_hosts import SshHostConfig
    from .vast_models import VastResourceProfile
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from execution_environments import ExecutionProvider
    from remote_hosts import SshHostConfig
    from vast_models import VastResourceProfile


@dataclass(frozen=True)
class RemoteConfiguration(ABC):
    """Describe one workflow-declared pool of remote execution capacity."""

    configuration_id: str
    display_name: str

    def __post_init__(self) -> None:
        """Reject ambiguous identities before provider discovery begins."""
        if not self.configuration_id.strip():
            raise ValueError("Remote configuration_id must not be empty.")
        if not self.display_name.strip():
            raise ValueError("Remote configuration display_name must not be empty.")

    @property
    @abstractmethod
    def provider(self) -> ExecutionProvider:
        """Return the provider represented by this configuration."""

    @property
    @abstractmethod
    def capacity_limit(self) -> int:
        """Return the maximum concurrent worker capacity contributed by this pool."""

    @abstractmethod
    def to_safe_dict(self) -> dict[str, Any]:
        """Return credential-free workflow and diagnostic metadata."""


@dataclass(frozen=True)
class ModalRemoteConfiguration(RemoteConfiguration):
    """Declare a Modal GPU type and its maximum concurrent container capacity."""

    gpu_type: str
    instance_count: int = 1

    def __post_init__(self) -> None:
        """Validate Modal capacity fields."""
        super().__post_init__()
        if not self.gpu_type.strip():
            raise ValueError("Modal gpu_type must not be empty.")
        if self.instance_count <= 0:
            raise ValueError("Modal instance_count must be positive.")

    @property
    def provider(self) -> ExecutionProvider:
        """Return the Modal provider identifier."""
        return ExecutionProvider.MODAL

    @property
    def capacity_limit(self) -> int:
        """Return the configured Modal container ceiling."""
        return self.instance_count

    def to_safe_dict(self) -> dict[str, Any]:
        """Return credential-free Modal configuration metadata."""
        return {
            "configuration_id": self.configuration_id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "gpu_type": self.gpu_type,
            "instance_count": self.instance_count,
        }


@dataclass(frozen=True)
class VastRemoteConfiguration(RemoteConfiguration):
    """Declare one Vast.ai marketplace capacity pool."""

    profile: VastResourceProfile

    def __post_init__(self) -> None:
        """Keep the wrapper identity aligned with the underlying Vast profile."""
        super().__post_init__()
        if self.profile.profile_id != self.configuration_id:
            raise ValueError(
                "Vast profile_id must match its remote configuration_id."
            )
        if self.profile.profile_name != self.display_name:
            raise ValueError(
                "Vast profile_name must match its remote configuration display_name."
            )

    @property
    def provider(self) -> ExecutionProvider:
        """Return the Vast.ai provider identifier."""
        return ExecutionProvider.VAST

    @property
    def capacity_limit(self) -> int:
        """Return the maximum number of managed Vast.ai instances."""
        return self.profile.maximum_instances

    def to_safe_dict(self) -> dict[str, Any]:
        """Return credential-free Vast configuration metadata."""
        return {
            "configuration_id": self.configuration_id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "maximum_instances": self.profile.maximum_instances,
            "search": self.profile.search_payload(),
        }


@dataclass(frozen=True)
class SshRemoteConfiguration(RemoteConfiguration):
    """Declare one SSH-accessible Docker host and its worker capacity."""

    host: SshHostConfig

    def __post_init__(self) -> None:
        """Keep workflow and host identities aligned."""
        super().__post_init__()
        if self.host.environment_id != self.configuration_id:
            raise ValueError(
                "SSH environment_id must match its remote configuration_id."
            )
        if self.host.display_name != self.display_name:
            raise ValueError(
                "SSH display_name must match its remote configuration display_name."
            )
        if self.host.maximum_workers <= 0:
            raise ValueError("SSH maximum_workers must be positive in a workflow.")

    @property
    def provider(self) -> ExecutionProvider:
        """Return the SSH Docker provider identifier."""
        return ExecutionProvider.SSH_DOCKER

    @property
    def capacity_limit(self) -> int:
        """Return the configured worker-container ceiling for this host."""
        return self.host.maximum_workers

    def to_safe_dict(self) -> dict[str, Any]:
        """Return credential-free SSH configuration metadata."""
        return {
            "configuration_id": self.configuration_id,
            "display_name": self.display_name,
            "provider": self.provider.value,
            "host": self.host.to_dict(),
        }


@dataclass(frozen=True)
class RemoteConfigurationSet:
    """Hold the complete authoritative remote capacity declaration for a workflow."""

    configurations: tuple[RemoteConfiguration, ...]

    def __post_init__(self) -> None:
        """Require at least one uniquely identified and named configuration."""
        if not self.configurations:
            raise ValueError(
                "Remote Execution Configurator requires at least one configuration."
            )
        configuration_ids = [
            configuration.configuration_id for configuration in self.configurations
        ]
        if len(configuration_ids) != len(set(configuration_ids)):
            raise ValueError("Remote configuration IDs must be unique.")
        normalized_names = [
            configuration.display_name.casefold()
            for configuration in self.configurations
        ]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("Remote configuration names must be unique.")
        modal_gpu_types = [
            configuration.gpu_type
            for configuration in self.configurations
            if isinstance(configuration, ModalRemoteConfiguration)
        ]
        if len(modal_gpu_types) != len(set(modal_gpu_types)):
            raise ValueError(
                "Each Modal GPU type may appear in only one remote configuration."
            )
        ssh_targets = [
            configuration.host.ssh_target
            for configuration in self.configurations
            if isinstance(configuration, SshRemoteConfiguration)
        ]
        if len(ssh_targets) != len(set(ssh_targets)):
            raise ValueError(
                "Each SSH destination may appear in only one remote configuration."
            )

    def to_safe_list(self) -> list[dict[str, Any]]:
        """Return ordered credential-free configuration metadata."""
        return [
            configuration.to_safe_dict()
            for configuration in self.configurations
        ]


__all__ = [
    "ModalRemoteConfiguration",
    "RemoteConfiguration",
    "RemoteConfigurationSet",
    "SshRemoteConfiguration",
    "VastRemoteConfiguration",
]
