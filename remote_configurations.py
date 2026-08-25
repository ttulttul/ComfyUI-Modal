"""Workflow-declared remote execution configuration models."""

from __future__ import annotations

import re
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

_R2_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")


@dataclass(frozen=True)
class RemoteExecutionConfiguration(ABC):
    """Describe one workflow-declared remote-execution configuration value."""

    configuration_id: str
    display_name: str

    def __post_init__(self) -> None:
        """Reject ambiguous identities before provider discovery begins."""
        if not self.configuration_id.strip():
            raise ValueError("Remote execution configuration_id must not be empty.")
        if not self.display_name.strip():
            raise ValueError(
                "Remote execution configuration display_name must not be empty."
            )

    @abstractmethod
    def to_safe_dict(self) -> dict[str, Any]:
        """Return credential-free workflow and diagnostic metadata."""


@dataclass(frozen=True)
class RemoteConfiguration(RemoteExecutionConfiguration, ABC):
    """Describe one workflow-declared pool of remote execution capacity."""

    @property
    @abstractmethod
    def provider(self) -> ExecutionProvider:
        """Return the provider represented by this configuration."""

    @property
    @abstractmethod
    def capacity_limit(self) -> int:
        """Return the maximum concurrent worker capacity contributed by this pool."""

@dataclass(frozen=True)
class StorageBackingConfiguration(RemoteExecutionConfiguration, ABC):
    """Describe storage shared by one or more remote execution targets."""

    @property
    @abstractmethod
    def storage_provider(self) -> str:
        """Return the stable storage-provider identifier."""


@dataclass(frozen=True)
class R2StorageBackingConfiguration(StorageBackingConfiguration):
    """Reference controller-held credentials for one Cloudflare R2 bucket."""

    account_id: str
    bucket: str
    credential_id: str
    jurisdiction: str = "default"
    key_prefix: str = "comfy-modal-cache/v1/blobs/sha256"
    write_back_mode: str = "async"

    def __post_init__(self) -> None:
        """Validate non-secret R2 workflow metadata."""
        super().__post_init__()
        if len(self.account_id) != 32 or any(
            character not in "0123456789abcdefABCDEF" for character in self.account_id
        ):
            raise ValueError(
                "Cloudflare R2 account ID must be 32 hexadecimal characters; "
                "use Login on the R2 Storage Configuration node."
            )
        if not _R2_BUCKET_PATTERN.fullmatch(self.bucket):
            raise ValueError(
                "Cloudflare R2 bucket must contain 3-63 lowercase letters, digits, "
                "or hyphens and begin and end with a letter or digit."
            )
        if not self.credential_id.strip():
            raise ValueError("Cloudflare R2 credential_id must not be empty.")
        if self.jurisdiction not in {"default", "eu", "fedramp", "us"}:
            raise ValueError("Cloudflare R2 jurisdiction is not supported.")
        if self.write_back_mode not in {"async", "off", "sync"}:
            raise ValueError("Cloudflare R2 write-back mode is not supported.")

    @property
    def storage_provider(self) -> str:
        """Return the Cloudflare R2 provider identifier."""
        return "cloudflare_r2"

    def to_safe_dict(self) -> dict[str, Any]:
        """Return R2 metadata without controller credentials."""
        return {
            "configuration_id": self.configuration_id,
            "display_name": self.display_name,
            "configuration_kind": "storage",
            "storage_provider": self.storage_provider,
            "account_id": self.account_id,
            "bucket": self.bucket,
            "jurisdiction": self.jurisdiction,
            "key_prefix": self.key_prefix,
            "write_back_mode": self.write_back_mode,
        }


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
    """Hold the authoritative capacity and storage declaration for a workflow."""

    configurations: tuple[RemoteExecutionConfiguration, ...]

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
        if not any(
            isinstance(configuration, RemoteConfiguration)
            for configuration in self.configurations
        ):
            raise ValueError(
                "Remote Execution Configurator requires at least one Modal, Vast.ai, "
                "or SSH capacity configuration."
            )
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
        storage_providers = [
            configuration.storage_provider
            for configuration in self.configurations
            if isinstance(configuration, StorageBackingConfiguration)
        ]
        if len(storage_providers) != len(set(storage_providers)):
            raise ValueError(
                "Each storage provider may appear only once in a Remote Execution "
                "Configurator."
            )

    @property
    def capacity_configurations(self) -> tuple[RemoteConfiguration, ...]:
        """Return only configurations that contribute execution capacity."""
        return tuple(
            configuration
            for configuration in self.configurations
            if isinstance(configuration, RemoteConfiguration)
        )

    @property
    def storage_configurations(self) -> tuple[StorageBackingConfiguration, ...]:
        """Return only shared storage backing configurations."""
        return tuple(
            configuration
            for configuration in self.configurations
            if isinstance(configuration, StorageBackingConfiguration)
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
    "RemoteExecutionConfiguration",
    "R2StorageBackingConfiguration",
    "SshRemoteConfiguration",
    "StorageBackingConfiguration",
    "VastRemoteConfiguration",
]
