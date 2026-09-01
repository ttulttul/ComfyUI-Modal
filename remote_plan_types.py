"""Shared value types for remote graph analysis and execution planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

if __package__:
    from .execution_environments import ExecutionAssignment
    from .remote_configurations import RemoteConfiguration, RemoteConfigurationSet
    from .remote_hosts import SshHostConfig
    from .sync_engine import SyncedAsset
    from .vast_service import VastService
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import ExecutionAssignment
    from remote_configurations import RemoteConfiguration, RemoteConfigurationSet
    from remote_hosts import SshHostConfig
    from sync_engine import SyncedAsset
    from vast_service import VastService


@dataclass(frozen=True)
class ComponentMemoryEstimate:
    """Describe conservative scheduler memory floors inferred from model assets."""

    minimum_vram_bytes: int
    minimum_ram_bytes: int
    model_asset_count: int = 0
    largest_model_bytes: int = 0


@dataclass(frozen=True)
class LinkedOutputRef:
    """Reference a node output slot within a prompt graph."""

    node_id: str
    output_index: int


@dataclass(frozen=True)
class InputTarget:
    """Describe a target input inside a remote component."""

    node_id: str
    input_name: str


@dataclass
class BoundaryInputSpec:
    """Describe one local-to-remote boundary value for a component."""

    proxy_input_name: str
    source: LinkedOutputRef
    io_type: str
    targets: list[InputTarget] = field(default_factory=list)


@dataclass
class BoundaryOutputSpec:
    """Describe one value exported across a remote component boundary."""

    proxy_output_name: str
    source: LinkedOutputRef
    io_type: str
    is_list: bool
    preview_target_node_ids: list[str] = field(default_factory=list)
    session_output: bool = False
    session_consumer_node_ids: list[str] = field(default_factory=list)
    local_materializer_node_id: str | None = None
    local_materializer_consumer_node_ids: list[str] = field(default_factory=list)


@dataclass
class StaticToMappedBoundarySpec:
    """Describe one static-phase output injected into each mapped item run."""

    proxy_name: str
    source: LinkedOutputRef
    io_type: str
    is_list: bool
    targets: list[InputTarget] = field(default_factory=list)


@dataclass(frozen=True)
class ProducedPhaseOutputSpec:
    """Describe one output published by an earlier split-proxy phase."""

    proxy_output_name: str
    source: LinkedOutputRef
    io_type: str
    is_list: bool
    session_output: bool = False


@dataclass
class RemoteComponentPlan:
    """Execution and rewrite plan for one connected remote component."""

    node_ids: list[str]
    representative_node_id: str
    boundary_inputs: list[BoundaryInputSpec]
    boundary_outputs: list[BoundaryOutputSpec]
    execute_node_ids: list[str]
    contains_output_node: bool
    mapped_boundary_input_name: str | None = None
    mapped_boundary_input_io_type: str | None = None
    mapped_boundary_source_node_id: str | None = None
    static_node_ids: list[str] = field(default_factory=list)
    mapped_node_ids: list[str] = field(default_factory=list)
    mapped_execute_node_ids: list[str] = field(default_factory=list)
    static_execute_node_ids: list[str] = field(default_factory=list)
    static_to_mapped_boundaries: list[StaticToMappedBoundarySpec] = field(
        default_factory=list
    )
    local_tap_node_ids: list[str] = field(default_factory=list)
    local_tap_terminal_node_ids: list[str] = field(default_factory=list)


@dataclass
class RewriteSummary:
    """Summary of the prompt rewrite performed for a queue request."""

    remote_node_ids: list[str] = field(default_factory=list)
    remote_component_ids: list[str] = field(default_factory=list)
    component_node_ids_by_representative: dict[str, list[str]] = field(
        default_factory=dict
    )
    component_dependency_ids_by_representative: dict[str, list[str]] = field(
        default_factory=dict
    )
    component_execution_stages: list[list[str]] = field(default_factory=list)
    mapped_component_ids: list[str] = field(default_factory=list)
    estimated_max_parallel_requests: int = 0
    max_parallel_requests_upper_bound: int | None = None
    requires_volume_reload: bool = False
    volume_reload_marker: str | None = None
    uploaded_volume_paths: list[str] = field(default_factory=list)
    rewritten_node_id_map: dict[str, str] = field(default_factory=dict)
    sandwiched_local_node_ids: list[str] = field(default_factory=list)
    parallel_local_branch_node_ids: list[str] = field(default_factory=list)
    synced_assets: list[SyncedAsset] = field(default_factory=list)
    custom_nodes_bundle: SyncedAsset | None = None
    artifact_finalizer_node_id: str | None = None
    execution_assignments_by_representative: dict[str, ExecutionAssignment] = field(
        default_factory=dict
    )
    execution_worker_indices_by_representative: dict[str, int] = field(
        default_factory=dict
    )
    custom_nodes_bundles_by_environment: dict[str, SyncedAsset | None] = field(
        default_factory=dict
    )
    remote_configurations: list[dict[str, Any]] = field(default_factory=list)
    execution_locations_by_environment: dict[str, str] = field(
        default_factory=dict
    )


@dataclass
class ComponentExecutionPlan:
    """Hold assignments and prepared provider state for one prompt rewrite."""

    assignments: dict[str, ExecutionAssignment]
    configuration_set: RemoteConfigurationSet | None = None
    configurations_by_id: dict[str, RemoteConfiguration] = field(
        default_factory=dict
    )
    safe_configurations: list[dict[str, Any]] = field(default_factory=list)
    ssh_hosts_by_id: dict[str, SshHostConfig] = field(default_factory=dict)
    vast_service: VastService | None = None
    vast_leases_by_environment: dict[str, Any] = field(default_factory=dict)
    resolved_llm_profiles: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _EnvironmentAssetPreparationResult:
    """Collect custom-node and prompt-asset results for one environment."""

    environment_id: str
    custom_nodes_bundle: SyncedAsset | None
    component_prompts: dict[str, dict[str, Any]]
    assets_by_component_id: dict[str, list[SyncedAsset]]
    asset_manifest_id: str | None = None


@dataclass(frozen=True)
class RemoteExpansionReason:
    """Describe why one upstream node had to join a remote component."""

    node_id: str
    class_type: str
    required_by_node_id: str
    required_by_class_type: str
    output_index: int
    io_type: str


@dataclass
class RemoteNodeAnalysis:
    """Structured dry-run result for context-menu remote expansion."""

    requested_node_ids: list[str] = field(default_factory=list)
    requested_workflow_node_paths: list[str] = field(default_factory=list)
    current_remote_node_ids: list[str] = field(default_factory=list)
    current_remote_workflow_node_paths: list[str] = field(default_factory=list)
    resolved_remote_node_ids: list[str] = field(default_factory=list)
    resolved_workflow_node_paths: list[str] = field(default_factory=list)
    added_node_ids: list[str] = field(default_factory=list)
    added_workflow_node_paths: list[str] = field(default_factory=list)
    sandwiched_local_node_ids: list[str] = field(default_factory=list)
    reasons: list[RemoteExpansionReason] = field(default_factory=list)


@dataclass(frozen=True)
class PromptGraphLink:
    """Describe one direct prompt dependency edge."""

    source_node_id: str
    source_output_index: int
    target_node_id: str
    target_input_name: str
    source_class_type: str
    target_class_type: str


class ModalPromptValidationError(ValueError):
    """Raised when a prompt cannot be executed with the current Modal transport."""
