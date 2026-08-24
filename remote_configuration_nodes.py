"""ComfyUI v3 nodes for workflow-scoped remote execution configuration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from comfy_api.latest import _io as io

if __package__:
    from .remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        SshRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from .remote_hosts import SshHostConfig
    from .settings import MODAL_GPU_TYPES, normalize_modal_gpu_selection
    from .vast_config_node import VastAILeaseConfiguration, profile_from_inputs
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from remote_configurations import (
        ModalRemoteConfiguration,
        RemoteConfiguration,
        RemoteConfigurationSet,
        SshRemoteConfiguration,
        VastRemoteConfiguration,
    )
    from remote_hosts import SshHostConfig
    from settings import MODAL_GPU_TYPES, normalize_modal_gpu_selection
    from vast_config_node import VastAILeaseConfiguration, profile_from_inputs

logger = logging.getLogger(__name__)

REMOTE_CONFIGURATION_IO_TYPE = "REMOTE_CONFIGURATION"
REMOTE_CONFIGURATION_SET_IO_TYPE = "REMOTE_CONFIGURATION_SET"
REMOTE_EXECUTION_CONFIGURATOR_NODE_ID = "RemoteExecutionConfigurator"
MODAL_REMOTE_CONFIGURATION_NODE_ID = "ModalRemoteConfiguration"
VAST_REMOTE_CONFIGURATION_NODE_ID = "VastRemoteConfiguration"
SSH_REMOTE_CONFIGURATION_NODE_ID = "SshRemoteConfiguration"
REMOTE_CONFIGURATION_INPUT_GROUP = "configurations"
REMOTE_CONFIGURATION_INPUT_PREFIX = "configuration_"
REMOTE_CONFIGURATION_MAX_INPUTS = 32

REMOTE_CONFIGURATION_NODE_IDS = frozenset(
    {
        REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
        MODAL_REMOTE_CONFIGURATION_NODE_ID,
        VAST_REMOTE_CONFIGURATION_NODE_ID,
        SSH_REMOTE_CONFIGURATION_NODE_ID,
    }
)

RemoteConfigurationType = io.Custom(REMOTE_CONFIGURATION_IO_TYPE)
RemoteConfigurationSetType = io.Custom(REMOTE_CONFIGURATION_SET_IO_TYPE)


class ModalConfiguration(io.ComfyNode):
    """Declare one workflow-scoped Modal capacity pool."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose a Modal GPU type and concurrent instance limit."""
        return io.Schema(
            node_id=MODAL_REMOTE_CONFIGURATION_NODE_ID,
            display_name="Modal Configuration",
            category="Remote Execution/Configuration",
            description=(
                "Contribute one Modal GPU type and capacity limit to a Remote "
                "Execution Configurator."
            ),
            inputs=[
                io.String.Input(
                    "configuration_name",
                    default="modal-default",
                    tooltip="Unique workflow-local name for this Modal capacity pool.",
                ),
                io.Combo.Input(
                    "gpu_type",
                    options=list(MODAL_GPU_TYPES),
                    default="RTX-PRO-6000",
                    tooltip="Modal GPU type used by this capacity pool.",
                ),
                io.Int.Input(
                    "instance_count",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Maximum concurrent Modal containers for this pool.",
                ),
            ],
            outputs=[RemoteConfigurationType.Output()],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **inputs: Any) -> io.NodeOutput:
        """Build one validated Modal configuration value."""
        configuration_id = _hidden_unique_id(cls) or str(
            inputs.pop("unique_id", "modal-configuration")
        )
        return io.NodeOutput(modal_configuration_from_inputs(configuration_id, inputs))


class VastConfiguration(io.ComfyNode):
    """Declare one workflow-scoped Vast.ai marketplace capacity pool."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Reuse the established Vast.ai marketplace controls with a typed output."""
        legacy_schema = VastAILeaseConfiguration.define_schema()
        return io.Schema(
            node_id=VAST_REMOTE_CONFIGURATION_NODE_ID,
            display_name="Vast.ai Configuration",
            category="Remote Execution/Configuration",
            description=(
                "Contribute one Vast.ai marketplace search and managed-instance "
                "limit to a Remote Execution Configurator."
            ),
            inputs=list(legacy_schema.inputs),
            outputs=[RemoteConfigurationType.Output()],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **inputs: Any) -> io.NodeOutput:
        """Build one validated Vast.ai configuration value."""
        configuration_id = _hidden_unique_id(cls) or str(
            inputs.pop("unique_id", "vast-configuration")
        )
        return io.NodeOutput(vast_configuration_from_inputs(configuration_id, inputs))


class SshConfiguration(io.ComfyNode):
    """Declare one workflow-scoped SSH Docker execution host."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose portable, credential-free host and scheduling controls."""
        return io.Schema(
            node_id=SSH_REMOTE_CONFIGURATION_NODE_ID,
            display_name="SSH Configuration",
            category="Remote Execution/Configuration",
            description=(
                "Contribute one SSH-accessible Docker host to a Remote Execution "
                "Configurator. Authentication remains in the local SSH agent/config."
            ),
            inputs=[
                io.String.Input(
                    "environment_id",
                    default="ssh-host",
                    tooltip="Stable lowercase workflow-local environment identity.",
                ),
                io.String.Input(
                    "display_name",
                    default="SSH host",
                    tooltip="Human-readable name used in plans and errors.",
                ),
                io.String.Input(
                    "ssh_target",
                    default="",
                    tooltip=(
                        "OpenSSH destination or alias that works "
                        "non-interactively."
                    ),
                ),
                io.String.Input(
                    "cost_usd_per_hour",
                    default="Unknown",
                    tooltip=(
                        "Marginal hourly cost used by cost-aware planning. Use "
                        "Unknown when no trustworthy price is available."
                    ),
                ),
                io.Int.Input(
                    "maximum_workers",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Maximum worker containers addressed on this host.",
                ),
                io.Float.Input(
                    "reserve_vram_gb",
                    default=0.0,
                    min=0.0,
                    max=4096.0,
                    step=0.1,
                    advanced=True,
                    tooltip="GPU VRAM withheld from scheduler admission checks.",
                ),
                io.String.Input(
                    "tags",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Comma-separated placement tags.",
                ),
                io.String.Input(
                    "docker_env_file",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Optional absolute env-file path already present on the host."
                    ),
                ),
            ],
            outputs=[RemoteConfigurationType.Output()],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **inputs: Any) -> io.NodeOutput:
        """Build one validated SSH host configuration value."""
        configuration_id = _hidden_unique_id(cls) or str(
            inputs.pop("unique_id", inputs.get("environment_id", "ssh-host"))
        )
        return io.NodeOutput(ssh_configuration_from_inputs(configuration_id, inputs))


class RemoteExecutionConfigurator(io.ComfyNode):
    """Anchor and validate the complete remote capacity declaration."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Accept a variable number of typed remote configurations."""
        configuration_template = io.Autogrow.TemplatePrefix(
            input=RemoteConfigurationType.Input("configuration"),
            prefix=REMOTE_CONFIGURATION_INPUT_PREFIX,
            min=1,
            max=REMOTE_CONFIGURATION_MAX_INPUTS,
        )
        return io.Schema(
            node_id=REMOTE_EXECUTION_CONFIGURATOR_NODE_ID,
            display_name="Remote Execution Configurator",
            category="Remote Execution/Configuration",
            description=(
                "Collect the Modal, Vast.ai, and SSH capacity pools available to "
                "this workflow's remote execution planner."
            ),
            inputs=[
                io.Autogrow.Input(
                    REMOTE_CONFIGURATION_INPUT_GROUP,
                    template=configuration_template,
                )
            ],
            outputs=[RemoteConfigurationSetType.Output()],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, configurations: io.Autogrow.Type) -> io.NodeOutput:
        """Assemble ordered configuration values into one validated set."""
        return io.NodeOutput(configuration_set_from_values(configurations))


def modal_configuration_from_inputs(
    configuration_id: str,
    inputs: Mapping[str, Any],
) -> ModalRemoteConfiguration:
    """Build one Modal configuration from queued widget inputs."""
    return ModalRemoteConfiguration(
        configuration_id=str(configuration_id).strip(),
        display_name=str(inputs.get("configuration_name") or "modal-default").strip(),
        gpu_type=normalize_modal_gpu_selection(inputs.get("gpu_type", "RTX-PRO-6000")),
        instance_count=int(inputs.get("instance_count", 1)),
    )


def vast_configuration_from_inputs(
    configuration_id: str,
    inputs: Mapping[str, Any],
) -> VastRemoteConfiguration:
    """Build one Vast configuration from queued widget inputs."""
    profile = profile_from_inputs(str(configuration_id).strip(), inputs)
    return VastRemoteConfiguration(
        configuration_id=profile.profile_id,
        display_name=profile.profile_name,
        profile=profile,
    )


def ssh_configuration_from_inputs(
    configuration_id: str,
    inputs: Mapping[str, Any],
) -> SshRemoteConfiguration:
    """Build one SSH configuration from queued widget inputs."""
    environment_id = str(inputs.get("environment_id") or configuration_id).strip()
    raw_tags = str(inputs.get("tags") or "")
    host = SshHostConfig(
        environment_id=environment_id,
        display_name=str(inputs.get("display_name") or environment_id).strip(),
        ssh_target=str(inputs.get("ssh_target") or "").strip(),
        cost_usd_per_second=_optional_hourly_cost_per_second(
            inputs.get("cost_usd_per_hour")
        ),
        maximum_workers=int(inputs.get("maximum_workers", 1)),
        reserve_vram_bytes=round(float(inputs.get("reserve_vram_gb", 0.0)) * 1024**3),
        tags=frozenset(tag.strip() for tag in raw_tags.split(",") if tag.strip()),
        docker_env_file=(
            str(inputs["docker_env_file"]).strip()
            if inputs.get("docker_env_file")
            else None
        ),
    )
    return SshRemoteConfiguration(
        configuration_id=host.environment_id,
        display_name=host.display_name,
        host=host,
    )


def configuration_set_from_values(
    values: Mapping[str, Any],
) -> RemoteConfigurationSet:
    """Validate ordered autogrow values as remote configurations."""
    configurations: list[RemoteConfiguration] = []
    for input_name, value in sorted(
        values.items(),
        key=lambda item: _configuration_input_sort_key(str(item[0])),
    ):
        if not isinstance(value, RemoteConfiguration):
            raise TypeError(
                f"Remote configurator input {input_name!r} did not receive a "
                "RemoteConfiguration value."
            )
        configurations.append(value)
    return RemoteConfigurationSet(tuple(configurations))


def compile_remote_configuration_set(
    prompt: Mapping[str, Any],
) -> RemoteConfigurationSet | None:
    """Compile the connected configurator branch before ComfyUI node execution."""
    configurator_nodes = [
        (str(node_id), prompt_node)
        for node_id, prompt_node in prompt.items()
        if isinstance(prompt_node, Mapping)
        and str(prompt_node.get("class_type") or "")
        == REMOTE_EXECUTION_CONFIGURATOR_NODE_ID
    ]
    if not configurator_nodes:
        return None
    if len(configurator_nodes) > 1:
        raise ValueError(
            "A workflow may contain only one Remote Execution Configurator."
        )

    configurator_id, configurator_node = configurator_nodes[0]
    raw_inputs = configurator_node.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError(
            f"Remote Execution Configurator {configurator_id!r} has invalid inputs."
        )
    connected_inputs = _serialized_configuration_inputs(raw_inputs)
    if not connected_inputs:
        raise ValueError(
            "Remote Execution Configurator requires at least one connected "
            "REMOTE_CONFIGURATION input."
        )

    configurations: list[RemoteConfiguration] = []
    linked_node_ids: set[str] = set()
    for input_name, input_value in connected_inputs:
        source_node_id = _configuration_link_source(input_name, input_value)
        if source_node_id in linked_node_ids:
            raise ValueError(
                f"Remote configuration node {source_node_id!r} is connected "
                "more than once."
            )
        linked_node_ids.add(source_node_id)
        source_node = prompt.get(source_node_id)
        if not isinstance(source_node, Mapping):
            raise ValueError(
                f"Remote configurator input {input_name!r} references missing node "
                f"{source_node_id!r}."
            )
        source_inputs = source_node.get("inputs")
        if not isinstance(source_inputs, Mapping):
            raise ValueError(
                f"Remote configuration node {source_node_id!r} has invalid inputs."
            )
        configurations.append(
            _configuration_from_prompt_node(
                source_node_id,
                str(source_node.get("class_type") or ""),
                source_inputs,
            )
        )
    return RemoteConfigurationSet(tuple(configurations))


def _serialized_configuration_inputs(
    raw_inputs: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    """Return Autogrow links from serialized or reconstructed input shapes."""
    flattened_prefix = (
        f"{REMOTE_CONFIGURATION_INPUT_GROUP}."
        f"{REMOTE_CONFIGURATION_INPUT_PREFIX}"
    )
    candidates: list[tuple[str, str, Any]] = []
    for raw_name, value in raw_inputs.items():
        input_name = str(raw_name)
        if input_name.startswith(flattened_prefix):
            leaf_name = input_name.removeprefix(
                f"{REMOTE_CONFIGURATION_INPUT_GROUP}."
            )
            candidates.append((leaf_name, input_name, value))
            continue
        if input_name.startswith(REMOTE_CONFIGURATION_INPUT_PREFIX):
            candidates.append((input_name, input_name, value))
            continue
        if (
            input_name == REMOTE_CONFIGURATION_INPUT_GROUP
            and isinstance(value, Mapping)
        ):
            for nested_name, nested_value in value.items():
                leaf_name = str(nested_name)
                if not leaf_name.startswith(REMOTE_CONFIGURATION_INPUT_PREFIX):
                    continue
                candidates.append(
                    (
                        leaf_name,
                        f"{REMOTE_CONFIGURATION_INPUT_GROUP}.{leaf_name}",
                        nested_value,
                    )
                )

    leaf_names = [leaf_name for leaf_name, _, _ in candidates]
    if len(leaf_names) != len(set(leaf_names)):
        raise ValueError(
            "Remote Execution Configurator contains duplicate Autogrow input slots."
        )
    return [
        (input_name, value)
        for _, input_name, value in sorted(
            candidates,
            key=lambda item: _configuration_input_sort_key(item[0]),
        )
    ]


def _configuration_link_source(input_name: str, value: Any) -> str:
    """Return the source node ID from one typed configurator link."""
    if (
        not isinstance(value, list)
        or len(value) != 2
        or isinstance(value[1], bool)
        or int(value[1]) != 0
    ):
        raise ValueError(
            f"Remote configurator input {input_name!r} must connect to output 0 "
            "of a configuration node."
        )
    return str(value[0])


def _configuration_from_prompt_node(
    node_id: str,
    class_type: str,
    inputs: Mapping[str, Any],
) -> RemoteConfiguration:
    """Build the typed configuration represented by one queued prompt node."""
    if class_type == MODAL_REMOTE_CONFIGURATION_NODE_ID:
        return modal_configuration_from_inputs(node_id, inputs)
    if class_type == VAST_REMOTE_CONFIGURATION_NODE_ID:
        return vast_configuration_from_inputs(node_id, inputs)
    if class_type == SSH_REMOTE_CONFIGURATION_NODE_ID:
        return ssh_configuration_from_inputs(node_id, inputs)
    raise ValueError(
        f"Remote configurator source node {node_id!r} has unsupported type "
        f"{class_type!r}."
    )


def _configuration_input_sort_key(input_name: str) -> tuple[int, str]:
    """Return numeric autogrow ordering with a stable fallback."""
    suffix = input_name.removeprefix(REMOTE_CONFIGURATION_INPUT_PREFIX)
    try:
        return (int(suffix), input_name)
    except ValueError:
        return (REMOTE_CONFIGURATION_MAX_INPUTS + 1, input_name)


def _optional_hourly_cost_per_second(value: Any) -> float | None:
    """Convert an optional hourly workflow price to a per-second scheduler rate."""
    normalized = str(value if value is not None else "").strip()
    if not normalized or normalized.casefold() in {"unknown", "any"}:
        return None
    try:
        hourly_cost = float(normalized)
    except ValueError as exc:
        raise ValueError(
            "cost_usd_per_hour must be Unknown or a non-negative number."
        ) from exc
    if hourly_cost < 0:
        raise ValueError("cost_usd_per_hour must not be negative.")
    return hourly_cost / 3600.0


def _hidden_unique_id(node_class: type[io.ComfyNode]) -> str | None:
    """Return the current v3 hidden unique ID when ComfyUI supplied one."""
    hidden = getattr(node_class, "hidden", None)
    unique_id = getattr(hidden, "unique_id", None)
    if unique_id is None:
        return None
    return str(unique_id)


__all__ = [
    "MODAL_REMOTE_CONFIGURATION_NODE_ID",
    "ModalConfiguration",
    "REMOTE_CONFIGURATION_IO_TYPE",
    "REMOTE_CONFIGURATION_INPUT_GROUP",
    "REMOTE_CONFIGURATION_NODE_IDS",
    "REMOTE_CONFIGURATION_SET_IO_TYPE",
    "REMOTE_EXECUTION_CONFIGURATOR_NODE_ID",
    "RemoteExecutionConfigurator",
    "SSH_REMOTE_CONFIGURATION_NODE_ID",
    "SshConfiguration",
    "VAST_REMOTE_CONFIGURATION_NODE_ID",
    "VastConfiguration",
    "compile_remote_configuration_set",
    "configuration_set_from_values",
    "modal_configuration_from_inputs",
    "ssh_configuration_from_inputs",
    "vast_configuration_from_inputs",
]
