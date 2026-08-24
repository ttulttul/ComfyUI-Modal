"""Disconnected ComfyUI v3 node for workflow-scoped Vast.ai lease profiles."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

from comfy_api.latest import _io as io

if __package__:
    from .vast_models import (
        VAST_CONFIG_NODE_ID,
        VAST_DEFAULT_IDLE_RETENTION_HOURS,
        VastResourceProfile,
    )
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from vast_models import (
        VAST_CONFIG_NODE_ID,
        VAST_DEFAULT_IDLE_RETENTION_HOURS,
        VastResourceProfile,
    )

logger = logging.getLogger(__name__)
VAST_ANY_VALUE = "Any"
_VAST_VERIFIED_ONLY_VALUE = "Verified only"


@dataclass(frozen=True)
class VastSelection:
    """Describe one distinct GPU selection rendered by the configuration node."""

    gpu_name: str
    gpu_count: int
    gpu_ram_mb: int
    hourly_cost_usd: float
    instance_count: int
    component_count: int


class VastAILeaseConfiguration(io.ComfyNode):
    """Declare one disconnected Vast.ai resource and retention profile."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose portable non-secret capacity and spending controls."""
        return io.Schema(
            node_id=VAST_CONFIG_NODE_ID,
            display_name="Vast.ai Lease Configuration",
            category="Remote Execution/Vast.ai",
            description=(
                "Configure a Vast.ai capacity pool for this workflow. This node "
                "remains disconnected; remote placement detects it automatically "
                "at queue time."
            ),
            inputs=[
                io.String.Input(
                    "profile_name",
                    default="vast-default",
                    tooltip="Unique workflow-local name for this Vast capacity pool.",
                ),
                io.String.Input(
                    "gpu_count",
                    default=VAST_ANY_VALUE,
                    tooltip="Use Any or enter the exact number of GPUs to rent.",
                ),
                io.String.Input(
                    "minimum_gpu_vram_gb",
                    default=VAST_ANY_VALUE,
                    tooltip=(
                        "Use Any to rely on the workflow memory estimate, or enter the "
                        "minimum VRAM in GiB for each GPU."
                    ),
                ),
                io.String.Input(
                    "minimum_total_tflops",
                    default=VAST_ANY_VALUE,
                    tooltip=(
                        "Use Any or enter minimum theoretical aggregate TFLOPS. Vast's "
                        "cross-architecture TFLOPS values are not directly comparable."
                    ),
                ),
                io.String.Input(
                    "minimum_cpu_ram_gb",
                    default=VAST_ANY_VALUE,
                    tooltip=(
                        "Use Any to rely on the workflow memory estimate, or enter the "
                        "minimum system RAM in GiB."
                    ),
                ),
                io.Float.Input(
                    "allocated_disk_gb",
                    default=200.0,
                    min=8.0,
                    max=10000.0,
                    step=1.0,
                    tooltip=(
                        "Instance disk allocation. Vast does not allow resizing it "
                        "after creation."
                    ),
                ),
                io.String.Input(
                    "maximum_hourly_cost_usd",
                    default=VAST_ANY_VALUE,
                    tooltip=(
                        "Use Any for no hard hourly price ceiling, or enter a positive "
                        "USD limit. Any may permit an expensive Vast-only rental."
                    ),
                ),
                io.Float.Input(
                    "idle_retention_hours",
                    default=VAST_DEFAULT_IDLE_RETENTION_HOURS,
                    min=0.0,
                    max=24.0 * 365.0,
                    step=1.0,
                    tooltip=(
                        "Keep the running instance after its last activity for this "
                        "many hours, then destroy it. Vast bills during this period."
                    ),
                ),
                io.String.Input(
                    "minimum_cpu_cores",
                    default=VAST_ANY_VALUE,
                    advanced=True,
                    tooltip=(
                        "Use Any or enter the minimum effective CPU core allocation."
                    ),
                ),
                io.String.Input(
                    "minimum_dlperf",
                    default=VAST_ANY_VALUE,
                    advanced=True,
                    tooltip="Use Any or enter an optional Vast DLPerf floor.",
                ),
                io.String.Input(
                    "minimum_download_mb_per_second",
                    default=VAST_ANY_VALUE,
                    advanced=True,
                    tooltip=(
                        "Use Any or enter the minimum advertised internet download "
                        "rate in MB/s."
                    ),
                ),
                io.String.Input(
                    "minimum_reliability",
                    default=VAST_ANY_VALUE,
                    advanced=True,
                    tooltip=(
                        "Use Any or enter the minimum Vast host reliability score "
                        "between 0 and 1."
                    ),
                ),
                io.String.Input(
                    "minimum_offer_duration_days",
                    default=VAST_ANY_VALUE,
                    advanced=True,
                    tooltip=(
                        "Use Any or require the offer to remain available for at least "
                        "this many days. This is separate from idle retention."
                    ),
                ),
                io.Combo.Input(
                    "verified_hosts_only",
                    options=[VAST_ANY_VALUE, _VAST_VERIFIED_ONLY_VALUE],
                    default=VAST_ANY_VALUE,
                    advanced=True,
                    tooltip="Use Any or exclude unverified Vast marketplace hosts.",
                ),
                io.String.Input(
                    "allowed_geolocations",
                    default=VAST_ANY_VALUE,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Use Any or enter comma-separated Vast locations or country "
                        "codes, such as US, CA."
                    ),
                ),
                io.Int.Input(
                    "maximum_instances",
                    default=1,
                    min=1,
                    max=16,
                    step=1,
                    advanced=True,
                    tooltip="Maximum simultaneously managed instances for this pool.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="STRING",
                    tooltip=(
                        "Markdown describing the Vast.ai GPU type or types selected "
                        "for this profile at queue time."
                    ),
                )
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **inputs: Any) -> io.NodeOutput:
        """Validate the profile and describe its actual queue-time Vast selection."""
        unique_id = _hidden_unique_id(cls) or str(
            inputs.pop("unique_id", "vast-config")
        )
        profile = profile_from_inputs(unique_id, inputs)
        return io.NodeOutput(
            vast_selection_markdown(
                _hidden_prompt(cls),
                profile_id=profile.profile_id,
                profile_name=profile.profile_name,
            )
        )

    @classmethod
    def fingerprint_inputs(cls, **inputs: Any) -> str:
        """Invalidate cached Markdown whenever queue-time placement changes."""
        unique_id = _hidden_unique_id(cls) or str(
            inputs.get("unique_id", "vast-config")
        )
        profile = profile_from_inputs(unique_id, inputs)
        return vast_selection_markdown(
            _hidden_prompt(cls),
            profile_id=profile.profile_id,
            profile_name=profile.profile_name,
        )


def profile_from_inputs(
    profile_id: str,
    inputs: Mapping[str, Any],
) -> VastResourceProfile:
    """Build one validated Vast resource profile from queued node inputs."""
    locations = tuple(
        location.strip()
        for location in str(inputs.get("allowed_geolocations") or "").split(",")
        if location.strip() and not _is_any(location)
    )
    return VastResourceProfile(
        profile_id=str(profile_id).strip(),
        profile_name=str(inputs.get("profile_name") or "vast-default").strip(),
        gpu_count=_optional_positive_int(inputs.get("gpu_count"), "gpu_count"),
        minimum_gpu_ram_mb=_optional_gb_to_mb(
            inputs.get("minimum_gpu_vram_gb"),
            "minimum_gpu_vram_gb",
        ),
        minimum_total_flops=_optional_non_negative_float(
            inputs.get("minimum_total_tflops"),
            "minimum_total_tflops",
        ),
        minimum_cpu_ram_mb=_optional_gb_to_mb(
            inputs.get("minimum_cpu_ram_gb"),
            "minimum_cpu_ram_gb",
        ),
        minimum_cpu_cores=_optional_non_negative_float(
            inputs.get("minimum_cpu_cores"),
            "minimum_cpu_cores",
        ),
        allocated_disk_gb=float(inputs.get("allocated_disk_gb", 200.0)),
        maximum_hourly_cost_usd=_optional_positive_float(
            inputs.get("maximum_hourly_cost_usd"),
            "maximum_hourly_cost_usd",
        ),
        idle_retention_seconds=float(
            inputs.get(
                "idle_retention_hours", VAST_DEFAULT_IDLE_RETENTION_HOURS
            )
        )
        * 3600.0,
        minimum_offer_duration_seconds=_optional_days_to_seconds(
            inputs.get("minimum_offer_duration_days")
        ),
        minimum_reliability=_optional_bounded_float(
            inputs.get("minimum_reliability"),
            "minimum_reliability",
            maximum=1.0,
        ),
        minimum_dlperf=_optional_non_negative_float(
            inputs.get("minimum_dlperf"),
            "minimum_dlperf",
        ),
        minimum_download_mb_per_second=_optional_non_negative_float(
            inputs.get("minimum_download_mb_per_second"),
            "minimum_download_mb_per_second",
        ),
        verified_only=_verified_only(inputs.get("verified_hosts_only")),
        allowed_geolocations=locations,
        maximum_instances=int(inputs.get("maximum_instances", 1)),
    )


def extract_vast_profiles(prompt: Mapping[str, Any]) -> tuple[VastResourceProfile, ...]:
    """Return every disconnected Vast configuration in one executable prompt."""
    profiles: list[VastResourceProfile] = []
    names: set[str] = set()
    for node_id, prompt_node in prompt.items():
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != VAST_CONFIG_NODE_ID:
            continue
        raw_inputs = prompt_node.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise ValueError(f"Vast configuration node {node_id!r} has invalid inputs.")
        profile = profile_from_inputs(str(node_id), raw_inputs)
        normalized_name = profile.profile_name.casefold()
        if normalized_name in names:
            raise ValueError(
                f"Vast profile name {profile.profile_name!r} appears more than once."
            )
        names.add(normalized_name)
        profiles.append(profile)
    return tuple(sorted(profiles, key=lambda profile: profile.profile_id))


def vast_selection_markdown(
    prompt: Mapping[str, Any] | None,
    *,
    profile_id: str,
    profile_name: str,
) -> str:
    """Return Markdown describing distinct Vast GPU selections for one profile."""
    selections = _vast_selections(prompt, profile_id=profile_id)
    heading = f"## Vast.ai selection for `{_escape_markdown_code(profile_name)}`"
    if not selections:
        return (
            f"{heading}\n\n"
            "No Vast.ai node type was selected for this execution."
        )
    rows = [
        "| GPU type | GPUs per instance | VRAM per GPU | Hourly price | "
        "Instances | Components |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows.extend(
        (
            "| {gpu} | {gpu_count} | {vram:.3f} GiB | ${price:.3f} | "
            "{instances} | {components} |"
        ).format(
            gpu=_escape_markdown_cell(selection.gpu_name),
            gpu_count=selection.gpu_count,
            vram=selection.gpu_ram_mb / 1024.0,
            price=selection.hourly_cost_usd,
            instances=selection.instance_count,
            components=selection.component_count,
        )
        for selection in selections
    )
    return f"{heading}\n\n" + "\n".join(rows)


def _vast_selections(
    prompt: Mapping[str, Any] | None,
    *,
    profile_id: str,
) -> tuple[VastSelection, ...]:
    """Extract and aggregate safe Vast placement metadata from a rewritten prompt."""
    if not isinstance(prompt, Mapping):
        return ()
    components_by_selection: dict[
        tuple[str, int, int, float], set[str]
    ] = defaultdict(set)
    instances_by_selection: dict[
        tuple[str, int, int, float], set[int]
    ] = defaultdict(set)
    for node_id, prompt_node in prompt.items():
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        payload = inputs.get("original_node_data")
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("execution_provider") or "") != "vast":
            continue
        if str(payload.get("vast_profile_id") or "") != profile_id:
            continue
        key = (
            str(payload.get("vast_gpu_name") or "Unknown GPU"),
            int(payload.get("vast_gpu_count") or 1),
            int(payload.get("vast_gpu_ram_mb") or 0),
            float(payload.get("vast_hourly_cost_usd") or 0.0),
        )
        components_by_selection[key].add(str(payload.get("component_id") or node_id))
        instance_id = int(payload.get("vast_instance_id") or 0)
        if instance_id > 0:
            instances_by_selection[key].add(instance_id)
    return tuple(
        VastSelection(
            gpu_name=key[0],
            gpu_count=key[1],
            gpu_ram_mb=key[2],
            hourly_cost_usd=key[3],
            instance_count=max(1, len(instances_by_selection[key])),
            component_count=len(component_ids),
        )
        for key, component_ids in sorted(
            components_by_selection.items(),
            key=lambda item: (item[0][3], item[0][0]),
        )
    )


def _hidden_prompt(
    node_class: type[VastAILeaseConfiguration],
) -> Mapping[str, Any] | None:
    """Return the v3 hidden prompt when ComfyUI supplied one."""
    hidden = getattr(node_class, "hidden", None)
    prompt = getattr(hidden, "prompt", None)
    return prompt if isinstance(prompt, Mapping) else None


def _hidden_unique_id(node_class: type[VastAILeaseConfiguration]) -> str | None:
    """Return the v3 hidden node identity when ComfyUI supplied one."""
    hidden = getattr(node_class, "hidden", None)
    unique_id = str(getattr(hidden, "unique_id", "") or "").strip()
    return unique_id or None


def _is_any(value: Any) -> bool:
    """Return whether a queued selector explicitly requests no constraint."""
    return value is None or str(value).strip().casefold() in {"", "any"}


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    """Parse Any or a positive integer selector."""
    if _is_any(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be Any or a positive integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be Any or a positive integer.") from exc
    if normalized <= 0 or float(value) != normalized:
        raise ValueError(f"{field_name} must be Any or a positive integer.")
    return normalized


def _optional_gb_to_mb(value: Any, field_name: str) -> int | None:
    """Convert Any or a GiB widget value to Vast's integer memory unit."""
    normalized = _optional_non_negative_float(value, field_name)
    return None if normalized is None else int(round(normalized * 1024.0))


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    """Parse Any or a finite non-negative floating-point selector."""
    return _optional_bounded_float(value, field_name, minimum=0.0)


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    """Parse Any or a finite positive floating-point selector."""
    normalized = _optional_bounded_float(value, field_name, minimum=0.0)
    if normalized is not None and normalized <= 0:
        raise ValueError(f"{field_name} must be Any or a positive number.")
    return normalized


def _optional_bounded_float(
    value: Any,
    field_name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float | None:
    """Parse Any or one finite floating-point selector within inclusive bounds."""
    if _is_any(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be Any or a number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be Any or a number.") from exc
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{field_name} must be Any or at least {minimum}.")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field_name} must be Any or at most {maximum}.")
    return normalized


def _optional_days_to_seconds(value: Any) -> float | None:
    """Convert Any or a non-negative day count to seconds."""
    normalized = _optional_non_negative_float(
        value,
        "minimum_offer_duration_days",
    )
    return None if normalized is None else normalized * 86400.0


def _verified_only(value: Any) -> bool:
    """Parse the Any/Verified-only selector and legacy Boolean workflow values."""
    if isinstance(value, bool):
        return value
    if _is_any(value):
        return False
    if str(value).strip().casefold() == _VAST_VERIFIED_ONLY_VALUE.casefold():
        return True
    raise ValueError("verified_hosts_only must be Any or Verified only.")


def _escape_markdown_cell(value: str) -> str:
    """Escape one untrusted marketplace label for a Markdown table cell."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _escape_markdown_code(value: str) -> str:
    """Escape backticks in one short Markdown inline-code value."""
    return value.replace("`", "\\`")


__all__ = [
    "VastAILeaseConfiguration",
    "VastSelection",
    "extract_vast_profiles",
    "profile_from_inputs",
    "vast_selection_markdown",
]
