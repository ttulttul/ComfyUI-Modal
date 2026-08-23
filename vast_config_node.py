"""Disconnected ComfyUI v3 node for workflow-scoped Vast.ai lease profiles."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from comfy_api.latest import _io as io

if __package__:
    from .vast_models import (
        VAST_CONFIG_NODE_ID,
        VAST_DEFAULT_IDLE_RETENTION_HOURS,
        VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS,
        VastResourceProfile,
    )
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from vast_models import (
        VAST_CONFIG_NODE_ID,
        VAST_DEFAULT_IDLE_RETENTION_HOURS,
        VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS,
        VastResourceProfile,
    )

logger = logging.getLogger(__name__)


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
                "Configure a Vast.ai capacity pool for this workflow. This node remains "
                "disconnected; remote placement detects it automatically at queue time."
            ),
            inputs=[
                io.String.Input(
                    "profile_name",
                    default="vast-default",
                    tooltip="Unique workflow-local name for this Vast capacity pool.",
                ),
                io.Int.Input(
                    "gpu_count",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="Exact number of GPUs to rent.",
                ),
                io.Float.Input(
                    "minimum_gpu_vram_gb",
                    default=24.0,
                    min=0.0,
                    max=1024.0,
                    step=1.0,
                    tooltip="Minimum nameplate VRAM required on each GPU.",
                ),
                io.Float.Input(
                    "minimum_total_tflops",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=1.0,
                    tooltip=(
                        "Minimum theoretical aggregate TFLOPS. Leave zero to avoid "
                        "filtering on theoretical compute."
                    ),
                ),
                io.Float.Input(
                    "minimum_cpu_ram_gb",
                    default=64.0,
                    min=1.0,
                    max=4096.0,
                    step=1.0,
                    tooltip="Minimum system RAM for the rented instance.",
                ),
                io.Float.Input(
                    "allocated_disk_gb",
                    default=200.0,
                    min=8.0,
                    max=10000.0,
                    step=1.0,
                    tooltip=(
                        "Instance disk allocation. Vast does not allow resizing it after "
                        "creation."
                    ),
                ),
                io.Float.Input(
                    "maximum_hourly_cost_usd",
                    default=1.0,
                    min=0.001,
                    max=128.0,
                    step=0.01,
                    tooltip="Hard maximum total hourly offer price in USD.",
                ),
                io.Float.Input(
                    "idle_retention_hours",
                    default=VAST_DEFAULT_IDLE_RETENTION_HOURS,
                    min=0.0,
                    max=24.0 * 365.0,
                    step=1.0,
                    tooltip=(
                        "Keep the running instance after its last activity for this many "
                        "hours, then destroy it. Vast bills during this period."
                    ),
                ),
                io.Float.Input(
                    "minimum_cpu_cores",
                    default=8.0,
                    min=1.0,
                    max=1024.0,
                    step=1.0,
                    advanced=True,
                    tooltip="Minimum effective CPU core allocation.",
                ),
                io.Float.Input(
                    "minimum_dlperf",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=1.0,
                    advanced=True,
                    tooltip="Optional Vast DLPerf floor; zero disables this filter.",
                ),
                io.Float.Input(
                    "minimum_download_mb_per_second",
                    default=100.0,
                    min=0.0,
                    max=100000.0,
                    step=10.0,
                    advanced=True,
                    tooltip="Minimum advertised internet download rate in MB/s.",
                ),
                io.Float.Input(
                    "minimum_reliability",
                    default=0.99,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip="Minimum Vast host reliability score.",
                ),
                io.Float.Input(
                    "minimum_offer_duration_days",
                    default=VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS,
                    min=0.0,
                    max=365.0,
                    step=1.0,
                    advanced=True,
                    tooltip=(
                        "Require the offer to remain available for at least this long. "
                        "This is separate from idle retention."
                    ),
                ),
                io.Boolean.Input(
                    "verified_hosts_only",
                    default=True,
                    advanced=True,
                    tooltip="Exclude unverified Vast marketplace hosts.",
                ),
                io.String.Input(
                    "allowed_geolocations",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Optional comma-separated Vast locations or country codes, such "
                        "as US, CA. Blank allows all locations."
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
            outputs=[],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **inputs: Any) -> io.NodeOutput:
        """Validate the profile while producing no runtime output."""
        unique_id = str(inputs.pop("unique_id", "vast-config"))
        profile_from_inputs(unique_id, inputs)
        return io.NodeOutput()


def profile_from_inputs(
    profile_id: str,
    inputs: Mapping[str, Any],
) -> VastResourceProfile:
    """Build one validated Vast resource profile from queued node inputs."""
    locations = tuple(
        location.strip()
        for location in str(inputs.get("allowed_geolocations") or "").split(",")
        if location.strip()
    )
    return VastResourceProfile(
        profile_id=str(profile_id).strip(),
        profile_name=str(inputs.get("profile_name") or "vast-default").strip(),
        gpu_count=int(inputs.get("gpu_count", 1)),
        minimum_gpu_ram_mb=_gb_to_mb(inputs.get("minimum_gpu_vram_gb", 24.0)),
        minimum_total_flops=float(inputs.get("minimum_total_tflops", 0.0)),
        minimum_cpu_ram_mb=_gb_to_mb(inputs.get("minimum_cpu_ram_gb", 64.0)),
        minimum_cpu_cores=float(inputs.get("minimum_cpu_cores", 8.0)),
        allocated_disk_gb=float(inputs.get("allocated_disk_gb", 200.0)),
        maximum_hourly_cost_usd=float(
            inputs.get("maximum_hourly_cost_usd", 1.0)
        ),
        idle_retention_seconds=float(
            inputs.get(
                "idle_retention_hours", VAST_DEFAULT_IDLE_RETENTION_HOURS
            )
        )
        * 3600.0,
        minimum_offer_duration_seconds=float(
            inputs.get(
                "minimum_offer_duration_days",
                VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS,
            )
        )
        * 86400.0,
        minimum_reliability=float(inputs.get("minimum_reliability", 0.99)),
        minimum_dlperf=float(inputs.get("minimum_dlperf", 0.0)),
        minimum_download_mb_per_second=float(
            inputs.get("minimum_download_mb_per_second", 100.0)
        ),
        verified_only=bool(inputs.get("verified_hosts_only", True)),
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


def _gb_to_mb(value: Any) -> int:
    """Convert a non-negative GiB-like widget value to Vast's integer MB unit."""
    normalized = float(value)
    if normalized < 0:
        raise ValueError("Vast memory values must not be negative.")
    return int(round(normalized * 1024))


__all__ = [
    "VastAILeaseConfiguration",
    "extract_vast_profiles",
    "profile_from_inputs",
]
