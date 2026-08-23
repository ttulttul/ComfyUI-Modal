"""Tests for the disconnected Vast.ai workflow configuration node."""

from __future__ import annotations

from typing import Any

import pytest


def _inputs(**overrides: Any) -> dict[str, Any]:
    """Return representative queued Vast configuration widget values."""
    values = {
        "profile_name": "large-video",
        "gpu_count": 1,
        "minimum_gpu_vram_gb": 48.0,
        "minimum_total_tflops": 90.0,
        "minimum_cpu_ram_gb": 96.0,
        "allocated_disk_gb": 300.0,
        "maximum_hourly_cost_usd": 1.25,
        "idle_retention_hours": 24.0,
        "minimum_cpu_cores": 12.0,
        "minimum_dlperf": 40.0,
        "minimum_download_mb_per_second": 200.0,
        "minimum_reliability": 0.995,
        "minimum_offer_duration_days": 14.0,
        "verified_hosts_only": True,
        "allowed_geolocations": "US, CA",
        "maximum_instances": 2,
    }
    values.update(overrides)
    return values


def test_configuration_node_is_disconnected_output_sink(
    vast_config_node_module: Any,
) -> None:
    """The node should survive prompt compilation without graph connections."""
    schema = vast_config_node_module.VastAILeaseConfiguration.define_schema()

    assert schema.node_id == "VastAILeaseConfiguration"
    assert schema.is_output_node is True
    assert schema.outputs == []


def test_extension_exports_configuration_node(
    extension_package: Any,
    vast_config_node_module: Any,
) -> None:
    """Register the Vast configuration through the ComfyUI v3 entrypoint."""
    assert (
        extension_package.VastAILeaseConfiguration
        is vast_config_node_module.VastAILeaseConfiguration
    )


def test_extracts_multiple_distinct_workflow_profiles(
    vast_config_node_module: Any,
) -> None:
    """Every disconnected configuration should become a scheduler candidate."""
    prompt = {
        "17": {"class_type": "VastAILeaseConfiguration", "inputs": _inputs()},
        "29": {
            "class_type": "VastAILeaseConfiguration",
            "inputs": _inputs(profile_name="h100", minimum_gpu_vram_gb=80.0),
        },
        "31": {"class_type": "KSampler", "inputs": {}},
    }

    profiles = vast_config_node_module.extract_vast_profiles(prompt)

    assert [profile.profile_name for profile in profiles] == ["large-video", "h100"]
    assert profiles[0].minimum_gpu_ram_mb == 48 * 1024
    assert profiles[0].minimum_cpu_ram_mb == 96 * 1024
    assert profiles[0].idle_retention_seconds == 24 * 3600
    assert profiles[1].minimum_gpu_ram_mb == 80 * 1024


def test_duplicate_profile_names_fail_before_rental(
    vast_config_node_module: Any,
) -> None:
    """Ambiguous optional affinity names should never reach the API."""
    prompt = {
        "1": {"class_type": "VastAILeaseConfiguration", "inputs": _inputs()},
        "2": {"class_type": "VastAILeaseConfiguration", "inputs": _inputs()},
    }

    with pytest.raises(ValueError, match="appears more than once"):
        vast_config_node_module.extract_vast_profiles(prompt)
