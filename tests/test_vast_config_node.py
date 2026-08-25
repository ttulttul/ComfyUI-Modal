"""Tests for the disconnected Vast.ai workflow configuration node."""

from __future__ import annotations

from types import SimpleNamespace
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
        "excluded_country_codes": "cn, RU, cn",
        "maximum_instances": 2,
    }
    values.update(overrides)
    return values


def test_configuration_node_is_disconnected_output_sink_with_markdown_output(
    vast_config_node_module: Any,
) -> None:
    """The node should survive prompt compilation without graph connections."""
    schema = vast_config_node_module.VastAILeaseConfiguration.define_schema()

    assert schema.node_id == "VastAILeaseConfiguration"
    assert schema.is_output_node is True
    assert len(schema.outputs) == 1
    assert schema.outputs[0].display_name == "STRING"


def test_configuration_node_defaults_marketplace_selectors_to_any(
    vast_config_node_module: Any,
) -> None:
    """A new node should maximize candidates until the user adds constraints."""
    schema = vast_config_node_module.VastAILeaseConfiguration.define_schema()
    defaults = {
        input_definition.id: input_definition.default
        for input_definition in schema.inputs
    }

    assert defaults == {
        "profile_name": "vast-default",
        "gpu_count": "Any",
        "minimum_gpu_vram_gb": "Any",
        "minimum_total_tflops": "Any",
        "minimum_cpu_ram_gb": "Any",
        "allocated_disk_gb": 200.0,
        "maximum_hourly_cost_usd": "Any",
        "idle_retention_hours": 24.0,
        "minimum_cpu_cores": "Any",
        "minimum_dlperf": "Any",
        "minimum_download_mb_per_second": "Any",
        "minimum_reliability": "Any",
        "minimum_offer_duration_days": "Any",
        "verified_hosts_only": "Any",
        "allowed_geolocations": "Any",
        "excluded_country_codes": "CN, RU",
        "maximum_instances": 1,
    }


def test_any_selectors_are_omitted_from_marketplace_profile(
    vast_config_node_module: Any,
) -> None:
    """Explicit Any values should not become accidental zero-value filters."""
    profile = vast_config_node_module.profile_from_inputs(
        "17",
        {
            "profile_name": "broad",
            "gpu_count": "Any",
            "minimum_gpu_vram_gb": "Any",
            "minimum_total_tflops": "Any",
            "minimum_cpu_ram_gb": "Any",
            "maximum_hourly_cost_usd": "Any",
            "minimum_cpu_cores": "Any",
            "minimum_dlperf": "Any",
            "minimum_download_mb_per_second": "Any",
            "minimum_reliability": "Any",
            "minimum_offer_duration_days": "Any",
            "verified_hosts_only": "Any",
            "allowed_geolocations": "Any",
            "excluded_country_codes": "Any",
        },
    )

    assert profile.gpu_count is None
    assert profile.minimum_gpu_ram_mb is None
    assert profile.minimum_cpu_ram_mb is None
    assert profile.maximum_hourly_cost_usd is None
    assert profile.verified_only is False
    assert profile.allowed_geolocations == ()
    assert profile.excluded_country_codes == ()
    payload = profile.search_payload()
    assert "num_gpus" not in payload
    assert "gpu_ram" not in payload
    assert "cpu_ram" not in payload
    assert "dph_total" not in payload
    assert "verified" not in payload
    assert "geolocation" not in payload


def test_country_exclusions_default_for_legacy_prompts_and_normalize_input(
    vast_config_node_module: Any,
) -> None:
    """Missing legacy inputs use the safe default; authored codes are normalized."""
    legacy_profile = vast_config_node_module.profile_from_inputs(
        "17",
        {"profile_name": "legacy"},
    )
    configured_profile = vast_config_node_module.profile_from_inputs(
        "18",
        {
            "profile_name": "configured",
            "excluded_country_codes": " cn, RU, cn ",
        },
    )

    assert legacy_profile.excluded_country_codes == ("CN", "RU")
    assert configured_profile.excluded_country_codes == ("CN", "RU")
    assert configured_profile.search_payload()["geolocation"] == {
        "notin": ["CN", "RU"]
    }


def test_country_exclusions_reject_invalid_and_conflicting_codes(
    vast_config_node_module: Any,
) -> None:
    """Malformed or contradictory workflow filters should fail before API access."""
    with pytest.raises(ValueError, match="two-letter country codes"):
        vast_config_node_module.profile_from_inputs(
            "17",
            {"profile_name": "invalid", "excluded_country_codes": "CHINA"},
        )
    with pytest.raises(ValueError, match="both allow and exclude"):
        vast_config_node_module.profile_from_inputs(
            "18",
            {
                "profile_name": "conflict",
                "allowed_geolocations": "CN",
                "excluded_country_codes": "CN, RU",
            },
        )


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


def test_selection_markdown_reports_only_this_profiles_actual_vast_types(
    vast_config_node_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The STRING output should summarize deduplicated queue-time lease choices."""
    selected_payload = {
        "execution_provider": "vast",
        "vast_profile_id": "17",
        "vast_gpu_name": "RTX PRO 6000 WS",
        "vast_gpu_count": 1,
        "vast_gpu_ram_mb": 97887,
        "vast_hourly_cost_usd": 1.25,
        "vast_instance_id": 9001,
    }
    prompt = {
        "component-a": {
            "inputs": {
                "original_node_data": {**selected_payload, "component_id": "a"}
            }
        },
        "component-b": {
            "inputs": {
                "original_node_data": {**selected_payload, "component_id": "b"}
            }
        },
        "other-profile": {
            "inputs": {
                "original_node_data": {
                    **selected_payload,
                    "vast_profile_id": "29",
                    "vast_gpu_name": "H200 NVL",
                }
            }
        },
    }
    monkeypatch.setattr(
        vast_config_node_module.VastAILeaseConfiguration,
        "hidden",
        SimpleNamespace(unique_id="17", prompt=prompt),
    )

    output = vast_config_node_module.VastAILeaseConfiguration.execute(
        **_inputs(profile_name="large-video")
    )

    markdown = output.result[0]
    assert "Vast.ai selection for `large-video`" in markdown
    assert "RTX PRO 6000 WS" in markdown
    assert "95.593 GiB" in markdown
    assert "$1.250" in markdown
    assert "| 1 | 2 |" in markdown
    assert "H200 NVL" not in markdown


def test_selection_markdown_says_when_vast_was_not_selected(
    vast_config_node_module: Any,
) -> None:
    """Automatic fallback to another provider should be explicit in the output."""
    markdown = vast_config_node_module.vast_selection_markdown(
        {},
        profile_id="17",
        profile_name="broad",
    )

    assert markdown.endswith(
        "No Vast.ai node type was selected for this execution."
    )


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
