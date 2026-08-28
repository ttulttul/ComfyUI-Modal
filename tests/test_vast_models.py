"""Tests for workflow-declared Vast.ai marketplace requirements."""

from __future__ import annotations

from typing import Any

import pytest


def test_profile_builds_documented_hard_filter_payload(
    vast_models_module: Any,
) -> None:
    """Search requests should express every user spending and capacity guard."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="17",
        profile_name="large-video",
        gpu_count=2,
        minimum_gpu_ram_mb=48 * 1024,
        minimum_total_flops=150.0,
        minimum_cpu_ram_mb=128 * 1024,
        minimum_cpu_cores=24.0,
        allocated_disk_gb=500.0,
        maximum_hourly_cost_usd=2.5,
        allowed_geolocations=("US", "CA"),
        excluded_country_codes=("CN", "RU"),
    )

    payload = profile.search_payload(limit=12)

    assert payload["num_gpus"] == {"eq": 2}
    assert payload["gpu_ram"] == {"gte": 48 * 1024}
    assert payload["total_flops"] == {"gte": 150.0}
    assert payload["cpu_ram"] == {"gte": 128 * 1024}
    assert payload["allocated_storage"] == 500.0
    assert payload["dph_total"] == {"lte": 2.5}
    assert payload["geolocation"] == {
        "in": ["US", "CA"],
        "notin": ["CN", "RU"],
    }
    assert payload["order"] == [["dph_total", "asc"]]
    assert profile.environment_id == "vast:17"


def test_default_profile_excludes_problematic_country_codes(
    vast_models_module: Any,
) -> None:
    """A broad profile should still avoid countries with unreliable model access."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="any",
        profile_name="any",
    )

    payload = profile.search_payload()

    assert set(payload) == {
        "limit",
        "type",
        "rentable",
        "rented",
        "gpu_arch",
        "cpu_arch",
        "gpu_frac",
        "gpu_display_active",
        "disk_space",
        "allocated_storage",
        "cuda_max_good",
        "direct_port_count",
        "geolocation",
        "order",
    }
    assert payload["geolocation"] == {"notin": ["CN", "RU"]}


def test_profile_can_disable_default_country_exclusions(
    vast_models_module: Any,
) -> None:
    """An explicit empty exclusion should restore an unrestricted geo search."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="anywhere",
        profile_name="anywhere",
        excluded_country_codes=(),
    )

    assert "geolocation" not in profile.search_payload()


def test_offer_in_excluded_country_is_rejected_client_side(
    vast_models_module: Any,
) -> None:
    """Provider results must not bypass an excluded-country marketplace filter."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="safe-egress",
        profile_name="safe-egress",
    )
    offer = vast_models_module.VastOffer.from_api(
        {
            "id": 42,
            "gpu_name": "GPU",
            "num_gpus": 1,
            "gpu_ram": 96 * 1024,
            "gpu_total_ram": 96 * 1024,
            "total_flops": 100,
            "cpu_ram": 128 * 1024,
            "cpu_cores_effective": 16,
            "disk_space": 500,
            "duration": 30 * 86400,
            "reliability": 0.999,
            "dlperf": 50,
            "inet_down": 1000,
            "cuda_max_good": 13.0,
            "direct_port_count": 1,
            "dph_total": 0.5,
            "verification": "verified",
            "geolocation": "Beijing, CN",
        }
    )

    assert offer.incompatibility_reason(profile) == "geolocation is excluded"


def test_offer_ranking_is_price_first_and_deterministic(
    vast_models_module: Any,
) -> None:
    """The cheapest compatible offer should win before quality tie-breakers."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="one",
        profile_name="one",
    )
    base = {
        "gpu_name": "GPU",
        "num_gpus": 1,
        "gpu_ram": 24 * 1024,
        "gpu_total_ram": 24 * 1024,
        "total_flops": 100,
        "cpu_ram": 64 * 1024,
        "cpu_cores_effective": 16,
        "disk_space": 500,
        "duration": 30 * 86400,
        "reliability": 0.999,
        "dlperf": 50,
        "inet_down": 1000,
        "cuda_max_good": 13.0,
        "direct_port_count": 1,
        "verification": "verified",
    }
    expensive = vast_models_module.VastOffer.from_api(
        {**base, "id": 2, "dph_total": 0.8}
    )
    cheap = vast_models_module.VastOffer.from_api(
        {**base, "id": 1, "dph_total": 0.4, "reliability": 0.991}
    )

    assert vast_models_module.compatible_offers((expensive, cheap), profile) == (
        cheap,
        expensive,
    )


def test_instance_prefers_console_direct_ssh_mapping(
    vast_models_module: Any,
) -> None:
    """The public port mapping should win when Vast's SSH proxy endpoint is stale."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 48627244,
            "actual_status": "running",
            "ssh_host": "ssh9.vast.ai",
            "ssh_port": 27244,
            "public_ipaddr": "172.125.76.174",
            "ports": {
                "22/tcp": [
                    {"HostIp": "0.0.0.0", "HostPort": "40638"},
                    {"HostIp": "::", "HostPort": "40638"},
                ]
            },
            "num_gpus": 1,
            "gpu_ram": 97887,
            "cpu_ram": 128625,
        }
    )

    assert instance.ssh_host == "172.125.76.174"
    assert instance.ssh_port == 40638
    assert instance.ssh_direct_host == "172.125.76.174"
    assert instance.ssh_direct_port == 40638
    assert instance.ssh_proxy_host == "ssh9.vast.ai"
    assert instance.ssh_proxy_port == 27244


def test_instance_falls_back_to_proxy_for_malformed_direct_mapping(
    vast_models_module: Any,
) -> None:
    """A partial direct mapping must not hide Vast's legacy proxy endpoint."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 17,
            "actual_status": "running",
            "ssh_host": "ssh7.vast.ai",
            "ssh_port": 22017,
            "public_ipaddr": "192.0.2.17",
            "ports": {"22/tcp": [{"HostPort": "invalid"}]},
            "num_gpus": 1,
            "gpu_ram": 24576,
            "cpu_ram": 65536,
        }
    )

    assert instance.ssh_host == "ssh7.vast.ai"
    assert instance.ssh_port == 22017
    assert instance.ssh_direct_host is None
    assert instance.ssh_direct_port is None
    assert instance.ssh_proxy_host == "ssh7.vast.ai"
    assert instance.ssh_proxy_port == 22017


def test_instance_does_not_retain_an_incomplete_proxy_pair(
    vast_models_module: Any,
) -> None:
    """Transient proxy publication must not create invalid durable lease state."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 43,
            "actual_status": "loading",
            "ssh_host": "ssh7.vast.ai",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "cpu_ram": 65536,
        }
    )

    assert instance.ssh_host == "ssh7.vast.ai"
    assert instance.ssh_port is None
    assert instance.ssh_proxy_host is None
    assert instance.ssh_proxy_port is None


def test_offer_is_revalidated_after_server_filtering(vast_models_module: Any) -> None:
    """A malformed server result must not bypass client-side constraints."""
    profile = vast_models_module.VastResourceProfile(
        profile_id="large",
        profile_name="large",
        minimum_gpu_ram_mb=80 * 1024,
    )
    offer = vast_models_module.VastOffer.from_api(
        {
            "id": 8,
            "gpu_name": "Small GPU",
            "num_gpus": 1,
            "gpu_ram": 24 * 1024,
            "gpu_total_ram": 24 * 1024,
            "total_flops": 100,
            "cpu_ram": 64 * 1024,
            "cpu_cores_effective": 16,
            "disk_space": 500,
            "duration": 30 * 86400,
            "reliability": 0.999,
            "dlperf": 50,
            "inet_down": 1000,
            "cuda_max_good": 13.0,
            "direct_port_count": 1,
            "dph_total": 0.2,
            "verification": "verified",
        }
    )

    assert offer.incompatibility_reason(profile) == "insufficient GPU RAM"
    assert vast_models_module.compatible_offers((offer,), profile) == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_name", "unsafe profile"),
        ("maximum_hourly_cost_usd", 0.0),
        ("minimum_reliability", 1.1),
        ("maximum_instances", 0),
        ("excluded_country_codes", ("CHINA",)),
    ],
)
def test_profile_rejects_invalid_or_unsafe_values(
    vast_models_module: Any,
    field: str,
    value: Any,
) -> None:
    """Invalid workflow widgets should fail before marketplace access."""
    arguments = {
        "profile_id": "profile",
        "profile_name": "profile",
        field: value,
    }
    with pytest.raises(ValueError):
        vast_models_module.VastResourceProfile(**arguments)


def test_launch_payload_uses_direct_ssh_and_fail_fast(
    vast_models_module: Any,
) -> None:
    """Managed instances should expose SSH without queuing unavailable offers."""
    spec = vast_models_module.VastInstanceLaunchSpec(
        image="ghcr.io/example/worker@sha256:abc",
        disk_gb=200,
        label="comfy-modal:owner:profile",
        onstart="/opt/comfy/bin/start",
        environment={"COMFY_MODAL_REMOTE_WORKER": "1"},
    )

    payload = spec.to_api_payload()

    assert payload["runtype"] == "ssh_direct"
    assert payload["cancel_unavail"] is True
    assert payload["target_state"] == "running"
    assert "price" not in payload


def test_instance_normalizes_provider_progress_and_terminal_state(
    vast_models_module: Any,
) -> None:
    """Vast's stopped image-pull state should be bounded, readable, and terminal."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 42,
            "actual_status": "loading",
            "intended_status": "stopped",
            "cur_state": "stopped",
            "status_msg": " unauthorized:\n authentication required " + "x" * 300,
            "num_gpus": 1,
            "gpu_ram": 48 * 1024,
            "cpu_ram": 96 * 1024,
        }
    )

    assert instance.current_state == "stopped"
    assert instance.status_message is not None
    assert "\n" not in instance.status_message
    assert len(instance.status_message) == 240
    assert instance.terminal_failure is True


def test_instance_falls_back_from_unknown_actual_status(
    vast_models_module: Any,
) -> None:
    """Provider progress should use current state when actual status is absent."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 43,
            "actual_status": None,
            "intended_status": "running",
            "cur_state": "loading",
            "status_msg": "allocating host capacity",
            "num_gpus": 1,
            "gpu_ram": 96 * 1024,
            "cpu_ram": 128 * 1024,
        }
    )

    assert instance.actual_status == "unknown"
    assert instance.lifecycle_status == "loading"
