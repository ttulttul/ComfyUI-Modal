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
    )

    payload = profile.search_payload(limit=12)

    assert payload["num_gpus"] == {"eq": 2}
    assert payload["gpu_ram"] == {"gte": 48 * 1024}
    assert payload["total_flops"] == {"gte": 150.0}
    assert payload["cpu_ram"] == {"gte": 128 * 1024}
    assert payload["allocated_storage"] == 500.0
    assert payload["dph_total"] == {"lte": 2.5}
    assert payload["geolocation"] == {"in": ["US", "CA"]}
    assert payload["order"] == [["dph_total", "asc"]]
    assert profile.environment_id == "vast:17"


def test_any_profile_omits_every_optional_marketplace_filter(
    vast_models_module: Any,
) -> None:
    """None-backed Any selectors should produce the broadest safe search."""
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
        "order",
    }


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
