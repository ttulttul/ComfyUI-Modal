"""Opt-in billable Vast provisioning and direct-worker readiness canary."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_vast


def test_live_vast_provision_and_worker_readiness(
    settings_module: object,
    vast_models_module: object,
    vast_service_module: object,
) -> None:
    """Rent the cheapest bounded offer, verify the worker, and destroy by default."""
    if os.getenv("COMFY_MODAL_RUN_LIVE_VAST") != "1":
        pytest.skip("Set COMFY_MODAL_RUN_LIVE_VAST=1 for the billable Vast canary.")
    settings = settings_module.get_settings()
    service = vast_service_module.VastService.from_environment(
        settings,
        repo_root=Path(__file__).resolve().parents[1],
    )
    profile = vast_models_module.VastResourceProfile(
        profile_id="live-canary",
        profile_name="live-canary",
        minimum_gpu_ram_mb=int(
            float(os.getenv("COMFY_MODAL_VAST_CANARY_VRAM_GB", "24")) * 1024
        ),
        minimum_cpu_ram_mb=int(
            float(os.getenv("COMFY_MODAL_VAST_CANARY_RAM_GB", "32")) * 1024
        ),
        allocated_disk_gb=float(
            os.getenv("COMFY_MODAL_VAST_CANARY_DISK_GB", "80")
        ),
        maximum_hourly_cost_usd=float(
            os.getenv("COMFY_MODAL_VAST_CANARY_MAX_USD_HOUR", "1.0")
        ),
        idle_retention_seconds=float(
            os.getenv("COMFY_MODAL_VAST_CANARY_IDLE_HOURS", "1")
        )
        * 3600,
    )

    async def scenario() -> None:
        """Run the live lifecycle while preserving the instance only by opt-in."""
        quote = await service.quote_best_profile(
            (profile,),
            minimum_vram_bytes=profile.minimum_gpu_ram_mb * 1024**2,
            minimum_ram_bytes=profile.minimum_cpu_ram_mb * 1024**2,
            predicted_execution_seconds=60.0,
        )
        lease = await service.acquire(quote)
        assert lease.ready_for_work
        assert lease.runtime_fingerprint == service.runtime_configuration.runtime_fingerprint
        if os.getenv("COMFY_MODAL_VAST_KEEP_CANARY") != "1":
            assert await service.lease_manager.destroy_owned_lease(lease.instance_id)

    asyncio.run(scenario())
