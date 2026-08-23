"""Tests for Vast marketplace quotation and scheduler integration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

from aiohttp import web


@asynccontextmanager
async def _running_app(app: web.Application) -> AsyncIterator[str]:
    """Serve one aiohttp application on an ephemeral loopback port."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    if server is None or not server.sockets:
        await runner.cleanup()
        raise RuntimeError("Test server did not expose a listening socket.")
    port = int(server.sockets[0].getsockname()[1])
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def test_quote_raises_memory_floors_and_prices_retention(
    vast_api_module: Any,
    vast_models_module: Any,
    vast_runtime_module: Any,
    vast_service_module: Any,
    vast_simulator_module: Any,
    tmp_path: Path,
) -> None:
    """A quote should select the cheapest offer after inferred memory and cooldown."""

    async def scenario() -> None:
        """Exercise the production client and service against the simulator."""
        simulator_state = vast_simulator_module.VastSimulatorState()
        simulator = vast_simulator_module.VastApiSimulator(simulator_state)
        async with _running_app(simulator.app) as base_url:
            settings = SimpleNamespace(app_name="test-owner")
            registry = vast_service_module.VastLeaseRegistry.for_user_directory(
                tmp_path
            )
            service = vast_service_module.VastService(
                settings=settings,
                repo_root=tmp_path,
                user_directory=tmp_path,
                api_client=vast_api_module.VastApiClient(
                    simulator_state.api_key,
                    base_url=base_url,
                ),
                runtime_configuration=vast_runtime_module.VastRuntimeConfiguration(
                    image="example.invalid/comfy-worker:test",
                    runtime_fingerprint="a" * 64,
                ),
                registry=registry,
            )
            profile = vast_models_module.VastResourceProfile(
                profile_id="77",
                profile_name="workflow-default",
                maximum_hourly_cost_usd=2.0,
                idle_retention_seconds=24 * 3600,
            )

            quote = await service.quote_best_profile(
                (profile,),
                minimum_vram_bytes=40 * 1024**3,
                minimum_ram_bytes=96 * 1024**3,
                predicted_execution_seconds=120.0,
            )
            scheduling_state = service.scheduling_state(quote)

            assert quote.offer.offer_id == 1002
            assert quote.profile.minimum_gpu_ram_mb == 40 * 1024
            assert quote.profile.minimum_cpu_ram_mb == 96 * 1024
            assert quote.predicted_incremental_cost_usd == (
                (24 * 3600 + 120) / 3600 * 0.74
            )
            assert scheduling_state.provider.value == "vast"
            assert scheduling_state.capabilities.maximum_vram_bytes == 48 * 1024**3

    asyncio.run(scenario())


def test_from_environment_requires_credential_before_other_setup(
    vast_service_module: Any,
    tmp_path: Path,
) -> None:
    """Vast-only selection should fail clearly without touching marketplace state."""
    settings = SimpleNamespace()

    try:
        vast_service_module.VastService.from_environment(
            settings,
            repo_root=tmp_path,
            environment={},
        )
    except RuntimeError as exc:
        assert "VAST_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing Vast credential was accepted.")


def test_prefetch_deduplicates_effective_profiles_and_searches_in_parallel(
    vast_models_module: Any,
    vast_runtime_module: Any,
    vast_service_module: Any,
    tmp_path: Path,
) -> None:
    """Planning should concurrently warm only distinct marketplace queries."""

    class DelayedApiClient:
        """Record overlapping searches without making network requests."""

        def __init__(self) -> None:
            """Initialize concurrency counters and searched VRAM floors."""
            self.active = 0
            self.peak = 0
            self.minimum_gpu_ram_mb: list[int] = []

        async def search_offers(self, profile: Any) -> tuple[Any, ...]:
            """Hold the event loop briefly so independent searches overlap."""
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.minimum_gpu_ram_mb.append(profile.minimum_gpu_ram_mb)
            await asyncio.sleep(0.02)
            self.active -= 1
            return ()

    async def scenario() -> None:
        """Prefetch repeated low floors and one raised floor."""
        api_client = DelayedApiClient()
        service = vast_service_module.VastService(
            settings=SimpleNamespace(app_name="test-owner"),
            repo_root=tmp_path,
            user_directory=tmp_path,
            api_client=api_client,
            runtime_configuration=vast_runtime_module.VastRuntimeConfiguration(
                image="example.invalid/comfy-worker:test",
                runtime_fingerprint="b" * 64,
            ),
            registry=vast_service_module.VastLeaseRegistry.for_user_directory(
                tmp_path
            ),
        )
        profile = vast_models_module.VastResourceProfile(
            profile_id="parallel",
            profile_name="parallel",
            minimum_gpu_ram_mb=48 * 1024,
        )
        requirements = (
            vast_service_module.VastSearchRequirements(0, 0),
            vast_service_module.VastSearchRequirements(16 * 1024**3, 0),
            vast_service_module.VastSearchRequirements(80 * 1024**3, 0),
        )

        await service.prefetch_offers((profile,), requirements)

        assert sorted(api_client.minimum_gpu_ram_mb) == [48 * 1024, 80 * 1024]
        assert api_client.peak == 2

    asyncio.run(scenario())
