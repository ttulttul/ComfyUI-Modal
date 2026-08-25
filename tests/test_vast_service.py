"""Tests for Vast marketplace quotation and scheduler integration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

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


def test_quote_raises_memory_floors_and_compares_effective_hourly_price(
    vast_api_module: Any,
    vast_models_module: Any,
    vast_runtime_module: Any,
    vast_service_module: Any,
    vast_simulator_module: Any,
    tmp_path: Path,
) -> None:
    """Scheduling should exclude retention cost while still exposing its estimate."""

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
            assert quote.predicted_incremental_cost_usd == 120 / 3600 * 0.74
            assert quote.predicted_retention_cost_usd == (
                (24 * 3600 + 120) / 3600 * 0.74
            )
            assert scheduling_state.provider.value == "vast"
            assert scheduling_state.cost_usd_per_second == 0.74 / 3600
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


def test_startup_status_translates_vast_image_progress(
    vast_models_module: Any,
    vast_service_module: Any,
) -> None:
    """Provider layer messages should become useful workflow status without hashes."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 48534708,
            "actual_status": "loading",
            "intended_status": "running",
            "cur_state": "running",
            "status_msg": "4c7d45f7c63c: Download complete",
            "num_gpus": 1,
            "gpu_ram": 48 * 1024,
            "cpu_ram": 96 * 1024,
        }
    )

    message = vast_service_module._vast_startup_status_message(instance)

    assert message == (
        "Vast.ai instance 48534708 is downloading the worker image (layer complete)"
    )
    assert "4c7d45f7c63c" not in message


def test_startup_status_uses_current_state_when_actual_status_is_unknown(
    vast_models_module: Any,
    vast_service_module: Any,
) -> None:
    """Missing actual status should not render a misleading unknown state."""
    instance = vast_models_module.VastInstance.from_api(
        {
            "id": 48597015,
            "actual_status": None,
            "intended_status": "running",
            "cur_state": "loading",
            "num_gpus": 1,
            "gpu_ram": 96 * 1024,
            "cpu_ram": 128 * 1024,
        }
    )

    assert vast_service_module._vast_startup_status_message(instance) == (
        "Vast.ai instance 48597015 is starting (loading)"
    )


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


def test_acquire_destroys_lease_when_worker_initialization_fails(
    vast_service_module: Any,
) -> None:
    """An SSH-unusable rental must be drained instead of reused next run."""

    class FakeRegistry:
        """Record that the unusable lease was marked before destruction."""

        def __init__(self) -> None:
            """Initialize the updated instance list."""
            self.updated_instance_ids: list[int] = []

        def update(self, instance_id: int, updater: Any) -> None:
            """Record the update without requiring a complete lease fixture."""
            del updater
            self.updated_instance_ids.append(instance_id)

    class FakeLeaseManager:
        """Return and destroy one deterministic unusable rental."""

        def __init__(self, lease: Any) -> None:
            """Retain the fake lease and destruction log."""
            self.lease = lease
            self.destroyed_instance_ids: list[int] = []

        async def ensure_lease(self, profile: Any, **kwargs: Any) -> Any:
            """Return the selected lease without provider calls."""
            del profile, kwargs
            return self.lease

        async def destroy_owned_lease(self, instance_id: int) -> bool:
            """Record cleanup of the unusable capacity."""
            self.destroyed_instance_ids.append(instance_id)
            return True

    lease = SimpleNamespace(instance_id=42, environment_id="vast:test:42")
    lease_manager = FakeLeaseManager(lease)
    registry = FakeRegistry()
    service = object.__new__(vast_service_module.VastService)
    service.lease_manager = lease_manager
    service.registry = registry

    def fail_runtime(_lease: Any) -> None:
        """Simulate the observed provider SSH key-exchange failure."""
        raise vast_service_module.VastSshError(
            "kex_exchange_identification: Connection closed by remote host"
        )

    service._initialize_runtime = fail_runtime
    messages: list[str] = []

    with pytest.raises(
        vast_service_module.VastSshError,
        match="Connection closed by remote host",
    ):
        asyncio.run(
            service.acquire(
                SimpleNamespace(profile=SimpleNamespace()),
                status_callback=messages.append,
            )
        )

    assert registry.updated_instance_ids == [42]
    assert lease_manager.destroyed_instance_ids == [42]
    assert messages[-1] == "Vast.ai worker initialization failed"


def test_acquire_replaces_instance_that_disappears_before_ssh(
    vast_service_module: Any,
) -> None:
    """A missing live contract should cold-start the next marketplace offer."""

    class FakeRegistry:
        """Record stale contract removal before replacement."""

        def __init__(self) -> None:
            """Initialize the removed instance log."""
            self.removed_instance_ids: list[int] = []

        def remove(self, instance_id: int) -> None:
            """Record removal of a provider-missing lease."""
            self.removed_instance_ids.append(instance_id)

    class FakeLeaseManager:
        """Return a vanished lease followed by a fresh replacement."""

        def __init__(self, leases: list[Any]) -> None:
            """Retain sequenced leases and offer exclusions."""
            self.leases = leases
            self.excluded_offer_ids: list[frozenset[int]] = []

        async def ensure_lease(
            self,
            profile: Any,
            *,
            excluded_offer_ids: frozenset[int],
            **kwargs: Any,
        ) -> Any:
            """Return the next lease while recording rejected offers."""
            del profile, kwargs
            self.excluded_offer_ids.append(excluded_offer_ids)
            return self.leases.pop(0)

    vanished = SimpleNamespace(
        instance_id=42,
        offer_id=1001,
        environment_id="vast:test:42",
    )
    replacement = SimpleNamespace(
        instance_id=43,
        offer_id=1002,
        environment_id="vast:test:43",
    )
    lease_manager = FakeLeaseManager([vanished, replacement])
    registry = FakeRegistry()
    service = object.__new__(vast_service_module.VastService)
    service.lease_manager = lease_manager
    service.registry = registry

    def initialize_runtime(lease: Any) -> None:
        """Report only the first provider contract as missing."""
        if lease.instance_id == vanished.instance_id:
            raise vast_service_module.VastInstanceNotFoundError(
                "Vast instance 42 does not exist."
            )

    service._initialize_runtime = initialize_runtime
    messages: list[str] = []

    acquired = asyncio.run(
        service.acquire(
            SimpleNamespace(profile=SimpleNamespace()),
            status_callback=messages.append,
        )
    )

    assert acquired is replacement
    assert registry.removed_instance_ids == [42]
    assert lease_manager.excluded_offer_ids == [frozenset(), frozenset({1001})]
    assert "Vast.ai instance disappeared; requesting a replacement" in messages
    assert messages[-1] == "Vast.ai worker is ready"
