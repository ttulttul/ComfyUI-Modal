"""Tests for Vast marketplace quotation and scheduler integration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
            settings = SimpleNamespace(
                app_name="test-owner",
                modal_gpu="RTX-PRO-6000",
            )
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


def test_from_environment_reuses_queue_time_runtime_expectation(
    vast_service_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Execution construction should not hash a moving local source tree again."""
    fingerprint = "f" * 64
    image = "ghcr.io/example/worker@sha256:" + "a" * 64
    settings = SimpleNamespace(
        app_name="test-owner",
        modal_gpu="RTX-PRO-6000",
        comfyui_root=tmp_path / "ComfyUI",
        custom_nodes_dir=None,
    )
    monkeypatch.setattr(
        vast_service_module,
        "discover_comfyui_user_directory",
        lambda _settings: tmp_path,
    )
    monkeypatch.setattr(
        vast_service_module,
        "build_vast_runtime_identity",
        lambda **_kwargs: pytest.fail("runtime identity was recomputed"),
    )

    service = vast_service_module.VastService.from_environment(
        settings,
        repo_root=tmp_path,
        environment={"VAST_API_KEY": "secret"},
        runtime_fingerprint=fingerprint,
        worker_image=image,
    )

    assert service.runtime_configuration.runtime_fingerprint == fingerprint
    assert service.runtime_configuration.image == image


def test_sync_engine_keeps_vast_lease_active_during_r2_writeback(
    settings_module: Any,
    vast_service_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Idle R2 transfers should hold lease activity and refresh both watchdog states."""
    events: list[tuple[str, Any]] = []

    class FakeLeaseManager:
        """Record the activity lifecycle selected for background write-back."""

        def begin_activity(self, instance_id: int) -> Any:
            """Return one active lease snapshot."""
            events.append(("begin", instance_id))
            return SimpleNamespace(state="active")

        def finish_activity(
            self,
            instance_id: int,
            *,
            idle_retention_seconds: float,
        ) -> Any:
            """Return one idle lease snapshot."""
            events.append(("finish", (instance_id, idle_retention_seconds)))
            return SimpleNamespace(state="idle")

    class FakeRuntime:
        """Record watchdog publications without opening SSH."""

        def __init__(self, **_kwargs: Any) -> None:
            """Accept the production runtime constructor arguments."""

        def update_watchdog(self, lease: Any) -> None:
            """Record one active or idle watchdog snapshot."""
            events.append(("watchdog", lease.state))

    class FakeVolume:
        """Accept the production Vast storage constructor arguments."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Initialize the no-op test backend."""

    monkeypatch.setattr(vast_service_module, "VastRuntimeManager", FakeRuntime)
    monkeypatch.setattr(vast_service_module, "VastSshVolumeBackend", FakeVolume)
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="vast",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=None,
    )
    service = SimpleNamespace(
        settings=settings,
        runtime_configuration=SimpleNamespace(
            remote_storage_root=PurePosixPath("/storage")
        ),
        lease_manager=FakeLeaseManager(),
        huggingface_asset_registry=None,
        huggingface_asset_discovery=None,
        r2_cache=None,
        _runner=lambda _lease: object(),
    )
    lease = SimpleNamespace(instance_id=42, idle_retention_seconds=90.0)

    engine = vast_service_module.VastService.sync_engine(service, lease)
    assert engine.r2_writeback_activity is not None
    with engine.r2_writeback_activity():
        events.append(("transfer", 42))

    assert events == [
        ("begin", 42),
        ("watchdog", "active"),
        ("transfer", 42),
        ("finish", (42, 90.0)),
        ("watchdog", "idle"),
    ]


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
            settings=SimpleNamespace(
                app_name="test-owner",
                modal_gpu="RTX-PRO-6000",
            ),
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


def test_acquire_replaces_lease_when_worker_initialization_fails(
    vast_service_module: Any,
) -> None:
    """An SSH-unusable rental must be destroyed and replaced in the same run."""

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
        """Return an unusable rental followed by a healthy replacement."""

        def __init__(self, leases: list[Any]) -> None:
            """Retain the fake leases, exclusions, and destruction log."""
            self.leases = leases
            self.destroyed_instance_ids: list[int] = []
            self.excluded_offer_ids: list[frozenset[int]] = []

        async def ensure_lease(
            self,
            profile: Any,
            *,
            excluded_offer_ids: frozenset[int],
            **kwargs: Any,
        ) -> Any:
            """Return each selected lease while recording rejected offers."""
            del profile, kwargs
            self.excluded_offer_ids.append(excluded_offer_ids)
            return self.leases.pop(0)

        async def destroy_owned_lease(self, instance_id: int) -> bool:
            """Record cleanup of the unusable capacity."""
            self.destroyed_instance_ids.append(instance_id)
            return True

    failed = SimpleNamespace(
        instance_id=42,
        offer_id=1001,
        environment_id="vast:test:42",
    )
    replacement = SimpleNamespace(
        instance_id=43,
        offer_id=1002,
        environment_id="vast:test:43",
    )
    lease_manager = FakeLeaseManager([failed, replacement])
    registry = FakeRegistry()
    service = object.__new__(vast_service_module.VastService)
    service.lease_manager = lease_manager
    service.registry = registry

    async def skip_preflight(*, status_callback: Any) -> None:
        """Keep this test focused on post-rental SSH replacement."""
        del status_callback

    service._preflight_published_image = skip_preflight

    def initialize_runtime(
        lease: Any,
        *,
        status_callback: Any,
    ) -> None:
        """Simulate the observed SSH failure only on the first rental."""
        if lease.instance_id == failed.instance_id:
            status_callback(
                "Vast SSH attempt 1 failed: Connection refused; retrying in 1.00s"
            )
            raise vast_service_module.VastSshError(
                "kex_exchange_identification: Connection closed by remote host"
            )

    service._initialize_runtime = initialize_runtime
    messages: list[str] = []

    acquired = asyncio.run(
        service.acquire(
            SimpleNamespace(profile=SimpleNamespace(), existing_lease=None),
            status_callback=messages.append,
        )
    )

    assert acquired is replacement
    assert registry.updated_instance_ids == [42]
    assert lease_manager.destroyed_instance_ids == [42]
    assert lease_manager.excluded_offer_ids == [frozenset(), frozenset({1001})]
    assert any("Vast SSH attempt 1 failed" in message for message in messages)
    assert any("failed worker setup" in message for message in messages)
    assert messages[-1] == "Vast.ai worker is ready"


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

    async def skip_preflight(*, status_callback: Any) -> None:
        """Keep this test focused on a disappeared provider contract."""
        del status_callback

    service._preflight_published_image = skip_preflight

    def initialize_runtime(
        lease: Any,
        *,
        status_callback: Any,
    ) -> None:
        """Report only the first provider contract as missing."""
        del status_callback
        if lease.instance_id == vanished.instance_id:
            raise vast_service_module.VastInstanceNotFoundError(
                "Vast instance 42 does not exist."
            )

    service._initialize_runtime = initialize_runtime
    messages: list[str] = []

    acquired = asyncio.run(
        service.acquire(
            SimpleNamespace(profile=SimpleNamespace(), existing_lease=None),
            status_callback=messages.append,
        )
    )

    assert acquired is replacement
    assert registry.removed_instance_ids == [42]
    assert lease_manager.excluded_offer_ids == [frozenset(), frozenset({1001})]
    assert "Vast.ai instance disappeared; requesting a replacement" in messages
    assert messages[-1] == "Vast.ai worker is ready"


def test_acquire_rebuilds_image_and_replaces_fingerprint_drift(
    vast_service_module: Any,
) -> None:
    """A stale published worker should be rebuilt once before capacity is retried."""

    class FakeRegistry:
        """Record stale worker draining before its destruction."""

        def __init__(self) -> None:
            """Initialize the updated instance log."""
            self.updated_instance_ids: list[int] = []

        def update(self, instance_id: int, updater: Any) -> None:
            """Record one stale lease update."""
            del updater
            self.updated_instance_ids.append(instance_id)

    class FakeLeaseManager:
        """Return one stale lease followed by a fresh-image lease."""

        def __init__(self, leases: list[Any]) -> None:
            """Retain sequenced leases and destruction calls."""
            self.leases = leases
            self.destroyed_instance_ids: list[int] = []

        async def ensure_lease(self, profile: Any, **kwargs: Any) -> Any:
            """Return the next lease without contacting Vast."""
            del profile, kwargs
            return self.leases.pop(0)

        async def destroy_owned_lease(self, instance_id: int) -> bool:
            """Record stale capacity cleanup."""
            self.destroyed_instance_ids.append(instance_id)
            return True

    class FakeImageBuilder:
        """Publish a deterministic replacement image."""

        def __init__(self) -> None:
            """Initialize requested fingerprint tracking."""
            self.fingerprints: list[str] = []

        def build_and_push(
            self,
            expected_fingerprint: str,
            *,
            status_callback: Any,
        ) -> str:
            """Emit progress and return a digest-pinned image."""
            self.fingerprints.append(expected_fingerprint)
            status_callback("Building replacement worker")
            return "ghcr.io/example/worker@sha256:" + "c" * 64

    stale = SimpleNamespace(
        instance_id=42,
        offer_id=1001,
        environment_id="vast:test:42",
    )
    replacement = SimpleNamespace(
        instance_id=43,
        offer_id=1001,
        environment_id="vast:test:43",
    )
    lease_manager = FakeLeaseManager([stale, replacement])
    image_builder = FakeImageBuilder()
    service = object.__new__(vast_service_module.VastService)
    service.lease_manager = lease_manager
    service.registry = FakeRegistry()
    service.image_builder = image_builder
    adopted_images: list[str] = []
    service._adopt_runtime_image = adopted_images.append

    async def skip_preflight(*, status_callback: Any) -> None:
        """Keep runtime drift as a defense-in-depth behavior in this test."""
        del status_callback

    service._preflight_published_image = skip_preflight

    def initialize_runtime(
        lease: Any,
        *,
        status_callback: Any,
    ) -> None:
        """Reject only the worker baked into the first lease."""
        del status_callback
        if lease.instance_id == stale.instance_id:
            raise vast_service_module.VastRuntimeFingerprintDriftError(
                expected_fingerprint="a" * 64,
                actual_fingerprint="b" * 64,
                protocol_version=1,
            )

    service._initialize_runtime = initialize_runtime
    messages: list[str] = []

    acquired = asyncio.run(
        service.acquire(
            SimpleNamespace(profile=SimpleNamespace(), existing_lease=None),
            status_callback=messages.append,
        )
    )

    assert acquired is replacement
    assert service.registry.updated_instance_ids == [42]
    assert lease_manager.destroyed_instance_ids == [42]
    assert image_builder.fingerprints == ["a" * 64]
    assert adopted_images == ["ghcr.io/example/worker@sha256:" + "c" * 64]
    assert "Building replacement worker" in messages
    assert "Vast worker image updated; requesting fresh capacity" in messages
    assert messages[-1] == "Vast.ai worker is ready"


def test_preflight_rebuilds_stale_image_before_capacity_request(
    vast_service_module: Any,
) -> None:
    """Registry drift should be repaired before lease acquisition can begin."""

    class FakeImageBuilder:
        """Return a replacement digest and record preflight inputs."""

        def __init__(self) -> None:
            """Initialize the call log."""
            self.calls: list[tuple[str, str]] = []

        def ensure_published_image(
            self,
            image: str,
            expected_fingerprint: str,
            *,
            status_callback: Any,
        ) -> str:
            """Simulate a stale registry image replacement."""
            self.calls.append((image, expected_fingerprint))
            status_callback("Published Vast worker image is stale")
            return "ghcr.io/example/worker@sha256:" + "d" * 64

    builder = FakeImageBuilder()
    service = object.__new__(vast_service_module.VastService)
    service.runtime_configuration = SimpleNamespace(
        image="ghcr.io/example/worker:v1",
        runtime_fingerprint="a" * 64,
    )
    service.image_builder = builder
    adopted: list[str] = []
    service._adopt_runtime_image = adopted.append
    messages: list[str] = []

    asyncio.run(service._preflight_published_image(status_callback=messages.append))

    assert builder.calls == [("ghcr.io/example/worker:v1", "a" * 64)]
    assert adopted == ["ghcr.io/example/worker@sha256:" + "d" * 64]
    assert messages[-1] == "Vast worker image updated before requesting capacity"


def test_acquire_preflights_an_existing_lease(
    vast_service_module: Any,
) -> None:
    """Lease reuse must resolve and validate its immutable image before readiness."""
    service = object.__new__(vast_service_module.VastService)
    preflight_calls: list[Any] = []
    acquired_lease = SimpleNamespace(instance_id=42)

    async def preflight(*, status_callback: Any) -> None:
        """Record the registry validation that precedes lease reuse."""
        preflight_calls.append(status_callback)

    async def acquire_with_replacement(*_args: Any, **_kwargs: Any) -> Any:
        """Return one already quoted lease without provider work."""
        return acquired_lease

    service._preflight_published_image = preflight
    service._acquire_with_replacement = acquire_with_replacement
    quote = SimpleNamespace(
        profile=SimpleNamespace(),
        existing_lease=acquired_lease,
    )

    result = asyncio.run(service.acquire(quote))

    assert result is acquired_lease
    assert preflight_calls == [None]


def test_proxy_endpoint_promotion_persists_the_working_route(
    vast_service_module: Any,
) -> None:
    """Successful proxy readiness must replace the dead direct primary endpoint."""

    @dataclass(frozen=True)
    class Lease:
        """Represent the endpoint fields changed by proxy promotion."""

        instance_id: int
        ssh_host: str
        ssh_port: int
        ssh_proxy_host: str
        ssh_proxy_port: int

    class UpdatingRegistry:
        """Apply one lease update in memory."""

        def __init__(self, lease: Any) -> None:
            """Retain the current lease."""
            self.lease = lease

        def update(self, instance_id: int, updater: Any) -> Any:
            """Apply and retain the requested endpoint change."""
            assert instance_id == self.lease.instance_id
            self.lease = updater(self.lease)
            return self.lease

    lease = Lease(
        instance_id=42,
        ssh_host="192.0.2.4",
        ssh_port=40112,
        ssh_proxy_host="ssh7.vast.ai",
        ssh_proxy_port=22017,
    )
    registry = UpdatingRegistry(lease)
    service = object.__new__(vast_service_module.VastService)
    service.registry = registry

    service._promote_proxy_endpoint(lease)

    assert registry.lease.ssh_host == "ssh7.vast.ai"
    assert registry.lease.ssh_port == 22017
