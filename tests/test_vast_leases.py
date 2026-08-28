"""Tests for persistent Vast.ai lease acquisition and idle cleanup."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest
from aiohttp import web


@dataclass
class FakeClock:
    """Controllable wall clock for lease-retention tests."""

    value: float = 1_000_000.0

    def __call__(self) -> float:
        """Return the current fake epoch."""
        return self.value

    def advance(self, seconds: float) -> None:
        """Move the fake epoch forward."""
        self.value += seconds


@asynccontextmanager
async def _running_simulator(app: web.Application) -> AsyncIterator[str]:
    """Serve one local simulator app on an ephemeral port."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    if server is None or not server.sockets:
        await runner.cleanup()
        raise RuntimeError("Simulator did not expose a listening socket.")
    port = int(server.sockets[0].getsockname()[1])
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _runtime_fingerprint() -> str:
    """Return a deterministic runtime SHA-256 value."""
    return "a" * 64


def _profile(vast_models_module: Any, **overrides: Any) -> Any:
    """Return one test profile with short retention."""
    values = {
        "profile_id": "node-17",
        "profile_name": "default",
        "idle_retention_seconds": 60.0,
    }
    values.update(overrides)
    return vast_models_module.VastResourceProfile(**values)


def _launch_factory(vast_models_module: Any) -> Any:
    """Return a deterministic launch-spec factory."""

    def factory(profile: Any, label: str) -> Any:
        """Build a launch spec matching one selected profile."""
        return vast_models_module.VastInstanceLaunchSpec(
            image="ghcr.io/example/comfy-vast@sha256:test",
            disk_gb=profile.allocated_disk_gb,
            label=label,
            onstart="/opt/comfy/start-worker",
            environment={"COMFY_MODAL_REMOTE_WORKER": "1"},
        )

    return factory


def test_manager_recovers_offer_race_then_reuses_persisted_lease(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Rent the next candidate after a race and avoid duplicate instances."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState(
            create_failures_remaining={1001: 1},
            create_failure_status=400,
            create_failure_message=(
                "error 404/3603: no_such_ask Instance type by id 1001 is not available."
            ),
            polls_until_running=1,
        )
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            manager = vast_leases_module.VastLeaseManager(
                api_client=vast_api_module.VastApiClient(
                    state.api_key,
                    base_url=base_url,
                    retry_attempts=1,
                ),
                registry=vast_leases_module.VastLeaseRegistry.for_user_directory(
                    tmp_path
                ),
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            profile = _profile(vast_models_module)

            first = await manager.ensure_lease(profile)
            second = await manager.ensure_lease(profile)

            assert first.instance_id == second.instance_id
            assert first.offer_id == 1002
            assert len(state.instances) == 1
            create_requests = [
                request
                for request in state.request_log
                if request["path"].startswith("/api/v0/asks/")
            ]
            assert [request["path"] for request in create_requests] == [
                "/api/v0/asks/1001/",
                "/api/v0/asks/1002/",
            ]

    asyncio.run(scenario())


def test_manager_excludes_offer_from_disappeared_instance_replacement(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A cold replacement must not immediately rent the vanished offer again."""

    async def scenario() -> None:
        """Exclude the cheapest offer and rent the next compatible candidate."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            manager = vast_leases_module.VastLeaseManager(
                api_client=vast_api_module.VastApiClient(
                    state.api_key,
                    base_url=base_url,
                ),
                registry=vast_leases_module.VastLeaseRegistry.for_user_directory(
                    tmp_path
                ),
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )

            lease = await manager.ensure_lease(
                _profile(vast_models_module),
                excluded_offer_ids=frozenset({1001}),
            )

            assert lease.offer_id == 1002
            create_requests = [
                request
                for request in state.request_log
                if request["path"].startswith("/api/v0/asks/")
            ]
            assert [request["path"] for request in create_requests] == [
                "/api/v0/asks/1002/"
            ]

    asyncio.run(scenario())


def test_manager_replaces_contract_that_disappears_during_provider_startup(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A contract lost before SSH readiness should rent the next offer."""

    class DisappearingClient(vast_api_module.VastApiClient):
        """Remove the first created contract during provider readiness."""

        def __init__(self, *args: Any, state: Any, **kwargs: Any) -> None:
            """Retain simulator state and initialize the disappearance count."""
            super().__init__(*args, **kwargs)
            self.state = state
            self.readiness_calls = 0

        async def wait_until_ready(
            self,
            instance_id: int,
            **kwargs: Any,
        ) -> Any:
            """Lose the first contract and delegate later readiness checks."""
            self.readiness_calls += 1
            if self.readiness_calls == 1:
                self.state.instances.pop(instance_id)
                raise vast_api_module.VastInstanceNotFoundError(
                    f"Vast instance {instance_id} does not exist."
                )
            return await super().wait_until_ready(instance_id, **kwargs)

    async def scenario() -> None:
        """Verify cold replacement and stale registry cleanup."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
            manager = vast_leases_module.VastLeaseManager(
                api_client=DisappearingClient(
                    state.api_key,
                    base_url=base_url,
                    state=state,
                ),
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )

            lease = await manager.ensure_lease(_profile(vast_models_module))

            assert lease.offer_id == 1002
            assert [record.instance_id for record in registry.load().leases] == [
                lease.instance_id
            ]
            create_requests = [
                request
                for request in state.request_log
                if request["path"].startswith("/api/v0/asks/")
            ]
            assert [request["path"] for request in create_requests] == [
                "/api/v0/asks/1001/",
                "/api/v0/asks/1002/",
            ]

    asyncio.run(scenario())


def test_manager_replaces_instance_that_times_out_during_provider_startup(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A loading instance that misses its deadline should be destroyed and replaced."""

    class TimingOutClient(vast_api_module.VastApiClient):
        """Timeout the first rental and allow its replacement to become ready."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize the deterministic readiness attempt counter."""
            super().__init__(*args, **kwargs)
            self.readiness_calls = 0

        async def wait_until_ready(
            self,
            instance_id: int,
            **kwargs: Any,
        ) -> Any:
            """Fail only the initial instance's provider-readiness wait."""
            self.readiness_calls += 1
            if self.readiness_calls == 1:
                raise TimeoutError(
                    f"Vast instance {instance_id} remained in 'loading'."
                )
            return await super().wait_until_ready(instance_id, **kwargs)

    async def scenario() -> None:
        """Verify API destruction, offer exclusion, and replacement telemetry."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
            manager = vast_leases_module.VastLeaseManager(
                api_client=TimingOutClient(state.api_key, base_url=base_url),
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            events: list[str] = []

            lease = await manager.ensure_lease(
                _profile(vast_models_module),
                event_callback=events.append,
            )

            assert lease.offer_id == 1002
            assert len(state.destroyed_instance_ids) == 1
            assert state.destroyed_instance_ids[0] != lease.instance_id
            assert [record.instance_id for record in registry.load().leases] == [
                lease.instance_id
            ]
            create_requests = [
                request
                for request in state.request_log
                if request["path"].startswith("/api/v0/asks/")
            ]
            assert [request["path"] for request in create_requests] == [
                "/api/v0/asks/1001/",
                "/api/v0/asks/1002/",
            ]
            assert "failed setup and was terminated" in events[0]
            assert "requesting a replacement" in events[0]

    asyncio.run(scenario())


def test_manager_bounds_replacements_when_every_instance_times_out(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A bad marketplace pool must not create an unbounded rental loop."""

    class AlwaysTimingOutClient(vast_api_module.VastApiClient):
        """Keep every newly created instance in a synthetic failed setup state."""

        async def wait_until_ready(
            self,
            instance_id: int,
            **kwargs: Any,
        ) -> Any:
            """Fail immediately while preserving the instance for API destruction."""
            del kwargs
            raise TimeoutError(f"Vast instance {instance_id} remained in 'loading'.")

    async def scenario() -> None:
        """Verify exactly an initial attempt plus one replacement."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            manager = vast_leases_module.VastLeaseManager(
                api_client=AlwaysTimingOutClient(state.api_key, base_url=base_url),
                registry=vast_leases_module.VastLeaseRegistry.for_user_directory(
                    tmp_path
                ),
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            events: list[str] = []

            with pytest.raises(
                vast_leases_module.VastLeaseStartupExhaustedError,
                match="after 2 attempts",
            ):
                await manager.ensure_lease(
                    _profile(vast_models_module),
                    event_callback=events.append,
                )

            assert len(state.destroyed_instance_ids) == 2
            assert "requesting a replacement" in events[0]
            assert "no further automatic attempts remain" in events[-1]

    asyncio.run(scenario())


def test_manager_adopts_legacy_lease_for_unchanged_worker_image(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Controller fingerprint drift should not rent again for the same worker image."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
            client = vast_api_module.VastApiClient(state.api_key, base_url=base_url)
            profile = _profile(vast_models_module)
            first_manager = vast_leases_module.VastLeaseManager(
                api_client=client,
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint="a" * 64,
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            first = await first_manager.ensure_lease(profile)
            second_manager = vast_leases_module.VastLeaseManager(
                api_client=client,
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint="b" * 64,
                worker_image="ghcr.io/example/worker@sha256:same",
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )

            second = await second_manager.ensure_lease(profile)

            assert second.instance_id == first.instance_id
            assert second.runtime_fingerprint == "b" * 64
            assert second.worker_image == "ghcr.io/example/worker@sha256:same"
            assert len(state.instances) == 1

    asyncio.run(scenario())


def test_active_work_suspends_idle_destruction_and_finish_resets_deadline(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Retention begins at terminal activity and never destroys active work."""

    async def scenario() -> None:
        clock = FakeClock()
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            manager = vast_leases_module.VastLeaseManager(
                api_client=vast_api_module.VastApiClient(
                    state.api_key, base_url=base_url
                ),
                registry=vast_leases_module.VastLeaseRegistry.for_user_directory(
                    tmp_path
                ),
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
                clock=clock,
            )
            profile = _profile(vast_models_module)
            lease = await manager.ensure_lease(profile)
            manager.begin_activity(lease.instance_id)
            clock.advance(600)

            assert await manager.destroy_expired() == ()
            finished = manager.finish_activity(
                lease.instance_id,
                idle_retention_seconds=profile.idle_retention_seconds,
            )
            assert finished.idle_deadline_epoch == clock.value + 60
            clock.advance(59)
            assert await manager.destroy_expired() == ()
            clock.advance(2)

            assert await manager.destroy_expired() == (lease.instance_id,)
            assert state.instances == {}
            assert manager.registry.load().leases == ()

    asyncio.run(scenario())


def test_reconcile_after_restart_clears_stale_activity_count(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A dead local process must not leave a lease permanently busy."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
            client = vast_api_module.VastApiClient(state.api_key, base_url=base_url)
            manager = vast_leases_module.VastLeaseManager(
                api_client=client,
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            lease = await manager.ensure_lease(_profile(vast_models_module))
            manager.begin_activity(lease.instance_id)

            restarted = vast_leases_module.VastLeaseManager(
                api_client=client,
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
            )
            reconciled = await restarted.reconcile()

            assert reconciled[0].active_invocations == 0
            assert registry.load().leases[0].active_invocations == 0

    asyncio.run(scenario())


def test_inventory_refresh_preserves_live_activity_count(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A UI inventory refresh must not make an active lease appear idle."""

    async def scenario() -> None:
        """Refresh provider state without applying restart reconciliation semantics."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            manager = vast_leases_module.VastLeaseManager(
                api_client=vast_api_module.VastApiClient(
                    state.api_key,
                    base_url=base_url,
                ),
                registry=vast_leases_module.VastLeaseRegistry.for_user_directory(
                    tmp_path
                ),
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            lease = await manager.ensure_lease(_profile(vast_models_module))
            manager.begin_activity(lease.instance_id)

            refreshed = await manager.refresh_owned_leases()

            assert refreshed[0].active_invocations == 1
            assert manager.registry.load().leases[0].active_invocations == 1

    asyncio.run(scenario())


def test_manual_destroy_requires_owned_matching_label(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Never destroy a reused instance identity whose ownership label changed."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            manager = vast_leases_module.VastLeaseManager(
                api_client=vast_api_module.VastApiClient(
                    state.api_key, base_url=base_url
                ),
                registry=vast_leases_module.VastLeaseRegistry.for_user_directory(
                    tmp_path
                ),
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            lease = await manager.ensure_lease(_profile(vast_models_module))
            state.instances[lease.instance_id]["label"] = "someone-else"

            with pytest.raises(RuntimeError, match="label no longer matches"):
                await manager.destroy_owned_lease(lease.instance_id)

            assert lease.instance_id in state.instances

    asyncio.run(scenario())


def test_destroy_removes_registry_when_instance_vanishes_after_ownership_check(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A destroy-time not-found response confirms cleanup of the owned lease."""

    class VanishingOnDestroyClient(vast_api_module.VastApiClient):
        """Delete the contract immediately before its destroy API response."""

        def __init__(self, *args: Any, state: Any, **kwargs: Any) -> None:
            """Retain simulator state for the synthetic provider race."""
            super().__init__(*args, **kwargs)
            self.state = state

        async def destroy_instance(self, instance_id: int) -> None:
            """Simulate Vast removing the contract between show and destroy."""
            self.state.instances.pop(instance_id, None)
            raise vast_api_module.VastInstanceNotFoundError(
                f"Vast instance {instance_id} does not exist."
            )

    async def scenario() -> None:
        """Verify the provider race does not retain stale local inventory."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
            manager = vast_leases_module.VastLeaseManager(
                api_client=VanishingOnDestroyClient(
                    state.api_key,
                    base_url=base_url,
                    state=state,
                ),
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            lease = await manager.ensure_lease(_profile(vast_models_module))

            assert await manager.destroy_owned_lease(lease.instance_id)
            assert registry.load().leases == ()
            assert lease.instance_id not in state.instances

    asyncio.run(scenario())


def test_explicit_force_destroy_can_terminate_active_owned_lease(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """The destructive UI path may kill active work after exact ownership checks."""

    async def scenario() -> None:
        """Acquire, mark busy, and explicitly force-destroy one owned lease."""
        state = vast_simulator_module.VastSimulatorState(polls_until_running=1)
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
            manager = vast_leases_module.VastLeaseManager(
                api_client=vast_api_module.VastApiClient(
                    state.api_key,
                    base_url=base_url,
                ),
                registry=registry,
                owner_id="comfy-owner",
                runtime_fingerprint=_runtime_fingerprint(),
                launch_spec_factory=_launch_factory(vast_models_module),
                startup_timeout_seconds=2.0,
            )
            lease = await manager.ensure_lease(_profile(vast_models_module))
            manager.begin_activity(lease.instance_id)

            with pytest.raises(RuntimeError, match="active work"):
                await manager.destroy_owned_lease(lease.instance_id)
            assert await manager.destroy_owned_lease(
                lease.instance_id,
                allow_active_work=True,
            )
            assert registry.load().leases == ()
            assert lease.instance_id not in state.instances

    asyncio.run(scenario())


def test_registry_is_versioned_atomic_and_credential_free(
    tmp_path: Any,
    vast_leases_module: Any,
) -> None:
    """Durable lease state must contain no API or SSH credentials."""
    registry = vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path)
    label = vast_leases_module.vast_managed_label(
        "owner", "profile", "b" * 64, "a" * 64
    )
    lease = vast_leases_module.VastLeaseRecord(
        instance_id=9,
        offer_id=7,
        owner_id="owner",
        profile_id="profile",
        profile_name="Default",
        profile_fingerprint="b" * 64,
        runtime_fingerprint="a" * 64,
        label=label,
        actual_status="running",
        ssh_host="ssh.example.invalid",
        ssh_port=2222,
        gpu_name="GPU",
        gpu_count=1,
        gpu_ram_mb=24576,
        cpu_ram_mb=65536,
        hourly_cost_usd=0.5,
        created_at_epoch=1.0,
        last_activity_at_epoch=2.0,
        idle_deadline_epoch=3.0,
    )

    registry.upsert(lease)

    assert registry.load().leases == (lease,)
    payload = registry.config_path.read_text(encoding="utf-8")
    assert json.loads(payload)["version"] == 1
    assert "api_key" not in payload.casefold()
    assert "private_key" not in payload.casefold()
    assert "instance_api_key" not in payload.casefold()


def test_retention_cost_accounts_for_existing_deadline(
    tmp_path: Any,
    vast_api_module: Any,
    vast_leases_module: Any,
    vast_models_module: Any,
) -> None:
    """Automatic placement should charge only the deadline extension on a warm lease."""
    clock = FakeClock()
    manager = vast_leases_module.VastLeaseManager(
        api_client=object(),
        registry=vast_leases_module.VastLeaseRegistry.for_user_directory(tmp_path),
        owner_id="owner",
        runtime_fingerprint=_runtime_fingerprint(),
        launch_spec_factory=_launch_factory(vast_models_module),
        clock=clock,
    )
    profile = _profile(
        vast_models_module,
        maximum_hourly_cost_usd=1.0,
        idle_retention_seconds=3600,
    )
    lease = vast_leases_module.VastLeaseRecord(
        instance_id=9,
        offer_id=7,
        owner_id="owner",
        profile_id="node-17",
        profile_name="default",
        profile_fingerprint=vast_leases_module.vast_profile_fingerprint(profile),
        runtime_fingerprint=_runtime_fingerprint(),
        label=vast_leases_module.vast_managed_label(
            "owner",
            "node-17",
            vast_leases_module.vast_profile_fingerprint(profile),
            _runtime_fingerprint(),
        ),
        actual_status="running",
        ssh_host="host",
        ssh_port=22,
        gpu_name="GPU",
        gpu_count=1,
        gpu_ram_mb=24576,
        cpu_ram_mb=65536,
        hourly_cost_usd=0.5,
        created_at_epoch=clock.value,
        last_activity_at_epoch=clock.value,
        idle_deadline_epoch=clock.value + 3500,
    )

    new_cost = manager.incremental_retention_cost_usd(
        profile,
        existing_lease=None,
        predicted_execution_seconds=60,
    )
    warm_cost = manager.incremental_retention_cost_usd(
        profile,
        existing_lease=lease,
        predicted_execution_seconds=60,
    )

    assert new_cost == pytest.approx(3660 / 3600)
    assert warm_cost == pytest.approx(160 / 3600 * 0.5)
