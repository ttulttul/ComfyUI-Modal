"""Contract tests for the Vast client against the stateful local simulator."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest
from aiohttp import web


@asynccontextmanager
async def _running_simulator(app: web.Application) -> AsyncIterator[str]:
    """Serve one simulator app on an ephemeral loopback port."""
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


def test_client_search_create_poll_manage_and_destroy_lifecycle(
    vast_api_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """The simulator should exercise the complete pre-SSH API lifecycle."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState(polls_until_running=2)
        simulator = vast_simulator_module.VastApiSimulator(state)
        async with _running_simulator(simulator.app) as base_url:
            client = vast_api_module.VastApiClient(
                state.api_key,
                base_url=base_url,
                retry_attempts=1,
            )
            account = await client.verify_credentials()
            profile = vast_models_module.VastResourceProfile(
                profile_id="17",
                profile_name="default",
            )
            offers = await client.search_offers(profile)

            assert account == {"id": 42, "credit": 100.0}
            assert [offer.offer_id for offer in offers] == [1001, 1002, 1003]
            launch = vast_models_module.VastInstanceLaunchSpec(
                image="ghcr.io/example/worker:latest",
                disk_gb=profile.allocated_disk_gb,
                label="comfy-modal:test:default",
                onstart="start-worker",
                environment={},
            )
            created = await client.create_instance(offers[0].offer_id, launch)
            instance = await client.wait_until_ready(
                created.instance_id,
                timeout_seconds=2.0,
                poll_interval_seconds=0.01,
            )

            assert instance.ready_for_ssh is True
            assert instance.ssh_host is not None
            assert instance.ssh_port is not None
            listed = await client.list_instances()
            assert [item.instance_id for item in listed] == [created.instance_id]

            await client.set_instance_state(created.instance_id, "stopped")
            assert (await client.show_instance(created.instance_id)).actual_status == "stopped"
            await client.set_instance_state(created.instance_id, "running")
            await client.destroy_instance(created.instance_id)

            assert await client.list_instances() == ()
            assert state.destroyed_instance_ids == [created.instance_id]
            assert all("Authorization" not in entry for entry in state.request_log)
            assert "instance-secret" not in repr(state.request_log)

    asyncio.run(scenario())


def test_simulator_enforces_capacity_price_and_location_filters(
    vast_api_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Offline offer selection should behave like the documented marketplace query."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState()
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            client = vast_api_module.VastApiClient(state.api_key, base_url=base_url)
            large_us_profile = vast_models_module.VastResourceProfile(
                profile_id="large",
                profile_name="large",
                minimum_gpu_ram_mb=48 * 1024,
                allowed_geolocations=("US",),
                maximum_hourly_cost_usd=1.0,
            )

            offers = await client.search_offers(large_us_profile)

            assert [offer.offer_id for offer in offers] == [1002]

    asyncio.run(scenario())


def test_offer_search_cache_retains_empty_results_until_ttl_and_can_refresh(
    vast_api_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Successful empty searches should be shared, expire, and support bypass."""

    async def scenario() -> None:
        """Exercise a shared cache across the short-lived clients used per prompt."""
        now = [100.0]
        state = vast_simulator_module.VastSimulatorState()
        cache = vast_api_module.VastOfferSearchCache(monotonic=lambda: now[0])
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            profile = vast_models_module.VastResourceProfile(
                profile_id="unavailable",
                profile_name="unavailable",
                maximum_hourly_cost_usd=0.01,
            )
            first_client = vast_api_module.VastApiClient(
                state.api_key,
                base_url=base_url,
                offer_cache=cache,
                offer_cache_ttl_seconds=3600.0,
            )
            second_client = vast_api_module.VastApiClient(
                state.api_key,
                base_url=base_url,
                offer_cache=cache,
                offer_cache_ttl_seconds=3600.0,
            )

            assert await first_client.search_offers(profile) == ()
            assert await second_client.search_offers(profile) == ()
            assert _offer_search_count(state) == 1

            assert (
                await second_client.search_offers(profile, force_refresh=True) == ()
            )
            assert _offer_search_count(state) == 2

            now[0] += 3601.0
            assert await first_client.search_offers(profile) == ()
            assert _offer_search_count(state) == 3

    asyncio.run(scenario())


def test_offer_disappearance_is_classified_without_secret_echo(
    vast_api_module: Any,
    vast_models_module: Any,
    vast_simulator_module: Any,
) -> None:
    """A rent race should be recoverable by the higher-level lease manager."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState(
            create_failures_remaining={1001: 1}
        )
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            client = vast_api_module.VastApiClient(
                state.api_key,
                base_url=base_url,
                retry_attempts=1,
            )
            profile = vast_models_module.VastResourceProfile(
                profile_id="one", profile_name="one"
            )
            offer = (await client.search_offers(profile))[0]
            launch = vast_models_module.VastInstanceLaunchSpec(
                image="worker",
                disk_gb=200,
                label="managed",
                onstart="start",
                environment={},
            )

            with pytest.raises(vast_api_module.VastOfferUnavailableError) as error:
                await client.create_instance(offer.offer_id, launch)

            assert state.api_key not in str(error.value)

    asyncio.run(scenario())


def test_invalid_api_key_is_classified(
    vast_api_module: Any,
    vast_simulator_module: Any,
) -> None:
    """Authentication failures should be actionable and never retried blindly."""

    async def scenario() -> None:
        state = vast_simulator_module.VastSimulatorState()
        async with _running_simulator(
            vast_simulator_module.create_vast_simulator_app(state)
        ) as base_url:
            client = vast_api_module.VastApiClient(
                "wrong-key", base_url=base_url, retry_attempts=1
            )
            with pytest.raises(vast_api_module.VastAuthenticationError):
                await client.verify_credentials()

    asyncio.run(scenario())


def test_client_rejects_non_loopback_plain_http(vast_api_module: Any) -> None:
    """Bearer credentials must not be sent over arbitrary plaintext HTTP."""
    with pytest.raises(ValueError, match="HTTPS or loopback"):
        vast_api_module.VastApiClient(
            "key",
            base_url="http://example.invalid",
        )


def _offer_search_count(state: Any) -> int:
    """Return the number of marketplace search requests seen by the simulator."""
    return sum(
        request["method"] == "POST" and request["path"] == "/api/v0/bundles/"
        for request in state.request_log
    )
