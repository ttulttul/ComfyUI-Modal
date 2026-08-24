"""Stateful local Vast.ai API simulator for tests and offline development."""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from aiohttp import web

logger = logging.getLogger(__name__)


def default_vast_simulator_offers() -> list[dict[str, Any]]:
    """Return deterministic offers spanning price and capacity tradeoffs."""
    common = {
        "rentable": True,
        "rented": False,
        "gpu_arch": "nvidia",
        "cpu_arch": "amd64",
        "gpu_frac": 1.0,
        "gpu_display_active": False,
        "direct_port_count": 4,
        "duration": 30 * 86400,
        "cuda_max_good": 13.0,
        "verification": "verified",
        "verified": True,
        "storage_cost": 0.12,
    }
    return [
        {
            **common,
            "id": 1001,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24 * 1024,
            "gpu_total_ram": 24 * 1024,
            "total_flops": 82.6,
            "cpu_ram": 64 * 1024,
            "cpu_cores": 16,
            "cpu_cores_effective": 12.0,
            "disk_space": 1000.0,
            "reliability": 0.995,
            "dlperf": 45.0,
            "inet_down": 800.0,
            "inet_up": 500.0,
            "dph_total": 0.42,
            "geolocation": "Vancouver, CA",
        },
        {
            **common,
            "id": 1002,
            "gpu_name": "RTX 6000 Ada",
            "num_gpus": 1,
            "gpu_ram": 48 * 1024,
            "gpu_total_ram": 48 * 1024,
            "total_flops": 91.1,
            "cpu_ram": 128 * 1024,
            "cpu_cores": 32,
            "cpu_cores_effective": 24.0,
            "disk_space": 2000.0,
            "reliability": 0.999,
            "dlperf": 62.0,
            "inet_down": 1200.0,
            "inet_up": 800.0,
            "dph_total": 0.74,
            "geolocation": "Seattle, US",
        },
        {
            **common,
            "id": 1003,
            "gpu_name": "H100 SXM",
            "num_gpus": 1,
            "gpu_ram": 80 * 1024,
            "gpu_total_ram": 80 * 1024,
            "total_flops": 989.0,
            "cpu_ram": 256 * 1024,
            "cpu_cores": 64,
            "cpu_cores_effective": 48.0,
            "disk_space": 4000.0,
            "reliability": 0.9998,
            "dlperf": 180.0,
            "inet_down": 2200.0,
            "inet_up": 1200.0,
            "dph_total": 1.96,
            "geolocation": "Dallas, US",
        },
    ]


@dataclass
class VastSimulatorState:
    """Hold mutable marketplace and instance state for the simulator."""

    api_key: str = "vast-test-key"
    offers: list[dict[str, Any]] = field(default_factory=default_vast_simulator_offers)
    instances: dict[int, dict[str, Any]] = field(default_factory=dict)
    destroyed_instance_ids: list[int] = field(default_factory=list)
    request_log: list[dict[str, Any]] = field(default_factory=list)
    create_failures_remaining: dict[int, int] = field(default_factory=dict)
    polls_until_running: int = 2
    next_instance_id: int = 9001

    def record(self, request: web.Request, body: object | None = None) -> None:
        """Record one request without retaining its authorization header."""
        self.request_log.append(
            {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
                "body": copy.deepcopy(body),
            }
        )


class VastApiSimulator:
    """Serve the Vast endpoints used by the extension on a local aiohttp app."""

    def __init__(self, state: VastSimulatorState | None = None) -> None:
        """Create one simulator around supplied or default mutable state."""
        self.state = state or VastSimulatorState()
        self.app = web.Application(middlewares=[self._authenticate])
        self._register_routes()

    @web.middleware
    async def _authenticate(
        self,
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        """Require the configured bearer token on API routes."""
        expected = f"Bearer {self.state.api_key}"
        if request.headers.get("Authorization") != expected:
            self.state.record(request)
            return web.json_response({"error": "Unauthorized"}, status=401)
        return await handler(request)

    def _register_routes(self) -> None:
        """Register the supported Vast API surface."""
        self.app.router.add_get("/api/v0/users/current/", self.current_user)
        self.app.router.add_post("/api/v0/bundles/", self.search_offers)
        self.app.router.add_put("/api/v0/asks/{offer_id}/", self.create_instance)
        self.app.router.add_get(
            "/api/v0/instances/{instance_id}/", self.show_instance
        )
        self.app.router.add_put(
            "/api/v0/instances/{instance_id}/", self.manage_instance
        )
        self.app.router.add_delete(
            "/api/v0/instances/{instance_id}/", self.destroy_instance
        )
        self.app.router.add_get("/api/v1/instances/", self.list_instances)

    async def current_user(self, request: web.Request) -> web.Response:
        """Return a deterministic account record."""
        self.state.record(request)
        return web.json_response(
            {
                "id": 42,
                "email": "simulator@example.invalid",
                "credit": 100.0,
                "ssh_key": "ssh-ed25519 simulator-key",
            }
        )

    async def search_offers(self, request: web.Request) -> web.Response:
        """Apply documented filter operators and deterministic ordering."""
        body = await _json_object(request)
        self.state.record(request, body)
        offers = [
            copy.deepcopy(offer)
            for offer in self.state.offers
            if _offer_matches(offer, body)
        ]
        order = body.get("order")
        if isinstance(order, list):
            for order_item in reversed(order):
                if not isinstance(order_item, list) or len(order_item) != 2:
                    continue
                field_name, direction = str(order_item[0]), str(order_item[1])
                offers.sort(
                    key=lambda offer: _sortable_value(offer.get(field_name)),
                    reverse=direction.casefold() == "desc",
                )
        limit = max(1, min(1000, int(body.get("limit", 100))))
        return web.json_response({"offers": offers[:limit]})

    async def create_instance(self, request: web.Request) -> web.Response:
        """Rent one available offer or simulate an availability race."""
        body = await _json_object(request)
        self.state.record(request, body)
        offer_id = int(request.match_info["offer_id"])
        offer = next(
            (candidate for candidate in self.state.offers if candidate["id"] == offer_id),
            None,
        )
        if offer is None or not offer.get("rentable") or offer.get("rented"):
            return web.json_response({"error": "Offer is no longer available"}, status=404)
        failures_remaining = self.state.create_failures_remaining.get(offer_id, 0)
        if failures_remaining > 0:
            self.state.create_failures_remaining[offer_id] = failures_remaining - 1
            return web.json_response({"error": "Offer is no longer available"}, status=404)
        if not body.get("cancel_unavail", False):
            return web.json_response(
                {"error": "Simulator requires cancel_unavail for deterministic tests."},
                status=400,
            )
        instance_id = self.state.next_instance_id
        self.state.next_instance_id += 1
        offer["rented"] = True
        instance = {
            **copy.deepcopy(offer),
            "id": instance_id,
            "ask_contract_id": offer_id,
            "actual_status": "loading",
            "intended_status": "running",
            "cur_state": "loading",
            "status_msg": "worker-layer: Download complete",
            "next_state": "running",
            "label": body.get("label"),
            "image_uuid": body.get("image"),
            "image_runtype": body.get("runtype", "ssh_direct"),
            "onstart": body.get("onstart"),
            "extra_env": body.get("env", {}),
            "disk_space": body.get("disk", offer.get("disk_space", 0)),
            "ssh_host": f"127.0.0.{(instance_id % 200) + 1}",
            "ssh_port": 22000 + instance_id % 1000,
            "_poll_count": 0,
        }
        self.state.instances[instance_id] = instance
        return web.json_response(
            {
                "success": True,
                "new_contract": instance_id,
                "instance_api_key": f"instance-secret-{instance_id}",
            }
        )

    async def show_instance(self, request: web.Request) -> web.Response:
        """Advance loading instances after the configured number of polls."""
        self.state.record(request)
        instance = self._instance(request)
        if instance is None:
            return web.json_response({"instances": None})
        instance["_poll_count"] = int(instance.get("_poll_count", 0)) + 1
        if (
            instance["actual_status"] == "loading"
            and instance["_poll_count"] >= self.state.polls_until_running
        ):
            instance["actual_status"] = "running"
            instance["cur_state"] = "running"
            instance["status_msg"] = "Worker ready"
        return web.json_response({"instances": _public_instance(instance)})

    async def list_instances(self, request: web.Request) -> web.Response:
        """Return all non-destroyed simulated instances in one page."""
        self.state.record(request)
        instances = [
            _public_instance(instance)
            for _instance_id, instance in sorted(self.state.instances.items())
        ]
        return web.json_response(
            {
                "success": True,
                "instances_found": len(instances),
                "total_instances": len(instances),
                "instances": instances,
                "next_token": None,
            }
        )

    async def manage_instance(self, request: web.Request) -> web.Response:
        """Set instance state or label using Vast's shared PUT route."""
        body = await _json_object(request)
        self.state.record(request, body)
        instance = self._instance(request)
        if instance is None:
            return web.json_response({"error": "Instance not found"}, status=404)
        state = body.get("state")
        if state is not None:
            if state not in {"running", "stopped"}:
                return web.json_response({"error": "Invalid state"}, status=400)
            instance["actual_status"] = state
            instance["intended_status"] = state
            instance["cur_state"] = state
            instance["next_state"] = state
        if "label" in body:
            instance["label"] = str(body["label"])
        return web.json_response({"success": True})

    async def destroy_instance(self, request: web.Request) -> web.Response:
        """Permanently remove one instance and release its offer."""
        self.state.record(request)
        instance_id = int(request.match_info["instance_id"])
        instance = self.state.instances.pop(instance_id, None)
        if instance is None:
            return web.json_response({"error": "Instance not found"}, status=404)
        offer_id = instance.get("ask_contract_id")
        for offer in self.state.offers:
            if offer.get("id") == offer_id:
                offer["rented"] = False
        self.state.destroyed_instance_ids.append(instance_id)
        return web.json_response(
            {"success": True, "msg": "Instance destroyed successfully"}
        )

    def _instance(self, request: web.Request) -> dict[str, Any] | None:
        """Return the instance named by one route."""
        return self.state.instances.get(int(request.match_info["instance_id"]))


def create_vast_simulator_app(
    state: VastSimulatorState | None = None,
) -> web.Application:
    """Return an aiohttp application implementing the local Vast simulator."""
    return VastApiSimulator(state).app


async def _json_object(request: web.Request) -> dict[str, Any]:
    """Read one JSON request body and require an object."""
    try:
        payload = await request.json()
    except ValueError as exc:
        raise web.HTTPBadRequest(text="Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="Request body must be a JSON object.")
    return payload


def _offer_matches(offer: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    """Apply the filter subset used by the extension's marketplace queries."""
    ignored_fields = {"limit", "order", "type", "allocated_storage"}
    for field_name, condition in query.items():
        if field_name in ignored_fields:
            continue
        if not isinstance(condition, Mapping):
            continue
        value = offer.get(field_name)
        if not _matches_condition(value, condition):
            return False
    query_type = str(query.get("type") or "on-demand")
    offer_type = str(offer.get("type") or "on-demand")
    return query_type == offer_type


def _matches_condition(value: object, condition: Mapping[str, Any]) -> bool:
    """Apply documented eq/neq/range/membership operators."""
    for operator, expected in condition.items():
        if operator == "eq" and value != expected:
            return False
        if operator == "neq" and value == expected:
            return False
        if operator == "gte" and not (_numeric(value) >= _numeric(expected)):
            return False
        if operator == "lte" and not (_numeric(value) <= _numeric(expected)):
            return False
        if operator == "gt" and not (_numeric(value) > _numeric(expected)):
            return False
        if operator == "lt" and not (_numeric(value) < _numeric(expected)):
            return False
        if operator in {"in", "notin"}:
            expected_values = expected if isinstance(expected, list) else []
            normalized_value = str(value or "").casefold()
            contained = any(
                normalized_value == str(candidate).casefold()
                or normalized_value.endswith(f", {str(candidate).casefold()}")
                for candidate in expected_values
            )
            if operator == "in" and not contained:
                return False
            if operator == "notin" and contained:
                return False
    return True


def _numeric(value: object) -> float:
    """Return a number used by simulator filter comparisons."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return float("-inf")


def _sortable_value(value: object) -> tuple[int, object]:
    """Return a stable key across missing, numeric, and textual fields."""
    if value is None:
        return (2, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _public_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Return one instance record without simulator-only fields."""
    return {
        key: copy.deepcopy(value)
        for key, value in instance.items()
        if not str(key).startswith("_")
    }


def _parser() -> argparse.ArgumentParser:
    """Return the standalone simulator command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--api-key", default="vast-test-key")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Vast simulator until interrupted."""
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    state = VastSimulatorState(api_key=arguments.api_key)
    logger.info(
        "Starting Vast API simulator on http://%s:%d with %d offers.",
        arguments.host,
        arguments.port,
        len(state.offers),
    )
    web.run_app(
        create_vast_simulator_app(state),
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":  # pragma: no cover - exercised manually.
    main()


__all__ = [
    "VastApiSimulator",
    "VastSimulatorState",
    "create_vast_simulator_app",
    "default_vast_simulator_offers",
    "main",
]
