"""Tests for per-prompt remote GPU time accounting."""

from __future__ import annotations

from typing import Any

import pytest

_RATES = {"RTX-PRO-6000": 0.001}


def _payload(prompt_id: str = "prompt-1", gpu: str = "RTX-PRO-6000") -> dict[str, Any]:
    """Return a minimal streamed payload with two titled nodes."""
    return {
        "prompt_id": prompt_id,
        "modal_gpu": gpu,
        "extra_data": {"client_id": "client-1"},
        "subgraph_prompt": {
            "116": {"class_type": "KSampler", "_meta": {"title": "Final pass"}},
            "117": {"class_type": "VAEDecode"},
        },
    }


def _run_invocation(
    gpu_usage_module: Any,
    events: list[tuple[float, dict[str, Any]]],
    *,
    finished_at: float,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Feed timed events through one invocation and return it."""
    invocation = gpu_usage_module.InvocationGpuUsage(
        payload=payload or _payload(), dispatched_at=0.0
    )
    for received_at, event in events:
        invocation.observe(received_at, event)
    return invocation


def _executing(node_id: str) -> dict[str, Any]:
    """Return a worker status event announcing that one node started."""
    return {"kind": "progress", "phase": "executing", "active_node_id": node_id}


def test_invocation_charges_each_node_until_the_next_one_starts(gpu_usage_module: Any) -> None:
    """Busy time splits into setup, per-node work, and returning results."""
    ledger = gpu_usage_module.GpuUsageLedger()
    invocation = _run_invocation(
        gpu_usage_module,
        [
            (5.0, {"kind": "remote_logs", "task_id": "ta-1"}),
            (7.0, _executing("116")),
            (8.0, {"kind": "progress", "event_type": "node_progress", "node_id": "116"}),
            (27.0, _executing("117")),
            (30.0, {"kind": "progress", "phase": "execution_success"}),
        ],
        finished_at=33.0,
    )

    client_id, snapshot = ledger.record(
        invocation, finished_at=33.0, succeeded=True, usd_per_second_by_gpu=_RATES
    )

    assert client_id == "client-1"
    assert snapshot["wait_seconds"] == pytest.approx(5.0)
    assert snapshot["busy_seconds"] == pytest.approx(28.0)
    assert snapshot["setup_seconds"] == pytest.approx(2.0)
    assert snapshot["return_seconds"] == pytest.approx(3.0)
    assert snapshot["estimated_cost_usd"] == pytest.approx(0.028)
    assert [
        (node["label"], node["seconds"], node["runs"]) for node in snapshot["nodes"]
    ] == [("Final pass #116", 20.0, 1), ("VAEDecode #117", 3.0, 1)]


def test_ledger_sums_items_per_node_and_prices_unknown_gpus_as_none(
    gpu_usage_module: Any,
) -> None:
    """Mapped items add up per node; a GPU without a price leaves the cost unknown."""
    ledger = gpu_usage_module.GpuUsageLedger()
    for _ in range(3):
        invocation = _run_invocation(
            gpu_usage_module,
            [(1.0, _executing("116")), (11.0, {"kind": "progress", "phase": "execution_success"})],
            finished_at=11.0,
        )
        _client_id, snapshot = ledger.record(
            invocation, finished_at=11.0, succeeded=True, usd_per_second_by_gpu=_RATES
        )
    assert snapshot["invocation_count"] == 3
    assert snapshot["nodes"][0]["seconds"] == pytest.approx(30.0)
    assert snapshot["nodes"][0]["runs"] == 3

    other = _run_invocation(
        gpu_usage_module,
        [(1.0, _executing("117"))],
        finished_at=2.0,
        payload=_payload(gpu="B300"),
    )
    _client_id, snapshot = ledger.record(
        other, finished_at=2.0, succeeded=False, usd_per_second_by_gpu=_RATES
    )
    assert snapshot["failed_invocation_count"] == 1
    assert snapshot["seconds_by_gpu"] == {"B300": 1.0, "RTX-PRO-6000": 30.0}
    assert snapshot["estimated_cost_usd"] is None


def test_call_without_node_events_counts_as_setup(gpu_usage_module: Any) -> None:
    """Session cleanup calls run no nodes, so their busy time is setup."""
    ledger = gpu_usage_module.GpuUsageLedger()
    invocation = _run_invocation(
        gpu_usage_module, [(2.0, {"kind": "remote_logs", "task_id": "ta-2"})], finished_at=4.0
    )

    _client_id, snapshot = ledger.record(
        invocation, finished_at=4.0, succeeded=True, usd_per_second_by_gpu=_RATES
    )

    assert snapshot["setup_seconds"] == pytest.approx(2.0)
    assert snapshot["nodes"] == []


def test_ledger_ignores_calls_without_a_prompt_and_keeps_recent_prompts(
    gpu_usage_module: Any,
) -> None:
    """Untracked calls are skipped and the ledger evicts its oldest prompts."""
    ledger = gpu_usage_module.GpuUsageLedger(limit=2)
    unowned = gpu_usage_module.InvocationGpuUsage(payload={}, dispatched_at=0.0)
    assert (
        ledger.record(unowned, finished_at=1.0, succeeded=True, usd_per_second_by_gpu=_RATES)
        is None
    )
    for prompt_id in ("a", "b", "c"):
        invocation = gpu_usage_module.InvocationGpuUsage(
            payload=_payload(prompt_id), dispatched_at=0.0
        )
        ledger.record(invocation, finished_at=1.0, succeeded=True, usd_per_second_by_gpu=_RATES)

    assert ledger.snapshot("a", _RATES) is None
    assert ledger.snapshot("c", _RATES)["prompt_id"] == "c"
