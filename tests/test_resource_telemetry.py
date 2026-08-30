"""Tests for backend-neutral remote memory telemetry."""

from __future__ import annotations

from typing import Any


def test_sampler_publishes_current_and_peak_memory(
    resource_telemetry_module: Any,
) -> None:
    """The terminal sample should preserve peaks observed during execution."""
    snapshots = iter(
        (
            resource_telemetry_module.ResourceMemorySnapshot(100, 1000, 200, 2000),
            resource_telemetry_module.ResourceMemorySnapshot(350, 1000, 800, 2000),
            resource_telemetry_module.ResourceMemorySnapshot(250, 1000, 500, 2000),
        )
    )
    events: list[dict[str, Any]] = []
    sampler = resource_telemetry_module.RemoteResourceTelemetrySampler(
        events.append,
        interval_seconds=60.0,
        sample_reader=lambda: next(snapshots),
    )

    sampler.start()
    sampler._publish_sample(active=True)
    sampler.stop()

    assert [event["active"] for event in events] == [True, True, False]
    assert events[-1]["cpu_memory_bytes"] == 250
    assert events[-1]["cpu_memory_peak_bytes"] == 350
    assert events[-1]["gpu_memory_bytes"] == 500
    assert events[-1]["gpu_memory_peak_bytes"] == 800
    assert events[-1]["cpu_memory_total_bytes"] == 1000
    assert events[-1]["gpu_memory_total_bytes"] == 2000


def test_snapshot_reader_returns_non_negative_memory(
    resource_telemetry_module: Any,
) -> None:
    """The production reader should work on CPU-only local test hosts."""
    snapshot = resource_telemetry_module.read_resource_memory_snapshot()

    assert snapshot.cpu_memory_bytes > 0
    assert snapshot.cpu_memory_total_bytes >= snapshot.cpu_memory_bytes
    assert snapshot.gpu_memory_bytes >= 0
    assert snapshot.gpu_memory_total_bytes >= snapshot.gpu_memory_bytes
