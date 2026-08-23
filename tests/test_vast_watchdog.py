"""Tests for the Vast in-instance idle self-destruction watchdog."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any

import pytest


@pytest.fixture(scope="module")
def watchdog_module(extension_package: Any) -> Any:
    """Return the remote Vast watchdog module."""
    return importlib.import_module(f"{extension_package.__name__}.remote.vast_watchdog")


@dataclass
class FakeDestroyer:
    """Record restricted self-destruction requests."""

    destroyed: list[int] = field(default_factory=list)

    async def destroy(self, instance_id: int) -> None:
        """Record one instance identity."""
        self.destroyed.append(instance_id)


def _write_snapshot(path: Any, snapshot: Any) -> None:
    """Write one canonical controller snapshot."""
    path.write_bytes(snapshot.to_json_bytes())


def test_expired_idle_snapshot_destroys_exact_instance(
    tmp_path: Any,
    watchdog_module: Any,
) -> None:
    """A verified elapsed deadline should invoke the restricted destroyer once."""
    state_path = tmp_path / "watchdog.json"
    snapshot = watchdog_module.VastWatchdogSnapshot(
        instance_id=42,
        owner_label="comfy-modal-vast:owner:profile",
        idle_deadline_epoch=100.0,
        active_invocations=0,
        updated_at_epoch=90.0,
    )
    _write_snapshot(state_path, snapshot)
    destroyer = FakeDestroyer()

    destroyed = asyncio.run(
        watchdog_module.run_watchdog_once(
            state_path=state_path,
            expected_instance_id=42,
            destroyer=destroyer,
            now_epoch=101.0,
        )
    )

    assert destroyed is True
    assert destroyer.destroyed == [42]


def test_active_or_unexpired_snapshot_preserves_instance(
    tmp_path: Any,
    watchdog_module: Any,
) -> None:
    """Neither active work nor a future deadline may trigger destruction."""
    state_path = tmp_path / "watchdog.json"
    destroyer = FakeDestroyer()
    active = watchdog_module.VastWatchdogSnapshot(
        instance_id=42,
        owner_label="owner",
        idle_deadline_epoch=100.0,
        active_invocations=1,
        updated_at_epoch=90.0,
    )
    _write_snapshot(state_path, active)
    assert (
        asyncio.run(
            watchdog_module.run_watchdog_once(
                state_path=state_path,
                expected_instance_id=42,
                destroyer=destroyer,
                now_epoch=101.0,
            )
        )
        is False
    )
    future = watchdog_module.VastWatchdogSnapshot(
        instance_id=42,
        owner_label="owner",
        idle_deadline_epoch=200.0,
        active_invocations=0,
        updated_at_epoch=90.0,
    )
    _write_snapshot(state_path, future)
    assert (
        asyncio.run(
            watchdog_module.run_watchdog_once(
                state_path=state_path,
                expected_instance_id=42,
                destroyer=destroyer,
                now_epoch=101.0,
            )
        )
        is False
    )
    assert destroyer.destroyed == []


def test_mismatched_instance_identity_fails_closed(
    tmp_path: Any,
    watchdog_module: Any,
) -> None:
    """Never apply a stale state file written for another instance."""
    state_path = tmp_path / "watchdog.json"
    snapshot = watchdog_module.VastWatchdogSnapshot(
        instance_id=7,
        owner_label="owner",
        idle_deadline_epoch=100.0,
        active_invocations=0,
        updated_at_epoch=90.0,
    )
    _write_snapshot(state_path, snapshot)

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            watchdog_module.run_watchdog_once(
                state_path=state_path,
                expected_instance_id=42,
                destroyer=FakeDestroyer(),
                now_epoch=101.0,
            )
        )

def test_malformed_state_never_becomes_an_expired_default(
    tmp_path: Any,
    watchdog_module: Any,
) -> None:
    """Corrupt controller state should preserve rather than destroy the instance."""
    state_path = tmp_path / "watchdog.json"
    state_path.write_text("{bad-json", encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable"):
        asyncio.run(
            watchdog_module.run_watchdog_once(
                state_path=state_path,
                expected_instance_id=42,
                destroyer=FakeDestroyer(),
                now_epoch=101.0,
            )
        )
