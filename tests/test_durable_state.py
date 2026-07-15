"""Tests for durable invocation metadata and binary object storage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest


def test_content_addressed_store_deduplicates_and_validates(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """Identical objects should share a path and be committed only once."""
    commits: list[bool] = []
    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        commit_callback=lambda: commits.append(True),
    )

    first_ref = store.put("results", b"durable-result")
    second_ref = store.put("results", b"durable-result")

    assert first_ref == second_ref
    assert store.get(first_ref) == b"durable-result"
    assert commits == [True]


def test_content_addressed_store_rejects_corruption(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """Reads should fail rather than returning corrupted durable output bytes."""
    store = durable_state_module.FileDurableObjectStore(tmp_path)
    object_ref = store.put("results", b"expected")
    (tmp_path / object_ref.object_path).write_bytes(b"corrupt")

    with pytest.raises(durable_state_module.DurableStateError, match="size mismatch"):
        store.get(object_ref)


def test_content_addressed_store_rejects_unsafe_reference_paths(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """Persisted references must never escape the configured object root."""
    store = durable_state_module.FileDurableObjectStore(tmp_path)
    object_ref = durable_state_module.DurableObjectRef(
        object_path="../outside.bin",
        sha256="0" * 64,
        size_bytes=0,
    )

    with pytest.raises(durable_state_module.DurableStateError, match="Unsafe"):
        store.get(object_ref)


def test_content_addressed_store_reloads_missing_mounted_object(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """A stale mounted Volume view should reload once before reporting a miss."""
    payload = b"arrived-from-another-container"
    digest = hashlib.sha256(payload).hexdigest()
    object_ref = durable_state_module.DurableObjectRef(
        object_path=f"results/{digest[:2]}/{digest}.bin",
        sha256=digest,
        size_bytes=len(payload),
    )
    reload_count = 0

    def reload_volume() -> None:
        """Simulate Modal exposing an object after Volume.reload()."""
        nonlocal reload_count
        reload_count += 1
        target_path = tmp_path / object_ref.object_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(payload)

    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        reload_callback=reload_volume,
    )

    assert store.get(object_ref) == payload
    assert reload_count == 1


def test_remote_invocation_id_is_stable_and_input_sensitive(
    durable_state_module: Any,
) -> None:
    """Retries should reuse an id while materially different inputs get a new id."""
    payload = {"component_id": "component-1", "payload_kind": "subgraph"}

    first_id = durable_state_module.stable_remote_invocation_id(payload, b"inputs-a")
    retry_id = durable_state_module.stable_remote_invocation_id(
        {**payload, "invocation_id": "ignored"},
        b"inputs-a",
    )
    changed_id = durable_state_module.stable_remote_invocation_id(payload, b"inputs-b")

    assert first_id == retry_id
    assert first_id != changed_id


def test_remote_invocation_record_round_trips_object_result(
    durable_state_module: Any,
) -> None:
    """Modal Dict-safe records should retain object-backed result metadata."""
    object_ref = durable_state_module.DurableObjectRef(
        object_path="results/ab/abcdef.bin",
        sha256="a" * 64,
        size_bytes=123,
    )
    record = durable_state_module.RemoteInvocationRecord(
        invocation_id="RIV_example",
        state="completed",
        attempt=2,
        created_at=10.0,
        updated_at=20.0,
        result_object=object_ref,
    )

    restored = durable_state_module.RemoteInvocationRecord.from_payload(
        record.to_payload()
    )

    assert restored == record
