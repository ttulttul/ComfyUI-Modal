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


def test_content_addressed_store_batches_commits_across_multiple_objects(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """One logical invocation should commit all new objects in one callback."""
    commits: list[bool] = []
    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        commit_callback=lambda: commits.append(True),
    )

    with store.batch_commits():
        store.put("bridges", b"bridge-inputs")
        store.put("bridges", b"bridge-output")
        store.put("results", b"invocation-result")

    assert commits == [True]


def test_content_addressed_store_transfers_deferred_batch_before_commit(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """A streaming worker's deferred writes should join its caller's completion batch."""
    commits: list[bool] = []
    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        commit_callback=lambda: commits.append(True),
    )

    with store.batch_commits(commit_on_exit=False) as worker_batch:
        store.put("bridges", b"worker-object")
    with store.batch_commits() as completion_batch:
        completion_batch.absorb(worker_batch)
        store.put("results", b"main-object")

    assert worker_batch.wrote_object is False
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


def test_content_addressed_store_reads_missing_object_from_committed_storage(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """A stale mount should use direct committed reads without materializing files."""
    payload = b"arrived-from-another-container"
    digest = hashlib.sha256(payload).hexdigest()
    object_ref = durable_state_module.DurableObjectRef(
        object_path=f"results/{digest[:2]}/{digest}.bin",
        sha256=digest,
        size_bytes=len(payload),
    )
    requested_paths: list[str] = []

    def read_committed_object(object_path: str) -> bytes:
        """Return one object from a simulated committed-volume API."""
        requested_paths.append(object_path)
        return payload

    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        committed_read_callback=read_committed_object,
    )

    assert store.get(object_ref) == payload
    assert requested_paths == [object_ref.object_path]
    assert not (tmp_path / object_ref.object_path).exists()


def test_content_addressed_store_recovers_truncated_mount_from_committed_storage(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """A stale partial mount should fall back to authoritative committed bytes."""
    payload = b"complete-durable-object"
    digest = hashlib.sha256(payload).hexdigest()
    object_ref = durable_state_module.DurableObjectRef(
        object_path=f"results/{digest[:2]}/{digest}.bin",
        sha256=digest,
        size_bytes=len(payload),
    )
    mounted_path = tmp_path / object_ref.object_path
    mounted_path.parent.mkdir(parents=True)
    mounted_path.write_bytes(payload[:-5])
    requested_paths: list[str] = []

    def read_committed_object(object_path: str) -> bytes:
        """Return the complete object from a simulated committed-volume API."""
        requested_paths.append(object_path)
        return payload

    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        committed_read_callback=read_committed_object,
    )

    assert store.get(object_ref) == payload
    assert requested_paths == [object_ref.object_path]


def test_content_addressed_store_reports_missing_committed_object(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """A missing mounted and committed object should remain a durable-state error."""
    object_ref = durable_state_module.DurableObjectRef(
        object_path="results/ab/missing.bin",
        sha256="a" * 64,
        size_bytes=1,
    )

    def read_missing_object(object_path: str) -> bytes:
        """Simulate Modal reporting that the committed object is absent."""
        raise FileNotFoundError(object_path)

    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        committed_read_callback=read_missing_object,
    )

    with pytest.raises(durable_state_module.DurableStateError, match="was not found"):
        store.get(object_ref)


def test_content_addressed_store_validates_direct_committed_read(
    durable_state_module: Any,
    tmp_path: Path,
) -> None:
    """Direct committed reads must retain content-address integrity checks."""
    expected_payload = b"expected"
    object_ref = durable_state_module.DurableObjectRef(
        object_path="results/ab/expected.bin",
        sha256=hashlib.sha256(expected_payload).hexdigest(),
        size_bytes=len(expected_payload),
    )

    def read_corrupt_object(object_path: str) -> bytes:
        """Return same-sized bytes with the wrong content digest."""
        del object_path
        return b"corrupt!"

    store = durable_state_module.FileDurableObjectStore(
        tmp_path,
        committed_read_callback=read_corrupt_object,
    )

    with pytest.raises(durable_state_module.DurableStateError, match="SHA256 mismatch"):
        store.get(object_ref)


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
