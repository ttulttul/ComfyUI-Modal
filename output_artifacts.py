"""Capture remote ComfyUI outputs and materialize them in a local output tree."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

if __package__:
    from .instance_identity import MODAL_APP_PREFIX
else:  # pragma: no cover - the stable cloud entrypoint imports this module top-level.
    from instance_identity import MODAL_APP_PREFIX

logger = logging.getLogger(__name__)

_REMOTE_EXECUTION_RESULT_MAGIC = b"CMODALR1"
_REMOTE_EXECUTION_RESULT_VERSION = 1
_RESULT_HEADER_LENGTH_BYTES = 8
_MAX_RESULT_HEADER_BYTES = 16 * 1024 * 1024
_EPOCH_DIGITS = 9
_COPY_BUFFER_BYTES = 1024 * 1024
_UNSAFE_IDENTIFIER_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")


class RemoteOutputArtifactError(RuntimeError):
    """Raised when a remote output artifact cannot be captured or restored safely."""


@dataclass(frozen=True)
class OutputFileState:
    """Filesystem attributes used to identify a new or replaced output file."""

    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class RemoteOutputSnapshot:
    """State of one remote ComfyUI output tree before payload execution."""

    output_directory: Path
    files: Mapping[str, OutputFileState]


@dataclass(frozen=True)
class RemoteOutputArtifact:
    """One regular file produced beneath the remote ComfyUI output directory."""

    relative_path: str
    payload: bytes


@dataclass(frozen=True)
class RemoteExecutionResult:
    """Serialized node outputs plus any files produced by the remote invocation."""

    outputs: bytes
    artifacts: tuple[RemoteOutputArtifact, ...] = ()
    completed_epoch: int | None = None


def _validated_relative_path(relative_path: str) -> Path:
    """Return a safe relative artifact path or raise a transport error."""
    if not relative_path or "\x00" in relative_path:
        raise RemoteOutputArtifactError(
            "Remote output artifact paths must not be empty."
        )
    path = Path(relative_path)
    if path.is_absolute() or path.name in {"", ".", ".."} or ".." in path.parts:
        raise RemoteOutputArtifactError(
            f"Unsafe remote output artifact path {relative_path!r}."
        )
    return path


def _iter_regular_output_files(
    output_directory: Path,
) -> Iterator[tuple[str, Path, OutputFileState]]:
    """Yield regular non-symlink files beneath an output directory in stable order."""
    if not output_directory.exists():
        return
    candidates = sorted(
        output_directory.rglob("*"),
        key=lambda path: path.as_posix(),
    )
    for candidate in candidates:
        if candidate.is_symlink():
            logger.debug(
                "Skipping symlink in remote ComfyUI output tree: %s",
                candidate,
            )
            continue
        try:
            candidate_stat = candidate.stat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(candidate_stat.st_mode):
            continue
        relative_path = candidate.relative_to(output_directory).as_posix()
        yield (
            relative_path,
            candidate,
            OutputFileState(
                size_bytes=candidate_stat.st_size,
                modified_ns=candidate_stat.st_mtime_ns,
                changed_ns=candidate_stat.st_ctime_ns,
            ),
        )


def snapshot_output_directory(output_directory: Path) -> RemoteOutputSnapshot:
    """Capture the files present before a remote ComfyUI payload executes."""
    resolved_directory = output_directory.expanduser().resolve()
    return RemoteOutputSnapshot(
        output_directory=resolved_directory,
        files={
            relative_path: file_state
            for relative_path, _candidate, file_state in _iter_regular_output_files(
                resolved_directory
            )
        },
    )


def collect_output_artifacts(
    snapshot: RemoteOutputSnapshot,
) -> tuple[RemoteOutputArtifact, ...]:
    """Read files created or replaced after a remote output snapshot."""
    artifacts: list[RemoteOutputArtifact] = []
    for relative_path, candidate, file_state in _iter_regular_output_files(
        snapshot.output_directory
    ):
        if snapshot.files.get(relative_path) == file_state:
            continue
        try:
            payload = candidate.read_bytes()
        except FileNotFoundError:
            logger.debug("Remote output disappeared before capture: %s", candidate)
            continue
        artifacts.append(
            RemoteOutputArtifact(
                relative_path=_validated_relative_path(relative_path).as_posix(),
                payload=payload,
            )
        )
    return tuple(artifacts)


def capture_execution_result(
    outputs: bytes,
    snapshot: RemoteOutputSnapshot,
    *,
    completed_epoch: int | None = None,
) -> bytes:
    """Bundle serialized node outputs with files changed during remote execution."""
    artifacts = collect_output_artifacts(snapshot)
    if not artifacts:
        return bytes(outputs)
    resolved_epoch = int(time.time()) if completed_epoch is None else completed_epoch
    return pack_remote_execution_result(
        RemoteExecutionResult(
            outputs=bytes(outputs),
            artifacts=artifacts,
            completed_epoch=resolved_epoch,
        )
    )


def _artifact_header(artifact: RemoteOutputArtifact) -> dict[str, Any]:
    """Return validated metadata for one binary artifact attachment."""
    relative_path = _validated_relative_path(artifact.relative_path).as_posix()
    return {
        "relative_path": relative_path,
        "size_bytes": len(artifact.payload),
        "sha256": hashlib.sha256(artifact.payload).hexdigest(),
    }


def pack_remote_execution_result(result: RemoteExecutionResult) -> bytes:
    """Encode one artifact-bearing execution result as a compact binary envelope."""
    if not result.artifacts:
        return bytes(result.outputs)
    if (
        result.completed_epoch is None
        or isinstance(result.completed_epoch, bool)
        or result.completed_epoch < 0
    ):
        raise RemoteOutputArtifactError(
            "Artifact-bearing remote results require a non-negative completion epoch."
        )
    artifact_headers = [_artifact_header(artifact) for artifact in result.artifacts]
    header = json.dumps(
        {
            "version": _REMOTE_EXECUTION_RESULT_VERSION,
            "completed_epoch": result.completed_epoch,
            "outputs_length": len(result.outputs),
            "artifacts": artifact_headers,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(header) > _MAX_RESULT_HEADER_BYTES:
        raise RemoteOutputArtifactError(
            "Remote output result header is unreasonably large."
        )
    return b"".join(
        (
            _REMOTE_EXECUTION_RESULT_MAGIC,
            struct.pack(">Q", len(header)),
            header,
            bytes(result.outputs),
            *(artifact.payload for artifact in result.artifacts),
        )
    )


def _decode_result_header(payload: bytes) -> tuple[Mapping[str, Any], int]:
    """Decode and validate the JSON header of an artifact result envelope."""
    header_length_offset = len(_REMOTE_EXECUTION_RESULT_MAGIC)
    header_offset = header_length_offset + _RESULT_HEADER_LENGTH_BYTES
    if len(payload) < header_offset:
        raise RemoteOutputArtifactError(
            "Remote output result is truncated before its header."
        )
    header_length = struct.unpack(">Q", payload[header_length_offset:header_offset])[0]
    if header_length > _MAX_RESULT_HEADER_BYTES:
        raise RemoteOutputArtifactError(
            "Remote output result header is unreasonably large."
        )
    body_offset = header_offset + header_length
    if body_offset > len(payload):
        raise RemoteOutputArtifactError(
            "Remote output result is truncated inside its header."
        )
    try:
        header = json.loads(payload[header_offset:body_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteOutputArtifactError(
            "Remote output result header is invalid JSON."
        ) from exc
    if not isinstance(header, Mapping):
        raise RemoteOutputArtifactError(
            "Remote output result header must be a mapping."
        )
    if header.get("version") != _REMOTE_EXECUTION_RESULT_VERSION:
        raise RemoteOutputArtifactError(
            f"Unsupported remote output result version {header.get('version')!r}."
        )
    return header, body_offset


def _validated_length(value: Any, field_name: str) -> int:
    """Return a non-negative integer attachment length from a result header."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RemoteOutputArtifactError(
            f"Remote output result {field_name} must be a non-negative integer."
        )
    return value


def _decode_artifact(
    metadata: Any,
    payload: bytes,
    offset: int,
) -> tuple[RemoteOutputArtifact, int]:
    """Decode and integrity-check one artifact attachment."""
    if not isinstance(metadata, Mapping):
        raise RemoteOutputArtifactError(
            "Remote output artifact metadata must be a mapping."
        )
    relative_path = _validated_relative_path(str(metadata.get("relative_path") or ""))
    size_bytes = _validated_length(metadata.get("size_bytes"), "artifact size")
    next_offset = offset + size_bytes
    if next_offset > len(payload):
        raise RemoteOutputArtifactError(
            "Remote output result is truncated inside an artifact."
        )
    artifact_payload = bytes(memoryview(payload)[offset:next_offset])
    expected_sha256 = str(metadata.get("sha256") or "")
    actual_sha256 = hashlib.sha256(artifact_payload).hexdigest()
    if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
        raise RemoteOutputArtifactError(
            "Remote output artifact "
            f"{relative_path.as_posix()!r} failed its SHA256 check."
        )
    return (
        RemoteOutputArtifact(
            relative_path=relative_path.as_posix(),
            payload=artifact_payload,
        ),
        next_offset,
    )


def unpack_remote_execution_result(payload: bytes | bytearray) -> RemoteExecutionResult:
    """Decode a result envelope, accepting legacy node-output bytes unchanged."""
    normalized_payload = bytes(payload)
    if not normalized_payload.startswith(_REMOTE_EXECUTION_RESULT_MAGIC):
        return RemoteExecutionResult(outputs=normalized_payload)
    header, next_offset = _decode_result_header(normalized_payload)
    outputs_length = _validated_length(header.get("outputs_length"), "outputs length")
    outputs_end = next_offset + outputs_length
    if outputs_end > len(normalized_payload):
        raise RemoteOutputArtifactError(
            "Remote output result is truncated inside node outputs."
        )
    outputs = bytes(memoryview(normalized_payload)[next_offset:outputs_end])
    artifact_metadata = header.get("artifacts")
    if not isinstance(artifact_metadata, list):
        raise RemoteOutputArtifactError(
            "Remote output result must include an artifacts list."
        )
    artifacts: list[RemoteOutputArtifact] = []
    next_offset = outputs_end
    for metadata in artifact_metadata:
        artifact, next_offset = _decode_artifact(
            metadata,
            normalized_payload,
            next_offset,
        )
        artifacts.append(artifact)
    if next_offset != len(normalized_payload):
        raise RemoteOutputArtifactError("Remote output result contains trailing bytes.")
    completed_epoch = _validated_length(
        header.get("completed_epoch"),
        "completion epoch",
    )
    return RemoteExecutionResult(
        outputs=outputs,
        artifacts=tuple(artifacts),
        completed_epoch=completed_epoch,
    )


def modal_app_identifier(app_name: str) -> str:
    """Return a filename-safe identifier for one uniquely named Modal app."""
    normalized_name = str(app_name).strip()
    generated_prefix = f"{MODAL_APP_PREFIX}-"
    identifier = (
        normalized_name[len(generated_prefix) :]
        if normalized_name.startswith(generated_prefix)
        else normalized_name
    )
    identifier = _UNSAFE_IDENTIFIER_CHARACTERS.sub("-", identifier).strip("-_")
    if identifier:
        return identifier
    return hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()[:11]


def _epoch_suffix(epoch: int) -> str:
    """Return the trailing nine decimal digits of a Unix epoch."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RemoteOutputArtifactError(
            "Remote output completion epoch must be non-negative."
        )
    return f"{epoch % (10**_EPOCH_DIGITS):0{_EPOCH_DIGITS}d}"


def _sha256_file(path: Path) -> str:
    """Hash one local file without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(_COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _collision_candidate(target_path: Path, collision_index: int) -> Path:
    """Return a non-overwriting alternative while retaining the required prefix."""
    return target_path.with_name(
        f"{target_path.stem}-{collision_index}{target_path.suffix}"
    )


def _publish_artifact_file(target_path: Path, payload: bytes) -> tuple[Path, bool]:
    """Atomically publish bytes without overwriting a different local output."""
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=".comfy-modal-output-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        collision_index = 1
        candidate = target_path
        while True:
            try:
                os.link(temporary_path, candidate)
                return candidate, True
            except FileExistsError:
                same_size = candidate.stat().st_size == len(payload)
                if same_size and _sha256_file(candidate) == payload_sha256:
                    return candidate, False
                collision_index += 1
                candidate = _collision_candidate(target_path, collision_index)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _local_artifact_target(
    output_directory: Path,
    relative_path: Path,
    prefixed_name: str,
) -> Path:
    """Resolve a local target whose parent cannot escape through a symlink."""
    target_parent = output_directory / relative_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = target_parent.resolve()
    if (
        resolved_parent != output_directory
        and output_directory not in resolved_parent.parents
    ):
        raise RemoteOutputArtifactError(
            f"Local output parent escapes the ComfyUI output tree: {target_parent}."
        )
    return resolved_parent / prefixed_name


def materialize_remote_output_artifacts(
    result: RemoteExecutionResult,
    *,
    output_directory: Path,
    app_name: str,
) -> tuple[Path, ...]:
    """Write downloaded artifacts beneath the local ComfyUI output directory."""
    if not result.artifacts:
        return ()
    if result.completed_epoch is None:
        raise RemoteOutputArtifactError(
            "Remote artifacts are missing their completion epoch."
        )
    resolved_output_directory = output_directory.expanduser().resolve()
    app_identifier = modal_app_identifier(app_name)
    prefix = f"remote-{app_identifier}-{_epoch_suffix(result.completed_epoch)}-"
    materialized_paths: list[Path] = []
    for artifact in result.artifacts:
        relative_path = _validated_relative_path(artifact.relative_path)
        target_path = _local_artifact_target(
            resolved_output_directory,
            relative_path,
            f"{prefix}{relative_path.name}",
        )
        materialized_path, created = _publish_artifact_file(
            target_path,
            artifact.payload,
        )
        logger.info(
            "%s remote ComfyUI output %s (%d bytes).",
            "Downloaded" if created else "Reused already-downloaded",
            materialized_path,
            len(artifact.payload),
        )
        materialized_paths.append(materialized_path)
    return tuple(materialized_paths)
