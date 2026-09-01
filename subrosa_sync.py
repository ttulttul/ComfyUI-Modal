"""Authenticated content-addressed asset synchronization for Subrosa."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
import requests

if __package__:
    from .runtime_environment import REMOTE_PYTHON_VERSION, custom_node_runtime_packages
    from .settings import ModalSyncSettings
    from .subrosa_credentials import SubrosaCredentialStore
    from .sync_engine import ModalAssetSyncEngine
    from .sync_protocols import CancellationCheck, SyncCancelledError
else:  # pragma: no cover - direct ComfyUI loading fallback.
    from runtime_environment import REMOTE_PYTHON_VERSION, custom_node_runtime_packages
    from settings import ModalSyncSettings
    from subrosa_credentials import SubrosaCredentialStore
    from sync_engine import ModalAssetSyncEngine
    from sync_protocols import CancellationCheck, SyncCancelledError

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 60.0
_UPLOAD_CHUNK_BYTES = 8 * 1024**2


class SubrosaAssetSyncError(RuntimeError):
    """Report a credential-safe Subrosa asset preparation failure."""


@dataclass(frozen=True)
class SubrosaAssetRecord:
    """Describe one account-owned object and its worker storage destination."""

    remote_path: str
    sha256: str
    size_bytes: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        """Return the control-plane manifest representation."""
        return {
            "remote_path": self.remote_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
        }


class _MemorySyncIndex:
    """Keep deduplication request-local so Subrosa verifies every account object."""

    def __init__(self) -> None:
        """Initialize an empty thread-safe record map."""
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a copied record when the current preparation already stored it."""
        with self._lock:
            record = self._records.get(key)
            return None if record is None else dict(record)

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Remember one record for the rest of the current preparation."""
        with self._lock:
            self._records[key] = dict(value)


@dataclass
class SubrosaAssetApi:
    """Call the extension-authenticated Subrosa asset endpoints."""

    relay_url: str
    token: str
    request: Callable[..., requests.Response] = requests.request

    @property
    def base_url(self) -> str:
        """Return the relay's HTTPS origin from its configured WebSocket URL."""
        parsed = urlparse(self.relay_url)
        if parsed.scheme not in {"ws", "wss"}:
            raise SubrosaAssetSyncError("Subrosa relay URL is invalid for asset sync.")
        scheme = "https" if parsed.scheme == "wss" else "http"
        if not parsed.netloc:
            raise SubrosaAssetSyncError("Subrosa relay URL is invalid for asset sync.")
        return f"{scheme}://{parsed.netloc}"

    def prepare_object(self, sha256: str, size_bytes: int) -> dict[str, Any]:
        """Return an upload capability unless the verified object already exists."""
        return self._json_request(
            "POST",
            "/api/v1/extension/assets/objects/prepare",
            payload={"sha256": sha256, "size_bytes": size_bytes},
            expected_statuses={200},
        )

    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        sha256: str,
        cancellation_check: CancellationCheck | None = None,
    ) -> bool:
        """Upload one file through a short-lived exact-object capability."""
        size_bytes = local_path.stat().st_size
        plan = self.prepare_object(sha256, size_bytes)
        if bool(plan.get("exists")):
            return False
        multipart = plan.get("multipart")
        if isinstance(multipart, Mapping):
            self._upload_multipart_file(
                local_path,
                remote_path,
                sha256=sha256,
                size_bytes=size_bytes,
                plan=multipart,
                cancellation_check=cancellation_check,
            )
            return True
        upload_url = str(plan.get("upload_url") or "")
        upload_headers = _string_mapping(plan.get("upload_headers"))
        if not upload_url:
            raise SubrosaAssetSyncError("Subrosa did not provide an asset upload URL.")
        upload_headers["Content-Length"] = str(size_bytes)
        with local_path.open("rb") as source:
            body = _iter_file_chunks(source.read, cancellation_check)
            self._upload(upload_url, body, upload_headers, remote_path)
        return True

    def upload_bytes(self, payload: bytes, remote_path: str, *, sha256: str) -> bool:
        """Upload one in-memory artifact through a short-lived capability."""
        plan = self.prepare_object(sha256, len(payload))
        if bool(plan.get("exists")):
            return False
        upload_url = str(plan.get("upload_url") or "")
        upload_headers = _string_mapping(plan.get("upload_headers"))
        if not upload_url:
            raise SubrosaAssetSyncError("Subrosa did not provide an asset upload URL.")
        upload_headers["Content-Length"] = str(len(payload))
        self._upload(upload_url, payload, upload_headers, remote_path)
        return True

    def create_manifest(
        self,
        *,
        environment_fingerprint: str,
        requirements: tuple[str, ...],
        assets: tuple[SubrosaAssetRecord, ...],
    ) -> str:
        """Create one immutable account manifest after every object is verified."""
        response = self._json_request(
            "POST",
            "/api/v1/extension/assets/manifests",
            payload={
                "environment_fingerprint": environment_fingerprint,
                "requirements": list(requirements),
                "assets": [asset.to_dict() for asset in assets],
            },
            expected_statuses={201},
        )
        manifest_id = str(response.get("manifest_id") or "").strip()
        if not manifest_id:
            raise SubrosaAssetSyncError("Subrosa returned an invalid asset manifest.")
        return manifest_id

    def _upload(
        self,
        upload_url: str,
        body: bytes | Iterator[bytes],
        headers: Mapping[str, str],
        remote_path: str,
    ) -> requests.Response:
        """Perform one credential-free R2 PUT without logging its signed URL."""
        try:
            response = self.request(
                "PUT",
                upload_url,
                headers=dict(headers),
                data=body,
                timeout=None,
            )
        except requests.RequestException as exc:
            raise SubrosaAssetSyncError(
                f"Subrosa asset upload failed for {remote_path}."
            ) from exc
        if response.status_code not in {200, 201, 204}:
            raise SubrosaAssetSyncError(
                f"Subrosa asset upload failed for {remote_path} with HTTP "
                f"{response.status_code}."
            )
        return response

    def _upload_multipart_file(
        self,
        local_path: Path,
        remote_path: str,
        *,
        sha256: str,
        size_bytes: int,
        plan: Mapping[str, Any],
        cancellation_check: CancellationCheck | None,
    ) -> None:
        """Upload a file larger than R2's single-PUT ceiling in bounded parts."""
        upload_id = str(plan.get("upload_id") or "").strip()
        part_size = int(plan.get("part_size_bytes") or 0)
        part_count = int(plan.get("part_count") or 0)
        if not upload_id or part_size <= 0 or part_count <= 0:
            raise SubrosaAssetSyncError("Subrosa returned an invalid multipart plan.")
        completed_parts: list[dict[str, Any]] = []
        try:
            with local_path.open("rb") as source:
                for part_number in range(1, part_count + 1):
                    if cancellation_check is not None and cancellation_check():
                        raise SyncCancelledError(
                            "Subrosa asset upload was cancelled."
                        )
                    payload = source.read(part_size)
                    if not payload:
                        raise SubrosaAssetSyncError(
                            "Subrosa multipart plan exceeded the local asset size."
                        )
                    capability = self._json_request(
                        "POST",
                        "/api/v1/extension/assets/objects/multipart/part",
                        payload={
                            "sha256": sha256,
                            "size_bytes": size_bytes,
                            "upload_id": upload_id,
                            "part_number": part_number,
                        },
                        expected_statuses={200},
                    )
                    upload_url = str(capability.get("upload_url") or "")
                    headers = _string_mapping(capability.get("upload_headers"))
                    headers["Content-Length"] = str(len(payload))
                    response = self._upload(
                        upload_url, payload, headers, remote_path
                    )
                    etag = str(response.headers.get("ETag") or "").strip()
                    if not etag:
                        raise SubrosaAssetSyncError(
                            "Subrosa multipart upload returned no ETag."
                        )
                    completed_parts.append(
                        {"part_number": part_number, "etag": etag}
                    )
            self._json_request(
                "POST",
                "/api/v1/extension/assets/objects/multipart/complete",
                payload={
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "upload_id": upload_id,
                    "parts": completed_parts,
                },
                expected_statuses={200},
            )
        except (OSError, SubrosaAssetSyncError, SyncCancelledError):
            self._abort_multipart(sha256, size_bytes, upload_id)
            raise

    def _abort_multipart(
        self,
        sha256: str,
        size_bytes: int,
        upload_id: str,
    ) -> None:
        """Best-effort cleanup of an incomplete R2 multipart upload."""
        try:
            self._json_request(
                "POST",
                "/api/v1/extension/assets/objects/multipart/abort",
                payload={
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "upload_id": upload_id,
                },
                expected_statuses={200},
            )
        except SubrosaAssetSyncError as exc:
            logger.warning("Subrosa multipart upload cleanup failed: %s", exc)

    def _json_request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any],
        expected_statuses: set[int],
    ) -> dict[str, Any]:
        """Send one token-authenticated JSON request and sanitize failures."""
        try:
            response = self.request(
                method,
                self.base_url + path,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json=dict(payload),
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise SubrosaAssetSyncError(
                "Subrosa asset preparation could not reach the service."
            ) from exc
        try:
            body = response.json()
        except requests.JSONDecodeError as exc:
            raise SubrosaAssetSyncError(
                f"Subrosa asset service returned invalid JSON with HTTP {response.status_code}."
            ) from exc
        if response.status_code not in expected_statuses:
            message = (
                str(body.get("error") or "asset request rejected")
                if isinstance(body, dict)
                else "asset request rejected"
            )
            raise SubrosaAssetSyncError(
                f"Subrosa asset preparation failed with HTTP {response.status_code}: {message}"
            )
        if not isinstance(body, dict):
            raise SubrosaAssetSyncError("Subrosa asset service returned an invalid object.")
        return dict(body)


class SubrosaAssetVolumeBackend:
    """Adapt Subrosa object uploads to the shared sync-engine storage protocol."""

    def __init__(self, api: SubrosaAssetApi) -> None:
        """Initialize an empty path-to-object manifest."""
        self.api = api
        self._records: dict[str, SubrosaAssetRecord] = {}
        self._lock = threading.Lock()

    @property
    def records(self) -> tuple[SubrosaAssetRecord, ...]:
        """Return every unique materialization target in deterministic order."""
        with self._lock:
            return tuple(self._records[path] for path in sorted(self._records))

    def exists(self, remote_path: str) -> bool:
        """Force the engine through upload planning for this account and request."""
        del remote_path
        return False

    def put_file(self, local_path: Path, remote_path: str) -> None:
        """Upload and record one local file."""
        self.put_file_cancellable(
            local_path,
            remote_path,
            cancellation_check=lambda: False,
        )

    def put_file_cancellable(
        self,
        local_path: Path,
        remote_path: str,
        *,
        cancellation_check: CancellationCheck,
    ) -> None:
        """Upload one local file while observing queue cancellation."""
        sha256 = _sha256_file(local_path, cancellation_check)
        self.api.upload_file(
            local_path,
            remote_path,
            sha256=sha256,
            cancellation_check=cancellation_check,
        )
        self._record(
            SubrosaAssetRecord(
                remote_path=remote_path,
                sha256=sha256,
                size_bytes=local_path.stat().st_size,
                kind=_asset_kind(remote_path),
            )
        )

    def put_bytes(self, payload: bytes, remote_path: str) -> None:
        """Upload and record one generated artifact."""
        sha256 = hashlib.sha256(payload).hexdigest()
        self.api.upload_bytes(payload, remote_path, sha256=sha256)
        self._record(
            SubrosaAssetRecord(
                remote_path=remote_path,
                sha256=sha256,
                size_bytes=len(payload),
                kind=_asset_kind(remote_path),
            )
        )

    def _record(self, record: SubrosaAssetRecord) -> None:
        """Record one remote path and reject contradictory duplicate mappings."""
        with self._lock:
            existing = self._records.get(record.remote_path)
            if existing is not None and existing != record:
                raise SubrosaAssetSyncError(
                    f"Subrosa asset path was assigned conflicting content: {record.remote_path}"
                )
            self._records[record.remote_path] = record


class SubrosaAssetSyncEngine(ModalAssetSyncEngine):
    """Reuse Modal-Sync discovery while publishing a Subrosa asset manifest."""

    volume: SubrosaAssetVolumeBackend

    def __init__(
        self,
        *,
        volume: SubrosaAssetVolumeBackend,
        settings: ModalSyncSettings,
        cancellation_check: CancellationCheck | None,
    ) -> None:
        """Bind request-local indexing and deterministic dependency identity."""
        super().__init__(
            volume=volume,
            settings=settings,
            sync_index=_MemorySyncIndex(),
            cancellation_check=cancellation_check,
        )
        self.requirements = _validated_remote_requirements(
            custom_node_runtime_packages(settings.custom_nodes_dir)
        )
        self.environment_fingerprint = _environment_fingerprint(self.requirements)
        self.asset_manifest_id: str | None = None

    def preflight_r2_access(self, *, status_callback: Any = None) -> None:
        """Skip unrelated BYO-R2 preflight; each managed PUT is independently signed."""
        del status_callback

    def finalize_manifest(self) -> str:
        """Publish the complete immutable environment manifest exactly once."""
        if self.asset_manifest_id is None:
            self.asset_manifest_id = self.volume.api.create_manifest(
                environment_fingerprint=self.environment_fingerprint,
                requirements=self.requirements,
                assets=self.volume.records,
            )
            logger.info(
                "Created Subrosa asset manifest id=%s assets=%d requirements=%d.",
                self.asset_manifest_id,
                len(self.volume.records),
                len(self.requirements),
            )
        return self.asset_manifest_id

    def _destination_label(self) -> str:
        """Name Subrosa truthfully in queue-time progress text."""
        return "Subrosa"


def subrosa_asset_sync_engine(
    settings: ModalSyncSettings,
    cancellation_check: CancellationCheck | None,
    *,
    relay_url: str,
    credential_id: str,
    credential_store: SubrosaCredentialStore | None = None,
) -> SubrosaAssetSyncEngine:
    """Build one authenticated request-local Subrosa synchronization engine."""
    store = credential_store or SubrosaCredentialStore()
    token = store.require(credential_id)
    api = SubrosaAssetApi(relay_url=relay_url, token=token)
    return SubrosaAssetSyncEngine(
        volume=SubrosaAssetVolumeBackend(api),
        settings=settings,
        cancellation_check=cancellation_check,
    )


def _iter_file_chunks(
    read: Callable[[int], bytes],
    cancellation_check: CancellationCheck | None,
) -> Iterator[bytes]:
    """Yield bounded file chunks and stop promptly when the queue is cancelled."""
    while True:
        if cancellation_check is not None and cancellation_check():
            if __package__:
                from .sync_protocols import SyncCancelledError
            else:  # pragma: no cover
                from sync_protocols import SyncCancelledError

            raise SyncCancelledError("Subrosa asset upload was cancelled.")
        chunk = read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


def _sha256_file(path: Path, cancellation_check: CancellationCheck) -> str:
    """Hash one file without buffering it and observe cancellation between chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in _iter_file_chunks(source.read, cancellation_check):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_fingerprint(requirements: tuple[str, ...]) -> str:
    """Hash the Linux Python ABI and ordered custom-node requirements."""
    payload = json.dumps(
        {"python": REMOTE_PYTHON_VERSION, "requirements": list(requirements)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_remote_requirements(requirements: tuple[str, ...]) -> tuple[str, ...]:
    """Reject local or credential-bearing dependency references before upload."""
    validated: list[str] = []
    for raw_requirement in requirements:
        if "${" in raw_requirement or "$(" in raw_requirement:
            raise SubrosaAssetSyncError(
                "A custom-node requirement contains an environment substitution. "
                "Use a public package specifier before running on Subrosa."
            )
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as exc:
            raise SubrosaAssetSyncError(
                f"Custom-node requirement is not remotely installable: {raw_requirement!r}."
            ) from exc
        if requirement.url is not None:
            parsed = urlparse(requirement.url)
            if parsed.scheme not in {"https", "git+https"} or not parsed.netloc:
                raise SubrosaAssetSyncError(
                    "Custom-node direct requirements must use a public HTTPS URL."
                )
            if parsed.username or parsed.password or parsed.query:
                raise SubrosaAssetSyncError(
                    "Custom-node requirement URLs must not contain credentials or query tokens."
                )
        validated.append(raw_requirement)
    return tuple(validated)


def _asset_kind(remote_path: str) -> str:
    """Classify one existing sync-engine storage path for diagnostics."""
    if remote_path.startswith("/custom_nodes/manifests/"):
        return "custom_nodes_manifest"
    if remote_path.startswith("/custom_nodes/entries/"):
        return "custom_node_archive"
    if remote_path.startswith("/custom_nodes/assets/"):
        return "custom_node_asset"
    return "model"


def _string_mapping(value: Any) -> dict[str, str]:
    """Normalize a JSON object containing required upload headers."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


__all__ = [
    "SubrosaAssetApi",
    "SubrosaAssetRecord",
    "SubrosaAssetSyncEngine",
    "SubrosaAssetSyncError",
    "SubrosaAssetVolumeBackend",
    "subrosa_asset_sync_engine",
]
