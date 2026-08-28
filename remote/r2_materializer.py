"""Transfer verified content-addressed files between worker storage and R2."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_RETRY_ATTEMPTS = 4
_CONNECT_TIMEOUT_SECONDS = 30.0
_READ_TIMEOUT_SECONDS = 120.0
_RETRIABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_SHA256_HEX_CHARACTERS = frozenset("0123456789abcdef")
_MAX_ERROR_RESPONSE_BYTES = 64 * 1024
_R2_ERROR_CODE_PATTERN = re.compile(
    rb"<(?:[A-Za-z0-9_.-]+:)?Code>\s*([A-Za-z][A-Za-z0-9]{0,63})\s*"
    rb"</(?:[A-Za-z0-9_.-]+:)?Code>"
)


class _R2HttpError(requests.HTTPError):
    """Retain only the safe R2 error code alongside an HTTP failure."""

    def __init__(
        self,
        message: str,
        *,
        response: requests.Response,
        r2_error_code: str | None,
    ) -> None:
        """Initialize an HTTP error without embedding its signed request URL."""
        super().__init__(message, response=response)
        self.r2_error_code = r2_error_code


@dataclass(frozen=True)
class _MaterializerTarget:
    """Hold one validated destination beneath the worker storage root."""

    storage_root: Path
    target_path: Path
    remote_path: str


@dataclass
class _LimitedReader:
    """Expose at most one multipart segment from an open binary file."""

    source: BinaryIO
    remaining: int

    def __len__(self) -> int:
        """Return the byte count requests should publish as Content-Length."""
        return self.remaining

    def read(self, size: int = -1) -> bytes:
        """Read no more than the bytes assigned to this upload part."""
        if self.remaining <= 0:
            return b""
        requested_size = self.remaining if size < 0 else min(size, self.remaining)
        payload = self.source.read(requested_size)
        self.remaining -= len(payload)
        return payload


def _required_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one required JSON object field."""
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"R2 materializer field {name!r} must be an object.")
    return value


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    """Return one required non-empty single-line string field."""
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"R2 materializer field {name!r} must be a non-empty string.")
    return value.strip()


def _required_non_negative_int(payload: Mapping[str, Any], name: str) -> int:
    """Return one required non-negative integer field."""
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"R2 materializer field {name!r} must be a non-negative integer."
        )
    return value


def _required_sha256(payload: Mapping[str, Any], name: str) -> str:
    """Return one normalized SHA-256 hex digest field."""
    value = _required_string(payload, name).casefold()
    if len(value) != 64 or any(
        character not in _SHA256_HEX_CHARACTERS for character in value
    ):
        raise ValueError(f"R2 materializer field {name!r} must be a SHA-256 digest.")
    return value


def _validated_target(request: Mapping[str, Any]) -> _MaterializerTarget:
    """Resolve one safe target path beneath an absolute storage root."""
    storage_root = Path(_required_string(request, "storage_root"))
    if not storage_root.is_absolute() or storage_root == Path("/"):
        raise ValueError("R2 materializer storage root must be absolute and non-root.")
    remote_path = _required_string(request, "remote_path").lstrip("/")
    relative_path = Path(remote_path)
    if not relative_path.parts or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        raise ValueError("R2 materializer remote path is unsafe.")
    resolved_root = storage_root.resolve()
    target_path = (resolved_root / relative_path).resolve()
    if not target_path.is_relative_to(resolved_root):
        raise ValueError("R2 materializer target escapes the storage root.")
    return _MaterializerTarget(
        storage_root=resolved_root,
        target_path=target_path,
        remote_path=relative_path.as_posix(),
    )


def _validated_url(value: object, allowed_host: str) -> str:
    """Require one credential-free HTTPS URL on the controller-approved host."""
    if not isinstance(value, str):
        raise TypeError("R2 materializer URL must be a string.")
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != allowed_host
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise ValueError("R2 materializer URL targets an unexpected origin.")
    return value


def _temporary_path(target_path: Path) -> Path:
    """Return the stable resumable temporary path for one target."""
    return target_path.with_name(f".{target_path.name}.r2.part")


def _response_is_retriable(response: requests.Response) -> bool:
    """Return whether one HTTP response should be retried."""
    return response.status_code in _RETRIABLE_STATUS_CODES


def _safe_r2_error_code(response: requests.Response) -> str | None:
    """Return one bounded syntax-validated R2 XML error code, when present."""
    payload = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            remaining = _MAX_ERROR_RESPONSE_BYTES - len(payload)
            payload.extend(chunk[:remaining])
            if len(payload) >= _MAX_ERROR_RESPONSE_BYTES:
                break
    except (AttributeError, OSError, TypeError, requests.RequestException):
        return None
    match = _R2_ERROR_CODE_PATTERN.search(payload)
    if match is None:
        return None
    return match.group(1).decode("ascii")


def _http_error(response: requests.Response, action: str) -> _R2HttpError:
    """Create one URL-free HTTP exception with an optional safe R2 code."""
    return _R2HttpError(
        f"{action} returned status {response.status_code}.",
        response=response,
        r2_error_code=_safe_r2_error_code(response),
    )


def _stream_response_to_file(
    response: requests.Response,
    temporary_path: Path,
    existing_size: int,
) -> None:
    """Append a ranged response or replace the partial file for a full response."""
    append = (
        existing_size > 0 and response.status_code == requests.codes.partial_content
    )
    mode = "ab" if append else "wb"
    with temporary_path.open(mode) as output_file:
        for chunk in response.iter_content(chunk_size=_BUFFER_BYTES):
            if chunk:
                output_file.write(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())


def _download_once(
    session: requests.Session,
    url: str,
    temporary_path: Path,
) -> None:
    """Perform one resumable download attempt."""
    existing_size = temporary_path.stat().st_size if temporary_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
    ) as response:
        if _response_is_retriable(response):
            raise _http_error(response, "R2 download")
        if response.status_code == requests.codes.requested_range_not_satisfiable:
            return
        if response.status_code >= 400:
            raise _http_error(response, "R2 download")
        _stream_response_to_file(response, temporary_path, existing_size)


def _verified_file_digest(path: Path, expected_size: int) -> str:
    """Return SHA-256 after enforcing the exact expected file size."""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"R2 transfer produced {actual_size} bytes, expected {expected_size}."
        )
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_delay(attempt: int) -> float:
    """Return one bounded exponential retry delay."""
    return min(8.0, 0.5 * (2**attempt))


def materialize_download(
    request: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Download, verify, and atomically publish one R2 object."""
    target = _validated_target(request)
    download = _required_mapping(request, "download")
    sha256 = _required_sha256(download, "sha256")
    size_bytes = _required_non_negative_int(download, "size_bytes")
    allowed_host = _required_string(download, "allowed_host")
    url = _validated_url(download.get("url"), allowed_host)
    target.target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(target.target_path)
    active_session = session or requests.Session()
    try:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                _download_once(active_session, url, temporary_path)
                actual_sha256 = _verified_file_digest(temporary_path, size_bytes)
                if actual_sha256 != sha256:
                    raise ValueError(
                        f"R2 download digest mismatch: expected {sha256}, found {actual_sha256}."
                    )
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, target.target_path)
                return {"size_bytes": size_bytes, "sha256": sha256}
            except (OSError, requests.RequestException) as exc:
                if attempt >= _RETRY_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"R2 download failed after retries: {exc}"
                    ) from exc
                time.sleep(_retry_delay(attempt))
    except ValueError:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if session is None:
            active_session.close()
    raise RuntimeError("R2 download retry loop exited unexpectedly.")


def materialize_preflight(
    request: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Verify effective worker-side R2 authorization without reading an object."""
    preflight = _required_mapping(request, "preflight")
    allowed_host = _required_string(preflight, "allowed_host")
    url = _validated_url(preflight.get("url"), allowed_host)
    active_session = session or requests.Session()
    try:
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                with active_session.get(
                    url,
                    headers={"Range": "bytes=0-0"},
                    stream=True,
                    allow_redirects=False,
                    timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
                ) as response:
                    status_code = response.status_code
                    if 200 <= status_code <= 299 or status_code in {
                        requests.codes.not_found,
                        requests.codes.requested_range_not_satisfiable,
                    }:
                        return {"authorized": True}
                    error = _http_error(response, "R2 worker preflight")
                    if _response_is_retriable(response):
                        raise error
                    raise error
            except (OSError, requests.RequestException) as exc:
                if attempt >= _RETRY_ATTEMPTS - 1 or (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and not _response_is_retriable(exc.response)
                ):
                    raise RuntimeError(
                        "R2 worker preflight failed after retries."
                    ) from exc
                time.sleep(_retry_delay(attempt))
    finally:
        if session is None:
            active_session.close()
    raise RuntimeError("R2 worker preflight retry loop exited unexpectedly.")


def _source_file(
    request: Mapping[str, Any], target: _MaterializerTarget
) -> tuple[Path, int]:
    """Return one verified local source file and declared size."""
    upload = _required_mapping(request, "upload")
    expected_sha256 = _required_sha256(upload, "sha256")
    size_bytes = _required_non_negative_int(upload, "size_bytes")
    if not target.target_path.is_file():
        raise FileNotFoundError(
            f"R2 write-back source is missing: {target.target_path}"
        )
    actual_size = target.target_path.stat().st_size
    if actual_size != size_bytes:
        raise ValueError(
            f"R2 write-back source has {actual_size} bytes, expected {size_bytes}."
        )
    actual_sha256 = _verified_file_digest(target.target_path, size_bytes)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"R2 write-back source digest mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}."
        )
    return target.target_path, size_bytes


def _put_single_upload(
    session: requests.Session,
    url: str,
    source_path: Path,
    size_bytes: int,
) -> None:
    """Upload one complete object with bounded retries."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            with source_path.open("rb") as source_file:
                response = session.put(
                    url,
                    data=source_file,
                    headers={"Content-Length": str(size_bytes)},
                    timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
                )
            if _response_is_retriable(response):
                raise _http_error(response, "R2 upload")
            if response.status_code >= 400:
                raise _http_error(response, "R2 upload")
            return
        except (OSError, requests.RequestException) as exc:
            if attempt >= _RETRY_ATTEMPTS - 1:
                raise RuntimeError(f"R2 upload failed after retries: {exc}") from exc
            time.sleep(_retry_delay(attempt))


def _put_upload_part(
    session: requests.Session,
    url: str,
    source_path: Path,
    offset: int,
    size_bytes: int,
) -> str:
    """Upload one bounded multipart segment and return its ETag."""
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            with source_path.open("rb") as source_file:
                source_file.seek(offset)
                limited_reader = _LimitedReader(source_file, size_bytes)
                response = session.put(
                    url,
                    data=limited_reader,
                    headers={"Content-Length": str(size_bytes)},
                    timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
                )
            if _response_is_retriable(response):
                raise _http_error(response, "R2 upload part")
            if response.status_code >= 400:
                raise _http_error(response, "R2 upload part")
            etag = response.headers.get("ETag", "").strip()
            if not etag:
                raise RuntimeError("R2 multipart upload response omitted its ETag.")
            return etag
        except (OSError, requests.RequestException) as exc:
            if attempt >= _RETRY_ATTEMPTS - 1:
                raise RuntimeError(
                    f"R2 multipart upload failed after retries: {exc}"
                ) from exc
            time.sleep(_retry_delay(attempt))
    raise RuntimeError("R2 multipart upload retry loop exited unexpectedly.")


def materialize_upload(
    request: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Upload one worker file through a controller-issued presigned plan."""
    target = _validated_target(request)
    upload = _required_mapping(request, "upload")
    source_path, size_bytes = _source_file(request, target)
    allowed_host = _required_string(upload, "allowed_host")
    mode = _required_string(upload, "mode").casefold()
    raw_urls = upload.get("urls")
    if not isinstance(raw_urls, list) or not raw_urls:
        raise ValueError("R2 upload plan must contain at least one URL.")
    urls = tuple(_validated_url(url, allowed_host) for url in raw_urls)
    active_session = session or requests.Session()
    try:
        if mode == "single":
            if len(urls) != 1:
                raise ValueError("Single-part R2 upload requires exactly one URL.")
            _put_single_upload(active_session, urls[0], source_path, size_bytes)
            return {"parts": []}
        if mode != "multipart":
            raise ValueError("R2 upload mode must be single or multipart.")
        part_size_bytes = _required_non_negative_int(upload, "part_size_bytes")
        if part_size_bytes <= 0:
            raise ValueError("R2 multipart part size must be positive.")
        expected_parts = (size_bytes + part_size_bytes - 1) // part_size_bytes
        if len(urls) != expected_parts:
            raise ValueError(
                f"R2 multipart plan contains {len(urls)} URLs, expected {expected_parts}."
            )
        parts: list[dict[str, object]] = []
        for index, url in enumerate(urls):
            offset = index * part_size_bytes
            part_size = min(part_size_bytes, size_bytes - offset)
            etag = _put_upload_part(active_session, url, source_path, offset, part_size)
            parts.append({"part_number": index + 1, "etag": etag})
        return {"parts": parts}
    finally:
        if session is None:
            active_session.close()


def process_request(request: Mapping[str, Any]) -> dict[str, object]:
    """Dispatch one validated R2 transfer request."""
    operation = _required_string(request, "operation").casefold()
    if operation == "preflight":
        return materialize_preflight(request)
    if operation == "download":
        return materialize_download(request)
    if operation == "upload":
        return materialize_upload(request)
    raise ValueError("R2 materializer operation must be download or upload.")


def _execute_payload() -> None:
    """Read, execute, and print one protected JSON materializer request."""
    payload = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(payload) > _MAX_REQUEST_BYTES:
        raise ValueError("R2 materializer request exceeds the maximum size.")
    request = json.loads(payload)
    if not isinstance(request, Mapping):
        raise TypeError("R2 materializer request must be a JSON object.")
    result = process_request(request)
    print(json.dumps(result, separators=(",", ":"), sort_keys=True), flush=True)


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    """Return a cycle-safe exception chain for credential-free classification."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _safe_error_details(exc: BaseException) -> tuple[str, int | None, str | None]:
    """Classify one failure without returning messages, URLs, or credentials."""
    chain = _exception_chain(exc)
    for error in chain:
        if isinstance(error, requests.HTTPError):
            response = error.response
            status_code = getattr(response, "status_code", None)
            r2_error_code = (
                error.r2_error_code if isinstance(error, _R2HttpError) else None
            )
            if isinstance(status_code, int) and 100 <= status_code <= 599:
                if 400 <= status_code <= 499:
                    return "http_client", status_code, r2_error_code
                if 500 <= status_code <= 599:
                    return "http_server", status_code, r2_error_code
                return "http", status_code, r2_error_code
    if any(isinstance(error, requests.Timeout) for error in chain):
        return "timeout", None, None
    if any(isinstance(error, requests.exceptions.SSLError) for error in chain):
        return "tls", None, None
    if any(isinstance(error, requests.ConnectionError) for error in chain):
        return "connection", None, None
    if any(isinstance(error, FileNotFoundError) for error in chain):
        return "source_missing", None, None
    if any(isinstance(error, json.JSONDecodeError) for error in chain):
        return "invalid_request", None, None
    if any(isinstance(error, TypeError) for error in chain):
        return "invalid_request", None, None
    if any(isinstance(error, ValueError) for error in chain):
        return "validation", None, None
    if any(isinstance(error, OSError) for error in chain):
        return "io", None, None
    return "transfer", None, None


def _safe_error_summary(exc: BaseException) -> str:
    """Return fixed diagnostic fields that cannot contain signed request data."""
    category, status_code, r2_error_code = _safe_error_details(exc)
    if status_code is None:
        return f"category={category}"
    if r2_error_code is not None:
        return (
            f"category={category} status={status_code} "
            f"r2_code={r2_error_code}"
        )
    return f"category={category} status={status_code}"


def main() -> int:
    """Run one request while preventing signed URLs from entering stderr."""
    try:
        _execute_payload()
    except (
        OSError,
        requests.RequestException,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"R2 materializer failed safely ({_safe_error_summary(exc)}).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through remote subprocesses.
    raise SystemExit(main())


__all__ = [
    "main",
    "materialize_download",
    "materialize_preflight",
    "materialize_upload",
    "process_request",
]
