"""Tests for worker-side signed Cloudflare R2 transfers."""

from __future__ import annotations

import hashlib
import io
import types
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import pytest


class FakeResponse:
    """Expose the requests response surface used by the materializer."""

    def __init__(
        self,
        payload: bytes = b"",
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store deterministic response bytes, status, and headers."""
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self) -> Self:
        """Enter a requests-compatible response context."""
        return self

    def __exit__(self, *args: object) -> None:
        """Leave a requests-compatible response context."""
        del args

    def iter_content(self, chunk_size: int) -> Iterable[bytes]:
        """Yield payload slices using the requested maximum chunk size."""
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset : offset + chunk_size]

    def raise_for_status(self) -> None:
        """Raise only for test response error statuses."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Record signed GET and PUT operations without network access."""

    def __init__(self, downloads: list[FakeResponse] | None = None) -> None:
        """Initialize queued downloads and uploaded payload records."""
        self.downloads = list(downloads or [])
        self.get_calls: list[tuple[str, dict[str, str]]] = []
        self.put_calls: list[tuple[str, bytes, dict[str, str]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        """Return the next queued signed download response."""
        self.get_calls.append((url, dict(kwargs.get("headers") or {})))
        return self.downloads.pop(0)

    def put(self, url: str, **kwargs: Any) -> FakeResponse:
        """Consume one file-like request body and return an ETag."""
        data = kwargs["data"]
        payload = data.read()
        headers = dict(kwargs.get("headers") or {})
        self.put_calls.append((url, payload, headers))
        return FakeResponse(headers={"ETag": f'"etag-{len(self.put_calls)}"'})


def _download_request(storage_root: Path, payload: bytes) -> dict[str, Any]:
    """Return one valid signed download request mapping."""
    return {
        "operation": "download",
        "storage_root": str(storage_root),
        "remote_path": "assets/model.safetensors",
        "download": {
            "url": "https://account.r2.cloudflarestorage.com/object?signature=secret",
            "allowed_host": "account.r2.cloudflarestorage.com",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
    }


def test_download_verifies_digest_and_publishes_atomically(
    r2_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """A matching signed download should appear only at its final verified path."""
    payload = b"verified-model-bytes"
    session = FakeSession([FakeResponse(payload)])
    request = _download_request(tmp_path / "storage", payload)

    result = r2_materializer_module.materialize_download(request, session=session)

    target = tmp_path / "storage" / "assets" / "model.safetensors"
    assert target.read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert list(target.parent.glob("*.r2.part")) == []


def test_corrupt_download_is_deleted_before_failure(
    r2_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """Digest mismatches must not leave cache bytes reusable on the worker."""
    expected = b"expected-model"
    request = _download_request(tmp_path / "storage", expected)
    request["download"]["size_bytes"] = len(b"corrupt-model!")
    session = FakeSession([FakeResponse(b"corrupt-model!")])

    with pytest.raises(ValueError, match="digest mismatch"):
        r2_materializer_module.materialize_download(request, session=session)

    target = tmp_path / "storage" / "assets" / "model.safetensors"
    assert target.exists() is False
    assert target.with_name(f".{target.name}.r2.part").exists() is False


def test_single_upload_uses_exact_content_length(
    r2_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """Small worker files should stream through one signed PUT."""
    storage_root = tmp_path / "storage"
    source = storage_root / "assets" / "model.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"small-model")
    session = FakeSession()
    request = {
        "operation": "upload",
        "storage_root": str(storage_root),
        "remote_path": "assets/model.bin",
        "upload": {
            "allowed_host": "account.r2.cloudflarestorage.com",
            "mode": "single",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size,
            "urls": ["https://account.r2.cloudflarestorage.com/object?secret=1"],
        },
    }

    result = r2_materializer_module.materialize_upload(request, session=session)

    assert result == {"parts": []}
    assert session.put_calls[0][1] == b"small-model"
    assert session.put_calls[0][2]["Content-Length"] == str(source.stat().st_size)


def test_multipart_upload_reads_only_each_planned_segment(
    r2_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """Multipart PUT bodies should contain non-overlapping bounded file slices."""
    storage_root = tmp_path / "storage"
    source = storage_root / "assets" / "large.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"abcdefghij")
    session = FakeSession()
    urls = [
        f"https://account.r2.cloudflarestorage.com/part-{part_number}?secret=1"
        for part_number in range(1, 4)
    ]
    request = {
        "operation": "upload",
        "storage_root": str(storage_root),
        "remote_path": "assets/large.bin",
        "upload": {
            "allowed_host": "account.r2.cloudflarestorage.com",
            "mode": "multipart",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size,
            "part_size_bytes": 4,
            "urls": urls,
        },
    }

    result = r2_materializer_module.materialize_upload(request, session=session)

    assert [call[1] for call in session.put_calls] == [b"abcd", b"efgh", b"ij"]
    assert result == {
        "parts": [
            {"part_number": 1, "etag": '"etag-1"'},
            {"part_number": 2, "etag": '"etag-2"'},
            {"part_number": 3, "etag": '"etag-3"'},
        ]
    }


def test_upload_rejects_worker_bytes_that_do_not_match_object_identity(
    r2_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """Write-back must not poison a content-addressed key with altered worker bytes."""
    storage_root = tmp_path / "storage"
    source = storage_root / "assets" / "model.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"altered-model")
    request = {
        "operation": "upload",
        "storage_root": str(storage_root),
        "remote_path": "assets/model.bin",
        "upload": {
            "allowed_host": "account.r2.cloudflarestorage.com",
            "mode": "single",
            "sha256": "f" * 64,
            "size_bytes": source.stat().st_size,
            "urls": ["https://account.r2.cloudflarestorage.com/object?secret=1"],
        },
    }

    with pytest.raises(ValueError, match="source digest mismatch"):
        r2_materializer_module.materialize_upload(request, session=FakeSession())


def test_main_never_prints_signed_url_on_transfer_failure(
    r2_materializer_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Subprocess diagnostics must not leak credential-bearing presigned URLs."""
    signed_url = "https://account.r2.cloudflarestorage.com/object?signature=secret"
    monkeypatch.setattr(
        r2_materializer_module.sys,
        "stdin",
        types.SimpleNamespace(buffer=io.BytesIO(b"{}")),
    )

    def fail_request(request: Any) -> dict[str, object]:
        """Raise the kind of URL-bearing message requests can produce."""
        del request
        raise RuntimeError(f"transfer failed for {signed_url}")

    monkeypatch.setattr(r2_materializer_module, "process_request", fail_request)

    assert r2_materializer_module.main() == 1
    captured = capsys.readouterr()
    assert "secret" not in captured.err
    assert "category=transfer" in captured.err


def test_main_reports_http_status_without_exposing_presigned_request(
    r2_materializer_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP failures should retain status and category but discard request details."""
    signed_url = "https://account.r2.cloudflarestorage.com/object?signature=secret"
    monkeypatch.setattr(
        r2_materializer_module.sys,
        "stdin",
        types.SimpleNamespace(buffer=io.BytesIO(b"{}")),
    )
    response = r2_materializer_module.requests.Response()
    response.status_code = 403
    response.url = signed_url

    def fail_request(request: Any) -> dict[str, object]:
        """Wrap the requests error in the same transfer failure used after retries."""
        del request
        try:
            raise r2_materializer_module.requests.HTTPError(
                f"403 Client Error for url: {signed_url}",
                response=response,
            )
        except r2_materializer_module.requests.HTTPError as exc:
            raise RuntimeError(f"R2 upload failed for {signed_url}") from exc

    monkeypatch.setattr(r2_materializer_module, "process_request", fail_request)

    assert r2_materializer_module.main() == 1
    captured = capsys.readouterr()
    assert "category=http_client status=403" in captured.err
    assert "secret" not in captured.err
    assert "cloudflarestorage.com" not in captured.err
    assert signed_url not in captured.err


def test_main_reports_safe_r2_error_code_without_exposing_response_body(
    r2_materializer_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R2's bounded symbolic code should survive while its message is discarded."""
    signed_url = "https://account.r2.cloudflarestorage.com/object?signature=secret"
    secret_message = "credential scope contains private-value"
    monkeypatch.setattr(
        r2_materializer_module.sys,
        "stdin",
        types.SimpleNamespace(buffer=io.BytesIO(b"{}")),
    )
    response = r2_materializer_module.requests.Response()
    response.status_code = 403
    response.url = signed_url
    response._content = (
        f"<Error><Code>AccessDenied</Code><Message>{secret_message}</Message></Error>"
    ).encode()
    response._content_consumed = True

    def fail_request(request: Any) -> dict[str, object]:
        """Raise the sanitized materializer HTTP error used by real transfers."""
        del request
        error = r2_materializer_module._http_error(response, "R2 upload")
        raise RuntimeError("R2 upload failed after retries") from error

    monkeypatch.setattr(r2_materializer_module, "process_request", fail_request)

    assert r2_materializer_module.main() == 1
    captured = capsys.readouterr()
    assert "category=http_client status=403 r2_code=AccessDenied" in captured.err
    assert "secret" not in captured.err
    assert secret_message not in captured.err
    assert "cloudflarestorage.com" not in captured.err
