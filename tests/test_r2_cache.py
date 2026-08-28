"""Tests for controller-side Cloudflare R2 cache planning."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest


class FakeS3Client:
    """Emulate the small S3 surface used by the R2 controller."""

    def __init__(self, module: Any) -> None:
        """Initialize an empty object store and operation log."""
        self.module = module
        self.objects: dict[str, int] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.pending_sizes: dict[str, int] = {}
        self.aborted_upload_ids: list[str] = []
        self.fail_upload_part_signing = False
        self.fail_bucket_validation = False

    def head_object(self, **kwargs: Any) -> dict[str, int]:
        """Return an exact stored size or an S3-compatible 404."""
        self.calls.append(("head_object", kwargs))
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise self.module.ClientError(
                {
                    "Error": {"Code": "NoSuchKey", "Message": "missing"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadObject",
            )
        return {"ContentLength": self.objects[key]}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, list[Any]]:
        """Accept a bounded bucket-access validation request."""
        self.calls.append(("list_objects_v2", kwargs))
        if self.fail_bucket_validation:
            raise ValueError("provider diagnostic containing secret-key")
        return {"Contents": []}

    def generate_presigned_url(self, operation: str, **kwargs: Any) -> str:
        """Return a deterministic URL on the configured account host."""
        self.calls.append((operation, kwargs))
        if operation == "upload_part" and self.fail_upload_part_signing:
            raise ValueError("signing failed")
        params = kwargs["Params"]
        return (
            "https://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.r2.cloudflarestorage.com/"
            f"bucket/{params['Key']}?operation={operation}"
        )

    def create_multipart_upload(self, **kwargs: Any) -> dict[str, str]:
        """Record declared metadata and return one upload ID."""
        self.calls.append(("create_multipart_upload", kwargs))
        metadata = kwargs["Metadata"]
        self.pending_sizes[str(kwargs["Key"])] = int(metadata["size-bytes"])
        return {"UploadId": "upload-1"}

    def complete_multipart_upload(self, **kwargs: Any) -> None:
        """Publish a previously created fake multipart object."""
        self.calls.append(("complete_multipart_upload", kwargs))
        key = str(kwargs["Key"])
        self.objects[key] = self.pending_sizes[key]

    def abort_multipart_upload(self, **kwargs: Any) -> None:
        """Record a best-effort multipart abort."""
        self.calls.append(("abort_multipart_upload", kwargs))
        self.aborted_upload_ids.append(str(kwargs["UploadId"]))

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, Any],
    ) -> None:
        """Publish one local file into the fake object map."""
        self.calls.append(
            (
                "upload_file",
                {
                    "filename": filename,
                    "bucket": bucket,
                    "key": key,
                    "ExtraArgs": ExtraArgs,
                },
            )
        )
        self.objects[key] = Path(filename).stat().st_size


def _configuration(r2_cache_module: Any, **overrides: Any) -> Any:
    """Return one valid small-threshold test configuration."""
    values = {
        "account_id": "a" * 32,
        "bucket": "bucket",
        "access_key_id": "access-key",
        "secret_access_key": "secret-key",
        "endpoint_url": (
            "https://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.r2.cloudflarestorage.com"
        ),
        "single_upload_max_bytes": 8,
        "multipart_part_bytes": 5 * 1024**2,
    }
    values.update(overrides)
    return r2_cache_module.R2CacheConfiguration(**values)


def test_environment_configuration_is_opt_in_and_hides_secrets(
    r2_cache_module: Any,
) -> None:
    """Disabled configuration should be inert and enabled secrets must not appear in repr."""
    assert r2_cache_module.R2CacheConfiguration.from_environment({}) is None
    environment = {
        "COMFY_MODAL_R2_ENABLED": "true",
        "COMFY_MODAL_R2_ACCOUNT_ID": "a" * 32,
        "COMFY_MODAL_R2_BUCKET": "models",
        "COMFY_MODAL_R2_ACCESS_KEY_ID": "controller-access",
        "COMFY_MODAL_R2_SECRET_ACCESS_KEY": "controller-secret",
        "COMFY_MODAL_R2_WRITE_BACK": "sync",
    }

    configuration = r2_cache_module.R2CacheConfiguration.from_environment(environment)

    assert configuration is not None
    assert configuration.endpoint_host.endswith("r2.cloudflarestorage.com")
    assert configuration.write_back_mode == "sync"
    assert "controller-access" not in repr(configuration)
    assert "controller-secret" not in repr(configuration)


def test_validate_bucket_access_uses_a_bounded_object_listing(
    r2_cache_module: Any,
) -> None:
    """Imported credentials should be checked without reading object contents."""
    configuration = _configuration(r2_cache_module)
    fake_s3 = FakeS3Client(r2_cache_module)

    r2_cache_module.R2CacheClient(
        configuration,
        s3_client=fake_s3,
    ).validate_bucket_access()

    assert fake_s3.calls == [
        ("list_objects_v2", {"Bucket": "bucket", "MaxKeys": 1})
    ]


def test_validate_bucket_access_returns_a_credential_safe_error(
    r2_cache_module: Any,
) -> None:
    """Provider diagnostics must not reflect imported secrets to the browser."""
    configuration = _configuration(r2_cache_module)
    fake_s3 = FakeS3Client(r2_cache_module)
    fake_s3.fail_bucket_validation = True
    client = r2_cache_module.R2CacheClient(configuration, s3_client=fake_s3)

    with pytest.raises(r2_cache_module.R2CacheError) as captured:
        client.validate_bucket_access()

    assert "secret-key" not in str(captured.value)
    assert "configured bucket" in str(captured.value)


def test_storage_usage_sums_every_paginated_object(
    r2_cache_module: Any,
) -> None:
    """Bucket usage should include every object across S3 listing pages."""

    class PaginatedS3Client(FakeS3Client):
        """Return two deterministic object-listing pages."""

        def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
            """Return the page selected by its continuation token."""
            self.calls.append(("list_objects_v2", kwargs))
            if kwargs.get("ContinuationToken") == "page-2":
                return {
                    "Contents": [{"Size": 7}],
                    "IsTruncated": False,
                }
            return {
                "Contents": [{"Size": 11}, {"Size": 13}],
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
            }

    fake_s3 = PaginatedS3Client(r2_cache_module)
    usage = r2_cache_module.R2CacheClient(
        _configuration(r2_cache_module),
        s3_client=fake_s3,
    ).storage_usage()

    assert usage == r2_cache_module.R2StorageUsage(size_bytes=31, object_count=3)
    assert fake_s3.calls == [
        ("list_objects_v2", {"Bucket": "bucket"}),
        (
            "list_objects_v2",
            {"Bucket": "bucket", "ContinuationToken": "page-2"},
        ),
    ]


def test_download_request_requires_an_exact_size_match(r2_cache_module: Any) -> None:
    """Cache hits should be signed only when their immutable size matches."""
    configuration = _configuration(r2_cache_module)
    fake_s3 = FakeS3Client(r2_cache_module)
    client = r2_cache_module.R2CacheClient(configuration, s3_client=fake_s3)
    sha256 = "b" * 64
    key = configuration.object_key(sha256)

    assert client.download_request(sha256, 12) is None
    fake_s3.objects[key] = 12
    request = client.download_request(sha256, 12)

    assert request is not None
    assert request.sha256 == sha256
    assert request.size_bytes == 12
    assert request.allowed_host == configuration.endpoint_host
    assert "get_object" in request.url

    fake_s3.objects[key] = 11
    assert client.download_request(sha256, 12) is None
    replacement = client.prepare_upload(sha256, 12)
    assert replacement is not None and replacement.mode == "multipart"


def test_worker_preflight_validates_controller_and_signs_short_read_probe(
    r2_cache_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker probes should use an absent key and a bounded presigned GET lifetime."""
    configuration = _configuration(r2_cache_module, url_ttl_seconds=3600)
    fake_s3 = FakeS3Client(r2_cache_module)
    client = r2_cache_module.R2CacheClient(configuration, s3_client=fake_s3)
    monkeypatch.setattr(r2_cache_module.secrets, "token_hex", lambda _bytes: "probe-id")

    request = client.worker_preflight_request()

    assert request.allowed_host == configuration.endpoint_host
    assert "operation=get_object" in request.url
    assert fake_s3.calls[0] == (
        "list_objects_v2",
        {"Bucket": configuration.bucket, "MaxKeys": 1},
    )
    sign_call = fake_s3.calls[1]
    assert sign_call[0] == "get_object"
    assert sign_call[1]["Params"]["Key"].endswith(
        "/.worker-preflight/probe-id"
    )
    assert sign_call[1]["ExpiresIn"] == 300


def test_single_and_multipart_upload_plans_are_completed(r2_cache_module: Any) -> None:
    """The controller should sign small PUTs and complete large multipart uploads."""
    configuration = _configuration(r2_cache_module)
    fake_s3 = FakeS3Client(r2_cache_module)
    client = r2_cache_module.R2CacheClient(configuration, s3_client=fake_s3)
    single_sha = "c" * 64
    multipart_sha = "d" * 64

    single = client.prepare_upload(single_sha, 8)
    multipart_size = 5 * 1024**2 + 1
    multipart = client.prepare_upload(multipart_sha, multipart_size)

    assert single is not None and single.mode == "single"
    put_call = next(call for call in fake_s3.calls if call[0] == "put_object")
    assert "Metadata" not in put_call[1]["Params"]
    assert multipart is not None and multipart.mode == "multipart"
    assert len(multipart.urls) == 2
    result = r2_cache_module.R2UploadResult(
        parts=(
            r2_cache_module.R2UploadedPart(1, '"etag-1"'),
            r2_cache_module.R2UploadedPart(2, '"etag-2"'),
        )
    )
    client.complete_upload(multipart, result)

    assert fake_s3.objects[multipart.key] == multipart_size
    assert fake_s3.aborted_upload_ids == []
    assert client.prepare_upload(multipart_sha, multipart_size) is None
    assert client.prepare_upload(multipart_sha, multipart_size, force=True) is not None


def test_local_prewarm_is_content_addressed_and_idempotent(
    r2_cache_module: Any,
    tmp_path: Path,
) -> None:
    """A local prewarm should upload once and then reuse the exact object."""
    configuration = _configuration(r2_cache_module)
    fake_s3 = FakeS3Client(r2_cache_module)
    client = r2_cache_module.R2CacheClient(configuration, s3_client=fake_s3)
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"model")
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    assert client.upload_local_file(source, sha256=sha256) is True
    assert client.upload_local_file(source, sha256=sha256) is False


def test_multipart_plan_failure_aborts_created_upload(r2_cache_module: Any) -> None:
    """A signing failure must not leave billable multipart data pending in R2."""
    configuration = _configuration(r2_cache_module)
    fake_s3 = FakeS3Client(r2_cache_module)
    fake_s3.fail_upload_part_signing = True
    client = r2_cache_module.R2CacheClient(configuration, s3_client=fake_s3)

    with pytest.raises(r2_cache_module.R2CacheError, match="signing failed"):
        client.prepare_upload("f" * 64, 5 * 1024**2 + 1)

    assert fake_s3.aborted_upload_ids == ["upload-1"]
