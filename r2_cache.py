"""Optional Cloudflare R2 content-addressed backing-cache controller."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

R2_ENABLED_ENV = "COMFY_MODAL_R2_ENABLED"
R2_ACCOUNT_ID_ENV = "COMFY_MODAL_R2_ACCOUNT_ID"
R2_BUCKET_ENV = "COMFY_MODAL_R2_BUCKET"
R2_ACCESS_KEY_ID_ENV = "COMFY_MODAL_R2_ACCESS_KEY_ID"
R2_SECRET_ACCESS_KEY_ENV = "COMFY_MODAL_R2_SECRET_ACCESS_KEY"
R2_ENDPOINT_URL_ENV = "COMFY_MODAL_R2_ENDPOINT_URL"
R2_KEY_PREFIX_ENV = "COMFY_MODAL_R2_KEY_PREFIX"
R2_WRITE_BACK_ENV = "COMFY_MODAL_R2_WRITE_BACK"
R2_URL_TTL_SECONDS_ENV = "COMFY_MODAL_R2_URL_TTL_SECONDS"
R2_MULTIPART_PART_MIB_ENV = "COMFY_MODAL_R2_MULTIPART_PART_MIB"
R2_SINGLE_UPLOAD_MAX_MIB_ENV = "COMFY_MODAL_R2_SINGLE_UPLOAD_MAX_MIB"

DEFAULT_R2_KEY_PREFIX = "comfy-modal-cache/v1/blobs/sha256"
DEFAULT_R2_URL_TTL_SECONDS = 6 * 60 * 60
DEFAULT_R2_MULTIPART_PART_MIB = 256
DEFAULT_R2_SINGLE_UPLOAD_MAX_MIB = 100
R2_MAX_URL_TTL_SECONDS = 7 * 24 * 60 * 60
R2_MAX_SINGLE_UPLOAD_BYTES = 5 * 1024**3
R2_MIN_MULTIPART_PART_BYTES = 5 * 1024**2
R2_MAX_MULTIPART_PART_BYTES = 5 * 1024**3
R2_MAX_MULTIPART_PARTS = 10_000
R2_WORKER_PREFLIGHT_TTL_SECONDS = 5 * 60
_HASH_BUFFER_BYTES = 4 * 1024 * 1024

_ACCOUNT_ID_PATTERN = re.compile(r"^[a-fA-F0-9]{32}$")
_BUCKET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_WRITE_BACK_MODES = frozenset({"async", "off", "sync"})
_R2_CLIENT_ERRORS = (
    Boto3Error,
    BotoCoreError,
    ClientError,
    OSError,
    RuntimeError,
    ValueError,
)


class R2CacheError(RuntimeError):
    """Raised when the configured R2 cache cannot complete an operation."""


@dataclass(frozen=True)
class R2StorageUsage:
    """Summarize the current objects stored in one R2 bucket."""

    size_bytes: int
    object_count: int


def _read_bool(value: object, *, name: str) -> bool:
    """Parse one conventional boolean environment value."""
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got {value!r}.")


def _read_positive_int(
    source: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    """Read one positive integer from an environment mapping."""
    raw_value = source.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {raw_value!r}."
        ) from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be positive.")
    return value


@dataclass(frozen=True)
class R2CacheConfiguration:
    """Hold validated controller-only Cloudflare R2 cache configuration."""

    account_id: str
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    endpoint_url: str
    key_prefix: str = DEFAULT_R2_KEY_PREFIX
    write_back_mode: str = "async"
    url_ttl_seconds: int = DEFAULT_R2_URL_TTL_SECONDS
    multipart_part_bytes: int = DEFAULT_R2_MULTIPART_PART_MIB * 1024**2
    single_upload_max_bytes: int = DEFAULT_R2_SINGLE_UPLOAD_MAX_MIB * 1024**2

    def __post_init__(self) -> None:
        """Reject unsafe endpoints, identifiers, and transfer bounds."""
        if not _ACCOUNT_ID_PATTERN.fullmatch(self.account_id):
            raise ValueError(
                "Cloudflare R2 account ID must be 32 hexadecimal characters."
            )
        if not _BUCKET_PATTERN.fullmatch(self.bucket):
            raise ValueError(
                "Cloudflare R2 bucket name contains unsupported characters."
            )
        for name, value in (
            ("access key ID", self.access_key_id),
            ("secret access key", self.secret_access_key),
        ):
            if not value or any(
                character in value for character in ("\x00", "\n", "\r")
            ):
                raise ValueError(
                    f"Cloudflare R2 {name} must be a non-empty single-line value."
                )
        parsed_endpoint = urlparse(self.endpoint_url)
        if (
            parsed_endpoint.scheme != "https"
            or not parsed_endpoint.hostname
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.port not in {None, 443}
            or parsed_endpoint.path not in {"", "/"}
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise ValueError(
                "Cloudflare R2 endpoint must be a credential-free HTTPS origin."
            )
        normalized_prefix = self.key_prefix.strip("/")
        if (
            not normalized_prefix
            or "//" in normalized_prefix
            or any(part in {"", ".", ".."} for part in normalized_prefix.split("/"))
            or any(
                character in normalized_prefix
                for character in ("\x00", "\n", "\r", "\\")
            )
        ):
            raise ValueError("Cloudflare R2 key prefix is unsafe.")
        object.__setattr__(self, "key_prefix", normalized_prefix)
        normalized_write_back = self.write_back_mode.strip().casefold()
        if normalized_write_back not in _WRITE_BACK_MODES:
            raise ValueError(
                "Cloudflare R2 write-back mode must be one of async, sync, or off."
            )
        if normalized_write_back == "sync":
            logger.warning(
                "Cloudflare R2 synchronous write-back is deprecated; using idle background write-back."
            )
            normalized_write_back = "async"
        object.__setattr__(self, "write_back_mode", normalized_write_back)
        if not 1 <= self.url_ttl_seconds <= R2_MAX_URL_TTL_SECONDS:
            raise ValueError(
                "Cloudflare R2 presigned URL lifetime must be 1-604800 seconds."
            )
        if (
            not R2_MIN_MULTIPART_PART_BYTES
            <= self.multipart_part_bytes
            <= R2_MAX_MULTIPART_PART_BYTES
        ):
            raise ValueError(
                "Cloudflare R2 multipart part size must be between 5 MiB and 5 GiB."
            )
        if not 1 <= self.single_upload_max_bytes <= R2_MAX_SINGLE_UPLOAD_BYTES:
            raise ValueError(
                "Cloudflare R2 single-upload threshold must be at most 5 GiB."
            )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> R2CacheConfiguration | None:
        """Return optional R2 configuration from controller environment variables."""
        source = os.environ if environment is None else environment
        if not _read_bool(source.get(R2_ENABLED_ENV, "false"), name=R2_ENABLED_ENV):
            return None
        required_names = (
            R2_ACCOUNT_ID_ENV,
            R2_BUCKET_ENV,
            R2_ACCESS_KEY_ID_ENV,
            R2_SECRET_ACCESS_KEY_ENV,
        )
        missing_names = [
            name for name in required_names if not str(source.get(name) or "").strip()
        ]
        if missing_names:
            raise RuntimeError(
                "Cloudflare R2 backing is enabled but required configuration is missing: "
                + ", ".join(missing_names)
                + "."
            )
        account_id = str(source[R2_ACCOUNT_ID_ENV]).strip()
        endpoint_url = str(source.get(R2_ENDPOINT_URL_ENV) or "").strip()
        if not endpoint_url:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        write_back_mode = str(source.get(R2_WRITE_BACK_ENV, "async")).strip()
        multipart_part_mib = _read_positive_int(
            source,
            R2_MULTIPART_PART_MIB_ENV,
            DEFAULT_R2_MULTIPART_PART_MIB,
        )
        single_upload_max_mib = _read_positive_int(
            source,
            R2_SINGLE_UPLOAD_MAX_MIB_ENV,
            DEFAULT_R2_SINGLE_UPLOAD_MAX_MIB,
        )
        return cls(
            account_id=account_id,
            bucket=str(source[R2_BUCKET_ENV]).strip(),
            access_key_id=str(source[R2_ACCESS_KEY_ID_ENV]).strip(),
            secret_access_key=str(source[R2_SECRET_ACCESS_KEY_ENV]).strip(),
            endpoint_url=endpoint_url.rstrip("/"),
            key_prefix=str(source.get(R2_KEY_PREFIX_ENV) or DEFAULT_R2_KEY_PREFIX),
            write_back_mode=write_back_mode,
            url_ttl_seconds=_read_positive_int(
                source,
                R2_URL_TTL_SECONDS_ENV,
                DEFAULT_R2_URL_TTL_SECONDS,
            ),
            multipart_part_bytes=multipart_part_mib * 1024**2,
            single_upload_max_bytes=single_upload_max_mib * 1024**2,
        )

    @property
    def endpoint_host(self) -> str:
        """Return the only host permitted in generated materialization URLs."""
        parsed = urlparse(self.endpoint_url)
        assert parsed.hostname is not None
        return parsed.hostname

    def object_key(self, sha256: str) -> str:
        """Return the immutable R2 object key for one SHA-256 digest."""
        normalized_sha256 = sha256.strip().casefold()
        if not _SHA256_PATTERN.fullmatch(normalized_sha256):
            raise ValueError("R2 cache object identity must be a SHA-256 hex digest.")
        return f"{self.key_prefix}/{normalized_sha256[:2]}/{normalized_sha256}"


@dataclass(frozen=True)
class R2DownloadRequest:
    """Describe one short-lived verified R2-to-worker download."""

    url: str = field(repr=False)
    allowed_host: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return the credential-sensitive request for protected standard input."""
        return {
            "url": self.url,
            "allowed_host": self.allowed_host,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class R2WorkerPreflightRequest:
    """Describe one short-lived read-only worker authorization probe."""

    url: str = field(repr=False)
    allowed_host: str

    def to_dict(self) -> dict[str, str]:
        """Return the credential-sensitive request for protected standard input."""
        return {"url": self.url, "allowed_host": self.allowed_host}


@dataclass(frozen=True)
class R2UploadPlan:
    """Describe one short-lived single-part or multipart worker upload."""

    key: str
    sha256: str
    size_bytes: int
    allowed_host: str
    mode: str
    urls: tuple[str, ...] = field(repr=False)
    part_size_bytes: int | None = None
    upload_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate a complete upload plan before sending it remotely."""
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("R2 upload plan identity must be a SHA-256 digest.")
        if self.size_bytes < 0:
            raise ValueError("R2 upload plan size must not be negative.")
        if self.mode not in {"single", "multipart"}:
            raise ValueError("R2 upload plan mode must be single or multipart.")
        if not self.urls:
            raise ValueError("R2 upload plan must contain at least one URL.")
        if self.mode == "multipart" and (
            self.part_size_bytes is None
            or self.part_size_bytes <= 0
            or not self.upload_id
        ):
            raise ValueError(
                "Multipart R2 upload plans require part size and upload ID."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the credential-sensitive plan for protected standard input."""
        return {
            "key": self.key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "allowed_host": self.allowed_host,
            "mode": self.mode,
            "urls": list(self.urls),
            "part_size_bytes": self.part_size_bytes,
        }


@dataclass(frozen=True)
class R2UploadedPart:
    """Identify one completed multipart upload part."""

    part_number: int
    etag: str


@dataclass(frozen=True)
class R2UploadResult:
    """Return the remote upload result needed for controller completion."""

    parts: tuple[R2UploadedPart, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> R2UploadResult:
        """Parse and validate one remote materializer result."""
        raw_parts = payload.get("parts", [])
        if not isinstance(raw_parts, list):
            raise TypeError("R2 upload result parts must be a list.")
        parts: list[R2UploadedPart] = []
        for raw_part in raw_parts:
            if not isinstance(raw_part, Mapping):
                raise TypeError("R2 upload result contains a malformed part.")
            part_number = raw_part.get("part_number")
            etag = raw_part.get("etag")
            if (
                isinstance(part_number, bool)
                or not isinstance(part_number, int)
                or part_number <= 0
            ):
                raise ValueError("R2 upload result contains an invalid part number.")
            if not isinstance(etag, str) or not etag.strip():
                raise ValueError("R2 upload result contains an invalid ETag.")
            if any(character in etag for character in ("\x00", "\n", "\r")):
                raise ValueError("R2 upload result ETag must be a single-line value.")
            parts.append(R2UploadedPart(part_number=part_number, etag=etag.strip()))
        return cls(parts=tuple(parts))


@dataclass
class R2CacheClient:
    """Use Cloudflare R2 as a content-addressed backing cache."""

    configuration: R2CacheConfiguration
    s3_client: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Create the S3-compatible client lazily when none was injected."""
        if self.s3_client is not None:
            return
        try:
            import boto3
            from botocore.config import Config
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cloudflare R2 backing requires boto3; install the project dependencies with uv sync."
            ) from exc
        self.s3_client = boto3.client(
            service_name="s3",
            endpoint_url=self.configuration.endpoint_url,
            aws_access_key_id=self.configuration.access_key_id,
            aws_secret_access_key=self.configuration.secret_access_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 4, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> R2CacheClient | None:
        """Return an enabled client or None when R2 backing is disabled."""
        configuration = R2CacheConfiguration.from_environment(environment)
        return cls(configuration) if configuration is not None else None

    @property
    def write_back_mode(self) -> str:
        """Return idle-background async or read-only off behavior."""
        return self.configuration.write_back_mode

    def validate_bucket_access(self) -> None:
        """Verify that the configured S3 credential can list its target bucket."""
        assert self.s3_client is not None
        try:
            self.s3_client.list_objects_v2(
                Bucket=self.configuration.bucket,
                MaxKeys=1,
            )
        except _R2_CLIENT_ERRORS as exc:
            raise R2CacheError(
                "Cloudflare rejected the R2 credentials or they do not grant "
                "object access to the configured bucket."
            ) from exc

    def worker_preflight_request(self) -> R2WorkerPreflightRequest:
        """Validate controller access and sign a read-only absent-object worker probe."""
        self.validate_bucket_access()
        sentinel_key = (
            f"{self.configuration.key_prefix}/.worker-preflight/"
            f"{secrets.token_hex(16)}"
        )
        assert self.s3_client is not None
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.configuration.bucket,
                    "Key": sentinel_key,
                },
                ExpiresIn=min(
                    self.configuration.url_ttl_seconds,
                    R2_WORKER_PREFLIGHT_TTL_SECONDS,
                ),
            )
        except _R2_CLIENT_ERRORS as exc:
            raise R2CacheError(
                "Unable to sign the Cloudflare R2 worker preflight request."
            ) from exc
        self._validate_generated_url(url)
        return R2WorkerPreflightRequest(
            url=url,
            allowed_host=self.configuration.endpoint_host,
        )

    def storage_usage(self) -> R2StorageUsage:
        """Return the exact current object bytes and count for the configured bucket."""
        assert self.s3_client is not None
        size_bytes = 0
        object_count = 0
        continuation_token: str | None = None
        while True:
            request: dict[str, Any] = {"Bucket": self.configuration.bucket}
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token
            try:
                response = self.s3_client.list_objects_v2(**request)
            except _R2_CLIENT_ERRORS as exc:
                raise R2CacheError(
                    "Unable to read storage usage for the configured R2 bucket."
                ) from exc
            if not isinstance(response, Mapping):
                raise R2CacheError("R2 returned an invalid storage-usage response.")
            contents = response.get("Contents", [])
            if not isinstance(contents, list):
                raise R2CacheError("R2 returned an invalid storage-usage object list.")
            for item in contents:
                if not isinstance(item, Mapping):
                    raise R2CacheError(
                        "R2 returned an invalid object in its storage-usage response."
                    )
                object_size = item.get("Size")
                if (
                    isinstance(object_size, bool)
                    or not isinstance(object_size, int)
                    or object_size < 0
                ):
                    raise R2CacheError(
                        "R2 returned an invalid object size in its storage-usage response."
                    )
                size_bytes += object_size
                object_count += 1
            if not response.get("IsTruncated", False):
                return R2StorageUsage(
                    size_bytes=size_bytes,
                    object_count=object_count,
                )
            raw_token = response.get("NextContinuationToken")
            next_token = str(raw_token or "").strip()
            if (
                not next_token
                or next_token == continuation_token
                or any(character in next_token for character in ("\x00", "\n", "\r"))
            ):
                raise R2CacheError(
                    "R2 returned an invalid continuation token for storage usage."
                )
            continuation_token = next_token

    def download_request(
        self,
        sha256: str,
        size_bytes: int,
    ) -> R2DownloadRequest | None:
        """Return a presigned download when the exact object exists in R2."""
        key = self.configuration.object_key(sha256)
        existing_size = self._head_size(key)
        if existing_size is None:
            return None
        if existing_size != size_bytes:
            logger.warning(
                "Ignoring corrupt R2 cache object key=%s size=%d expected_size=%d.",
                key,
                existing_size,
                size_bytes,
            )
            return None
        assert self.s3_client is not None
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.configuration.bucket, "Key": key},
                ExpiresIn=self.configuration.url_ttl_seconds,
            )
        except _R2_CLIENT_ERRORS as exc:
            raise R2CacheError(
                f"Unable to sign R2 download for {key!r}: {exc}"
            ) from exc
        self._validate_generated_url(url)
        return R2DownloadRequest(
            url=url,
            allowed_host=self.configuration.endpoint_host,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def prepare_upload(
        self,
        sha256: str,
        size_bytes: int,
        *,
        force: bool = False,
    ) -> R2UploadPlan | None:
        """Create presigned worker upload URLs unless the exact object already exists."""
        key = self.configuration.object_key(sha256)
        existing_size = self._head_size(key)
        if existing_size is not None:
            if existing_size == size_bytes and not force:
                return None
            if existing_size != size_bytes:
                logger.warning(
                    "Replacing corrupt R2 cache object key=%s size=%d expected_size=%d.",
                    key,
                    existing_size,
                    size_bytes,
                )
        if size_bytes <= self.configuration.single_upload_max_bytes:
            return self._single_upload_plan(key, sha256, size_bytes)
        return self._multipart_upload_plan(key, sha256, size_bytes)

    def complete_upload(self, plan: R2UploadPlan, result: R2UploadResult) -> None:
        """Publish a completed upload and verify its resulting object size."""
        assert self.s3_client is not None
        if plan.mode == "multipart":
            expected_parts = len(plan.urls)
            expected_part_numbers = tuple(range(1, expected_parts + 1))
            actual_part_numbers = tuple(part.part_number for part in result.parts)
            if actual_part_numbers != expected_part_numbers:
                self.abort_upload(plan)
                raise R2CacheError(
                    "R2 multipart upload returned invalid part numbers: "
                    f"{actual_part_numbers!r}; expected {expected_part_numbers!r}."
                )
            try:
                self.s3_client.complete_multipart_upload(
                    Bucket=self.configuration.bucket,
                    Key=plan.key,
                    UploadId=plan.upload_id,
                    MultipartUpload={
                        "Parts": [
                            {"PartNumber": part.part_number, "ETag": part.etag}
                            for part in result.parts
                        ]
                    },
                )
            except _R2_CLIENT_ERRORS as exc:
                self.abort_upload(plan)
                raise R2CacheError(
                    f"Unable to complete R2 multipart upload for {plan.key!r}: {exc}"
                ) from exc
        elif result.parts:
            raise R2CacheError(
                "Single-part R2 upload returned unexpected part metadata."
            )
        existing_size = self._head_size(plan.key)
        if existing_size != plan.size_bytes:
            raise R2CacheError(
                f"Completed R2 cache object {plan.key!r} has size {existing_size}, "
                f"expected {plan.size_bytes}."
            )

    def abort_upload(self, plan: R2UploadPlan) -> None:
        """Best-effort abort one incomplete multipart upload."""
        if plan.mode != "multipart" or plan.upload_id is None:
            return
        self._abort_multipart_key(plan.key, plan.upload_id)

    def _abort_multipart_key(self, key: str, upload_id: str) -> None:
        """Best-effort abort a multipart upload before or after plan creation."""
        assert self.s3_client is not None
        try:
            self.s3_client.abort_multipart_upload(
                Bucket=self.configuration.bucket,
                Key=key,
                UploadId=upload_id,
            )
        except _R2_CLIENT_ERRORS as exc:
            logger.warning("Unable to abort R2 multipart upload key=%s: %s", key, exc)

    def upload_local_file(self, local_path: Path, *, sha256: str) -> bool:
        """Prewarm one local file into R2 and report whether it was uploaded."""
        resolved_path = local_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"R2 prewarm source not found: {resolved_path}")
        size_bytes = resolved_path.stat().st_size
        key = self.configuration.object_key(sha256)
        actual_sha256 = _sha256_file(resolved_path)
        if actual_sha256 != sha256:
            raise ValueError(
                f"R2 prewarm digest mismatch for {resolved_path}: "
                f"expected {sha256}, found {actual_sha256}."
            )
        existing_size = self._head_size(key)
        if existing_size is not None:
            if existing_size == size_bytes:
                return False
            logger.warning(
                "Replacing corrupt R2 prewarm object key=%s size=%d expected_size=%d.",
                key,
                existing_size,
                size_bytes,
            )
        assert self.s3_client is not None
        try:
            self.s3_client.upload_file(
                str(resolved_path),
                self.configuration.bucket,
                key,
                ExtraArgs={
                    "Metadata": {"sha256": sha256, "size-bytes": str(size_bytes)}
                },
            )
        except _R2_CLIENT_ERRORS as exc:
            raise R2CacheError(
                f"Unable to upload {resolved_path} to R2: {exc}"
            ) from exc
        if self._head_size(key) != size_bytes:
            raise R2CacheError(f"R2 prewarm verification failed for {resolved_path}.")
        return True

    def _single_upload_plan(
        self,
        key: str,
        sha256: str,
        size_bytes: int,
    ) -> R2UploadPlan:
        """Return one presigned PutObject plan."""
        assert self.s3_client is not None
        try:
            url = self.s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.configuration.bucket, "Key": key},
                ExpiresIn=self.configuration.url_ttl_seconds,
            )
        except _R2_CLIENT_ERRORS as exc:
            raise R2CacheError(f"Unable to sign R2 upload for {key!r}: {exc}") from exc
        self._validate_generated_url(url)
        return R2UploadPlan(
            key=key,
            sha256=sha256,
            size_bytes=size_bytes,
            allowed_host=self.configuration.endpoint_host,
            mode="single",
            urls=(url,),
        )

    def _multipart_upload_plan(
        self,
        key: str,
        sha256: str,
        size_bytes: int,
    ) -> R2UploadPlan:
        """Create an R2 multipart upload and presign every uniform part."""
        assert self.s3_client is not None
        part_size_bytes = max(
            self.configuration.multipart_part_bytes,
            math.ceil(size_bytes / R2_MAX_MULTIPART_PARTS),
        )
        part_size_bytes = math.ceil(part_size_bytes / (1024**2)) * 1024**2
        if part_size_bytes > R2_MAX_MULTIPART_PART_BYTES:
            raise R2CacheError(
                "R2 object is too large for the supported multipart limits."
            )
        part_count = math.ceil(size_bytes / part_size_bytes)
        upload_id: str | None = None
        try:
            created = self.s3_client.create_multipart_upload(
                Bucket=self.configuration.bucket,
                Key=key,
                Metadata={"sha256": sha256, "size-bytes": str(size_bytes)},
            )
            raw_upload_id = created["UploadId"]
            if not isinstance(raw_upload_id, str) or not raw_upload_id.strip():
                raise ValueError("R2 returned an invalid multipart upload ID.")
            upload_id = raw_upload_id.strip()
            urls = tuple(
                self.s3_client.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": self.configuration.bucket,
                        "Key": key,
                        "UploadId": upload_id,
                        "PartNumber": part_number,
                    },
                    ExpiresIn=self.configuration.url_ttl_seconds,
                )
                for part_number in range(1, part_count + 1)
            )
            for url in urls:
                self._validate_generated_url(url)
        except (KeyError, TypeError, *_R2_CLIENT_ERRORS) as exc:
            if upload_id is not None:
                self._abort_multipart_key(key, upload_id)
            raise R2CacheError(
                f"Unable to create R2 multipart upload for {key!r}: {exc}"
            ) from exc
        assert upload_id is not None
        return R2UploadPlan(
            key=key,
            sha256=sha256,
            size_bytes=size_bytes,
            allowed_host=self.configuration.endpoint_host,
            mode="multipart",
            urls=urls,
            part_size_bytes=part_size_bytes,
            upload_id=upload_id,
        )

    def _head_size(self, key: str) -> int | None:
        """Return an object's size, or None when R2 reports a cache miss."""
        assert self.s3_client is not None
        try:
            response = self.s3_client.head_object(
                Bucket=self.configuration.bucket,
                Key=key,
            )
        except _R2_CLIENT_ERRORS as exc:
            if self._is_missing_object_error(exc):
                return None
            raise R2CacheError(
                f"Unable to inspect R2 cache object {key!r}: {exc}"
            ) from exc
        content_length = response.get("ContentLength")
        if (
            isinstance(content_length, bool)
            or not isinstance(content_length, int)
            or content_length < 0
        ):
            raise R2CacheError(f"R2 returned an invalid size for cache object {key!r}.")
        return content_length

    @staticmethod
    def _is_missing_object_error(error: BaseException) -> bool:
        """Return whether a boto-compatible error represents an absent object."""
        response = getattr(error, "response", None)
        if not isinstance(response, Mapping):
            return False
        error_payload = response.get("Error")
        metadata = response.get("ResponseMetadata")
        code = (
            str(error_payload.get("Code") or "")
            if isinstance(error_payload, Mapping)
            else ""
        )
        status = (
            metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
        )
        return code in {"404", "NoSuchKey", "NotFound"} or status == 404

    def _validate_generated_url(self, url: object) -> None:
        """Require boto to return a URL bound to the configured R2 HTTPS origin."""
        if not isinstance(url, str):
            raise R2CacheError("R2 client returned a non-string presigned URL.")
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != self.configuration.endpoint_host
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.fragment)
        ):
            raise R2CacheError(
                "R2 client returned a presigned URL for an unexpected host."
            )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local prewarm source."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(_HASH_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "R2CacheClient",
    "R2CacheConfiguration",
    "R2CacheError",
    "R2DownloadRequest",
    "R2StorageUsage",
    "R2UploadPlan",
    "R2UploadResult",
    "R2UploadedPart",
    "R2WorkerPreflightRequest",
]
