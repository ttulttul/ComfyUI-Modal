"""Download and atomically publish one verified Hugging Face asset on a worker."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ and "." in __package__:
    from ..huggingface_assets import HuggingFaceAssetSource, sha256_file
else:  # pragma: no cover - direct remote entrypoint compatibility.
    from huggingface_assets import HuggingFaceAssetSource, sha256_file

logger = logging.getLogger(__name__)
DownloadFile = Callable[..., str]


@dataclass(frozen=True)
class HuggingFaceMaterializationRequest:
    """Describe one verified Hugging Face file publication request."""

    source: HuggingFaceAssetSource
    storage_root: Path
    remote_path: str
    token: str | None = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "HuggingFaceMaterializationRequest":
        """Build and validate one request received over the SSH standard input."""
        source_payload = payload.get("source")
        if not isinstance(source_payload, Mapping):
            raise ValueError("Hugging Face materialization source must be an object.")
        token_value = payload.get("token")
        token = None if token_value is None or token_value == "" else str(token_value)
        if token is not None and any(character in token for character in "\x00\n\r"):
            raise ValueError("Hugging Face token must be a single-line value.")
        return cls(
            source=HuggingFaceAssetSource.from_dict(source_payload),
            storage_root=Path(str(payload.get("storage_root") or "")),
            remote_path=str(payload.get("remote_path") or ""),
            token=token,
        )


@dataclass(frozen=True)
class HuggingFaceMaterializationResult:
    """Report whether one verified target was newly published or already present."""

    target_path: str
    created: bool
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a safe JSON-compatible result without credentials or cache paths."""
        return {
            "created": self.created,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "target_path": self.target_path,
        }


def _validated_target_path(storage_root: Path, remote_path: str) -> Path:
    """Resolve one requested target strictly below the configured storage root."""
    root = storage_root.expanduser().resolve()
    if not root.is_absolute() or root == Path("/"):
        raise ValueError("Hugging Face materialization storage_root must be absolute and non-root.")
    relative = Path(remote_path.lstrip("/"))
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\x00" in remote_path
    ):
        raise ValueError("Hugging Face materialization remote_path is unsafe.")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Hugging Face materialization target escapes storage_root.") from exc
    return target


def _validate_file(path: Path, source: HuggingFaceAssetSource) -> None:
    """Require one file to match the registered size and content digest."""
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face download did not produce a file: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.size_bytes:
        raise ValueError(
            "Hugging Face asset size mismatch for "
            f"{source.display_reference}: expected {source.size_bytes}, found {actual_size}."
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != source.sha256:
        raise ValueError(
            "Hugging Face asset SHA-256 mismatch for "
            f"{source.display_reference}: expected {source.sha256}, found {actual_sha256}."
        )


def _atomic_publish(source_path: Path, target_path: Path) -> None:
    """Publish one verified local file atomically, preferring a same-device hard link."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        try:
            os.link(source_path, temporary_path)
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                raise
            shutil.copyfile(source_path, temporary_path)
            with temporary_path.open("rb") as copied_file:
                os.fsync(copied_file.fileno())
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize_huggingface_asset(
    request: HuggingFaceMaterializationRequest,
    *,
    download_file: DownloadFile | None = None,
) -> HuggingFaceMaterializationResult:
    """Fetch, verify, and atomically publish one immutable Hugging Face file."""
    target_path = _validated_target_path(request.storage_root, request.remote_path)
    if target_path.exists():
        _validate_file(target_path, request.source)
        return HuggingFaceMaterializationResult(
            target_path=request.remote_path,
            created=False,
            size_bytes=request.source.size_bytes,
            sha256=request.source.sha256,
        )

    if download_file is None:
        from huggingface_hub import hf_hub_download

        download_file = hf_hub_download
    cache_dir = request.storage_root.expanduser().resolve() / "huggingface-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading Hugging Face asset source=%s size_bytes=%d.",
        request.source.display_reference,
        request.source.size_bytes,
    )
    downloaded_path = Path(
        download_file(
            repo_id=request.source.repo_id,
            filename=request.source.filename,
            revision=request.source.revision,
            token=request.token,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    _validate_file(downloaded_path, request.source)
    _atomic_publish(downloaded_path, target_path)
    _validate_file(target_path, request.source)
    logger.info(
        "Published verified Hugging Face asset source=%s target=%s.",
        request.source.display_reference,
        request.remote_path,
    )
    return HuggingFaceMaterializationResult(
        target_path=request.remote_path,
        created=True,
        size_bytes=request.source.size_bytes,
        sha256=request.source.sha256,
    )


def main() -> int:
    """Read one credential-bearing request from stdin and emit a safe JSON result."""
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    payload = json.load(sys.stdin)
    if not isinstance(payload, Mapping):
        raise ValueError("Hugging Face materialization request must be a JSON object.")
    request = HuggingFaceMaterializationRequest.from_dict(payload)
    result = materialize_huggingface_asset(request)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in the remote worker.
    raise SystemExit(main())


__all__ = [
    "HuggingFaceMaterializationRequest",
    "HuggingFaceMaterializationResult",
    "main",
    "materialize_huggingface_asset",
]
