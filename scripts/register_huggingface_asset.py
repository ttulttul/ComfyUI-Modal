"""Register one local model file as an immutable Hugging Face-backed asset."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_assets import (  # noqa: E402
    HuggingFaceAssetRegistry,
    HuggingFaceAssetSource,
    sha256_file,
)
from settings import discover_comfyui_user_directory, get_settings  # noqa: E402

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser for one explicit provenance association."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_path", type=Path, help="Existing local model file.")
    parser.add_argument("repo_id", help="Hugging Face model repository, owner/model.")
    parser.add_argument(
        "filename",
        help="Repository-relative filename corresponding to the local file.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Revision to resolve to an immutable commit; defaults to main.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Explicit registry JSON path; defaults below the ComfyUI user directory.",
    )
    return parser


def _field(value: object, name: str) -> object | None:
    """Read one field from a Hugging Face object or test mapping."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _sibling_lfs_sha256(sibling: object) -> str | None:
    """Return the raw content SHA-256 advertised for one Hub LFS/Xet file."""
    lfs = _field(sibling, "lfs")
    if lfs is None:
        return None
    value = _field(lfs, "sha256") or _field(lfs, "oid")
    normalized = str(value or "").strip().lower()
    return normalized or None


def resolve_huggingface_asset_source(
    local_path: Path,
    *,
    repo_id: str,
    filename: str,
    revision: str,
    api: Any,
) -> HuggingFaceAssetSource:
    """Resolve Hub metadata and require it to match the supplied local file exactly."""
    resolved_path = local_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Local model asset not found: {resolved_path}")
    model_info = api.model_info(
        repo_id,
        revision=revision,
        files_metadata=True,
    )
    exact_revision = str(_field(model_info, "sha") or "").strip().lower()
    siblings = _field(model_info, "siblings")
    if not isinstance(siblings, Sequence):
        raise ValueError(f"Hugging Face repository {repo_id!r} returned no file metadata.")
    sibling = next(
        (
            item
            for item in siblings
            if str(_field(item, "rfilename") or "") == filename
        ),
        None,
    )
    if sibling is None:
        raise ValueError(
            f"Hugging Face repository {repo_id!r} has no file {filename!r} at "
            f"revision {revision!r}."
        )
    local_sha256 = sha256_file(resolved_path)
    remote_sha256 = _sibling_lfs_sha256(sibling)
    if remote_sha256 is None:
        raise ValueError(
            f"Hugging Face file {repo_id}/{filename} does not expose an LFS/Xet "
            "content SHA-256 and cannot be registered safely."
        )
    if remote_sha256 != local_sha256:
        raise ValueError(
            "Local file SHA-256 does not match Hugging Face metadata: "
            f"local={local_sha256} remote={remote_sha256}."
        )
    remote_size_value = _field(sibling, "size") or _field(_field(sibling, "lfs"), "size")
    remote_size = int(remote_size_value or 0)
    local_size = resolved_path.stat().st_size
    if remote_size != local_size:
        raise ValueError(
            "Local file size does not match Hugging Face metadata: "
            f"local={local_size} remote={remote_size}."
        )
    return HuggingFaceAssetSource(
        repo_id=repo_id,
        revision=exact_revision,
        filename=filename,
        sha256=local_sha256,
        size_bytes=local_size,
    )


def _registry(explicit_path: Path | None) -> HuggingFaceAssetRegistry:
    """Return the explicit registry or discover the persistent ComfyUI registry."""
    if explicit_path is not None:
        return HuggingFaceAssetRegistry(explicit_path.expanduser().resolve())
    user_directory = discover_comfyui_user_directory(get_settings())
    if user_directory is None:
        raise RuntimeError(
            "Unable to discover the ComfyUI user directory; pass --registry explicitly."
        )
    return HuggingFaceAssetRegistry.for_user_directory(user_directory)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one association against the Hub and persist its provenance record."""
    arguments = _parser().parse_args(argv)
    from huggingface_hub import HfApi

    source = resolve_huggingface_asset_source(
        arguments.local_path,
        repo_id=arguments.repo_id,
        filename=arguments.filename,
        revision=arguments.revision,
        api=HfApi(),
    )
    registry = _registry(arguments.registry)
    registry.upsert(source)
    print(
        f"Registered {source.display_reference} as {source.sha256} in "
        f"{registry.config_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
