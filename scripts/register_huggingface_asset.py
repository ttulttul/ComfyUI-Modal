"""Register one local model file as an immutable Hugging Face-backed asset."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
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
from huggingface_discovery import (  # noqa: E402
    HuggingFaceAssetHint,
    resolve_huggingface_asset_hint,
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
    local_sha256 = sha256_file(resolved_path)
    source = resolve_huggingface_asset_hint(
        HuggingFaceAssetHint(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            evidence="manual diagnostic override",
        ),
        sha256=local_sha256,
        size_bytes=resolved_path.stat().st_size,
        api=api,
    )
    if source is None:
        raise ValueError(
            "Local file does not match Hugging Face metadata for "
            f"{repo_id}/{filename} at revision {revision!r}."
        )
    return source


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
