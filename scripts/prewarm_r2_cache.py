"""Prewarm the optional Cloudflare R2 backing cache from local files."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from r2_cache import R2CacheClient

logger = logging.getLogger(__name__)
_BUFFER_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Expand files and directories into unique regular files in stable order."""
    discovered: dict[Path, None] = {}
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            discovered[path] = None
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and not candidate.is_symlink():
                    discovered[candidate.resolve()] = None
            continue
        raise FileNotFoundError(f"R2 prewarm path not found: {path}")
    return tuple(discovered)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for local cache prewarming."""
    parser = argparse.ArgumentParser(
        description=(
            "Upload files into the configured content-addressed Cloudflare R2 cache."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to hash and upload recursively.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prewarm configured R2 storage and report uploads versus cache hits."""
    arguments = build_argument_parser().parse_args(argv)
    cache = R2CacheClient.from_environment()
    if cache is None:
        raise RuntimeError(
            "Set COMFY_MODAL_R2_ENABLED=true and configure the R2 controller credentials."
        )
    files = discover_files(arguments.paths)
    uploaded_count = 0
    for index, path in enumerate(files, start=1):
        digest = sha256_file(path)
        uploaded = cache.upload_local_file(path, sha256=digest)
        uploaded_count += int(uploaded)
        logger.info(
            "R2 prewarm %d/%d %s: %s",
            index,
            len(files),
            "uploaded" if uploaded else "already cached",
            path,
        )
    logger.info(
        "R2 prewarm complete: %d uploaded, %d already cached.",
        uploaded_count,
        len(files) - uploaded_count,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())


__all__ = ["build_argument_parser", "discover_files", "main", "sha256_file"]
