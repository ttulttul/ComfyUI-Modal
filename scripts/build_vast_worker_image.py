"""Build and optionally push the shared worker image used by Vast.ai leases."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Sequence

from runtime_environment import build_remote_runtime_identity
from settings import get_settings
from ssh_runtime import export_worker_image_context

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        required=True,
        help="Versioned registry tag, for example ghcr.io/owner/comfy-worker:v0.4.0.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the built tag and print its registry digest reference.",
    )
    return parser


def _validate_tag(tag: str) -> str:
    """Require an explicit version tag rather than a mutable latest reference."""
    normalized = tag.strip()
    final_component = normalized.rsplit("/", maxsplit=1)[-1]
    if not normalized or ":" not in final_component:
        raise ValueError("Vast worker image must use an explicit version tag.")
    if normalized.casefold().endswith(":latest"):
        raise ValueError("Vast worker image must not use the mutable latest tag.")
    return normalized


def _run(command: Sequence[str], *, input_payload: bytes | None = None) -> str:
    """Run one local Docker command and return trimmed standard output."""
    completed = subprocess.run(
        tuple(command),
        input=input_payload,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command {command[0]!r} failed with status {completed.returncode}."
        )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def build_image(tag: str, *, push: bool) -> str:
    """Build the current runtime, optionally push it, and return its image reference."""
    image_tag = _validate_tag(tag)
    repo_root = Path(__file__).resolve().parents[1]
    settings = get_settings()
    identity = build_remote_runtime_identity(
        repo_root=repo_root,
        comfyui_root=settings.comfyui_root,
        custom_nodes_dir=settings.custom_nodes_dir,
        settings=settings,
    )
    context = export_worker_image_context(
        repo_root=repo_root,
        settings=settings,
        identity=identity,
    )
    logger.info(
        "Building Vast worker image tag=%s fingerprint=%s context_bytes=%d.",
        image_tag,
        identity.fingerprint,
        len(context),
    )
    _run(
        (
            "docker",
            "build",
            "--pull",
            "--label",
            f"comfy.remote.runtime-fingerprint={identity.fingerprint}",
            "-t",
            image_tag,
            "-",
        ),
        input_payload=context,
    )
    if not push:
        return image_tag
    _run(("docker", "push", image_tag))
    repository_digests = _run(
        (
            "docker",
            "image",
            "inspect",
            "--format",
            "{{join .RepoDigests \"\\n\"}}",
            image_tag,
        )
    )
    digest = next(
        (line for line in repository_digests.splitlines() if "@sha256:" in line),
        None,
    )
    if digest is None:
        raise RuntimeError("Registry push succeeded but no repository digest is available.")
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested image and print the exact runtime configuration value."""
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    image_reference = build_image(arguments.tag, push=arguments.push)
    print(f"COMFY_MODAL_VAST_IMAGE={image_reference}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
