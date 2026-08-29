"""Build and optionally push the shared worker image used by Vast.ai leases."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from string import Formatter
from typing import Any, Sequence
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_environment import (
    build_vast_runtime_identity,
    remote_runtime_dependency_fingerprint,
)
from settings import get_settings
from ssh_runtime import (
    DEPENDENCY_FINGERPRINT_LABEL,
    RUNTIME_FINGERPRINT_LABEL,
    export_worker_dependency_image_context,
    export_worker_source_overlay_context,
)

logger = logging.getLogger(__name__)

DEFAULT_TAG_TEMPLATE = "ghcr.io/{owner}/comfy-modal-worker:v{version}"
VAST_WORKER_PLATFORM = "linux/amd64"
SOURCE_FINGERPRINT_RESULT_PREFIX = "COMFY_MODAL_VAST_SOURCE_FINGERPRINT="
_SHA256_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help=(
            "Explicit versioned registry tag. By default, derive a GHCR tag from "
            "pyproject.toml."
        ),
    )
    parser.add_argument(
        "--owner",
        help=(
            "Container registry owner used by the default tag template. Defaults to "
            "the GitHub owner in pyproject.toml's Repository URL."
        ),
    )
    parser.add_argument(
        "--tag-template",
        default=DEFAULT_TAG_TEMPLATE,
        help=(
            "Default tag template when --tag is omitted. Supports {owner} and "
            "{version}."
        ),
    )
    parser.add_argument(
        "--comfyui-root",
        type=Path,
        help=(
            "ComfyUI source checkout to embed. Defaults to the active installation, "
            "COMFYUI_ROOT, ~/git/ComfyUI, or ~/git/Latest_ComfyUI."
        ),
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the built tag and print its registry digest reference.",
    )
    parser.add_argument(
        "--expected-fingerprint",
        type=_runtime_fingerprint,
        help=(
            "Abort before building if current source no longer matches this SHA-256 "
            "runtime fingerprint. Used by automatic publication."
        ),
    )
    return parser


def _runtime_fingerprint(value: str) -> str:
    """Return one normalized SHA-256 runtime fingerprint for argparse."""
    normalized = value.strip().casefold()
    if not _SHA256_FINGERPRINT_PATTERN.fullmatch(normalized):
        raise argparse.ArgumentTypeError(
            "runtime fingerprint must be a 64-character lowercase SHA-256 value"
        )
    return normalized


def _project_metadata(repo_root: Path = REPO_ROOT) -> Mapping[str, Any]:
    """Load and return the PEP 621 project metadata from pyproject.toml."""
    pyproject_path = repo_root / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f"Unable to read project metadata from {pyproject_path}."
        ) from error
    project = document.get("project")
    if not isinstance(project, Mapping):
        raise ValueError(f"Project metadata is missing from {pyproject_path}.")
    return project


def _project_version(project: Mapping[str, Any]) -> str:
    """Return the normalized project version used in the worker image tag."""
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml must define a non-empty project.version.")
    normalized = version.strip()
    if any(character in normalized for character in "/:@"):
        raise ValueError(
            "project.version contains characters that are invalid in a tag."
        )
    return normalized


def _repository_url(project: Mapping[str, Any]) -> str:
    """Return the repository URL declared in PEP 621 project metadata."""
    urls = project.get("urls")
    if not isinstance(urls, Mapping):
        raise ValueError("pyproject.toml must define project.urls.Repository.")
    repository = next(
        (
            value
            for key, value in urls.items()
            if str(key).casefold() == "repository" and isinstance(value, str)
        ),
        None,
    )
    if repository is None or not repository.strip():
        raise ValueError("pyproject.toml must define project.urls.Repository.")
    return repository.strip()


def _github_owner(repository_url: str) -> str:
    """Extract a normalized GitHub owner from an HTTPS, SSH, or SCP-style URL."""
    candidate = repository_url.strip()
    if candidate.startswith("git@github.com:"):
        repository_path = candidate.removeprefix("git@github.com:")
    else:
        parsed = urlparse(candidate)
        if parsed.hostname != "github.com":
            raise ValueError(
                "Repository URL must point to github.com; pass --owner or --tag."
            )
        repository_path = parsed.path.lstrip("/")
    return _normalize_owner(repository_path.split("/", maxsplit=1)[0])


def _normalize_owner(owner: str) -> str:
    """Normalize and validate a GitHub-compatible container registry owner."""
    normalized = owner.strip().casefold()
    if not normalized or any(
        not (character.isascii() and (character.isalnum() or character == "-"))
        for character in normalized
    ):
        raise ValueError(
            "Unable to infer a valid GHCR owner; pass --owner or --tag."
        )
    return normalized


def _render_tag_template(tag_template: str, *, owner: str, version: str) -> str:
    """Render a tag template containing only simple owner and version fields."""
    try:
        parsed_fields = tuple(Formatter().parse(tag_template))
    except ValueError as error:
        raise ValueError("Tag template is malformed.") from error
    invalid_field = any(
        field_name not in {None, "owner", "version"}
        or bool(format_spec)
        or conversion is not None
        for _, field_name, format_spec, conversion in parsed_fields
    )
    if invalid_field:
        raise ValueError(
            "Tag template may contain only the {owner} and {version} fields."
        )
    return tag_template.format(owner=owner, version=version)


def _resolve_image_tag(
    explicit_tag: str | None,
    *,
    owner: str | None,
    tag_template: str,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Return an explicit tag or render the project-derived default image tag."""
    if explicit_tag is not None:
        return _validate_tag(explicit_tag)
    project = _project_metadata(repo_root)
    resolved_owner = (
        _github_owner(_repository_url(project))
        if owner is None
        else _normalize_owner(owner)
    )
    rendered = _render_tag_template(
        tag_template,
        owner=resolved_owner,
        version=_project_version(project),
    )
    return _validate_tag(rendered)


def _validate_tag(tag: str) -> str:
    """Require an explicit version tag rather than a mutable latest reference."""
    normalized = tag.strip()
    final_component = normalized.rsplit("/", maxsplit=1)[-1]
    if not normalized or ":" not in final_component:
        raise ValueError("Vast worker image must use an explicit version tag.")
    if normalized.casefold().endswith(":latest"):
        raise ValueError("Vast worker image must not use the mutable latest tag.")
    return normalized


def _run(
    command: Sequence[str],
    *,
    input_payload: bytes | None = None,
    capture_stdout: bool = True,
) -> str:
    """Run one Docker command, optionally inheriting stdout for live progress."""
    completed = subprocess.run(
        tuple(command),
        input=input_payload,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=None,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command {command[0]!r} failed with status {completed.returncode}."
        )
    if completed.stdout is None:
        return ""
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _docker_build_command(
    image_tag: str,
    runtime_fingerprint: str,
    *,
    dependency_fingerprint: str | None = None,
    pull: bool = False,
) -> tuple[str, ...]:
    """Return the architecture-pinned Docker command for one worker image."""
    command = [
        "docker",
        "build",
        "--platform",
        VAST_WORKER_PLATFORM,
        "--label",
        f"{RUNTIME_FINGERPRINT_LABEL}={runtime_fingerprint}",
    ]
    if dependency_fingerprint is not None:
        command.extend(
            ("--label", f"{DEPENDENCY_FINGERPRINT_LABEL}={dependency_fingerprint}")
        )
    if pull:
        command.append("--pull")
    command.extend(("-t", image_tag, "-"))
    return tuple(command)


def _docker_dependency_build_command(
    image_tag: str,
    dependency_fingerprint: str,
) -> tuple[str, ...]:
    """Return the Docker command for a stable registry-backed dependency base."""
    return (
        "docker",
        "build",
        "--platform",
        VAST_WORKER_PLATFORM,
        "--pull",
        "--label",
        f"{DEPENDENCY_FINGERPRINT_LABEL}={dependency_fingerprint}",
        "-t",
        image_tag,
        "-",
    )


def _image_repository(image_tag: str) -> str:
    """Return an image repository with any tag or digest removed."""
    without_digest = image_tag.split("@", maxsplit=1)[0]
    final_slash = without_digest.rfind("/")
    final_colon = without_digest.rfind(":")
    return (
        without_digest[:final_colon]
        if final_colon > final_slash
        else without_digest
    )


def _dependency_image_tag(image_tag: str, dependency_fingerprint: str) -> str:
    """Return the stable registry tag for one dependency manifest."""
    return f"{_image_repository(image_tag)}:deps-{dependency_fingerprint[:16]}"


def _local_image_label(image: str, label: str) -> str | None:
    """Return one local Docker image label when the image is inspectable."""
    try:
        value = _run(
            (
                "docker",
                "image",
                "inspect",
                "--format",
                f'{{{{index .Config.Labels "{label}"}}}}',
                image,
            )
        )
    except RuntimeError:
        return None
    return value or None


def _pull_current_dependency(image: str, dependency_fingerprint: str) -> bool:
    """Pull and validate a reusable dependency image from its registry."""
    logger.info("Checking registry for Vast worker dependency image tag=%s.", image)
    try:
        _run(
            ("docker", "pull", "--platform", VAST_WORKER_PLATFORM, image),
            capture_stdout=False,
        )
    except RuntimeError:
        logger.info(
            "No reusable Vast worker dependency image was pulled tag=%s.",
            image,
        )
        return False
    actual = _local_image_label(image, DEPENDENCY_FINGERPRINT_LABEL)
    if actual == dependency_fingerprint:
        logger.info(
            "Reusing registry-backed Vast worker dependency image tag=%s.",
            image,
        )
        return True
    logger.warning(
        "Ignoring Vast dependency image with unexpected label tag=%s expected=%s "
        "actual=%s.",
        image,
        dependency_fingerprint[:12],
        (actual or "missing")[:12],
    )
    return False


def _repository_digest(image_tag: str) -> str:
    """Resolve one pushed or pulled tag to an immutable repository digest."""
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
    repository = _image_repository(image_tag)
    digest = next(
        (
            line
            for line in repository_digests.splitlines()
            if line.startswith(f"{repository}@sha256:")
        ),
        None,
    )
    if digest is None:
        raise RuntimeError(
            "Registry operation succeeded but no repository digest is available "
            f"for {image_tag}."
        )
    return digest


def build_image(
    tag: str,
    *,
    push: bool,
    comfyui_root: Path | None = None,
    expected_fingerprint: str | None = None,
) -> str:
    """Build the current runtime, optionally push it, and return its image reference."""
    image_tag = _validate_tag(tag)
    repo_root = REPO_ROOT
    settings = get_settings()
    if comfyui_root is not None:
        resolved_comfyui_root = comfyui_root.expanduser().resolve()
        if not (resolved_comfyui_root / "main.py").is_file() or not (
            resolved_comfyui_root / "nodes.py"
        ).is_file():
            raise ValueError(
                f"ComfyUI root {resolved_comfyui_root} must contain main.py and nodes.py."
            )
        custom_nodes_dir = resolved_comfyui_root / "custom_nodes"
        settings = replace(
            settings,
            comfyui_root=resolved_comfyui_root,
            custom_nodes_dir=(
                custom_nodes_dir if custom_nodes_dir.is_dir() else None
            ),
        )
    identity = build_vast_runtime_identity(
        repo_root=repo_root,
        comfyui_root=settings.comfyui_root,
        custom_nodes_dir=settings.custom_nodes_dir,
        settings=settings,
    )
    print(f"{SOURCE_FINGERPRINT_RESULT_PREFIX}{identity.fingerprint}", flush=True)
    if (
        expected_fingerprint is not None
        and identity.fingerprint != expected_fingerprint
    ):
        raise RuntimeError(
            "Local runtime source changed after ComfyUI started: expected "
            f"{expected_fingerprint[:12]}, found {identity.fingerprint[:12]}. "
            "Restart ComfyUI before publishing a replacement worker image."
        )
    dependency_fingerprint = remote_runtime_dependency_fingerprint(identity)
    dependency_tag = _dependency_image_tag(image_tag, dependency_fingerprint)
    if not _pull_current_dependency(dependency_tag, dependency_fingerprint):
        dependency_context = export_worker_dependency_image_context(
            repo_root=repo_root,
            settings=settings,
        )
        logger.info(
            "Building Vast worker dependency image tag=%s fingerprint=%s "
            "context_bytes=%d.",
            dependency_tag,
            dependency_fingerprint,
            len(dependency_context),
        )
        _run(
            _docker_dependency_build_command(
                dependency_tag,
                dependency_fingerprint,
            ),
            input_payload=dependency_context,
            capture_stdout=False,
        )
        if push:
            logger.info("Pushing Vast worker dependency image tag=%s.", dependency_tag)
            _run(("docker", "push", dependency_tag), capture_stdout=False)
    dependency_image = (
        _repository_digest(dependency_tag) if push else dependency_tag
    )
    context = export_worker_source_overlay_context(
        repo_root=repo_root,
        settings=settings,
        identity=identity,
        dependency_image=dependency_image,
    )
    logger.info(
        "Applying Vast worker source overlay tag=%s fingerprint=%s "
        "dependency_image=%s context_bytes=%d.",
        image_tag,
        identity.fingerprint,
        dependency_image,
        len(context),
    )
    _run(
        _docker_build_command(
            image_tag,
            identity.fingerprint,
            dependency_fingerprint=dependency_fingerprint,
        ),
        input_payload=context,
        capture_stdout=False,
    )
    logger.info("Finished building Vast worker image tag=%s.", image_tag)
    if not push:
        return image_tag
    logger.info("Pushing Vast worker image tag=%s.", image_tag)
    _run(("docker", "push", image_tag), capture_stdout=False)
    logger.info("Resolving immutable digest for Vast worker image tag=%s.", image_tag)
    return _repository_digest(image_tag)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested image and print the exact runtime configuration value."""
    arguments = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    image_tag = _resolve_image_tag(
        arguments.tag,
        owner=arguments.owner,
        tag_template=arguments.tag_template,
    )
    image_reference = build_image(
        image_tag,
        push=arguments.push,
        comfyui_root=arguments.comfyui_root,
        expected_fingerprint=arguments.expected_fingerprint,
    )
    print(f"COMFY_MODAL_VAST_IMAGE={image_reference}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
