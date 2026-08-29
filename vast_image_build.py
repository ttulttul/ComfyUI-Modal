"""Build and publish a current worker image after Vast source drift."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

if __package__:
    from .vast_image_registry import (
        VastImageNotFoundError,
        VastImageRegistryError,
        published_image_metadata,
    )
else:  # pragma: no cover - direct debugging imports.
    from vast_image_registry import (
        VastImageNotFoundError,
        VastImageRegistryError,
        published_image_metadata,
    )

logger = logging.getLogger(__name__)

VAST_IMAGE_BUILD_COMMAND = (
    "uv",
    "run",
    "python",
    "scripts/build_vast_worker_image.py",
    "--push",
)
_IMAGE_RESULT_PREFIX = "COMFY_MODAL_VAST_IMAGE="
_SOURCE_FINGERPRINT_RESULT_PREFIX = "COMFY_MODAL_VAST_SOURCE_FINGERPRINT="
_DIGEST_REFERENCE_PATTERN = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_IN_TEXT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_MAXIMUM_STATUS_LINE_CHARACTERS = 300
_BUILD_LOCK = threading.Lock()
_BUILT_IMAGES_BY_FINGERPRINT: dict[str, str] = {}

VastImageBuildStatusCallback = Callable[[str], None]


class VastWorkerImageBuildError(RuntimeError):
    """Report that the automatic Vast worker image build could not finish."""


@dataclass(frozen=True)
class VastWorkerImageBuilder:
    """Run the repository image builder and stream bounded progress messages."""

    repo_root: Path
    comfyui_root: Path | None
    modal_gpu: str | None = None
    environment: Mapping[str, str] | None = None

    def ensure_published_image(
        self,
        image: str,
        expected_fingerprint: str,
        *,
        status_callback: VastImageBuildStatusCallback | None = None,
    ) -> str:
        """Return a current published image, rebuilding before rental if stale."""
        self._emit(status_callback, "Checking the published Vast worker image")
        try:
            published_image = published_image_metadata(image)
        except VastImageNotFoundError:
            self._emit(
                status_callback,
                "Published Vast worker image is missing; building before requesting "
                "capacity",
            )
            return self.build_and_push(
                expected_fingerprint,
                status_callback=status_callback,
            )
        except (VastImageRegistryError, ValueError) as exc:
            raise VastWorkerImageBuildError(
                self._manual_build_message(
                    "Unable to inspect the configured Vast worker image before "
                    f"renting capacity: {exc} Ensure the image is publicly readable."
                )
            ) from exc
        actual_fingerprint = published_image.runtime_fingerprint
        if actual_fingerprint == expected_fingerprint:
            if published_image.immutable_image is None:
                raise VastWorkerImageBuildError(
                    self._manual_build_message(
                        "The registry did not return an immutable linux/amd64 digest "
                        f"for {image!r}."
                    )
                )
            self._emit(status_callback, "Published Vast worker image is current")
            return published_image.immutable_image
        actual_summary = (
            actual_fingerprint[:12]
            if actual_fingerprint is not None
            else "missing label"
        )
        logger.warning(
            "Published Vast worker image is stale image=%s expected=%s actual=%s.",
            image,
            expected_fingerprint[:12],
            actual_summary,
        )
        self._emit(
            status_callback,
            "Published Vast worker image is stale; rebuilding before requesting "
            "capacity",
        )
        return self.build_and_push(
            expected_fingerprint,
            status_callback=status_callback,
        )

    def build_and_push(
        self,
        expected_fingerprint: str,
        *,
        status_callback: VastImageBuildStatusCallback | None = None,
    ) -> str:
        """Build once per fingerprint and return the published digest reference."""
        with _BUILD_LOCK:
            cached = _BUILT_IMAGES_BY_FINGERPRINT.get(expected_fingerprint)
            if cached is not None:
                self._emit(
                    status_callback,
                    "Using the worker image published by another workflow",
                )
                return cached
            image = self._run_build(
                expected_fingerprint=expected_fingerprint,
                status_callback=status_callback,
            )
            _BUILT_IMAGES_BY_FINGERPRINT[expected_fingerprint] = image
            return image

    def _run_build(
        self,
        *,
        expected_fingerprint: str,
        status_callback: VastImageBuildStatusCallback | None,
    ) -> str:
        """Execute the image builder and extract its immutable result reference."""
        command = self._automatic_build_command(expected_fingerprint)
        command_text = shlex.join(command)
        self._emit(status_callback, "Building the current Vast worker image")
        environment = dict(os.environ if self.environment is None else self.environment)
        environment.setdefault("BUILDKIT_PROGRESS", "plain")
        if self.comfyui_root is not None:
            environment["COMFYUI_ROOT"] = str(self.comfyui_root)
        if self.modal_gpu is not None:
            environment["COMFY_MODAL_GPU"] = self.modal_gpu
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise VastWorkerImageBuildError(
                self._manual_build_message(
                    f"Unable to start {command[0]!r}: {exc}"
                )
            ) from exc
        image_reference, source_fingerprint, recent_output = self._consume_output(
            process.stdout,
            status_callback=status_callback,
        )
        return_code = process.wait()
        if (
            source_fingerprint is not None
            and source_fingerprint != expected_fingerprint
        ):
            raise VastWorkerImageBuildError(
                "Automatic Vast worker image publication stopped because local "
                "runtime source changed after ComfyUI started: expected "
                f"{expected_fingerprint[:12]}, found "
                f"{source_fingerprint[:12]}. Restart ComfyUI and retry the "
                "workflow; no manual image build is needed."
            )
        if return_code != 0:
            diagnostic = self._failure_diagnostic(recent_output)
            raise VastWorkerImageBuildError(
                self._manual_build_message(
                    f"{command_text} exited with status {return_code}: {diagnostic}"
                )
            )
        if image_reference is None:
            raise VastWorkerImageBuildError(
                self._manual_build_message(
                    "The build completed without returning an immutable image digest."
                )
            )
        if source_fingerprint is None:
            raise VastWorkerImageBuildError(
                self._manual_build_message(
                    "The build did not report its baked runtime fingerprint."
                )
            )
        self._emit(status_callback, "Published the current Vast worker image")
        return image_reference

    @staticmethod
    def _automatic_build_command(expected_fingerprint: str) -> tuple[str, ...]:
        """Run with the exact interpreter hosting the active ComfyUI process."""
        return (
            sys.executable,
            "scripts/build_vast_worker_image.py",
            "--push",
            "--expected-fingerprint",
            expected_fingerprint,
        )

    @staticmethod
    def _failure_diagnostic(recent_output: tuple[str, ...]) -> str:
        """Return the last substantive build line instead of traceback footer noise."""
        ignored_lines = {"<no Python frame>", "[stderr]", "]"}
        for line in reversed(recent_output):
            if line not in ignored_lines and not line.startswith("Current thread "):
                return line
        return recent_output[-1] if recent_output else "no build output"

    def _consume_output(
        self,
        output: IO[str] | None,
        *,
        status_callback: VastImageBuildStatusCallback | None,
    ) -> tuple[str | None, str | None, tuple[str, ...]]:
        """Forward output and return its digest, source identity, and diagnostics."""
        if output is None:
            return None, None, ()
        image_reference: str | None = None
        source_fingerprint: str | None = None
        recent_output: list[str] = []
        for raw_line in output:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(_IMAGE_RESULT_PREFIX):
                candidate = line.removeprefix(_IMAGE_RESULT_PREFIX).strip()
                if _DIGEST_REFERENCE_PATTERN.fullmatch(candidate):
                    image_reference = candidate
                continue
            if line.startswith(_SOURCE_FINGERPRINT_RESULT_PREFIX):
                candidate = line.removeprefix(
                    _SOURCE_FINGERPRINT_RESULT_PREFIX
                ).strip()
                if _FINGERPRINT_PATTERN.fullmatch(candidate):
                    source_fingerprint = candidate
                    continue
            bounded = self._safe_status_line(line)
            recent_output.append(bounded)
            del recent_output[:-10]
            logger.info("Vast worker image build: %s", bounded)
            self._emit(status_callback, f"Vast image build: {bounded}")
        return image_reference, source_fingerprint, tuple(recent_output)

    @staticmethod
    def _safe_status_line(line: str) -> str:
        """Bound one progress line and remove terminal codes and exact digests."""
        without_ansi = _ANSI_ESCAPE_PATTERN.sub("", line)
        redacted = _DIGEST_IN_TEXT_PATTERN.sub("sha256:[redacted]", without_ansi)
        printable = "".join(
            character for character in redacted if character.isprintable()
        )
        return printable[:_MAXIMUM_STATUS_LINE_CHARACTERS]

    def _manual_build_message(self, cause: str) -> str:
        """Return an actionable error without leaking subprocess environment data."""
        recovery_command = shlex.join(VAST_IMAGE_BUILD_COMMAND)
        return (
            f"Automatic Vast worker image publication failed. {cause} Run `"
            + recovery_command
            + "` from "
            + str(self.repo_root)
            + ", set COMFY_MODAL_VAST_IMAGE to the printed digest, restart ComfyUI, "
            "and retry the workflow."
        )

    @staticmethod
    def _emit(
        status_callback: VastImageBuildStatusCallback | None,
        message: str,
    ) -> None:
        """Send one build update when workflow progress is available."""
        if status_callback is not None:
            status_callback(message)


__all__ = [
    "VAST_IMAGE_BUILD_COMMAND",
    "VastWorkerImageBuildError",
    "VastWorkerImageBuilder",
]
