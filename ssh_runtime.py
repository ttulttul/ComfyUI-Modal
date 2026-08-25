"""OCI image construction and warm-worker lifecycle for SSH Docker hosts."""

from __future__ import annotations

import io
import json
import logging
import shlex
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

if __package__:
    from .runtime_environment import (
        COMFYUI_RUNTIME_SOURCE_DIRECTORIES,
        COMFYUI_RUNTIME_SOURCE_FILES,
        REMOTE_LLAMA_CPP_SERVER_IMAGE,
        REMOTE_PYTHON_VERSION,
        RemoteRuntimeIdentity,
        build_remote_runtime_identity,
        custom_node_runtime_packages,
        remote_accelerator_packages,
        remote_accelerator_validation_command,
        remote_apt_packages,
        remote_compiler_validation_command,
        remote_runtime_packages,
        remote_runtime_validation_command,
        select_remote_torch_build,
    )
    from .settings import ModalSyncSettings
    from .ssh_docker import SshDockerController, SshDockerError
else:  # pragma: no cover - remote entrypoint compatibility.
    from runtime_environment import (
        COMFYUI_RUNTIME_SOURCE_DIRECTORIES,
        COMFYUI_RUNTIME_SOURCE_FILES,
        REMOTE_LLAMA_CPP_SERVER_IMAGE,
        REMOTE_PYTHON_VERSION,
        RemoteRuntimeIdentity,
        build_remote_runtime_identity,
        custom_node_runtime_packages,
        remote_accelerator_packages,
        remote_accelerator_validation_command,
        remote_apt_packages,
        remote_compiler_validation_command,
        remote_runtime_packages,
        remote_runtime_validation_command,
        select_remote_torch_build,
    )
    from settings import ModalSyncSettings
    from ssh_docker import SshDockerController, SshDockerError

logger = logging.getLogger(__name__)

_REMOTE_REPO_ROOT = Path("/opt/comfy-remote/repo")
_REMOTE_COMFYUI_ROOT = Path("/opt/comfy-remote/ComfyUI")
_REMOTE_STORAGE_ROOT = Path("/storage")
_RUNTIME_LABEL = "comfy.remote.runtime-fingerprint"
_ENVIRONMENT_LABEL = "comfy.remote.environment-id"
_WORKER_LABEL = "comfy.remote.worker-index"
_LARGE_DOWNLOAD_RESUME_RETRIES = 20
_LARGE_DOWNLOAD_TIMEOUT_SECONDS = 120
_WORKER_LIFECYCLE_LOCKS_GUARD = threading.Lock()
_WORKER_LIFECYCLE_LOCKS: dict[tuple[str, int], threading.Lock] = {}


def _emit_runtime_status(
    status_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    """Publish one SSH runtime lifecycle message when a callback is available."""
    if status_callback is not None:
        status_callback(message)


@dataclass(frozen=True)
class SshRuntimeSpec:
    """Describe one immutable SSH worker image and warm container."""

    identity: RemoteRuntimeIdentity
    image_tag: str
    container_name: str
    storage_volume_name: str
    worker_index: int


@dataclass
class SshRuntimeManager:
    """Build fingerprinted images and reconcile warm worker containers."""

    controller: SshDockerController
    repo_root: Path
    settings: ModalSyncSettings

    def runtime_spec(self, worker_index: int = 0) -> SshRuntimeSpec:
        """Return the deterministic image and worker identity for one host."""
        if worker_index < 0:
            raise ValueError("worker_index must not be negative.")
        identity = build_remote_runtime_identity(
            repo_root=self.repo_root,
            comfyui_root=self.settings.comfyui_root,
            custom_nodes_dir=self.settings.custom_nodes_dir,
            settings=self.settings,
        )
        fingerprint_short = identity.fingerprint[:16]
        environment_id = self.controller.host.environment_id
        return SshRuntimeSpec(
            identity=identity,
            image_tag=f"comfy-remote:{fingerprint_short}",
            container_name=(
                f"comfy-remote-{environment_id}-{fingerprint_short}-w{worker_index}"
            ),
            storage_volume_name=self.controller.host.resolved_storage_volume_name,
            worker_index=worker_index,
        )

    def ensure_worker(
        self,
        worker_index: int = 0,
        status_callback: Callable[[str], None] | None = None,
    ) -> SshRuntimeSpec:
        """Ensure a compatible image and running warm worker exist."""
        lifecycle_lock = _worker_lifecycle_lock(
            self.controller.host.environment_id,
            worker_index,
        )
        with lifecycle_lock:
            spec = self.runtime_spec(worker_index)
            environment_id = self.controller.host.environment_id
            _emit_runtime_status(
                status_callback,
                f"Checking SSH runtime environment={environment_id}",
            )
            self.controller.ensure_volume(spec.storage_volume_name)
            if not self._image_is_current(spec):
                _emit_runtime_status(
                    status_callback,
                    f"Building SSH runtime environment={environment_id} "
                    f"image={spec.image_tag}",
                )
                self._build_image(spec)
            self._remove_stale_worker_containers(spec)
            if not self._container_is_current_and_running(spec):
                _emit_runtime_status(
                    status_callback,
                    f"Starting SSH worker environment={environment_id}",
                )
                self._replace_worker_container(spec)
            _emit_runtime_status(
                status_callback,
                f"Waiting for SSH worker environment={environment_id}",
            )
            self._wait_until_ready(spec)
            _emit_runtime_status(status_callback, "Ready for remote execution")
            return spec

    def stop_worker(self, worker_index: int = 0) -> bool:
        """Stop and remove one exact managed worker container when present."""
        spec = self.runtime_spec(worker_index)
        inspected = self.controller.docker(
            ("container", "inspect", spec.container_name),
            check=False,
        )
        if inspected.returncode != 0:
            return False
        self.controller.docker(("rm", "-f", spec.container_name))
        return True

    def stop_all_workers(self) -> tuple[str, ...]:
        """Remove every container owned by this configured environment."""
        removed_names: list[str] = []
        for worker in self.controller.list_managed_workers():
            self.controller.remove_managed_worker(worker.container_name)
            removed_names.append(worker.container_name)
        return tuple(removed_names)

    def _remove_stale_worker_containers(self, spec: SshRuntimeSpec) -> None:
        """Remove superseded images' container for the same logical worker slot."""
        for worker in self.controller.list_managed_workers():
            if worker.worker_index != spec.worker_index:
                continue
            if worker.container_name == spec.container_name:
                continue
            logger.info(
                "Removing stale SSH worker environment=%s worker_index=%d "
                "container=%s.",
                self.controller.host.environment_id,
                spec.worker_index,
                worker.container_name,
            )
            self.controller.remove_managed_worker(worker.container_name)

    def _image_is_current(self, spec: SshRuntimeSpec) -> bool:
        """Return whether the expected fingerprinted image is available."""
        result = self.controller.docker(
            (
                "image",
                "inspect",
                "--format",
                f'{{{{index .Config.Labels "{_RUNTIME_LABEL}"}}}}',
                spec.image_tag,
            ),
            check=False,
        )
        return (
            result.returncode == 0
            and result.stdout_text.strip() == spec.identity.fingerprint
        )

    def _container_is_current_and_running(self, spec: SshRuntimeSpec) -> bool:
        """Return whether one managed worker is running the expected image."""
        result = self.controller.docker(
            ("container", "inspect", spec.container_name),
            check=False,
        )
        if result.returncode != 0:
            return False
        try:
            payload = json.loads(result.stdout_text)
        except json.JSONDecodeError:
            return False
        if (
            not isinstance(payload, list)
            or not payload
            or not isinstance(payload[0], dict)
        ):
            return False
        container = payload[0]
        state = container.get("State")
        config = container.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        return bool(isinstance(state, dict) and state.get("Running")) and bool(
            isinstance(labels, dict)
            and labels.get(_RUNTIME_LABEL) == spec.identity.fingerprint
            and labels.get(_ENVIRONMENT_LABEL) == self.controller.host.environment_id
            and labels.get(_WORKER_LABEL) == str(spec.worker_index)
        )

    def _build_image(self, spec: SshRuntimeSpec) -> None:
        """Stream a deterministic Docker build context to the remote daemon."""
        context = self._build_context(spec)
        logger.info(
            "Building SSH runtime environment=%s image=%s fingerprint=%s "
            "context_bytes=%d.",
            self.controller.host.environment_id,
            spec.image_tag,
            spec.identity.fingerprint,
            len(context),
        )
        self.controller.docker(
            (
                "build",
                "--pull",
                "--load",
                "--label",
                f"{_RUNTIME_LABEL}={spec.identity.fingerprint}",
                "-t",
                spec.image_tag,
                "-",
            ),
            input_payload=context,
            timeout_seconds=max(3600.0, self.settings.startup_timeout_seconds),
        )

    def _build_context(self, spec: SshRuntimeSpec) -> bytes:
        """Return a tar build context containing only worker runtime sources."""
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            dockerfile = self._dockerfile(spec).encode("utf-8")
            dockerfile_info = tarfile.TarInfo("Dockerfile")
            dockerfile_info.size = len(dockerfile)
            dockerfile_info.mtime = 0
            dockerfile_info.mode = 0o644
            archive.addfile(dockerfile_info, io.BytesIO(dockerfile))
            for source_path, archive_path in self._runtime_context_files():
                info = archive.gettarinfo(str(source_path), arcname=archive_path)
                info.mtime = 0
                with source_path.open("rb") as source_file:
                    archive.addfile(info, source_file)
        return output.getvalue()

    def _runtime_context_files(self) -> Iterable[tuple[Path, str]]:
        """Yield deterministic repository and ComfyUI files for the image."""
        resolved_repo_root = self.repo_root.resolve()
        ignored_directory_names = {
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "tests",
        }
        repo_candidates = sorted(
            path
            for path in resolved_repo_root.rglob("*")
            if path.is_file()
            and not ignored_directory_names.intersection(
                path.relative_to(resolved_repo_root).parts
            )
            and (path.suffix == ".py" or path.name in {"llm_profiles.json"})
        )
        for source_path in repo_candidates:
            relative_path = source_path.relative_to(resolved_repo_root).as_posix()
            yield source_path, f"repo/{relative_path}"

        comfyui_root = self.settings.comfyui_root
        if comfyui_root is None or not comfyui_root.is_dir():
            raise SshDockerError(
                "SSH execution requires a local ComfyUI checkout to build the "
                "worker image."
            )
        resolved_comfyui_root = comfyui_root.resolve()
        for source_path in sorted(resolved_comfyui_root.rglob("*")):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(resolved_comfyui_root)
            if ignored_directory_names.intersection(relative_path.parts):
                continue
            top_level_name = relative_path.parts[0]
            if top_level_name not in COMFYUI_RUNTIME_SOURCE_DIRECTORIES and (
                source_path.suffix != ".py"
                and relative_path.as_posix() not in COMFYUI_RUNTIME_SOURCE_FILES
            ):
                continue
            yield source_path, f"comfyui/{relative_path.as_posix()}"

    def _dockerfile(self, spec: SshRuntimeSpec) -> str:
        """Return the immutable Dockerfile for one runtime identity."""
        torch_build = select_remote_torch_build(self.settings.modal_gpu)
        lines = [
            f"FROM python:{REMOTE_PYTHON_VERSION}-slim-bookworm AS python-runtime",
            f"FROM {REMOTE_LLAMA_CPP_SERVER_IMAGE}",
            "COPY --from=python-runtime /usr/local /usr/local",
            "ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 "
            "CC=/usr/bin/gcc CXX=/usr/bin/g++",
            _docker_run(
                "apt-get",
                "update",
                "&&",
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                *remote_apt_packages(),
                "&&",
                "rm",
                "-rf",
                "/var/lib/apt/lists/*",
            ),
            f"RUN {remote_compiler_validation_command()}",
            _pip_install(remote_runtime_packages()),
        ]
        for layer in torch_build.install_layers:
            layer_arguments = [
                "python",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--index-url",
                layer.index_url,
            ]
            if layer.extra_options:
                layer_arguments.extend(shlex.split(layer.extra_options))
            layer_arguments.extend(layer.packages)
            lines.append(f"RUN {shlex.join(layer_arguments)}")
        lines.extend(
            [
                f"RUN {torch_build.validation_command()}",
                _pip_install(
                    remote_accelerator_packages(self.settings.modal_gpu),
                    resume_retries=_LARGE_DOWNLOAD_RESUME_RETRIES,
                    timeout_seconds=_LARGE_DOWNLOAD_TIMEOUT_SECONDS,
                ),
                f"RUN {remote_accelerator_validation_command(self.settings.modal_gpu)}",
            ]
        )
        custom_packages = custom_node_runtime_packages(self.settings.custom_nodes_dir)
        if custom_packages:
            lines.append(_pip_install(custom_packages))
            lines.append(_pip_install(remote_runtime_packages()))
        lines.append(f"RUN {remote_runtime_validation_command()}")
        lines.extend(
            [
                "COPY repo /opt/comfy-remote/repo",
                "COPY comfyui /opt/comfy-remote/ComfyUI",
                (
                    f"ENV PYTHONPATH={_REMOTE_REPO_ROOT}:{_REMOTE_COMFYUI_ROOT} "
                    "LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH} "
                    f"COMFYUI_ROOT={_REMOTE_COMFYUI_ROOT} "
                    f"COMFY_MODAL_COMFYUI_ROOT={_REMOTE_COMFYUI_ROOT} "
                    f"COMFY_MODAL_LOCAL_STORAGE_ROOT={_REMOTE_STORAGE_ROOT} "
                    f"COMFY_MODAL_REMOTE_STORAGE_ROOT={_REMOTE_STORAGE_ROOT} "
                    "COMFY_MODAL_EXECUTION_MODE=local COMFY_MODAL_REMOTE_WORKER=1 "
                    "COMFY_MODAL_LLM_EXECUTION_TARGET=ssh_docker "
                    f"COMFY_MODAL_RUNTIME_FINGERPRINT={spec.identity.fingerprint}"
                ),
                "WORKDIR /opt/comfy-remote/repo",
                "HEALTHCHECK NONE",
                'ENTRYPOINT ["python","-m","remote.ssh_worker","serve"]',
                "",
            ]
        )
        return "\n".join(lines)

    def _replace_worker_container(self, spec: SshRuntimeSpec) -> None:
        """Replace one exact managed worker with a compatible warm container."""
        existing = self.controller.docker(
            ("container", "inspect", spec.container_name),
            check=False,
        )
        if existing.returncode == 0:
            self.controller.remove_managed_worker(spec.container_name)
        gpu_arguments = self._gpu_arguments(spec.worker_index)
        environment_file_arguments = self._environment_file_arguments()
        launched = self.controller.docker(
            (
                "run",
                "-d",
                "--name",
                spec.container_name,
                "--restart",
                "unless-stopped",
                "--init",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--shm-size",
                "8g",
                "--label",
                f"{_RUNTIME_LABEL}={spec.identity.fingerprint}",
                "--label",
                f"{_ENVIRONMENT_LABEL}={self.controller.host.environment_id}",
                "--label",
                f"{_WORKER_LABEL}={spec.worker_index}",
                "-v",
                f"{spec.storage_volume_name}:{_REMOTE_STORAGE_ROOT}",
                *environment_file_arguments,
                *gpu_arguments,
                spec.image_tag,
            ),
            timeout_seconds=120.0,
            check=False,
        )
        if launched.returncode == 0:
            return
        if self._concurrent_worker_became_ready(spec):
            logger.info(
                "Adopting concurrently launched SSH worker environment=%s "
                "worker_index=%d container=%s.",
                self.controller.host.environment_id,
                spec.worker_index,
                spec.container_name,
            )
            return
        error_text = launched.stderr_text.strip() or launched.stdout_text.strip()
        failure_detail = error_text or (
            f"Docker exited with status {launched.returncode}"
        )
        raise SshDockerError(
            f"Could not start SSH worker {spec.container_name!r} on "
            f"{self.controller.host.ssh_target!r}: "
            f"{failure_detail}"
        )

    def _concurrent_worker_became_ready(self, spec: SshRuntimeSpec) -> bool:
        """Return whether another launcher won a short race for this worker name."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._container_is_current_and_running(spec):
                return True
            time.sleep(0.1)
        return False

    def _gpu_arguments(self, worker_index: int) -> tuple[str, ...]:
        """Return Docker GPU-selection arguments for this worker."""
        capabilities = self.controller.host.capabilities
        if capabilities is None:
            raise SshDockerError(
                "SSH worker launch requires probed GPU capabilities."
            )
        if not capabilities.gpus:
            raise SshDockerError(
                "SSH worker launch requires at least one discovered NVIDIA GPU."
            )
        if not capabilities.nvidia_container_runtime:
            raise SshDockerError(
                "SSH worker launch requires the NVIDIA container runtime."
            )
        gpu = capabilities.gpus[worker_index % len(capabilities.gpus)]
        return ("--gpus", f"device={gpu.uuid}")

    def _environment_file_arguments(self) -> tuple[str, ...]:
        """Return an optional administrator-managed remote Docker env-file argument."""
        environment_file = self.controller.host.docker_env_file
        if environment_file is None:
            return ()
        return ("--env-file", environment_file)

    def _wait_until_ready(self, spec: SshRuntimeSpec) -> None:
        """Wait for the worker socket and verify its runtime fingerprint."""
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        last_error = "worker has not responded"
        while time.monotonic() < deadline:
            result = self.controller.docker(
                (
                    "exec",
                    spec.container_name,
                    "python",
                    "-m",
                    "remote.ssh_worker",
                    "runtime-info",
                ),
                check=False,
            )
            if result.returncode == 0:
                try:
                    runtime = json.loads(result.stdout_text)
                except json.JSONDecodeError:
                    last_error = "worker returned invalid runtime metadata"
                else:
                    if runtime.get("runtime_fingerprint") == spec.identity.fingerprint:
                        return
                    last_error = (
                        "worker fingerprint does not match the requested runtime"
                    )
            else:
                last_error = result.stderr_text.strip() or "worker is still starting"
            time.sleep(0.5)
        raise SshDockerError(
            f"SSH worker {spec.container_name!r} did not become ready: {last_error}."
        )


def export_worker_image_context(
    *,
    repo_root: Path,
    settings: ModalSyncSettings,
    identity: RemoteRuntimeIdentity,
) -> bytes:
    """Return the shared deterministic worker build context without a remote host."""
    manager = SshRuntimeManager.__new__(SshRuntimeManager)
    manager.repo_root = repo_root
    manager.settings = settings
    spec = SshRuntimeSpec(
        identity=identity,
        image_tag=f"comfy-remote:{identity.fingerprint[:16]}",
        container_name="context-only",
        storage_volume_name="context-only",
        worker_index=0,
    )
    return manager._build_context(spec)


def _worker_lifecycle_lock(
    environment_id: str,
    worker_index: int,
) -> threading.Lock:
    """Return the process-wide lifecycle lock for one environment worker slot."""
    key = (environment_id, worker_index)
    with _WORKER_LIFECYCLE_LOCKS_GUARD:
        lock = _WORKER_LIFECYCLE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WORKER_LIFECYCLE_LOCKS[key] = lock
        return lock


def _pip_install(
    packages: Iterable[str],
    *,
    resume_retries: int | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Return one deterministic Docker RUN instruction for pip packages."""
    arguments = [
        "python",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
    ]
    if resume_retries is not None:
        if resume_retries < 0:
            raise ValueError("resume_retries must not be negative.")
        arguments.extend(("--resume-retries", str(resume_retries)))
    if timeout_seconds is not None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        arguments.extend(("--timeout", str(timeout_seconds)))
    arguments.extend(packages)
    return f"RUN {shlex.join(arguments)}"


def _docker_run(*arguments: str) -> str:
    """Return one Docker RUN line with explicit shell operators preserved."""
    rendered = " ".join(
        argument if argument in {"&&", "||", ";"} else shlex.quote(argument)
        for argument in arguments
    )
    return f"RUN {rendered}"
