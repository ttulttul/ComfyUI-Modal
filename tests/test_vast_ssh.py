"""Tests for direct Vast SSH command and storage adapters."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


class FakeRunner:
    """Record direct remote commands and emulate file existence."""

    def __init__(self, module: Any) -> None:
        """Initialize empty calls and the result type under test."""
        self.module = module
        self.calls: list[dict[str, Any]] = []
        self.existing: set[str] = set()

    def run(
        self,
        remote_argv: Any,
        *,
        input_payload: bytes | None = None,
        input_file: Path | None = None,
        timeout_seconds: float | None = None,
        check: bool = True,
        cancellation_check: Any | None = None,
    ) -> Any:
        """Record one command and infer test/atomic-write behavior."""
        arguments = tuple(remote_argv)
        self.calls.append(
            {
                "argv": arguments,
                "input": input_payload,
                "input_file": input_file,
                "timeout": timeout_seconds,
                "check": check,
                "cancellation_check": cancellation_check,
            }
        )
        if arguments[:2] == ("test", "-f"):
            return self.module.VastSshCommandResult(
                stdout=b"",
                stderr=b"",
                returncode=0 if arguments[2] in self.existing else 1,
            )
        if arguments[:2] == ("python", "-c"):
            self.existing.add(arguments[-1])
        if arguments[:2] == ("python", "/storage/runtime-tools") or (
            len(arguments) == 2 and arguments[1].endswith(".pyz")
        ):
            request = __import__("json").loads(input_payload or b"{}")
            self.existing.add(f"/storage/{request['remote_path']}")
        return self.module.VastSshCommandResult(stdout=b"", stderr=b"", returncode=0)


def test_command_uses_dedicated_accept_new_trust_store(
    tmp_path: Path,
    vast_ssh_module: Any,
) -> None:
    """Dynamic Vast hosts should not weaken or pollute global known_hosts state."""
    connection = vast_ssh_module.VastSshConnection(
        host="ssh.example.invalid",
        port=22345,
        user="root",
        known_hosts_path=(tmp_path / "vast-known-hosts").resolve(),
        identity_file=(tmp_path / "identity").resolve(),
    )
    runner = vast_ssh_module.VastSshRunner(connection)

    command = runner.command(("python", "-c", "print('safe')"))

    assert command[:3] == ["ssh", "-p", "22345"]
    assert "StrictHostKeyChecking=accept-new" in command
    assert f"UserKnownHostsFile={connection.known_hosts_path}" in command
    assert "ClearAllForwardings=yes" in command
    assert command[-2] == "root@ssh.example.invalid"
    assert command[-1] == "python -c 'print('\"'\"'safe'\"'\"')'"
    assert connection.known_hosts_path.stat().st_mode & 0o777 == 0o600


def test_connection_rejects_option_injection(
    tmp_path: Path,
    vast_ssh_module: Any,
) -> None:
    """A marketplace-provided hostname cannot become an OpenSSH option."""
    with pytest.raises(ValueError, match="option prefix"):
        vast_ssh_module.VastSshConnection(
            host="-oProxyCommand=bad",
            port=22,
            known_hosts_path=(tmp_path / "known-hosts").resolve(),
        )


def test_runner_passes_file_handle_to_openssh_stdin(
    tmp_path: Path,
    vast_ssh_module: Any,
    monkeypatch: Any,
) -> None:
    """The concrete runner must use subprocess stdin rather than a bytes input buffer."""
    connection = vast_ssh_module.VastSshConnection(
        host="ssh.example.invalid",
        port=22,
        known_hosts_path=(tmp_path / "known-hosts").resolve(),
    )
    runner = vast_ssh_module.VastSshRunner(connection)
    asset_path = tmp_path / "large-model.safetensors"
    asset_path.write_bytes(b"chunked-input")
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Require a live file-backed stdin handle inside subprocess.run."""
        observed["command"] = command
        assert "input" not in kwargs
        input_handle = kwargs["stdin"]
        observed["payload"] = input_handle.read()
        return vast_ssh_module.subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=b"",
            stderr=b"",
        )

    monkeypatch.setattr(vast_ssh_module.subprocess, "run", fake_run)

    runner.run(("cat",), input_file=asset_path)

    assert observed["payload"] == b"chunked-input"
    assert observed["command"][-1] == "cat"


def test_runner_retries_transient_disconnect_and_reopens_input_file(
    tmp_path: Path,
    vast_ssh_module: Any,
    monkeypatch: Any,
) -> None:
    """Killed upload connections should retry from the beginning with jittered backoff."""
    connection = vast_ssh_module.VastSshConnection(
        host="ssh.example.invalid",
        port=22,
        known_hosts_path=(tmp_path / "known-hosts").resolve(),
    )
    delays: list[float] = []
    runner = vast_ssh_module.VastSshRunner(
        connection,
        sleep=delays.append,
        random_unit=lambda: 0.5,
    )
    asset_path = tmp_path / "archive.zip"
    asset_path.write_bytes(b"complete-archive")
    responses = iter(
        (
            (255, b"Connection closed by 54.80.37.79 port 11714"),
            (255, b"kex_exchange_identification: read: Connection reset by peer"),
            (0, b""),
        )
    )
    uploaded_payloads: list[bytes] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Return two transport failures followed by one successful upload."""
        input_handle = kwargs["stdin"]
        uploaded_payloads.append(input_handle.read())
        returncode, stderr = next(responses)
        return vast_ssh_module.subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=b"",
            stderr=stderr,
        )

    monkeypatch.setattr(vast_ssh_module.subprocess, "run", fake_run)

    result = runner.run(("python", "upload.py"), input_file=asset_path)

    assert result.returncode == 0
    assert uploaded_payloads == [b"complete-archive"] * 3
    assert delays == [0.5, 1.0]


def test_runner_does_not_retry_ssh_authentication_failure(
    tmp_path: Path,
    vast_ssh_module: Any,
    monkeypatch: Any,
) -> None:
    """Credential rejection should remain immediate and actionable."""
    connection = vast_ssh_module.VastSshConnection(
        host="ssh.example.invalid",
        port=22,
        known_hosts_path=(tmp_path / "known-hosts").resolve(),
    )
    delays: list[float] = []
    runner = vast_ssh_module.VastSshRunner(connection, sleep=delays.append)
    call_count = 0

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Return OpenSSH's public-key rejection."""
        nonlocal call_count
        del kwargs
        call_count += 1
        return vast_ssh_module.subprocess.CompletedProcess(
            args=command,
            returncode=255,
            stdout=b"",
            stderr=b"root@host: Permission denied (publickey).",
        )

    monkeypatch.setattr(vast_ssh_module.subprocess, "run", fake_run)

    with pytest.raises(vast_ssh_module.VastSshError, match="Permission denied"):
        runner.run(("true",))

    assert call_count == 1
    assert delays == []


def test_runner_bounds_repeated_transient_disconnects(
    tmp_path: Path,
    vast_ssh_module: Any,
    monkeypatch: Any,
) -> None:
    """Persistent connection loss should fail after the configured attempt budget."""
    connection = vast_ssh_module.VastSshConnection(
        host="ssh.example.invalid",
        port=22,
        known_hosts_path=(tmp_path / "known-hosts").resolve(),
    )
    delays: list[float] = []
    runner = vast_ssh_module.VastSshRunner(
        connection,
        retry_attempts=3,
        sleep=delays.append,
        random_unit=lambda: 0.5,
    )

    def fake_run(command: list[str], **kwargs: object) -> object:
        """Always report the killed connection observed against Vast."""
        del kwargs
        return vast_ssh_module.subprocess.CompletedProcess(
            args=command,
            returncode=255,
            stdout=b"",
            stderr=b"Connection closed by 54.80.37.79 port 11714",
        )

    monkeypatch.setattr(vast_ssh_module.subprocess, "run", fake_run)

    with pytest.raises(
        vast_ssh_module.VastSshError,
        match="after 3 transport attempts",
    ):
        runner.run(("true",), check=False)

    assert delays == [0.5, 1.0]


def test_volume_backend_writes_atomically_and_caches_existence(
    vast_ssh_module: Any,
) -> None:
    """Direct instance storage should satisfy the sync engine's volume protocol."""
    runner = FakeRunner(vast_ssh_module)
    backend = vast_ssh_module.VastSshVolumeBackend(runner=runner)

    assert backend.exists("models/aa/model.bin") is False
    backend.put_bytes(b"weights", "models/aa/model.bin")
    assert backend.exists("models/aa/model.bin") is True

    write_call = next(call for call in runner.calls if call["argv"][:2] == ("python", "-c"))
    assert write_call["input"] == b"weights"
    assert write_call["argv"][-1] == "/storage/models/aa/model.bin"
    assert write_call["argv"][-2] == str(len(b"weights"))
    assert "os.replace" in write_call["argv"][2]
    assert len([call for call in runner.calls if call["argv"][:2] == ("test", "-f")]) == 1


def test_volume_backend_streams_files_without_reading_them_into_memory(
    vast_ssh_module: Any,
    tmp_path: Path,
) -> None:
    """Large fallback uploads should hand a file stream to OpenSSH instead of bytes."""
    runner = FakeRunner(vast_ssh_module)
    backend = vast_ssh_module.VastSshVolumeBackend(runner=runner)
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"streamed-model")

    backend.put_file(asset_path, "assets/model.safetensors")

    write_call = next(call for call in runner.calls if call["argv"][:2] == ("python", "-c"))
    assert write_call["input"] is None
    assert write_call["input_file"] == asset_path.resolve()
    assert "copyfileobj" in write_call["argv"][2]
    assert backend.exists("assets/model.safetensors") is True


def test_atomic_writer_refuses_to_publish_truncated_upload(
    vast_ssh_module: Any,
    tmp_path: Path,
) -> None:
    """A killed SSH stream must not publish partial content at the final path."""
    target = tmp_path / "remote" / "archive.zip"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            vast_ssh_module._ATOMIC_STDIN_WRITER,
            "100",
            str(target),
        ],
        input=b"truncated",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert target.exists() is False
    assert list(target.parent.glob(".*.tmp")) == []


def test_huggingface_materialization_keeps_token_out_of_command_arguments(
    vast_ssh_module: Any,
    huggingface_assets_module: Any,
    tmp_path: Path,
) -> None:
    """Private Hub credentials should travel only inside SSH standard input."""
    runner = FakeRunner(vast_ssh_module)
    backend = vast_ssh_module.VastSshVolumeBackend(runner=runner)
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"registered-model")
    source = huggingface_assets_module.HuggingFaceAssetSource(
        repo_id="owner/private-model",
        revision="a" * 40,
        filename="model.safetensors",
        sha256=huggingface_assets_module.sha256_file(asset_path),
        size_bytes=asset_path.stat().st_size,
    )

    assert backend.materialize_huggingface_file(
        source,
        "assets/model.safetensors",
        token="hf_private_token",
    ) is True

    materialize_call = next(
        call
        for call in runner.calls
        if len(call["argv"]) == 2
        and call["argv"][0] == "python"
        and call["argv"][-1].endswith(".pyz")
    )
    assert "hf_private_token" not in " ".join(materialize_call["argv"])
    assert b"hf_private_token" in materialize_call["input"]
    assert backend.exists("assets/model.safetensors") is True


def test_materializer_bundle_is_self_contained_and_executable(
    vast_ssh_module: Any,
    tmp_path: Path,
) -> None:
    """The uploaded zipapp should load without worker-image package sources."""
    bundle_path = tmp_path / "materializer.pyz"
    bundle = vast_ssh_module._huggingface_materializer_bundle()
    bundle_path.write_bytes(bundle)

    assert vast_ssh_module._huggingface_materializer_bundle() == bundle

    completed = subprocess.run(
        [sys.executable, str(bundle_path)],
        input=b"{}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode != 0
    assert b"No module named" not in completed.stderr
    assert b"source must be an object" in completed.stderr


def test_runner_terminates_active_ssh_process_when_cancelled(
    tmp_path: Path,
    vast_ssh_module: Any,
    monkeypatch: Any,
) -> None:
    """Prompt cancellation should terminate an in-flight OpenSSH subprocess."""
    runner = vast_ssh_module.VastSshRunner(
        vast_ssh_module.VastSshConnection(
            host="ssh.example.invalid",
            port=22,
            known_hosts_path=(tmp_path / "known-hosts").resolve(),
        )
    )

    class FakeProcess:
        """Represent a running SSH process that only exits after termination."""

        returncode = -15

        def __init__(self) -> None:
            """Track termination calls."""
            self.terminated = False

        def terminate(self) -> None:
            """Record graceful process termination."""
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            """Return the terminated status immediately."""
            del timeout
            return self.returncode

        def kill(self) -> None:
            """Fail if graceful termination unexpectedly needs escalation."""
            raise AssertionError("kill should not be needed")

        def communicate(self, **kwargs: Any) -> tuple[bytes, bytes]:
            """Remain active until the cancellation check fires."""
            del kwargs
            raise subprocess.TimeoutExpired("ssh", 0.25)

    process = FakeProcess()
    monkeypatch.setattr(
        vast_ssh_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    checks = iter((False, True))

    with pytest.raises(vast_ssh_module.VastSshCancelledError, match="cancelled"):
        runner.run(("python", "worker.py"), cancellation_check=lambda: next(checks))

    assert process.terminated is True


@pytest.mark.parametrize("path", ["../secret", "a/../../secret", ".", ""])
def test_volume_backend_rejects_path_traversal(
    path: str,
    vast_ssh_module: Any,
) -> None:
    """Remote uploads must remain below the dedicated storage root."""
    backend = vast_ssh_module.VastSshVolumeBackend(runner=FakeRunner(vast_ssh_module))
    with pytest.raises(ValueError, match="Unsafe Vast storage path"):
        backend.put_bytes(b"secret", path)
