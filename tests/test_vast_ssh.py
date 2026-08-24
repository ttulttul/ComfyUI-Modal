"""Tests for direct Vast SSH command and storage adapters."""

from __future__ import annotations

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
        if arguments[:3] == ("python", "-m", "remote.huggingface_materializer"):
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
        if call["argv"][:3] == ("python", "-m", "remote.huggingface_materializer")
    )
    assert "hf_private_token" not in " ".join(materialize_call["argv"])
    assert b"hf_private_token" in materialize_call["input"]
    assert backend.exists("assets/model.safetensors") is True


@pytest.mark.parametrize("path", ["../secret", "a/../../secret", ".", ""])
def test_volume_backend_rejects_path_traversal(
    path: str,
    vast_ssh_module: Any,
) -> None:
    """Remote uploads must remain below the dedicated storage root."""
    backend = vast_ssh_module.VastSshVolumeBackend(runner=FakeRunner(vast_ssh_module))
    with pytest.raises(ValueError, match="Unsafe Vast storage path"):
        backend.put_bytes(b"secret", path)
