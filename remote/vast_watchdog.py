"""In-instance Vast.ai idle watchdog using the restricted container API key."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_WATCHDOG_STATE_PATH = Path("/storage/comfy-vast-watchdog.json")
DEFAULT_VAST_API_BASE_URL = "https://console.vast.ai"
_MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class VastWatchdogSnapshot:
    """Describe the activity state controlling self-destruction."""

    instance_id: int
    owner_label: str
    idle_deadline_epoch: float
    active_invocations: int
    updated_at_epoch: float

    def __post_init__(self) -> None:
        """Validate safety-sensitive watchdog state."""
        if (
            isinstance(self.instance_id, bool)
            or not isinstance(self.instance_id, int)
            or self.instance_id <= 0
        ):
            raise ValueError("Watchdog instance_id must be a positive integer.")
        if not self.owner_label.strip():
            raise ValueError("Watchdog owner_label must not be empty.")
        if self.active_invocations < 0:
            raise ValueError("Watchdog active_invocations must not be negative.")
        for field_name, value in (
            ("idle_deadline_epoch", self.idle_deadline_epoch),
            ("updated_at_epoch", self.updated_at_epoch),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Watchdog {field_name} must be finite and non-negative.")

    def to_json_bytes(self) -> bytes:
        """Return canonical JSON bytes suitable for an atomic remote write."""
        return (
            json.dumps(asdict(self), separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "VastWatchdogSnapshot":
        """Build a validated snapshot from decoded JSON."""
        return cls(
            instance_id=int(payload["instance_id"]),
            owner_label=str(payload["owner_label"]),
            idle_deadline_epoch=float(payload["idle_deadline_epoch"]),
            active_invocations=int(payload.get("active_invocations", 0)),
            updated_at_epoch=float(payload["updated_at_epoch"]),
        )


class VastDestroyer(Protocol):
    """Destroy the current Vast instance through a restricted API key."""

    async def destroy(self, instance_id: int) -> None:
        """Permanently destroy one instance."""


class VastRestrictedInstanceClient:
    """Use Vast's injected per-instance key for self-destruction only."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_VAST_API_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Configure the restricted credential and safe endpoint."""
        normalized_key = api_key.strip()
        normalized_url = base_url.rstrip("/")
        if not normalized_key:
            raise ValueError("CONTAINER_API_KEY is required for Vast self-destruction.")
        if not normalized_url.startswith(
            ("https://", "http://127.0.0.1:", "http://localhost:")
        ):
            raise ValueError("Vast watchdog API URL must use HTTPS or loopback HTTP.")
        if timeout_seconds <= 0:
            raise ValueError("Vast watchdog timeout must be positive.")
        self._api_key = normalized_key
        self._base_url = normalized_url
        self._timeout_seconds = timeout_seconds

    async def destroy(self, instance_id: int) -> None:
        """Permanently destroy the current exact instance."""
        if instance_id <= 0:
            raise ValueError("Vast watchdog instance identity must be positive.")
        timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.delete(
                f"{self._base_url}/api/v0/instances/{instance_id}/",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                allow_redirects=False,
            ) as response:
                body = await _read_bounded_body(response)
                if 200 <= response.status < 300:
                    return
                raise RuntimeError(
                    f"Vast self-destruction returned HTTP {response.status}: "
                    f"{_safe_response_message(body)}"
                )


def read_watchdog_snapshot(path: Path) -> VastWatchdogSnapshot:
    """Read and validate one controller-authored watchdog state file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Vast watchdog state {path} is unreadable.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Vast watchdog state must be a JSON object.")
    return VastWatchdogSnapshot.from_mapping(payload)


def watchdog_should_destroy(
    snapshot: VastWatchdogSnapshot,
    *,
    expected_instance_id: int,
    now_epoch: float,
) -> bool:
    """Return whether verified idle state has reached its deadline."""
    if snapshot.instance_id != expected_instance_id:
        raise ValueError(
            "Vast watchdog state instance identity does not match CONTAINER_ID."
        )
    if snapshot.active_invocations:
        return False
    return snapshot.idle_deadline_epoch <= now_epoch


async def run_watchdog_once(
    *,
    state_path: Path,
    expected_instance_id: int,
    destroyer: VastDestroyer,
    now_epoch: float | None = None,
) -> bool:
    """Evaluate one snapshot and destroy the instance when safely expired."""
    snapshot = read_watchdog_snapshot(state_path)
    if not watchdog_should_destroy(
        snapshot,
        expected_instance_id=expected_instance_id,
        now_epoch=time.time() if now_epoch is None else now_epoch,
    ):
        return False
    logger.warning(
        "Vast lease idle deadline elapsed; destroying instance=%d owner=%s deadline=%.3f.",
        expected_instance_id,
        snapshot.owner_label,
        snapshot.idle_deadline_epoch,
    )
    await destroyer.destroy(expected_instance_id)
    return True


async def watch_until_destroyed(
    *,
    state_path: Path,
    expected_instance_id: int,
    destroyer: VastDestroyer,
    poll_interval_seconds: float,
    missing_state_grace_seconds: float,
) -> None:
    """Poll controller state until idle destruction succeeds."""
    if poll_interval_seconds <= 0 or missing_state_grace_seconds < 0:
        raise ValueError("Watchdog polling must be positive and grace non-negative.")
    started_at = time.monotonic()
    while True:
        try:
            destroyed = await run_watchdog_once(
                state_path=state_path,
                expected_instance_id=expected_instance_id,
                destroyer=destroyer,
            )
        except FileNotFoundError:
            if time.monotonic() - started_at >= missing_state_grace_seconds:
                logger.error(
                    "Vast watchdog state is still absent after %.0fs; preserving the "
                    "instance because no verified idle deadline exists.",
                    missing_state_grace_seconds,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error(
                "Vast watchdog could not verify or apply idle state; preserving instance: %s",
                exc,
            )
        else:
            if destroyed:
                return
        await asyncio.sleep(poll_interval_seconds)


async def _read_bounded_body(response: aiohttp.ClientResponse) -> bytes:
    """Read a small diagnostics body from the destroy response."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Vast watchdog response exceeded 1 MiB.")
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_response_message(body: bytes) -> str:
    """Return a bounded response message without reflecting credentials."""
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "non-JSON response"
    if isinstance(payload, Mapping):
        for field_name in ("msg", "message", "error"):
            value = payload.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip().replace("\n", " ")[:500]
    return "request rejected"


def _parser() -> argparse.ArgumentParser:
    """Return the standalone watchdog argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_WATCHDOG_STATE_PATH)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--missing-state-grace", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the in-instance watchdog from Vast-provided environment state."""
    arguments = _parser().parse_args(argv)
    instance_id_value = os.getenv("CONTAINER_ID")
    api_key = os.getenv("CONTAINER_API_KEY")
    if not instance_id_value or not api_key:
        raise RuntimeError(
            "Vast watchdog requires CONTAINER_ID and CONTAINER_API_KEY."
        )
    try:
        instance_id = int(instance_id_value)
    except ValueError as exc:
        raise RuntimeError("CONTAINER_ID must be an integer.") from exc
    base_url = os.getenv("VAST_API_BASE_URL", DEFAULT_VAST_API_BASE_URL)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        watch_until_destroyed(
            state_path=arguments.state,
            expected_instance_id=instance_id,
            destroyer=VastRestrictedInstanceClient(api_key, base_url=base_url),
            poll_interval_seconds=arguments.poll_interval,
            missing_state_grace_seconds=arguments.missing_state_grace,
        )
    )


if __name__ == "__main__":  # pragma: no cover - remote process entrypoint.
    main()


__all__ = [
    "DEFAULT_VAST_API_BASE_URL",
    "DEFAULT_WATCHDOG_STATE_PATH",
    "VastRestrictedInstanceClient",
    "VastWatchdogSnapshot",
    "read_watchdog_snapshot",
    "run_watchdog_once",
    "watch_until_destroyed",
    "watchdog_should_destroy",
]
