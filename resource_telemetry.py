"""Backend-neutral CPU and GPU memory telemetry for remote execution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import resource
import sys
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class ResourceMemorySnapshot:
    """Capture one process-RAM and visible-GPU memory observation."""

    cpu_memory_bytes: int
    cpu_memory_total_bytes: int
    gpu_memory_bytes: int
    gpu_memory_total_bytes: int


class RemoteResourceTelemetrySampler:
    """Sample remote memory on a daemon thread and publish cumulative peaks."""

    def __init__(
        self,
        callback: Callable[[dict[str, Any]], None],
        *,
        interval_seconds: float = _RESOURCE_SAMPLE_INTERVAL_SECONDS,
        sample_reader: Callable[[], ResourceMemorySnapshot] | None = None,
    ) -> None:
        """Initialize a sampler with an injectable reader for deterministic tests."""
        self._callback = callback
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._sample_reader = sample_reader or read_resource_memory_snapshot
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_sequence = 0
        self._cpu_memory_peak_bytes = 0
        self._gpu_memory_peak_bytes = 0
        self._sampling_error_logged = False

    def start(self) -> None:
        """Publish an initial sample and start periodic collection."""
        if self._thread is not None:
            return
        self._publish_sample(active=True)
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="remote-resource-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop periodic collection and publish one terminal peak sample."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 2.0))
        self._publish_sample(active=False)

    def _sample_until_stopped(self) -> None:
        """Publish observations at a bounded cadence until execution finishes."""
        while not self._stop_event.wait(self._interval_seconds):
            self._publish_sample(active=True)

    def _publish_sample(self, *, active: bool) -> None:
        """Read, peak, and publish one transport-safe telemetry event."""
        try:
            snapshot = self._sample_reader()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if not self._sampling_error_logged:
                logger.warning("Remote resource telemetry sampling failed: %s", exc)
                self._sampling_error_logged = True
            return
        self._cpu_memory_peak_bytes = max(
            self._cpu_memory_peak_bytes,
            snapshot.cpu_memory_bytes,
        )
        self._gpu_memory_peak_bytes = max(
            self._gpu_memory_peak_bytes,
            snapshot.gpu_memory_bytes,
        )
        self._sample_sequence += 1
        try:
            self._callback(
                {
                    "event_type": "resource_telemetry",
                    "active": active,
                    "sample_sequence": self._sample_sequence,
                    "sampled_at": time.time(),
                    "cpu_memory_bytes": snapshot.cpu_memory_bytes,
                    "cpu_memory_peak_bytes": self._cpu_memory_peak_bytes,
                    "cpu_memory_total_bytes": snapshot.cpu_memory_total_bytes,
                    "gpu_memory_bytes": snapshot.gpu_memory_bytes,
                    "gpu_memory_peak_bytes": self._gpu_memory_peak_bytes,
                    "gpu_memory_total_bytes": snapshot.gpu_memory_total_bytes,
                }
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Remote resource telemetry callback was unavailable: %s", exc)


def read_resource_memory_snapshot() -> ResourceMemorySnapshot:
    """Read current process RSS and aggregate visible CUDA device memory."""
    cpu_memory_bytes, cpu_memory_total_bytes = _read_cpu_memory()
    gpu_memory_bytes, gpu_memory_total_bytes = _read_gpu_memory()
    return ResourceMemorySnapshot(
        cpu_memory_bytes=cpu_memory_bytes,
        cpu_memory_total_bytes=cpu_memory_total_bytes,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_memory_total_bytes=gpu_memory_total_bytes,
    )


def _read_cpu_memory() -> tuple[int, int]:
    """Return current process RSS and host/container RAM capacity."""
    try:
        import psutil
    except ModuleNotFoundError:
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024
        return max(0, peak_rss), 0
    try:
        root_process = psutil.Process()
        processes = [root_process, *root_process.children(recursive=True)]
        process_rss = 0
        for process in processes:
            try:
                process_rss += int(process.memory_info().rss)
            except (OSError, psutil.Error):
                continue
        total_memory = int(psutil.virtual_memory().total)
    except (OSError, psutil.Error) as exc:
        raise RuntimeError("Unable to read process memory usage.") from exc
    return max(0, process_rss), max(0, total_memory)


def _read_gpu_memory() -> tuple[int, int]:
    """Return used and total memory across every CUDA device visible to the worker."""
    try:
        import torch
    except ModuleNotFoundError:
        return 0, 0
    try:
        if not torch.cuda.is_available():
            return 0, 0
        used_bytes = 0
        total_bytes = 0
        for device_index in range(torch.cuda.device_count()):
            free_device_bytes, total_device_bytes = torch.cuda.mem_get_info(device_index)
            used_bytes += max(0, int(total_device_bytes) - int(free_device_bytes))
            total_bytes += max(0, int(total_device_bytes))
        return used_bytes, total_bytes
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError("Unable to read CUDA memory usage.") from exc
