"""Idle-gated, deduplicated Cloudflare R2 write-back coordination."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Protocol

if __package__:
    from .r2_cache import (
        R2CacheClient,
        R2CacheError,
        R2DownloadRequest,
        R2UploadPlan,
    )
    from .sync_protocols import (
        CancellationCheck,
        CancellableR2WriteBackBackend,
        R2MaterializingBackend,
        R2WorkerPreflightBackend,
        SyncStatusCallback,
        _ContentAddressedSyncResult,
        _ContentAddressedSyncSpec,
        _R2MaterializationOutcome,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from r2_cache import R2CacheClient, R2CacheError, R2DownloadRequest, R2UploadPlan
    from sync_protocols import (
        CancellationCheck,
        CancellableR2WriteBackBackend,
        R2MaterializingBackend,
        R2WorkerPreflightBackend,
        SyncStatusCallback,
        _ContentAddressedSyncResult,
        _ContentAddressedSyncSpec,
        _R2MaterializationOutcome,
    )

logger = logging.getLogger(__name__)


class R2WriteBackCancelled(RuntimeError):
    """Signal that an idle write-back yielded to foreground workflow activity."""


@dataclass(frozen=True)
class _R2WriteBackJob:
    """Describe one deduplicated best-effort cache population operation."""

    key: tuple[str, ...]
    callback: Callable[[CancellationCheck], None] = field(compare=False, repr=False)


class R2WriteBackCoordinator:
    """Run cache population only while no remote workflow is active."""

    def __init__(self, *, max_workers: int = 2, max_pending_jobs: int = 1024) -> None:
        """Start bounded daemon workers for opportunistic write-back jobs."""
        if max_workers <= 0 or max_pending_jobs <= 0:
            raise ValueError("R2 write-back coordinator bounds must be positive.")
        self._condition = threading.Condition()
        self._pending: OrderedDict[tuple[str, ...], _R2WriteBackJob] = OrderedDict()
        self._known_keys: set[tuple[str, ...]] = set()
        self._active_prompts: set[str] = set()
        self._active_cancellations: set[threading.Event] = set()
        self._active_jobs = 0
        self._max_pending_jobs = max_pending_jobs
        self._workers = tuple(
            threading.Thread(
                target=self._worker,
                name=f"comfy-r2-writeback-{worker_index + 1}",
                daemon=True,
            )
            for worker_index in range(max_workers)
        )
        for worker in self._workers:
            worker.start()

    def begin_prompt(self, prompt_id: str) -> None:
        """Reserve foreground priority and preempt active cache transfers."""
        normalized_prompt_id = str(prompt_id).strip()
        if not normalized_prompt_id:
            raise ValueError("R2 write-back prompt ID must not be empty.")
        with self._condition:
            self._active_prompts.add(normalized_prompt_id)
            for cancellation in self._active_cancellations:
                cancellation.set()
            self._condition.notify_all()

    def finish_prompt(self, prompt_id: str) -> None:
        """Release one prompt reservation and wake idle cache workers."""
        normalized_prompt_id = str(prompt_id).strip()
        if not normalized_prompt_id:
            return
        with self._condition:
            self._active_prompts.discard(normalized_prompt_id)
            self._condition.notify_all()

    def reset_prompt_reservations_for_tests(self) -> None:
        """Clear synthetic prompt reservations left by isolated queue tests."""
        with self._condition:
            self._active_prompts.clear()
            for cancellation in self._active_cancellations:
                cancellation.set()
            self._condition.notify_all()

    def submit(
        self,
        key: tuple[str, ...],
        callback: Callable[[CancellationCheck], None],
    ) -> bool:
        """Queue one deduplicated job without waiting for capacity or execution."""
        if not key or any(not str(part) for part in key):
            raise ValueError("R2 write-back job key must contain non-empty values.")
        with self._condition:
            if key in self._known_keys:
                return False
            if len(self._pending) >= self._max_pending_jobs:
                logger.warning(
                    "Dropping R2 write-back because the idle queue reached %d jobs key=%s.",
                    self._max_pending_jobs,
                    key[-1],
                )
                return False
            job = _R2WriteBackJob(key=key, callback=callback)
            self._pending[key] = job
            self._known_keys.add(key)
            self._condition.notify()
            return True

    def wait_for_idle(self, timeout_seconds: float | None = None) -> bool:
        """Wait until pending and active jobs drain, primarily for tests and shutdown."""
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        with self._condition:
            while self._pending or self._active_jobs:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def _worker(self) -> None:
        """Run queued jobs only during foreground-idle windows."""
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: bool(self._pending) and not self._active_prompts
                )
                key, job = self._pending.popitem(last=False)
                cancellation = threading.Event()
                self._active_cancellations.add(cancellation)
                self._active_jobs += 1
            cancelled = False
            try:
                job.callback(cancellation.is_set)
            except R2WriteBackCancelled:
                cancelled = True
            except (
                AssertionError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    "Unhandled R2 write-back job failure key=%s: %s",
                    key[-1],
                    exc,
                )
            finally:
                with self._condition:
                    self._active_cancellations.discard(cancellation)
                    self._active_jobs -= 1
                    if cancelled:
                        self._pending.setdefault(key, job)
                    else:
                        self._known_keys.discard(key)
                    self._condition.notify_all()


_R2_WRITE_BACK_COORDINATOR = R2WriteBackCoordinator()


def begin_r2_writeback_prompt(prompt_id: str) -> None:
    """Pause opportunistic cache population for one foreground prompt."""
    _R2_WRITE_BACK_COORDINATOR.begin_prompt(prompt_id)


def finish_r2_writeback_prompt(prompt_id: str) -> None:
    """Resume cache population after one foreground prompt leaves the queue."""
    _R2_WRITE_BACK_COORDINATOR.finish_prompt(prompt_id)


def _never_cancel() -> bool:
    """Return false for write-backs without a cooperative cancellation source."""
    return False


def _emit_sync_status(
    status_callback: SyncStatusCallback | None,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Emit one synchronization status update when available."""
    if status_callback is not None:
        status_callback(message, current, total)


def _format_r2_download_status(
    asset_name: str,
    *,
    item_index: int | None,
    total_items: int | None,
) -> str:
    """Return one queue-time worker-side R2 download status."""
    if item_index is not None and total_items is not None and total_items > 1:
        return (
            f"Downloading asset {item_index}/{total_items} from Cloudflare R2: "
            f"{asset_name}"
        )
    return f"Downloading asset from Cloudflare R2: {asset_name}"


class R2TransferHost(Protocol):
    """Define engine services used by R2 transfer coordination."""

    def _destination_label(self) -> str:
        """Return a user-facing worker destination."""

    def _raise_if_cancelled(self) -> None:
        """Raise when foreground synchronization was cancelled."""

    def _store_sync_record(
        self,
        *,
        sync_key: str,
        remote_path: str,
        source_description: str,
    ) -> None:
        """Persist one successful synchronization record."""


@dataclass
class R2TransferManager:
    """Own R2 preflight, read-through, and background write-back behavior."""

    _host: R2TransferHost
    volume: Any
    r2_cache: R2CacheClient | None = None
    cancellation_check: CancellationCheck | None = None
    r2_writeback_activity: Callable[[], AbstractContextManager[None]] | None = None

    def preflight_r2_access(
        self,
        *,
        status_callback: SyncStatusCallback | None = None,
    ) -> None:
        """Fail before asset transfer when a worker cannot use configured R2 URLs."""
        self._host._raise_if_cancelled()
        if self.r2_cache is None or not isinstance(
            self.volume,
            R2WorkerPreflightBackend,
        ):
            return
        destination = self._host._destination_label()
        _emit_sync_status(
            status_callback,
            f"Checking Cloudflare R2 access from {destination}",
        )
        try:
            request = self.r2_cache.worker_preflight_request()
        except (R2CacheError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise R2CacheError(
                "Cloudflare R2 controller validation failed before the worker "
                f"preflight: {exc}"
            ) from exc
        try:
            self.volume.preflight_r2_access(
                request,
                cancellation_check=self.cancellation_check,
            )
        except InterruptedError as exc:
            self._host._raise_if_cancelled()
            raise R2CacheError(
                f"Cloudflare R2 worker preflight was interrupted for {destination}."
            ) from exc
        except (R2CacheError, OSError, RuntimeError, TypeError, ValueError) as exc:
            diagnostic = str(exc)
            hint = (
                " Remove Client IP Address Filtering from the bucket-scoped R2 "
                "API token; dynamic worker egress addresses cannot be reliably "
                "allowlisted."
                if "r2_code=AccessDenied" in diagnostic
                else ""
            )
            raise R2CacheError(
                f"Cloudflare R2 is reachable from the controller but not from "
                f"{destination}.{hint} Safe worker diagnostic: {diagnostic}"
            ) from exc
        self._host._raise_if_cancelled()
        logger.info(
            "Validated Cloudflare R2 access from %s before asset transfer.",
            destination,
        )

    def _materialize_r2_source(
        self,
        spec: _ContentAddressedSyncSpec,
        size_bytes: int,
    ) -> _R2MaterializationOutcome:
        """Try one signed worker-side R2 download and preserve upload fallback."""
        if self.r2_cache is None or not isinstance(self.volume, R2MaterializingBackend):
            return _R2MaterializationOutcome(result=None)
        request: R2DownloadRequest | None = None
        try:
            request = self.r2_cache.download_request(spec.sha256, size_bytes)
            if request is None:
                return _R2MaterializationOutcome(result=None)
            _emit_sync_status(
                spec.status_callback,
                _format_r2_download_status(
                    spec.local_path.name,
                    item_index=spec.status_current,
                    total_items=spec.status_total,
                ),
                spec.status_current,
                spec.status_total,
            )
            self.volume.materialize_r2_file(
                request,
                spec.remote_path,
                cancellation_check=self.cancellation_check,
            )
        except InterruptedError as exc:
            raise SyncCancelledError("Remote workflow preparation was cancelled.") from exc
        except (R2CacheError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "Cloudflare R2 materialization failed for SHA-256 %s; "
                "falling back to the existing upload path: %s",
                spec.sha256,
                exc,
            )
            return _R2MaterializationOutcome(
                result=None,
                refresh_required=request is not None,
            )
        self._host._store_sync_record(
            sync_key=spec.sync_key,
            remote_path=spec.remote_path,
            source_description=f"Cloudflare R2 SHA-256 {spec.sha256}",
        )
        logger.info(
            "Materialized SHA-256 %s from Cloudflare R2 at %s.",
            spec.sha256,
            spec.remote_path,
        )
        return _R2MaterializationOutcome(
            result=_ContentAddressedSyncResult(
                remote_path=spec.remote_path,
                uploaded=True,
            )
        )

    def _schedule_r2_writeback(
        self,
        *,
        sha256: str,
        size_bytes: int,
        remote_path: str,
        force: bool = False,
    ) -> None:
        """Queue one best-effort R2 write without delaying foreground progress."""
        if not self._r2_writeback_enabled():
            return
        assert self.r2_cache is not None
        job_key = (
            *self._r2_writeback_job_prefix(),
            sha256,
            str(size_bytes),
            remote_path,
            "force" if force else "normal",
        )
        _R2_WRITE_BACK_COORDINATOR.submit(
            job_key,
            lambda cancellation_check: self._write_back_r2_file(
                sha256,
                size_bytes,
                remote_path,
                force,
                cancellation_check=cancellation_check,
            ),
        )

    def _r2_writeback_job_prefix(self) -> tuple[str, str, str]:
        """Return a stable cache namespace used to deduplicate background jobs."""
        if self.r2_cache is None:
            raise RuntimeError("R2 write-back job requested without an R2 cache.")
        configuration = getattr(self.r2_cache, "configuration", None)
        if configuration is not None:
            return (
                str(configuration.endpoint_url),
                str(configuration.bucket),
                str(configuration.key_prefix),
            )
        return (
            type(self.r2_cache).__module__,
            type(self.r2_cache).__qualname__,
            str(id(self.r2_cache)),
        )

    def _r2_writeback_enabled(self) -> bool:
        """Return whether this engine can populate its configured R2 cache."""
        return bool(
            self.r2_cache is not None
            and self.r2_cache.write_back_mode != "off"
            and isinstance(self.volume, R2MaterializingBackend)
        )

    def _write_back_r2_file(
        self,
        sha256: str,
        size_bytes: int,
        remote_path: str,
        force: bool = False,
        *,
        cancellation_check: CancellationCheck = _never_cancel,
    ) -> None:
        """Upload one worker-resident file to R2 without exposing permanent keys."""
        if self.r2_cache is None or not isinstance(
            self.volume,
            R2MaterializingBackend,
        ):
            return
        plan: R2UploadPlan | None = None
        try:
            if cancellation_check():
                raise R2WriteBackCancelled("R2 write-back yielded before activity.")
            activity = (
                self.r2_writeback_activity()
                if self.r2_writeback_activity is not None
                else nullcontext()
            )
            with activity:
                if cancellation_check():
                    raise R2WriteBackCancelled("R2 write-back yielded before signing.")
                plan = self.r2_cache.prepare_upload(sha256, size_bytes, force=force)
                if plan is None:
                    return
                if cancellation_check():
                    raise R2WriteBackCancelled("R2 write-back yielded after signing.")
                if isinstance(self.volume, CancellableR2WriteBackBackend):
                    result = self.volume.upload_r2_file_cancellable(
                        plan,
                        remote_path,
                        cancellation_check=cancellation_check,
                    )
                else:
                    result = self.volume.upload_r2_file(plan, remote_path)
                if cancellation_check():
                    raise R2WriteBackCancelled(
                        "R2 write-back yielded before completion."
                    )
                self.r2_cache.complete_upload(plan, result)
                logger.info("Wrote SHA-256 %s back to Cloudflare R2.", sha256)
        except (InterruptedError, R2WriteBackCancelled) as exc:
            if plan is not None:
                self.r2_cache.abort_upload(plan)
            raise R2WriteBackCancelled(str(exc)) from exc
        except (R2CacheError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if plan is not None:
                self.r2_cache.abort_upload(plan)
            logger.warning("Cloudflare R2 write-back failed for SHA-256 %s: %s", sha256, exc)

    def wait_for_r2_writebacks(self) -> None:
        """Wait for this engine's currently scheduled R2 writes to finish."""
        _R2_WRITE_BACK_COORDINATOR.wait_for_idle()

