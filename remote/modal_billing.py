"""Modal hourly billing queries and bounded interval caches."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import importlib
import logging
import os
import threading
from typing import Any, Callable, Mapping

from ..settings import (
    ModalSyncSettings,
    get_settings,
    modal_deployment_app_name,
    settings_for_modal_gpu,
)

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback environments.
    modal = None

_MODAL_BILLING_COLLECTION_BUFFER = timedelta(minutes=10)
_MODAL_BILLING_INTERVAL = timedelta(hours=1)
_MODAL_HOURLY_BILLING_CACHE_LOCK = threading.Lock()
_MODAL_HOURLY_BILLING_CACHE: dict[
    tuple[str, str, datetime], "ModalHourlyBillingStatus"
] = {}
_MODAL_HOURLY_BILLING_ERROR_CACHE: dict[tuple[str, str, datetime], str] = {}
_MODAL_HOURLY_BILLING_CACHE_LIMIT = 64


class ModalBillingStatusError(RuntimeError):
    """Raised when Modal hourly billing data cannot be queried."""


@dataclass(frozen=True)
class ModalHourlyBillingStatus:
    """Describe one completed hourly billing interval for a GPU-specific app."""

    app_id: str | None
    app_name: str
    environment_name: str
    modal_gpu: str
    interval_start: datetime
    interval_end: datetime
    app_cost_usd_before_credits: Decimal
    has_usage: bool
    fetched_at: datetime
    next_refresh_at: datetime

    def as_dict(self) -> dict[str, str | float | bool | None]:
        """Return a JSON-serializable representation for the frontend."""
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "environment_name": self.environment_name,
            "modal_gpu": self.modal_gpu,
            "resolution": "hour",
            "interval_start": self.interval_start.isoformat(),
            "interval_end": self.interval_end.isoformat(),
            "app_cost_usd_before_credits": float(self.app_cost_usd_before_credits),
            "has_usage": self.has_usage,
            "fetched_at": self.fetched_at.isoformat(),
            "next_refresh_at": self.next_refresh_at.isoformat(),
            "collection_buffer_seconds": int(
                _MODAL_BILLING_COLLECTION_BUFFER.total_seconds()
            ),
        }


def _modal_environment_name() -> str | None:
    """Return the active Modal environment name when explicitly configured."""
    environment_name = os.getenv("MODAL_ENVIRONMENT")
    if environment_name is None:
        return None
    normalized = environment_name.strip()
    return normalized or None


def _resolved_modal_environment_name() -> str | None:
    """Return the explicitly selected Modal environment, if one is configured."""
    object_module = importlib.import_module("modal._object")
    environments_module = importlib.import_module("modal.environments")
    environment = environments_module.ensure_env(_modal_environment_name())
    environment_name = object_module._get_environment_name(environment)
    if environment_name is None:
        return None
    normalized_environment_name = str(environment_name).strip()
    return normalized_environment_name or None


def _completed_modal_billing_interval(
    now: datetime,
) -> tuple[datetime, datetime, datetime]:
    """Return the buffered completed hour and its next eligible refresh time."""
    normalized_now = (
        now.replace(tzinfo=timezone.utc)
        if now.tzinfo is None
        else now.astimezone(timezone.utc)
    )
    buffered_now = normalized_now - _MODAL_BILLING_COLLECTION_BUFFER
    interval_end = buffered_now.replace(minute=0, second=0, microsecond=0)
    interval_start = interval_end - _MODAL_BILLING_INTERVAL
    next_refresh_at = (
        interval_end + _MODAL_BILLING_INTERVAL + _MODAL_BILLING_COLLECTION_BUFFER
    )
    return interval_start, interval_end, next_refresh_at


def _modal_billing_row_value(row: Any, key: str) -> Any:
    """Read one billing row field from a mapping or SDK dataclass."""
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def _modal_billing_cost(value: Any) -> Decimal:
    """Normalize one Modal billing cost without losing decimal precision."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ModalBillingStatusError(
            f"Modal returned an invalid billing cost {value!r}."
        ) from exc


def _prune_modal_hourly_billing_cache() -> None:
    """Bound retained hourly billing records across apps and GPU selections."""
    retained_keys = set(_MODAL_HOURLY_BILLING_CACHE) | set(
        _MODAL_HOURLY_BILLING_ERROR_CACHE
    )
    overflow = len(retained_keys) - _MODAL_HOURLY_BILLING_CACHE_LIMIT
    if overflow <= 0:
        return
    oldest_keys = sorted(retained_keys, key=lambda cache_key: cache_key[2])[:overflow]
    for cache_key in oldest_keys:
        _MODAL_HOURLY_BILLING_CACHE.pop(cache_key, None)
        _MODAL_HOURLY_BILLING_ERROR_CACHE.pop(cache_key, None)


def _matching_modal_hourly_billing_rows(
    rows: Any,
    *,
    app_name: str,
    environment_name: str | None,
    interval_start: datetime,
) -> tuple[list[Any], str]:
    """Select every historical app identity from one billing environment."""
    candidate_rows = [
        row
        for row in rows
        if str(_modal_billing_row_value(row, "description") or "") == app_name
        and _modal_billing_row_value(row, "interval_start") == interval_start
    ]
    if environment_name is not None:
        matching_rows = [
            row
            for row in candidate_rows
            if str(_modal_billing_row_value(row, "environment_name") or "")
            == environment_name
        ]
        return matching_rows, environment_name
    candidate_environment_names = {
        str(_modal_billing_row_value(row, "environment_name") or "").strip()
        for row in candidate_rows
    }
    if len(candidate_environment_names) > 1:
        raise ModalBillingStatusError(
            f"Modal returned billing for app {app_name!r} in multiple "
            "environments; set MODAL_ENVIRONMENT to select one."
        )
    reported_environment_name = next(
        (
            str(_modal_billing_row_value(row, "environment_name") or "").strip()
            for row in candidate_rows
            if str(_modal_billing_row_value(row, "environment_name") or "").strip()
        ),
        "<workspace default>",
    )
    return candidate_rows, reported_environment_name


def _modal_hourly_billing_status_from_rows(
    rows: Any,
    *,
    app_name: str,
    environment_name: str | None,
    modal_gpu: str,
    interval_start: datetime,
    interval_end: datetime,
    fetched_at: datetime,
    next_refresh_at: datetime,
) -> ModalHourlyBillingStatus:
    """Build one app billing status from an hourly workspace report."""
    matching_rows, reported_environment_name = _matching_modal_hourly_billing_rows(
        rows,
        app_name=app_name,
        environment_name=environment_name,
        interval_start=interval_start,
    )
    app_cost = sum(
        (
            _modal_billing_cost(_modal_billing_row_value(row, "cost"))
            for row in matching_rows
        ),
        start=Decimal("0"),
    )
    app_ids = {
        str(app_id)
        for row in matching_rows
        if (app_id := _modal_billing_row_value(row, "object_id"))
    }
    return ModalHourlyBillingStatus(
        app_id=next(iter(app_ids)) if len(app_ids) == 1 else None,
        app_name=app_name,
        environment_name=reported_environment_name,
        modal_gpu=modal_gpu,
        interval_start=interval_start,
        interval_end=interval_end,
        app_cost_usd_before_credits=app_cost,
        has_usage=bool(matching_rows),
        fetched_at=fetched_at,
        next_refresh_at=next_refresh_at,
    )


def _fetch_modal_hourly_billing_synchronously(
    report: Callable[..., Any],
    error_types: tuple[type[BaseException], ...],
    *,
    app_name: str,
    environment_name: str | None,
    modal_gpu: str,
    interval_start: datetime,
    interval_end: datetime,
    fetched_at: datetime,
    next_refresh_at: datetime,
) -> ModalHourlyBillingStatus:
    """Fetch and cache one hourly report while serializing concurrent UI polls."""
    cache_key = (environment_name or "<workspace default>", app_name, interval_start)
    with _MODAL_HOURLY_BILLING_CACHE_LOCK:
        cached_status = _MODAL_HOURLY_BILLING_CACHE.get(cache_key)
        if cached_status is not None:
            return cached_status
        cached_error = _MODAL_HOURLY_BILLING_ERROR_CACHE.get(cache_key)
        if cached_error is not None:
            raise ModalBillingStatusError(cached_error)
        try:
            rows = report(start=interval_start, end=interval_end, resolution="h")
            status = _modal_hourly_billing_status_from_rows(
                rows,
                app_name=app_name,
                environment_name=environment_name,
                modal_gpu=modal_gpu,
                interval_start=interval_start,
                interval_end=interval_end,
                fetched_at=fetched_at,
                next_refresh_at=next_refresh_at,
            )
        except error_types as exc:
            cached_error = str(exc)
            _MODAL_HOURLY_BILLING_ERROR_CACHE[cache_key] = cached_error
            _prune_modal_hourly_billing_cache()
            raise ModalBillingStatusError(cached_error) from exc
        _MODAL_HOURLY_BILLING_CACHE[cache_key] = status
        _prune_modal_hourly_billing_cache()
        return status


async def get_hourly_modal_app_billing(
    modal_gpu: str,
    settings: ModalSyncSettings | None = None,
    *,
    now: datetime | None = None,
) -> ModalHourlyBillingStatus:
    """Return cached billing for the latest buffered hour of one GPU app."""
    if modal is None:
        raise ModalBillingStatusError("The Modal SDK is unavailable.")
    resolved_settings = settings_for_modal_gpu(settings or get_settings(), modal_gpu)
    app_name = modal_deployment_app_name(resolved_settings)
    requested_now = now or datetime.now(timezone.utc)
    fetched_at = (
        requested_now.replace(tzinfo=timezone.utc)
        if requested_now.tzinfo is None
        else requested_now.astimezone(timezone.utc)
    )
    interval_start, interval_end, next_refresh_at = _completed_modal_billing_interval(
        fetched_at
    )
    try:
        environment_name = _resolved_modal_environment_name()
        billing_module = importlib.import_module("modal.billing")
        exception_module = importlib.import_module("modal.exception")
    except (ModuleNotFoundError, AttributeError, RuntimeError) as exc:
        raise ModalBillingStatusError(
            "The installed Modal SDK does not expose hourly billing reports."
        ) from exc
    report = getattr(billing_module, "workspace_billing_report", None)
    if not callable(report):
        raise ModalBillingStatusError(
            "The installed Modal SDK does not expose workspace_billing_report()."
        )
    modal_error_type = getattr(exception_module, "Error", RuntimeError)
    billing_error_types = (
        modal_error_type,
        ModalBillingStatusError,
        OSError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    try:
        return await asyncio.to_thread(
            _fetch_modal_hourly_billing_synchronously,
            report,
            billing_error_types,
            app_name=app_name,
            environment_name=environment_name,
            modal_gpu=resolved_settings.modal_gpu,
            interval_start=interval_start,
            interval_end=interval_end,
            fetched_at=fetched_at,
            next_refresh_at=next_refresh_at,
        )
    except billing_error_types as exc:
        raise ModalBillingStatusError(
            f"Unable to fetch Modal hourly billing: {exc}"
        ) from exc
