"""Modal app-existence checks used before constructing the cloud entrypoint."""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

try:
    from .settings import modal_deployment_app_name
except ImportError:  # pragma: no cover - flat Modal-container import.
    from settings import modal_deployment_app_name

logger = logging.getLogger(__name__)


def _modal_exception_types(
    modal_module: Any,
    names: Sequence[str],
) -> tuple[type[BaseException], ...]:
    """Return available Modal exception types from a Modal SDK-like module."""
    exception_namespace = getattr(modal_module, "exception", None)
    if exception_namespace is None:
        return tuple()

    error_types: list[type[BaseException]] = []
    for error_name in names:
        error_type = getattr(exception_namespace, error_name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            error_types.append(error_type)
    return tuple(error_types)


def _modal_lookup_create_flag_name(app_lookup: Callable[..., Any]) -> str | None:
    """Return the App.lookup keyword that disables creation when available."""
    try:
        lookup_signature = inspect.signature(app_lookup)
    except (TypeError, ValueError):
        return None
    for flag_name in ("create_if_missing", "create"):
        if flag_name in lookup_signature.parameters:
            return flag_name
    return None


def _modal_app_exists_via_sdk(modal_module: Any, app_name: str) -> bool | None:
    """Check for a Modal app through a non-creating SDK lookup when possible."""
    app_namespace = getattr(modal_module, "App", None)
    app_lookup = getattr(app_namespace, "lookup", None)
    if not callable(app_lookup):
        return None

    create_flag_name = _modal_lookup_create_flag_name(app_lookup)
    if create_flag_name is None:
        return None

    not_found_errors = _modal_exception_types(modal_module, ("NotFoundError",))
    try:
        app_lookup(app_name, **{create_flag_name: False})
    except not_found_errors:
        return False
    return True


def _app_name_from_modal_app_list_entry(entry: Mapping[str, Any]) -> str | None:
    """Extract an app name from one ``modal app list --json`` entry."""
    for key in ("name", "app_name", "Name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _modal_app_exists_via_cli(app_name: str) -> bool | None:
    """Check for a Modal app by parsing the CLI's JSON output when available."""
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        return None

    command = [modal_cli, "app", "list", "--json"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Unable to list Modal apps through the Modal CLI: %s", exc)
        return None
    if completed.returncode != 0:
        logger.warning(
            "Modal CLI app list failed with exit code %s: %s",
            completed.returncode,
            completed.stderr.strip(),
        )
        return None

    try:
        listed_apps = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("Modal CLI app list returned invalid JSON: %s", exc)
        return None
    if not isinstance(listed_apps, list):
        return None

    return any(
        _app_name_from_modal_app_list_entry(entry) == app_name
        for entry in listed_apps
        if isinstance(entry, Mapping)
    )


def _running_inside_modal_container(modal_module: Any) -> bool:
    """Return whether this import is happening inside a Modal worker container."""
    if os.getenv("MODAL_TASK_ID"):
        return True
    is_local = getattr(modal_module, "is_local", None)
    if callable(is_local):
        try:
            return not bool(is_local())
        except RuntimeError:
            return False
    return False


def _existing_modal_app_error_message(app_name: str) -> str:
    """Return the user-facing error for refusing to overwrite a Modal app."""
    return (
        f"Modal app {app_name!r} already exists. Delete the existing app before launching "
        "ComfyUI-Modal with this configuration, especially after changing the workflow GPU "
        "target or the COMFY_MODAL_GPU compatibility fallback, "
        f"for example: modal app stop {app_name!r}."
    )


def guard_against_existing_modal_app(
    settings: Any,
    modal_module: Any,
    *,
    error_type: type[RuntimeError],
) -> None:
    """Fail local app construction when the configured Modal app already exists."""
    if _running_inside_modal_container(modal_module):
        return

    app_name = modal_deployment_app_name(settings)
    app_exists = _modal_app_exists_via_sdk(modal_module, app_name)
    if app_exists is None:
        app_exists = _modal_app_exists_via_cli(app_name)
    if app_exists:
        raise error_type(_existing_modal_app_error_message(app_name))
