"""Administrative operations for persistent Modal objects."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

if __package__:
    from .settings import ModalSyncSettings
else:  # pragma: no cover - flat import inside the Modal container.
    from settings import ModalSyncSettings

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional local dependency.
    modal = None


def _modal_not_found_error_types() -> tuple[type[BaseException], ...]:
    """Return Modal SDK exception classes that mean a named object is absent."""
    if modal is None:
        return ()
    exception_namespace = getattr(modal, "exception", None)
    candidates = [
        getattr(exception_namespace, "NotFoundError", None),
        getattr(exception_namespace, "InvalidError", None),
    ]
    return tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, type) and issubclass(candidate, BaseException)
    )


def _modal_cache_dict_names(settings: ModalSyncSettings) -> list[str]:
    """Return the configured persistent Modal Dict names used as local-reset caches."""
    return [
        settings.interrupt_dict_name,
        settings.node_output_cache_dict_name,
        settings.session_bridge_dict_name,
        settings.sync_index_dict_name,
        settings.snapshot_profile_dict_name,
    ]


async def _call_modal_sdk(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a Modal SDK method without blocking the aiohttp event loop."""
    async_method = getattr(method, "aio", None)
    if callable(async_method):
        return await async_method(*args, **kwargs)
    return await asyncio.to_thread(method, *args, **kwargs)


async def _delete_modal_named_object(
    namespace: Any, name: str, *, object_label: str
) -> None:
    """Delete a named Modal object using the supported manager API when available."""
    objects_manager = getattr(namespace, "objects", None)
    manager_delete = getattr(objects_manager, "delete", None)
    if callable(manager_delete):
        await _call_modal_sdk(manager_delete, name, allow_missing=True)
        return

    instance = await _call_modal_sdk(namespace.from_name, name, create_if_missing=False)
    delete_method = getattr(instance, "delete", None)
    if not callable(delete_method):
        raise RuntimeError(f"Modal {object_label} {name!r} does not expose delete().")
    await _call_modal_sdk(delete_method, name)


async def delete_modal_cache_dicts(settings: ModalSyncSettings) -> dict[str, Any]:
    """Delete all configured Modal Dict caches and return a reset summary."""
    if modal is None:
        raise RuntimeError("Modal SDK is unavailable; cannot delete Modal caches.")
    modal_dict = getattr(modal, "Dict", None)
    if modal_dict is None:
        raise RuntimeError(
            "Modal SDK does not expose modal.Dict; cannot delete Modal caches."
        )

    deleted: list[str] = []
    skipped: list[str] = []
    not_found_errors = _modal_not_found_error_types()
    for dict_name in _modal_cache_dict_names(settings):
        try:
            await _call_modal_sdk(
                modal_dict.from_name,
                dict_name,
                create_if_missing=False,
            )
        except not_found_errors:
            skipped.append(dict_name)
            continue
        await _delete_modal_named_object(modal_dict, dict_name, object_label="Dict")
        deleted.append(dict_name)

    logger.info("Deleted Modal cache Dicts deleted=%s skipped=%s.", deleted, skipped)
    return {"deleted": deleted, "skipped": skipped}


async def delete_modal_volume(settings: ModalSyncSettings) -> dict[str, Any]:
    """Delete the configured Modal Volume and return a reset summary."""
    if modal is None:
        raise RuntimeError("Modal SDK is unavailable; cannot delete Modal volume.")
    modal_volume = getattr(modal, "Volume", None)
    if modal_volume is None:
        raise RuntimeError(
            "Modal SDK does not expose modal.Volume; cannot delete Modal volume."
        )

    try:
        await _call_modal_sdk(
            modal_volume.from_name,
            settings.volume_name,
            create_if_missing=False,
        )
    except _modal_not_found_error_types():
        logger.info("Skipped deleting missing Modal Volume %s.", settings.volume_name)
        return {"deleted": [], "skipped": [settings.volume_name]}

    await _delete_modal_named_object(
        modal_volume, settings.volume_name, object_label="Volume"
    )
    logger.info("Deleted Modal Volume %s.", settings.volume_name)
    return {"deleted": [settings.volume_name], "skipped": []}
