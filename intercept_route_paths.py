"""Route-path derivation for the ComfyUI prompt interception API."""

from __future__ import annotations


def _analysis_route_path(route_path: str) -> str:
    """Return the sibling HTTP route used for dry-run remote-node expansion."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/analyze_remote_nodes"
    return f"{route_path.rstrip('/')}/analyze_remote_nodes"


def _progress_state_route_path(route_path: str) -> str:
    """Return the sibling HTTP route used for Modal UI event replay."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/progress_state"
    return f"{route_path.rstrip('/')}/progress_state"


def _container_status_route_path(route_path: str) -> str:
    """Return the sibling HTTP route used for active Modal container status."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/container_status"
    return f"{route_path.rstrip('/')}/container_status"


def _delete_modal_caches_route_path(route_path: str) -> str:
    """Return the sibling HTTP route used to delete persistent Modal cache Dicts."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/delete_caches"
    return f"{route_path.rstrip('/')}/delete_caches"


def _delete_modal_volume_route_path(route_path: str) -> str:
    """Return the sibling HTTP route used to delete the configured Modal Volume."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/delete_volume"
    return f"{route_path.rstrip('/')}/delete_volume"


def _cancel_preparation_route_path(route_path: str) -> str:
    """Return the sibling route used to cancel queue-time remote preparation."""
    if route_path.endswith("/queue_prompt"):
        return f"{route_path.removesuffix('/queue_prompt')}/cancel_preparation"
    return f"{route_path.rstrip('/')}/cancel_preparation"


def _remote_environments_route_path(route_path: str) -> str:
    """Return the provider-neutral environment management route."""
    del route_path
    return "/remote/environments"


def _remote_environment_probe_route_path(route_path: str) -> str:
    """Return the provider-neutral host capability probe route."""
    del route_path
    return "/remote/environments/probe"


def _remote_environment_bootstrap_route_path(route_path: str) -> str:
    """Return the provider-neutral worker bootstrap route."""
    del route_path
    return "/remote/environments/bootstrap"


def _remote_environment_status_route_path(route_path: str) -> str:
    """Return the provider-neutral worker status route."""
    del route_path
    return "/remote/environments/status"


def _remote_environment_stop_route_path(route_path: str) -> str:
    """Return the provider-neutral worker shutdown route."""
    del route_path
    return "/remote/environments/stop"
