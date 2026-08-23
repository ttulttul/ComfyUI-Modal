"""Asynchronous Vast.ai REST API client with bounded retries and redaction."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import aiohttp

if __package__:
    from .vast_models import (
        VAST_API_BASE_URL,
        VastInstance,
        VastInstanceLaunchSpec,
        VastOffer,
        VastResourceProfile,
        compatible_offers,
    )
else:  # pragma: no cover - direct simulator and debugging imports.
    from vast_models import (
        VAST_API_BASE_URL,
        VastInstance,
        VastInstanceLaunchSpec,
        VastOffer,
        VastResourceProfile,
        compatible_offers,
    )

logger = logging.getLogger(__name__)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class VastApiError(RuntimeError):
    """Raised when the Vast API rejects or cannot complete an operation."""


class VastAuthenticationError(VastApiError):
    """Raised when the configured Vast API key is invalid or insufficient."""


class VastOfferUnavailableError(VastApiError):
    """Raised when a marketplace offer disappears before instance creation."""


class VastInstanceNotFoundError(VastApiError):
    """Raised when a managed Vast instance no longer exists."""


@dataclass(frozen=True)
class VastCreateResult:
    """Return the non-secret result of a successful instance rental."""

    instance_id: int


class VastApiClient:
    """Call the Vast.ai API without exposing bearer or per-instance secrets."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = VAST_API_BASE_URL,
        request_timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Configure one client using an injected or internally managed session."""
        normalized_key = api_key.strip()
        if not normalized_key or any(
            character in normalized_key for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("A non-empty single-line Vast API key is required.")
        normalized_base_url = base_url.rstrip("/")
        if not normalized_base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Vast API base URL must use HTTPS or loopback HTTP.")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive.")
        self._api_key = normalized_key
        self._base_url = normalized_base_url
        self._request_timeout_seconds = request_timeout_seconds
        self._retry_attempts = retry_attempts
        self._session = session

    async def verify_credentials(self) -> dict[str, Any]:
        """Return a small non-secret account summary after validating the API key."""
        payload = await self._request_json("GET", "/api/v0/users/current/")
        return {
            "id": payload.get("id"),
            "credit": payload.get("credit"),
        }

    async def search_offers(
        self,
        profile: VastResourceProfile,
        *,
        limit: int = 25,
    ) -> tuple[VastOffer, ...]:
        """Return locally revalidated compatible offers in best-price order."""
        payload = await self._request_json(
            "POST",
            "/api/v0/bundles/",
            json_payload=profile.search_payload(limit=limit),
        )
        raw_offers = payload.get("offers")
        if isinstance(raw_offers, Mapping):
            raw_offer_records: Sequence[object] = [raw_offers]
        elif isinstance(raw_offers, list):
            raw_offer_records = raw_offers
        else:
            raise VastApiError("Vast offer search returned an invalid offers field.")
        try:
            offers = tuple(
                VastOffer.from_api(offer)
                for offer in raw_offer_records
                if isinstance(offer, Mapping)
            )
        except ValueError as exc:
            raise VastApiError("Vast offer search returned a malformed offer.") from exc
        return compatible_offers(offers, profile)

    async def create_instance(
        self,
        offer_id: int,
        launch_spec: VastInstanceLaunchSpec,
    ) -> VastCreateResult:
        """Rent one offer and return only its new instance identity."""
        payload = await self._request_json(
            "PUT",
            f"/api/v0/asks/{offer_id}/",
            json_payload=launch_spec.to_api_payload(),
            offer_creation=True,
        )
        if not payload.get("success", False):
            raise VastApiError(_safe_error_message(payload, "Vast did not create the instance."))
        try:
            instance_id = int(payload["new_contract"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VastApiError(
                "Vast created an instance but omitted its contract identity."
            ) from exc
        if instance_id <= 0:
            raise VastApiError("Vast returned an invalid instance identity.")
        return VastCreateResult(instance_id=instance_id)

    async def show_instance(self, instance_id: int) -> VastInstance:
        """Return one normalized instance state."""
        payload = await self._request_json(
            "GET", f"/api/v0/instances/{_positive_id(instance_id)}/"
        )
        raw_instance = payload.get("instances")
        if not isinstance(raw_instance, Mapping):
            raise VastApiError("Vast show-instance returned an invalid instance field.")
        try:
            return VastInstance.from_api(raw_instance)
        except ValueError as exc:
            raise VastApiError("Vast show-instance returned malformed state.") from exc

    async def list_instances(self) -> tuple[VastInstance, ...]:
        """Return every visible instance, following Vast keyset pagination."""
        instances: list[VastInstance] = []
        next_token: str | None = None
        for _page in range(100):
            query = f"?next_token={next_token}" if next_token else ""
            payload = await self._request_json("GET", f"/api/v1/instances/{query}")
            raw_instances = payload.get("instances")
            if not isinstance(raw_instances, list):
                raise VastApiError("Vast instance listing returned invalid state.")
            try:
                instances.extend(
                    VastInstance.from_api(instance)
                    for instance in raw_instances
                    if isinstance(instance, Mapping)
                )
            except ValueError as exc:
                raise VastApiError("Vast instance listing returned malformed state.") from exc
            raw_next_token = payload.get("next_token")
            next_token = str(raw_next_token).strip() if raw_next_token else None
            if not next_token:
                return tuple(instances)
        raise VastApiError("Vast instance listing exceeded the pagination limit.")

    async def set_instance_state(self, instance_id: int, state: str) -> None:
        """Set one instance to running or stopped."""
        if state not in {"running", "stopped"}:
            raise ValueError("Vast instance state must be 'running' or 'stopped'.")
        payload = await self._request_json(
            "PUT",
            f"/api/v0/instances/{_positive_id(instance_id)}/",
            json_payload={"state": state},
        )
        if not payload.get("success", False):
            raise VastApiError(_safe_error_message(payload, "Vast did not update the instance."))

    async def destroy_instance(self, instance_id: int) -> None:
        """Permanently destroy one exact Vast instance."""
        payload = await self._request_json(
            "DELETE", f"/api/v0/instances/{_positive_id(instance_id)}/"
        )
        if not payload.get("success", False):
            raise VastApiError(_safe_error_message(payload, "Vast did not destroy the instance."))

    async def wait_until_ready(
        self,
        instance_id: int,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 10.0,
    ) -> VastInstance:
        """Poll until one instance exposes a running SSH endpoint."""
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("Vast readiness timeout and poll interval must be positive.")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        last_instance: VastInstance | None = None
        while loop.time() < deadline:
            last_instance = await self.show_instance(instance_id)
            if last_instance.ready_for_ssh:
                return last_instance
            if last_instance.actual_status in {"error", "exited", "destroyed"}:
                raise VastApiError(
                    f"Vast instance {instance_id} entered terminal state "
                    f"{last_instance.actual_status!r} before SSH was ready."
                )
            await asyncio.sleep(min(poll_interval_seconds, max(0.0, deadline - loop.time())))
        last_status = last_instance.actual_status if last_instance is not None else "unknown"
        raise TimeoutError(
            f"Vast instance {instance_id} did not become SSH-ready within "
            f"{timeout_seconds:.0f}s; last status was {last_status!r}."
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        offer_creation: bool = False,
    ) -> dict[str, Any]:
        """Execute one authenticated request with bounded transient retries."""
        last_error: VastApiError | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                status, body, retry_after = await self._request_once(
                    method,
                    path,
                    json_payload=json_payload,
                )
            except (aiohttp.ClientError, TimeoutError) as exc:
                last_error = VastApiError(
                    f"Vast API request failed before receiving a response: {type(exc).__name__}."
                )
                if attempt >= self._retry_attempts:
                    raise last_error from exc
                await asyncio.sleep(_retry_delay(attempt, None))
                continue
            payload = _decode_response_json(body)
            if 200 <= status < 300:
                return payload
            error_message = _safe_error_message(
                payload,
                f"Vast API returned HTTP {status}.",
            )
            if status in {401, 403}:
                raise VastAuthenticationError(error_message)
            if status == 404:
                if offer_creation:
                    raise VastOfferUnavailableError(error_message)
                if path.startswith("/api/v0/instances/"):
                    raise VastInstanceNotFoundError(error_message)
                raise VastApiError(error_message)
            last_error = VastApiError(error_message)
            if status not in _RETRIABLE_STATUS_CODES or attempt >= self._retry_attempts:
                raise last_error
            await asyncio.sleep(_retry_delay(attempt, retry_after))
        if last_error is not None:
            raise last_error
        raise VastApiError("Vast API request ended without a result.")

    async def _request_once(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None,
    ) -> tuple[int, bytes, str | None]:
        """Return one response status, bounded body, and Retry-After value."""
        session = self._session
        owns_session = session is None
        if session is None:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._request_timeout_seconds)
            )
        try:
            async with session.request(
                method,
                f"{self._base_url}{path}",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                json=dict(json_payload) if json_payload is not None else None,
                allow_redirects=False,
            ) as response:
                body = await _read_bounded_body(response)
                return response.status, body, response.headers.get("Retry-After")
        finally:
            if owns_session:
                await session.close()


async def _read_bounded_body(response: aiohttp.ClientResponse) -> bytes:
    """Read one API response without accepting an unbounded body."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise VastApiError("Vast API response exceeded the 8 MiB safety limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_response_json(body: bytes) -> dict[str, Any]:
    """Decode one JSON object while keeping invalid responses secret-safe."""
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VastApiError("Vast API returned a non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise VastApiError("Vast API returned a non-object JSON response.")
    return payload


def _safe_error_message(payload: Mapping[str, Any], fallback: str) -> str:
    """Return a bounded error message without reflecting token-like fields."""
    for field_name in ("msg", "message", "error", "detail"):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            normalized = value.strip().replace("\n", " ").replace("\r", " ")
            if "api_key" not in normalized.casefold() and "token" not in normalized.casefold():
                return normalized[:1000]
    return fallback


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Return one bounded retry delay with Retry-After precedence."""
    if retry_after:
        try:
            return min(30.0, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(8.0, 0.5 * 2 ** (attempt - 1))


def _positive_id(instance_id: int) -> int:
    """Return a positive non-boolean Vast identity."""
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("Vast instance identity must be a positive integer.")
    return instance_id


__all__ = [
    "VastApiClient",
    "VastApiError",
    "VastAuthenticationError",
    "VastCreateResult",
    "VastInstanceNotFoundError",
    "VastOfferUnavailableError",
]
