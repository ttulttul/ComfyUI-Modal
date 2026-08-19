"""ComfyUI nodes for OpenAI-compatible Modal inference endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol
from urllib.parse import urlsplit

import aiohttp
import torch
from comfy_api.latest import _io as io
from PIL import Image

logger = logging.getLogger(__name__)

MODAL_ENDPOINT_CHAT_NODE_ID = "ModalEndpointChat"
MODAL_KEY_ENV = "MODAL_KEY"
MODAL_SECRET_ENV = "MODAL_SECRET"
_KEYRING_SERVICE = "ComfyUI Modal-Sync"
_KEYRING_WRITE_TEST_USER = "__credential_store_write_test__"
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_TOKEN_CREATION_TIMEOUT_SECONDS = 120
_TOKEN_CREATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModalProxyCredentials:
    """Modal proxy-token credentials used to authenticate endpoint requests."""

    key: str
    secret: str

    def validate(self) -> None:
        """Reject missing values and Modal API tokens with incompatible prefixes."""
        if not self.key or not self.secret:
            raise ValueError(
                "Modal proxy-token credentials must include both key and secret."
            )
        if not self.key.startswith("wk-") or not self.secret.startswith("ws-"):
            raise ValueError(
                "Modal endpoint credentials must be a proxy-token pair with wk- and ws- prefixes."
            )


class CredentialStore(Protocol):
    """Storage interface for a Modal proxy-token pair."""

    def load(self) -> ModalProxyCredentials | None:
        """Load a stored credential pair when one exists."""

    def save(self, credentials: ModalProxyCredentials) -> None:
        """Persist one credential pair securely."""

    def ensure_writable(self) -> None:
        """Verify that a newly created one-time secret can be persisted."""


class ProxyTokenCreator(Protocol):
    """Creation interface for Modal proxy tokens."""

    def create(self) -> ModalProxyCredentials:
        """Create and return a new Modal proxy-token pair."""


class ComfyUISecretManager:
    """Store Modal credentials in the operating-system vault for this ComfyUI node."""

    def __init__(self, keyring_module: Any | None = None) -> None:
        """Use an injected keyring implementation or import the system keyring package."""
        if keyring_module is None:
            import keyring

            keyring_module = keyring
        self._keyring = keyring_module

    def load(self) -> ModalProxyCredentials | None:
        """Load the pair without ever logging either credential value."""
        try:
            key = self._keyring.get_password(_KEYRING_SERVICE, MODAL_KEY_ENV)
            secret = self._keyring.get_password(_KEYRING_SERVICE, MODAL_SECRET_ENV)
        except self._keyring.errors.KeyringError as exc:
            raise RuntimeError(
                "ComfyUI could not access a secure operating-system credential vault for Modal tokens."
            ) from exc
        if not key and not secret:
            return None
        if not key or not secret:
            raise RuntimeError(
                "The ComfyUI Modal credential vault contains only one half of the proxy-token pair."
            )
        credentials = ModalProxyCredentials(key=key, secret=secret)
        credentials.validate()
        return credentials

    def save(self, credentials: ModalProxyCredentials) -> None:
        """Persist both values in the OS vault, removing a partial write on failure."""
        credentials.validate()
        try:
            self._keyring.set_password(_KEYRING_SERVICE, MODAL_KEY_ENV, credentials.key)
            self._keyring.set_password(
                _KEYRING_SERVICE, MODAL_SECRET_ENV, credentials.secret
            )
        except self._keyring.errors.KeyringError as exc:
            self._delete_partial_credentials()
            raise RuntimeError(
                "ComfyUI could not save the new Modal proxy token in the secure credential vault."
            ) from exc

    def ensure_writable(self) -> None:
        """Write and remove a harmless probe before creating a one-time Modal secret."""
        try:
            self._keyring.set_password(
                _KEYRING_SERVICE,
                _KEYRING_WRITE_TEST_USER,
                "write-test",
            )
            self._keyring.delete_password(_KEYRING_SERVICE, _KEYRING_WRITE_TEST_USER)
        except self._keyring.errors.KeyringError as exc:
            raise RuntimeError(
                "ComfyUI cannot create a Modal proxy token because its secure credential vault is not writable."
            ) from exc

    def _delete_partial_credentials(self) -> None:
        """Best-effort cleanup for a credential pair whose write was interrupted."""
        for username in (MODAL_KEY_ENV, MODAL_SECRET_ENV):
            try:
                self._keyring.delete_password(_KEYRING_SERVICE, username)
            except self._keyring.errors.KeyringError:
                logger.warning(
                    "Could not remove a partial Modal credential-vault entry."
                )


class ModalCliProxyTokenCreator:
    """Create Modal proxy tokens through the authenticated Modal CLI."""

    def create(self) -> ModalProxyCredentials:
        """Run a compatible Modal CLI and parse its machine-readable response."""
        commands = self._candidate_commands()
        if not commands:
            raise RuntimeError(
                "No Modal CLI is available. Install uv or Modal, then run `modal setup`."
            )
        for command_index, command in enumerate(commands):
            result = self._run_command(command)
            if result.returncode == 0:
                return self._parse_credentials(result.stdout)
            if command_index + 1 < len(commands) and self._is_unsupported_cli(result):
                logger.info(
                    "Trying a current Modal CLI because the installed CLI lacks proxy-token support."
                )
                continue
            detail = self._safe_error_detail(result.stderr or result.stdout)
            raise RuntimeError(f"Modal proxy-token creation failed: {detail}")
        raise RuntimeError(
            "Modal proxy-token creation failed before returning credentials."
        )

    @staticmethod
    def _candidate_commands() -> list[list[str]]:
        """Return installed and uv-isolated Modal CLI commands in preference order."""
        suffix = ["workspace", "proxy-tokens", "create", "--json"]
        commands: list[list[str]] = []
        modal_command = shutil.which("modal")
        if modal_command:
            commands.append([modal_command, *suffix])
        uvx_command = shutil.which("uvx")
        if uvx_command:
            commands.append([uvx_command, "--from", "modal>=1.5.4", "modal", *suffix])
        if not modal_command:
            commands.insert(0, [sys.executable, "-m", "modal", *suffix])
        return commands

    @staticmethod
    def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Execute a token-creation command without invoking a shell."""
        try:
            return subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=_TOKEN_CREATION_TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "The Modal proxy-token creation command could not complete."
            ) from exc

    @staticmethod
    def _is_unsupported_cli(result: subprocess.CompletedProcess[str]) -> bool:
        """Return whether the failure indicates an absent or outdated Modal CLI."""
        output = f"{result.stdout}\n{result.stderr}".lower()
        return (
            "no such command 'workspace'" in output
            or "no module named 'modal'" in output
        )

    @staticmethod
    def _parse_credentials(output: str) -> ModalProxyCredentials:
        """Parse the CLI's JSON response and validate the proxy-token prefixes."""
        try:
            payload = json.loads(output)
            credentials = ModalProxyCredentials(
                key=str(payload["Modal-Key"]),
                secret=str(payload["Modal-Secret"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Modal created a token but returned an unrecognized response; the secret was not stored."
            ) from exc
        credentials.validate()
        return credentials

    @staticmethod
    def _safe_error_detail(output: str) -> str:
        """Return a bounded CLI error without echoing possible token material."""
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        safe_lines = [line for line in lines if "wk-" not in line and "ws-" not in line]
        return " ".join(safe_lines)[-1000:] or "the CLI returned no diagnostic output"


class ModalCredentialResolver:
    """Resolve Modal endpoint credentials from environment, vault, or the CLI."""

    def __init__(self, store: CredentialStore, creator: ProxyTokenCreator) -> None:
        """Configure secure storage and token creation implementations."""
        self._store = store
        self._creator = creator

    def resolve(self) -> ModalProxyCredentials:
        """Return credentials, creating and storing a pair only when none exists."""
        environment_credentials = self._from_environment()
        if environment_credentials is not None:
            return environment_credentials
        with _TOKEN_CREATION_LOCK:
            environment_credentials = self._from_environment()
            if environment_credentials is not None:
                return environment_credentials
            stored_credentials = self._store.load()
            if stored_credentials is not None:
                return stored_credentials
            self._store.ensure_writable()
            logger.info("No Modal proxy token found; creating one with the Modal CLI.")
            created_credentials = self._creator.create()
            self._store.save(created_credentials)
            return created_credentials

    @staticmethod
    def _from_environment() -> ModalProxyCredentials | None:
        """Read the authoritative MODAL_KEY and MODAL_SECRET environment pair."""
        key = os.getenv(MODAL_KEY_ENV, "").strip()
        secret = os.getenv(MODAL_SECRET_ENV, "").strip()
        if not key and not secret:
            return None
        if not key or not secret:
            raise RuntimeError(
                "Set both MODAL_KEY and MODAL_SECRET, or remove both so ComfyUI can use its credential vault."
            )
        credentials = ModalProxyCredentials(key=key, secret=secret)
        credentials.validate()
        return credentials


def _normalize_endpoint_url(endpoint_url: str) -> str:
    """Validate a Modal Direct URL and return its origin without an API suffix."""
    value = endpoint_url.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Modal endpoint URL must be an absolute HTTPS URL.")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname != "modal.direct" and not hostname.endswith(".modal.direct"):
        raise ValueError("Modal endpoint URL must use a modal.direct hostname.")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError(
            "Modal endpoint URL cannot contain credentials or a nonstandard port."
        )
    path = parsed.path.rstrip("/")
    allowed_paths = ("", "/v1", "/v1/chat/completions")
    if path not in allowed_paths or parsed.query or parsed.fragment:
        raise ValueError(
            "Modal endpoint URL must be the endpoint origin, optionally ending in /v1 or /v1/chat/completions."
        )
    return f"https://{hostname}"


def _tensor_image_to_data_uri(image: torch.Tensor) -> str:
    """Encode one ComfyUI image tensor as an inline PNG data URI."""
    normalized = image.detach().to(device="cpu", dtype=torch.float32)
    if normalized.ndim != 3 or normalized.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Each Modal endpoint image must have HWC shape with 1, 3, or 4 channels."
        )
    normalized = torch.nan_to_num(normalized).clamp(0.0, 1.0)
    pixels = normalized.mul(255).round().to(torch.uint8).numpy()
    if pixels.shape[-1] == 1:
        pixels = pixels[:, :, 0]
    pil_image = Image.fromarray(pixels)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_content_parts(images: torch.Tensor | None) -> list[dict[str, Any]]:
    """Convert an optional ComfyUI image batch to Chat Completions content parts."""
    if images is None:
        return []
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(
            "Modal endpoint images must be an HWC image or BHWC image batch."
        )
    return [
        {
            "type": "image_url",
            "image_url": {"url": _tensor_image_to_data_uri(image), "detail": "auto"},
        }
        for image in images
    ]


def _mapping_from_file_input(file_input: Any) -> Mapping[str, Any]:
    """Convert a built-in OpenAI input-file object to a plain mapping."""
    if isinstance(file_input, Mapping):
        return file_input
    model_dump = getattr(file_input, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(
        "Modal endpoint files must come from an OpenAI ChatGPT Input Files node."
    )


def _file_content_part(file_input: Any) -> dict[str, Any]:
    """Translate a ComfyUI OpenAI input file to a Chat Completions file part."""
    payload = _mapping_from_file_input(file_input)
    if payload.get("type") == "file" and isinstance(payload.get("file"), Mapping):
        return dict(payload)
    file_data = payload.get("file_data")
    if not isinstance(file_data, str) or not file_data:
        raise ValueError(
            "Modal endpoint files must contain inline file_data; provider-specific file IDs cannot be reused."
        )
    filename = payload.get("filename")
    safe_filename = str(filename) if filename else "attachment"
    return {
        "type": "file",
        "file": {"filename": safe_filename, "file_data": file_data},
    }


def _file_content_parts(files: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Convert optional file inputs to Chat Completions content parts."""
    if files is None:
        return []
    return [_file_content_part(file_input) for file_input in files]


def _user_content(
    prompt: str,
    images: torch.Tensor | None,
    files: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    """Build one multimodal user-message content list."""
    return [
        {"type": "text", "text": prompt},
        *_image_content_parts(images),
        *_file_content_parts(files),
    ]


async def _read_json_response(response: aiohttp.ClientResponse) -> Mapping[str, Any]:
    """Read one bounded JSON response body."""
    chunks: list[bytes] = []
    byte_count = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        byte_count += len(chunk)
        if byte_count > _MAX_RESPONSE_BYTES:
            raise RuntimeError(
                "Modal endpoint response exceeded the 8 MiB safety limit."
            )
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Modal endpoint returned a non-JSON response.") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Modal endpoint returned a JSON value instead of an object.")
    return payload


def _response_error_message(payload: Mapping[str, Any]) -> str:
    """Extract a bounded provider error message from a response object."""
    error = payload.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
        return str(error["message"])[:2000]
    return json.dumps(payload, ensure_ascii=False)[:2000]


def _chat_response_text(payload: Mapping[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible Chat Completions response."""
    choices = payload.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, str | bytes)
        or not choices
    ):
        raise RuntimeError(
            "Modal endpoint response did not contain a chat-completion choice."
        )
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping) or not isinstance(
        first_choice.get("message"), Mapping
    ):
        raise TypeError(
            "Modal endpoint response choice did not contain an assistant message."
        )
    content = first_choice["message"].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        text_parts = [
            str(part["text"])
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)
    raise RuntimeError("Modal endpoint assistant message did not contain text content.")


class ModalEndpointClient:
    """Async HTTP client for one authenticated Modal inference endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        credentials: ModalProxyCredentials,
        timeout_seconds: int,
    ) -> None:
        """Validate request settings before any credential-bearing network call."""
        self._origin = _normalize_endpoint_url(endpoint_url)
        credentials.validate()
        self._credentials = credentials
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    @property
    def endpoint_hostname(self) -> str:
        """Return the already validated endpoint hostname for safe diagnostics."""
        hostname = urlsplit(self._origin).hostname
        if hostname is None:
            raise RuntimeError("Validated Modal endpoint lost its hostname.")
        return hostname

    def _headers(self) -> dict[str, str]:
        """Return Modal's separate proxy-token authentication headers."""
        return {
            "Content-Type": "application/json",
            "Modal-Key": self._credentials.key,
            "Modal-Secret": self._credentials.secret,
        }

    async def discover_model(self, session: aiohttp.ClientSession) -> str:
        """Select the first model advertised by the endpoint's OpenAI-compatible API."""
        payload = await self._request_json(session, "GET", "/v1/models")
        models = payload.get("data")
        if (
            isinstance(models, Sequence)
            and not isinstance(models, str | bytes)
            and models
        ):
            first_model = models[0]
            if isinstance(first_model, Mapping) and isinstance(
                first_model.get("id"), str
            ):
                return str(first_model["id"])
        raise RuntimeError(
            "Modal endpoint did not advertise a model at /v1/models; enter its Hugging Face model ID."
        )

    async def complete(
        self,
        prompt: str,
        model: str,
        images: torch.Tensor | None,
        files: Sequence[Any] | None,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Submit one non-streaming multimodal Chat Completions request."""
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            resolved_model = model.strip() or await self.discover_model(session)
            payload = self._request_payload(
                prompt,
                resolved_model,
                images,
                files,
                system_prompt,
                max_tokens,
                temperature,
            )
            response = await self._request_json(
                session, "POST", "/v1/chat/completions", payload
            )
        return _chat_response_text(response)

    @staticmethod
    def _request_payload(
        prompt: str,
        model: str,
        images: torch.Tensor | None,
        files: Sequence[Any] | None,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible non-streaming chat request."""
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {"role": "user", "content": _user_content(prompt, images, files)}
        )
        return {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Make one credential-bearing request without following redirects."""
        async with session.request(
            method,
            f"{self._origin}{path}",
            headers=self._headers(),
            json=payload,
            allow_redirects=False,
        ) as response:
            response_payload = await _read_json_response(response)
            if 200 <= response.status < 300:
                return response_payload
            detail = _response_error_message(response_payload)
            raise RuntimeError(
                f"Modal endpoint returned HTTP {response.status}: {detail}"
            )


def _modal_endpoint_primary_inputs() -> list[io.Input]:
    """Return the primary prompt and multimodal input declarations."""
    return [
        io.String.Input(
            "prompt",
            default="",
            multiline=True,
            tooltip="Text input sent to the hosted model.",
        ),
        io.String.Input(
            "endpoint_url",
            default="",
            tooltip="Modal Direct endpoint URL, without or with the /v1 suffix.",
        ),
        io.String.Input(
            "model",
            default="",
            tooltip=(
                "Base or custom Hugging Face model ID. Leave blank to use the first model "
                "advertised by the endpoint."
            ),
        ),
        io.Image.Input(
            "images",
            optional=True,
            tooltip="Optional image batch for a vision-capable hosted model.",
        ),
        io.Custom("OPENAI_INPUT_FILES").Input(
            "files",
            optional=True,
            tooltip="Optional files from the built-in OpenAI ChatGPT Input Files node.",
        ),
    ]


def _modal_endpoint_advanced_inputs() -> list[io.Input]:
    """Return optional generation and timeout input declarations."""
    return [
        io.String.Input(
            "system_prompt",
            default="",
            multiline=True,
            optional=True,
            advanced=True,
            tooltip="Optional system instructions for the hosted model.",
        ),
        io.Int.Input(
            "max_tokens",
            default=4096,
            min=1,
            max=131072,
            advanced=True,
            tooltip="Maximum number of generated tokens.",
        ),
        io.Float.Input(
            "temperature",
            default=0.7,
            min=0.0,
            max=2.0,
            step=0.05,
            advanced=True,
            tooltip="Sampling temperature.",
        ),
        io.Int.Input(
            "timeout_seconds",
            default=600,
            min=1,
            max=3600,
            advanced=True,
            tooltip="Total request timeout, including a possible cold start.",
        ),
    ]


def _modal_endpoint_inputs() -> list[io.Input]:
    """Return all user-facing V3 inputs for the endpoint chat node."""
    return [*_modal_endpoint_primary_inputs(), *_modal_endpoint_advanced_inputs()]


class ModalEndpointChat(io.ComfyNode):
    """Generate text with an OpenAI-compatible Modal hosted model endpoint."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose inputs aligned with ComfyUI's built-in OpenAI ChatGPT node."""
        return io.Schema(
            node_id=MODAL_ENDPOINT_CHAT_NODE_ID,
            display_name="Modal Endpoint Chat",
            category="Modal/text",
            essentials_category="Text Generation",
            description=(
                "Generate text with a Modal hosted model endpoint. Supports prompts, image batches, "
                "and files from OpenAI ChatGPT Input Files."
            ),
            inputs=_modal_endpoint_inputs(),
            outputs=[io.String.Output(display_name="response")],
            not_idempotent=True,
            is_experimental=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        endpoint_url: str,
        model: str = "",
        images: torch.Tensor | None = None,
        files: Sequence[Any] | None = None,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout_seconds: int = 600,
    ) -> io.NodeOutput:
        """Resolve credentials and make one Modal endpoint inference call."""
        if not prompt and images is None and not files:
            raise ValueError(
                "Modal Endpoint Chat requires a prompt, image, or file input."
            )
        resolver = ModalCredentialResolver(
            store=ComfyUISecretManager(),
            creator=ModalCliProxyTokenCreator(),
        )
        credentials = await asyncio.to_thread(resolver.resolve)
        client = ModalEndpointClient(endpoint_url, credentials, timeout_seconds)
        logger.info("Calling Modal hosted-model endpoint %s.", client.endpoint_hostname)
        response = await client.complete(
            prompt=prompt,
            model=model,
            images=images,
            files=files,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return io.NodeOutput(response)


__all__ = [
    "ComfyUISecretManager",
    "ModalCliProxyTokenCreator",
    "ModalCredentialResolver",
    "ModalEndpointChat",
    "ModalEndpointClient",
    "ModalProxyCredentials",
]
