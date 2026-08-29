"""llama.cpp server backend for resident multimodal LLM inference."""

from __future__ import annotations

import base64
import gc
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

if __package__:
    from .llm_inputs import multimodal_messages as _multimodal_messages
    from .llm_profiles import LLMModelProfile
    from .llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from .llm_types import (
        BackendGenerationResult,
        LLMGenerationSettings,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
    )
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_inputs import multimodal_messages as _multimodal_messages
    from llm_profiles import LLMModelProfile
    from llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from llm_types import (
        BackendGenerationResult,
        LLMGenerationSettings,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
    )

logger = logging.getLogger(__name__)

class LlamaCppServerBackend:
    """Run one curated GGUF model through a resident llama.cpp CUDA server."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Start a private loopback server for one immutable GGUF artifact."""
        from transformers import AutoProcessor

        self.profile = profile
        self.snapshot_path = snapshot_path
        self.model_path = self._model_path()
        self.mmproj_path = self._mmproj_path()
        self.port = self._available_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        log_descriptor, log_path = tempfile.mkstemp(
            prefix="comfy-llama-cpp-", suffix=".log"
        )
        os.close(log_descriptor)
        self._log_path = Path(log_path)
        self._log_file = self._log_path.open("w+b")
        progress_callback(
            LLMProgressEvent(
                stage="processor",
                message="Loading GGUF tokenizer",
                indeterminate=True,
            )
        )
        self.processor = AutoProcessor.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.reasoning_parser: ReasoningOutputParser = create_reasoning_parser(
            profile,
            self.processor.tokenizer,
        )
        progress_callback(
            LLMProgressEvent(
                stage="engine",
                message="Loading GGUF weights into llama.cpp",
                indeterminate=True,
            )
        )
        self.process = self._start_server()
        try:
            self._wait_until_ready()
        except (RuntimeError, TimeoutError):
            self.unload()
            raise
        progress_callback(
            LLMProgressEvent(
                stage="ready",
                message="llama.cpp engine ready",
                value=1,
                maximum=1,
                unit="model",
            )
        )

    def _model_path(self) -> Path:
        """Return the required staged GGUF model path."""
        filename = str(self.profile.backend_option("model_filename", "")).strip()
        if not filename or Path(filename).name != filename:
            raise ValueError(
                f"GGUF profile {self.profile.profile_id!r} has no safe model filename."
            )
        model_path = self.snapshot_path / filename
        if not model_path.is_file():
            raise RuntimeError(f"Staged GGUF model is missing at {model_path}.")
        return model_path

    def _mmproj_path(self) -> Path | None:
        """Return the optional staged multimodal projector path."""
        filename = self.profile.backend_option("mmproj_filename")
        if filename is None:
            return None
        normalized_filename = str(filename).strip()
        if not normalized_filename or Path(normalized_filename).name != normalized_filename:
            raise ValueError(
                f"GGUF profile {self.profile.profile_id!r} has no safe multimodal "
                "projector filename."
            )
        mmproj_path = self.snapshot_path / normalized_filename
        if not mmproj_path.is_file():
            raise RuntimeError(
                f"Staged GGUF multimodal projector is missing at {mmproj_path}."
            )
        return mmproj_path

    @staticmethod
    def _available_port() -> int:
        """Reserve and release one loopback port for the private server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _server_command(self) -> list[str]:
        """Return the bounded CUDA llama-server command for this profile."""
        binary = str(
            self.profile.backend_option("server_binary", "/app/llama-server")
        ).strip()
        context_size = int(
            self.profile.backend_option("context_size", self.profile.max_context_tokens)
        )
        gpu_layers = int(self.profile.backend_option("gpu_layers", 999))
        command = [
            binary,
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(context_size),
            "--parallel",
            "1",
            "--n-gpu-layers",
            str(gpu_layers),
            "--cache-type-k",
            str(self.profile.backend_option("cache_type_k", "q8_0")),
            "--cache-type-v",
            str(self.profile.backend_option("cache_type_v", "q8_0")),
            "--flash-attn",
            "on",
            "--no-webui",
        ]
        if self.mmproj_path is not None:
            command.extend(("--mmproj", str(self.mmproj_path)))
        return command

    def _start_server(self) -> subprocess.Popen[bytes]:
        """Launch llama-server without exposing a network listener."""
        command = self._server_command()
        environment = self._server_environment(command)
        logger.info(
            "Starting llama.cpp profile=%s model=%s context=%s port=%d.",
            self.profile.profile_id,
            self.model_path,
            self.profile.backend_option(
                "context_size", self.profile.max_context_tokens
            ),
            self.port,
        )
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                env=environment,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Unable to start llama.cpp for profile {self.profile.profile_id!r}: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _server_environment(command: Sequence[str]) -> dict[str, str]:
        """Expose shared libraries installed beside the llama-server binary."""
        environment = os.environ.copy()
        binary_directory = str(Path(command[0]).resolve().parent)
        existing_path = environment.get("LD_LIBRARY_PATH", "")
        path_entries = [entry for entry in existing_path.split(":") if entry]
        if binary_directory not in path_entries:
            path_entries.insert(0, binary_directory)
        environment["LD_LIBRARY_PATH"] = ":".join(path_entries)
        return environment

    def _log_tail(self, maximum_bytes: int = 8192) -> str:
        """Return the bounded tail of the private server log."""
        try:
            self._log_file.flush()
            with self._log_path.open("rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - maximum_bytes))
                return log_file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _wait_until_ready(self) -> None:
        """Wait until llama.cpp reports that the model is loaded."""
        timeout_seconds = float(
            self.profile.backend_option("server_startup_timeout_seconds", 900)
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"llama.cpp exited with code {return_code} while loading "
                    f"{self.profile.profile_id!r}. Log tail:\n{self._log_tail()}"
                )
            try:
                with urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                    if response.status == 200:
                        return
            except HTTPError as exc:
                if exc.code != 503:
                    raise RuntimeError(
                        f"llama.cpp health check failed with HTTP {exc.code}."
                    ) from exc
            except (URLError, TimeoutError, OSError):
                pass
            time.sleep(0.25)
        raise TimeoutError(
            f"llama.cpp did not load profile {self.profile.profile_id!r} within "
            f"{timeout_seconds:.0f} seconds. Log tail:\n{self._log_tail()}"
        )

    def _prompt(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
    ) -> str:
        """Render one text-only chat prompt with the pinned tokenizer template."""
        if prepared_inputs.images or prepared_inputs.video is not None:
            raise ValueError(f"GGUF profile {self.profile.profile_id!r} is text-only.")
        prompt = self.processor.apply_chat_template(
            _multimodal_messages(prepared_inputs),
            add_generation_prompt=True,
            tokenize=False,
            **reasoning_chat_template_kwargs(
                self.profile,
                settings.enable_reasoning,
            ),
        )
        if not isinstance(prompt, str):
            raise RuntimeError("The GGUF tokenizer did not return a text prompt.")
        return prompt

    @staticmethod
    def _image_data_uri(image: Image.Image) -> str:
        """Encode one normalized image for llama.cpp's private chat endpoint."""
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _chat_messages(
        self,
        prepared_inputs: PreparedLLMInputs,
    ) -> list[dict[str, Any]]:
        """Return OpenAI-compatible text and image chat messages."""
        messages: list[dict[str, Any]] = []
        if prepared_inputs.system_prompt:
            messages.append(
                {"role": "system", "content": prepared_inputs.system_prompt}
            )
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_data_uri(image)},
            }
            for image in prepared_inputs.images
        ]
        content.append({"type": "text", "text": prepared_inputs.prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _post_json(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Submit one JSON request to the private llama.cpp server."""
        request = Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama.cpp request to {endpoint} failed with HTTP {exc.code}: "
                f"{error_body}"
            ) from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"llama.cpp request to {endpoint} failed: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"llama.cpp request to {endpoint} returned a non-object response."
            )
        return decoded

    def _completion(
        self, payload: Mapping[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Submit one non-streaming completion request to the private server."""
        return self._post_json("/completion", payload, timeout_seconds)

    def _chat_completion(
        self, payload: Mapping[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Submit one multimodal OpenAI-compatible chat request."""
        return self._post_json("/v1/chat/completions", payload, timeout_seconds)

    @staticmethod
    def _chat_response_content(response: Mapping[str, Any]) -> tuple[str, int, int]:
        """Extract text and token counts from one chat-completion response."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp chat completion returned no choices.")
        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp chat completion omitted message content.")
        usage = response.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else {}
        return (
            content,
            int(usage_mapping.get("prompt_tokens", 0)),
            int(usage_mapping.get("completion_tokens", 0)),
        )

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Generate one response and report bounded llama.cpp telemetry."""
        progress_callback(
            LLMProgressEvent(
                stage="prefill",
                message="Prefill / waiting for llama.cpp",
                value=0,
                maximum=settings.max_new_tokens,
                unit="tokens",
                indeterminate=True,
            )
        )
        started_at = time.perf_counter()
        timeout_seconds = max(900.0, settings.max_new_tokens * 10.0)
        if prepared_inputs.images:
            response = self._chat_completion(
                {
                    "model": self.profile.profile_id,
                    "messages": self._chat_messages(prepared_inputs),
                    "max_tokens": settings.max_new_tokens,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "seed": settings.seed,
                    "repeat_penalty": 1.05,
                    "chat_template_kwargs": reasoning_chat_template_kwargs(
                        self.profile,
                        settings.enable_reasoning,
                    ),
                },
                timeout_seconds=timeout_seconds,
            )
            content, input_tokens, output_tokens = self._chat_response_content(response)
            output_token_ids: list[int] = []
        else:
            response = self._completion(
                {
                    "prompt": self._prompt(prepared_inputs, settings),
                    "n_predict": settings.max_new_tokens,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "seed": settings.seed,
                    "repeat_penalty": 1.05,
                    "cache_prompt": True,
                    "return_tokens": True,
                },
                timeout_seconds=timeout_seconds,
            )
            content = response.get("content")
            tokens = response.get("tokens")
            if not isinstance(content, str) or not isinstance(tokens, list):
                raise RuntimeError(
                    "llama.cpp completion omitted string content or generated tokens."
                )
            output_token_ids = [int(token) for token in tokens]
            output_tokens = len(output_token_ids)
            input_tokens = int(response.get("tokens_evaluated", 0))
        completed_at = time.perf_counter()
        elapsed_seconds = completed_at - started_at
        timings = response.get("timings")
        timing_mapping = timings if isinstance(timings, Mapping) else {}
        tokens_per_second = timing_mapping.get("predicted_per_second")
        resolved_tokens_per_second = (
            float(tokens_per_second)
            if isinstance(tokens_per_second, int | float)
            else (output_tokens / elapsed_seconds if elapsed_seconds > 0 else None)
        )
        progress_callback(
            LLMProgressEvent(
                stage="generating",
                message="Generated with llama.cpp",
                value=output_tokens,
                maximum=settings.max_new_tokens,
                unit="tokens",
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=resolved_tokens_per_second,
            )
        )
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        reasoning_output = reasoning_parser.extract(content, output_token_ids)
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
            tokens_per_second=resolved_tokens_per_second,
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Return the GGUF artifact and server configuration."""
        return {
            "llama_cpp_model_filename": self.model_path.name,
            "llama_cpp_mmproj_filename": (
                self.mmproj_path.name if self.mmproj_path is not None else None
            ),
            "llama_cpp_context_size": int(
                self.profile.backend_option(
                    "context_size",
                    self.profile.max_context_tokens,
                )
            ),
            "llama_cpp_cache_type_k": self.profile.backend_option(
                "cache_type_k", "q8_0"
            ),
            "llama_cpp_cache_type_v": self.profile.backend_option(
                "cache_type_v", "q8_0"
            ),
        }

    def unload(self) -> None:
        """Stop the private server and remove its bounded diagnostic log."""
        process = getattr(self, "process", None)
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()
            self._log_file = None
        log_path = getattr(self, "_log_path", None)
        if isinstance(log_path, Path):
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
        self.processor = None
        gc.collect()



