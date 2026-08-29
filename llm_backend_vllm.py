"""vLLM backend for resident multimodal inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Coroutine
import uuid

if __package__:
    from .llm_backend_transformers import (
        _safetensor_shard_count,
        _weight_progress_message,
    )
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
    from .vllm_instrumentation import (
        _install_accurate_triton_compile_listener,
        _vllm_execution_policy,
        _vllm_execution_setting,
    )
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_backend_transformers import (
        _safetensor_shard_count,
        _weight_progress_message,
    )
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
    from vllm_instrumentation import (
        _install_accurate_triton_compile_listener,
        _vllm_execution_policy,
        _vllm_execution_setting,
    )

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024**3
_VLLM_SAFETENSORS_LOAD_STRATEGY = "prefetch"

class _AsyncLoopRunner:
    """Keep one asyncio loop alive for a resident asynchronous vLLM engine."""

    def __init__(self) -> None:
        """Start a daemon thread and wait until its event loop is available."""
        self.loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="modal-llm-vllm-loop",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()

    def _run_loop(self) -> None:
        """Own and run the backend event loop until explicit shutdown."""
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()
        self.loop.close()

    def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Run one coroutine on the resident loop and return its result."""
        if not self._thread.is_alive():
            raise RuntimeError("The resident vLLM event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        """Stop and join the resident event-loop thread."""
        if not self._thread.is_alive():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logger.warning("Resident vLLM event-loop thread did not stop promptly.")


@dataclass(frozen=True)
class _VLLMStreamState:
    """Retain the final cumulative output and request timing boundaries."""

    request_output: Any
    started_at: float
    first_token_at: float | None

class VLLMMultimodalBackend:
    """Run Qwen multimodal checkpoints through an asynchronous vLLM engine."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Load an immutable local snapshot under explicit co-residency budgets."""
        from transformers import AutoProcessor
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        if os.getenv("COMFY_MODAL_REMOTE_WORKER") == "1":
            _install_accurate_triton_compile_listener()

        self.profile = profile
        self.snapshot_path = snapshot_path
        self.execution_setting = _vllm_execution_setting(profile)
        self.execution_mode, self.enforce_eager = _vllm_execution_policy(profile)
        progress_callback(
            LLMProgressEvent(
                stage="processor",
                message="Loading processor",
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
        quantization = str(profile.backend_option("quantization", "")).strip()
        shard_count = _safetensor_shard_count(snapshot_path)
        self._log_engine_configuration(quantization, shard_count)
        progress_callback(
            LLMProgressEvent(
                stage="engine",
                message=_weight_progress_message(shard_count) + " + engine warmup",
                value=0 if shard_count else None,
                maximum=shard_count,
                unit="shards" if shard_count else None,
                indeterminate=True,
            )
        )
        engine_args = self._engine_arguments(AsyncEngineArgs, quantization)
        self.llm = self._start_engine(AsyncLLMEngine, engine_args)
        progress_callback(
            LLMProgressEvent(
                stage="ready",
                message="vLLM engine ready",
                value=shard_count,
                maximum=shard_count,
                unit="shards" if shard_count else None,
            )
        )

    def _log_engine_configuration(
        self,
        quantization: str,
        shard_count: int | None,
    ) -> None:
        """Log the immutable vLLM configuration before its expensive startup."""
        logger.info(
            "Loading asynchronous vLLM profile=%s path=%s quantization=%s "
            "mode=%s enforce_eager=%s safetensors_load_strategy=%s "
            "max_model_len=%d kv_cache_gib=%.1f shards=%s compile_cache=%s.",
            self.profile.profile_id,
            self.snapshot_path,
            quantization or "auto",
            self.execution_mode,
            self.enforce_eager,
            _VLLM_SAFETENSORS_LOAD_STRATEGY,
            int(
                self.profile.backend_option(
                    "max_model_len",
                    self.profile.max_context_tokens,
                )
            ),
            int(self.profile.backend_option("kv_cache_memory_bytes", 0))
            / _BYTES_PER_GIB,
            shard_count,
            os.getenv("VLLM_CACHE_ROOT", "<ephemeral-default>"),
        )

    def _engine_arguments(self, argument_class: Any, quantization: str) -> Any:
        """Build explicit AsyncLLM co-residency arguments for this profile."""
        return argument_class(
            model=str(self.snapshot_path),
            tokenizer=str(self.snapshot_path),
            trust_remote_code=False,
            dtype=self.profile.dtype,
            quantization=quantization or None,
            max_model_len=int(
                self.profile.backend_option(
                    "max_model_len",
                    self.profile.max_context_tokens,
                )
            ),
            kv_cache_memory_bytes=int(
                self.profile.backend_option(
                    "kv_cache_memory_bytes",
                    12 * _BYTES_PER_GIB,
                )
            ),
            enforce_eager=self.enforce_eager,
            safetensors_load_strategy=_VLLM_SAFETENSORS_LOAD_STRATEGY,
            disable_custom_all_reduce=True,
            attention_config={"backend": "TRITON_ATTN"},
            generation_config="vllm",
            limit_mm_per_prompt={"image": self.profile.max_images, "video": 1},
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Return the execution and persistent-cache settings used by vLLM."""
        return {
            "vllm_execution_setting": self.execution_setting,
            "vllm_execution_mode": self.execution_mode,
            "vllm_enforce_eager": self.enforce_eager,
            "vllm_safetensors_load_strategy": _VLLM_SAFETENSORS_LOAD_STRATEGY,
            "vllm_compile_cache_root": os.getenv("VLLM_CACHE_ROOT"),
        }

    def _start_engine(self, engine_class: Any, engine_args: Any) -> Any:
        """Start AsyncLLM and clean up its loop if initialization fails."""
        self._loop_runner = _AsyncLoopRunner()
        engine_created = False
        try:
            engine = self._loop_runner.run(
                self._create_engine(engine_class, engine_args)
            )
            engine_created = True
            return engine
        finally:
            if not engine_created:
                self._loop_runner.close()

    @staticmethod
    async def _create_engine(engine_class: Any, engine_args: Any) -> Any:
        """Construct AsyncLLM while its long-lived event loop is current."""
        return engine_class.from_engine_args(engine_args)

    def _request(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
    ) -> dict[str, Any]:
        """Build one vLLM prompt with direct in-process multimodal data."""
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
            raise RuntimeError("The multimodal processor did not return a text prompt.")
        multimodal_data: dict[str, Any] = {}
        if prepared_inputs.images:
            multimodal_data["image"] = list(prepared_inputs.images)
        if prepared_inputs.video is not None:
            import numpy as np

            multimodal_data["video"] = np.stack(
                [np.asarray(frame) for frame in prepared_inputs.video.frames]
            )
        request: dict[str, Any] = {"prompt": prompt}
        if multimodal_data:
            request["multi_modal_data"] = multimodal_data
        return request

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Generate one response while streaming cumulative token telemetry."""
        return self._loop_runner.run(
            self._generate_async(prepared_inputs, settings, progress_callback)
        )

    async def _generate_async(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Consume vLLM's async output stream and report every token update."""
        from vllm import SamplingParams
        from vllm.v1.engine.exceptions import EngineDeadError

        self._report_prefill(settings, progress_callback)
        sampling_params = self._sampling_params(SamplingParams, settings)
        request_id = f"modal-llm-{uuid.uuid4().hex}"
        finished = False
        try:
            stream_state = await self._consume_stream(
                prepared_inputs,
                settings,
                sampling_params,
                request_id,
                progress_callback,
            )
            finished = True
        except (EngineDeadError, RuntimeError) as error:
            logger.exception(
                "vLLM generation failed for profile=%s.",
                self.profile.profile_id,
            )
            raise RuntimeError(
                f"vLLM generation failed for profile {self.profile.profile_id!r}: "
                f"{error}"
            ) from None
        finally:
            if not finished:
                await self._abort_request(request_id, EngineDeadError)
        return self._generation_result(stream_state, settings)

    @staticmethod
    def _report_prefill(
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Publish the indeterminate interval before vLLM yields a token."""
        progress_callback(
            LLMProgressEvent(
                stage="prefill",
                message="Prefill / waiting for first token",
                value=0,
                maximum=settings.max_new_tokens,
                unit="tokens",
                indeterminate=True,
            )
        )

    def _sampling_params(
        self,
        parameter_class: Any,
        settings: LLMGenerationSettings,
    ) -> Any:
        """Translate backend-neutral settings into vLLM sampling parameters."""
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        return parameter_class(
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=settings.seed,
            skip_special_tokens=not reasoning_parser.requires_boundary_tokens,
        )

    async def _consume_stream(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        sampling_params: Any,
        request_id: str,
        progress_callback: LLMProgressCallback,
    ) -> _VLLMStreamState:
        """Consume cumulative AsyncLLM outputs and return final stream state."""
        started_at = time.perf_counter()
        first_token_at: float | None = None
        request_output: Any | None = None
        output_stream = self.llm.generate(
            self._request(prepared_inputs, settings),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        async for streamed_output in output_stream:
            if not streamed_output.outputs:
                continue
            request_output = streamed_output
            now = time.perf_counter()
            output_tokens = len(streamed_output.outputs[0].token_ids)
            if output_tokens > 0 and first_token_at is None:
                first_token_at = now
            self._report_token_progress(
                settings,
                output_tokens,
                started_at,
                first_token_at,
                now,
                progress_callback,
            )
        if request_output is None or not request_output.outputs:
            raise RuntimeError("vLLM returned no generation candidate.")
        return _VLLMStreamState(
            request_output=request_output,
            started_at=started_at,
            first_token_at=first_token_at,
        )

    @staticmethod
    def _report_token_progress(
        settings: LLMGenerationSettings,
        output_tokens: int,
        started_at: float,
        first_token_at: float | None,
        now: float,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Publish one cumulative token count with live timing telemetry."""
        elapsed_seconds = now - started_at
        progress_callback(
            LLMProgressEvent(
                stage="generating",
                message="Generating",
                value=output_tokens,
                maximum=settings.max_new_tokens,
                unit="tokens",
                elapsed_seconds=elapsed_seconds,
                time_to_first_token_seconds=(
                    first_token_at - started_at if first_token_at is not None else None
                ),
                tokens_per_second=(
                    output_tokens / elapsed_seconds
                    if output_tokens > 0 and elapsed_seconds > 0
                    else None
                ),
            )
        )

    async def _abort_request(
        self, request_id: str, engine_error: type[Exception]
    ) -> None:
        """Best-effort abort one request without masking its original failure."""
        try:
            await self.llm.abort(request_id)
        except (engine_error, RuntimeError) as abort_error:
            logger.debug(
                "Unable to abort failed vLLM request %s: %s",
                request_id,
                abort_error,
            )

    def _generation_result(
        self,
        stream_state: _VLLMStreamState,
        settings: LLMGenerationSettings,
    ) -> BackendGenerationResult:
        """Convert the final cumulative vLLM output into the backend contract."""
        request_output = stream_state.request_output
        candidate = request_output.outputs[0]
        output_tokens = len(candidate.token_ids)
        completed_at = time.perf_counter()
        elapsed_seconds = completed_at - stream_state.started_at
        time_to_first_token_seconds = (
            stream_state.first_token_at - stream_state.started_at
            if stream_state.first_token_at is not None
            else None
        )
        tokens_per_second = (
            output_tokens / elapsed_seconds if elapsed_seconds > 0 else None
        )
        native_reasoning = getattr(candidate, "reasoning", None)
        if native_reasoning is None:
            native_reasoning = getattr(candidate, "reasoning_content", None)
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        reasoning_output = reasoning_parser.extract(
            str(candidate.text),
            candidate.token_ids,
            native_reasoning=(
                str(native_reasoning) if native_reasoning is not None else None
            ),
        )
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=len(request_output.prompt_token_ids),
            output_tokens=output_tokens,
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
            time_to_first_token_seconds=time_to_first_token_seconds,
            tokens_per_second=tokens_per_second,
        )

    def unload(self) -> None:
        """Shut down the vLLM engine and release its CUDA allocations."""
        import torch

        llm = self.llm
        self.llm = None
        self.processor = None
        if llm is not None:
            self._loop_runner.run(self._shutdown_engine(llm))
        self._loop_runner.close()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    async def _shutdown_engine(llm: Any) -> None:
        """Shut down AsyncLLM on the event loop that owns its output task."""
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
            await asyncio.sleep(0)



