# Local And Modal Resident LLM Architecture

The `Modal LLM` V3 node runs multimodal language-model inference in the current ComfyUI execution process. An unmarked node runs locally on Apple Silicon through MLX-VLM. A node marked `Run on Modal` runs in the persistent RemoteEngine process on the workflow's selected Modal GPU. Neither target creates a separate HTTP inference service.

## Target Selection And Request Path

1. The node exposes text, IMAGE, native VIDEO, and `OPENAI_INPUT_FILES` inputs plus the existing `Run on Modal` toggle.
2. Queue rewriting leaves an unmarked node in the local graph. Marking it places the node in a normal remote component.
3. Both targets inspect a Hugging Face repository and `config.json` before weight download, validate the target-specific architecture and quantization policy, and pin the exact commit in a content-addressed schema-v2 profile.
4. Local execution downloads the approved snapshot beneath `<ComfyUI models>/modal_llm` and loads it through pinned MLX-VLM. Modal execution uses the CPU-only ModelStager and shared Modal Volume before allocating a GPU worker. Snapshot identity is the repository plus exact revision, independent of runtime tuning, so execution-profile changes reuse the same files.
5. The node converts ComfyUI tensors directly to processor inputs, samples video uniformly into timestamped frames, and extracts bounded text from supported files. The local adapter supplies sampled video frames as an ordered image sequence because the shared input layer has already done the sampling and timestamp annotation.
6. A target-specific process-global `ResidentLLMManager` reuses or loads the immutable profile under a process lock.
7. A shared model-aware parser separates generated reasoning from final content using exact token boundaries, with a native engine field taking precedence when available.
8. The clean response, JSON telemetry, and separate reasoning string return through normal ComfyUI node outputs.

An unmarked LLM may sit between remote regions. The planner keeps it local and creates the same transport boundaries it would for any other local node; only the values cross those boundaries.

## Apple-Local Runtime

Local execution currently requires Apple Silicon macOS and exactly `mlx-vlm==0.6.15`. Install it into the interpreter that launches ComfyUI:

```bash
uv pip install --python <comfyui-venv>/bin/python \
  "mlx-vlm==0.6.15" "psutil>=7,<8" \
  "huggingface-hub==1.28.0" "hf-xet==1.6.0"
```

The default snapshot root follows ComfyUI's model directory and registers the `modal_llm` model folder. `COMFY_MODAL_LOCAL_LLM_STORAGE_ROOT` overrides it. The compatibility registry currently permits unquantized or native MLX checkpoints for SmolVLM, Muse-Glimmer, and Qwen3.5 architectures. CUDA-oriented FP8, ModelOpt FP4, and unknown formats are rejected before download; users should choose an unquantized repository or an `mlx-community` conversion.

For Qwen3.5, the per-request `enable_reasoning` control is passed to MLX-VLM's hard `enable_thinking` switch. When enabled, MLX also receives a thinking budget equal to half of `max_new_tokens`; when disabled, the direct-response parser leaves the reasoning output empty. This keeps the local and Modal node contract aligned while ensuring that a verbose small model cannot consume the entire local allowance before emitting a final answer.

## Residency And Memory

The shared manager implements a serialized, immutable-profile LRU for both targets. Before a cold load it asks ComfyUI to release idle managed models, evicts older resident LLMs as needed, and rejects the load when the model estimate plus configured reserve does not fit.

Apple-local admission uses `psutil.virtual_memory().available` and total system unified memory. The node's `local_reserve_free_memory_gb` defaults to 4 GiB, the local LRU defaults to one model, and evictions clear MLX caches. `COMFY_MODAL_LOCAL_LLM_MAX_RESIDENT_MODELS` changes the LRU limit. `COMFY_MODAL_LOCAL_LLM_RESERVE_FREE_GB` supplies the default to programmatic calls that omit a reserve.

Modal admission keeps the existing `torch.cuda.mem_get_info()` path, `reserve_free_vram_gb` control, and `COMFY_MODAL_LLM_MAX_RESIDENT_MODELS` setting. Executions are serialized within each process; retaining more than one model does not promise simultaneous kernels.

Telemetry exposes `execution_target`, `device`, `reasoning_enabled`, generic available/total-memory fields, cache status, resident profiles, and ComfyUI-managed model names. Existing Modal GPU memory keys remain present for backward compatibility.

## Security And Reproducibility

- Repository IDs are metadata-inspected before weight download or accelerator allocation.
- Every generated profile pins a 40-character repository commit and includes the execution target in its content identity.
- Existing Modal generated-profile identities remain unchanged for compatibility; `execution_target=modal` is intentionally omitted from their historical digest shape.
- `trust_remote_code` remains false.
- Snapshot allow-patterns omit Python source and pickle weight formats.
- Both loaders consume only the completed local snapshot path.
- Text and PDF sizes are bounded before prompt construction.
- Public models need no token. Local gated access reads `HF_TOKEN` from ComfyUI's environment; remote gated access reads it from the selected Modal secret collection.
- `hf-xet` is pinned and build-validated in the CPU staging image and is explicit in the Apple-local extra. `huggingface_hub` automatically selects it for Xet-backed repositories and receives `HF_TOKEN` or the legacy `HUGGING_FACE_HUB_TOKEN` when configured.
- Xet adaptive concurrency needs no tuning. `HF_XET_HIGH_PERFORMANCE=1` is reserved for high-bandwidth hosts with at least 64 GiB of RAM and should not be enabled on the default 16 GiB Modal staging worker.

Changing target requires resolving the original Hugging Face reference again. A generated profile created for Modal cannot be silently loaded through MLX, and vice versa.

## Backend Boundary

The Modal compatibility registry selects Transformers for Muse-Glimmer and vLLM for Qwen3.5, including reviewed block-FP8 and NVIDIA ModelOpt NVFP4 checkpoints. The Apple registry selects MLX-VLM for reviewed unquantized or MLX-format SmolVLM, Muse-Glimmer, and Qwen3.5 checkpoints.

Reasoning extraction remains backend-neutral:

- A profile selects an immutable `reasoning_parser`; known architectures provide a compatibility fallback for older profiles.
- Each backend supplies raw generated token IDs and any native separated reasoning field to the same parser contract.
- Each request applies the profile parser when reasoning is enabled or a direct-response parser when it is disabled.
- Transformers and MLX retain reasoning boundary tokens until parsing; resident vLLM uses its cumulative async token stream.
- The node appends `reasoning` after the pre-existing `response` and `metadata_json` outputs, preserving saved workflow link indices.
- `output_tokens` counts all generated tokens and `reasoning_tokens` counts reasoning content without boundary markers.

## Modal vLLM Execution And Disk Caches

`COMFY_MODAL_LLM_VLLM_EXECUTION_MODE` accepts `auto`, `eager`, or `throughput` and defaults to `auto`. A container starts its first distinct workflow with eager mode. When the RemoteEngine sees a second ComfyUI prompt id, it permanently promotes that container to throughput mode. If an eager vLLM engine is resident, the next LLM request unloads and rebuilds it once with `enforce_eager=False`, allowing vLLM to select its hybrid compiled and CUDA-graph path. Persistent compilation artifacts reduce that promotion cost. The manager re-runs ComfyUI memory release after eviction and polls CUDA free memory for at most `COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS`, which defaults to 15 seconds. If the required model-plus-reserve threshold is still unavailable, the worker is marked unsafe for reuse and scheduled for retirement while the local dispatcher retries the same durable invocation once with a fresh worker-affinity identity. A retry that follows auto promotion carries throughput state into the fresh container instead of reverting to eager. Multiple ordinary components, LLM nodes, mapped phases, or durable retries with the same prompt id remain one workflow. Pin `eager` to avoid compilation entirely or `throughput` to compile on the first request. The setting is deployment-scoped rather than part of the model profile, so it cannot create another copy of identical weights. Every setting uses safetensor prefetch for the Modal Volume mount.

Metadata reports `vllm_execution_setting`, the effective `vllm_execution_mode`, `vllm_auto_promoted`, and the bounded `vllm_observed_workflow_count`. Promotion also emits an indeterminate `engine` progress stage before the compiled engine initialization stages.

The weight Volume retains immutable repository revisions. Legacy profile-keyed weight directories are recognized from their completion marker and reused in place so an older deployed app can continue resolving its historical path. A CPU staging call commits the Volume only when it downloaded weights or created a generated manifest.

Compilation artifacts use a separate Modal Volume so weight/custom-node reloads cannot invalidate open JIT cache files. `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH` point into a namespace derived from GPU type, Python version, Torch build, and pinned vLLM package. This allows ordinary source redeployments to reuse compatible compiled artifacts while keeping incompatible accelerator stacks separate. `COMFY_MODAL_LLM_COMPILE_CACHE_VOLUME_NAME` overrides the default `<weight-volume>-llm-compile-cache` name.

`scripts/benchmark_modal_llm.py` creates an isolated, billable benchmark app. A cold cycle stops that app, invokes the real CPU staging and GPU execution path once, then immediately repeats on the resident engine. Multiple cold cycles reuse the persistent disk caches while forcing a new container and engine, separating four states: first compilation, cached compilation on a new container, first model load in a container, and resident model reuse. The JSON report records the exact source and workload alongside vLLM telemetry.

On 2026-08-19, the harness ran `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4` on RTX PRO 6000 with one synthetic image and 128 output tokens. The uncached throughput engine spent 138.0 seconds in `torch.compile`; a later container directly loaded the 81 MB AOT cache and reduced compilation to 22.4 seconds. Two genuine cached cold containers took 133.4 and 174.9 seconds of node-reported load time, versus 93.0 seconds for eager. Resident throughput repeats generated at 38.4-39.1 tokens per second with 49-51 ms TTFT; eager generated at 17.2 tokens per second with 151 ms TTFT. Thus the throughput profile roughly doubled steady-state decode speed but retained a 40-82 second cached cold-load penalty in this sample. The JSON artifacts are the authoritative comparison because image deployment time can be independently cold and is included only in wall time.

## Progress And Cancellation

Both staging paths pass a custom progress adapter into Hugging Face snapshot download. Local and remote execution emit the same typed stages for metadata inspection or curated-profile resolution, snapshot lookup, concurrent-download waiting, download preparation and transfer, input preparation, memory admission, processor/model loading, and token generation. Pre-inference staging events carry their model reference so a multi-node remote component renders them on the matching LLM node rather than on its graph representative. The standard ComfyUI progress bar therefore works locally, while the existing streamed remote event path continues to feed Modal overlays. Work without a real completion fraction remains indeterminate.

Each generated token checks ComfyUI's interruption hook through the node progress callback. Remote vLLM also aborts its active async request before propagating cancellation. Final time-to-first-token and throughput are retained in `metadata_json`.

## Live Validation

On 2026-08-19, Apple-local validation on a 64 GiB Mac loaded the exact `mlx-community/Qwen3.5-2B-4bit` revision `674aaa7240b91e8012fcad5d791b7dfe5ba90207`, produced the requested text response, accepted a real `[1,32,32,3]` ComfyUI IMAGE tensor and identified its dominant red colour, then reported a resident cache hit on the second request. The text pass generated at approximately 193 tokens per second. This validates one representative 4-bit MLX model, not every model that fits in unified memory.

The same date's Modal canaries validated generated vLLM FP8, Transformers, and vLLM ModelOpt FP4 profiles on B300 and RTX PRO 6000 workers. Model-specific capacity and compatibility remain separate from backend availability on both targets.
