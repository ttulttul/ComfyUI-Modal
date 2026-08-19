# Local And Modal Resident LLM Architecture

The `Modal LLM` V3 node runs multimodal language-model inference in the current ComfyUI execution process. An unmarked node runs locally on Apple Silicon through MLX-VLM. A node marked `Run on Modal` runs in the persistent RemoteEngine process on the workflow's selected Modal GPU. Neither target creates a separate HTTP inference service.

## Target Selection And Request Path

1. The node exposes text, IMAGE, native VIDEO, and `OPENAI_INPUT_FILES` inputs plus the existing `Run on Modal` toggle.
2. Queue rewriting leaves an unmarked node in the local graph. Marking it places the node in a normal remote component.
3. Both targets inspect a Hugging Face repository and `config.json` before weight download, validate the target-specific architecture and quantization policy, and pin the exact commit in a content-addressed schema-v2 profile.
4. Local execution downloads the approved snapshot beneath `<ComfyUI models>/modal_llm` and loads it through pinned MLX-VLM. Modal execution uses the CPU-only ModelStager and shared Modal Volume before allocating a GPU worker.
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

## Progress And Cancellation

Both staging paths pass a custom progress adapter into Hugging Face snapshot download. Local and remote execution emit the same typed stages for profile inspection, input preparation, memory admission, processor/model loading, and token generation. The standard ComfyUI progress bar therefore works locally, while the existing streamed remote event path continues to feed Modal overlays. Work without a real completion fraction remains indeterminate.

Each generated token checks ComfyUI's interruption hook through the node progress callback. Remote vLLM also aborts its active async request before propagating cancellation. Final time-to-first-token and throughput are retained in `metadata_json`.

## Live Validation

On 2026-08-19, Apple-local validation on a 64 GiB Mac loaded the exact `mlx-community/Qwen3.5-2B-4bit` revision `674aaa7240b91e8012fcad5d791b7dfe5ba90207`, produced the requested text response, accepted a real `[1,32,32,3]` ComfyUI IMAGE tensor and identified its dominant red colour, then reported a resident cache hit on the second request. The text pass generated at approximately 193 tokens per second. This validates one representative 4-bit MLX model, not every model that fits in unified memory.

The same date's Modal canaries validated generated vLLM FP8, Transformers, and vLLM ModelOpt FP4 profiles on B300 and RTX PRO 6000 workers. Model-specific capacity and compatibility remain separate from backend availability on both targets.
