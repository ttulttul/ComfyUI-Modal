# Resident Modal LLM Architecture

Modal LLM runs multimodal language-model inference inside the persistent RemoteEngine process. It is not an HTTP client and does not create an independently scaling inference service.

## Request Path

1. The V3 node exposes text, IMAGE, native VIDEO, and OPENAI_INPUT_FILES inputs.
2. Queue rewriting places the marked node in a normal remote component.
3. The local dispatcher scans the complete payload, including split phases, for curated profiles or Hugging Face model IDs.
4. For a model ID, the CPU resolver inspects Hub metadata and `config.json`, validates architecture, quantization, safetensors inventory, security status, and download budget, then pins the exact commit in a content-addressed schema-v2 manifest.
5. The CPU ModelStager downloads only the approved revision-pinned safetensors, processor, and tokenizer assets to the shared Modal Volume.
6. The stager writes a completion marker and commits the Volume.
7. The dispatcher rewrites the model ID to the immutable generated profile ID, and the GPU worker reloads that exact Volume revision before executing the component.
8. The node converts ComfyUI tensors directly to processor inputs, samples video frames uniformly, and extracts bounded text from supported files.
9. ResidentLLMManager reuses or loads the profile, generates under a process lock, and reports progress and cancellation.
10. A model-aware output parser separates generated reasoning from final content using exact token boundaries, with a native engine field taking precedence when available.
11. The clean response, JSON telemetry, and separate reasoning string return through the normal Modal-Sync node-output transport.

## Residency And Memory

The manager is module-global in the warm RemoteEngine process. Its LRU is keyed by immutable profile ID and retains up to COMFY_MODAL_LLM_MAX_RESIDENT_MODELS.

Before loading a new LLM, the manager:

1. asks ComfyUI's memory manager to release idle managed models if needed;
2. checks torch.cuda.mem_get_info();
3. evicts least-recently used LLMs until the profile estimate plus reserve fits;
4. fails with measured free/total VRAM rather than attempting an unsafe load.

ComfyUI sees the LLM's real CUDA allocation when calculating free memory, even though it does not own the Transformers object. The configurable reserve gives subsequent image/video nodes headroom. Executions are serialized within a worker; co-residency does not promise simultaneous CUDA kernels.

## Security And Reproducibility

- Repository IDs are metadata-inspected on CPU before any weight download or GPU allocation.
- Every profile pins a 40-character repository commit.
- Generated profile IDs include the SHA-256 digest of all runtime-defining fields.
- `trust_remote_code` must remain false.
- The remote model loader uses `local_files_only=true`.
- The CPU stager is the only network download path.
- Snapshot allow-patterns omit Python source and pickle weight formats.
- Text and PDF input sizes are bounded before prompt construction.
- The compatibility policy participates in the remote runtime fingerprint.

Compatibility-policy changes remain deployment changes. New repositories using a reviewed architecture and quantization can generate immutable profiles without changing the image.

## Current Backend Boundary

The compatibility registry selects Transformers for Muse-Glimmer and vLLM for Qwen3.5. Qwen block-FP8 and NVIDIA ModelOpt NVFP4 checkpoints share the vLLM adapter. Generated vLLM profiles set an explicit KV-cache byte budget and a conservative 32K default context instead of allowing the serving engine to reserve nearly all available GPU memory.

Reasoning extraction is deliberately model-aware and backend-neutral:

- A profile selects a top-level immutable `reasoning_parser`; known architectures provide a compatibility fallback for previously generated profiles.
- Each backend supplies raw generated token IDs and any native separated reasoning field to the same parser contract.
- A native field wins when an engine supplies one. This supports future engine APIs without changing the node contract.
- Transformers generation has no standard separated reasoning result, so reasoning models retain boundary tokens until the parser splits the token sequence and decodes each channel with special tokens removed.
- Offline vLLM `LLM.generate` returns completion text and token IDs. Its OpenAI-serving reasoning parser is not the resident node's execution path, so the same token-aware fallback applies.
- Qwen3.5 chat templating explicitly enables thinking, and its parser treats tokens before `</think>` as reasoning. If generation stops before that terminator, the entire partial output is reasoning and `response` remains empty.

The node appends `reasoning` after the pre-existing `response` and `metadata_json` outputs to preserve saved workflow link indices. `output_tokens` remains the total generated count; `reasoning_tokens` counts reasoning content tokens excluding boundary markers.

Every selectable NVIDIA GPU runtime uses Python 3.13 and pins Torch 2.13.0, torchvision 0.28.0, Transformers 5.15.0, and the CUDA 13.0 vLLM 0.27.1 wheel. This covers Modal's T4-through-Blackwell choices and ensures a workflow's GPU price/capacity selection never removes its LLM backend. Python 3.13 is necessary because a FlashInfer communications module imported by vLLM's kernel warmup evaluates `array.array[int]` at runtime. The initial vLLM policy uses a 12 GiB BF16 KV cache, one request at a time, and eager execution to keep graph-capture allocations predictable beside ComfyUI. The worker sets `VLLM_USE_FLASHINFER_SAMPLER=0` and selects Triton full attention so optional FlashInfer sampling and Blackwell TRT-LLM attention kernels do not require an `nvcc` JIT compiler, while avoiding FlashAttention 4's unsupported Qwen3.5 used-sequence prefill at head dimension 256. Qwen's separate GDN kernel remains available. Torch and vLLM imports are validated while building every image; model-specific capacity remains enforced separately from backend availability.

## Live GPU Validation

On 2026-08-19, the generated-profile canary resolved exact Hub revisions, staged immutable snapshots, loaded each model, accepted a real ComfyUI IMAGE input, and generated non-empty output for:

- `orcarouter/Qwen3.8-27B-Uncensored-FP8` at `9228df5c6c9c509e1019f83b4e085cf643118bac` through vLLM FP8;
- `meta-models/Muse-Glimmer-30B` at `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` through Transformers;
- `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4` at `faf7945020c138c8ef864ab1644273f3158f85fa` through vLLM ModelOpt FP4.

The live co-residency canary can select a curated profile or Hub ID through `COMFY_MODAL_LLM_CANARY_PROFILE`. It retains that LLM, executes a real ComfyUI VAE encode on the same single B300 worker, and then requires the second LLM call to report a resident cache hit. The 2026-08-19 run passed with the generated Orcarouter FP8 profile retained across a real Flux VAE encode on one B300 worker.

The same date's isolated RTX PRO 6000 canary rebuilt the universal image from its custom-node requirements, validated the final pinned dependency versions, and passed both the remote runtime handshake and real generated-profile inference for `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4`. This specifically covers the non-B300 path that previously omitted vLLM. It does not imply that every model fits or supports every selectable GPU; profile compatibility and measured-VRAM admission remain independent checks.
