# Resident Modal LLM Architecture

Modal LLM runs multimodal language-model inference inside the persistent RemoteEngine process. It is not an HTTP client and does not create an independently scaling inference service.

## Request Path

1. The V3 node exposes text, IMAGE, native VIDEO, and OPENAI_INPUT_FILES inputs.
2. Queue rewriting places the marked node in a normal remote component.
3. The local dispatcher scans the complete payload, including split phases, for fixed curated profile IDs.
4. A CPU-only ModelStager downloads missing revision-pinned Hugging Face snapshots to the shared Modal Volume.
5. The stager writes a completion marker and commits the Volume.
6. The GPU worker reloads that exact volume revision before executing the component.
7. The node converts ComfyUI tensors directly to processor inputs, samples video frames uniformly, and extracts bounded text from supported files.
8. ResidentLLMManager reuses or loads the profile, generates under a process lock, and reports per-token progress and cancellation.
9. The response and JSON telemetry return through the normal Modal-Sync node-output transport.

## Residency And Memory

The manager is module-global in the warm RemoteEngine process. Its LRU is keyed by curated profile ID and retains up to COMFY_MODAL_LLM_MAX_RESIDENT_MODELS.

Before loading a new LLM, the manager:

1. asks ComfyUI's memory manager to release idle managed models if needed;
2. checks torch.cuda.mem_get_info();
3. evicts least-recently used LLMs until the profile estimate plus reserve fits;
4. fails with measured free/total VRAM rather than attempting an unsafe load.

ComfyUI sees the LLM's real CUDA allocation when calculating free memory, even though it does not own the Transformers object. The configurable reserve gives subsequent image/video nodes headroom. Executions are serialized within a worker; co-residency does not promise simultaneous CUDA kernels.

## Security And Reproducibility

- Arbitrary repository IDs are not accepted from workflows.
- Every profile pins a 40-character repository commit.
- trust_remote_code must remain false.
- The remote model loader uses local_files_only=true.
- The CPU stager is the only network download path.
- Text and PDF input sizes are bounded before prompt construction.
- The profile registry participates in the remote runtime fingerprint.

Add a model by reviewing and extending llm_profiles.json, then validate its exact Transformers/CUDA combination on the target GPU. A new profile is a deployment change, not merely a user-provided string.

## Current Backend Boundary

The backend-neutral request and resident-manager boundary permits a later vLLM backend, but the initial backend is Hugging Face Transformers. That choice reuses the image's pinned PyTorch/CUDA stack and supports direct multimodal processor inputs without another server lifecycle. vLLM should be added only after its compiled wheel is proven compatible with the B300 CUDA/PyTorch image and its KV cache has an explicit memory budget.
