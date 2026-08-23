# ComfyUI Modal-Sync

> [!WARNING]
> This project is still alpha. Expect missing features, rough edges, and breaking changes.

ComfyUI Modal-Sync is a ComfyUI custom node extension for running selected parts of a workflow through Modal or user-managed Docker hosts reached over SSH. You mark nodes with `Run on Modal` (the legacy-compatible remote marker), or enable workflow-wide automatic placement; the extension rewrites the queued prompt into transport-aware remote components, syncs required assets, and returns remote outputs and generated files to the local ComfyUI graph.

## Overview

Modal-Sync provides:

- a ComfyUI frontend extension with a `Run on Modal` toggle, workflow-level GPU selection, and remote execution overlays
- a resident `Modal LLM` node for text, image, video, and bounded file understanding on either Apple Silicon or the same Modal GPU worker as ComfyUI
- a `Modal Endpoint Chat` node for prompt, image, and file inference through Modal hosted-model endpoints
- a queue route at `/modal/queue_prompt` that intercepts normal prompt submission
- queue-time graph partitioning and proxy-node rewrite for selected remote regions
- local in-process execution mode for development and tests
- Modal-backed remote execution with deployed-app lookup and first-run auto-deploy
- self-hosted execution through persistent, fingerprinted Docker workers over SSH
- capability-aware, cost-aware provider and host selection with workflow-level policy controls
- model asset sync and optional `custom_nodes/` package sync
- streamed remote status, progress, preview, and UI payload relay

Remote execution is component-based. A remote component may contain several marked nodes, and Modal-Sync may expand the component upstream when a marked node depends on a non-transportable ComfyUI runtime object.

## Quick Start

Install this repository under ComfyUI's `custom_nodes/` directory:

```bash
cd ~/git/ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Modal
```

Restart ComfyUI. On startup it should load:

- [`web/modal_toggle.js`](web/modal_toggle.js), the frontend toggle and overlay extension
- [`web/remote_environments.js`](web/remote_environments.js), the SSH host manager
- [`api_intercept.py`](api_intercept.py), the queue rewrite route
- [`modal_executor_node.py`](modal_executor_node.py), the internal proxy node registry
- [`modal_llm_node.py`](modal_llm_node.py), the resident multimodal LLM node

Start with local mode while building or debugging workflows:

```bash
export COMFY_MODAL_EXECUTION_MODE=local
```

Local mode still exercises marker resolution, prompt rewrite, sync planning, serialization, and proxy execution, but the rewritten component runs in the local ComfyUI process instead of in Modal.

Use remote mode with working Modal credentials:

```bash
export COMFY_MODAL_EXECUTION_MODE=remote
```

Remote workers receive workflow API keys through a named Modal secret collection. Create the
default `comfy` collection from ComfyUI's local `.env` file once; the file itself is not copied
into the image or container:

```bash
<comfyui-venv>/bin/python -m modal secret create comfy \
  --from-dotenv /path/to/ComfyUI/.env
```

To use a different collection, set its name before starting ComfyUI:

```bash
export COMFY_MODAL_SECRET_NAME=my-comfy-secrets
export COMFY_MODAL_EXECUTION_MODE=remote
```

Every key in that Modal collection is available to remote custom nodes through `os.environ`.
The selected collection must already exist in the active Modal environment before deployment.

When ComfyUI starts in remote mode, Modal-Sync checks whether the supported Modal SDK is importable. If it is missing, the extension installs the pinned `modal==1.4.2` package into the exact Python interpreter running ComfyUI. The installer prefers `uv pip` and falls back to `python -m pip`; startup logs report detection, the selected command, and the result. If neither installer is available or installation fails, startup continues with the existing local-mirror fallback and logs an actionable error.

Modal authentication remains user-managed. Run `<comfyui-venv>/bin/python -m modal setup` when the current user does not already have working credentials. To install the SDK manually before startup, run `uv pip install --python <comfyui-venv>/bin/python "modal==1.4.2"`.

Each ComfyUI environment receives its own Modal app name on first startup. Modal-Sync generates 64 random bits, persists them as lowercase hexadecimal in `<ComfyUI user directory>/.comfy-modal-sync-instance-id`, and uses an unpadded URL-safe Base64 encoding in names such as `comfy-modal-sync-R7uqpQZ6S1A`. The identity file is published atomically, so concurrent first startups converge on one name. It lives in ComfyUI's effective user directory and therefore follows `--user-directory`; keep the file when upgrading if the deployment and its state should remain associated with the same ComfyUI environment. `COMFY_MODAL_APP_NAME` remains an authoritative override for deliberately shared or externally managed deployments.

For repository development, `uv sync --extra remote --extra local-apple --group test` installs the pinned Modal SDK plus the guarded Apple-local backend where its platform markers apply. Remote mode uses the stable cloud entrypoint in [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py). On first use, Modal-Sync can auto-deploy the configured Modal app if it does not exist.

### Self-hosted SSH Docker environments

Self-hosted execution reuses the same component planner, asset synchronization, custom-node bundle, serialization, progress stream, cancellation, remote-session, and output-artifact paths as Modal. The provider-specific layer is limited to host discovery, OCI image construction, named-volume storage, warm-worker lifecycle, and a framed binary relay over SSH.

Open **Settings → Remote Execution: SSH environments**, or right-click an eligible node and choose **Remote Execution → Manage SSH hosts…**. Add one or more hosts using an SSH destination or alias that already works from the account running ComfyUI. The extension deliberately stores no password or private-key path; authentication, jump hosts, ports, and key selection belong in the user's normal SSH agent and `~/.ssh/config`.

Each host must provide:

- Linux on `x86_64`
- a non-interactive SSH login with an already trusted host key
- a working Docker CLI and daemon available to that login
- NVIDIA drivers plus Docker's NVIDIA runtime or CDI configuration for GPU work
- outbound package/image access during the first fingerprinted worker build

Use **Save and probe** to discover CPU architecture, RAM, available Docker disk, Docker/runtime state, and every GPU's name, UUID, compute capability, driver, total VRAM, and free VRAM. Use **Build / update worker** to create the host's content-addressed storage volume, stream a minimal local ComfyUI/runtime build context to Docker, and start a persistent worker. This bootstrap is optional: the first assigned component reconciles the same image and worker automatically.

The host form also controls:

- **Cost USD/hour**: converted to per-second cost for planning. Provisioned hosts can report this automatically with the Docker daemon label `comfy.remote.cost-usd-per-second=<value>`; the form value overrides that label. Arbitrary machines expose no universal authoritative amortized or leased price, so leaving both sources blank means “unknown,” not zero.
- **Max workers**: the number of fingerprinted containers the planner may address. Worker indices are distributed across discovered GPU UUIDs.
- **Reserve VRAM GB**: capacity withheld from the compatibility check for desktop or other local use.
- **Tags**, **Enabled**, and **Drain**: placement metadata and safe admission controls. Draining blocks new components without killing a running worker.

**Refresh workers** lists only containers carrying the selected environment's ownership label. **Stop workers** rechecks that label before removing each exact container, so it cannot target unrelated Docker workloads. A source/runtime fingerprint change removes the superseded container for the same worker slot before starting its replacement; named storage is retained.

Workflow policy is stored under `workflow.extra.remote_execution` and travels with the workflow:

- **Modal only** preserves existing behavior and is the default for old workflows.
- **Self-hosted only** requires a ready, compatible SSH host and fails before queueing when none is available.
- **Automatic** considers ready SSH hosts and the selected Modal GPU, rejects hard incompatibilities such as insufficient VRAM or a provider-specific backend, then prefers known lowest predicted cost with deterministic tie-breaking. Provider-constrained nodes form their own acyclic components, and cross-provider outputs are materialized through ComfyUI instead of being hidden in one provider's session storage.
- **Automatically place eligible nodes** makes the planner select the executable prompt without requiring each node toggle. Leave it off to keep manual component boundaries.

Before comparing cost, the planner resolves model files referenced by each remote component and derives conservative memory floors. It budgets the largest primary model that may be resident, adds LoRA/adapter/ControlNet weights, applies 20% weight overhead, and reserves another 4 GiB for activations and runtime state. RAM receives the resident weights plus 4 GiB. Hosts are checked against currently free VRAM and available RAM when the capability probe reports them, falling back to total capacity only when live availability is unknown. An undersized zero-cost host is therefore rejected before any model upload; automatic policy can fall back to a compatible Modal GPU, while self-hosted-only policy returns the per-host memory shortfall.

Images are immutable and named from the local node-pack source, selected ComfyUI source, custom-node requirements, and runtime-shaping settings. Warm containers use a persistent named volume mounted at `/storage`; assets and custom-node bundles are content-addressed, and a remote volume epoch prevents a deleted/recreated volume from reusing stale local index records. Requests and tensor payloads travel as bounded, length-prefixed binary frames through `ssh docker exec`; no daemon TCP port, worker port, or inbound firewall rule is required. Generated ComfyUI files are bundled into the result, integrity-checked, and atomically restored beneath the local output directory just as they are for Modal.

Controller SSH sessions disable forwards inherited from the selected alias, so existing development tunnels cannot collide with worker-management connections. Image builds explicitly load their result into the remote daemon, including when its selected BuildKit builder uses the isolated `docker-container` driver. The worker image includes ComfyUI's top-level Python modules together with its selected package trees, so headless startup can import `nodes`, `execution`, `folder_paths`, and related runtime entry modules. The worker entrypoint keeps Modal-only package initialization lazy, allowing the same source tree to run as a top-level OCI application while ordinary ComfyUI imports retain package-relative behavior. Derived worker images disable health checks inherited from provider base images because readiness is established by the private worker socket rather than a base image's HTTP process. They also expose `/app` through `LD_LIBRARY_PATH`, and the GGUF adapter defensively exposes the selected server binary's directory to its child process, so shared llama.cpp libraries copied beside `llama-server` remain loadable. Custom-node requirements are installed into the image, except packages owned by the immutable Python/CUDA runtime—such as NumPy, OpenCV, Pillow, PyTorch, Triton, and vLLM—whose custom-node constraints are ignored before the final supported pins are applied. A custom node that is genuinely incompatible with those supported versions will fail explicitly during import rather than silently replacing the worker stack.

Enabled hosts are re-probed concurrently immediately before every placement decision, so recently saved free-VRAM and available-RAM values cannot outlive another workload that began moments later. An unreachable host is excluded from new automatic placements, allowing Modal to remain the fallback; self-hosted-only workflows fail with the per-host rejection reason. A broken SSH/Docker relay removes and recreates its assigned worker slot and retries the stable invocation once, while remote application errors are returned without blind retry.

Successful components write a bounded 20-sample timing window to `<ComfyUI user directory>/comfyui-modal/execution-history.sqlite3`. The planner uses the per-environment median for the same workload signature—node types plus cost-shaping inputs such as dimensions, steps, frames, batch size, model profile, and token limit—so a faster but more expensive-per-second GPU can win when its predicted total cost is lower. Until observations exist, planning uses a conservative common 60-second estimate.

Self-hosted worker environment secrets are administrator-managed. The optional **Remote Docker env file** field accepts a safe absolute path that already exists on the SSH host and passes only that path to Docker's `--env-file`; its contents never cross the API or enter the workflow/registry. The extension intentionally does not copy local `.env` values. Modal execution continues to use the named Modal secret collection described above.

The deployed image uses Python 3.13 plus an exact ComfyUI support and CUDA package set, including ComfyUI's current `comfy-aimdo==0.4.13` and `comfy-kitchen==0.2.31` pins and the import-time dependencies used by built-in extras such as Math Expression and GLSL. Python 3.13 is also required by the FlashInfer communications annotations imported during vLLM 0.27.1 kernel warmup. Its headless PromptServer shim instantiates ComfyUI's `NodeReplaceManager`, allowing current built-in and custom extensions to register replacement nodes during remote initialization. Remote prompt executors mirror both legacy scalar and current active/inactive RAM-pressure cache arguments from ComfyUI, and the cache adapter supports both synchronous legacy access and current coroutine-based cache operations. Remote pre-execution validation finalizes V3 dynamic input schemas against the live prompt before checking required sockets, so nodes using `io.Autogrow` accept expanded paths such as `images.image0` exactly as local ComfyUI does. The image's local build context is limited to the ComfyUI source packages, top-level Python modules, and runtime configuration needed by the headless worker; model directories, custom nodes, caches, tests, virtual environments, user data, and unknown top-level directories stay out of the image snapshot. Before every process's first remote invocation, Modal-Sync compares the deployed worker's runtime fingerprint with the local source, ComfyUI source, custom-node requirements, and runtime-shaping settings. A missing or mismatched fingerprint is treated as stale and replaced automatically when `COMFY_MODAL_AUTO_DEPLOY=true`; replacement uses Modal's non-interactive SDK stop API when available and an explicitly confirmed CLI fallback, so app shutdown cannot stall on a hidden terminal prompt.

### Calling A Modal Hosted Model Endpoint

`Modal Endpoint Chat` is a separate V3 node for Modal's hosted-model endpoints. Its layout follows ComfyUI's built-in `OpenAI ChatGPT` node: provide a prompt, an optional `IMAGE` batch, and optional files from `OpenAI ChatGPT Input Files`. Supply a Modal Direct endpoint such as:

```text
https://your-workspace--your-endpoint.us-west.modal.direct
```

The node calls the endpoint's OpenAI-compatible `/v1/chat/completions` API. Enter the base or custom Hugging Face model ID in `model`, or leave it blank to select the first ID advertised by `/v1/models`. Image and file content is sent using the OpenAI Chat Completions multimodal format; the hosted model and serving recipe must support the supplied content types.

Modal endpoints scale to zero. While the first replica is starting, Modal Server returns an empty HTTP 503 response rather than queueing the request. The node treats that response as a cold-start signal and retries model discovery or completion with bounded backoff until the advanced `timeout_seconds` deadline. Other empty or non-JSON responses retain their HTTP status in the ComfyUI error instead of being reported only as a JSON decoding failure.

Authentication is resolved in this order:

1. `MODAL_KEY` and `MODAL_SECRET` from the ComfyUI process environment. These must be proxy-token values with `wk-` and `ws-` prefixes, not Modal API credentials with `ak-` and `as-` prefixes.
2. A previously saved pair in the operating-system credential vault under the `ComfyUI Modal-Sync` service.
3. A new pair created with `modal workspace proxy-tokens create --json`, using a current CLI through `uvx` when the installed Modal CLI is too old.

Automatically created values go directly to the OS vault—Keychain on macOS or the configured native `keyring` backend on other platforms. They are never written into the workflow, ComfyUI settings JSON, logs, or a plaintext file. Because newly created proxy tokens are scoped in RBAC-enabled workspaces, the node also authorizes vault-backed tokens for the advanced `environment` setting before use; it defaults to `main`, treats a blank saved value as `main`, and should be changed when the endpoint is hosted in another Modal environment. A successful association is cached for the ComfyUI process so inference calls do not repeatedly invoke the CLI. Environment-supplied credentials remain administrator-managed and are not modified.

ComfyUI Cloud's Settings → Secrets service is cloud-only; when that service injects `MODAL_KEY` and `MODAL_SECRET` into the process environment, the first resolution path applies. A local/headless installation without a secure keyring backend must supply both environment variables rather than allowing automatic creation.

The CLI must already be authenticated with Modal. Run `<comfyui-venv>/bin/python -m modal setup` if necessary. An owner or manager can also associate an externally managed token manually:

```bash
modal workspace proxy-tokens allow wk-... main
```

For credential safety, the node accepts only HTTPS `modal.direct` origins, refuses redirects, and never exposes its own `Run on Modal` toggle.

### Running A Resident LLM Beside ComfyUI

`Modal LLM` is a V3 multimodal language-model node with two execution targets controlled by the existing node toggle:

- Leave `Run on Modal` disabled to run the node locally on Apple Silicon. The advanced `local_mlx_engine` control defaults to `auto`: registered text-only targets use `mlx-dspark`, while image/video requests and targets without a registered speculative drafter use MLX-VLM. Choose either engine explicitly to override that selection. This is intended for MacBooks whose unified memory can hold an LLM while heavier image or video nodes run on Modal.
- Enable `Run on Modal` to run the node inside the same Modal GPU worker as its surrounding remote component.
- The visible toggle is authoritative at queue time. A disabled LLM node bypasses Modal setup, deployment, asset sync, and remote dispatch entirely, even if an older saved workflow contains stale remote-marker metadata.

The queue planner treats an unmarked Modal LLM as an ordinary local node, including when it sits between two remote regions. Inputs and outputs cross those region boundaries through the existing transport layer; the model itself stays on the Mac.

Apple-local inference requires macOS on Apple Silicon and the optional pinned runtime in the Python environment that launches ComfyUI:

```bash
uv pip install --python <comfyui-venv>/bin/python \
  "mlx-vlm==0.6.15" "mlx-dspark==0.13.1" "psutil>=7,<8" \
  "huggingface-hub==1.28.0" "hf-xet==1.6.0"
```

Repository developers can instead run `uv sync --extra local-apple --extra remote --group test`. Other local platforms receive an actionable error and can enable `Run on Modal`. Local snapshots are stored under `<ComfyUI models>/modal_llm`; set `COMFY_MODAL_LOCAL_LLM_STORAGE_ROOT` to use another writable location.

The node accepts:

- a text prompt and optional system prompt
- an `IMAGE` batch
- one native ComfyUI `VIDEO`, uniformly sampled into timestamped frames
- UTF-8 text or text-based PDF values from `OpenAI ChatGPT Input Files`
- a per-request reasoning toggle for models whose profile supports thinking
- an `auto`, `mlx-vlm`, or `mlx-dspark` local-engine selector
- bounded generation, sampling, seed, video-frame, VRAM-reserve, and residency controls

It returns three strings: the clean final `response`, compact `metadata_json`, and a separate `reasoning` channel. The reasoning output is appended after the two original outputs so existing response and metadata links retain their indices. `enable_reasoning` defaults to on and passes the model's hard thinking switch through the local MLX, remote Transformers, and remote vLLM backends; disabling it requests a direct answer and leaves `reasoning` empty. Models without a configured reasoning parser ignore the switch. For thinking models, `max_new_tokens` covers reasoning and final-response tokens together. MLX caps enabled MLX-VLM thinking at half that budget so it retains room for a final answer, while mlx-dspark and the remote backends retain their model-native thinking behavior. Telemetry includes the selected backend and speculative mode/drafter when applicable, execution target, device, requested reasoning state, token timing and rate, detected reasoning tokens and parser, cold/warm model status, media counts, available memory, resident LLM profiles, and ComfyUI-managed models still resident after generation. The node's ordinary progress panel carries compact stages for profile/input preparation, snapshot download, processor and weight loading, engine warmup where applicable, prefill, and generation. Prompt cancellation remains wired through ComfyUI's interruption hook without a separate inference server.

Modal LLM uses immutable schema-v2 model profiles. Checked-in entries in [`llm_profiles.json`](llm_profiles.json) remain available, while generated profiles content-address the execution target, exact Hugging Face revision, compatibility-policy version, backend, quantization, reasoning parser, context/media limits, expected memory, and remote-code policy. The local registry selects MLX-VLM for reviewed SmolVLM, Muse-Glimmer, and Qwen3.5 architectures; the Modal registry selects Transformers or vLLM for its reviewed CUDA formats. SSH CUDA workers additionally carry a pinned official `llama-server` for curated GGUF profiles that vLLM cannot load. Qwen3.5 profiles select the shared `qwen3` reasoning parser. Unknown or target-incompatible architectures and quantizations fail compatibility inspection before weight download.

Enter either a curated profile name or a Hugging Face `owner/model` ID directly in the node. Use `owner/model@revision` to resolve an explicit branch, tag, or commit. Each target performs metadata and compatibility inspection before downloading safetensors, pins the exact commit in a target-specific generated profile, and reuses its completed snapshot on later executions. New weight directories are keyed by repository and exact revision rather than the full runtime profile, so eager and throughput deployments do not duplicate the same checkpoint. Existing profile-keyed directories are recognized by their completion marker and reused in place, preserving compatibility with older deployed apps. Public models need no token. For gated local models, set `HF_TOKEN` in ComfyUI's environment; for gated Modal models, include it in the selected Modal secret collection; for gated SSH models, expose it through that host's configured Docker environment file.

The curated `huihui-qwen3.8-27b-abliterated-q2-k-gguf` profile pins `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` to one 10.12 GiB `Q2_K` file and its exact 888 MiB BF16 multimodal projector rather than downloading every quant in that repository. It stages separately pinned tokenizer metadata from the source checkpoint, runs text/image/file requests through the private loopback llama.cpp server, and uses an 8K context plus Q8 KV cache under a conservative 16 GiB VRAM floor. Set the node's `reserve_free_vram_gb` low enough for the selected SSH GPU.

mlx-dspark resolution is text-only because its Qwen path intentionally omits the vision tower. Its registered DSpark or DFlash drafter is independently security-checked, pinned to an exact Hub commit, staged beneath the same local model root, and reported in output metadata.

Modal-Sync explicitly installs `hf-xet` beside `huggingface-hub` for Apple-local, Modal, and SSH-host staging. Hugging Face automatically uses the Rust Xet transfer backend for Xet-backed repositories; no alternate download command is required. When `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is present, Modal-Sync passes it to repository inspection, config retrieval, and the snapshot download without logging its value. Xet's adaptive concurrency remains enabled by default. Advanced users may set `HF_XET_HIGH_PERFORMANCE=1` in the matching local, Modal-secret, or SSH Docker environment only on high-bandwidth machines with at least 64 GiB of RAM; the default 16 GiB Modal staging worker is intentionally not configured for that mode.

Before automatic placement, the planner performs metadata-only resolution for raw Hugging Face IDs and applies the generated profile's conservative VRAM and RAM floors, so an inexpensive small GPU cannot win cost ranking for a checkpoint it cannot load. For Modal execution, the local dispatcher then calls the deployed CPU-only `ModelStager`, which writes the immutable snapshot and completion marker to the shared Modal Volume before the selected GPU worker starts. For SSH execution, the controller runs the same provider-neutral resolver and stager inside the selected persistent worker, writes the profile and weights to that host's named Docker volume, streams progress back to ComfyUI, and rewrites the payload to the immutable generated profile before GPU execution. Cancelling during SSH staging stops and later recreates that managed worker while preserving resumable partial files in its named volume. A linked/dynamic remote `model_profile` is rejected because it cannot be staged before GPU allocation. For local execution, resolution and staging happen directly in the ComfyUI process before MLX allocation.

Each target and local engine holds a process-global serialized LRU. A cache hit reuses the processor and weights across node and workflow executions. Switching local engines unloads the other engine's residents so separate caches cannot silently double unified-memory use. Before a cold load, the manager asks ComfyUI to release idle managed models, evicts older resident LLMs if necessary, and enforces the target's reserve: `local_reserve_free_memory_gb` defaults to 4 GiB for macOS unified memory, while `reserve_free_vram_gb` remains the Modal GPU control. Local MLX defaults to one resident model and can be changed with `COMFY_MODAL_LOCAL_LLM_MAX_RESIDENT_MODELS`; `COMFY_MODAL_LOCAL_LLM_RESERVE_FREE_GB` supplies the programmatic default when the runner is called without a node value. All backends pass generated token IDs through the same model-aware reasoning abstraction.

For a single-user B300 session, start conservatively with one container and one active component:

```bash
export COMFY_MODAL_MAX_CONTAINERS=1
export COMFY_MODAL_MAX_INFLIGHT_CALLS=1
export COMFY_MODAL_LLM_MAX_RESIDENT_MODELS=2
export COMFY_MODAL_LLM_RESERVE_FREE_GB=24
export COMFY_MODAL_LLM_VLLM_EXECUTION_MODE=auto
```

The Modal secret collection may include `HF_TOKEN` for gated profiles. Public profiles do not require it. Model staging can take several minutes the first time, but it consumes CPU and network resources rather than billed B300 time; later requests reuse the committed Volume snapshot.

Modal vLLM deployments default to `COMFY_MODAL_LLM_VLLM_EXECUTION_MODE=auto`. The first distinct workflow handled by a container uses eager mode for lower startup latency. When that container receives a second workflow, it unloads an eager resident vLLM engine once and rebuilds it in throughput mode, reusing the persistent compilation cache from then on. After eviction, the manager asks ComfyUI to release memory again and waits up to `COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS` (15 seconds by default) for CUDA free-memory accounting to recover. If the admission threshold is still unmet, Modal retires that dirty worker and retries the invocation once on a fresh affinity identity; an auto-promoted retry remains in throughput mode. Components and ordinary durable retries carrying the same ComfyUI prompt id remain part of one workflow and cannot trigger promotion. Set the execution-mode variable to `eager` or `throughput` to pin either behavior. All settings force safetensor prefetch from the mounted weight Volume.

vLLM, TorchInductor, Triton, CUDA, Torch extension, and Numba compilation artifacts persist in a separate `<weight-volume>-llm-compile-cache` Modal Volume, namespaced by GPU and pinned accelerator runtime. Workers reload that Volume once before runtime initialization, then leave the mounted snapshot unchanged after native compiler artifacts may be memory-mapped. Future-component warmup discovery follows each component's executable dependency closure, so disconnected LLM nodes and serialized workflow-planning metadata cannot load an LLM on an ordinary Comfy worker. Eligible LLM workers execute content-addressed text and vision warmup requests and write completion manifests. Because vLLM starts `EngineCore` with multiprocessing spawn, Modal-Sync replaces the pickled EngineCore entrypoint with a top-level wrapper that installs Triton's cache-aware compilation listener inside the spawned compiler process. The listener logs `cache_hit`, artifact identity, and stage timings for every specialization and publishes a live-process marker. A genuine disk-cache miss advances a process-shared signal and explicitly commits the Volume after either representative warmup or the executable LLM subgraph; persistent hits avoid redundant commits, and missing EngineCore telemetry can never be reported as a hit. Override the Volume name with `COMFY_MODAL_LLM_COMPILE_CACHE_VOLUME_NAME`. Telemetry records the configured setting, effective mode, auto-promotion state, eager flag, load strategy, and compilation-cache root.

## Using It In ComfyUI

Build the workflow normally. Modal-Sync does not replace standard nodes; it adds remote execution controls to the existing graph.

Good remote candidates are nodes that:

- consume large model files
- perform expensive tensor work
- accept and return values that can cross the local/remote boundary

To mark a remote region:

1. Enable `Run on Modal` on each node that should belong to the remote island.
2. Confirm the node shows the blue remote-execution border.
3. Queue the workflow using ComfyUI's normal queue action.

The toggle stores `properties.is_modal_remote = true` in workflow metadata. The editor graph is not rewritten when you toggle a node; rewrite happens only when the prompt is queued.

The node context menu includes a `Modal` submenu for bulk changes. Its `GPU` submenu lists Modal's supported single-GPU targets and marks the current workflow selection. The choice is saved as `workflow.extra.comfy_modal.gpu`, so it follows the graph when the workflow is saved, shared, and loaded again. `Enable on Upstream Nodes` asks the backend which extra upstream nodes must join the selected remote island when a boundary would otherwise contain local-only runtime objects. `Disable on Upstream Nodes`, `Enable All Nodes`, and `Disable All Nodes` apply the corresponding marker changes to the current graph or selection.

### Canvas State

The frontend shows remote state directly on the canvas:

- blue border: marked for Modal, idle
- orange pulsing border: queue-time setup or upload work
- yellow pulsing border: dispatched locally and waiting for Modal execution feedback
- pulsing green border: ready and waiting
- pulsing purple border: executing remotely
- steady green border: finished for the current run
- red border on a remote node: queue-time or execution failure
- neutral `!` badge on an otherwise normally styled local node: the planner found a remote-to-local-to-remote path through that node, so the local re-entry may force extra component splits and data transfers
- numbered badge: remote component assignment for the current prompt

These state colours work with both the legacy LiteGraph renderer and ComfyUI's experimental Nodes 2.0 renderer. The legacy renderer uses its canvas foreground hook; Nodes 2.0 receives an equivalent DOM border, glow, fill, and component badge on each `.lg-node`, driven by the same palette and throttled animation state. Local re-entry warnings cover every local node in a path between remote regions, including multi-node local chains. They preserve the node's normal theme and add only a neutral `!` badge; hovering the badge asks whether the node was intended to run on Modal. The warnings refresh when the planner analyzes a changed remote selection or queues a prompt, and clear immediately when you change a `Run on Modal` flag so stale advice is not left on the canvas.

Subgraph nodes inherit a derived decoration whenever any nested descendant is set to run on Modal, including through multiple levels of nested subgraphs. The container remains a local organizational node: the derived highlight does not change its `Run on Modal` setting. During execution, a `Σ` badge identifies the decoration as a descendant summary and its tooltip lists the descendant phase counts. When descendants have conflicting phases, the border uses the most operationally important state in this order: error, cancelling, active, starting, setup, ready, finalizing, then complete. For example, a subgraph containing one active node and one starting node is purple for active work while the tooltip preserves both counts.

Remote sampler-style progress is rendered in a small temporary panel near the node. Each active progress bar includes a smoothed iterations-per-second (`it/s`) rate derived from successive streamed progress samples; LLM token streams use engine-reported `tok/s`, show token count and TTFT, and reuse the header for compact startup-stage text. Parallel mapped lanes report their rates independently. Static progress redraws happen only when progress events arrive, while indeterminate LLM work, pulsing node phases, setup lane placeholders, and short fade-outs use a throttled canvas animation loop. Preview images and ComfyUI UI payloads emitted by remote nodes are streamed back into the local PromptServer while the remote component is still running. When the browser regains focus, Modal-Sync replays recent UI events and reconciles them against ComfyUI queue/history state so cancelled or completed prompts do not leave stale progress bars behind.

Cancelling a local prompt propagates a targeted interrupt to the active Modal work. If Modal is still deploying, provisioning, or slow to observe the interrupt, the local proxy releases the ComfyUI prompt after the configured grace period while remote cleanup continues.

## Batched And Mapped Workflows

`Modal Map Input` is a pass-through adapter node whose special meaning activates during queue-time rewrite. Put it before a remote-marked region when one boundary input should fan out across Modal workers.

Mapped execution currently supports:

- scalar primitive values as one-item maps
- Python lists
- `IMAGE` batches
- `LATENT` batches and LATENT-like mappings
- other supported batched tensor values split on dimension `0`

One `Modal Map Input` boundary is supported per remote component. Non-mapped boundary inputs are broadcast unchanged to every item. Mapped outputs are reassembled in item order, concatenating batchable tensors when possible and otherwise preserving an ordered list.

Ordinary remote components without `Modal Map Input` still preserve ComfyUI's zipped batch behavior at the remote boundary. If a compatible batch reaches a primitive socket such as `seed: INT`, Modal-Sync itemizes it instead of injecting the whole list into the primitive widget input. If the target node declares `INPUT_IS_LIST`, Modal-Sync runs the component once as an ordinary subgraph. Scheduler-list output contracts remain stable for singleton runs: an ordinary one-item `IMAGE` result stays wrapped as one `[B,H,W,C]` value instead of losing its batch dimension when it returns to ComfyUI.

Mapped components can contain both one-time execute targets and per-item execute targets. For example, two remote samplers may share one upstream model loader while only one sampler fans out over latents. Modal-Sync keeps the invariant upstream work separate from the per-item work so the sibling branch still runs once.

Mapped progress is summarized at the global status pill and representative node with counts such as `3/16`. The global pill shows the workflow's selected GPU on a smaller second line throughout setup, app rebuilding, execution, and finalization. While a Modal workflow is active, it also lists every active container owned by this ComfyUI instance as starting or running, with a short container id and elapsed runtime. The same poll stream integrates active container-seconds for the prompt's selected GPU and shows an estimated accumulated GPU cost plus the current per-minute burn rate. This estimate uses Modal's published prices captured on `2026-08-13`; it intentionally excludes CPU, memory, storage, credits, reservations, and any warm-container lifetime after the prompt UI becomes terminal.

The pill also fetches Modal's actual metered app cost for the GPU-specific app selected by the workflow. Modal's finest billing-report resolution is one hour, and partial current hours are excluded, so Modal-Sync requests the latest completed UTC hour after a ten-minute collection buffer. The result is cached by app, environment, and interval; frequent container polling therefore produces at most one billing API request per report hour. The displayed value is the complete app cost before credits and reservations, not a per-prompt or GPU-only cost. Billing-report access depends on the Modal workspace plan and credentials; failure leaves live container monitoring and the local GPU estimate intact.

The frontend polls the local `/modal/container_status` route quickly while assignments change, slows down once the list is stable, backs off after failures, reduces frequency in background tabs, and stops polling when no Modal workflow remains active. The backend marshals low-level container queries onto Modal's managed event loop and runs billing reads off the ComfyUI event loop. Node-local bars remain reserved for real streamed node progress from executing remote nodes. Their iterations-per-second labels are painted as compact black overlays after the full-width progress fill, so viewer-stable text cannot consume the meter width at low canvas zoom.

## How Modal-Sync Works

When a prompt is queued:

1. The frontend sends the prompt and `extra_pnginfo.workflow` metadata to `POST /modal/queue_prompt`.
2. The backend resolves marked workflow nodes onto queued prompt node ids, including nested subgraph ids such as `195:27`. Reusable definitions under `workflow.definitions.subgraphs` are expanded through each matching subgraph instance before markers are mapped.
3. Remote-marked nodes are partitioned into cost-aware, provider-aware components. Direct remote edges carrying large tensor or media values such as `LATENT`, `IMAGE`, `MASK`, `SIGMAS`, `AUDIO`, or `VIDEO` normally stay inside one component; inexpensive scalar edges may remain component boundaries. A backend that requires SSH cuts a provider boundary even across an expensive edge, and the partitioner refuses any later merge that would make the coarse component graph cyclic.
4. Components also expand across non-transportable inputs such as `MODEL`, `CLIP`, `VAE`, or `CONDITIONING`.
5. A terminal `SaveVideo` fed by a remote `VIDEO` is also placed remotely as an artifact sink. Its encoded file is downloaded into local `output/`; raw frames and audio are not exported merely to re-encode the same video locally.
6. Each component is replaced with one or more generated `ModalUniversalExecutor_<hash>` proxy nodes.
7. Referenced model assets and, when enabled, `custom_nodes/` packages are mirrored into storage.
8. An internal `ModalArtifactFinalizer` output sink is connected to every generated proxy so ComfyUI executes remote components even when their terminal nodes only save files and expose no normal ComfyUI output.
9. The rewritten prompt is submitted to ComfyUI's normal execution queue.
10. Local nodes execute normally until a proxy node is reached.
11. The proxy dispatches local or Modal execution. Values needed by local nodes are materialized as normal ComfyUI outputs, while values consumed only by later remote components travel through the local graph as small Modal-backed references.

Boundary-crossing values must be transportable. Supported evaluated values include:

- `IMAGE`
- `VIDEO`
- `AUDIO`
- `MASK`
- `LATENT`
- `SIGMAS`
- `INT`
- `FLOAT`
- `BOOLEAN`
- `STRING`

`LATENT` transport includes both ordinary tensor samples and ComfyUI's multimodal
`NestedTensor` wrapper used for paired video/audio latent samples.

Automatic mapped execution preserves semantic tensor batches for aggregate consumers.
For example, the `IMAGE` input to `CreateVideo` remains one ordered frame sequence
instead of being split into one remote invocation per frame.
When mapped outputs cannot be concatenated because their shapes differ, Modal-Sync
publishes them as ordered ComfyUI list outputs so ordinary downstream nodes execute
once per item instead of receiving an invalid Python list as one tensor-like value.
That scheduler-list contract propagates through later remote components whose outputs
depend on the mapped input, including components that become mapped only at runtime.

ComfyUI runtime objects such as `MODEL`, `CONDITIONING`, `CLIP`, `VAE`, `NOISE`, and `CONTROL_NET` cannot cross the local/remote boundary directly. Modal-Sync either expands the remote island so those values are produced remotely, keeps local preview/UI branches local, or fails queue-time validation with a boundary error.

If a rewritten graph could create a local scheduler cycle, Modal-Sync logs compact diagnostics for the proxy graph: node classes, dependency edges, proxy payload summaries, planned stages, and detected cycle paths.

## Remote Runtime Behavior

Remote mode prefers a persistent deployed Modal app over ephemeral `app.run()` execution. First-run auto-deploy is enabled by default and can replace missing, stale, unversioned, or protocol-incompatible deployed apps. The extension does not create a persistent web endpoint.

Modal hardware is fixed at deploy time. Choose the target from `Modal` → `GPU`; the next remote run looks up or automatically builds the persistent Modal app dedicated to that GPU. New workflows default to `RTX-PRO-6000`. The legacy `A100` target retains the configured base app name for deployment compatibility, while every other target—including the new default—uses a readable suffix such as `-gpu-rtx-pro-6000` or `-gpu-b300`. GPU-specific app identity prevents one target from contacting an existing class for another GPU merely to inspect its runtime fingerprint. The GPU apps continue to share the configured asset Volume and durable cache names. If you upgrade this node pack and expect changed remote behavior, redeploy the affected GPU app once so it picks up the new code and class options.

Every supported Modal GPU image uses the same Python 3.13, PyTorch 2.13.0, CUDA 13.0, and vLLM 0.27.1 binary contract. The selectable GPUs range from T4 compute capability 7.5 through Blackwell, matching CUDA 13 and vLLM's supported NVIDIA range. After installing bundled custom-node requirements, the image reapplies the deployment-owned runtime pins, installs PyTorch and its companion packages, and then installs the immutable CUDA 13.0 vLLM wheel from its release-commit URL. The TorchAudio layer uses the official CPU-only wheel with `--no-deps`; installing the ordinary PyPI wheel can replace companion packages and invalidate native extensions. Count suffixes such as `B300:2` remain supported and are normalized before build selection. Every ordered package layer is logged and recorded in the runtime fingerprint. Deployment-specific environment values and source mounts are layered after these dependencies, so ordinary source changes reuse the expensive image layers. Build-time handshakes import Torch, TorchVision, TorchAudio, vLLM, NumPy, and OpenCV and verify the expected CUDA and package versions before Modal deploys the app. Smaller GPUs may still reject a particular profile through the measured VRAM admission check, but they cannot fail merely because the selected image omitted vLLM.

CPU memory snapshots are enabled by default. GPU memory snapshots are also enabled by default in current settings, but useful GPU snapshot work is limited to stable loader profiles derived from root literal model-loader nodes. Each loader profile records direct-start and GPU-snapshot warmup timings in its snapshot-profile `Dict`; the first two warmups sample both modes and later starts select the lower median rather than assuming snapshots are universally faster. Generic no-profile and isolated LLM workers use direct startup. Independent root CLIP, UNET, and checkpoint loader plans run concurrently under a small worker limit. Their shared absolute-asset lookup remains installed until every overlapping loader exits, so one completed load cannot break a still-running sibling. A real dispatch joins an exact matching speculative warmup—app, GPU, affinity, loader profile, and startup mode—instead of racing a duplicate cold container. After an automatic deployment, the first important profile is seeded and joined before workflow execution continues.

Warm containers can reuse loaded model state, `PromptExecutor` state, remote session bridge values, and worker-local loader cache entries across compatible requests. The default Modal `scaledown_window` is `600` seconds with `min_containers=0`, so compute can scale down to zero between runs while still benefiting from warm reuse when capacity remains alive. Once a streamed remote component has demonstrably started, the planner can also prepare the nearest reachable component assigned to a different worker-affinity group. This one-component lookahead is deduplicated per prompt and affinity, never blocks the current component, respects a configured container cap below two by declining speculation, and warms literal checkpoint, UNET, CLIP, and dual-CLIP loaders when present. If Modal has no spare capacity, the ordinary downstream invocation remains the fallback.

When the planner detects a remote → local → remote gap, those remote proxies use a stable parameterized worker slot and an independently autoscaled local-gap pool. After the producing remote phase finishes, Modal-Sync sends a lightweight pulse every `15` seconds for at most `900` seconds; dispatching the next remote phase stops the pulse immediately. This protects a loaded worker during a temporarily long local LLM or transformation without continuing active pulses after the remote continuation begins or after a terminal remote phase. The pool then follows its configured idle cooldown.

Independent Modal-backed components can overlap through ComfyUI's async proxy path and the local Modal call executor. Large tensor and media edges are deliberately co-located first, so this parallelism is reserved for genuinely independent branches, inexpensive scalar boundaries, mapped execution, and graph shapes that require a split for correctness. Ordinary components with several remote execution targets remain one proxy unless a local re-entry dependency would create a scheduler cycle or a mixed local-preview/remote-continuation boundary requires ordered phases. Each Modal GPU container handles one active workflow execution at a time, so parallel ready components can scale out across containers instead of multiplexing several active executions onto one worker. `COMFY_MODAL_MAX_INFLIGHT_CALLS` bounds local dispatch independently from the local CPU count and the remote autoscaler.

If a remote output also feeds both later remote work and a local branch that cannot transitively reach any later remote component, Modal-Sync splits the producer from the remote continuation. The producer publishes a small durable bridge reference; the downstream remote proxy consumes that reference immediately, while an asynchronous local materializer downloads the underlying value for the independent preview or save branch. A dispatch gate starts every nearest downstream Modal continuation before releasing the local download, so the preview still appears while later remote inference is in flight without putting transfer, rendering, or display work on its critical path. When these ordered phases place `Modal LLM` between ordinary image or video phases, the planner assigns the LLM and Comfy phases to separate reusable worker-affinity groups so a warm worker holding heavyweight Comfy models is not selected for the LLM load. Deployment protocol checks use a parameterless probe for backward compatibility, but actual execution always rebinds an affinity-aware worker handle, including after protocol compatibility has been cached. Local branches that do feed remote execution remain ordinary dependencies and preserve their original ordering.

When a component output is consumed by other remote components, Modal-Sync can keep the continuation remote. The producer returns a small durable bridge reference through the local ComfyUI proxy instead of returning the underlying `IMAGE`, `AUDIO`, `LATENT`, `MASK`, `SIGMAS`, `VIDEO`, or scalar value. A downstream remote component resolves that reference from the worker's warm bridge cache when possible, or restores it from shared Modal storage when Modal schedules the components on different containers. An independent local-only consumer gets the ordinary ComfyUI value through the parallel materializer described above. A local branch that later returns to remote execution still uses normal materialization and dependency ordering. Deployment identity is validated through the CPU-only model stager before the first GPU dispatch. Once that protocol and fingerprint are cached as current, dispatch and speculative warmup construct only affinity-bound GPU class instances. A backward-compatible parameterless GPU probe is retained solely for older deployments that do not expose CPU identity metadata.

Split proxies, mapped phases, and remote-to-remote component edges use component-local remote sessions. Durable bridge metadata and small serialized values are stored in a Modal `Dict`. Serialized bridge values above `COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES` are stored as integrity-checked, content-addressed objects on the shared Modal Volume, with only the object reference retained in the Dict record. Bridge records do not retain producer inputs when the output is directly restorable or a literal node plan can rebuild it; linked loader plans retain only the boundary inputs used by their reduced dependency subgraph. The complete producer input signature still participates in the bridge key, so omitting recovery inputs cannot make distinct prompts collide. Durable writes loop until every byte is accepted, flush to storage, and verify the staged file's size and SHA-256 before publication. The producer commits new objects before publishing a completed invocation; a consumer whose mounted Volume snapshot is absent, truncated, or corrupt retries through the authoritative `Volume.read_file()` API without reloading the mount or disturbing memory-mapped models. Terminal-result diagnostics time worker publication, durable completion, invocation-record writes, Modal result yielding, local receipt, and stream close separately so post-compute stalls can be assigned to a specific transport boundary. `NOISE` remains deliberately excluded from generic serialization because ComfyUI represents it as a deferred strategy object rather than a realized tensor. Literal `RandomNoise` and `KSamplerSelect` bridge producers are instead rebuilt directly on the downstream worker, and fallback producer replay is rejected whenever its full dependency closure includes sampler work.

Each remote payload also carries a stable invocation id. The local proxy keeps volatile execution metadata outside ComfyUI's cache surface and rehydrates it using both the queued prompt id and proxy node id, so a second queued workflow cannot overwrite the active workflow's warmup or session context. V3 proxies read hidden prompt metadata and node ids from ComfyUI's execution-specific class clone, with a list-normalizing legacy `kwargs` fallback, keeping back-to-back queued workflows isolated. The worker records its lifecycle in a shared Modal `Dict` and replays a completed result when the local client or Modal retries the same call. An overlapping delivery waits briefly for the active attempt to publish a terminal state instead of failing immediately. If a streamed call loses its consumer, the worker cancels unfinished compute and marks that attempt failed before the retry proceeds; uncommitted Volume writes remain unpublished. Large completed results use the same content-addressed Volume store, while failed attempts remain retryable.

After every remote payload, including the final component that clears its remote session, Modal-Sync compares the remote ComfyUI `output/` tree with its pre-execution snapshot. New or replaced regular files are bundled into the durable result and downloaded into the corresponding local `output/` subdirectory. Each downloaded filename is prefixed `remote-<app_id>-<epoch>-`, where `app_id` is the unique Modal app suffix and `epoch` is the trailing nine digits of the remote completion time. Artifact paths and SHA-256 digests are validated, symlinks and escaping paths are rejected, identical retry downloads are reused, and a differing local collision receives a numeric suffix instead of being overwritten. A terminal local `SaveVideo` whose input comes from a remote `VIDEO` is automatically executed as a remote artifact sink, preserving its configured filename and encoding while avoiding transport of the decoded frame and audio tensors. Nonterminal local video consumers retain ordinary lossless VIDEO transport.

The proxy emits its internal completion token only after the remote result and its artifacts have been materialized locally. The finalizer consumes those tokens without producing a user-visible value, making artifact-only remote branches part of ComfyUI's required output path while preserving normal downstream graph behavior.

Remote boundary hydration does not alter the prompt metadata exposed to custom nodes. The PromptExecutor uses hydrated tensors and runtime objects for dependency execution, while hidden `PROMPT` inputs receive a separately preserved JSON-safe graph. Metadata-writing output nodes such as VideoHelperSuite can therefore serialize the prompt into an image or video without encountering tensor values introduced by the remote boundary.

Remote subgraph runs can persist transport-safe node outputs into a shared Modal `Dict` using ComfyUI input-signature semantics. The cache skips non-serializable outputs and entries above the configured size cap.

Tensor and byte boundary values use a versioned binary envelope, so safetensors bytes cross Modal directly instead of expanding through base64 JSON. Readers remain compatible with legacy JSON payloads during deployment replacement. Streamed progress uses a bounded queue that coalesces stale progress when a consumer falls behind while preserving result, error, and completion events.

Cancellation uses a shared Modal `Dict` control store plus local polling of ComfyUI cancellation. Remote workers retire themselves after poisoned CUDA/runtime failures or stuck cancellation, while deterministic prompt and custom-node errors preserve healthy warm workers.

## Asset And Custom Node Sync

The sync engine automatically looks for inputs that resolve to files ending in:

- `.safetensors`
- `.ckpt`
- `.pt`
- `.vae`

Absolute paths and model names resolvable through ComfyUI `folder_paths` work. Arbitrary unresolved strings do not sync. If a remote-marked node depends on a model filename that cannot be resolved locally, prompt queueing fails instead of sending a broken remote request.

In remote mode, assets and custom-node archives are uploaded into the configured Modal volume. In local mode, the default backend is a local mirror used for development and tests.

Asset paths are planned once per queued prompt. Repeated references across nodes or remote components share one hash, sync-index lookup, and upload decision while every component still receives the same content-addressed remote path and reload metadata.

Custom-node sync is enabled by default in remote mode and disabled by default in local mode. When enabled, Modal-Sync packages `custom_nodes/` as a whole-tree manifest plus content-addressed code archives for each top-level custom-node package. Package-owned model artifacts such as `.pth`, `.safetensors`, `.gguf`, and `.onnx` files are stored separately and linked into the extracted package on the worker, so a code edit does not recompress or reupload a multi-gigabyte model. Nested virtual environments, caches, compiled Python artifacts, logs, and temporary files are excluded. Unchanged code and asset digests are reused through a Modal `Dict` sync index instead of probing the volume for many marker files. The remote image includes the GL and GLib runtime libraries required when a custom-node dependency installs GUI-enabled OpenCV, even though Modal-Sync's core runtime uses the headless OpenCV wheel.

Warm workers resolve committed assets through a local read-through cache when their mounted Volume snapshot is stale. This keeps newly uploaded models available without forcing `Volume.reload()` to close mmap-backed model files, and reused sync-index entries are still checked for runtime visibility before execution.

When a synced top-level custom-node package has a `requirements.txt`, those requirements are folded into the Modal image build. `-r other-file.txt` includes are followed relative to the declaring package; pip option and constraint lines are ignored.

Warm workers call `vol.reload()` only for uploaded mounted-volume paths that the current payload can reference. Reload markers are deduped across one queued workflow so multiple components do not repeatedly reload the same asset snapshot.

Durable bridge objects and oversized invocation results share one content-addressed object store. Writes produced during one invocation are committed to the Modal volume as one completion batch on the stream-owning request lifecycle, before completed invocation metadata is published for retry replay. When a warm worker's mounted snapshot predates an object produced by another worker, Modal-Sync reads that committed object directly through `Volume.read_file()` instead of reloading the mount and disturbing memory-mapped model files.

## Configuration

Boolean values accept `1`, `true`, `yes`, `on`, `0`, `false`, `no`, and `off`.

### Routing And Metadata

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_ROUTE_PATH` | `/modal/queue_prompt` | Queue endpoint registered by the backend. |
| `COMFY_MODAL_MARKER_PROPERTY` | `is_modal_remote` | Workflow property used to mark remote nodes. |

### Paths And Sync

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFYUI_ROOT` | auto-discovered | Preferred ComfyUI checkout root for tests and path resolution. |
| `COMFY_MODAL_COMFYUI_ROOT` | auto-discovered | Modal-Sync-specific ComfyUI checkout override, used after `COMFYUI_ROOT`. |
| `COMFY_MODAL_CUSTOM_NODES_DIR` | auto-discovered | `custom_nodes` directory to bundle and mirror. |
| `COMFY_MODAL_LOCAL_STORAGE_ROOT` | `/tmp/comfyui-modal-sync-storage` | Local mirror root for local mode, tests, and dry runs. |
| `COMFY_MODAL_REMOTE_STORAGE_ROOT` | `/storage` | Mounted storage root inside the Modal container. |
| `COMFY_MODAL_CUSTOM_NODES_ARCHIVE` | `custom_nodes_bundle.zip` | Base archive name used for custom-node bundle paths. |
| `COMFY_MODAL_SYNC_CUSTOM_NODES` | `false` in local mode, `true` otherwise | Force-enable or disable custom-node bundle sync. |

### Deployment

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_EXECUTION_MODE` | `local` | Set to `remote` for Modal-backed execution. |
| `MODAL_ENVIRONMENT` | active Modal profile or workspace default | Select the Modal environment used for app deployment, lookup, status, and billing. Set it explicitly when the same app name exists in multiple environments. |
| `COMFY_MODAL_APP_NAME` | `comfy-modal-sync-<instance_id>` | Explicit Modal app name override; otherwise derived from the persistent per-ComfyUI identity. |
| `COMFY_MODAL_INSTANCE_ID_PATH` | `<ComfyUI user directory>/.comfy-modal-sync-instance-id` | Override the persistent 64-bit identity file location. |
| `COMFY_MODAL_SECRET_NAME` | `comfy` | Existing Modal secret collection injected into every remote worker as environment variables. |
| `COMFY_MODAL_VOLUME_NAME` | `comfy-universal-storage` | Modal volume name for synced assets and bundles. |
| `COMFY_MODAL_AUTO_DEPLOY` | `true` | Deploy or replace the configured app when lookup fails or its runtime fingerprint is stale. |
| `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK` | `false` | Allow the older temporary `app.run()` fallback when deployed lookup fails. |
| `COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR` | `true` | Make a remote worker exit after surfacing a crash. |

### Modal State Stores

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_INTERRUPT_DICT_NAME` | `<app_name>-interrupts` | Shared Modal `Dict` for cancellation flags. |
| `COMFY_MODAL_NODE_CACHE_DICT_NAME` | `<app_name>-node-cache` | Shared Modal `Dict` for persisted transport-safe node outputs. |
| `COMFY_MODAL_SESSION_BRIDGE_DICT_NAME` | `<app_name>-session-bridges` | Shared Modal `Dict` for durable session bridge metadata. |
| `COMFY_MODAL_INVOCATION_DICT_NAME` | `<app_name>-invocations` | Shared Modal `Dict` for idempotent invocation lifecycle and result metadata. |
| `COMFY_MODAL_SYNC_INDEX_DICT_NAME` | `<app_name>-sync-index` | Shared Modal `Dict` for mirrored asset and bundle digests. |
| `COMFY_MODAL_SNAPSHOT_PROFILE_DICT_NAME` | `<app_name>-snapshot-profiles` | Shared Modal `Dict` for loader snapshot profile records. |
| `COMFY_MODAL_NODE_CACHE_MAX_BYTES` | `5242880` | Maximum raw output size eligible for persisted node caching; set `0` to disable. |
| `COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES` | `4194304` | Maximum serialized bridge input or output size retained inline before Volume offload. |
| `COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES` | `4194304` | Maximum completed invocation result retained inline before Volume offload. |

### Runtime Sizing And Warmup

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_GPU` | `RTX-PRO-6000` | Compatibility fallback for workflows or API clients without `workflow.extra.comfy_modal.gpu`. The context-menu workflow selection is authoritative for normal UI runs and also selects the compatible pinned PyTorch CUDA wheel set. |
| `COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT` | `true` | Enable Modal CPU memory snapshots. |
| `COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT` | `true` | Enable Modal GPU memory snapshots for profiled loader states. |
| `COMFY_MODAL_SCALEDOWN_WINDOW` | `600` | Seconds to keep idle Modal containers warm. |
| `COMFY_MODAL_LOCAL_GAP_KEEPALIVE_SECONDS` | `900` | Maximum time to retain the matching remote worker slot while a remote → local → remote workflow gap runs; set `0` to disable pulses. |
| `COMFY_MODAL_LOCAL_GAP_KEEPALIVE_INTERVAL_SECONDS` | `15` | Seconds between lightweight worker-retention pulses during a local gap. |
| `COMFY_MODAL_MIN_CONTAINERS` | `0` | Minimum warm containers. |
| `COMFY_MODAL_MAX_CONTAINERS` | unset | Optional upper bound on simultaneously scaled Modal containers. |
| `COMFY_MODAL_BUFFER_CONTAINERS` | unset | Optional spare warm containers above current load. |
| `COMFY_MODAL_MAX_INFLIGHT_CALLS` | `4` | Maximum local Modal calls dispatched at once; mapped fan-out is clamped to this budget. |
| `COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS` | `3600` | Maximum runtime for one Modal workflow call. |
| `COMFY_MODAL_STARTUP_TIMEOUT_SECONDS` | `900` | Maximum Modal container startup and snapshot-restore time. |
| `COMFY_MODAL_LLM_MAX_RESIDENT_MODELS` | `2` | Maximum Transformers LLM profiles retained per warm GPU worker before LRU eviction. |
| `COMFY_MODAL_LLM_RESERVE_FREE_GB` | `24.0` | Default minimum free VRAM retained for ComfyUI-managed image and video models. The node can override this per request. |
| `COMFY_MODAL_LLM_VLLM_EXECUTION_MODE` | `auto` | `auto` uses eager on a container's first workflow and promotes to throughput on its second; `eager` and `throughput` pin one mode for the container lifetime. |
| `COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS` | `15.0` | Maximum post-eviction wait for CUDA free memory before retiring the worker and retrying once on a fresh worker. |
| `COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE` | `256` | Maximum buffered remote progress/result envelopes; stale progress is coalesced when full. |
| `COMFY_MODAL_ENABLE_PROACTIVE_WARMUP` | `true` | Start background warmup from runtime parallelism signals and one-component cross-affinity planner lookahead. |
| `COMFY_MODAL_ENABLE_LOADER_PREWARM` | `true` | During warmup, execute synthetic loader prompts for root literal model-loader nodes. |
| `COMFY_MODAL_LOADER_PREWARM_WORKERS` | `2` | Maximum number of independent root CLIP, UNET, and checkpoint loader plans executed concurrently in one worker. |
| `COMFY_MODAL_PROACTIVE_WARMUP_HEAD_START_SECONDS` | `2.0` | Bounded wait for exact mapped warmup slots before lane seeding starts. |
| `COMFY_MODAL_MAX_LOADER_PREWARMS_PER_COMPONENT` | reserved | Recognized in the settings environment signature, but not currently consumed by runtime settings. |

### Cancellation And Logs

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS` | `2.0` | How long the local proxy waits after propagating cancellation before releasing the local prompt. |
| `COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS` | `1.0` | How long a Modal worker waits after observing cancellation before exiting if execution is still stuck. |
| `COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS` | `false` | Mirror live Modal container logs into local ComfyUI stderr during streamed executions. |

## Troubleshooting

- App not found or deleted: leave `COMFY_MODAL_AUTO_DEPLOY=true` so the next lookup can deploy the stable cloud entrypoint again.
- Changed the workflow GPU: queue the next remote run and Modal-Sync will look up or build the GPU-specific app without starting the previously selected GPU app.
- Remote mode still uses local mirror storage: restart ComfyUI with `COMFY_MODAL_EXECUTION_MODE=remote` and the Modal SDK available so sync and invocation resolve the same mode.
- Missing custom node class in Modal: ensure custom-node sync is enabled, check the worker logs for import failures, and confirm the package's Python dependencies are present in its `requirements.txt`.
- `UNETLoader` reports `Could not detect model type` for a synced Flux-style model: Modal-Sync aliases saved RMSNorm `.weight` keys to ComfyUI's `.scale` key form in memory before remote model detection, but the model still has to be supported by the ComfyUI checkout packaged into the Modal app.
- Boundary validation fails on `MODEL`, `CLIP`, `VAE`, `CONDITIONING`, or similar values: include the upstream producer in the remote island or use `Enable on Upstream Nodes`.
- ComfyUI reports `Dependency cycle detected` after rewrite: inspect local `comfy.log` for the Modal-Sync proxy graph diagnostics and cycle path.
- Cancellation appears to finish locally while Modal is still busy: the local prompt has been released after the grace window, and remote cleanup or worker retirement may still be completing.
- Remote runtime behavior does not reflect a local code update: redeploy the configured Modal app so the deployed class uses the current code.

## Development

Manage the project with `uv`.

The `add-generic-execution-environments` development line is introducing a
provider-neutral execution model alongside Modal. Remote environment records
deliberately contain only SSH destinations, scheduling limits, tags, discovered
capabilities, and user-supplied cost rates. SSH authentication remains in the
user's normal SSH configuration and agent; private keys and passwords are not
stored in workflows or node-pack configuration.

```bash
uv sync --group test
uv run pytest
```

Tests look for ComfyUI in `COMFYUI_ROOT` first, then `COMFY_MODAL_COMFYUI_ROOT`, then an installed parent checkout, then `~/git/ComfyUI`.

The live Modal canaries are opt-in because they authenticate, deploy when needed, start GPU containers, and therefore may incur Modal charges. They validate the deployed runtime fingerprint, binary tensor transport plus durable duplicate replay, two-call remote concurrency through a shared barrier, and prompt cancellation propagation:

```bash
COMFY_MODAL_RUN_LIVE_CANARIES=1 \
COMFY_MODAL_EXECUTION_MODE=remote \
uv run --extra remote pytest -q tests/test_live_modal_canary.py
```

The canaries use the normal `COMFY_MODAL_APP_NAME`, environment, GPU, timeout, and container-limit settings. The parallel canary skips when either the local in-flight limit or the configured Modal container limit is below two. All ordinary tests remain local-only and do not require Modal credentials.

The 2026-08-19 non-B300 regression canary passed on RTX PRO 6000 with the final reconciled Torch 2.13.0/CUDA 13.0/vLLM 0.27.1 image and real inference using the generated profile for `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4`. Model capacity and quantization compatibility are still checked per GPU; the universal image guarantees backend availability, not that every checkpoint can execute on every card.

The standalone benchmark harness uses a dedicated app, one GPU container, deterministic inputs, and fixed worker affinity. Each cold cycle stops only that benchmark app, runs one cold-container inference, then runs the requested number of resident-engine repeats. Two cold cycles reveal whether the persistent compilation cache improves the second container boot. The command is billable and stops its benchmark app afterward unless `--keep-app` is supplied:

```bash
uv run --extra remote python scripts/benchmark_modal_llm.py \
  --mode throughput \
  --gpu RTX-PRO-6000 \
  --cold-cycles 2 \
  --warm-runs 1 \
  --output /tmp/modal-llm-throughput.json
```

Run the same command with `--mode eager` and a different output path for a matched baseline. The JSON artifact includes the source revision, complete workload configuration, wall time, model load time, generation time, TTFT, token rate, output count, response digest, and resident-cache status for every invocation. Reasoning is disabled and a deterministic synthetic image is included by default; use `--enable-reasoning` or `--text-only` to change those dimensions explicitly.

In a 2026-08-19 RTX PRO 6000 run with Qwen3.8 27B NVFP4 and 128 output tokens, persistent AOT reuse reduced `torch.compile` from 138.0 to as little as 22.4 seconds. Warm throughput mode reached 38.4-39.1 tokens per second with roughly 50 ms TTFT, compared with 17.2 tokens per second and 151 ms in eager mode. Cached throughput cold initialization remained slower, so eager is still useful for short-lived apps while throughput is preferred when the resident engine will serve repeated calls.

To run tests against a temporary checkout:

```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/comfyui-modal-test/ComfyUI
UV_PROJECT_ENVIRONMENT=/tmp/comfyui-modal-test-env uv sync --group test
COMFYUI_ROOT=/tmp/comfyui-modal-test/ComfyUI \
  /tmp/comfyui-modal-test-env/bin/python -m pytest
```

The repository is structured as a ComfyUI Registry node pack with registry metadata in [`pyproject.toml`](pyproject.toml) and a publish workflow in [`.github/workflows/publish_action.yml`](.github/workflows/publish_action.yml). The registry pack name is `modal-sync`, the display name is `Modal Sync`, and the current publisher id is `ttulttul`.

[`modal_test_workflow.json`](modal_test_workflow.json) is a checked-in smoke artifact from a successful Modal-path run, not a pristine authoring workflow.

## Current Limitations

- Remote execution is component-based. If you leave a local gap in the middle of a remote chain, the boundary still has to be transport-safe.
- Real Modal execution depends on a working Modal SDK environment and a storage backend visible to Modal workers.
- Non-JSON, non-bytes, non-tensor payloads are not supported across the current local/remote boundary.
- Workflow artifacts captured after a remote run may include internal proxy nodes such as `ModalUniversalExecutor`; they are useful as regression fixtures, but should not be treated as clean source workflows.
