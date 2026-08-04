# ComfyUI Modal-Sync

> [!WARNING]
> This project is still alpha. Expect missing features, rough edges, and breaking changes.

ComfyUI Modal-Sync is a ComfyUI custom node extension for running selected parts of a workflow through Modal. You mark nodes with `Run on Modal`; Modal-Sync rewrites the queued prompt into transport-aware remote components, syncs required assets, and returns remote outputs to the local ComfyUI graph.

## Overview

Modal-Sync provides:

- a ComfyUI frontend extension with a `Run on Modal` toggle and remote execution overlays
- a queue route at `/modal/queue_prompt` that intercepts normal prompt submission
- queue-time graph partitioning and proxy-node rewrite for selected remote regions
- local in-process execution mode for development and tests
- Modal-backed remote execution with deployed-app lookup and first-run auto-deploy
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
- [`api_intercept.py`](api_intercept.py), the queue rewrite route
- [`modal_executor_node.py`](modal_executor_node.py), the internal proxy node registry

Start with local mode while building or debugging workflows:

```bash
export COMFY_MODAL_EXECUTION_MODE=local
```

Local mode still exercises marker resolution, prompt rewrite, sync planning, serialization, and proxy execution, but the rewritten component runs in the local ComfyUI process instead of in Modal.

Use remote mode with working Modal credentials:

```bash
export COMFY_MODAL_EXECUTION_MODE=remote
```

When ComfyUI starts in remote mode, Modal-Sync checks whether the supported Modal SDK is importable. If it is missing, the extension installs the pinned `modal==1.4.2` package into the exact Python interpreter running ComfyUI. The installer prefers `uv pip` and falls back to `python -m pip`; startup logs report detection, the selected command, and the result. If neither installer is available or installation fails, startup continues with the existing local-mirror fallback and logs an actionable error.

Modal authentication remains user-managed. Run `<comfyui-venv>/bin/python -m modal setup` when the current user does not already have working credentials. To install the SDK manually before startup, run `uv pip install --python <comfyui-venv>/bin/python "modal==1.4.2"`.

Each ComfyUI environment receives its own Modal app name on first startup. Modal-Sync generates 64 random bits, persists them as lowercase hexadecimal in `<ComfyUI user directory>/.comfy-modal-sync-instance-id`, and uses an unpadded URL-safe Base64 encoding in names such as `comfy-modal-sync-R7uqpQZ6S1A`. The identity file is published atomically, so concurrent first startups converge on one name. It lives in ComfyUI's effective user directory and therefore follows `--user-directory`; keep the file when upgrading if the deployment and its state should remain associated with the same ComfyUI environment. `COMFY_MODAL_APP_NAME` remains an authoritative override for deliberately shared or externally managed deployments.

For repository development, `uv sync --extra remote --group test` installs the same pinned SDK. Remote mode uses the stable cloud entrypoint in [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py). On first use, Modal-Sync can auto-deploy the configured Modal app if it does not exist.

The deployed image uses Python 3.11 plus an exact ComfyUI support and CUDA package set, including the ComfyUI-pinned `comfy-aimdo` and `comfy-kitchen` releases required by current memory-management and quantization imports. Remote prompt executors mirror both legacy scalar and current active/inactive RAM-pressure cache arguments from ComfyUI, and the cache adapter supports both synchronous legacy access and current coroutine-based cache operations. The image's local build context is limited to the ComfyUI source packages, top-level Python modules, and runtime configuration needed by the headless worker; model directories, custom nodes, caches, tests, virtual environments, user data, and unknown top-level directories stay out of the image snapshot. Before every process's first remote invocation, Modal-Sync compares the deployed worker's runtime fingerprint with the local source, ComfyUI source, custom-node requirements, and runtime-shaping settings. A missing or mismatched fingerprint is treated as stale and replaced automatically when `COMFY_MODAL_AUTO_DEPLOY=true`; replacement uses Modal's non-interactive SDK stop API when available and an explicitly confirmed CLI fallback, so app shutdown cannot stall on a hidden terminal prompt.

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

The node context menu includes a `Modal` submenu for bulk changes. `Enable on Upstream Nodes` asks the backend which extra upstream nodes must join the selected remote island when a boundary would otherwise contain local-only runtime objects. `Disable on Upstream Nodes`, `Enable All Nodes`, and `Disable All Nodes` apply the corresponding marker changes to the current graph or selection.

### Canvas State

The frontend shows remote state directly on the canvas:

- blue border: marked for Modal, idle
- orange pulsing border: queue-time setup or upload work
- yellow pulsing border: dispatched locally and waiting for Modal execution feedback
- pulsing green border: ready and waiting
- pulsing purple border: executing remotely
- steady green border: finished for the current run
- red border: queue-time or execution failure
- numbered badge: remote component assignment for the current prompt

Remote sampler-style progress is rendered in a small temporary panel near the node. Static progress redraws happen only when progress events arrive, while pulsing node phases, setup lane placeholders, and short fade-outs use a throttled canvas animation loop. Preview images and ComfyUI UI payloads emitted by remote nodes are streamed back into the local PromptServer while the remote component is still running. When the browser regains focus, Modal-Sync replays recent UI events and reconciles them against ComfyUI queue/history state so cancelled or completed prompts do not leave stale progress bars behind.

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

Ordinary remote components without `Modal Map Input` still preserve ComfyUI's zipped batch behavior at the remote boundary. If a compatible batch reaches a primitive socket such as `seed: INT`, Modal-Sync itemizes it instead of injecting the whole list into the primitive widget input. If the target node declares `INPUT_IS_LIST`, Modal-Sync runs the component once as an ordinary subgraph.

Mapped components can contain both one-time execute targets and per-item execute targets. For example, two remote samplers may share one upstream model loader while only one sampler fans out over latents. Modal-Sync keeps the invariant upstream work separate from the per-item work so the sibling branch still runs once.

Mapped progress is summarized at the global status pill and representative node with counts such as `3/16`. Node-local bars remain reserved for real streamed node progress from executing remote nodes.

## How Modal-Sync Works

When a prompt is queued:

1. The frontend sends the prompt and `extra_pnginfo.workflow` metadata to `POST /modal/queue_prompt`.
2. The backend resolves marked workflow nodes onto queued prompt node ids, including nested subgraph ids such as `195:27`.
3. Remote-marked nodes are partitioned into transport-aware components.
4. Components expand upstream when required by non-transportable inputs such as `MODEL`, `CLIP`, `VAE`, or `CONDITIONING`.
5. Each component is replaced with one or more generated `ModalUniversalExecutor_<hash>` proxy nodes.
6. Referenced model assets and, when enabled, `custom_nodes/` packages are mirrored into storage.
7. The rewritten prompt is submitted to ComfyUI's normal execution queue.
8. Local nodes execute normally until a proxy node is reached.
9. The proxy serializes boundary inputs, dispatches local or Modal execution, deserializes returned outputs, and exposes them as normal ComfyUI outputs.

Boundary-crossing values must be transportable. Supported evaluated values include:

- `IMAGE`
- `MASK`
- `LATENT`
- `SIGMAS`
- `NOISE`
- `INT`
- `FLOAT`
- `BOOLEAN`
- `STRING`

ComfyUI runtime objects such as `MODEL`, `CONDITIONING`, `CLIP`, `VAE`, and `CONTROL_NET` cannot cross the local/remote boundary directly. Modal-Sync either expands the remote island so those values are produced remotely, keeps local preview/UI branches local, or fails queue-time validation with a boundary error.

If a rewritten graph could create a local scheduler cycle, Modal-Sync logs compact diagnostics for the proxy graph: node classes, dependency edges, proxy payload summaries, planned stages, and detected cycle paths.

## Remote Runtime Behavior

Remote mode prefers a persistent deployed Modal app over ephemeral `app.run()` execution. First-run auto-deploy is enabled by default and can replace missing, stale, unversioned, or protocol-incompatible deployed apps. The extension does not create a persistent web endpoint.

Modal hardware is fixed at deploy time. If you change `COMFY_MODAL_GPU`, stop/delete the existing Modal app or redeploy it so the remote class is built with the new GPU type. If you upgrade this node pack and expect changed remote behavior, redeploy once so the Modal app picks up the new code and class options.

CPU memory snapshots are enabled by default. GPU memory snapshots are also enabled by default in current settings, but useful GPU snapshot work is limited to stable loader profiles derived from root literal model-loader nodes. Generic no-profile workers skip GPU snapshot prewarm because they do not provide the model-loaded cold-start win.

Warm containers can reuse loaded model state, `PromptExecutor` state, remote session bridge values, and worker-local loader cache entries across compatible requests. The default Modal `scaledown_window` is `600` seconds with `min_containers=0`, so compute can scale down to zero between runs while still benefiting from warm reuse when capacity remains alive.

Independent Modal-backed components can overlap through ComfyUI's async proxy path and the local Modal call executor. Each Modal GPU container handles one active workflow execution at a time, so parallel ready components can scale out across containers instead of multiplexing several active executions onto one worker. `COMFY_MODAL_MAX_INFLIGHT_CALLS` bounds local dispatch independently from the local CPU count and the remote autoscaler.

Split proxies and mapped phases use prompt-scoped remote sessions for live non-transportable values. Durable bridge metadata is stored in a Modal `Dict` so later phases can rehydrate selected values after container churn. Oversized serialized bridge inputs and outputs are stored as integrity-checked, content-addressed objects on the shared Modal Volume instead of being embedded in `Dict` records. Sampler-producing bridges are not replayed as a fallback; losing those values is surfaced as a session-state error.

Each remote payload also carries a stable invocation id. The worker records its lifecycle in a shared Modal `Dict` and replays a completed result when the local client or Modal retries the same call. An overlapping delivery waits briefly for the active attempt to publish a terminal state instead of failing immediately. If a streamed call loses its consumer, the worker cancels unfinished compute and marks that attempt failed before the retry proceeds; uncommitted Volume writes remain unpublished. Large completed results use the same content-addressed Volume store, while failed attempts remain retryable.

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
| `COMFY_MODAL_APP_NAME` | `comfy-modal-sync-<instance_id>` | Explicit Modal app name override; otherwise derived from the persistent per-ComfyUI identity. |
| `COMFY_MODAL_INSTANCE_ID_PATH` | `<ComfyUI user directory>/.comfy-modal-sync-instance-id` | Override the persistent 64-bit identity file location. |
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
| `COMFY_MODAL_GPU` | `A100` | Modal GPU type requested by the deployed remote class. |
| `COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT` | `true` | Enable Modal CPU memory snapshots. |
| `COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT` | `true` | Enable Modal GPU memory snapshots for profiled loader states. |
| `COMFY_MODAL_SCALEDOWN_WINDOW` | `600` | Seconds to keep idle Modal containers warm. |
| `COMFY_MODAL_MIN_CONTAINERS` | `0` | Minimum warm containers. |
| `COMFY_MODAL_MAX_CONTAINERS` | unset | Optional upper bound on simultaneously scaled Modal containers. |
| `COMFY_MODAL_BUFFER_CONTAINERS` | unset | Optional spare warm containers above current load. |
| `COMFY_MODAL_MAX_INFLIGHT_CALLS` | `4` | Maximum local Modal calls dispatched at once; mapped fan-out is clamped to this budget. |
| `COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS` | `3600` | Maximum runtime for one Modal workflow call. |
| `COMFY_MODAL_STARTUP_TIMEOUT_SECONDS` | `900` | Maximum Modal container startup and snapshot-restore time. |
| `COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE` | `256` | Maximum buffered remote progress/result envelopes; stale progress is coalesced when full. |
| `COMFY_MODAL_ENABLE_PROACTIVE_WARMUP` | `true` | Start background warmup from runtime parallelism signals such as mapped fan-out. |
| `COMFY_MODAL_ENABLE_LOADER_PREWARM` | `true` | During warmup, execute synthetic loader prompts for root literal model-loader nodes. |
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
- Changed `COMFY_MODAL_GPU`: delete or stop the old Modal app before redeploying; hardware is fixed at deploy time.
- Remote mode still uses local mirror storage: restart ComfyUI with `COMFY_MODAL_EXECUTION_MODE=remote` and the Modal SDK available so sync and invocation resolve the same mode.
- Missing custom node class in Modal: ensure custom-node sync is enabled, check the worker logs for import failures, and confirm the package's Python dependencies are present in its `requirements.txt`.
- `UNETLoader` reports `Could not detect model type` for a synced Flux-style model: Modal-Sync aliases saved RMSNorm `.weight` keys to ComfyUI's `.scale` key form in memory before remote model detection, but the model still has to be supported by the ComfyUI checkout packaged into the Modal app.
- Boundary validation fails on `MODEL`, `CLIP`, `VAE`, `CONDITIONING`, or similar values: include the upstream producer in the remote island or use `Enable on Upstream Nodes`.
- ComfyUI reports `Dependency cycle detected` after rewrite: inspect local `comfy.log` for the Modal-Sync proxy graph diagnostics and cycle path.
- Cancellation appears to finish locally while Modal is still busy: the local prompt has been released after the grace window, and remote cleanup or worker retirement may still be completing.
- Remote runtime behavior does not reflect a local code update: redeploy the configured Modal app so the deployed class uses the current code.

## Development

Manage the project with `uv`.

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
