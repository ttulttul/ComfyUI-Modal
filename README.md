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

Use remote mode when the ComfyUI environment has the `modal` package and working Modal credentials:

```bash
export COMFY_MODAL_EXECUTION_MODE=remote
```

Remote mode uses the stable cloud entrypoint in [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py). On first use, Modal-Sync can auto-deploy the configured Modal app if it does not exist.

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

Remote sampler-style progress is rendered in a small temporary panel near the node. Preview images and ComfyUI UI payloads emitted by remote nodes are streamed back into the local PromptServer while the remote component is still running.

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

Independent Modal-backed components can overlap through ComfyUI's async proxy path and the local Modal call executor. Each Modal GPU container handles one active workflow execution at a time, so parallel ready components can scale out across containers instead of multiplexing several active executions onto one worker.

Split proxies and mapped phases use prompt-scoped remote sessions for live non-transportable values. Durable bridge metadata is stored in a Modal `Dict` so later phases can rehydrate selected values after container churn. Sampler-producing bridges are not replayed as a fallback; losing those values is surfaced as a session-state error.

Remote subgraph runs can persist transport-safe node outputs into a shared Modal `Dict` using ComfyUI input-signature semantics. The cache skips non-serializable outputs and entries above the configured size cap.

Cancellation uses a shared Modal `Dict` control store plus local polling of ComfyUI cancellation. Remote workers can retire themselves after crashes or stuck cancellation so a bad warm GPU container is not left idle and billable.

## Asset And Custom Node Sync

The sync engine automatically looks for inputs that resolve to files ending in:

- `.safetensors`
- `.ckpt`
- `.pt`
- `.vae`

Absolute paths and model names resolvable through ComfyUI `folder_paths` work. Arbitrary unresolved strings do not sync. If a remote-marked node depends on a model filename that cannot be resolved locally, prompt queueing fails instead of sending a broken remote request.

In remote mode, assets and custom-node archives are uploaded into the configured Modal volume. In local mode, the default backend is a local mirror used for development and tests.

Custom-node sync is enabled by default in remote mode and disabled by default in local mode. When enabled, Modal-Sync packages `custom_nodes/` as a whole-tree manifest plus content-addressed archives for each top-level custom-node package. Unchanged package digests are reused through a Modal `Dict` sync index instead of probing the volume for many marker files.

When a synced top-level custom-node package has a `requirements.txt`, those requirements are folded into the Modal image build. `-r other-file.txt` includes are followed relative to the declaring package; pip option and constraint lines are ignored.

Warm workers call `vol.reload()` only for uploaded mounted-volume paths that the current payload can reference. Reload markers are deduped across one queued workflow so multiple components do not repeatedly reload the same asset snapshot.

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
| `COMFY_MODAL_APP_NAME` | `comfy-modal-sync` | Modal app name. |
| `COMFY_MODAL_VOLUME_NAME` | `comfy-universal-storage` | Modal volume name for synced assets and bundles. |
| `COMFY_MODAL_AUTO_DEPLOY` | `true` | Deploy or replace the configured app when lookup requires it. |
| `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK` | `false` | Allow the older temporary `app.run()` fallback when deployed lookup fails. |
| `COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR` | `true` | Make a remote worker exit after surfacing a crash. |

### Modal State Stores

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_INTERRUPT_DICT_NAME` | `<app_name>-interrupts` | Shared Modal `Dict` for cancellation flags. |
| `COMFY_MODAL_NODE_CACHE_DICT_NAME` | `<app_name>-node-cache` | Shared Modal `Dict` for persisted transport-safe node outputs. |
| `COMFY_MODAL_SESSION_BRIDGE_DICT_NAME` | `<app_name>-session-bridges` | Shared Modal `Dict` for durable session bridge metadata. |
| `COMFY_MODAL_SYNC_INDEX_DICT_NAME` | `<app_name>-sync-index` | Shared Modal `Dict` for mirrored asset and bundle digests. |
| `COMFY_MODAL_SNAPSHOT_PROFILE_DICT_NAME` | `<app_name>-snapshot-profiles` | Shared Modal `Dict` for loader snapshot profile records. |
| `COMFY_MODAL_NODE_CACHE_MAX_BYTES` | `5242880` | Maximum raw output size eligible for persisted node caching; set `0` to disable. |

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
