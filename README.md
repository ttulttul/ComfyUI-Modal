# ComfyUI Modal-Sync

> [!WARNING]
> This project is still alpha. Expect missing features, rough edges, and breaking changes.

`ComfyUI Modal-Sync` is a ComfyUI custom node extension for marking parts of a workflow for remote execution. The frontend stores a per-node `is_modal_remote` flag in workflow metadata, the backend groups connected marked nodes into a remote component, and a proxy node executes that component either locally or through Modal.

## Status

- The queue rewrite, serialization, and sync pipeline are implemented.
- `local` execution mode is the default development path and exercises the same rewrite and transport flow in-process.
- `remote` execution targets a deployed Modal app and can auto-deploy it on first use.
- Remote execution happens per connected remote component, not per individual marked node.
- Remote components can expand upstream automatically when a marked node depends on non-transportable Comfy runtime objects.

## What is implemented

- A frontend extension in [`web/modal_toggle.js`](web/modal_toggle.js) that adds a `Run on Modal` toggle, draws remote-state overlays, and queues prompts through `/modal/queue_prompt`.
- A backend route in [`api_intercept.py`](api_intercept.py) that reads workflow metadata, resolves nested subgraph markers, partitions remote components, syncs assets, and rewrites each component into a signature-preserving proxy node.
- A dynamic proxy registry in [`modal_executor_node.py`](modal_executor_node.py) that mirrors exported output counts and output types so ComfyUI validation still succeeds after rewrite.
- A content-addressable sync engine in [`sync_engine.py`](sync_engine.py) for model files and the optional `custom_nodes/` bundle.
- A remote runtime in [`remote/modal_app.py`](remote/modal_app.py) with local fallback execution for tests and Modal-backed execution when the SDK is available.
- A stable Modal cloud entrypoint in [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py) for deployable remote execution.

## How it works

1. You mark nodes with `Run on Modal`.
2. Queueing goes to `/modal/queue_prompt` instead of ComfyUI's normal `/prompt` route.
3. The backend reads `extra_pnginfo.workflow`, including nested subgraph metadata, and resolves those markers onto queued prompt node ids.
4. Connected remote-marked nodes are grouped into components, with upstream auto-expansion when required by non-transportable inputs.
5. Each component is replaced with a generated `ModalUniversalExecutor_<hash>` proxy node.
6. Boundary inputs are serialized, referenced assets are synced, and the component executes locally or remotely.
7. Exported boundary outputs are deserialized back into normal ComfyUI values for the rest of the graph.

## Installation

Install this repository under ComfyUI's `custom_nodes/` directory:

```bash
cd ~/git/ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Modal
```

When ComfyUI starts, it should load:

- the backend route from [`api_intercept.py`](api_intercept.py)
- the internal proxy node definitions from [`modal_executor_node.py`](modal_executor_node.py)
- the frontend extension from [`web/modal_toggle.js`](web/modal_toggle.js)

## Execution modes

### Local mode

Use local mode during development:

```bash
export COMFY_MODAL_EXECUTION_MODE=local
```

This still exercises prompt rewrite, asset sync, and serialization, but the proxied component runs in-process instead of on Modal.

By default, local mode skips hashing and zipping the entire `custom_nodes/` tree because that work is unnecessary for in-process execution and can make queue requests look stalled on larger ComfyUI installs.

Force custom-node bundle sync in local mode with:

```bash
export COMFY_MODAL_SYNC_CUSTOM_NODES=true
```

### Remote Modal mode

Remote mode requires:

- the `modal` Python package in the ComfyUI environment
- working Modal credentials
- a reachable Modal app and volume configuration
- `COMFY_MODAL_EXECUTION_MODE` set to something other than `local`

The normal remote path is deployed-app lookup by name. On first remote use, the extension can auto-deploy the stable cloud entrypoint from [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py) if the configured app does not exist yet.

```bash
export COMFY_MODAL_EXECUTION_MODE=remote
```

Remote deployment behavior:

- First-run auto-deploy is enabled by default.
- Persistent deployed apps are preferred over ephemeral `app.run()` execution.
- The extension does not create a persistent web endpoint for you.
- `COMFY_MODAL_AUTO_DEPLOY=false` disables first-run auto-deploy.
- `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true` re-enables the older temporary `app.run()` fallback path.

Remote runtime behavior:

- The deployed class reads its GPU type from `COMFY_MODAL_GPU`.
- CPU memory snapshots are enabled by default.
- GPU memory snapshots remain opt-in.
- The default `scaledown_window` is `600` seconds with `min_containers=0`.
- Warm containers can reuse loader state and `PromptExecutor` state across compatible requests.

If you change `COMFY_MODAL_GPU`, redeploy the Modal app or delete it and let auto-deploy recreate it. Modal hardware is fixed at deploy time.

## Using it in ComfyUI

### 1. Build your workflow

Build the workflow normally. This extension does not replace the editor's regular nodes. It adds a `Run on Modal` toggle to standard nodes.

Good candidates for remote execution are nodes that:

- consume large model files
- do expensive tensor work
- accept and return values that can cross the local/remote boundary cleanly

### 2. Understand the transport boundary

Boundary-crossing values must be transportable. In practice, supported evaluated values include tensor-like and primitive types such as:

- `IMAGE`
- `MASK`
- `LATENT`
- `SIGMAS`
- `NOISE`
- `INT`
- `FLOAT`
- `BOOLEAN`
- `STRING`

Comfy runtime objects such as `MODEL`, `CONDITIONING`, `CLIP`, `VAE`, and `CONTROL_NET` cannot cross the current local/remote boundary. If a marked node still depends on those values from the local side, queueing fails fast with a validation error.

To reduce manual graph editing, Modal-Sync can auto-expand a remote component upstream when a marked node depends on non-transportable inputs. In practice, you can often mark the downstream sampler or custom node you care about and let the backend pull the required upstream loaders or builders into the same remote island.

### 3. Mark remote regions

For each connected region you want to offload:

1. Find the `Run on Modal` toggle on the node.
2. Enable it on every node that should be part of the same remote island.
3. Confirm the node shows the blue remote-execution border.

The toggle stores `properties.is_modal_remote = true` in workflow metadata. The visible graph is not rewritten in the editor. Rewrite happens only at queue time.

### 4. Queue the workflow

Use ComfyUI's normal queue action. The frontend intercepts submission and sends the prompt to `/modal/queue_prompt`.

During queue-time processing, the backend:

1. Inspects `extra_pnginfo.workflow`, including nested saved subgraph fragments.
2. Resolves nested markers onto the actual queued prompt ids, including composed ids like `24:23` when needed.
3. Replaces each connected remote component with an internal `ModalUniversalExecutor_<hash>` proxy node.
4. Mirrors referenced model assets into storage.
5. Optionally syncs a zipped `custom_nodes/` bundle.
6. Submits the rewritten prompt to ComfyUI's normal queue.

### 5. Watch execution state

The frontend shows remote state directly on the canvas:

- blue border: marked for Modal, idle
- orange pulsing border: queued or still being prepared
- pulsing green border: ready in Modal and waiting
- pulsing purple border: currently executing remotely
- steady green border: finished for the current run
- red border: queue-time or execution failure

The extension also keeps a global activity badge visible during queue-time sync and remote execution, including the period before the prompt is formally queued.

### 6. Asset sync expectations

The sync engine automatically looks for inputs that resolve to files ending in:

- `.safetensors`
- `.ckpt`
- `.pt`
- `.vae`

In practice:

- absolute file paths work
- model names resolvable through ComfyUI `folder_paths` work
- arbitrary unresolved strings do not sync

If a remote-marked node depends on a model filename that cannot be resolved to a local file, prompt queueing fails instead of continuing with a broken remote request.

## Configuration

### Routing and metadata

- `COMFY_MODAL_ROUTE_PATH`: Override the queue endpoint. Default: `/modal/queue_prompt`.
- `COMFY_MODAL_MARKER_PROPERTY`: Override the workflow property used to mark nodes. Default: `is_modal_remote`.

### Local storage and sync

- `COMFY_MODAL_LOCAL_STORAGE_ROOT`: Local mirror root used for sync tests and dry runs. Default: `/tmp/comfyui-modal-sync-storage`.
- `COMFY_MODAL_CUSTOM_NODES_DIR`: Override the `custom_nodes` directory to bundle and mirror.
- `COMFY_MODAL_SYNC_CUSTOM_NODES`: Force-enable or disable custom-node bundle sync. Default: disabled in `local` mode, enabled otherwise.

### Execution and deployment

- `COMFY_MODAL_EXECUTION_MODE`: Set to `local` for in-process fallback execution. Default: `local`.
- `COMFY_MODAL_APP_NAME` and `COMFY_MODAL_VOLUME_NAME`: Override Modal app and volume naming.
- `COMFY_MODAL_AUTO_DEPLOY`: Automatically deploy the Modal app on first remote invocation when lookup fails. Default: `true`.
- `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK`: Re-enable slow `app.run()` fallback in remote mode. Default: `false`.

### Modal runtime sizing

- `COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT`: Enable Modal CPU memory snapshots. Default: `true`.
- `COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT`: Enable Modal GPU memory snapshots. Default: `false`.
- `COMFY_MODAL_GPU`: Modal GPU type to request when deploying the remote class. Default: `A100`.
- `COMFY_MODAL_SCALEDOWN_WINDOW`: Keep idle containers warm for this many seconds. Default: `600`.
- `COMFY_MODAL_MIN_CONTAINERS`: Keep at least this many containers warm. Default: `0`.

## Development

- Manage the project with `uv`.
- Run the test suite with `uv run pytest`.
- [`pyproject.toml`](pyproject.toml) provides pytest discovery configuration for `uv`.
- [`modal_test_workflow.json`](modal_test_workflow.json) is a checked-in smoke artifact from a successful Modal-path run, not a pristine authoring workflow.

## Current limitations

- Remote execution is component-based. If you leave a local gap in the middle of a would-be remote chain, the boundary still has to be transport-safe.
- The default sync backend is still a local mirror used for development and tests. Real Modal execution depends on a working Modal SDK environment.
- Non-JSON, non-bytes, non-tensor payloads are not supported across the current local/remote boundary.
- Workflow artifacts captured after a remote run may include internal proxy nodes such as `ModalUniversalExecutor`; they are useful as regression fixtures, but should not be treated as clean source workflows.
