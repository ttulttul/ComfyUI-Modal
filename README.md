# ComfyUI Modal-Sync

> [!WARNING]
> This is an early alpha-level project. It is not ready for general use yet, the execution model is still incomplete, and you should expect missing features, hard limitations, and breaking changes.

`ComfyUI Modal-Sync` is a ComfyUI custom node extension that lets you mark workflow regions for remote execution. The frontend stores a per-node `is_modal_remote` flag in workflow metadata, the backend groups connected remote-marked nodes into a remote component before queueing, and the proxy forwards that component through a serialization layer that is safe for tensors and plain JSON values.

## What is implemented

- A frontend extension in [web/modal_toggle.js](/home/ksimpson/git/ComfyUI-Modal/web/modal_toggle.js) that injects a `Run on Modal` toggle into every non-internal node, draws live remote-state overlays on marked nodes, and submits queued prompts to `/modal/queue_prompt`.
- A backend route in [api_intercept.py](/home/ksimpson/git/ComfyUI-Modal/api_intercept.py) that reads the workflow snapshot from `extra_pnginfo.workflow`, identifies connected remote-marked node components, syncs referenced assets, and rewrites each component into one signature-preserving Modal proxy node.
- A dynamic proxy registry in [modal_executor_node.py](/home/ksimpson/git/ComfyUI-Modal/modal_executor_node.py) that mirrors the exported output count and output types of each remote component so ComfyUI validation still succeeds after rewrite.
- A content-addressable sync engine in [sync_engine.py](/home/ksimpson/git/ComfyUI-Modal/sync_engine.py) that mirrors model files and a zipped `custom_nodes/` bundle into storage keyed by SHA256.
- A remote runtime skeleton in [remote/modal_app.py](/home/ksimpson/git/ComfyUI-Modal/remote/modal_app.py) that supports local fallback execution for tests and can call into Modal when the SDK is available.
- A stable Modal cloud entry in [comfyui_modal_sync_cloud.py](/home/ksimpson/git/ComfyUI-Modal/comfyui_modal_sync_cloud.py) that packages the local ComfyUI checkout plus the core ComfyUI Python runtime dependencies into the remote image.

## Repository layout

```text
.
├── __init__.py
├── api_intercept.py
├── modal_executor_node.py
├── remote/
│   ├── __init__.py
│   └── modal_app.py
├── serialization.py
├── settings.py
├── sync_engine.py
├── tests/
└── web/
    └── modal_toggle.js
```

## Runtime behavior

1. Queueing a prompt posts to `/modal/queue_prompt`.
2. The route loads the workflow metadata from `extra_data.extra_pnginfo.workflow`.
3. Connected regions of nodes with `properties.is_modal_remote = true` are replaced with generated `ModalUniversalExecutor_<hash>` proxy nodes.
4. File-like widget inputs ending in `.safetensors`, `.ckpt`, `.pt`, or `.vae` are resolved to local files and mirrored into storage.
5. The executor serializes the component boundary inputs with `safetensors`, forwards the remote component payload to the runtime, and deserializes the exported boundary outputs back into ComfyUI values.

## HOWTO

This pack is meant to be installed as a normal ComfyUI custom node package, then used by marking specific workflow nodes for remote execution.

### 1. Install the pack into ComfyUI

Put this repository under your ComfyUI `custom_nodes/` directory. For example:

```bash
cd ~/git/ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Modal
```

When ComfyUI starts, it should load:

- the backend route from [api_intercept.py](/home/ksimpson/git/ComfyUI-Modal/api_intercept.py)
- the internal proxy node definitions from [modal_executor_node.py](/home/ksimpson/git/ComfyUI-Modal/modal_executor_node.py)
- the frontend extension from [web/modal_toggle.js](/home/ksimpson/git/ComfyUI-Modal/web/modal_toggle.js)

### 2. Decide whether to run in local fallback mode or real Modal mode

For development, the current implementation supports a local fallback executor:

```bash
export COMFY_MODAL_EXECUTION_MODE=local
```

That path still exercises the same prompt rewrite, asset sync, and serialization logic, but it executes the proxied node in-process instead of calling Modal.

In `local` mode, the extension now skips hashing and zipping the entire `custom_nodes/` tree by default. That work is unnecessary for in-process execution and can make queue requests appear stuck on larger ComfyUI installs.

If you want to force custom-node bundle sync even in local mode, set:

```bash
export COMFY_MODAL_SYNC_CUSTOM_NODES=true
```

For real Modal execution, you need:

- the `modal` Python package available in the ComfyUI environment
- a valid Modal account/session
- a reachable Modal app/volume configuration
- `COMFY_MODAL_EXECUTION_MODE` set to something other than `local`

The remote runtime skeleton lives in [remote/modal_app.py](/home/ksimpson/git/ComfyUI-Modal/remote/modal_app.py). The extension is structured for Modal, but you still need to supply a working Modal environment.

When `COMFY_MODAL_EXECUTION_MODE` is remote, the runtime now tries a deployed Modal class lookup first via the configured app name. If no deployed app is available, it falls back to starting an ephemeral `app.run()` session locally for the invocation. That fallback is slower, but it avoids the old "Function has not been hydrated" failure mode.

Important distinction:
- the current fallback can create a temporary Modal app session automatically
- it does not auto-deploy a persistent Modal app
- it does not auto-create a persistent web endpoint
- the actual Modal cloud service now lives in a stable importable module, [comfyui_modal_sync_cloud.py](/home/ksimpson/git/ComfyUI-Modal/comfyui_modal_sync_cloud.py), because ComfyUI’s custom-node loader does not assign Modal-safe module names to node-pack directories
- the Modal image now installs the core ComfyUI runtime Python packages automatically, but remote execution may still surface additional environment gaps as broader workflows are exercised
- the Modal image filter now preserves internal ComfyUI Python packages such as `comfy/ldm/models` while still excluding top-level runtime asset folders like `models/` and `output/`
- the remote worker now initializes ComfyUI's built-in extra nodes and API nodes on first use, and loads extracted custom-node bundles through ComfyUI's normal `custom_nodes` registry path
- mirrored asset references such as `/assets/<sha>_model.safetensors` are now materialized to container-local absolute paths before remote execution, and the worker patches ComfyUI's `folder_paths` lookups so normal loader nodes still accept them
- in real remote mode, queue-time asset sync now uploads into the named Modal volume instead of only mirroring to local disk, and the worker reloads that mounted volume before each execution so warm containers can see newly uploaded assets
- the queue route now performs Modal volume SDK calls from a worker thread instead of directly on the aiohttp request loop, and missing marker paths in the volume are treated as normal cache misses rather than fatal errors
- the remote worker now force-loads ComfyUI's top-level `utils` package from the bundled ComfyUI source tree before API nodes initialize, which avoids third-party or stray `utils` modules shadowing `utils.install_util`

If you want a stable reusable Modal deployment, that is still a separate step outside the current node-pack behavior.

### 3. Open your workflow in ComfyUI

Build your workflow normally. This pack does not replace regular nodes in the editor. Instead, it adds a `Run on Modal` toggle to standard nodes.

The best candidates for remote execution are nodes that:

- consume large model files
- do expensive tensor work
- have inputs and outputs that are JSON values, bytes, or tensors

Avoid marking nodes remote if they depend on local-only process state or return objects that cannot be serialized by [serialization.py](/home/ksimpson/git/ComfyUI-Modal/serialization.py).

Under the current implementation, a remote-marked component can only consume or export boundary-crossing values whose evaluated values are transportable. In practice that means tensor-like and primitive types such as:

- `IMAGE`
- `MASK`
- `LATENT`
- `SIGMAS`
- `NOISE`
- `INT`
- `FLOAT`
- `BOOLEAN`
- `STRING`

Inputs or outputs that evaluate to Comfy runtime objects such as `MODEL`, `CONDITIONING`, `CLIP`, `VAE`, `CONTROL_NET`, and similar internal types cannot cross the current local/remote boundary. If you mark a node like `KSampler` remote while its `model` or `positive`/`negative` inputs are still produced locally, the queue request will now be rejected immediately with a validation error instead of failing later during execution.

### 4. Mark the nodes you want to offload

For each connected region you want to run remotely:

1. Find the `Run on Modal` toggle on the node.
2. Enable it on every node that should belong to the same remote island.
3. Confirm the node shows the blue remote-execution border.

That toggle stores `properties.is_modal_remote = true` in the workflow metadata. Nothing is rewritten in the canvas itself. The rewrite happens only when you queue the prompt.

### 5. Queue the workflow

Use the normal queue button in ComfyUI.

The frontend extension intercepts queue submission and sends the prompt to `/modal/queue_prompt` instead of `/prompt`. On the backend:

1. The request handler inspects the workflow snapshot in `extra_pnginfo.workflow`.
2. Connected remote-marked nodes are partitioned into remote components and each component is replaced with an internal `ModalUniversalExecutor_<hash>` proxy node.
3. Referenced model assets are mirrored into storage.
4. If custom-node syncing is enabled, the local `custom_nodes/` directory is zipped and mirrored if its content hash changed.
5. The rewritten prompt is submitted to the normal ComfyUI prompt queue.

While that is happening, the frontend now shows remote-node state directly on the canvas:

- blue border: marked for Modal, idle
- orange pulsing border: queued or still being prepared for remote execution
- green border with light green shading: the remote component is actively executing
- red border: queue-time or execution failure detected

Because the Modal queue route can spend noticeable time hashing, syncing, and creating a remote runtime before the prompt is formally queued, the frontend also emits a temporary synthetic running state into ComfyUI's normal queue/execution UI. Those events now mirror ComfyUI's native websocket payload shapes, including the temporary "Waiting for a machine" initialization notification, so the built-in queue indicators stay alive during the preparatory phase instead of looking idle until the backend finally returns the queued prompt id.

### 6. What happens during execution

When execution reaches a rewritten component proxy:

- the proxy receives the remote component metadata plus the evaluated boundary inputs
- tensor boundary inputs are serialized with `safetensors`
- the proxy calls the remote runtime
- the remote runtime executes the full connected remote subgraph
- exported boundary outputs are serialized back to the local process and returned to the rest of the workflow

This is why the backend generates proxy node classes that preserve the exported output signature: downstream validation still has to see the right output count and types.

### 7. Asset expectations

The current sync engine automatically looks for prompt inputs that resolve to files ending in:

- `.safetensors`
- `.ckpt`
- `.pt`
- `.vae`

Those files are mirrored into content-addressed storage. In practice that means:

- absolute file paths work
- model names that ComfyUI can resolve through `folder_paths` work
- arbitrary unresolved strings do not sync

If a remote-marked node depends on a model filename that cannot be resolved to a local file, prompt queueing will fail instead of silently continuing.

### 8. Operational caveats

Use this pack with the current limits in mind:

- The rewrite is component-based, but only for connected regions that are explicitly marked remote.
- Only the component boundary is serialized. Non-transportable Comfy runtime objects still cannot cross between local and remote regions.
- Real Modal execution is scaffolded, but you still need to wire the actual Modal environment and credentials.
- Non-JSON, non-bytes, non-tensor payloads are not supported across the remote boundary.

## Configuration

These environment variables are supported:

- `COMFY_MODAL_ROUTE_PATH`: Override the queue endpoint. Default: `/modal/queue_prompt`.
- `COMFY_MODAL_MARKER_PROPERTY`: Override the workflow property used to mark nodes. Default: `is_modal_remote`.
- `COMFY_MODAL_LOCAL_STORAGE_ROOT`: Local mirror root used for sync tests and dry runs. Default: `/tmp/comfyui-modal-sync-storage`.
- `COMFY_MODAL_CUSTOM_NODES_DIR`: Override the `custom_nodes` directory to bundle and mirror.
- `COMFY_MODAL_EXECUTION_MODE`: Set to `local` for in-process fallback execution. Default: `local`.
- `COMFY_MODAL_SYNC_CUSTOM_NODES`: Force-enable or disable custom-node bundle sync. Default: disabled in `local` mode, enabled otherwise.
- `COMFY_MODAL_APP_NAME` and `COMFY_MODAL_VOLUME_NAME`: Override Modal app and volume naming.

## Development

- Run tests with `uv run pytest`.
- The project includes a minimal [pyproject.toml](/home/ksimpson/git/ComfyUI-Modal/pyproject.toml) so pytest discovery is consistent under `uv`.

## Current limitations

- Remote execution now happens per connected remote component, not per disconnected set of marked nodes. If you leave a local gap in the middle of a would-be remote chain, the boundary still has to be transport-safe.
- The default sync backend is a local mirror used for development and tests. The Modal runtime entrypoint is present, but real cloud execution still depends on a working Modal SDK environment.
- Non-JSON, non-bytes, non-tensor values cannot cross the local/remote boundary yet. Unsupported boundary types raise immediately instead of silently degrading.
