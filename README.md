# ComfyUI Modal-Sync

`ComfyUI Modal-Sync` is a ComfyUI custom node extension that lets you mark individual nodes for remote execution. The frontend stores a per-node `is_modal_remote` flag in workflow metadata, the backend rewrites those nodes into a Modal proxy before queueing, and the proxy forwards execution payloads through a serialization layer that is safe for tensors and plain JSON values.

## What is implemented

- A frontend extension in [web/modal_toggle.js](/home/ksimpson/git/ComfyUI-Modal/web/modal_toggle.js) that injects a `Run on Modal` toggle into every non-internal node, draws a blue border around remote nodes, and submits queued prompts to `/modal/queue_prompt`.
- A backend route in [api_intercept.py](/home/ksimpson/git/ComfyUI-Modal/api_intercept.py) that reads the workflow snapshot from `extra_pnginfo.workflow`, identifies remote-marked nodes, syncs referenced assets, and rewrites each remote node into a signature-preserving Modal proxy node.
- A dynamic proxy registry in [modal_executor_node.py](/home/ksimpson/git/ComfyUI-Modal/modal_executor_node.py) that mirrors the original node’s output count and output types so ComfyUI validation still succeeds after rewrite.
- A content-addressable sync engine in [sync_engine.py](/home/ksimpson/git/ComfyUI-Modal/sync_engine.py) that mirrors model files and a zipped `custom_nodes/` bundle into storage keyed by SHA256.
- A remote runtime skeleton in [remote/modal_app.py](/home/ksimpson/git/ComfyUI-Modal/remote/modal_app.py) that supports local fallback execution for tests and can call into Modal when the SDK is available.

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
3. Nodes with `properties.is_modal_remote = true` are replaced with a generated `ModalUniversalExecutor_<hash>` node id.
4. File-like widget inputs ending in `.safetensors`, `.ckpt`, `.pt`, or `.vae` are resolved to local files and mirrored into storage.
5. The executor serializes tensors with `safetensors`, forwards the payload to the remote runtime, and deserializes the returned outputs back into ComfyUI values.

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

### 3. Open your workflow in ComfyUI

Build your workflow normally. This pack does not replace regular nodes in the editor. Instead, it adds a `Run on Modal` toggle to standard nodes.

The best candidates for remote execution are nodes that:

- consume large model files
- do expensive tensor work
- have inputs and outputs that are JSON values, bytes, or tensors

Avoid marking nodes remote if they depend on local-only process state or return objects that cannot be serialized by [serialization.py](/home/ksimpson/git/ComfyUI-Modal/serialization.py).

### 4. Mark the nodes you want to offload

For each node you want to run remotely:

1. Find the `Run on Modal` toggle on the node.
2. Enable it.
3. Confirm the node shows the blue remote-execution border.

That toggle stores `properties.is_modal_remote = true` in the workflow metadata. Nothing is rewritten in the canvas itself. The rewrite happens only when you queue the prompt.

### 5. Queue the workflow

Use the normal queue button in ComfyUI.

The frontend extension intercepts queue submission and sends the prompt to `/modal/queue_prompt` instead of `/prompt`. On the backend:

1. The request handler inspects the workflow snapshot in `extra_pnginfo.workflow`.
2. Every node marked with `is_modal_remote` is replaced with an internal `ModalUniversalExecutor_<hash>` proxy node.
3. Referenced model assets are mirrored into storage.
4. If custom-node syncing is enabled, the local `custom_nodes/` directory is zipped and mirrored if its content hash changed.
5. The rewritten prompt is submitted to the normal ComfyUI prompt queue.

### 6. What happens during execution

When execution reaches a rewritten node:

- the proxy receives the original node metadata plus the evaluated upstream inputs
- tensor inputs are serialized with `safetensors`
- the proxy calls the remote runtime
- the remote runtime imports the original node class and executes it
- outputs are serialized back to the local process and returned to the rest of the workflow

This is why the backend generates proxy node classes that preserve the original output signature: downstream validation still has to see the right output count and types.

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

- The rewrite is node-by-node, not subgraph-by-subgraph.
- Remote-to-remote chains still bounce through the local boundary between nodes.
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

- The graph rewrite is per-node, not whole-subgraph partitioning. Remote-to-remote chains will currently round-trip through the local executor boundary between nodes.
- The default sync backend is a local mirror used for development and tests. The Modal runtime entrypoint is present, but real cloud execution still depends on a working Modal SDK environment.
- Non-JSON, non-bytes, non-tensor values cannot be serialized across the remote boundary yet. Unsupported types raise immediately instead of silently degrading.
