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

## Configuration

These environment variables are supported:

- `COMFY_MODAL_ROUTE_PATH`: Override the queue endpoint. Default: `/modal/queue_prompt`.
- `COMFY_MODAL_MARKER_PROPERTY`: Override the workflow property used to mark nodes. Default: `is_modal_remote`.
- `COMFY_MODAL_LOCAL_STORAGE_ROOT`: Local mirror root used for sync tests and dry runs. Default: `/tmp/comfyui-modal-sync-storage`.
- `COMFY_MODAL_CUSTOM_NODES_DIR`: Override the `custom_nodes` directory to bundle and mirror.
- `COMFY_MODAL_EXECUTION_MODE`: Set to `local` for in-process fallback execution. Default: `local`.
- `COMFY_MODAL_APP_NAME` and `COMFY_MODAL_VOLUME_NAME`: Override Modal app and volume naming.

## Development

- Run tests with `uv run pytest`.
- The project includes a minimal [pyproject.toml](/home/ksimpson/git/ComfyUI-Modal/pyproject.toml) so pytest discovery is consistent under `uv`.

## Current limitations

- The graph rewrite is per-node, not whole-subgraph partitioning. Remote-to-remote chains will currently round-trip through the local executor boundary between nodes.
- The default sync backend is a local mirror used for development and tests. The Modal runtime entrypoint is present, but real cloud execution still depends on a working Modal SDK environment.
- Non-JSON, non-bytes, non-tensor values cannot be serialized across the remote boundary yet. Unsupported types raise immediately instead of silently degrading.
