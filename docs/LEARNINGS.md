# Learnings

## 2026-04-17

- The workflow metadata already rides along in `extra_data.extra_pnginfo.workflow`, so the backend can discover `node.properties.is_modal_remote` without any extra frontend metadata channel.
- A single wildcard-output proxy node is not enough for server-side graph rewriting. ComfyUI validates downstream link indices and output arity against the replacement class, so the backend has to generate proxy node classes that mirror the original node’s output signature before validation runs.
- Keeping the sync engine content-addressable makes prompt rewrites cheap after the first upload. The route can block on sync safely as long as repeated requests collapse to a hash marker check.
- Hashing and zipping the entire `custom_nodes/` tree inside the queue route is too expensive for `COMFY_MODAL_EXECUTION_MODE=local`. Local mode should skip custom-node bundle sync by default, with an explicit override when that path needs to be exercised.
