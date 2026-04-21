# Up Next

## Hybrid split-proxy rollout

### Goal

Allow transportable static outputs from a hybrid remote component to unblock downstream local nodes before the mapped sibling branch finishes.

Concrete shape:

```text
Node4_Remote -> Node9_Local
Node12_Remote -> Node13_Local
Node12_Remote -> Node39_Remote
Node39_Remote -> Node40_Local
```

Desired behavior:

- `Node4_Remote` completes inside the static proxy.
- `Node9_Local` starts as soon as that static proxy returns.
- `Node39_Remote` continues in the mapped proxy using shared remote-only state from the same prompt-scoped session.

### Implemented

- Hybrid components with both static and mapped remote subsets now rewrite into two ordinary `subgraph` proxies instead of one hybrid `mapped_subgraph` proxy.
- Ordinary remote components with multiple local-exporting execute branches can now also rewrite into ordered ordinary `subgraph` proxies when they only share remote-only upstream state.
- The static proxy returns normal transportable outputs immediately to local consumers.
- Static-to-mapped non-transportable boundaries now return opaque session refs instead of raw runtime objects.
- The remote runtime stores those bridge values in a prompt-scoped in-memory session and resolves them when the mapped proxy runs.
- The mapped proxy clears the session after completion.
- Prompt metadata is recomputed after rewrite so execution stages and dependency edges reflect the real split proxies.
- A checked-in workflow-shaped regression artifact now locks the split static-plus-mapped rewrite behavior against future drift.
- Session create, reuse, resolve, storage, and cleanup now emit explicit observability logs in the shared store and both runtime entrypoints.
- Streamed remote executions now also surface their `MODAL_TASK_ID`, and the local runtime can optionally mirror that container's live logs into local stderr, preferring `modal container logs -f` with an SDK fallback only when the CLI is unavailable.

### Why this shape

Partial-completion semantics for one proxy would require ComfyUI scheduler behavior that does not exist today. Split proxies keep completion semantics simple: each proxy either returns or it does not.

### Current state

- Online testing confirmed the intended split-proxy behavior: early local unblocking works, session-backed split phases can reuse remote-only state, and unchanged second runs can now be skipped locally instead of dispatching a redundant remote proxy call.
- Session-backed bridges are durable now. The fast path still uses prompt-scoped in-memory session values, but replay metadata also lives in a second Modal `Dict`, so later split phases can recover from container churn without replaying dead session refs.
- Split-phase caching and batching have been hardened around the real workflow edges we saw online: provenance-based cache keys for non-transportable boundary inputs, correct handling of local `ModalMapInput` markers, broadcast `CONDITIONING`, per-item session-ref batches, primitive over-list inputs, singleton semantic lists, and `INPUT_IS_LIST` targets.
- The frontend and observability paths are also in better shape: component-scoped progress lanes no longer flicker, single-node components clear their purple state correctly, and `COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS=true` can mirror remote container logs into local stderr during streamed runs.

### Remaining work

- Keep watching mirrored live logs under broader scale-out for real session misses or excessive replay after container churn.
- If production still shows too many split-proxy session misses, strengthen worker/session routing instead of piling on more replay logic. The likely next step is stronger affinity or an explicit leased-worker/session model rather than best-effort class-instance affinity.
