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

### Remaining work

- Online testing confirmed that `RemoteEngine(session_affinity_key)` kept the split static and mapped proxies on the same deployed Modal worker for the tested workflow.
- Online testing also exposed one implicit-mapped runtime edge: list-backed `CONDITIONING` inputs must stay broadcast across mapped item runs. That regression is now covered so future batching changes do not reintroduce the `convert_cond` failure shape.
- Online testing also exposed a primitive-socket validation edge: upstream or boundary-fed `list(INT)` values are legitimate for nodes that rely on ComfyUI's over-list execution semantics, so only raw widget list literals should fail early.
- Online testing also exposed a boundary-normalization edge: singleton wrapper cleanup must preserve single-entry semantic lists like `CONDITIONING` while still collapsing true scalar wrappers and wrapped prompt links.
- Online testing also exposed one more implicit-batching edge: if a split boundary input lands on an `INPUT_IS_LIST` node, outer Modal fan-out is the wrong model. That case now stays as one ordinary remote subgraph run so ComfyUI can handle any downstream over-list behavior internally without duplicating scalar sibling outputs.
- Combined local+remote logs also exposed the real remaining split-proxy risk under load: a later phase can still land on a different Modal container and miss the prompt-scoped session bucket entirely. We now avoid killing that second container on session misses, but the underlying product fix is still stronger container affinity or explicit leased-worker/session routing.
- Rewrite also has to honor local `ModalMapInput` markers. That case is now covered so `list(INT)` outputs from local nodes like `NextSeeds` still become mapped remote item runs instead of ordinary remote subgraphs that pass the whole list into one sampler call.
- Implicit batching also has to recognize per-item remote session refs for nontransportable types. Lists of `MODEL`/`CONDITIONING` session refs now split item-by-item, while raw conditioning payload lists still stay broadcast.
- Implicit mapped cleanup now also runs through one dedicated final payload instead of letting each `::static` or `::item:*` sub-run inherit `clear_remote_session=True`. That keeps shared prompt-scoped state alive until all sibling mapped runs finish.
- Frontend progress lanes are now scoped by remote component representative, and streamed runs still accept the proxy's native completion event. That fixes both the stuck-purple single-node component case and the purple/green flicker when two different remote components are sampling in parallel with the same lane numbers.
- Run with `COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS=true` during broader production workflows and watch the mirrored session lifecycle logs for any real session misses under scale-out.
- If production logs ever show split-proxy session misses under real scale-out, add a stronger affinity fallback or explicit leased-worker/session routing instead of relying on best-effort class-instance affinity.
