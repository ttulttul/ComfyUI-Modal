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

### Performance findings

- Upload and sync overhead are already small in the captured runs. Queue-time rewrite and sync took about `1.25s` on the first run and `0.214s` on the second run, and the custom-nodes bundle was already mirrored in both runs.
- Volume reload is not currently a material cost for this workflow. Both runs logged `Skipping modal_volume_reload ... because no new assets were uploaded`.
- The dominant recurring non-sampling cost is remote model loading inside the static phase. On both runs, `UNETLoader` took about `14.4s` and `CLIPLoader` took about `10.0s`, so roughly `24-25s` per run is spent reloading models even when workflow structure is otherwise unchanged.
- The second run proved that local ComfyUI proxy caching is now helping, but it also showed the next bottleneck clearly: the mapped phase still has to replay the static producer work to rebuild session-backed state before sampling starts.
- Persisted Modal `Dict` caching is helping some small nodes, but it is not touching the dominant cost centers in this workflow because the expensive static loader outputs are not being reused from a warm in-process loader cache.
- Modal container startup is still expensive, but keeping workers hot full-time is not the preferred answer for this project. Performance work should prioritize better reuse within a worker lifecycle and less replay when a worker is already available.

### Optimization plan

1. Add warm-worker loader reuse for heavy model nodes.
   Focus on `UNETLoader` and `CLIPLoader` first, since the logs show they dominate non-sampling runtime.
   Reuse loaded outputs across compatible requests on the same Modal worker using a stable loader-input key and clone-or-safe-share semantics.
   Acceptance criteria: on a same-worker repeat run, those loader nodes log `Loader cache hit` instead of `Loader cache miss`, and the static phase drops by roughly the combined `~24s` loader cost seen in the current logs.

2. Avoid full static-phase replay for session-backed mapped runs when bridge refs are still valid locally.
   The second run shows that local proxy caching skipped the static proxy dispatch, but the mapped phase still replays the static producer branch to rebuild session-backed state.
   Add a fast path where the mapped phase can resolve durable bridge refs directly into the current session without rerunning the whole static producer subgraph when the required bridge values are already reconstructible from cached durable metadata or retained in-process state.
   Acceptance criteria: on an unchanged repeat run, the mapped phase no longer re-enters the full static producer prompt before `KSampler` starts, and time-to-sampling shrinks materially.

3. Add targeted observability for replay cost breakdown.
   Emit explicit timers for bridge-record lookup, replayed static-subgraph execution, session-value restoration, and loader-cache hit/miss counts.
   Right now the logs make replay visible, but not cheap to quantify automatically.
   Acceptance criteria: one log block per mapped phase shows how much time was spent in direct bridge resolution versus replay and how many heavy loaders were reused.

4. Reduce remote custom-node initialization churn where possible.
   `init_external_custom_nodes` is only about `~1s`, so it is not the first optimization target, but it is still worth trimming after loader reuse lands.
   Investigate whether the worker can avoid repeated import-error noise and repeated custom-node init work once the bundle digest is unchanged.
   Acceptance criteria: repeated runs on the same worker avoid redundant custom-node initialization work and produce cleaner logs.

5. Keep monitoring session-routing quality under real scale-out.
   This is durability work with performance impact, not the first speed target.
   If session churn still forces replay too often under scale-out, the next structural change is stronger worker/session routing or an explicit leased-worker/session model.
   Acceptance criteria: mirrored logs show low replay frequency for ordinary repeat workflows, and session misses remain rare under moderate parallel use.
