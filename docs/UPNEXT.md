# Up Next

## Early local delivery from hybrid remote components

### Problem

Some remote outputs are logically ready before the rest of the remote work is done, but the local ComfyUI graph still cannot consume them yet.

Concrete failing shape:

```text
Node4_Remote -> Node9_Local
Node12_Remote -> Node13_Local
Node12_Remote -> Node39_Remote
Node39_Remote -> Node40_Local
```

Observed behavior:

- `Node4_Remote` finishes inside the remote run.
- `Node9_Local` still does not start until `Node39_Remote` finishes.

### Why it happens

This case is being rewritten as one hybrid `mapped_subgraph` proxy instead of separate remote proxies for the static and mapped branches.

- Queue-time rewrite builds one payload with `static_phase` and `mapped_phase`.
- The remote runtime executes the static phase first, then the mapped phase.
- Static outputs are merged with mapped outputs and returned once at the end of the proxy run.
- Local ComfyUI can only unblock downstream nodes when that proxy returns.

So even though `Node4_Remote` is done, its result is still trapped inside the running proxy invocation.

### Scope of current streaming

Current streaming is only enough for UI-facing events:

- progress updates
- `executed` UI payloads
- preview-image boundary events

It is not arbitrary local-graph input delivery. A finished remote node output does not currently become a usable local edge value while sibling remote work in the same hybrid proxy is still running.

### Evidence

Existing logs for the failing prompt show the static/mapped split inside one remote component:

- `Node4` is persisted under `component=1::static`
- `Node39` executes under the mapped side of that same component

That matches the observed behavior: `Node4` finishes, but `Node9_Local` still waits for the single proxy result.

### Implementation directions

Two designs look viable:

1. Partial-completion semantics for one proxy node.
   The proxy would need to emit boundary outputs incrementally and give the local scheduler a safe way to treat those outputs as ready before the proxy has fully completed.

2. Session-scoped remote state shared across multiple proxies.
   Queue-time rewrite would split the static and mapped work into separate proxies, while the remote side keeps non-transportable state like `MODEL` alive in a shared session/cache so the mapped proxy can reuse it without serializing it across the boundary.

### Recommendation

The second option looks safer with ComfyUI's execution model.

- Keep proxy completion semantics simple: a proxy either returns or it does not.
- Split hybrid components only when the mapped side can reconnect to prior remote state through a session handle.
- Treat transportable outputs as normal boundaries and non-transportable outputs as in-session references.

That would allow `Node4_Remote -> Node9_Local` to unblock early without pretending that ComfyUI supports partially completed node outputs natively.
