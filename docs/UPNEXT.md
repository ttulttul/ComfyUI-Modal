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
- The static proxy returns normal transportable outputs immediately to local consumers.
- Static-to-mapped non-transportable boundaries now return opaque session refs instead of raw runtime objects.
- The remote runtime stores those bridge values in a prompt-scoped in-memory session and resolves them when the mapped proxy runs.
- The mapped proxy clears the session after completion.
- Prompt metadata is recomputed after rewrite so execution stages and dependency edges reflect the real split proxies.

### Why this shape

Partial-completion semantics for one proxy would require ComfyUI scheduler behavior that does not exist today. Split proxies keep completion semantics simple: each proxy either returns or it does not.

### Remaining work

- Online testing confirmed that `RemoteEngine(session_affinity_key)` kept the split static and mapped proxies on the same deployed Modal worker for the tested workflow.
- Add a broader workflow-level regression on top of the new proxy-level test coverage if we need full scheduler-path protection beyond the current split-proxy execution test.
- Watch live logs for session create/reuse/cleanup and add a fallback path if deployed affinity turns out not to be sticky enough.
