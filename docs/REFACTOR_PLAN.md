# Modularity Refactoring Plan

Branch context: written on `add-r2-backing`, 2026-08-29. Goal: reduce the size of
individual Python files and improve modularity **without changing functionality**.
The repo is ~106k lines of Python; three files (`api_intercept.py` 10.1k,
`remote/modal_app.py` 9.1k, `comfyui_modal_sync_cloud.py` 8.1k) hold ~26k of it,
with a second tier of 1–3k-line modules behind them.

Target: no production module larger than ~1,500 lines; each extracted module owns
one responsibility and its own mutable state.

---

## Ground truths that shape the plan

These were verified against the source and must be respected by every step.

1. **The intra-repo import graph is an acyclic DAG.** No cycles anywhere in the
   49 root modules. Extractions are mechanical, not architectural surgery.

2. **`comfyui_modal_sync_cloud.py` must keep its name and top-level location.**
   It is *not* imported by `__init__.py`; it is loaded by
   `remote/modal_app.py::_load_modal_cloud_module` via
   `spec_from_file_location` under the stable flat module name
   `comfyui_modal_sync_cloud` (see `docs/LEARNINGS.md` — Modal needs a stable
   importable module name). It puts the repo root on `sys.path` and uses **flat
   absolute imports** (`from runtime_environment import ...`). Extracted helpers
   must be importable both flat (in the container) and as package members
   (locally) — the existing dual-import pattern already solves this.

3. **`remote/modal_app.py` has NO single-file deployment constraint.** It runs
   only in the local ComfyUI host process. The deployed Modal app is
   `comfyui_modal_sync_cloud.py`, and the image ships the **whole repo** via
   `image.add_local_dir(_REPO_ROOT, ...)` — new files anywhere in the repo are
   automatically included. Splitting `modal_app.py` is deployment-safe.

4. **Moving/adding `.py` files changes `repo_source_digest`** (the runtime
   fingerprint walks the tree recursively), forcing one image rebuild/redeploy
   per landed phase. Expected, not a blocker.

5. **Tests are the dominant risk, not production imports.** Production public
   surfaces are tiny (`api_intercept.py` exports exactly one name used by
   `__init__.py`: `setup_modal_queue_route`; `remote/__init__.py` lazily exports
   three names). But `tests/conftest.py` hands whole module objects to tests as
   fixtures, and tests read *and monkeypatch* dozens of private names and
   module-level mutable globals. Two hard rules follow:
   - A moved **function** that internal code still calls through the old module
     namespace can be re-exported, but `monkeypatch.setattr(old_module, ...)`
     only works if the *calling* code resolves the symbol through that same
     namespace at call time. When a function moves, update the tests to patch
     its new home module in the same commit.
   - A moved **mutable global** (cache dict, lock, flag) must NOT be re-exported
     via `from x import _CACHE` — tests that clear/patch it would silently
     operate on a different binding. Keep each mutable global in exactly one
     module and update tests to reach it there.

6. **The dual-import pattern is mandatory** for every new root-level module:
   ```python
   if __package__:
       from .foo import bar
   else:  # flat import inside the Modal container / pytest top-level
       from foo import bar
   ```
   29 modules already carry this block; replicate it verbatim in new files.

7. **Do not create generic new top-level package names** (`llm/`, `sync/`,
   `utils/`, `storage/`). The repo root itself goes on `sys.path` in the
   container alongside the ComfyUI checkout and site-packages, so any top-level
   name becomes globally importable there (`llm` collides with a real PyPI
   package). New subpackages either nest under one new distinctive name
   (Phase 7) or use distinctive flat filenames.

8. **`remote/` ships to SSH/Vast workers as a top-level package** with a
   minimal-dependency contract (`remote/r2_materializer.py` deliberately imports
   only `requests`; workers run `python -m remote.ssh_worker`). Host-side
   modules may be added under `remote/` (Modal ships the whole repo anyway), but
   the worker entrypoints must never import them.

9. Two tests grep **raw source text** of `comfyui_modal_sync_cloud.py` for the
   literals `worker_affinity_key: str = modal.parameter` and
   `def keepalive_for_local_gap` — both must remain literally in that file.

Process rules per repo convention (AGENTS.md): one commit per numbered step,
full test suite (`uv run pytest`) green before each commit, README/LEARNINGS
updates with major steps.

---

## Phase 0 — Preparation (cheap, unblocks everything)

0.1 **Consolidate mid-file globals.** `comfyui_modal_sync_cloud.py` declares
    globals at lines ~1803–1818 and ~6468 rather than in its header;
    `remote/modal_app.py` has a 240-line state block. Move all module-level
    state declarations to the top of each file, grouped by the cluster that owns
    them, with a comment naming the cluster. Pure motion inside the file; makes
    every later extraction diff reviewable.

0.2 **Extract shared constants.** `_BOUNDARY_INPUT_SIGNATURES_KEY` and
    `_PRIMITIVE_WIDGET_INPUT_TYPES` are duplicated verbatim between
    `comfyui_modal_sync_cloud.py` and `remote/modal_app.py`. Create
    `remote_protocol` additions (or a new `boundary_constants.py`) holding one
    copy; both files import it (dual-import pattern).

0.3 **Decide the `modal` guard strategy once.** `remote/modal_app.py` (and
    future split modules) use `try: import modal / except: modal = None`, and
    tests monkeypatch `module.modal`. Standardize: every extracted module gets
    its own guard, and tests patch the module actually under test. Do not
    introduce a shared guard module — it breaks attribute-level monkeypatching.

0.4 **Add an import-shape test.** A small test that (a) imports every root
    module both as package member and flat with the repo root on `sys.path`,
    and (b) asserts `comfyui_modal_sync_cloud` still defines the names
    `remote/modal_app.py` reads off it (`app`, `RemoteEngine`, the six
    exception classes with `__module__ == "comfyui_modal_sync_cloud"`). This is
    the tripwire for every later phase.

---

## Phase 1 — Trivial, high-confidence extractions from `api_intercept.py`

`api_intercept.py` (10,061 lines) has exactly **one** production consumer:
`__init__.py` imports `setup_modal_queue_route`. Everything else is test
surface. Keep `api_intercept.py` as the shim/aggregator throughout; shrink it
stepwise. Module-level mutable state is already narrowly scoped (5 items), so
extraction order is driven by dependency, smallest first.

1.1 `remote_plan_types.py` — the 14 dataclasses + `ModalPromptValidationError`
    (current lines ~206–411). Everything else imports from here;
    `ModalPromptValidationError` is raised from ~25 sites, so it must move
    first. `api_intercept.py` re-imports all names (tests keep working —
    dataclasses/exceptions are read-only surface, safe to re-export).

1.2 `modal_hardware.py` — GPU VRAM/cost tables + hardware payload builders
    (~413–556, ~145 lines). No mutable state.

1.3 `intercept_route_paths.py` — the 11 route-path derivation functions
    (~8601–8672). Trivial.

1.4 `modal_admin_ops.py` — `delete_modal_cache_dicts`, `delete_modal_volume`,
    `_call_modal_sdk`, `_modal_not_found_error_types` (~8673–8789).
    Self-contained Modal SDK wrapper.

1.5 `modal_ui_events.py` — status emission + per-client event ring buffer
    (~2509–2625). Owns `_MODAL_UI_EVENTS_BY_CLIENT` + its lock; move state and
    functions together, update the (few) test references to the new home.

Each step: `git mv`-style extraction, dual-import block, re-export from
`api_intercept.py` for functions/types only, run suite, commit.

---

## Phase 2 — Structural split of `api_intercept.py`

Order matters: each module depends only on earlier ones.

2.1 `remote_graph_analysis.py` (~1,600 lines from ~3237–4709) — workflow↔prompt
    node mapping, transportability rules, consumer maps, graph partitioning
    (`_remote_component_partition_groups`), topological ordering,
    `analyze_remote_node_selection`. Pure functions over arguments; the
    highest-value, lowest-risk large extraction in the repo. Tests reach
    `_remote_component_partition_groups`, `_component_topological_order`,
    `_component_execution_stages`, `_build_consumer_map` etc. through the
    module fixture — update `tests/test_configured_remote_planning.py` and
    friends to import the new module.

2.2 `component_planning.py` (~1,360 lines from ~4710–6074) —
    `_build_component_plan(s)`, boundary closures, dependency replication,
    session-boundary marking, `validate_remote_component_transport_compatibility`.
    Depends on 1.1 + 2.1 only.

2.3 `prompt_rewrite.py` (~1,530 lines from ~6125–7654) —
    `_build_component_payload` (592-line function), `_rewrite_component_into_proxy`
    (528 lines), parallelization, keepalive, prewarm hooks. While extracting,
    decompose the two giant functions: each has 3 nested builders that become
    module-level private functions with typed signatures (per AGENTS.md's
    ~50-line function guideline). Behavior-neutral: same call sequence, named
    instead of nested.

2.4 `execution_scheduling.py` (~1,700 lines from ~557–678, 979–2508) —
    environment state, capacity, `_plan_component_execution_assignments`,
    Vast/SSH candidate selection and reclaim, backend construction/stamping.
    Internally entangled across providers — extract as ONE module, do not split
    by provider. This is where most `monkeypatch.setattr(api_intercept, ...)`
    targets live (`R2CredentialStore`, `R2CacheClient`, `SshDockerController`,
    `_ssh_host_registry`, `_refresh_r2_storage_usage`); update the tests that
    patch them in the same commit. The R2 storage-usage cache
    (`_R2_STORAGE_USAGE_CACHE` + lock) moves here whole —
    `tests/test_configured_r2_backing.py` clears it directly, so point that
    test at the new module.

2.5 `queue_bridge.py` (~350 lines from ~8520–8600 + 8790–9059) —
    `_queue_prompt_json`, the `prompt_queue` monkeypatch bridge
    (`_install_modal_interrupt_queue_bridge`), and the remote-preparation
    registry. State lives as attributes on the `prompt_server`/`prompt_queue`
    objects (string attr keys), so this is cleanly portable.

2.6 Split the route layer. `setup_modal_queue_route` is 1,000 lines with 17
    nested aiohttp handlers closing over ~15 locals. Introduce a frozen
    `RouteContext` dataclass (settings, sync engine, host registries, vast
    registry, derived paths) and split handlers into per-domain registrars,
    each taking `(prompt_server, ctx)`:
    - `routes_r2.py` — R2 usage/keychain routes
    - `routes_remote_environments.py` — environment CRUD/probe/bootstrap/stop
    - `routes_vast.py` — vast status/verify/reap/destroy
    - `routes_modal_containers.py` — progress/container status/stop, cache and
      volume deletion
    - `routes_queue.py` — the 381-line `modal_queue_prompt` handler, decomposed
      into named stages (validate → analyze → plan → rewrite → dispatch)
    `setup_modal_queue_route` remains in `api_intercept.py` as a ~50-line
    aggregator that builds the context, calls the registrars, and preserves the
    `_ROUTE_REGISTERED` guard (tests monkeypatch it there — leave it in place).

End state: `api_intercept.py` ≈ 400–600 lines of aggregation + re-exports.

---

## Phase 3 — Split `remote/modal_app.py` (host-side, deployment-safe)

Keep `remote/modal_app.py` as the orchestrator holding the public entrypoints
(`invoke_remote_engine`, `invoke_remote_engine_async`, `execute_node_locally`,
`execute_subgraph_locally` re-exports for `remote/__init__.py`'s lazy
`__getattr__`) plus the hard-to-separate invocation core. Extract in
cleanliness order; every extracted module gets its own `modal` guard and
`logger` (check `tests/test_logging.py` for logger-name assertions).

3.1 `remote/modal_billing.py` (~440 lines) — billing status, own globals and
    exceptions; consumed only by `api_intercept` routes.
3.2 `remote/modal_container_logs.py` (~360 lines) — container list/stop + log
    streaming state.
3.3 `remote/host_session_bridge.py` (~920 lines) — session-bridge record
    building, offload, rehydration, `materialize_remote_session_bridge_ref_locally`.
    Local-exec functions import it (one direction, no cycle).
3.4 `remote/local_execution.py` (~1,270 lines) — `_NullPromptServer`,
    custom-node bundle materialization, `execute_node_locally`,
    prompt validation, subgraph trimming/normalization, local subgraph
    execution. Update `remote/__init__.py` lazy map in lockstep.
3.5 `remote/local_ui_events.py` (~500 lines) — PromptServer event emitters;
    already imported piecemeal by `ssh_executor.py` / `vast_executor.py` —
    point those imports at the new module.
3.6 `remote/modal_deployment.py` (~750 lines) — cloud-module loading (keep the
    `threading.local` settings-override setter and its reader together in this
    module), fingerprint/version checks, app stop/replace, auto-deploy. Where
    auto-deploy calls LLM staging and speculative prewarm, accept injected
    callables from the orchestrator rather than importing upward.
3.7 `remote/modal_warmup.py` (~1,175 lines) — warmup/prewarm/snapshot-profile
    records, keepalive, `ensure_remote_warm_capacity`,
    `boost_mapped_component_warmup`. Owns its executors: move the three
    import-time `ThreadPoolExecutor`s behind lazy accessors while relocating.
3.8 `remote/mapped_execution.py` (~1,215 lines) — phase splitting, mapped/
    implicit-batch invocation, aggregation.
3.9 `remote/modal_llm_profile_staging.py` (~290 lines) — staged-profile
    registry (name avoids colliding with root `llm_staging.py`).
3.10 **Delete the vestigial tail** (~99 lines): the never-deployed
    `modal.App(...)` + shadow `RemoteEngine` at the bottom of `modal_app.py`.
    Its only consumer is a debug-log guard that falls through unconditionally.
    This also removes an import-time `get_settings()` + `modal.App()` side
    effect. Verify no test references `remote.modal_app.app` before deleting.

Residual `modal_app.py`: `_invoke_remote_call_with_interrupts`, the 490-line
`_consume_remote_payload_stream` (decompose by event kind into named handlers:
progress, preview, boundary output, log, interrupt), payload invocation +
LLM-memory recovery retry, and the public entrypoints — roughly 1,400 lines.

---

## Phase 4 — Split `comfyui_modal_sync_cloud.py` (constrained)

The file keeps its name, location, the `modal.App`/image/`RemoteEngine`/
`ModelStager` tail, all six exception classes (their `__module__` is asserted),
the two grep'd literals, and the four injected dunder globals
(`__comfy_modal_settings_override__`, `__comfy_modal_gpu__`,
`__comfy_modal_app_name__`, `__comfy_modal_secret_name__`). Everything else can
move to flat root modules with a `cloud_` prefix (distinctive flat names per
ground truth 7), imported by the cloud module via the dual-import pattern.

**Prerequisite — 4.0 runtime store injection.** Helpers currently reach the
Modal `Dict`s/`Volume` (defined only in the tail) via `globals().get(...)`:
`session_bridge_cache`, `invocation_records`, `vol`, `snapshot_profiles`,
`node_output_cache`, `interrupt_flags`. `globals()` lookups break once code
moves file. Create `cloud_runtime_context.py` with a module-level registry and
typed accessors (`session_bridge_store()`, `invocation_record_store()`, ...);
the tail of `comfyui_modal_sync_cloud.py` registers the live objects right
after creating them. Convert the six existing accessor functions to delegate to
it *before* moving anything. This is the single enabling change for the phase.

Then extract, cleanest first:

4.1 `cloud_app_guard.py` — Modal app-existence probing (~135 lines).
4.2 `cloud_image_env.py` — image ignore filters + pip/torch layer builders +
    cls options/secret (~235 lines).
4.3 `cloud_durable_invocation.py` — invocation records, wait-for-running,
    canary barrier, output capture (~545 lines).
4.4 `cloud_session_bridge.py` — the 805-line session-bridge cluster + snapshot
    profile store. Its caches (`_REMOTE_SESSION_BRIDGE_VALUE_CACHE` etc.) move
    with it; update the ~dozen tests that touch them via the
    `modal_cloud_module` fixture to reach the new module.
4.5 `cloud_prompt_server_shims.py` — `_NullPromptServer`,
    `_HeadlessPromptQueue`, `_TracingPromptServer` (~465 lines).
4.6 `cloud_comfy_bootstrap.py` — custom-node bundle materialization, ComfyUI
    runtime init, node-class registration, loader-cache and folder-paths
    monkeypatches (~1,300 lines). This is the global-mutation core — move it as
    ONE unit; its patch-depth bookkeeping globals travel with it.
4.7 `cloud_node_output_cache.py` — persisted node-output cache (~845 lines +
    boundary signatures).
4.8 `cloud_prompt_execution.py` — executor management, local node execution,
    prompt validation, subgraph normalization/execution, mapped/phased
    execution (~1,500 lines). Decompose the 260-line
    `_execute_subgraph_prompt` into named stages while moving.
4.9 `cloud_streaming.py` — bounded stream buffer + `_stream_remote_payload_events`
    (~300 lines). `remote/ssh_worker.py` calls
    `execution_kernel._stream_remote_payload_events` — keep a delegating
    `_stream_remote_payload_events` name in `comfyui_modal_sync_cloud.py`
    (thin wrapper is fine here; ssh_worker calls it, tests don't patch it).
4.10 `cloud_volume_reload.py` — volume reload/hydration (~440 lines) and
    `cloud_prewarm.py` — snapshot prewarm + warm-container prep (~580 lines).
    `_clear_warm_remote_caches` reaches across loader/bridge/executor caches —
    reimplement it as calls to per-module `clear_warm_caches()` functions so
    each cache stays private to its owner.

End state: `comfyui_modal_sync_cloud.py` ≈ 800–1,000 lines — sys.path
bootstrap, exceptions, runtime-context registration, the Modal app/image/
service classes, and thin glue.

**De-dup opportunity (do after 4.x and 3.4, as its own step):** local node
execution, node/execution module loading, and prompt validation exist in
near-identical form in both `comfyui_modal_sync_cloud.py` and
`remote/modal_app.py`. Once both are extracted (`cloud_prompt_execution.py` vs
`remote/local_execution.py`), diff them and merge into one shared module if
they are truly identical; keep them separate if they have diverged
deliberately. Treat as behavior-risky; do last with extra test attention.

---

## Phase 5 — Mid-size modules

5.1 **`modal_llm_runtime.py` (2,769) → five modules.** Also fixes a dependency
    lie: `local_llm_runtime.py` (Apple/MLX) currently imports shared types and
    even a private helper from the "modal" runtime.
    - `llm_types.py` — `PreparedLLMInputs`, `LLMGenerationSettings`,
      `LLMProgressEvent`, `BackendGenerationResult`, `LLMInferenceResult`,
      `LLMBackend` protocol, `ResidentModel` (+ promote `_coerce_positive_int`
      to a public helper here).
    - `llm_inputs.py` — media/file preparation (tensor→PIL, frame sampling,
      PDF/base64 handling, chat templating) — pure and independently testable.
    - `vllm_instrumentation.py` — Triton compile-miss telemetry, vLLM
      engine-core monkeypatching, `VLLMExecutionModeController` (global
      process-side-effect code, kept away from inference logic).
    - `llm_backend_transformers.py`, `llm_backend_llamacpp.py`,
      `llm_backend_vllm.py` — one backend each (flat distinctive names until
      Phase 7).
    - `modal_llm_runtime.py` residual — `ResidentLLMManager`, env readers,
      singleton, `run_modal_llm_inference` / `prewarm_modal_llm_profile`
      (~700 lines).
    `local_llm_runtime.py` then imports only `llm_types`/`llm_inputs`.
    `local_llm_runtime.py` itself is cohesive — leave it alone.

5.2 **`sync_engine.py` (2,049) → package-style split.** The 1,375-line
    `ModalAssetSyncEngine` dataclass has ~55 methods spanning six concerns.
    Extract collaborators the engine composes (not subclasses):
    - `sync_protocols.py` — the 7 protocols + result/spec dataclasses
    - `sync_backends.py` — `LocalMirrorVolume`, `LocalFileSyncIndex`,
      `ModalDictSyncIndex`, `ModalVolumeBackend`, `_ModalSdkCaller`
    - `sync_hashing.py` — hashing + on-disk hash cache
    - `sync_custom_nodes.py` — custom-nodes bundle/archive/partition pipeline
      (largest self-contained slice)
    - `sync_r2_transfer.py` — R2 materialization/upload/write-back pool
    - `sync_engine.py` residual — `ModalAssetSyncEngine` as a ~400-line
      coordinator + `resolve_model_path` + status formatters (collapse the five
      near-identical `_format_*_status` functions into one parameterized
      helper).
    Budget for `tests/test_sync_engine.py`: 23 `monkeypatch.setattr` targets
    relocate; update patch targets alongside each extraction.

5.3 **`llm_staging.py` (1,053):** extract `snapshot_lease.py` — the ~280-line
    cross-process file lease (owner identity, heartbeat, stale expiry, wait
    loop) is a general-purpose lock with nothing HF-specific. Optionally also
    `supplied_llm_profiles.py` (~165 lines of profile persistence). The rest is
    cohesive staging logic.

5.4 **`modal_executor_node.py` (1,218):** extract
    - `remote_executor_router.py` — client protocol + Modal client + provider
      router (~160 lines; already has its own test file)
    - `proxy_payloads.py` — payload normalization/sanitization/registry
      (~240 lines)
    - `proxy_node_factory.py` — dynamic proxy class factory +
      `ensure_*_registered` node-mapping mutation (~300 lines)
    leaving the five static v3 node classes + concurrency-slot management.
    **Caveat:** `tests/test_modal_executor_node.py` is 16,550 lines — the
    heaviest test coupling in the repo. Do 5.4 after splitting that test file
    (Phase 6), or accept a large but mechanical test-import update.

5.5 **Leave alone:** `local_llm_runtime.py`, `vast_leases.py` (single coherent
    lease-lifecycle domain; optionally extract `vast_lease_records.py`
    (errors + record + registry persistence, ~380 lines) — zero test
    monkeypatching makes it cheap), and everything under 600 lines.

---

## Phase 6 — Test-suite mirroring

Large test files block production splits and are themselves unreadable:
`test_modal_executor_node.py` (16.5k), `test_api_intercept.py` (7k),
`test_modal_llm.py` (3k), `test_sync_engine.py` (2.4k).

- Split each along the same seams as the production modules they exercise
  (e.g. `test_remote_graph_analysis.py`, `test_component_planning.py`,
  `test_prompt_rewrite.py`, `test_routes_*.py` out of `test_api_intercept.py`).
- Move shared fixtures into `tests/conftest.py` or per-area
  `tests/fixtures/` modules as they surface.
- Keep the `conftest.py` `import_module(f"{PACKAGE_NAME}.<mod>")` fixture
  pattern; add fixtures for each new module. 51 of ~57 dynamic imports already
  live in `conftest.py`, so this stays a mostly one-file change per phase.

This phase interleaves with Phases 2–5 (split a test file in the same commit
series as its production counterpart) rather than happening at the end.

---

## Phase 7 (optional, last) — Package restructure

Once files are small, optionally group the ~70 flat root modules into one new
top-level package (single distinctive name, e.g. `modalsync/`), nesting
`core/`, `providers/{modal,vast,ssh}/`, `llm/`, `storage/` beneath it.
Feasibility is verified: ComfyUI package loading doesn't constrain sub-layout,
`remote/` proves subpackages work in all load contexts, the fingerprint walk
and `add_local_dir` handle subdirectories, and conftest uses dotted paths.

It is deliberately last because it is a pure `git mv` + import rewrite whose
risk (rewriting every dual-import `else:` branch, one per module) is only
worth taking after the suite fully covers the new module boundaries. If the
flat root with distinctive prefixes (`cloud_*`, `sync_*`, `llm_*`, `vast_*`)
proves readable enough, skip this phase — it adds churn without reducing any
file's size.

---

## Verification protocol (every step)

1. `uv run pytest` fully green before commit (repo rule).
2. The Phase 0.4 import-shape test passes (both import styles, cloud-module
   surface intact).
3. `git diff --stat` for the step shows moves + import edits only; any logic
   change (decomposing a giant function into named stages) is its own commit
   with before/after behavior pinned by existing tests.
4. After each phase touching `comfyui_modal_sync_cloud.py` or files it imports:
   one live Modal canary run (`tests/test_live_modal_canary.py`) since the
   repo digest and image will rebuild.
5. Update `docs/LEARNINGS.md` when a step surfaces a non-obvious constraint.

## Sequencing summary and effort

| Order | Phase | Risk | Outcome |
|---|---|---|---|
| 1 | 0 prep | low | seams visible, tripwire test in place |
| 2 | 1 api_intercept trivial extractions | low | −1,000 lines from the 10k file |
| 3 | 2 api_intercept structural split | medium | 10.1k → ~500-line aggregator + 10 modules |
| 4 | 3 modal_app split | medium | 9.1k → ~1.4k orchestrator + 9 modules |
| 5 | 4 cloud module split | medium-high | 8.1k → ~1k entrypoint + 11 `cloud_*` modules |
| 6 | 5 mid-size modules | medium | no second-tier file above ~1.5k |
| 7 | 6 test mirroring | low | interleaved with 2–5 |
| 8 | 7 package restructure | medium | optional |

The three giant files are independent of each other — Phases 2, 3, and 4 can
land in any order (Phase 2 first is recommended: smallest test blast radius per
step and it unblocks route-level readability that the R2 work on this branch
touches).
