# ComfyUI Modal-Sync

> [!WARNING]
> This project is still alpha. Expect missing features, rough edges, and breaking changes.

`ComfyUI Modal-Sync` is a ComfyUI custom node extension for marking parts of a workflow for remote execution. The frontend stores a per-node `is_modal_remote` flag in workflow metadata, the backend partitions the selected remote graph into transport-aware components, and a proxy node executes each component either locally or through Modal.

## Status

- The queue rewrite, serialization, and sync pipeline are implemented.
- `local` execution mode is the default development path and exercises the same rewrite and transport flow in-process.
- `remote` execution targets a deployed Modal app and can auto-deploy it on first use.
- Remote execution happens per transport-aware remote component, not per individual marked node.
- Remote components can expand upstream automatically when a marked node depends on non-transportable Comfy runtime objects.

## What is implemented

- A frontend extension in [`web/modal_toggle.js`](web/modal_toggle.js) that adds a `Run on Modal` toggle, draws remote-state overlays, and queues prompts through `/modal/queue_prompt`.
- A backend route in [`api_intercept.py`](api_intercept.py) that reads workflow metadata, resolves nested subgraph markers, partitions remote components, syncs assets, and rewrites each component into a signature-preserving proxy node.
- A dynamic proxy registry in [`modal_executor_node.py`](modal_executor_node.py) that mirrors exported output counts and output types so ComfyUI validation still succeeds after rewrite.
- A content-addressable sync engine in [`sync_engine.py`](sync_engine.py) for model files and the optional `custom_nodes/` bundle.
- A remote runtime in [`remote/modal_app.py`](remote/modal_app.py) with local fallback execution for tests and Modal-backed execution when the SDK is available.
- A stable Modal cloud entrypoint in [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py) for deployable remote execution.

## How it works

1. You mark nodes with `Run on Modal`.
2. Queueing goes to `/modal/queue_prompt` instead of ComfyUI's normal `/prompt` route.
3. The backend reads `extra_pnginfo.workflow`, including nested subgraph metadata, and resolves those markers onto queued prompt node ids.
4. Remote-marked nodes are partitioned into transport-aware components, with upstream auto-expansion when required by non-transportable inputs.
   If the coarse transport-aware split would create a cyclic quotient between components, the backend merges that SCC back into one larger remote component before rewrite.
5. Each component is replaced with a generated `ModalUniversalExecutor_<hash>` proxy node.
6. Boundary inputs are serialized, referenced assets are synced, and the component executes locally or remotely.
7. Exported boundary outputs are deserialized back into normal ComfyUI values for the rest of the graph.

The list above is the shortest accurate summary. If you want the execution path in a more algorithmic form, the pipeline below follows the same request from the frontend toggle through queue-time rewrite, remote dispatch, remote execution, and reintegration into the local graph.

### Remote Execution Pipeline

- **1. Frontend queue interception**
  - The user marks workflow nodes with `Run on Modal`.
  - The frontend intercepts the normal queue action.
  - It sends the prompt and workflow metadata to `POST /modal/queue_prompt`.

- **2. Queue-time rewrite on the local ComfyUI server**
  1. Read `extra_pnginfo.workflow` and find nodes marked for remote execution.
  2. Expand the remote set upstream when required by non-transportable inputs.
  3. Partition the marked graph into transport-aware remote components.
  4. Validate that each component boundary only uses supported transport types.
  5. Sync referenced assets to storage.
     - Model files are mirrored into remote storage.
     - The `custom_nodes` bundle may also be synced.
  6. Build a payload for each component.
     - Include the subgraph prompt.
     - Include boundary input specs.
     - Include boundary output specs.
     - Include execute target node ids.
     - Include asset and volume-reload metadata.
  7. Rewrite each remote component into a single dynamic Modal proxy node.
  8. Submit the rewritten prompt into ComfyUI's normal execution queue.

- **3. Local execution reaches a proxy node**
  - Local nodes execute normally until a rewritten Modal proxy node is reached.
  - The proxy node serializes its boundary inputs.
  - It checks the payload kind.
    - If the payload is a normal remote component, it dispatches a `subgraph` payload.
    - If the payload uses `ModalMapInput`, it dispatches one `mapped_subgraph` payload whose explicit static and mapped phases are executed inside a single runtime.

- **4. Dispatch decision**
  1. Check `COMFY_MODAL_EXECUTION_MODE`.
  2. If execution mode is `local`, or the Modal SDK is unavailable:
     - Execute the same payload locally in-process.
  3. Otherwise:
     - Look up the deployed Modal `RemoteEngine`.
     - If lookup fails:
       1. Try first-run auto-deploy.
       2. If allowed, fall back to ephemeral `app.run()`.
     - Invoke the remote payload.
       - Prefer `execute_payload_stream` when available.
       - Otherwise use `execute_payload`.

- **5. Remote execution inside the Modal container**
  1. Optionally reload the Modal volume if newly uploaded assets require it.
  2. Extract the synced `custom_nodes` bundle if present.
  3. Initialize the ComfyUI runtime.
  4. Deserialize the boundary inputs.
  5. Inject those inputs into the remote subgraph prompt.
  6. Run `PromptExecutor` for the component's `execute_node_ids`.
  7. Collect the declared boundary outputs.
  8. Serialize the outputs for transport back to the caller.

- **6. Streaming and local UI updates**
  - While the remote component runs, streamed events may be sent back.
    - Status changes
    - Numeric progress
    - Executed UI payloads
    - Preview images
  - The local frontend updates node overlays and global status from those events.

- **7. Return to the local workflow**
  1. The proxy node receives the serialized result.
  2. It deserializes the returned outputs.
  3. It exposes those outputs as normal ComfyUI node outputs.
  4. Downstream local nodes continue executing normally.

## Installation

Install this repository under ComfyUI's `custom_nodes/` directory:

```bash
cd ~/git/ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Modal
```

When ComfyUI starts, it should load:

- the backend route from [`api_intercept.py`](api_intercept.py)
- the internal proxy node definitions from [`modal_executor_node.py`](modal_executor_node.py)
- the frontend extension from [`web/modal_toggle.js`](web/modal_toggle.js)

## Execution modes

### Local mode

Use local mode during development:

```bash
export COMFY_MODAL_EXECUTION_MODE=local
```

This still exercises prompt rewrite, asset sync, and serialization, but the proxied component runs in-process instead of on Modal.

By default, local mode skips hashing and zipping the entire `custom_nodes/` tree because that work is unnecessary for in-process execution and can make queue requests look stalled on larger ComfyUI installs.

Force custom-node bundle sync in local mode with:

```bash
export COMFY_MODAL_SYNC_CUSTOM_NODES=true
```

### Remote Modal mode

Remote mode requires:

- the `modal` Python package in the ComfyUI environment
- working Modal credentials
- a reachable Modal app and volume configuration
- `COMFY_MODAL_EXECUTION_MODE` set to something other than `local`

The normal remote path is deployed-app lookup by name. On first remote use, the extension can auto-deploy the stable cloud entrypoint from [`comfyui_modal_sync_cloud.py`](comfyui_modal_sync_cloud.py) if the configured app does not exist yet.
If a previous attempt imported that cloud entrypoint only partially before failing, the loader now discards the stale cached module and retries the import cleanly before auto-deploy.
The local and remote subgraph runners also canonicalize malformed singleton-list wrappers in rewritten prompts, including socket indexes like `["node", [0]]`, scalar inputs like `[4]`, and wrapped execute node ids like `["12"]`, before handing the prompt to ComfyUI.
When the remote PromptExecutor still fails, the Modal worker now logs the normalized execute-node ids, boundary outputs, suspicious remaining wrapped prompt inputs, and the underlying ComfyUI `execution_error` payload to make the bad field visible.

```bash
export COMFY_MODAL_EXECUTION_MODE=remote
```

Remote deployment behavior:

- First-run auto-deploy is enabled by default.
- Persistent deployed apps are preferred over ephemeral `app.run()` execution.
- The extension does not create a persistent web endpoint for you.
- `COMFY_MODAL_AUTO_DEPLOY=false` disables first-run auto-deploy.
- `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK=true` re-enables the older temporary `app.run()` fallback path.

Remote runtime behavior:

- The deployed class reads its GPU type from `COMFY_MODAL_GPU`.
- CPU memory snapshots are enabled by default.
- GPU memory snapshots remain opt-in.
- The default `scaledown_window` is `600` seconds with `min_containers=0`.
- Warm containers can reuse loader state and `PromptExecutor` state across compatible requests.
- Each Modal GPU container now handles one active workflow execution at a time. If multiple remote components become ready in parallel, Modal can scale them out across multiple containers instead of multiplexing several executions onto one GPU worker.
- Remote proxy nodes now execute through ComfyUI's async node path, so independent Modal-backed components can overlap instead of being forced through one blocking local proxy at a time.
- The local Modal call executor keeps multiple worker threads available, which removes the previous `max_workers=1` bottleneck when several remote components are ready at once.
- Pure `ModalMapInput` components still use one explicit `mapped_subgraph` payload when the whole remote island is per-item work.
- Hybrid `ModalMapInput` components with a one-time static branch now rewrite into two ordinary proxy nodes: a static `subgraph` proxy that can return transportable outputs early, and a mapped `subgraph` proxy that reuses the static branch's remote-only values through prompt-scoped session refs.
- Ordinary remote components can now also split into ordered `subgraph` proxies when multiple execute branches only share remote-only upstream state. That lets one branch unblock local downstream work as soon as its transportable outputs are ready while later remote phases reconnect to shared remote-only values through prompt-scoped session refs.
- Those session refs keep values like `MODEL` remote-only while still letting the mapped proxy reconnect to them later, and the mapped proxy clears the prompt-scoped session once it finishes.
- Deployed split proxies now also pass a session-affinity key into `RemoteEngine` lookup so the static and mapped halves can reuse that prompt-scoped remote state on the same worker.
- Session observability now logs the full prompt-scoped lifecycle for split proxies: session creation, reuse, ref resolution, value storage, and cleanup in both local and cloud runtime paths.
- Opt-in live remote container log mirroring can now follow the active Modal worker's `task_id` during streamed executions and forward those lines into the local ComfyUI process stderr. That makes session create/reuse/cleanup messages visible locally without having to run `modal container logs ...` in a second terminal, and the watcher now prefers the Modal CLI path to avoid SDK task-context warnings in the local process.
- The remote worker only forwards numeric node progress for meaningful progress states. Trivial `0/1` updates from ordinary upstream nodes are ignored so sampler-style progress bars stay attached to nodes like `KSampler` instead of appearing on loaders or text encoders.
- Static sub-runs inside hybrid mapped execution now also forward their lane-less execute-node progress even when coarse per-phase status chatter is suppressed, so one-time upstream samplers still show real progress bars while mapped item runs keep their quieter stream behavior.
- Those suppressed static progress events now emit an explicit clear when the static sub-run finishes, so one-time upstream sampler bars disappear promptly at completion instead of sticking at `100%` until the whole prompt ends.
- Persisted remote node-cache hits are now idempotent within one execution: if a node output was restored from the distributed Modal Dict under a given `NC_...` key and nothing invalidated that key during the run, the persist phase skips rewriting that same record instead of churning the Dict on a pure cache hit.
- The distributed cache hot path is also lighter now: `restored-hit` nodes short-circuit before any output serialization work in the persist phase, and restore fan-outs independent Modal Dict lookups so mapped item runs spend less time waiting on node-cache bookkeeping.
- Mapped remote components can now contain both one-time execute targets and per-item execute targets. A common case is two remote `KSampler` nodes sharing one upstream `Load Diffusion Model`, where only the sampler fed by `ModalMapInput` should fan out per latent while the sibling sampler still runs exactly once.
- Queue-time rewrite now materializes explicit static-to-mapped boundaries for invariant upstream outputs. That lets the mapped phase consume only the values it really depends on instead of carrying the whole coarse component prompt back through every item run.
- Hybrid mapped sub-runs now trim their prompt down to the dependency closure of the specific `execute_node_ids` they are about to run. That keeps a static-only sub-run from validating or executing the unrelated mapped branch just because both branches originally lived in the same coarse remote component.
- Hybrid sub-run trimming also tolerates stale execute target ids that are no longer present in the current subgraph prompt. Those ids are now dropped before dependency resolution instead of crashing the remote worker with a `KeyError`.
- Boundary inputs injected back into a remote subgraph are now normalized through the same singleton-wrapper cleanup path as stored prompt inputs, so a proxied scalar like `[4]` no longer reaches `PromptExecutor` as `int([4])`.
- Before handing a remote prompt to `PromptExecutor`, Modal-Sync now preflights primitive widget inputs and rejects raw list values on `INT`/`FLOAT`/`BOOLEAN`/`STRING` sockets with a specific node/input error. When `PromptExecutor` still fails, the surfaced remote exception message now includes the failing node context and current inputs instead of only the inner exception string.
- Remote cancellation now uses a shared Modal `Dict` control store instead of a second RPC lane into the execution class, so per-container execution concurrency can stay at `1` without losing interrupt propagation.
- The blocking Modal request bridge now polls ComfyUI cancellation itself, including the streamed `execute_payload_stream.remote_gen(...)` path, so a local cancel still writes the shared interrupt flag even if the outer proxy wrapper has not noticed the interrupt yet.
- If a remote execution crashes instead of being interrupted normally, the worker now schedules its own container process to exit after surfacing the exception. That retires the bad warm GPU container instead of leaving it idle and billable after a hard failure.
- If a run is cancelled and restarted quickly, or a warm container is still releasing heavy model files, the remote worker now gives Modal volume reload a longer bounded retry window so recently released files can close before the next request needs a fresh `vol.reload()`.
- Mapped remote execution now carries a per-request volume reload marker, so one warm container only performs `vol.reload()` once for that uploaded asset snapshot even if the local scheduler fans the component out into many per-item Modal calls.
- Volume reload dedupe is now request-wide, not component-wide. If one queued workflow rewrites into several remote components, they all share the same reload marker so a warm container only reloads once for that overall synced asset set instead of once per component.
- Reload eligibility is now narrower than that request-wide marker. Each payload carries only the newly uploaded mounted-volume paths it can actually reference, so a component that does not touch a fresh asset no longer forces a reload just because some other component in the same queued workflow uploaded one.
- Queue-time prompt rewrite now emits a best-effort staged parallelism plan for the remote component DAG. Modal-Sync uses that estimate to start background container warmup as soon as assets are synced instead of waiting for the first proxy node to hit Modal.
- When a mapped or implicitly batched component finally hydrates its input batch, Modal-Sync upgrades that queue-time estimate with the exact per-item lane count and tops up background warmup again. That keeps early warmup conservative when fan-out size is unknown, but still lets the runtime race ahead of the heavy component once the true item count is available.
- If `vol.reload()` still reports open files after that bounded retry window, the worker now checks whether the payload's referenced mounted-volume paths are already visible on disk. When they are, Modal-Sync proceeds and records the reload marker instead of failing on a redundant reload.
- The worker now logs mounted-volume reload diagnostics when open-file retries start, including the uploaded paths and referenced paths for that payload, so you can see exactly which files were relevant when reload got stuck behind open handles.
- Custom-node bundle sync now treats the hash-addressed bundle path itself as authoritative. If ComfyUI crashes after uploading `/custom_nodes/<hash>_custom_nodes_bundle.zip` but before writing its marker, the next run backfills the marker and reuses the existing bundle instead of trying to rebuild or reupload it.

If you change `COMFY_MODAL_GPU`, redeploy the Modal app or delete it and let auto-deploy recreate it. Modal hardware is fixed at deploy time.

If you upgrade this node pack and expect changed remote runtime behavior such as faster interrupt handling, redeploy the Modal app once so the deployed class picks up the new code and class options.

## Using it in ComfyUI

### 1. Build your workflow

Build the workflow normally. This extension does not replace the editor's regular nodes. It adds a `Run on Modal` toggle to standard nodes.

Good candidates for remote execution are nodes that:

- consume large model files
- do expensive tensor work
- accept and return values that can cross the local/remote boundary cleanly

For batched workflows, you can also insert `Modal Map Input` before a remote-marked region. When that boundary input resolves to a Python list, an `IMAGE` batch, a `LATENT` batch, or another supported batched tensor value, Modal-Sync can split it into per-item remote executions and refill the available Modal slots locally until the batch completes.

### 2. Understand the transport boundary

Boundary-crossing values must be transportable. In practice, supported evaluated values include tensor-like and primitive types such as:

- `IMAGE`
- `MASK`
- `LATENT`
- `SIGMAS`
- `NOISE`
- `INT`
- `FLOAT`
- `BOOLEAN`
- `STRING`

Comfy runtime objects such as `MODEL`, `CONDITIONING`, `CLIP`, `VAE`, and `CONTROL_NET` cannot cross the current local/remote boundary. If a marked node still depends on those values from the local side, queueing fails fast with a validation error.

To reduce manual graph editing, Modal-Sync can auto-expand a remote component upstream when a marked node depends on non-transportable inputs. In practice, you can often mark the downstream sampler or custom node you care about and let the backend pull the required upstream loaders or builders into the same remote island.

### 3. Mark remote regions

For each connected region you want to offload:

1. Find the `Run on Modal` toggle on the node.
2. Enable it on every node that should be part of the same remote island.
3. Confirm the node shows the blue remote-execution border.

If queue-time validation tells you a remote node still depends on local-only runtime objects, you no longer have to hunt those upstream nodes manually. Right-click the node and use `Modal: Include Required Upstream Nodes` to ask the backend which extra nodes must join that remote island. If multiple nodes are selected in the current graph, the same context menu expands the whole selection at once.

The toggle stores `properties.is_modal_remote = true` in workflow metadata. The visible graph is not rewritten in the editor. Rewrite happens only at queue time.

### 3a. Map batched inputs across Modal

`Modal Map Input` is a pass-through adapter node. Its queue-time meaning only activates when it sits inside a remote-marked component.

Current mapped-execution rules:

- one `Modal Map Input` boundary per remote component
- downstream remote nodes reachable from that map marker stay in the same mapped remote component
- mapped inputs may currently be Python lists, `IMAGE` batches, `LATENT` batches, and other batched tensors split on dimension `0`
- ordinary remote components without `Modal Map Input` still preserve ComfyUI's usual zipped batch semantics at the remote boundary: if multiple boundary inputs arrive as compatible batches, Modal-Sync fans that component out per item instead of injecting the raw list into primitive widget sockets like `seed: INT`
- that implicit batching path now partitions one-time and per-item execute nodes too, so if two remote samplers share the same upstream `MODEL` but only one sampler receives a multi-item batch input, the sibling sampler still runs once instead of repeating once per batch item
- non-mapped boundary inputs are broadcast unchanged to every per-item execution
- mapped outputs are reassembled in item order, concatenating batchable tensors back together when possible
- per-item remote node status updates are suppressed, and streamed UI events from mapped item runs are filtered to the nodes that actually belong to that per-item payload so static sibling branches do not repaint the UI on every item
- aggregate mapped-batch completion now shades the global Modal status pill like a progress bar and shows counts such as `3/16`, while node-local progress bars remain reserved for real streamed node progress only
- the representative remote node also shows a small non-bar batch badge like `3/16` during mapped execution so you can see which remote island owns the batch without implying that the node itself emits sampler-style progress
- prompt-wide global status metadata is now merged across later per-component status updates instead of being overwritten by whichever remote component reports most recently, so the global pill keeps the correct total remote node count even when several components are running in parallel
- the global status pill now also reports queue-time setup detail such as packaging the custom-nodes ZIP, uploading named assets, and finalizing the run by receiving Modal outputs back into the local workflow
- once the rewritten prompt is accepted by the Modal queue route, all remote-marked nodes now flip to the ready/green state immediately rather than waiting for each component's first execution event; purple remains reserved for the node that is actively executing
- the ready-state transition now uses the accepted queue response's resolved remote-node list and component membership, so backend-expanded upstream nodes do not stay orange just because the websocket setup event arrived after the HTTP success response
- remote progress overlays now render in a temporary panel below the node instead of inside the node body, so sampler progress stays visible even when the node contains covered text widgets or other controls
- streamed per-item progress lanes now follow the real executing node id instead of being forced onto the component representative node, so concurrent remote samplers no longer all paint their progress bars onto the first node in the remote island
- mapped progress events now carry both the representative display id and the real executing node id, and the frontend resolves component membership from any node in that component so mapped fan-out runs do not leave sibling remote nodes stuck orange
- mapped node highlighting now derives its purple/active state from live per-node lane or numeric progress, so overlapping `A -> B` work across different containers can keep both remote nodes active at once while ordinary non-mapped runs still use the older single-active-node behavior
- worker lanes no longer emit a placeholder progress bar on the component representative node before real node progress arrives, which avoids leaving stray lane bars behind on the first node in the remote island
- the proxy node itself now opts into ComfyUI `INPUT_IS_LIST` handling and unwraps singleton list wrappers on both `original_node_data` and ordinary inputs before dispatch, so list-valued mapped inputs reach the internal Modal scheduler without causing the whole proxy node to be auto-mapped once per item
- static and per-item sub-runs prune the rewritten prompt to only the nodes they actually depend on before handing it to `PromptExecutor`, so shared-upstream hybrid components do not validate the other branch's batched inputs during the wrong sub-run

### 4. Queue the workflow

Use ComfyUI's normal queue action. The frontend intercepts submission and sends the prompt to `/modal/queue_prompt`.

During queue-time processing, the backend:

1. Inspects `extra_pnginfo.workflow`, including nested saved subgraph fragments.
2. Resolves nested markers onto the actual queued prompt ids, including composed ids like `24:23` when needed.
3. Replaces each transport-aware remote component with an internal `ModalUniversalExecutor_<hash>` proxy node.
4. Mirrors referenced model assets into storage.
5. Optionally syncs a zipped `custom_nodes/` bundle.
6. Submits the rewritten prompt to ComfyUI's normal queue.

### 5. Watch execution state

The frontend shows remote state directly on the canvas:

- blue border: marked for Modal, idle
- orange pulsing border: queued or still being prepared
- pulsing green border: ready in Modal and waiting
- pulsing purple border: currently executing remotely
- steady green border: finished for the current run
- red border: queue-time or execution failure

When a remote node reports numeric progress through ComfyUI's progress hooks, the active Modal-marked node now also shows an in-node progress bar and percentage locally while the remote work is running. That is especially useful for sampler-style nodes such as `KSampler`.

Cancelling a prompt in local ComfyUI now propagates to the active Modal component as a targeted remote interrupt. In practice, long-running remote samplers, including streamed subgraph runs, no longer continue burning time after you hit cancel locally.

Remote nodes that emit ComfyUI UI outputs also stream those `executed` payloads and preview frames back into the local PromptServer while the remote subgraph is still running. That means image previews can appear during a Modal run instead of only after the final proxy node returns.

For direct local `PreviewImage` consumers of a remote boundary `IMAGE` output, the relay also synthesizes the local preview-node UI event as soon as that remote boundary image is ready. This improves the common "remote decode -> local preview" case even though the proxy node itself still returns its formal outputs only when the remote component finishes.

Remote subgraph runs also persist transport-safe node outputs into a shared Modal `Dict` using ComfyUI's `CacheKeySetInputSignature` semantics. In practice that lets later workflow runs reuse small and medium cached outputs, including image tensors, even after Modal scales down and starts a fresh container, while still skipping non-serializable outputs and entries above the configured size cap.

The extension also keeps a global activity badge visible during queue-time sync and remote execution, including the period before the prompt is formally queued. During setup it now surfaces more specific stages such as packaging the `custom_nodes` ZIP and uploading named assets, and after remote execution finishes it can briefly report that it is still receiving Modal outputs before the local proxy returns.

### 6. Asset sync expectations

The sync engine automatically looks for inputs that resolve to files ending in:

- `.safetensors`
- `.ckpt`
- `.pt`
- `.vae`

In practice:

- absolute file paths work
- model names resolvable through ComfyUI `folder_paths` work
- arbitrary unresolved strings do not sync

If a remote-marked node depends on a model filename that cannot be resolved to a local file, prompt queueing fails instead of continuing with a broken remote request.

## Configuration

### Routing and metadata

- `COMFY_MODAL_ROUTE_PATH`: Override the queue endpoint. Default: `/modal/queue_prompt`.
- `COMFY_MODAL_MARKER_PROPERTY`: Override the workflow property used to mark nodes. Default: `is_modal_remote`.

### Local storage and sync

- `COMFY_MODAL_LOCAL_STORAGE_ROOT`: Local mirror root used for sync tests and dry runs. Default: `/tmp/comfyui-modal-sync-storage`.
- `COMFY_MODAL_CUSTOM_NODES_DIR`: Override the `custom_nodes` directory to bundle and mirror.
- `COMFY_MODAL_SYNC_CUSTOM_NODES`: Force-enable or disable custom-node bundle sync. Default: disabled in `local` mode, enabled otherwise.

### Execution and deployment

- `COMFY_MODAL_EXECUTION_MODE`: Set to `local` for in-process fallback execution. Default: `local`.
- `COMFY_MODAL_APP_NAME` and `COMFY_MODAL_VOLUME_NAME`: Override Modal app and volume naming.
- `COMFY_MODAL_INTERRUPT_DICT_NAME`: Override the shared Modal `Dict` used for remote cancellation flags. Default: `<app_name>-interrupts`.
- `COMFY_MODAL_NODE_CACHE_DICT_NAME`: Override the shared Modal `Dict` used for persisted transport-safe node outputs. Default: `<app_name>-node-cache`.
- `COMFY_MODAL_NODE_CACHE_MAX_BYTES`: Maximum raw output size eligible for persisted node caching. Default: `5242880` (5 MiB). Set to `0` to disable the persisted node-output cache.
- `COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR`: When true, a remote execution crash makes the worker exit its own container after returning the error. Default: `true`.
- `COMFY_MODAL_AUTO_DEPLOY`: Automatically deploy the Modal app on first remote invocation when lookup fails. Default: `true`.
- `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK`: Re-enable slow `app.run()` fallback in remote mode. Default: `false`.
- `COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS`: Mirror live remote Modal container logs into the local ComfyUI stderr stream during streamed executions. Default: `false`. Modal-Sync prefers `modal container logs <task_id> -f` and only falls back to the installed Modal Python SDK log stream when the CLI is unavailable.

### Modal runtime sizing

- `COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT`: Enable Modal CPU memory snapshots. Default: `true`.
- `COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT`: Enable Modal GPU memory snapshots. Default: `false`.
- `COMFY_MODAL_GPU`: Modal GPU type to request when deploying the remote class. Default: `A100`.
- `COMFY_MODAL_SCALEDOWN_WINDOW`: Keep idle containers warm for this many seconds. Default: `600`.
- `COMFY_MODAL_MIN_CONTAINERS`: Keep at least this many containers warm. Default: `0`.
- `COMFY_MODAL_MAX_CONTAINERS`: Optional upper bound on simultaneously scaled Modal containers.
- `COMFY_MODAL_BUFFER_CONTAINERS`: Optional number of spare warm containers Modal should try to keep ready above current load.
- `COMFY_MODAL_ENABLE_PROACTIVE_WARMUP`: Start best-effort background warmup RPCs based on queue-time and runtime parallelism estimates. Default: `true`.

## Development

- Manage the project with `uv`.
- Install test dependencies with `uv sync --group test`.
- The test dependency group includes the local packages this extension needs directly plus the minimum runtime packages required to import a ComfyUI checkout during pytest.
- Tests look for ComfyUI in `COMFYUI_ROOT` first, then fall back to `~/git/ComfyUI`.
- To run tests against a temporary checkout, clone ComfyUI somewhere disposable and point `COMFYUI_ROOT` at it.

```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/comfyui-modal-test/ComfyUI
UV_PROJECT_ENVIRONMENT=/tmp/comfyui-modal-test-env uv sync --group test
COMFYUI_ROOT=/tmp/comfyui-modal-test/ComfyUI \
  /tmp/comfyui-modal-test-env/bin/python -m pytest
```

- `uv run pytest` should also work in a normal environment once the `test` dependency group has been synced into the active project environment.
- The repository is now structured as a ComfyUI Registry node pack with registry metadata in [`pyproject.toml`](pyproject.toml) and a publish workflow in [`.github/workflows/publish_action.yml`](.github/workflows/publish_action.yml).
- Before publishing, create a Comfy Registry publisher and API key, then store the token in the GitHub Actions secret `REGISTRY_ACCESS_TOKEN`.
- The registry pack name is `modal-sync`, the display name is `Modal Sync`, and the current publisher id is set to `ttulttul` to match the GitHub origin owner.
- [`modal_test_workflow.json`](modal_test_workflow.json) is a checked-in smoke artifact from a successful Modal-path run, not a pristine authoring workflow.
- Implicit mapped execution only treats explicitly batchable transport types and list-wrapped primitive sockets as per-item inputs. List-backed semantic types like `CONDITIONING` stay broadcast across mapped item runs so the remote worker does not accidentally slice them into invalid fragments.
- Remote primitive-socket validation only rejects raw widget list literals now. If a primitive socket like `INT` is hydrated from a boundary input or upstream output and carries a list value, the runtime leaves it alone so ComfyUI can apply its normal over-list execution semantics for nodes like `NextSeeds`.
- Boundary input normalization only unwraps singleton wrappers for scalar-like values and true prompt links. Single-entry semantic lists like `CONDITIONING` must stay wrapped as lists, or ComfyUI will misread `[[tensor, meta]]` as `[tensor, meta]` and crash during conditioning conversion.
- Prompt-scoped remote session misses are treated as routing/state errors now, not poisoned-container crashes. If a split proxy lands on the wrong Modal container and its session bucket is missing, the worker logs the miss but does not self-terminate; the real fix for that class of failure is stronger container affinity or leased session routing.
- `ModalMapInput` is a queue-time mapping marker even when the marker node itself stays local. If a local `ModalMapInput` feeds a remote node, rewrite should still produce a `mapped_subgraph` so list(INT) values like `NextSeeds` outputs are itemized before the remote sampler runs.
- Implicit mapped execution distinguishes semantic lists from lists of remote session refs. Raw `CONDITIONING` lists stay broadcast, but a list of prompt-scoped session refs for `MODEL`/`CONDITIONING` is treated as per-item mapped state and split item-by-item before the downstream remote node executes.
- Implicit mapped execution cannot let each static or per-item sub-run inherit `clear_remote_session=True`. Shared prompt-scoped sessions now stay alive across all sibling mapped items and are cleared exactly once through a dedicated final cleanup payload, which prevents later item runs from seeing `RemoteSessionStateError` after an earlier sibling finishes first.

## Current limitations

- Remote execution is component-based. If you leave a local gap in the middle of a would-be remote chain, the boundary still has to be transport-safe.
- The default sync backend is still a local mirror used for development and tests. Real Modal execution depends on a working Modal SDK environment.
- Non-JSON, non-bytes, non-tensor payloads are not supported across the current local/remote boundary.
- Workflow artifacts captured after a remote run may include internal proxy nodes such as `ModalUniversalExecutor`; they are useful as regression fixtures, but should not be treated as clean source workflows.
