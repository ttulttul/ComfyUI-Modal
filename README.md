# ComfyUI Modal-Sync

## Breaking Change: Remote Execution Configurator Required

> [!IMPORTANT]
> Starting with version 0.4.0, every workflow that uses remote execution must include a **Remote Execution Configurator** connected to at least one **Modal Configuration**, **Vast.ai Configuration**, or **SSH Configuration** node. Existing workflows that relied only on the legacy workflow-wide provider and GPU settings must add and connect these nodes before they can run remotely. Optional storage backends, including **R2 Storage Configuration**, connect to the same Configurator.

> [!WARNING]
> This project is still alpha. Expect missing features, rough edges, and breaking changes.

ComfyUI Modal-Sync is a ComfyUI custom node extension that runs selected parts of a workflow on remote GPUs. Mark the expensive nodes for remote execution — or let the planner place them automatically — and queue the workflow as usual. The extension partitions the graph, syncs the required models and custom nodes, executes the remote portions on the back-end of your choice, and streams progress, previews, and outputs back into your local ComfyUI session.

Three execution back-ends are supported today:

- **Modal** — serverless GPU containers with automatic deployment and scale-to-zero
- **Self-hosted SSH hosts** — your own machines, reached over plain SSH and running fingerprinted Docker workers
- **Vast.ai** — marketplace GPU instances rented automatically, priced and selected per workflow

The back-end layer is pluggable. Each provider implements only host discovery, provisioning, and transport; graph partitioning, cost-aware scheduling, asset sync, serialization, progress streaming, cancellation, and output handling are shared across all providers. This makes it practical to add more back-ends in the future. (The project began as a Modal integration — hence the name — and Modal is now one provider among several.)

## Table of Contents

- [Breaking Change: Remote Execution Configurator Required](#breaking-change-remote-execution-configurator-required)
- [Highlights](#highlights)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Execution Back-Ends](#execution-back-ends)
  - [Workflow Provider Policy](#workflow-provider-policy)
  - [Modal](#modal)
  - [Self-Hosted SSH Hosts](#self-hosted-ssh-hosts)
  - [Vast.ai](#vastai)
  - [Cost- And Capability-Aware Scheduling](#cost--and-capability-aware-scheduling)
- [Using It In ComfyUI](#using-it-in-comfyui)
  - [Marking Nodes For Remote Execution](#marking-nodes-for-remote-execution)
  - [Canvas State](#canvas-state)
  - [Batched And Mapped Workflows](#batched-and-mapped-workflows)
- [LLM Nodes](#llm-nodes)
  - [Modal LLM](#modal-llm)
  - [Modal Endpoint Chat](#modal-endpoint-chat)
- [Asset And Custom Node Sync](#asset-and-custom-node-sync)
  - [Optional Cloudflare R2 Backing Cache](#optional-cloudflare-r2-backing-cache)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Current Limitations](#current-limitations)

## Highlights

- A `Run Remotely` node toggle, workflow-scoped provider capacity, and optional fully automatic node placement
- Three interchangeable back-ends — Modal, self-hosted SSH hosts, and Vast.ai — behind one shared planner and transport layer
- Capability- and cost-aware scheduling: the planner checks VRAM/RAM requirements against each candidate and picks the lowest predicted total cost
- Automatic sync of referenced model files and, optionally, your `custom_nodes/` packages
- Streamed remote status, sampler progress, preview images, and UI payloads rendered live on the local canvas
- A resident `Modal LLM` node for text, image, video, and file understanding — locally on Apple Silicon or on the same remote GPU as the rest of the component
- A `Modal Endpoint Chat` node for inference through Modal hosted-model endpoints
- A local in-process execution mode for development and testing, no credentials required

## How It Works

Remote execution is component-based: contiguous groups of marked nodes become *remote components*, and each component runs as a unit on one back-end. When a prompt is queued:

1. The frontend sends the prompt and workflow metadata to `POST /modal/queue_prompt` instead of the normal queue route.
2. Remote markers are resolved onto queued prompt node ids, including nodes nested inside subgraphs.
3. Marked nodes are partitioned into components. Edges carrying large tensor or media values (`LATENT`, `IMAGE`, `VIDEO`, …) are kept inside one component where possible; components also expand upstream across values that cannot cross the local/remote boundary, such as `MODEL`, `CLIP`, `VAE`, and `CONDITIONING`.
4. The planner assigns each component to a back-end according to the workflow's provider policy, hardware compatibility, and predicted cost.
5. Referenced model assets — and, when enabled, custom-node packages — are synced to the selected back-end's storage.
6. Each component is replaced with a generated proxy node, and the rewritten prompt is submitted to ComfyUI's normal execution queue.
7. Local nodes run normally. When a proxy is reached, it dispatches the component remotely, relays streamed progress and previews, and materializes remote outputs — including generated files, which are downloaded into your local `output/` directory — back into the local graph.

Values crossing the local/remote boundary must be transportable. Supported types are `IMAGE`, `VIDEO`, `AUDIO`, `MASK`, `LATENT`, `SIGMAS`, `INT`, `FLOAT`, `BOOLEAN`, and `STRING`. Runtime objects such as `MODEL`, `CONDITIONING`, `CLIP`, and `VAE` cannot cross the boundary; Modal-Sync either expands the remote component so they are produced remotely, or fails queue-time validation with a boundary error.

List-valued transportable outputs cross through ComfyUI normally, even when both components use the same remote environment. This preserves ComfyUI's `OUTPUT_IS_LIST` item mapping and cache behavior; provider-local session bridges remain reserved for values whose scalar/list identity is unchanged by replacing the value with one bridge reference.

When one component's output feeds only another component on the same remote environment, the value stays on the remote side as a small durable reference instead of round-tripping through the local machine. Modal can also use its shared durable store to download the same value asynchronously for an independent local branch. Vast and SSH workers have provider-local bridge stores, so a transportable output that also feeds a local preview, save, or other local-only branch instead crosses through ComfyUI normally.

## Installation

Install this repository under ComfyUI's `custom_nodes/` directory and restart ComfyUI:

```bash
cd ~/git/ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Modal
```

Start with local mode while building or debugging workflows. Local mode exercises the full pipeline — marker resolution, prompt rewrite, sync planning, serialization, and proxy execution — but runs the rewritten components in the local ComfyUI process, so it needs no credentials:

```bash
export COMFY_MODAL_EXECUTION_MODE=local
```

Switch to remote mode when you are ready to run on a real back-end:

```bash
export COMFY_MODAL_EXECUTION_MODE=remote
```

## Execution Back-Ends

### Workflow Capacity Configuration

New workflows declare every available remote capacity pool explicitly:

1. Add a **Remote Execution Configurator** node.
2. Add any number of **Modal Configuration**, **Vast.ai Configuration**, and **SSH Configuration** nodes.
3. Optionally add an **R2 Storage Configuration** node to provide shared object storage to Vast.ai and SSH targets.
4. Connect every capacity and storage node's `REMOTE_CONFIGURATION` output to the configurator's growing input group.

The connected graph is authoritative. Connecting only Modal configurations produces a Modal-only plan; connecting several providers lets the planner choose among all of them. Each capacity configuration has its own name and limit, so one workflow can offer several Modal GPU types, several independent Vast marketplace searches, and several SSH hosts at the same time. Storage configurations are sibling values in the same configurator but never become scheduler targets or contribute capacity. All configuration nodes are concrete ComfyUI v3 data nodes, so the branch remains valid when ComfyUI executes it after queue-time planning.

Capacity limits are concurrency ceilings, not eager provisioning requests. Modal containers scale as work needs them, Vast instances are quoted during planning and rented only for selected slots, and SSH worker containers are prepared lazily on the declared host. Sequential components can reuse one slot; parallel components consume distinct slots or wait when that is cheaper than another environment.

The planner compiles configuration inputs before ordinary ComfyUI node execution, resolves the complete component DAG, and assigns every component before it performs a billable Vast acquisition. Hard provider, VRAM, RAM, and architecture constraints are applied before cost ranking. The `REMOTE_CONFIGURATION_SET` output is also available to local graph consumers for inspection.

**Automatically place eligible nodes** remains an independent workflow option. Leave it off to mark component boundaries with `Run Remotely`, or enable it from the **Remote Execution** menu to let the planner select eligible nodes.

Workflows saved before the configurator was introduced continue to use their saved provider policy, workflow Modal GPU, disconnected Vast lease nodes, and installation-wide SSH registry. When a configurator is present, those legacy capacity sources are not added to the new plan.

### Modal

Authentication is user-managed — run Modal's setup once if the current user does not already have working credentials:

```bash
<comfyui-venv>/bin/python -m modal setup
```

If the pinned Modal SDK (`modal==1.4.2`) is not importable when ComfyUI starts in remote mode, Modal-Sync installs it into the ComfyUI Python environment automatically (preferring `uv pip`, falling back to `pip`).

Remote workers receive workflow API keys through a named Modal secret collection. Create the default `comfy` collection once from ComfyUI's local `.env` file (the file itself is never copied into the image):

```bash
<comfyui-venv>/bin/python -m modal secret create comfy \
  --from-dotenv /path/to/ComfyUI/.env
```

Every key in the collection is available to remote custom nodes through `os.environ`. To use a different collection, set `COMFY_MODAL_SECRET_NAME` before starting ComfyUI.

Each ComfyUI installation gets its own persistent Modal app name, derived from an identity file in the ComfyUI user directory, so multiple installations do not collide. Set `COMFY_MODAL_APP_NAME` to share or externally manage a deployment.

Each **Modal Configuration** selects a GPU type and maximum concurrent container count. Every GPU target gets its own deployed app, so several configurations can contribute different Modal GPU types to the same plan. First-run auto-deploy is enabled by default and also replaces stale or incompatible deployments; after upgrading this node pack, redeploy once (or leave `COMFY_MODAL_AUTO_DEPLOY=true`) so remote workers pick up the new code. Warm containers reuse loaded models across compatible requests and scale down to zero after an idle window (`COMFY_MODAL_SCALEDOWN_WINDOW`, default 600 seconds).

### Self-Hosted SSH Hosts

Self-hosted execution runs the same workers on machines you control, reached over SSH and managed through Docker. Add one **SSH Configuration** per machine and connect it to the configurator. The extension deliberately stores no passwords or private-key paths — authentication, jump hosts, ports, and key selection stay in your normal SSH agent and `~/.ssh/config`.

The installation-wide **Settings → Remote Execution: SSH environments** manager remains available for legacy workflows and host preflight. New configured workflows carry their own credential-free SSH destination and scheduling snapshot, including the probed GPU capabilities, so their execution does not depend on mutating that global registry. Older queued payloads without a capability snapshot are re-probed before launch. GPU workers are always started with the selected GPU device request; missing GPU/runtime capability is an explicit setup error instead of a silent CPU-only container. The worker image also pins and validates the PyAV API required by the copied ComfyUI source, so a stale video dependency fails during image construction instead of at the first remote node import.

Worker images embed every file beneath ComfyUI's curated runtime package directories, including tokenizer vocabularies, model configuration JSON, SentencePiece data, and other non-Python resources. Those files participate in the runtime fingerprint, so adding or changing required package data automatically replaces a stale SSH worker and is reflected in the next published Vast worker image. On self-hosted SSH machines, the heavyweight apt, PyTorch, CUDA, accelerator, and custom-requirement stack is retained as a separately fingerprinted local dependency image. A normal source edit therefore rebuilds only the roughly 52 MiB source overlay; the multi-gigabyte base is rebuilt only when its package inputs change. Before a build, superseded images tagged and labeled by this extension are removed when Docker no longer has a container reference to them, bounding the retained cache without pruning unrelated host images. The image also retains an explicit `gcc`/`g++` build toolchain and exports `CC`/`CXX`, because Triton may compile native launchers lazily when a ComfyUI node first executes; the dependency build validates that toolchain before publication.

Parallel components assigned to the same SSH worker slot share one lifecycle operation. If two launchers race for the deterministic container name, the loser adopts the correctly labeled, fingerprint-matching worker that won the race; containers without the configured environment ownership label are never removed automatically.

Each host must provide:

- Linux on `x86_64`
- a non-interactive SSH login with an already trusted host key
- a working Docker CLI and daemon with the Buildx `default` builder available to that login
- NVIDIA drivers plus Docker's NVIDIA runtime or CDI configuration for GPU work
- outbound package/image access during the first worker build

Use **Save and probe** to discover CPU architecture, RAM, Docker disk and runtime state, and each GPU's name, compute capability, driver, and total/free VRAM. Use **Build / update worker** to build the host's worker image and start a persistent container — or skip it, since the first component assigned to the host builds the same image automatically.

The host form also controls placement:

- **Cost USD/hour** — used for cost-based planning. Leaving it blank means "unknown," not zero, so an unpriced host is not mistaken for a free one.
- **Max workers** — how many worker containers the planner may address; workers are distributed across the discovered GPUs.
- **Reserve VRAM GB** — capacity withheld from compatibility checks, e.g. for desktop use.
- **Tags**, **Enabled**, and **Drain** — placement metadata and safe admission controls. Draining blocks new components without killing a running worker.

**Refresh workers** and **Stop workers** operate only on containers carrying this extension's ownership label, so they cannot touch unrelated Docker workloads. Worker images are immutable and fingerprinted from the node-pack source, ComfyUI source, custom-node requirements, and runtime settings; a fingerprint change replaces the worker while its named storage volume is retained. All traffic travels over the existing SSH connection — no daemon TCP port, worker port, or inbound firewall rule is required.

Worker environment secrets are administrator-managed: the optional **Remote Docker env file** field names a file that already exists on the host and is passed to Docker's `--env-file`. Its contents never cross the API or enter the workflow.

### Vast.ai

Vast.ai capacity is declared with one or more **Vast.ai Configuration** nodes connected to the configurator. Each node declares an independent named marketplace search and maximum instance count. Marketplace selectors default to the explicit value **Any**, so a new profile starts broad and the user can add GPU-count, per-GPU VRAM, TFLOPS, RAM, reliability, duration, price, verification, location, or network constraints one at a time. The advanced **Excluded country codes** box sends Vast's `geolocation.notin` filter and defaults to `CN, RU`, avoiding regions where Hugging Face and other model origins are commonly inaccessible; clear the box to allow every country. Disk allocation, idle retention, and the maximum number of managed instances remain required launch/lifecycle controls. **Any** hourly price removes the hard spending ceiling; offers are still ranked by price, but a Vast-only workflow may rent an expensive offer when no cheaper capacity exists.

The typed node remains local and travels to the planner only through its configurator connection. The older disconnected **Vast.ai Lease Configuration** node and its Markdown selection output remain supported for legacy workflows.

Marketplace searches for the distinct effective resource profiles in one plan run in parallel, with a maximum of eight concurrent requests. Successful results, including an empty result, are cached in the ComfyUI process for 60 minutes and shared across workflow submissions. Lease selection and acquisition remain ordered so later components can reuse capacity rented by earlier components. Restart ComfyUI to clear the in-memory cache; a rental availability race automatically forces one fresh marketplace search.

At queue time the planner raises **Any** or lower VRAM and RAM floors to whatever the workflow's models actually require, queries current on-demand offers, and revalidates every enabled constraint locally. Cross-provider selection compares Vast's effective hourly compute price over the predicted execution time; configured idle retention is tracked as a separate billing estimate and does not make Vast appear more expensive than Modal or SSH during placement. Numeric memory constraints use GiB, matching the existing scheduler and Vast offer-capacity calculations. Vast's marketed GPU names do not always equal their reported capacity exactly; for example, a reported 95.x GiB card does not satisfy a literal 96 GiB floor. The default idle retention is 24 hours, after which an in-instance watchdog destroys the idle lease.

Vast startup uses every available lifecycle field instead of relying only on `actual_status`. Legitimate image-download and loading states retain the full configured readiness window, while an instance that continuously reports no lifecycle state, provider message, or SSH host fails after a two-minute grace period. When Vast publishes both a direct public mapping for container port 22 and an SSH proxy, the controller uses the same direct endpoint shown in the Vast console; this avoids stale proxy endpoints that close every handshake. Vast's SSH launch mode injects the account key before running the configured `onstart` script; that script normalizes `/root`, `.ssh`, and `authorized_keys` ownership and modes before starting the worker, preventing `sshd` StrictModes failures inherited from a custom image. Before every worker-readiness SSH probe, the controller rechecks that the exact Vast contract still exists. A contract that disappears during provider startup or worker initialization is removed from the local registry immediately, its marketplace offer is excluded, and the next compatible offer is cold-started as a replacement. A contract that remains non-SSH-ready for the configured startup window, or cannot initialize its worker afterward, is marked unusable and terminated through the Vast API before the controller cold-starts one replacement from a different offer. A reused contract consumes only the remainder of its original startup window, so restarting ComfyUI cannot restart the full wait. If the replacement also fails, acquisition stops with the failed instance IDs, cleanup outcome, and final provider/setup cause instead of entering an unbounded rental loop. Worker-readiness polling makes one connection per interval rather than nesting the normal command retry budget inside every probe. Commands on an established worker and interrupted asset transfers still retry transient connection closures.

Direct Vast SSH operations tolerate bounded connection-level closures, resets, and handshake interruptions with jittered exponential retries. Authentication rejection and deterministic remote-command failures remain immediate. Streamed uploads reopen the local source from byte zero on every attempt, and the remote atomic writer verifies the expected byte count before publishing, so a killed connection cannot leave a truncated file at the content-addressed destination.

Cancelling a workflow also cancels queue-time remote preparation. File hashing checks the prompt-scoped cancellation signal between chunks, and active Vast SSH uploads or Hugging Face downloads terminate their local SSH process instead of continuing behind ComfyUI's cancelling status. Vast Hugging Face acquisition runs from a small content-addressed tool bundle uploaded by the controller, so an older worker image cannot force a multi-gigabyte fallback upload merely because it lacks the newest materializer module. The launch environment does not overwrite the runtime fingerprint baked into the published worker image. An owned lease may be adopted across controller-only fingerprint drift when its configured worker image is unchanged and its worker protocol is still compatible, avoiding a duplicate rental; changing the configured image still requires matching capacity. Stale registry entries are discarded whether Vast reports a missing contract through HTTP 404 or its v0 `instances: null` response.

Set credentials and the worker image in the ComfyUI process environment. Neither value is written to workflows, the lease registry, API responses, or logs:

```bash
export VAST_API_KEY='...'
export COMFY_MODAL_VAST_IMAGE='ghcr.io/owner/comfy-modal-worker@sha256:...'
```

Build the shared SSH/Vast worker image locally, push it to a registry, and use the digest printed by the script (the image must be readable by the rented instance; the current implementation supports public images):

```bash
uv run python scripts/build_vast_worker_image.py --push
```

The builder always targets `linux/amd64`, including when it runs on an Apple Silicon Mac, because Vast GPU hosts and the pinned CUDA/vLLM wheels are x86_64. Docker Desktop uses its cross-architecture emulation for the build. `--tag` is optional. When omitted, its effective default is `ghcr.io/<owner>/comfy-modal-worker:v<version>`: the builder reads `<version>` from `project.version` and `<owner>` from the GitHub Repository URL in `pyproject.toml`. Use `--owner` or `--tag-template` to change those derived defaults, or pass `--tag` to supply the complete image tag explicitly.

Open **Settings → Remote Execution: Vast.ai leases** to verify the credential, inspect hourly price, activity, and idle deadlines, and destroy expired or idle leases. Manual destruction rechecks the live instance's ownership label and refuses to destroy active work. The same controls are available at `/remote/vast/status`, `/remote/vast/verify`, `/remote/vast/reap`, and `/remote/vast/destroy`.

For offline development and CI, a local API simulator stands in for the live service — see [Development](#development).

### Cost- And Capability-Aware Scheduling

Before comparing costs, the planner resolves the model files referenced by each remote component and derives conservative memory floors: the largest resident model plus LoRA/adapter/ControlNet weights, a weight-overhead margin, and a fixed reserve for activations and runtime state. Candidates are checked against live free VRAM and RAM where a probe reports them, so an undersized host is rejected before any model upload. When an earlier ComfyUI prompt already owns a configured SSH host, a later queued prompt is admitted against that host's real total capacity rather than its transient free memory; prompt-level execution serialization ensures the earlier workflow releases the capacity first. If a persistent managed SSH worker is idle but still holds model memory from an earlier process, the planner safely recycles that owned worker and re-probes the host before admission. Unrelated GPU use receives no such exemption. Enabled SSH hosts are re-probed immediately before every placement decision, and an unreachable host is simply excluded, letting another provider serve as the fallback under automatic policy.

Provider boundaries are retained when values can be serialized between workers. When a `MODEL`, `CLIP`, `VAE`, `NOISE`, or `SAMPLER` producer is safely reproducible from loader or literal inputs, the planner replicates that dependency on the downstream worker instead of transferring the in-memory object or forcing unrelated phases onto the same provider. Non-reproducible runtime objects remain hard co-location constraints, and the strictest required provider applies to their complete component.

Completed components record execution timings per environment and workload signature. The planner uses the median of past runs to predict total cost, so a faster but more expensive GPU can win when its predicted total is lower; until history exists, it uses a conservative 60-second estimate.

## Using It In ComfyUI

### Marking Nodes For Remote Execution

Build the workflow normally — Modal-Sync does not replace standard nodes; it adds remote execution controls to the existing graph. Good remote candidates consume large model files, perform expensive tensor work, and exchange values that can cross the local/remote boundary.

1. Enable `Run Remotely` on each node that should belong to the remote region.
2. Confirm the node shows the blue remote-execution border.
3. Queue the workflow using ComfyUI's normal queue action.

The toggle stores `properties.is_modal_remote = true` in workflow metadata; the graph itself is only rewritten when the prompt is queued.

The node context menu includes a `Modal` submenu for bulk operations: a `GPU` submenu for the workflow's Modal GPU target, `Enable on Upstream Nodes` (which asks the backend which upstream nodes must join the remote region to keep the boundary transportable), and `Enable`/`Disable All Nodes`.

### Canvas State

Remote state is shown directly on the canvas, in both the legacy LiteGraph renderer and the Nodes 2.0 renderer:

- blue border: marked remote, idle
- orange pulsing border: queue-time setup or upload work
- yellow pulsing border: dispatched and waiting for remote execution feedback
- pulsing green border: ready and waiting
- pulsing purple border: executing remotely
- steady green border: finished for the current run
- red border: queue-time or execution failure
- neutral `!` badge on a local node: the planner found a remote → local → remote path through it, which may force extra component splits and transfers
- numbered badge: remote component assignment for the current prompt

Subgraph containers inherit a derived summary decoration (marked with a `Σ` badge) whenever any nested descendant runs remotely, using the most operationally important descendant state for the border color.

Active remote nodes show a small progress panel with a Modal, SSH terminal, or Vast.ai icon beside every bar, per-bar `it/s` rates, human-readable `B/s` through `TB/s` transfer rates for file downloads, and `tok/s`, token counts, and TTFT for LLM nodes. LLM preparation is reported on the actual LLM node—not the remote component's representative node—with distinct metadata inspection, snapshot lookup, concurrent-download wait, download preparation, transfer, cached, and ready states. When a prompt is scheduled across more than one remote environment, the panel also names where each component is running — the Modal container id or the SSH host name. Preview images and UI payloads from remote nodes stream back into the local session while the component is still running, and progress state is reconciled against queue history when the browser regains focus so cancelled prompts do not leave stale bars behind.

Remote planning and capacity acquisition count as active queue work even though they happen before the rewritten prompt enters ComfyUI's native execution queue. During Vast startup, the canvas receives readable updates for rental, worker-image download, SSH readiness, and runtime initialization. A provider-side terminal image-pull failure is reported immediately rather than appearing as an empty queue until the startup timeout expires. A rental that never produces an SSH-usable worker is marked unusable and destroyed after the bounded initialization wait so a later workflow cannot silently reuse it.

For workflows using a **Remote Execution Configurator**, that node is also the primary planning surface. As soon as scheduling finishes, the scheduler result is streamed to the panel before any Vast rental or SSH-readiness wait begins. The browser retains that plan across late node-panel mounts, a full page refresh, tab refocus, queue failure, and extension reloads, while removing any orphaned prior panel before remounting. The panel follows the node's current width as it is resized, including shrinking within narrow layouts instead of retaining its initial DOM-widget width. The execution plan is grouped into one section per concrete target, with labeled components beneath it and the best-known machine type, GPU/HBM or VRAM capacity, system RAM, predicted completion time, and predicted cost. Modal system RAM is explicitly shown as provider-managed because it is assigned dynamically rather than inferred as a fixed machine capacity. Configuration names and selected targets therefore remain inspectable even when capacity never becomes ready. The **Managed capacity** area independently lists every lease or container owned by this installation, including idle leftovers from earlier workflows, with provider, resource ID, lifecycle state, activity, GPU/RAM, and hourly price where available. Its **Reload** button refreshes the list, while each exact Vast lease or Modal container has a destructive **KILL** button with confirmation; ownership is revalidated by the server, and killing active capacity is explicitly allowed because the confirmation warns that its workflow will fail. A successful Vast kill clears the entire retained execution plan because removing capacity invalidates all of its scheduling assumptions; delayed events from that prompt cannot restore it. These controls opt into pointer handling inside ComfyUI's otherwise click-through canvas overlay, provide immediate pressed/loading feedback, and isolate their pointer gestures from node dragging. Connected storage backends are summarized beneath the execution plan rather than presented as scheduler targets; an R2 summary includes its bucket, current storage use and object count, jurisdiction, cache policy, and optional key prefix without exposing credentials. Bucket usage is read through the existing bucket-scoped S3 credential and cached for five minutes so repeated workflow submissions do not repeatedly enumerate the bucket. The storage area's **Reload** button bypasses that cache, retrieves fresh aggregate state, and updates the retained plan without queueing a workflow. Each selected environment has its own status and progress lane, so parallel Vast, Modal, and self-hosted setup or execution updates cannot overwrite each other; once a newer prompt owns the panel, delayed events from an older prompt remain isolated. Planned environments begin in a waiting state. When a planned Vast configuration resolves to a concrete lease, its existing lane is promoted in place so the configuration and instance identities do not briefly appear as duplicate capacity. Vast capacity then reports rental, worker-image download, initialization, and asset preparation before becoming ready. SSH rows distinguish completed asset synchronization from worker readiness and stream the later runtime check, image build, container start, and health wait; they report ready only after the worker socket answers with the expected fingerprint. A bounded controller pool prepares custom nodes and prompt assets concurrently across distinct environments, while preserving component order and request-cache reuse within each environment. A prompt-wide failure persistently terminates every environment lane and clears its in-progress meter, including when a later plan render or panel remount occurs. Live preparation and execution status use the same durable event stream. Queue-history reconciliation reads the prompt's actual terminal result, preserving a failure or interruption in the Configurator and replacing stale submission text when the prompt is no longer executing. Legacy workflows without a configurator continue to use the global status pill.

Queue-time preparation appears in ComfyUI's queue through the server-side preparation bridge, including the native job metadata required by the queue panel, while the Configurator panel and global status pill show its detailed phase. The frontend acknowledges the prompt immediately so ComfyUI owns a visible, cancellable job while remote capacity is still starting; ComfyUI's standard Jobs cancellation also signals the prompt's preparation task. Preparation deliberately does not emit fake native node-execution events: doing so starts ComfyUI's execution watchdog, which can incorrectly mark a node as stalled or failed during a healthy multi-minute asset download. A preparation failure remains a queue/setup error with its original message rather than being recast as a node execution crash.

While a Modal workflow is active, the global status pill lists this instance's active containers, an estimated accumulated GPU cost with the current burn rate (based on Modal's published GPU prices; excludes CPU, memory, storage, credits, and reservations), and — where the workspace plan allows billing-report access — Modal's actual metered app cost at hourly resolution.

Cancelling a local prompt propagates a targeted interrupt to the active remote work. If the back-end is slow to observe it, the local prompt is released after a grace period while remote cleanup continues.

### Batched And Mapped Workflows

`Modal Map Input` is a pass-through adapter node that fans one boundary input out across remote workers. Place it before a remote-marked region; at queue time each item of the mapped input becomes its own remote invocation.

Mapped execution supports scalar primitives (as one-item maps), Python lists, `IMAGE` batches, `LATENT` batches, and other batched tensors split on dimension 0. One `Modal Map Input` boundary is supported per remote component; non-mapped boundary inputs are broadcast unchanged to every item, and mapped outputs are reassembled in item order — concatenated where shapes allow, otherwise returned as an ordered list so downstream nodes execute once per item. Mapped components can mix one-time work (a shared model loader) with per-item work (a fanned-out sampler); the shared work runs once.

Ordinary components without `Modal Map Input` preserve ComfyUI's normal zipped-batch behavior at the remote boundary, and semantic batches destined for aggregate consumers (for example the frame sequence feeding `CreateVideo`) stay intact instead of being split per item.

Mapped progress is summarized on the global status pill and a representative node with counts such as `3/16`. Fan-out is clamped to `COMFY_MODAL_MAX_INFLIGHT_CALLS`.

## LLM Nodes

### Modal LLM

`Modal LLM` is a multimodal language-model node with two execution targets, controlled by the standard node toggle:

- **Toggle off** — run locally on Apple Silicon via MLX. Intended for MacBooks whose unified memory can hold an LLM while heavier image or video nodes run remotely. The `local_mlx_engine` control defaults to `auto` (text-only registered targets use `mlx-dspark` with speculative decoding; image/video requests use MLX-VLM).
- **Toggle on** — run inside the same remote GPU worker as the surrounding remote component.

The node accepts a prompt and optional system prompt, an `IMAGE` batch, one native ComfyUI `VIDEO` (uniformly sampled into timestamped frames), UTF-8 text or text-based PDF files, a per-request reasoning toggle, and bounded generation/sampling/memory controls. It returns three strings: the final `response`, compact `metadata_json`, and a separate `reasoning` channel.

Enter either a curated profile name from [`llm_profiles.json`](llm_profiles.json) or a Hugging Face `owner/model` ID directly (use `owner/model@revision` to pin a branch, tag, or commit). Each target inspects the model's metadata and compatibility before downloading weights, pins the exact commit, and reuses the completed snapshot on later runs. Unknown or target-incompatible architectures fail inspection before any weight download. Public models need no token; for gated models, set `HF_TOKEN` in ComfyUI's environment (local), in the Modal secret collection (Modal), or in the host's Docker env file (SSH).

**GGUF exception:** do not enter the raw Hugging Face repository ID for a GGUF model. Raw `owner/model` values use the Transformers-compatible resolver, which requires `config.json` and safetensors; GGUF repositories normally provide neither. Select a curated GGUF profile instead so Modal-Sync knows the exact `.gguf` weight, optional multimodal projector, tokenizer source, and `llama.cpp` backend. For example, use `huihui-qwen3.8-27b-abliterated-q2-k-gguf`, not `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`.

Apple-local inference requires macOS on Apple Silicon and the optional pinned runtime in the ComfyUI Python environment:

```bash
uv pip install --python <comfyui-venv>/bin/python \
  "mlx-vlm==0.6.15" "mlx-dspark==0.13.1" "psutil>=7,<8" \
  "huggingface-hub==1.28.0" "hf-xet==1.6.0"
```

Local snapshots are stored under `<ComfyUI models>/modal_llm`; set `COMFY_MODAL_LOCAL_LLM_STORAGE_ROOT` to relocate them.

Loaded models are cached in a per-target LRU and reused across executions. Before a cold load, the manager asks ComfyUI to release idle managed models, evicts older resident LLMs if needed, and enforces a free-memory reserve (`local_reserve_free_memory_gb` for macOS unified memory, `reserve_free_vram_gb` for remote GPUs). On Modal, vLLM-backed profiles default to eager mode for a container's first workflow and are promoted to throughput mode — backed by a persistent compilation cache — when the container serves a second one; pin either behavior with `COMFY_MODAL_LLM_VLLM_EXECUTION_MODE`.

Model staging is deliberately kept off billed GPU time: weights are downloaded by a CPU-only stager (Modal) or inside the persistent worker (SSH) before the GPU allocation begins, and the planner applies each model's VRAM/RAM floors before placement so a too-small GPU cannot win cost ranking for a checkpoint it cannot load. The planner's already validated, credential-free model profile travels with the execution payload, so the worker reports **Using planner-resolved metadata** instead of repeating Hugging Face inspection. Before downloading, the stager checks that the missing artifacts fit while retaining a free-disk reserve. Snapshot leases carry a host, PID, process-start identity, and heartbeat; a dead same-host owner is reclaimed immediately, while a live download cannot lose its lease merely because it runs for a long time. Cancelling a Vast or SSH staging phase targets that exact owner process without recycling the whole worker, and every provider aborts a staging stream that produces no progress within the configured deadline.

### Modal Endpoint Chat

`Modal Endpoint Chat` is a separate node for Modal's hosted-model endpoints, laid out like ComfyUI's built-in `OpenAI ChatGPT` node: a prompt, an optional `IMAGE` batch, and optional files from `OpenAI ChatGPT Input Files`. Point it at a Modal Direct endpoint:

```text
https://your-workspace--your-endpoint.us-west.modal.direct
```

The node calls the endpoint's OpenAI-compatible `/v1/chat/completions` API. Enter a model ID, or leave it blank to use the first ID advertised by `/v1/models`. Because Modal endpoints scale to zero, an empty 503 during cold start is retried with bounded backoff until the node's `timeout_seconds` deadline.

Authentication is resolved in this order:

1. `MODAL_KEY` and `MODAL_SECRET` from the ComfyUI process environment — proxy-token values with `wk-`/`ws-` prefixes, not `ak-`/`as-` API credentials.
2. A previously saved pair in the operating-system credential vault under the `ComfyUI Modal-Sync` service.
3. A new pair created with `modal workspace proxy-tokens create --json` (requires an authenticated Modal CLI).

Automatically created tokens go directly to the OS vault (Keychain on macOS, the native `keyring` backend elsewhere) and are never written into the workflow, settings, or logs. In RBAC-enabled workspaces, vault-backed tokens are authorized for the node's `environment` setting (default `main`) before use. A headless installation without a secure keyring must supply both environment variables instead.

For credential safety, the node accepts only HTTPS `modal.direct` origins, refuses redirects, and has no `Run Remotely` toggle of its own.

## Asset And Custom Node Sync

The sync engine looks for node inputs that resolve to model files ending in `.safetensors`, `.ckpt`, `.gguf`, `.pt`, or `.vae`. Absolute paths and model names resolvable through ComfyUI's `folder_paths` both work; arbitrary unresolved strings do not sync, and a remote-marked node depending on an unresolvable model filename fails at queue time rather than sending a broken remote request.

Assets are content-addressed and uploaded once: repeated references across nodes and components share one hash and upload decision, and unchanged files are skipped on later runs. In local mode, a local mirror stands in for remote storage.

For Vast.ai, a single-file model can instead be downloaded directly from Hugging Face over the rented instance's data-center connection. This is automatic for models installed through ComfyUI Manager, files or symlinks in the standard Hugging Face cache, Hugging Face Git checkouts, browser downloads that retain their source URL, and safetensors that embed an official Hub source URL. These sources are treated only as candidates: Modal-Sync resolves the revision to an exact 40-character commit and requires the Hub's LFS/Xet SHA-256 and size to match the local bytes before persisting the learned mapping below `<ComfyUI user directory>/comfyui-modal/huggingface-assets.json`.

Future Vast cache misses use `huggingface_hub` with `hf-xet` and high-performance mode inside the worker, verify the downloaded size and SHA-256 again, then atomically publish the file under the same content-addressed `/storage/assets/...` path used by ordinary sync. The configurator panel reports both automatic source identification and the Hugging Face download while preparation is active. Because Hugging Face does not provide a global reverse lookup from an arbitrary digest to every repository path, an originless file simply falls back to the normal streamed SSH upload. No user action is required.

The `scripts/register_huggingface_asset.py` utility remains available as a diagnostic override for a file whose installation metadata was removed. It is not part of normal model installation or workflow use.

Public files need no token. For private or gated files, set `HF_TOKEN` in the local ComfyUI process. Modal-Sync uses it for metadata verification and sends it only through the protected SSH standard input of the transient materializer; it is not stored in the workflow, registry, Vast launch environment, command arguments, or logs. If Hugging Face is unavailable, authorization fails, or verification rejects the downloaded bytes, preparation reports the fallback and streams the local file through SSH without buffering the complete model in ComfyUI memory.

The worker-side materializer is part of the versioned runtime source. Rebuild and publish the SSH/Vast worker image after upgrading before expecting an already configured Vast deployment to use this path.

Custom-node sync is enabled by default in remote mode. When enabled, Modal-Sync packages `custom_nodes/` as content-addressed code archives per package, with package-owned model artifacts (`.pth`, `.safetensors`, `.gguf`, `.onnx`) stored separately so a code edit never re-uploads a multi-gigabyte model. Nested virtual environments, caches, compiled artifacts, and logs are excluded. When a synced package has a `requirements.txt`, those requirements are folded into the worker image build (with `-r` includes followed; pip options and constraints ignored).

### Optional Cloudflare R2 Backing Cache

Vast.ai and self-hosted SSH environments can optionally share a Cloudflare R2 bucket as a content-addressed backing cache. On a local sync-index miss, Modal-Sync first adopts an existing file from persistent worker storage, then prefers a verified Hugging Face origin when one is known, then checks R2 by SHA-256 and exact byte size. An R2 hit is downloaded directly over the worker's data-center connection, verified again, and atomically published into `/storage`; an R2 failure falls back to the existing local-to-worker SSH upload.

New local or Hugging Face transfers are written back to R2 by default on a two-worker background executor, so the current workflow does not wait for cache population. Use `COMFY_MODAL_R2_WRITE_BACK=sync` when cache durability must be confirmed before preparation completes, or `off` for read-only use. The cache also covers custom-node archives, manifests, and package-owned model assets. Modal execution does not use R2 because its native persistent volume already supplies the fast shared layer.

Permanent R2 access credentials remain only in the local ComfyUI controller. Workers receive expiring, exact-object S3-compatible GET/PUT URLs through protected process standard input; URLs do not enter workflow data, Docker/SSH command arguments, launch environments, or logs. Downloads are SHA-256 and size verified rather than trusting an S3 ETag. Files above the configured single-PUT threshold use multipart upload, which also avoids buffering model files in controller memory.

For a configured workflow, add **R2 Storage Configuration**, connect it to the same **Remote Execution Configurator** as the Vast.ai or SSH capacity nodes, choose the private bucket name, and select **Login to Cloudflare** directly on the R2 node. Login does not queue or execute the workflow. The browser uses Cloudflare Authorization Code + PKCE, and the controller creates or reuses the requested bucket. Cloudflare does not expose API-token creation to third-party OAuth clients, so the node then opens a secure credential-import dialog. Follow its link to **R2 > Manage R2 API Tokens**, create an **Object Read & Write** token restricted to the configured bucket, and paste the one-time **Access Key ID** and **Secret Access Key** values into the dialog. The controller verifies bucket access before storing the pair in the operating-system credential vault through `keyring`. The workflow contains only the account ID, bucket, jurisdiction, cache policy, and a random opaque credential reference.

Register one public Cloudflare OAuth client for the extension before using the Login button. Configure it for Authorization Code + PKCE, token endpoint authentication `none`, response type `code`, and the callback shown below. Do not create or distribute a client secret for the desktop client. In Cloudflare's friendly-name scope picker, enable **Account Settings Read** and **Workers R2 Storage Write**. Do not enable **openid** or **Access: Service Tokens Write**; neither is required for R2 provisioning, and Zero Trust service tokens are unrelated to R2 S3 credentials.

```bash
export COMFY_MODAL_CLOUDFLARE_OAUTH_CLIENT_ID='your-public-client-id'
# Register this exact callback, adjusting the loopback port when needed:
export COMFY_MODAL_CLOUDFLARE_OAUTH_REDIRECT_URI='http://127.0.0.1:8188/remote/storage/r2/oauth/callback'
# Defaults shown explicitly; these must match the scopes on the registered client:
export COMFY_MODAL_CLOUDFLARE_OAUTH_SCOPES='account-settings.read workers-r2.write'
```

When ComfyUI is opened through an HTTP loopback URL, the callback is derived automatically if the redirect variable is omitted. A remote, reverse-proxied, or HTTPS installation must set the registered redirect URI explicitly. When the OAuth grant contains more than one Cloudflare account, enter the desired 32-character account ID on the node and select Login again. R2 must already be activated for that account; Cloudflare's first-time checkout remains an interactive dashboard operation.

ComfyUI normally rejects all cross-site browser requests through its origin-only middleware. Cloudflare's redirect is necessarily cross-site, so the extension installs a narrow exception ahead of that guard for `GET /remote/storage/r2/oauth/callback` only. The callback remains protected by its random, single-use OAuth state and PKCE verifier; all other cross-site ComfyUI requests continue through the standard guard.

The environment-only configuration remains supported for legacy workflows and scripts. Create a private R2 bucket and an API token scoped to that bucket, then configure the ComfyUI process:

```bash
export COMFY_MODAL_R2_ENABLED=true
export COMFY_MODAL_R2_ACCOUNT_ID='0123456789abcdef0123456789abcdef'
export COMFY_MODAL_R2_BUCKET='comfy-model-cache'
export COMFY_MODAL_R2_ACCESS_KEY_ID='...'
export COMFY_MODAL_R2_SECRET_ACCESS_KEY='...'
```

The first ordinary cache miss populates R2 automatically. To seed known files or directories before renting a GPU, run:

```bash
uv run python scripts/prewarm_r2_cache.py /path/to/models /path/to/custom-file.safetensors
```

## Configuration

Boolean values accept `1`, `true`, `yes`, `on`, `0`, `false`, `no`, and `off`.

### Routing And Metadata

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_ROUTE_PATH` | `/modal/queue_prompt` | Queue endpoint registered by the backend. |
| `COMFY_MODAL_MARKER_PROPERTY` | `is_modal_remote` | Workflow property used to mark remote nodes. |

### Paths And Sync

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFYUI_ROOT` | auto-discovered | Preferred ComfyUI checkout root for tests and path resolution. |
| `COMFY_MODAL_COMFYUI_ROOT` | auto-discovered | Modal-Sync-specific ComfyUI checkout override, used after `COMFYUI_ROOT`. |
| `COMFY_MODAL_CUSTOM_NODES_DIR` | auto-discovered | `custom_nodes` directory to bundle and mirror. |
| `COMFY_MODAL_LOCAL_STORAGE_ROOT` | `/tmp/comfyui-modal-sync-storage` | Local mirror root for local mode, tests, and dry runs. |
| `COMFY_MODAL_REMOTE_STORAGE_ROOT` | `/storage` | Mounted storage root inside remote workers. |
| `COMFY_MODAL_CUSTOM_NODES_ARCHIVE` | `custom_nodes_bundle.zip` | Base archive name used for custom-node bundle paths. |
| `COMFY_MODAL_SYNC_CUSTOM_NODES` | `false` in local mode, `true` otherwise | Force-enable or disable custom-node bundle sync. |
| `COMFY_MODAL_R2_ENABLED` | `false` | Enable the shared R2 backing cache for Vast.ai and SSH Docker storage. |
| `COMFY_MODAL_R2_ACCOUNT_ID` | unset | Cloudflare account ID used to derive the R2 S3 endpoint. |
| `COMFY_MODAL_R2_BUCKET` | unset | Private R2 bucket containing content-addressed objects. |
| `COMFY_MODAL_R2_ACCESS_KEY_ID` | unset | Controller-only R2 API-token access key ID. |
| `COMFY_MODAL_R2_SECRET_ACCESS_KEY` | unset | Controller-only R2 API-token secret access key. |
| `COMFY_MODAL_R2_ENDPOINT_URL` | account R2 endpoint | Optional credential-free HTTPS S3 endpoint override. |
| `COMFY_MODAL_R2_KEY_PREFIX` | `comfy-modal-cache/v1/blobs/sha256` | Immutable object-key namespace shared by compatible installations. |
| `COMFY_MODAL_R2_WRITE_BACK` | `async` | Cache population mode: `async`, `sync`, or `off`. |
| `COMFY_MODAL_R2_URL_TTL_SECONDS` | `21600` | Presigned URL lifetime, from 1 second through R2's 7-day limit. |
| `COMFY_MODAL_R2_MULTIPART_PART_MIB` | `256` | Multipart upload part size, from 5 MiB through 5 GiB. |
| `COMFY_MODAL_R2_SINGLE_UPLOAD_MAX_MIB` | `100` | Maximum file size sent with one signed PUT, capped at 5 GiB. |
| `COMFY_MODAL_CLOUDFLARE_OAUTH_CLIENT_ID` | unset | Public Cloudflare OAuth client ID used by the R2 node Login button. |
| `COMFY_MODAL_CLOUDFLARE_OAUTH_REDIRECT_URI` | loopback page origin | Exact registered OAuth callback; required for non-loopback ComfyUI URLs. |
| `COMFY_MODAL_CLOUDFLARE_OAUTH_SCOPES` | `account-settings.read workers-r2.write` | Space- or comma-separated scope IDs configured on the Cloudflare OAuth client. |

### Modal Deployment

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_EXECUTION_MODE` | `local` | Set to `remote` for remote execution. |
| `MODAL_ENVIRONMENT` | active Modal profile or workspace default | Modal environment used for app deployment, lookup, status, and billing. |
| `COMFY_MODAL_APP_NAME` | `comfy-modal-sync-<instance_id>` | Explicit Modal app name override; otherwise derived from the persistent per-installation identity. |
| `COMFY_MODAL_INSTANCE_ID_PATH` | `<ComfyUI user directory>/.comfy-modal-sync-instance-id` | Override the persistent identity file location. |
| `COMFY_MODAL_SECRET_NAME` | `comfy` | Modal secret collection injected into every remote worker as environment variables. |
| `COMFY_MODAL_VOLUME_NAME` | `comfy-universal-storage` | Modal volume name for synced assets and bundles. |
| `COMFY_MODAL_AUTO_DEPLOY` | `true` | Deploy or replace the configured app when lookup fails or its runtime fingerprint is stale. |
| `COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK` | `false` | Allow temporary `app.run()` execution when deployed lookup fails. |
| `COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR` | `true` | Make a remote worker exit after surfacing a crash. |

### Self-Hosted SSH And Vast.ai

| Variable | Default | Purpose |
| --- | --- | --- |
| `VAST_API_KEY` | unset | Vast.ai account API key, required for Vast execution. |
| `COMFY_MODAL_VAST_IMAGE` | unset | Immutable worker image reference (digest form recommended), required for Vast execution. |
| `COMFY_MODAL_VAST_API_BASE_URL` | Vast.ai production API | Override for the local API simulator; non-loopback values must be HTTPS. |
| `COMFY_MODAL_VAST_SSH_IDENTITY_FILE` | SSH default identity | Absolute private-key path when the Vast account does not use the default SSH identity. |

New workflow SSH host settings (destinations, cost, worker limits, VRAM reserve, tags) live in **SSH Configuration** nodes. The **Settings → Remote Execution: SSH environments** panel remains the legacy installation-wide registry and preflight interface.

### Modal State Stores

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_INTERRUPT_DICT_NAME` | `<app_name>-interrupts` | Shared Modal `Dict` for cancellation flags. |
| `COMFY_MODAL_NODE_CACHE_DICT_NAME` | `<app_name>-node-cache` | Shared Modal `Dict` for persisted transport-safe node outputs. |
| `COMFY_MODAL_SESSION_BRIDGE_DICT_NAME` | `<app_name>-session-bridges` | Shared Modal `Dict` for durable session bridge metadata. |
| `COMFY_MODAL_INVOCATION_DICT_NAME` | `<app_name>-invocations` | Shared Modal `Dict` for idempotent invocation lifecycle and result metadata. |
| `COMFY_MODAL_SYNC_INDEX_DICT_NAME` | `<app_name>-sync-index` | Shared Modal `Dict` for mirrored asset and bundle digests. |
| `COMFY_MODAL_SNAPSHOT_PROFILE_DICT_NAME` | `<app_name>-snapshot-profiles` | Shared Modal `Dict` for loader snapshot profile records. |
| `COMFY_MODAL_NODE_CACHE_MAX_BYTES` | `5242880` | Maximum raw output size eligible for persisted node caching; `0` disables. |
| `COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES` | `4194304` | Maximum serialized bridge value retained inline before Volume offload. |
| `COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES` | `4194304` | Maximum completed invocation result retained inline before Volume offload. |

### Runtime Sizing And Warmup

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_GPU` | `RTX-PRO-6000` | Fallback Modal GPU for workflows or API clients without `workflow.extra.comfy_modal.gpu`; the context-menu selection is authoritative for normal UI runs. |
| `COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT` | `true` | Enable Modal CPU memory snapshots. |
| `COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT` | `true` | Enable Modal GPU memory snapshots for profiled loader states. |
| `COMFY_MODAL_SCALEDOWN_WINDOW` | `600` | Seconds to keep idle Modal containers warm. |
| `COMFY_MODAL_LOCAL_GAP_KEEPALIVE_SECONDS` | `900` | Maximum time to retain a matching worker slot across a remote → local → remote workflow gap; `0` disables pulses. |
| `COMFY_MODAL_LOCAL_GAP_KEEPALIVE_INTERVAL_SECONDS` | `15` | Seconds between worker-retention pulses during a local gap. |
| `COMFY_MODAL_MIN_CONTAINERS` | `0` | Minimum warm containers. |
| `COMFY_MODAL_MAX_CONTAINERS` | unset | Optional upper bound on simultaneously scaled Modal containers. |
| `COMFY_MODAL_BUFFER_CONTAINERS` | unset | Optional spare warm containers above current load. |
| `COMFY_MODAL_MAX_INFLIGHT_CALLS` | `4` | Maximum local remote calls dispatched at once; mapped fan-out is clamped to this budget. |
| `COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS` | `3600` | Maximum runtime for one remote workflow call. |
| `COMFY_MODAL_STARTUP_TIMEOUT_SECONDS` | `900` | Maximum container startup and snapshot-restore time. |
| `COMFY_MODAL_LLM_MAX_RESIDENT_MODELS` | `2` | Maximum LLM profiles retained per warm GPU worker before LRU eviction. |
| `COMFY_MODAL_LLM_RESERVE_FREE_GB` | `24.0` | Default minimum free VRAM retained for ComfyUI-managed image and video models; the node can override per request. |
| `COMFY_MODAL_LLM_VLLM_EXECUTION_MODE` | `auto` | `auto` uses eager for a container's first workflow and promotes to throughput on its second; `eager` and `throughput` pin one mode. |
| `COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS` | `15.0` | Maximum post-eviction wait for CUDA free memory before retiring the worker and retrying once on a fresh worker. |
| `COMFY_MODAL_LLM_MIN_FREE_DISK_GB` | `8.0` | Free storage that must remain after staging the model bytes not already present in a resumable snapshot. |
| `COMFY_MODAL_LLM_STAGE_LEASE_TIMEOUT_SECONDS` | `7200` | Maximum wait for a foreign or legacy snapshot lease; dead local process owners are reclaimed immediately and live leases heartbeat. |
| `COMFY_MODAL_LLM_STAGE_LEASE_HEARTBEAT_STALE_SECONDS` | `300` | Age after which a structured lease with a missing heartbeat is reclaimable, including after its former worker container disappears. Must exceed two heartbeat intervals. |
| `COMFY_MODAL_LLM_STAGE_NO_PROGRESS_TIMEOUT_SECONDS` | `600` | Maximum silence between model-staging events before the controller terminates the transport and requests provider-side cancellation. |
| `COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE` | `256` | Maximum buffered remote progress/result envelopes; stale progress is coalesced when full. |
| `COMFY_MODAL_ENABLE_PROACTIVE_WARMUP` | `true` | Start background warmup from runtime parallelism signals and planner lookahead. |
| `COMFY_MODAL_ENABLE_LOADER_PREWARM` | `true` | During warmup, execute synthetic loader prompts for root literal model-loader nodes. |
| `COMFY_MODAL_LOADER_PREWARM_WORKERS` | `2` | Maximum independent loader plans executed concurrently in one worker during prewarm. |

### Cancellation And Logs

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS` | `2.0` | How long the local proxy waits after propagating cancellation before releasing the local prompt. |
| `COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS` | `1.0` | How long a remote worker waits after observing cancellation before exiting if execution is still stuck. |
| `COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS` | `false` | Mirror live Modal container logs into local ComfyUI stderr during streamed executions. |

## Troubleshooting

- **App not found or deleted**: leave `COMFY_MODAL_AUTO_DEPLOY=true` so the next lookup deploys the app again.
- **Changed the workflow GPU**: just queue the next remote run — Modal-Sync looks up or builds the GPU-specific app without touching the previous one.
- **Remote mode still uses local mirror storage**: restart ComfyUI with `COMFY_MODAL_EXECUTION_MODE=remote` and the Modal SDK available so sync and invocation resolve the same mode.
- **Missing custom node class on the remote worker**: ensure custom-node sync is enabled, check the worker logs for import failures, and confirm the package's Python dependencies are listed in its `requirements.txt`.
- **`UNETLoader` reports `Could not detect model type` for a synced Flux-style model**: Modal-Sync aliases saved RMSNorm `.weight` keys to ComfyUI's `.scale` form before remote model detection, but the model still has to be supported by the ComfyUI checkout packaged into the worker.
- **Boundary validation fails on `MODEL`, `CLIP`, `VAE`, `CONDITIONING`, or similar**: include the upstream producer in the remote region, or use `Enable on Upstream Nodes`.
- **`Dependency cycle detected` after rewrite**: inspect local `comfy.log` for the proxy-graph diagnostics and cycle path.
- **Cancellation finishes locally while the back-end is still busy**: the local prompt is released after the grace window; remote cleanup or worker retirement may still be completing.
- **Remote behavior does not reflect a local code update**: redeploy the configured Modal app, or rebuild the SSH/Vast worker image, so remote workers use the current code.

## Development

Manage the project with `uv`:

```bash
uv sync --group test
uv run pytest
```

Tests look for ComfyUI in `COMFYUI_ROOT` first, then `COMFY_MODAL_COMFYUI_ROOT`, then an installed parent checkout, then `~/git/ComfyUI`. To run against a temporary checkout:

```bash
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/comfyui-modal-test/ComfyUI
UV_PROJECT_ENVIRONMENT=/tmp/comfyui-modal-test-env uv sync --group test
COMFYUI_ROOT=/tmp/comfyui-modal-test/ComfyUI \
  /tmp/comfyui-modal-test-env/bin/python -m pytest
```

For full local development including the remote and Apple-local extras, use `uv sync --extra remote --extra local-apple --group test`.

### Vast.ai API Simulator

Vast development and CI need no live credentials or billable instances. Start the stateful local simulator:

```bash
uv run python scripts/run_vast_api_simulator.py --port 8099 --api-key vast-test-key
```

It implements the account check, offer search, offer-rental races, instance lifecycle, and destruction paths used by the extension, with default offers covering 24 GB, 48 GB, and 80 GB GPU tiers. Point the production client at it for offline work:

```bash
export VAST_API_KEY=vast-test-key
export COMFY_MODAL_VAST_API_BASE_URL=http://127.0.0.1:8099
```

The production client accepts plaintext HTTP only for loopback addresses; live credentials are sent only to HTTPS endpoints.

### Live Canaries And Benchmarks

Live canaries are opt-in because they authenticate against real providers, start real GPU capacity, and may incur charges. All ordinary tests remain local-only.

The Modal canaries validate the deployed runtime fingerprint, binary tensor transport with durable duplicate replay, two-call remote concurrency, and cancellation propagation:

```bash
COMFY_MODAL_RUN_LIVE_CANARIES=1 \
COMFY_MODAL_EXECUTION_MODE=remote \
uv run --extra remote pytest -q tests/test_live_modal_canary.py
```

The Vast canary rents bounded capacity, verifies the real worker, and destroys it by default (set `COMFY_MODAL_VAST_KEEP_CANARY=1` to deliberately retain the lease, with a one-hour idle retention):

```bash
COMFY_MODAL_RUN_LIVE_VAST=1 uv run pytest -q tests/test_live_vast_canary.py
```

[`scripts/benchmark_modal_llm.py`](scripts/benchmark_modal_llm.py) is a billable standalone harness for measuring cold-start and resident-engine LLM performance on a dedicated Modal app; see its `--help` for options.

### Registry Metadata

The repository is structured as a ComfyUI Registry node pack with metadata in [`pyproject.toml`](pyproject.toml) and a publish workflow in [`.github/workflows/publish_action.yml`](.github/workflows/publish_action.yml). The registry pack name is `modal-sync`, the display name is `Modal Sync`, and the current publisher id is `ttulttul`.

[`modal_test_workflow.json`](modal_test_workflow.json) is a checked-in smoke artifact from a successful remote run, not a pristine authoring workflow.

## Current Limitations

- Remote execution is component-based: a local gap in the middle of a remote chain still requires transport-safe values at each boundary.
- Non-JSON, non-bytes, non-tensor payloads are not supported across the local/remote boundary.
- The Vast.ai back-end currently supports public worker images only; private-registry credentials are deliberately not stored in workflows.
- Workflow files captured after a remote run may include internal proxy nodes such as `ModalUniversalExecutor`; they are useful as regression fixtures but are not clean source workflows.
