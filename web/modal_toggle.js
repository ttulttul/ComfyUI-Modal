import { app } from "../../scripts/app.js";
import { PromptExecutionError, api } from "../../scripts/api.js";
import { openRemoteEnvironmentManager } from "./remote_environments.js";

const REMOTE_PROPERTY = "is_modal_remote";
const REMOTE_WIDGET_NAME = "Run Remotely";
const LEGACY_REMOTE_WIDGET_NAME = "Run on Modal";
const MODAL_ROUTE = "/modal/queue_prompt";
const MODAL_ANALYZE_ROUTE = MODAL_ROUTE.replace(/\/queue_prompt$/, "/analyze_remote_nodes");
const MODAL_PROGRESS_STATE_ROUTE = MODAL_ROUTE.replace(/\/queue_prompt$/, "/progress_state");
const MODAL_CONTAINER_STATUS_ROUTE = MODAL_ROUTE.replace(/\/queue_prompt$/, "/container_status");
const MODAL_DELETE_CACHES_ROUTE = MODAL_ROUTE.replace(/\/queue_prompt$/, "/delete_caches");
const MODAL_DELETE_VOLUME_ROUTE = MODAL_ROUTE.replace(/\/queue_prompt$/, "/delete_volume");
const MODAL_CANCEL_PREPARATION_ROUTE = MODAL_ROUTE.replace(
  /\/queue_prompt$/,
  "/cancel_preparation",
);
const COMFY_QUEUE_ROUTE = "/queue";
const COMFY_HISTORY_ROUTE = "/history";
const INTERNAL_NODE_PREFIX = "ModalUniversalExecutor";
const LOCAL_MODAL_NODE_IDS = new Set([
  "ModalEndpointChat",
  "VastAILeaseConfiguration",
  "ModalRemoteConfiguration",
  "VastRemoteConfiguration",
  "SshRemoteConfiguration",
  "RemoteExecutionConfigurator",
]);
const WORKFLOW_MODAL_CONFIG_KEY = "comfy_modal";
const WORKFLOW_MODAL_GPU_KEY = "gpu";
const WORKFLOW_REMOTE_CONFIG_KEY = "remote_execution";
const WORKFLOW_REMOTE_POLICY_KEY = "policy";
const WORKFLOW_REMOTE_AUTO_PLACE_KEY = "auto_place";
const EXECUTION_POLICIES = [
  ["modal", "Modal only"],
  ["self_hosted", "Self-hosted only"],
  ["vast", "Vast.ai only"],
  ["automatic", "Automatic (lowest cost compatible)"],
];
const DEFAULT_MODAL_GPU = "RTX-PRO-6000";
const MODAL_GPU_TYPES = [
  "T4",
  "L4",
  "A10",
  "L40S",
  "A100",
  "A100-40GB",
  "A100-80GB",
  "RTX-PRO-6000",
  "H100",
  "H100!",
  "H200",
  "B200",
  "B200+",
  "B300",
];

const IDLE_BORDER_COLOR = "#1d9bf0";
const SETUP_BORDER_COLOR = "#f59e0b";
const STARTING_BORDER_COLOR = "#eab308";
const FINALIZING_BORDER_COLOR = "#3b82f6";
const CANCELLING_BORDER_COLOR = "#fb7185";
const READY_ACTIVE_COMPONENT_BORDER_COLOR = "#22c55e";
const READY_INACTIVE_COMPONENT_BORDER_COLOR = "#166534";
const ACTIVE_BORDER_COLOR = "#a855f7";
const COMPLETE_BORDER_COLOR = "#004FA4";
const COMPLETE_FILL_COLOR = "#001C71";
const FINALIZING_NODE_BORDER_COLOR = "#00358A";
const ERROR_BORDER_COLOR = "#ef4444";
const LOCAL_BOTTLENECK_BADGE_BORDER_COLOR = "rgba(148, 163, 184, 0.72)";
const LOCAL_BOTTLENECK_TOOLTIP = "Did you mean to make this node execute on Modal?";
const REMOTE_LOCATION_ICON_SOURCES = {
  modal: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 368 192"><path fill="#34D058" d="M0 128 73 0l37 64-73 128z"/><path fill="#00B75A" d="m110 64 74 0-74 128H37z"/><path fill="#8AEF74" d="M73 0h74l37 64h-74z"/><path fill="#34D058" d="M184 64 220 0l74 128-37 64z"/><path fill="#A7F59A" d="M220 0h74l74 128h-74z"/><path fill="#00B75A" d="m294 128h74l-37 64h-74z"/></svg>',
  )}`,
  ssh_docker: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 73 73"><rect x="1" y="1" width="71" height="71" rx="14" fill="#fff" stroke="#000" stroke-width="2"/><path d="M60.24 17.02H12.66A1.66 1.66 0 0 0 11 18.68v8.26h50.9v-8.26a1.66 1.66 0 0 0-1.66-1.66Zm-17.2 7.18a2.21 2.21 0 1 1 0-4.43 2.21 2.21 0 0 1 0 4.43Zm6.03 0a2.21 2.21 0 1 1 0-4.43 2.21 2.21 0 0 1 0 4.43Zm6.03 0a2.21 2.21 0 1 1 0-4.43 2.21 2.21 0 0 1 0 4.43ZM11 30.26v23.83c0 .92.74 1.66 1.66 1.66h47.58c.92 0 1.66-.74 1.66-1.66V30.26H11Zm23.41 14.01-5.44 4.62a1.66 1.66 0 0 1-2.15-2.53l3.95-3.35-3.95-3.36a1.66 1.66 0 1 1 2.15-2.53l5.44 4.62a1.66 1.66 0 0 1 0 2.53ZM45 49.29h-6.55a1.66 1.66 0 1 1 0-3.32H45a1.66 1.66 0 1 1 0 3.32Z"/></svg>',
  )}`,
};

const STATE_SETUP = "setup";
const STATE_STARTING = "starting";
const STATE_WAITING = "waiting";
const STATE_FINALIZING = "finalizing";
const STATE_CANCELLING = "cancelling";
const EXECUTION_PHASE = "executing";
const STATE_READY = "ready";
const STATE_ACTIVE = "active";
const STATE_COMPLETE = "complete";
const STATE_ERROR = "error";
const ERROR_CLEAR_DELAY_MS = 5000;
const TERMINAL_PROMPT_RETENTION_MS = 60000;
const PROGRESS_FADE_MS = 900;
const REFOCUS_STALE_PROMPT_GRACE_MS = 30000;
const MODAL_ANIMATION_FRAME_INTERVAL_MS = 100;
const ITERATION_RATE_SMOOTHING_FACTOR = 0.35;
const CONTAINER_STATUS_FAST_POLL_MS = 1500;
const CONTAINER_STATUS_STABLE_POLL_MS = 5000;
const CONTAINER_STATUS_HIDDEN_POLL_MS = 15000;
const CONTAINER_STATUS_MAX_BACKOFF_MS = 30000;

const modalNodeStates = new Map();
const modalNodeProgress = new Map();
const modalNodeProgressLanes = new Map();
const modalNodeBatchProgress = new Map();
const modalNodeCachedStates = new Map();
const modalNodeClearTimers = new Map();
const modalPromptStates = new Map();
const modalTerminalPromptStates = new Map();
const modalCancellingPromptIds = new Set();
const modalQueuedPromptIds = new Set();
const modalSandwichedLocalNodeIds = new Set();
const modalRemoteDescendantNodeIdsByAncestor = new Map();
const syntheticPromptUiStates = new Map();
const modalGlobalStatusStates = new Map();
const remoteLocationIconImages = new Map();
const remoteConfiguratorPanels = new Map();

let animationFrameHandle = null;
let modalLastAnimationRedrawAt = 0;
let modalGlobalStatusElement = null;
let modalVisibilityRefreshInFlight = null;
let modalReplayedEventUpdatedAtMs = null;
let vueNodeObserver = null;
let vueNodeSyncScheduled = false;
let remoteDescendantIndexRebuildScheduled = false;
let modalContainerStatuses = [];
let modalContainerStatusPromptId = null;
let modalContainerStatusLoaded = false;
let modalContainerStatusError = null;
let modalContainerStatusTimer = null;
let modalContainerStatusPollInFlight = false;
let modalContainerStatusUnchangedPolls = 0;
let modalContainerStatusFailureCount = 0;
let modalContainerEstimatedCostUsd = 0;
let modalContainerCostUpdatedAtSeconds = null;
let modalHourlyBillingStatus = null;
let modalHourlyBillingError = null;

/**
 * Return whether a node should show the Modal toggle.
 * @param {LGraphNode} node
 * @returns {boolean}
 */
function isEligibleNode(node) {
  return (
    Boolean(node?.comfyClass) &&
    !String(node.comfyClass).startsWith(INTERNAL_NODE_PREFIX) &&
    !LOCAL_MODAL_NODE_IDS.has(String(node.comfyClass))
  );
}

/**
 * Read the remote execution flag from node properties.
 * @param {LGraphNode} node
 * @returns {boolean}
 */
function isRemoteNode(node) {
  return Boolean(node?.properties?.[REMOTE_PROPERTY]);
}

/**
 * Return whether a node is a subgraph container with Modal-enabled descendants.
 * @param {LGraphNode} node
 * @returns {boolean}
 */
function hasRemoteDescendants(node) {
  const workflowPath = workflowNodePath(node) || nodeId(node);
  return (modalRemoteDescendantNodeIdsByAncestor.get(workflowPath)?.size ?? 0) > 0;
}

/**
 * Return whether the latest planner result identified a local re-entry bottleneck.
 * @param {LGraphNode} node
 * @returns {boolean}
 */
function isSandwichedLocalNode(node) {
  if (isRemoteNode(node)) {
    return false;
  }
  const workflowPath = workflowNodePath(node) || nodeId(node);
  return modalSandwichedLocalNodeIds.has(workflowPath);
}

/**
 * Replace the local re-entry warnings with one planner result.
 * @param {Array<string | number>} nodeIds
 */
function setSandwichedLocalNodeIds(nodeIds) {
  modalSandwichedLocalNodeIds.clear();
  for (const nodeIdValue of nodeIds ?? []) {
    modalSandwichedLocalNodeIds.add(String(nodeIdValue));
  }
  refreshNodeDecorations();
}

/**
 * Clear planner warnings after the user changes remote selection.
 */
function clearSandwichedLocalNodeWarnings() {
  if (modalSandwichedLocalNodeIds.size === 0) {
    return;
  }
  modalSandwichedLocalNodeIds.clear();
  refreshNodeDecorations();
}

/**
 * Return a websocket event detail payload.
 * @param {CustomEvent | object} event
 * @returns {Record<string, any>}
 */
function eventDetail(event) {
  return event?.detail ?? event ?? {};
}

/**
 * Build a prompt id for queue requests.
 * @returns {string}
 */
function createPromptId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `modal-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/**
 * Return the current monotonic clock value for UI state ordering.
 * @returns {number}
 */
function nowMs() {
  return Date.now();
}

/**
 * Return the event clock used for progress-rate samples, preserving backend time during replay.
 * @returns {number}
 */
function progressEventTimestampMs() {
  return modalReplayedEventUpdatedAtMs ?? nowMs();
}

/**
 * Derive a smoothed iterations-per-second rate from two numeric progress samples.
 * @param {{ value?: number, max?: number, updatedAt?: number, iterationRate?: number | null } | null | undefined} previousState
 * @param {number} value
 * @param {number} maxValue
 * @param {number} updatedAt
 * @returns {number | null}
 */
function progressIterationRate(previousState, value, maxValue, updatedAt) {
  if (!previousState) {
    return null;
  }
  const previousValue = Number(previousState.value);
  const previousMax = Number(previousState.max);
  const previousUpdatedAt = Number(previousState.updatedAt);
  const previousRate =
    previousState.iterationRate == null ? Number.NaN : Number(previousState.iterationRate);
  if (
    !Number.isFinite(previousValue) ||
    !Number.isFinite(previousMax) ||
    !Number.isFinite(previousUpdatedAt) ||
    previousMax !== maxValue ||
    value < previousValue
  ) {
    return null;
  }
  if (value === previousValue || updatedAt <= previousUpdatedAt) {
    return Number.isFinite(previousRate) && previousRate >= 0 ? previousRate : null;
  }
  const elapsedSeconds = (updatedAt - previousUpdatedAt) / 1000;
  const sampleRate = (value - previousValue) / elapsedSeconds;
  if (!Number.isFinite(sampleRate) || sampleRate < 0) {
    return null;
  }
  if (!Number.isFinite(previousRate) || previousRate < 0) {
    return sampleRate;
  }
  return (
    previousRate * (1 - ITERATION_RATE_SMOOTHING_FACTOR) +
    sampleRate * ITERATION_RATE_SMOOTHING_FACTOR
  );
}

/**
 * Format a progress rate for the compact label over a node progress bar.
 * @param {number | null | undefined} iterationRate
 * @param {string | null | undefined} unit
 * @returns {string}
 */
function formatIterationRate(iterationRate, unit = null) {
  const safeRate = iterationRate == null ? Number.NaN : Number(iterationRate);
  const rateUnit = unit === "tokens" ? "tok/s" : "it/s";
  if (!Number.isFinite(safeRate) || safeRate < 0) {
    return `— ${rateUnit}`;
  }
  const fractionDigits = safeRate < 10 ? 2 : safeRate < 100 ? 1 : 0;
  return `${safeRate.toFixed(fractionDigits)} ${rateUnit}`;
}

/**
 * Trim one canvas label to fit a fixed width.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} text
 * @param {number} maxWidth
 * @returns {string}
 */
function fitCanvasText(ctx, text, maxWidth) {
  const safeText = String(text ?? "");
  if (!safeText || ctx.measureText(safeText).width <= maxWidth) {
    return safeText;
  }
  let end = safeText.length;
  while (end > 1 && ctx.measureText(`${safeText.slice(0, end)}…`).width > maxWidth) {
    end -= 1;
  }
  return `${safeText.slice(0, end)}…`;
}

/**
 * Return a lazily loaded provider icon for the canvas location line.
 * @param {string | null | undefined} provider
 * @returns {HTMLImageElement | null}
 */
function remoteLocationIcon(provider) {
  const safeProvider = String(provider ?? "");
  const source = REMOTE_LOCATION_ICON_SOURCES[safeProvider];
  if (!source || typeof Image !== "function") {
    return null;
  }
  if (!remoteLocationIconImages.has(safeProvider)) {
    const icon = new Image();
    icon.decoding = "async";
    icon.src = source;
    icon.addEventListener?.("load", refreshNodeDecorations, { once: true });
    remoteLocationIconImages.set(safeProvider, icon);
  }
  return remoteLocationIconImages.get(safeProvider) ?? null;
}

/**
 * Draw an iteration-rate label over a progress bar with a tight opaque backing.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number | null | undefined} iterationRate
 * @param {number} rightX
 * @param {number} centerY
 * @param {number} barHeight
 * @param {number} scale
 * @param {string | null | undefined} unit
 */
function drawIterationRateOverlay(
  ctx,
  iterationRate,
  rightX,
  centerY,
  barHeight,
  scale,
  unit = null,
) {
  const label = formatIterationRate(iterationRate, unit);
  const metrics = ctx.measureText(label);
  const paddingX = 3 / scale;
  const paddingY = 1 / scale;
  const measuredHeight =
    Number(metrics.actualBoundingBoxAscent ?? 0) +
    Number(metrics.actualBoundingBoxDescent ?? 0);
  const boxWidth = metrics.width + paddingX * 2;
  const boxHeight = Math.min(barHeight, Math.max(7 / scale, measuredHeight) + paddingY * 2);
  const boxX = rightX - boxWidth;
  const boxY = centerY - boxHeight / 2;

  ctx.fillStyle = "rgba(0, 0, 0, 0.9)";
  ctx.beginPath();
  ctx.roundRect(boxX, boxY, boxWidth, boxHeight, Math.min(2 / scale, boxHeight / 2));
  ctx.fill();
  ctx.fillStyle = "#e2e8f0";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(label, rightX - paddingX, centerY);
}

/**
 * Remove terminal prompt markers after their stale-event guard window expires.
 */
function pruneTerminalPromptStates() {
  const cutoffMs = nowMs() - TERMINAL_PROMPT_RETENTION_MS;
  for (const [promptId, terminalState] of modalTerminalPromptStates.entries()) {
    if ((terminalState?.terminalAt ?? 0) < cutoffMs) {
      modalTerminalPromptStates.delete(promptId);
    }
  }
}

/**
 * Return whether late events for a prompt should be ignored.
 * @param {string} promptId
 * @returns {boolean}
 */
function isPromptTerminal(promptId) {
  if (!promptId) {
    return false;
  }
  pruneTerminalPromptStates();
  return modalTerminalPromptStates.has(promptId);
}

/**
 * Mark a prompt as terminal so late remote events cannot resurrect stale UI state.
 * @param {string} promptId
 * @param {string} phase
 */
function markPromptTerminal(promptId, phase) {
  if (!promptId) {
    return;
  }
  modalCancellingPromptIds.delete(promptId);
  modalTerminalPromptStates.set(promptId, {
    phase,
    terminalAt: nowMs(),
  });
}

/**
 * Allow a newly queued prompt id to receive state updates.
 * @param {string} promptId
 */
function clearPromptTerminal(promptId) {
  if (!promptId) {
    return;
  }
  modalTerminalPromptStates.delete(promptId);
  modalCancellingPromptIds.delete(promptId);
  clearPromptQueued(promptId);
}

/**
 * Return whether one prompt has a user-requested cancellation in progress.
 * @param {string} promptId
 * @returns {boolean}
 */
function isPromptCancelling(promptId) {
  return Boolean(promptId) && modalCancellingPromptIds.has(promptId);
}

/**
 * Read a node id as a stable string.
 * @param {LGraphNode} node
 * @returns {string}
 */
function nodeId(node) {
  return String(node?.id ?? "");
}

/**
 * Ensure the global Modal execution status badge exists.
 * @returns {HTMLDivElement | null}
 */
function ensureGlobalStatusElement() {
  if (typeof document === "undefined") {
    return null;
  }
  if (modalGlobalStatusElement?.isConnected) {
    return modalGlobalStatusElement;
  }

  const element = document.createElement("div");
  element.id = "comfy-modal-global-status";
  element.style.position = "fixed";
  element.style.top = "14px";
  element.style.right = "18px";
  element.style.zIndex = "9999";
  element.style.display = "none";
  element.style.alignItems = "flex-start";
  element.style.gap = "10px";
  element.style.padding = "10px 14px";
  element.style.borderRadius = "16px";
  element.style.border = "1px solid rgba(255, 255, 255, 0.16)";
  element.style.background = "rgba(15, 23, 42, 0.94)";
  element.style.boxShadow = "0 10px 30px rgba(0, 0, 0, 0.28)";
  element.style.color = "#f8fafc";
  element.style.fontFamily = "ui-sans-serif, system-ui, sans-serif";
  element.style.fontSize = "13px";
  element.style.fontWeight = "600";
  element.style.pointerEvents = "none";
  element.innerHTML =
    '<span class="modal-status-dot"></span><span class="modal-status-copy"><span class="modal-status-text"></span><span class="modal-status-gpu" hidden></span><span class="modal-status-cost" hidden></span><span class="modal-status-billing" hidden></span><span class="modal-status-containers" hidden></span></span>';
  document.body.appendChild(element);
  modalGlobalStatusElement = element;
  return element;
}

/**
 * Remove orphaned global status entries that no longer have any live prompt state.
 */
function pruneGlobalStatusStates() {
  pruneTerminalPromptStates();
  for (const promptId of Array.from(modalGlobalStatusStates.keys())) {
    if (isPromptTerminal(promptId)) {
      modalGlobalStatusStates.delete(promptId);
      continue;
    }
    if (modalPromptStates.has(promptId) || syntheticPromptUiStates.has(promptId)) {
      continue;
    }
    modalGlobalStatusStates.delete(promptId);
  }
}

/**
 * Return all current node states for a prompt id.
 * @param {string} promptId
 * @returns {Array<{ phase: string, promptId: string }>}
 */
function promptNodeStates(promptId) {
  return Array.from(modalNodeStates.values()).filter((state) => state?.promptId === promptId);
}

/**
 * Return the best known remote node count for one prompt.
 * @param {string} promptId
 * @param {number} fallbackCount
 * @returns {number}
 */
function promptRemoteNodeCount(promptId, fallbackCount = 1) {
  const promptState = modalPromptStates.get(promptId);
  const remoteCount = promptState?.remoteNodeIds?.length ?? 0;
  return Math.max(1, Number(remoteCount || fallbackCount) || 1);
}

/**
 * Derive the effective global phase for one prompt from its live node state.
 * @param {string} promptId
 * @param {string} phase
 * @returns {string}
 */
function effectiveGlobalStatusPhase(promptId, phase) {
  const promptState = modalPromptStates.get(promptId);
  const nodeStates = promptNodeStates(promptId);

  if (phase === STATE_ERROR) {
    return STATE_ERROR;
  }
  if (phase === STATE_CANCELLING || nodeStates.some((state) => state.phase === STATE_CANCELLING)) {
    return STATE_CANCELLING;
  }
  if (phase === STATE_SETUP || phase === STATE_STARTING || phase === STATE_WAITING) {
    return phase;
  }
  if (nodeStates.some((state) => state.phase === STATE_ERROR)) {
    return STATE_ERROR;
  }
  if (phase === STATE_FINALIZING) {
    if (
      promptState?.hasRemoteExecutionStarted ||
      promptActiveNodeIsLive(promptId) ||
      nodeStates.some((state) => state.phase === STATE_ACTIVE || state.phase === STATE_READY)
    ) {
      return EXECUTION_PHASE;
    }
    return STATE_FINALIZING;
  }
  if (promptState?.hasRemoteExecutionStarted && promptActiveNodeIsLive(promptId)) {
    return EXECUTION_PHASE;
  }
  if (promptState?.hasRemoteExecutionStarted && nodeStates.some((state) => state.phase === STATE_ACTIVE)) {
    return EXECUTION_PHASE;
  }
  if (
    promptState?.hasRemoteExecutionStarted &&
    nodeStates.some((state) => state.phase === STATE_READY)
  ) {
    return EXECUTION_PHASE;
  }
  return phase;
}

/**
 * Return whether one prompt still has active remote work that should keep the global pill visible.
 * @param {string} promptId
 * @returns {boolean}
 */
function promptHasLiveRemoteWork(promptId) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    return false;
  }

  if (promptActiveNodeIsLive(promptId)) {
    return true;
  }

  if (promptState.remoteNodeIds.some((nodeIdValue) => hasLiveNodeProgress(nodeIdValue, promptId))) {
    return true;
  }

  return promptState.remoteNodeIds.some((nodeIdValue) => {
    const nodeState = modalNodeStates.get(String(nodeIdValue));
    return (
      nodeState?.promptId === promptId &&
      [STATE_SETUP, STATE_STARTING, STATE_READY, STATE_ACTIVE].includes(nodeState.phase)
    );
  });
}

/**
 * Return whether the prompt-wide active node still represents live execution.
 * @param {string} promptId
 * @returns {boolean}
 */
function promptActiveNodeIsLive(promptId) {
  const promptState = modalPromptStates.get(promptId);
  const activeNodeId = promptState?.activeNodeId;
  if (!activeNodeId) {
    return false;
  }
  if (hasLiveNodeProgress(activeNodeId, promptId)) {
    return true;
  }
  const activeNodeState = modalNodeStates.get(String(activeNodeId));
  return activeNodeState?.promptId === promptId && activeNodeState.phase === STATE_ACTIVE;
}

/**
 * Return whether any older Modal prompt is still visually or remotely active.
 * @param {string | null} excludedPromptId
 * @returns {boolean}
 */
function hasActiveModalPrompt(excludedPromptId = null) {
  for (const promptId of modalPromptStates.keys()) {
    if (promptId === excludedPromptId || isPromptTerminal(promptId)) {
      continue;
    }
    if (syntheticPromptUiStates.has(promptId) || promptHasLiveRemoteWork(promptId)) {
      return true;
    }
  }
  return false;
}

/**
 * Mark a submitted prompt as queued behind an already active Modal prompt.
 * @param {string} promptId
 * @returns {boolean}
 */
function markPromptQueuedBehindActiveModal(promptId) {
  if (!promptId || !hasActiveModalPrompt(promptId)) {
    return false;
  }
  modalQueuedPromptIds.add(promptId);
  return true;
}

/**
 * Return whether queue-time UI updates should be suppressed for one prompt.
 * @param {string} promptId
 * @returns {boolean}
 */
function isPromptQueuedBehindActiveModal(promptId) {
  return modalQueuedPromptIds.has(promptId);
}

/**
 * Allow normal execution UI updates for a prompt once ComfyUI actually starts it.
 * @param {string} promptId
 */
function clearPromptQueued(promptId) {
  modalQueuedPromptIds.delete(promptId);
}

/**
 * Drop one prompt's stale global-status entry once it no longer has live remote work.
 * @param {string} promptId
 */
function reconcilePromptGlobalStatus(promptId) {
  if (!promptId) {
    return;
  }
  const globalStatusState = modalGlobalStatusStates.get(promptId);
  if (!globalStatusState) {
    return;
  }
  if (syntheticPromptUiStates.has(promptId)) {
    return;
  }
  if (globalStatusState.phase === STATE_ERROR || globalStatusState.phase === STATE_SETUP) {
    return;
  }
  if (promptHasLiveRemoteWork(promptId)) {
    refreshGlobalStatusElement();
    return;
  }
  modalGlobalStatusStates.delete(promptId);
  refreshGlobalStatusElement();
}

/**
 * Return the most important active global Modal state.
 * @returns {{ phase: string, promptId: string, nodeCount: number, modalGpu: string | null } | null}
 */
function currentGlobalStatus() {
  pruneGlobalStatusStates();
  if (modalGlobalStatusStates.size === 0) {
    return null;
  }

  const phases = Array.from(modalGlobalStatusStates.entries()).map(([promptId, state]) => ({
    promptId,
    phase: effectiveGlobalStatusPhase(promptId, state.phase),
    nodeCount: state.nodeCount,
    batchValue: state.batchValue ?? null,
    batchMax: state.batchMax ?? null,
    statusMessage: state.statusMessage ?? null,
    statusCurrent: state.statusCurrent ?? null,
    statusTotal: state.statusTotal ?? null,
    modalGpu: state.modalGpu ?? null,
    executionLabel: state.executionLabel ?? null,
    updatedAt: state.updatedAt,
  }));
  phases.sort((left, right) => right.updatedAt - left.updatedAt);

  return (
    phases.find((state) => state.phase === STATE_ERROR) ??
    phases.find((state) => state.phase === STATE_CANCELLING) ??
    phases.find((state) => state.phase === STATE_SETUP) ??
    phases.find((state) => state.phase === STATE_STARTING) ??
    phases.find((state) => state.phase === STATE_WAITING) ??
    phases.find((state) => state.phase === EXECUTION_PHASE) ??
    phases.find((state) => state.phase === STATE_FINALIZING) ??
    phases[0]
  );
}

/**
 * Return whether active workflow state should poll Modal for container status.
 * @param {{ phase?: string, modalGpu?: string | null } | null} activeState
 * @returns {boolean}
 */
function shouldPollModalContainerStatus(activeState) {
  return Boolean(
    activeState && activeState.phase !== STATE_ERROR && activeState.modalGpu,
  );
}

/**
 * Return a provider-aware message while planned remote capacity starts.
 * @param {Iterable<string>} providers
 * @returns {string}
 */
function remoteCapacityWaitingMessage(providers) {
  const providerSet = new Set(Array.from(providers, (provider) => String(provider)));
  if (providerSet.size !== 1) {
    return "Waiting for remote capacity";
  }
  if (providerSet.has("modal")) {
    return "Waiting for Modal startup";
  }
  if (providerSet.has("vast")) {
    return "Waiting for Vast.ai instance";
  }
  if (providerSet.has("ssh_docker")) {
    return "Waiting for self-hosted worker";
  }
  return "Waiting for remote capacity";
}

/**
 * Return one compact provider and environment label for the global status pill.
 * @param {Record<string, any>} assignment
 * @param {Map<string, Record<string, any>>} configurationsById
 * @param {string | null} fallbackModalGpu
 * @returns {string}
 */
function remoteExecutionEnvironmentLabel(
  assignment,
  configurationsById,
  fallbackModalGpu,
) {
  if (assignment?.provider === "modal") {
    const configuration = configurationsById.get(
      String(assignment?.configuration_id ?? ""),
    );
    const modalGpu = configuration?.gpu_type ?? fallbackModalGpu;
    return modalGpu ? `Modal ${modalGpu}` : "Modal";
  }
  if (assignment?.provider === "vast") {
    const destination = String(
      assignment?.execution_location ?? assignment?.environment_id ?? "capacity",
    );
    return `Vast.ai ${destination}`;
  }
  return String(
    assignment?.execution_location ??
      assignment?.environment_id ??
      "self-hosted",
  );
}

/**
 * Normalize one container-list payload for stable rendering and comparison.
 * @param {any} payload
 * @returns {Array<Record<string, any>>}
 */
function normalizedModalContainerStatuses(payload) {
  const containers = Array.isArray(payload?.containers) ? payload.containers : [];
  return containers
    .filter((container) => container?.container_id)
    .map((container) => ({
      containerId: String(container.container_id),
      appName: String(container.app_name ?? ""),
      modalGpu: String(container.modal_gpu ?? ""),
      estimatedGpuCostPerSecond:
        Math.max(0, Number(container.estimated_gpu_cost_per_second)) || 0,
      state: container.state === "running" ? "running" : "starting",
      enqueuedAt: Number(container.enqueued_at) || null,
      startedAt: Number(container.started_at) || null,
    }))
    .sort((left, right) =>
      (left.enqueuedAt ?? left.startedAt ?? 0) - (right.enqueuedAt ?? right.startedAt ?? 0) ||
      left.containerId.localeCompare(right.containerId),
    );
}

/**
 * Normalize one completed Modal hourly billing record.
 * @param {any} payload
 * @returns {Record<string, any> | null}
 */
function normalizedModalHourlyBillingStatus(payload) {
  const billing = payload?.billing;
  if (!billing?.app_name || !billing?.interval_end) {
    return null;
  }
  return {
    appName: String(billing.app_name),
    environmentName: String(billing.environment_name ?? ""),
    modalGpu: String(billing.modal_gpu ?? ""),
    intervalStart: String(billing.interval_start ?? ""),
    intervalEnd: String(billing.interval_end),
    appCostUsdBeforeCredits:
      Math.max(0, Number(billing.app_cost_usd_before_credits)) || 0,
    hasUsage: Boolean(billing.has_usage),
    fetchedAt: String(billing.fetched_at ?? ""),
    nextRefreshAt: String(billing.next_refresh_at ?? ""),
  };
}

/**
 * Return a stable signature for one container status response.
 * @param {Array<Record<string, any>>} containers
 * @returns {string}
 */
function modalContainerStatusSignature(containers) {
  return containers
    .map(
      (container) =>
        `${container.containerId}:${container.state}:${container.startedAt ?? ""}:${container.estimatedGpuCostPerSecond}`,
    )
    .join("|");
}

/**
 * Return containers whose billed GPU identity matches the active prompt.
 * @param {string | null} promptId
 * @param {Array<Record<string, any>>} containers
 * @returns {Array<Record<string, any>>}
 */
function modalPromptCostContainers(promptId, containers) {
  const selectedModalGpu = modalPromptStates.get(String(promptId ?? ""))?.modalGpu;
  if (!selectedModalGpu) {
    return containers;
  }
  return containers.filter((container) => container.modalGpu === selectedModalGpu);
}

/**
 * Return the estimated GPU burn rate for active running containers.
 * @param {Array<Record<string, any>>} containers
 * @param {string | null} promptId
 * @returns {number}
 */
function modalContainerGpuBurnRate(
  containers = modalContainerStatuses,
  promptId = modalContainerStatusPromptId,
) {
  return modalPromptCostContainers(promptId, containers)
    .filter((container) => container.state === "running" && container.startedAt)
    .reduce(
      (total, container) => total + Math.max(0, container.estimatedGpuCostPerSecond),
      0,
    );
}

/**
 * Add one successful status interval to the prompt-scoped GPU cost estimate.
 * @param {string} promptId
 * @param {Array<Record<string, any>>} nextStatuses
 * @param {number} polledAtSeconds
 */
function updateModalContainerCostEstimate(promptId, nextStatuses, polledAtSeconds) {
  if (!promptId || !Number.isFinite(polledAtSeconds)) {
    return;
  }
  const promptStartedAtSeconds = Math.max(
    0,
    Number(modalPromptStates.get(promptId)?.startedAt ?? nowMs()) / 1000,
  );
  const promptStatuses = modalPromptCostContainers(promptId, nextStatuses);
  const previousPromptStatuses = modalPromptCostContainers(
    promptId,
    modalContainerStatuses,
  );
  let intervalCostUsd = 0;

  if (modalContainerCostUpdatedAtSeconds == null) {
    for (const container of promptStatuses) {
      if (container.state !== "running" || !container.startedAt) {
        continue;
      }
      const intervalStartedAt = Math.max(promptStartedAtSeconds, container.startedAt);
      intervalCostUsd +=
        Math.max(0, polledAtSeconds - intervalStartedAt) *
        container.estimatedGpuCostPerSecond;
    }
  } else {
    const previousIntervalSeconds = Math.max(
      0,
      polledAtSeconds - modalContainerCostUpdatedAtSeconds,
    );
    intervalCostUsd +=
      previousIntervalSeconds *
      modalContainerGpuBurnRate(previousPromptStatuses, promptId);

    const previousRunningContainerIds = new Set(
      previousPromptStatuses
        .filter((container) => container.state === "running" && container.startedAt)
        .map((container) => container.containerId),
    );
    for (const container of promptStatuses) {
      if (
        container.state !== "running" ||
        !container.startedAt ||
        previousRunningContainerIds.has(container.containerId)
      ) {
        continue;
      }
      const intervalStartedAt = Math.max(
        promptStartedAtSeconds,
        modalContainerCostUpdatedAtSeconds,
        container.startedAt,
      );
      intervalCostUsd +=
        Math.max(0, polledAtSeconds - intervalStartedAt) *
        container.estimatedGpuCostPerSecond;
    }
  }

  modalContainerEstimatedCostUsd += Math.max(0, intervalCostUsd);
  modalContainerCostUpdatedAtSeconds = polledAtSeconds;
}

/**
 * Return the accumulated estimate plus the active interval since the last poll.
 * @returns {number}
 */
function liveModalContainerCostEstimate() {
  if (modalContainerCostUpdatedAtSeconds == null) {
    return modalContainerEstimatedCostUsd;
  }
  const unpolledSeconds = Math.max(0, nowMs() / 1000 - modalContainerCostUpdatedAtSeconds);
  return modalContainerEstimatedCostUsd +
    unpolledSeconds * modalContainerGpuBurnRate(modalContainerStatuses);
}

/**
 * Return the next adaptive container status poll delay.
 * @returns {number}
 */
function modalContainerStatusPollDelay() {
  if (modalContainerStatusFailureCount > 0) {
    return Math.min(
      CONTAINER_STATUS_MAX_BACKOFF_MS,
      CONTAINER_STATUS_STABLE_POLL_MS * 2 ** (modalContainerStatusFailureCount - 1),
    );
  }
  if (typeof document !== "undefined" && document.visibilityState !== "visible") {
    return CONTAINER_STATUS_HIDDEN_POLL_MS;
  }
  return modalContainerStatusUnchangedPolls >= 2
    ? CONTAINER_STATUS_STABLE_POLL_MS
    : CONTAINER_STATUS_FAST_POLL_MS;
}

/**
 * Clear scheduled polling and prompt-scoped container UI state.
 */
function stopModalContainerStatusPolling() {
  if (modalContainerStatusTimer != null) {
    clearTimeout(modalContainerStatusTimer);
    modalContainerStatusTimer = null;
  }
  modalContainerStatuses = [];
  modalContainerStatusPromptId = null;
  modalContainerStatusLoaded = false;
  modalContainerStatusError = null;
  modalContainerStatusUnchangedPolls = 0;
  modalContainerStatusFailureCount = 0;
  modalContainerEstimatedCostUsd = 0;
  modalContainerCostUpdatedAtSeconds = null;
  modalHourlyBillingStatus = null;
  modalHourlyBillingError = null;
}

/**
 * Schedule the next container status query while Modal work remains active.
 * @param {number | null} requestedDelayMs
 */
function scheduleModalContainerStatusPoll(requestedDelayMs = null) {
  const activeState = currentGlobalStatus();
  if (!shouldPollModalContainerStatus(activeState)) {
    stopModalContainerStatusPolling();
    return;
  }
  if (modalContainerStatusPollInFlight || modalContainerStatusTimer != null) {
    return;
  }
  const delayMs = requestedDelayMs ?? modalContainerStatusPollDelay();
  modalContainerStatusTimer = setTimeout(() => {
    modalContainerStatusTimer = null;
    pollModalContainerStatus();
  }, Math.max(0, delayMs));
}

/**
 * Poll active containers through the local ComfyUI route.
 * @returns {Promise<void>}
 */
async function pollModalContainerStatus() {
  const activeState = currentGlobalStatus();
  if (!shouldPollModalContainerStatus(activeState) || typeof api.fetchApi !== "function") {
    stopModalContainerStatusPolling();
    return;
  }

  const requestedPromptId = String(activeState.promptId ?? "");
  const requestedModalGpu = String(activeState.modalGpu ?? DEFAULT_MODAL_GPU);
  modalContainerStatusPollInFlight = true;
  try {
    const statusUrl =
      `${MODAL_CONTAINER_STATUS_ROUTE}?modal_gpu=${encodeURIComponent(requestedModalGpu)}`;
    const response = await api.fetchApi(statusUrl, { method: "GET" });
    if (response.status !== 200) {
      throw new Error(`Modal container status returned HTTP ${response.status}.`);
    }
    const payload = await response.json();
    if (modalContainerStatusPromptId !== requestedPromptId) {
      return;
    }
    const nextStatuses = normalizedModalContainerStatuses(payload);
    modalHourlyBillingStatus = normalizedModalHourlyBillingStatus(payload);
    modalHourlyBillingError = payload?.billing_error
      ? String(payload.billing_error)
      : null;
    const polledAtSeconds = Number(payload?.polled_at) || nowMs() / 1000;
    updateModalContainerCostEstimate(requestedPromptId, nextStatuses, polledAtSeconds);
    const changed =
      modalContainerStatusSignature(nextStatuses) !==
      modalContainerStatusSignature(modalContainerStatuses);
    modalContainerStatuses = nextStatuses;
    modalContainerStatusLoaded = true;
    modalContainerStatusError = null;
    modalContainerStatusFailureCount = 0;
    modalContainerStatusUnchangedPolls = changed
      ? 0
      : modalContainerStatusUnchangedPolls + 1;
  } catch (error) {
    modalContainerStatusLoaded = true;
    modalContainerStatusError = String(error?.message ?? error);
    modalHourlyBillingError = modalContainerStatusError;
    modalContainerStatusFailureCount += 1;
    console.debug("Unable to poll Modal container status.", error);
  } finally {
    modalContainerStatusPollInFlight = false;
    refreshGlobalStatusElement();
    scheduleModalContainerStatusPoll();
  }
}

/**
 * Ensure polling follows the prompt currently shown by the global status pill.
 * @param {{ promptId?: string, phase?: string } | null} activeState
 */
function syncModalContainerStatusPolling(activeState) {
  if (!shouldPollModalContainerStatus(activeState)) {
    stopModalContainerStatusPolling();
    return;
  }
  if (modalContainerStatusPromptId !== activeState.promptId) {
    stopModalContainerStatusPolling();
    modalContainerStatusPromptId = activeState.promptId;
    scheduleModalContainerStatusPoll(0);
    return;
  }
  scheduleModalContainerStatusPoll();
}

/**
 * Force an immediate status refresh after the browser returns to the foreground.
 */
function requestImmediateModalContainerStatusPoll() {
  if (modalContainerStatusTimer != null) {
    clearTimeout(modalContainerStatusTimer);
    modalContainerStatusTimer = null;
  }
  scheduleModalContainerStatusPoll(0);
}

/**
 * Format a compact elapsed time for one running container.
 * @param {number | null} startedAtSeconds
 * @returns {string}
 */
function formatModalContainerAge(startedAtSeconds) {
  if (!startedAtSeconds) {
    return "";
  }
  const elapsedSeconds = Math.max(0, Math.floor(nowMs() / 1000 - startedAtSeconds));
  if (elapsedSeconds < 60) {
    return `${elapsedSeconds}s`;
  }
  const minutes = Math.floor(elapsedSeconds / 60);
  const seconds = elapsedSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

/**
 * Format a compact estimated US-dollar amount.
 * @param {number} amountUsd
 * @returns {string}
 */
function formatEstimatedModalUsd(amountUsd) {
  const safeAmount = Math.max(0, Number(amountUsd) || 0);
  if (safeAmount < 0.01) {
    return `$${safeAmount.toFixed(4)}`;
  }
  if (safeAmount < 1) {
    return `$${safeAmount.toFixed(3)}`;
  }
  return `$${safeAmount.toFixed(2)}`;
}

/**
 * Render the prompt-scoped estimated GPU cost and current burn rate.
 * @param {HTMLElement | null} costElement
 */
function renderModalCostEstimate(costElement) {
  if (!costElement) {
    return;
  }
  const estimatedCostUsd = liveModalContainerCostEstimate();
  const burnRatePerMinuteUsd = modalContainerGpuBurnRate() * 60;
  if (estimatedCostUsd <= 0 && burnRatePerMinuteUsd <= 0) {
    costElement.textContent = "";
    costElement.hidden = true;
    return;
  }
  costElement.textContent =
    `Estimated GPU cost ${formatEstimatedModalUsd(estimatedCostUsd)}` +
    (burnRatePerMinuteUsd > 0
      ? ` · ${formatEstimatedModalUsd(burnRatePerMinuteUsd)}/min`
      : "");
  costElement.hidden = false;
}

/**
 * Format the end of one completed hourly Modal billing interval.
 * @param {string} intervalEnd
 * @returns {string}
 */
function formatModalBillingIntervalEnd(intervalEnd) {
  const intervalEndDate = new Date(intervalEnd);
  if (!Number.isFinite(intervalEndDate.getTime())) {
    return "unknown time";
  }
  return intervalEndDate.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

/**
 * Render actual app billing for Modal's latest buffered completed hour.
 * @param {HTMLElement | null} billingElement
 * @param {{ modalGpu?: string | null } | null} activeState
 */
function renderModalHourlyBilling(billingElement, activeState) {
  if (!billingElement) {
    return;
  }
  if (modalHourlyBillingError) {
    billingElement.textContent = "Hourly billing temporarily unavailable";
    billingElement.title = modalHourlyBillingError;
    billingElement.hidden = false;
    return;
  }
  if (
    !modalHourlyBillingStatus ||
    modalHourlyBillingStatus.modalGpu !== String(activeState?.modalGpu ?? DEFAULT_MODAL_GPU)
  ) {
    billingElement.textContent = "Checking hourly Modal billing…";
    billingElement.title = "Modal billing reports use completed hourly intervals.";
    billingElement.hidden = false;
    return;
  }

  const reportedCost = formatEstimatedModalUsd(
    modalHourlyBillingStatus.appCostUsdBeforeCredits,
  );
  const intervalEnd = formatModalBillingIntervalEnd(
    modalHourlyBillingStatus.intervalEnd,
  );
  billingElement.textContent =
    `Reported app cost ${reportedCost} · hour ending ${intervalEnd}`;
  billingElement.title =
    `${modalHourlyBillingStatus.appName} in ${modalHourlyBillingStatus.environmentName}; ` +
    "actual Modal metered cost before credits and reservations. " +
    "Hourly reports exclude the partial current hour and use a 10-minute collection buffer.";
  billingElement.hidden = false;
}

/**
 * Render all active Modal containers beneath the GPU line.
 * @param {HTMLElement | null} containerElement
 */
function renderModalContainerStatuses(containerElement) {
  if (!containerElement) {
    return;
  }
  containerElement.replaceChildren();
  containerElement.hidden = false;

  if (modalContainerStatusError) {
    const line = document.createElement("span");
    line.textContent = "Container status temporarily unavailable";
    line.className = "modal-status-container modal-status-container-error";
    containerElement.appendChild(line);
    return;
  }
  if (!modalContainerStatusLoaded) {
    const line = document.createElement("span");
    line.textContent = "Checking Modal containers…";
    line.className = "modal-status-container";
    containerElement.appendChild(line);
    return;
  }
  if (modalContainerStatuses.length === 0) {
    const line = document.createElement("span");
    line.textContent = "Waiting for container assignment";
    line.className = "modal-status-container";
    containerElement.appendChild(line);
    return;
  }

  modalContainerStatuses.forEach((container, index) => {
    const line = document.createElement("span");
    const shortId = container.containerId.slice(-8);
    const age = formatModalContainerAge(container.startedAt);
    const state = container.state === "running" ? "Running" : "Starting";
    line.textContent = `Container ${index + 1} · ${state}${age ? ` ${age}` : ""} · ${shortId}`;
    line.className = `modal-status-container modal-status-container-${container.state}`;
    containerElement.appendChild(line);
  });
}

/**
 * Redraw the global Modal execution badge.
 */
function refreshGlobalStatusElement() {
  const element = ensureGlobalStatusElement();
  if (!element) {
    return;
  }

  const activeState = currentGlobalStatus();
  if (!activeState) {
    element.style.display = "none";
    element.dataset.phase = "";
    element.dataset.modalGpu = "";
    stopModalContainerStatusPolling();
    return;
  }

  const promptState = modalPromptStates.get(String(activeState.promptId ?? ""));
  const configuratorPanel = remoteConfiguratorPanel(promptState?.configuratorNodeId);
  if (configuratorPanel) {
    element.style.display = "none";
    element.dataset.phase = activeState.phase;
    element.dataset.modalGpu = activeState.modalGpu ?? "";
    stopModalContainerStatusPolling();
    renderRemoteConfiguratorStatus(configuratorPanel, activeState);
    return;
  }

  const dot = element.querySelector(".modal-status-dot");
  const text = element.querySelector(".modal-status-text");
  const gpuText = element.querySelector(".modal-status-gpu");
  const costText = element.querySelector(".modal-status-cost");
  const billingText = element.querySelector(".modal-status-billing");
  const containerText = element.querySelector(".modal-status-containers");
  const nodeLabel = activeState.nodeCount === 1 ? "node" : "nodes";
  const hasBatchProgress =
    activeState.phase === EXECUTION_PHASE &&
    Number(activeState.batchMax ?? 0) > 1;
  const batchValue = hasBatchProgress
    ? Math.max(0, Math.min(Number(activeState.batchMax), Number(activeState.batchValue ?? 0)))
    : 0;
  const batchMax = hasBatchProgress ? Math.max(1, Number(activeState.batchMax)) : 1;
  const batchRatio = hasBatchProgress ? batchValue / batchMax : 0;
  const hasStatusProgress = Number(activeState.statusTotal ?? 0) > 1;
  const statusValue = hasStatusProgress
    ? Math.max(
        0,
        Math.min(Number(activeState.statusTotal), Number(activeState.statusCurrent ?? 0)),
      )
    : 0;
  const statusMax = hasStatusProgress ? Math.max(1, Number(activeState.statusTotal)) : 1;
  const statusRatio = hasStatusProgress ? statusValue / statusMax : 0;

  element.style.display = "inline-flex";
  element.dataset.phase = activeState.phase;
  element.dataset.modalGpu = activeState.modalGpu ?? "";
  gpuText.textContent = activeState.modalGpu ?? "";
  gpuText.hidden = !activeState.modalGpu;
  if (activeState.modalGpu) {
    syncModalContainerStatusPolling(activeState);
    renderModalCostEstimate(costText);
    renderModalHourlyBilling(billingText, activeState);
    renderModalContainerStatuses(containerText);
  } else {
    stopModalContainerStatusPolling();
    for (const modalOnlyElement of [costText, billingText, containerText]) {
      if (modalOnlyElement) {
        modalOnlyElement.textContent = "";
        modalOnlyElement.hidden = true;
      }
    }
  }

  if (activeState.phase === STATE_SETUP) {
    element.style.borderColor = "rgba(245, 158, 11, 0.55)";
    element.style.background = hasStatusProgress
      ? `linear-gradient(90deg, rgba(180, 83, 9, 0.94) 0%, rgba(180, 83, 9, 0.94) ${(
          statusRatio * 100
        ).toFixed(2)}%, rgba(61, 42, 9, 0.94) ${(statusRatio * 100).toFixed(2)}%, rgba(61, 42, 9, 0.94) 100%)`
      : "rgba(61, 42, 9, 0.94)";
    dot.style.background = SETUP_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(245, 158, 11, 0.18)";
    dot.style.animation = "modal-status-pulse 1.1s ease-in-out infinite";
    text.textContent = activeState.statusMessage ?? "Preparing remote workflow";
  } else if (activeState.phase === STATE_STARTING) {
    element.style.borderColor = "rgba(234, 179, 8, 0.58)";
    element.style.background = "rgba(54, 45, 6, 0.94)";
    dot.style.background = STARTING_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(234, 179, 8, 0.2)";
    dot.style.animation = "modal-status-pulse 0.85s ease-in-out infinite";
    text.textContent = activeState.statusMessage ?? "Starting remote component";
  } else if (activeState.phase === STATE_WAITING) {
    element.style.borderColor = "rgba(245, 158, 11, 0.55)";
    element.style.background = "rgba(61, 42, 9, 0.94)";
    dot.style.background = SETUP_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(245, 158, 11, 0.18)";
    dot.style.animation = "modal-status-pulse 1.1s ease-in-out infinite";
    text.textContent = activeState.statusMessage ?? "Waiting for remote startup";
  } else if (activeState.phase === EXECUTION_PHASE) {
    element.style.borderColor = "rgba(34, 197, 94, 0.55)";
    element.style.background = hasBatchProgress
      ? `linear-gradient(90deg, rgba(22, 163, 74, 0.92) 0%, rgba(22, 163, 74, 0.92) ${(
          batchRatio * 100
        ).toFixed(2)}%, rgba(8, 49, 28, 0.94) ${(batchRatio * 100).toFixed(2)}%, rgba(8, 49, 28, 0.94) 100%)`
      : "rgba(8, 49, 28, 0.94)";
    dot.style.background = ACTIVE_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(34, 197, 94, 0.18)";
    dot.style.animation = "modal-status-pulse 1.1s ease-in-out infinite";
    const executionLabel = activeState.executionLabel
      ? ` via ${activeState.executionLabel}`
      : "";
    text.textContent = hasBatchProgress
      ? `Remote workflow running on ${activeState.nodeCount} ${nodeLabel}${executionLabel} · ${Math.round(batchValue)}/${Math.round(batchMax)}`
      : `Remote workflow running on ${activeState.nodeCount} ${nodeLabel}${executionLabel}`;
  } else if (activeState.phase === STATE_FINALIZING) {
    element.style.borderColor = "rgba(59, 130, 246, 0.55)";
    element.style.background = "rgba(15, 23, 42, 0.94)";
    dot.style.background = FINALIZING_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(59, 130, 246, 0.18)";
    dot.style.animation = "modal-status-pulse 1.1s ease-in-out infinite";
    text.textContent = activeState.statusMessage ?? "Receiving remote outputs";
  } else if (activeState.phase === STATE_CANCELLING) {
    element.style.borderColor = "rgba(251, 113, 133, 0.58)";
    element.style.background = "rgba(76, 5, 25, 0.94)";
    dot.style.background = CANCELLING_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(251, 113, 133, 0.2)";
    dot.style.animation = "modal-status-pulse 0.75s ease-in-out infinite";
    text.textContent = activeState.statusMessage ?? "Cancelling remote workflow";
  } else if (activeState.phase === STATE_ERROR) {
    element.style.borderColor = "rgba(239, 68, 68, 0.55)";
    element.style.background = "rgba(69, 10, 10, 0.94)";
    dot.style.background = ERROR_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(239, 68, 68, 0.18)";
    dot.style.animation = "none";
    text.textContent = "Remote workflow failed";
  } else {
    element.style.borderColor = "rgba(29, 155, 240, 0.5)";
    element.style.background = "rgba(15, 23, 42, 0.94)";
    dot.style.background = IDLE_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(29, 155, 240, 0.18)";
    dot.style.animation = "none";
    text.textContent = "Remote workflow active";
  }

  dot.style.width = "10px";
  dot.style.height = "10px";
  dot.style.borderRadius = "999px";
  dot.style.display = "inline-block";
  dot.style.marginTop = "2px";
}

/**
 * Record one prompt's global Modal execution phase.
 * @param {string} promptId
 * @param {string} phase
 * @param {number} nodeCount
 * @param {{ message?: string | null, current?: number | null, total?: number | null, modalGpu?: string | null } | null} details
 */
function setGlobalStatusPhase(promptId, phase, nodeCount, details = null) {
  if (!promptId) {
    return;
  }
  if (isPromptTerminal(promptId) && phase !== STATE_ERROR) {
    return;
  }
  const existingState = modalGlobalStatusStates.get(promptId);
  const promptState = ensurePromptState(promptId);
  const detailModalGpu = MODAL_GPU_TYPES.includes(details?.modalGpu) ? details.modalGpu : null;
  if (promptState && detailModalGpu) {
    promptState.modalGpu = detailModalGpu;
  }
  modalGlobalStatusStates.set(promptId, {
    phase: effectiveGlobalStatusPhase(promptId, phase),
    nodeCount: promptRemoteNodeCount(promptId, nodeCount),
    batchValue: existingState?.batchValue ?? null,
    batchMax: existingState?.batchMax ?? null,
    statusMessage: details?.message ?? existingState?.statusMessage ?? null,
    statusCurrent: details?.current ?? existingState?.statusCurrent ?? null,
    statusTotal: details?.total ?? existingState?.statusTotal ?? null,
    modalGpu: detailModalGpu ?? existingState?.modalGpu ?? promptState?.modalGpu ?? null,
    executionLabel:
      existingState?.executionLabel ?? promptState?.executionLabel ?? null,
    updatedAt: nowMs(),
  });
  refreshRemoteConfiguratorPanelForPrompt(promptId);
  refreshGlobalStatusElement();
}

/**
 * Record aggregate mapped-batch progress for one prompt's global Modal status pill.
 * @param {string} promptId
 * @param {number} value
 * @param {number} maxValue
 */
function setGlobalStatusBatchProgress(promptId, value, maxValue) {
  if (!promptId) {
    return;
  }
  if (isPromptTerminal(promptId)) {
    return;
  }
  const existingState = modalGlobalStatusStates.get(promptId);
  const safeMaxValue = Math.max(1, Number(maxValue) || 1);
  const safeValue = Math.max(0, Math.min(safeMaxValue, Number(value) || 0));
  modalGlobalStatusStates.set(promptId, {
    phase: effectiveGlobalStatusPhase(promptId, existingState?.phase ?? EXECUTION_PHASE),
    nodeCount: promptRemoteNodeCount(promptId, existingState?.nodeCount ?? 1),
    batchValue: safeValue,
    batchMax: safeMaxValue,
    statusMessage: existingState?.statusMessage ?? null,
    statusCurrent: existingState?.statusCurrent ?? null,
    statusTotal: existingState?.statusTotal ?? null,
    modalGpu:
      existingState?.modalGpu ?? modalPromptStates.get(promptId)?.modalGpu ?? null,
    executionLabel:
      existingState?.executionLabel ??
      modalPromptStates.get(promptId)?.executionLabel ??
      null,
    updatedAt: nowMs(),
  });
  refreshRemoteConfiguratorPanelForPrompt(promptId);
  refreshGlobalStatusElement();
}

/**
 * Clear one prompt from the global Modal execution badge.
 * @param {string} promptId
 */
function clearGlobalStatusPhase(promptId) {
  if (!promptId) {
    return;
  }
  modalGlobalStatusStates.delete(promptId);
  refreshGlobalStatusElement();
}

/**
 * Show immediate cancellation feedback for one Modal prompt.
 * @param {string} promptId
 */
function markPromptCancellationRequested(promptId) {
  if (!promptId || isPromptTerminal(promptId)) {
    return;
  }
  const promptState = modalPromptStates.get(promptId);
  const stateNodeIds = Array.from(modalNodeStates.entries())
    .filter(([, state]) => state?.promptId === promptId)
    .map(([nodeIdValue]) => nodeIdValue);
  const remoteNodeIds = promptState?.remoteNodeIds?.length ? promptState.remoteNodeIds : stateNodeIds;
  if (remoteNodeIds.length === 0 && !modalGlobalStatusStates.has(promptId)) {
    return;
  }
  modalCancellingPromptIds.add(promptId);
  clearPromptQueued(promptId);
  endSyntheticExecutionUi(promptId);
  setPromptActiveNode(promptId, null);
  setGlobalStatusPhase(promptId, STATE_CANCELLING, remoteNodeIds.length || 1, {
    message: "Cancelling remote workflow",
  });
  if (remoteNodeIds.length > 0) {
    setNodesPhase(remoteNodeIds, STATE_CANCELLING, promptId);
  }
}

/**
 * Remove older cancellation UI before a newer Modal prompt takes over the global badge.
 * @param {string} activePromptId
 */
function clearSupersededCancellingPrompts(activePromptId) {
  for (const [promptId, state] of Array.from(modalGlobalStatusStates.entries())) {
    if (promptId === activePromptId) {
      continue;
    }
    if (effectiveGlobalStatusPhase(promptId, state.phase) !== STATE_CANCELLING) {
      continue;
    }
    if (promptHasLiveRemoteWork(promptId)) {
      continue;
    }
    markPromptTerminal(promptId, "superseded_by_new_prompt");
    clearPromptRemoteNodeVisuals(promptId);
    clearPromptRemoteStates(promptId);
  }
}

/**
 * Return prompt ids targeted by one ComfyUI interrupt request.
 * @param {any} resource
 * @param {any} options
 * @returns {string[]}
 */
function promptIdsFromInterruptRequest(resource, options) {
  const route = String(resource?.url ?? resource ?? "");
  const method = String(options?.method ?? resource?.method ?? "GET").toUpperCase();
  if (!route.includes("/interrupt") || method !== "POST") {
    return [];
  }

  const body = options?.body ?? resource?.body;
  if (typeof body === "string" && body.trim()) {
    try {
      const payload = JSON.parse(body);
      if (payload?.prompt_id != null) {
        return [String(payload.prompt_id)];
      }
    } catch (error) {
      console.debug("Unable to parse ComfyUI interrupt request body for Modal cancellation UI.", error);
    }
  }

  return activeModalUiPromptIds();
}

/**
 * Patch fetchApi so Modal prompts show cancellation feedback as soon as the user clicks cancel.
 */
function patchInterruptFeedback() {
  if (api.__modalInterruptFeedbackPatched || typeof api.fetchApi !== "function") {
    return;
  }
  const originalFetchApi = api.fetchApi;
  api.fetchApi = async function modalFetchApi(resource, options) {
    const promptIds = promptIdsFromInterruptRequest(resource, options);
    for (const promptId of promptIds) {
      markPromptCancellationRequested(promptId);
    }
    if (promptIds.length > 0) {
      await Promise.all(promptIds.map(async (promptId) => {
        try {
          await originalFetchApi.call(this, MODAL_CANCEL_PREPARATION_ROUTE, {
            method: "POST",
            body: JSON.stringify({ prompt_id: promptId }),
          });
        } catch (error) {
          console.warn(`Unable to cancel remote preparation for prompt ${promptId}.`, error);
        }
      }));
    }
    return originalFetchApi.call(this, resource, options);
  };
  api.__modalInterruptFeedbackPatched = true;
}

/**
 * Replay one Modal UI event received from the backend refocus buffer.
 * @param {{ event?: string, payload?: object }} eventRecord
 */
function replayModalUiEvent(eventRecord) {
  const eventName = String(eventRecord?.event ?? "");
  const payload = eventRecord?.payload;
  if (!payload || typeof payload !== "object") {
    return;
  }
  const updatedAtSeconds = Number(eventRecord?.updated_at ?? 0);
  modalReplayedEventUpdatedAtMs = Number.isFinite(updatedAtSeconds) && updatedAtSeconds > 0
    ? updatedAtSeconds * 1000
    : null;
  try {
    if (eventName === "modal_status") {
      handleModalStatus({ detail: payload });
    } else if (eventName === "modal_progress") {
      handleModalProgress({ detail: payload });
    }
  } finally {
    modalReplayedEventUpdatedAtMs = null;
  }
}

/**
 * Fetch recent Modal UI events that may have arrived while the browser was backgrounded.
 * @returns {Promise<void>}
 */
async function replayModalUiEventsAfterVisibilityChange() {
  const clientId = api.clientId == null ? "" : String(api.clientId);
  if (!clientId || typeof api.fetchApi !== "function") {
    return;
  }

  const response = await api.fetchApi(
    `${MODAL_PROGRESS_STATE_ROUTE}?client_id=${encodeURIComponent(clientId)}`,
    { method: "GET" },
  );
  if (response.status !== 200) {
    return;
  }

  const payload = await response.json();
  const events = Array.isArray(payload?.events) ? payload.events : [];
  for (const eventRecord of events) {
    replayModalUiEvent(eventRecord);
  }
}

/**
 * Return prompt ids that currently have temporary Modal UI state.
 * @returns {string[]}
 */
function activeModalUiPromptIds() {
  const promptIds = new Set();
  for (const promptId of modalPromptStates.keys()) {
    promptIds.add(String(promptId));
  }
  for (const promptId of modalGlobalStatusStates.keys()) {
    promptIds.add(String(promptId));
  }
  for (const promptId of modalQueuedPromptIds.values()) {
    promptIds.add(String(promptId));
  }
  for (const promptId of syntheticPromptUiStates.keys()) {
    promptIds.add(String(promptId));
  }
  for (const state of modalNodeStates.values()) {
    if (state?.promptId) {
      promptIds.add(String(state.promptId));
    }
  }
  for (const progressState of modalNodeProgress.values()) {
    if (progressState?.promptId) {
      promptIds.add(String(progressState.promptId));
    }
  }
  for (const laneState of modalNodeProgressLanes.values()) {
    if (laneState?.promptId) {
      promptIds.add(String(laneState.promptId));
    }
  }
  for (const batchState of modalNodeBatchProgress.values()) {
    if (batchState?.promptId) {
      promptIds.add(String(batchState.promptId));
    }
  }
  for (const cachedState of modalNodeCachedStates.values()) {
    if (cachedState?.promptId) {
      promptIds.add(String(cachedState.promptId));
    }
  }
  return Array.from(promptIds).filter((promptId) => promptId && !isPromptTerminal(promptId));
}

/**
 * Return the oldest age in milliseconds of a prompt's temporary Modal UI state.
 * @param {string} promptId
 * @returns {number}
 */
function promptUiStateAgeMs(promptId) {
  const updatedAts = [];
  const promptState = modalPromptStates.get(promptId);
  if (promptState?.startedAt) {
    updatedAts.push(promptState.startedAt);
  }
  const globalStatusState = modalGlobalStatusStates.get(promptId);
  if (globalStatusState?.updatedAt) {
    updatedAts.push(globalStatusState.updatedAt);
  }
  for (const state of modalNodeStates.values()) {
    if (state?.promptId === promptId && state?.updatedAt) {
      updatedAts.push(state.updatedAt);
    }
  }
  for (const progressState of modalNodeProgress.values()) {
    if (progressState?.promptId === promptId && progressState?.updatedAt) {
      updatedAts.push(progressState.updatedAt);
    }
  }
  for (const laneState of modalNodeProgressLanes.values()) {
    if (laneState?.promptId !== promptId) {
      continue;
    }
    for (const laneProgress of laneState.lanes.values()) {
      if (laneProgress?.updatedAt) {
        updatedAts.push(laneProgress.updatedAt);
      }
    }
  }
  for (const batchState of modalNodeBatchProgress.values()) {
    if (batchState?.promptId === promptId && batchState?.updatedAt) {
      updatedAts.push(batchState.updatedAt);
    }
  }
  for (const cachedState of modalNodeCachedStates.values()) {
    if (cachedState?.promptId === promptId && cachedState?.cachedAt) {
      updatedAts.push(cachedState.cachedAt);
    }
  }
  if (updatedAts.length === 0) {
    return 0;
  }
  return Math.max(0, nowMs() - Math.min(...updatedAts));
}

/**
 * Return prompt ids reported by ComfyUI's queue endpoint.
 * @param {any} queuePayload
 * @returns {Set<string>}
 */
function promptIdsFromQueuePayload(queuePayload) {
  const promptIds = new Set();
  for (const queueName of ["queue_running", "queue_pending"]) {
    const queueEntries = Array.isArray(queuePayload?.[queueName]) ? queuePayload[queueName] : [];
    for (const entry of queueEntries) {
      if (Array.isArray(entry) && entry.length > 1) {
        promptIds.add(String(entry[1]));
      } else if (entry?.prompt_id != null) {
        promptIds.add(String(entry.prompt_id));
      }
    }
  }
  return promptIds;
}

/**
 * Return whether one ComfyUI history response contains a prompt id.
 * @param {any} historyPayload
 * @param {string} promptId
 * @returns {boolean}
 */
function historyPayloadHasPrompt(historyPayload, promptId) {
  if (!historyPayload || typeof historyPayload !== "object") {
    return false;
  }
  if (Object.prototype.hasOwnProperty.call(historyPayload, promptId)) {
    return true;
  }
  return String(historyPayload.prompt_id ?? "") === promptId;
}

/**
 * Fetch JSON from the ComfyUI API when the endpoint returns a successful response.
 * @param {string} route
 * @returns {Promise<any | null>}
 */
async function fetchComfyJson(route) {
  if (typeof api.fetchApi !== "function") {
    return null;
  }
  const response = await api.fetchApi(route, { method: "GET" });
  if (response.status !== 200) {
    return null;
  }
  return response.json();
}

/**
 * Clear refocus-stale Modal visuals once ComfyUI reports a prompt has finished.
 * @param {string} promptId
 * @param {string} phase
 */
function clearRefocusCompletedPrompt(promptId, phase) {
  markPromptTerminal(promptId, phase);
  endSyntheticExecutionUi(promptId);
  clearGlobalStatusPhase(promptId);
  clearPromptRemoteStates(promptId);
}

/**
 * Reconcile temporary Modal UI state against ComfyUI queue/history after refocus.
 * @returns {Promise<void>}
 */
async function reconcileModalUiAfterVisibilityChange() {
  const promptIds = activeModalUiPromptIds();
  if (promptIds.length === 0 || typeof api.fetchApi !== "function") {
    return;
  }

  const queuePayload = await fetchComfyJson(COMFY_QUEUE_ROUTE);
  if (!queuePayload) {
    return;
  }
  const queuedPromptIds = promptIdsFromQueuePayload(queuePayload);
  for (const promptId of promptIds) {
    if (queuedPromptIds.has(promptId)) {
      continue;
    }
    const historyPayload = await fetchComfyJson(
      `${COMFY_HISTORY_ROUTE}/${encodeURIComponent(promptId)}`,
    );
    if (historyPayloadHasPrompt(historyPayload, promptId)) {
      clearRefocusCompletedPrompt(promptId, "execution_success");
    } else if (promptUiStateAgeMs(promptId) > REFOCUS_STALE_PROMPT_GRACE_MS) {
      clearRefocusCompletedPrompt(promptId, "stale_refocus_cleanup");
    }
  }
}

/**
 * Refresh the badge and canvas when the tab regains visibility.
 */
function refreshModalUiAfterVisibilityChange() {
  requestImmediateModalContainerStatusPoll();
  refreshGlobalStatusElement();
  if (Array.from(modalNodeStates.values()).length > 0) {
    ensureAnimationLoop();
  }
  refreshNodeDecorations();
  if (modalVisibilityRefreshInFlight) {
    return;
  }
  modalVisibilityRefreshInFlight = replayModalUiEventsAfterVisibilityChange()
    .then(() => reconcileModalUiAfterVisibilityChange())
    .catch((error) => {
      console.warn("Unable to refresh Modal UI state after visibility change.", error);
    })
    .finally(() => {
      modalVisibilityRefreshInFlight = null;
      refreshGlobalStatusElement();
      if (Array.from(modalNodeStates.values()).length > 0) {
        ensureAnimationLoop();
      }
      refreshNodeDecorations();
    });
}

/**
 * Return the prompt metadata bucket, creating it if needed.
 * @param {string} promptId
 * @returns {{ startedAt: number, modalGpu: string | null, executionLabel: string | null, scheduledEnvironmentCount: number, remoteNodeIds: string[], componentsByRepresentative: Map<string, string[]>, componentNodeIdsByMember: Map<string, string[]>, representativeNodeIdByMember: Map<string, string>, componentLabelByMember: Map<string, string>, executionAssignmentsByRepresentative: Map<string, object>, executionAssignmentByMember: Map<string, object>, laneNodeIdsByLane: Map<string, string> }}
 */
function ensurePromptState(promptId) {
  if (isPromptTerminal(promptId)) {
    return null;
  }
  if (!modalPromptStates.has(promptId)) {
    const startedAt = modalReplayedEventUpdatedAtMs ?? nowMs();
    modalPromptStates.set(promptId, {
      startedAt,
      modalGpu: null,
      executionLabel: null,
      scheduledEnvironmentCount: 0,
      remoteNodeIds: [],
      componentsByRepresentative: new Map(),
      componentNodeIdsByMember: new Map(),
      representativeNodeIdByMember: new Map(),
      componentLabelByMember: new Map(),
      executionAssignmentsByRepresentative: new Map(),
      executionAssignmentByMember: new Map(),
      configuratorNodeId: null,
      remoteExecutionPlanAssignments: {},
      remoteExecutionConfigurations: [],
      descendantNodeIdsByAncestor: new Map(),
      laneNodeIdsByLane: new Map(),
      activeNodeId: null,
      hasStreamedProgress: false,
      hasRemoteExecutionStarted: false,
    });
  }
  return modalPromptStates.get(promptId);
}

/**
 * Clear prompt-scoped progress and cache maps even when prompt metadata is already gone.
 * @param {string} promptId
 */
function clearPromptProgressStates(promptId) {
  for (const [nodeIdValue, progressState] of Array.from(modalNodeProgress.entries())) {
    if (progressState?.promptId === promptId) {
      modalNodeProgress.delete(nodeIdValue);
    }
  }
  for (const [nodeIdValue, laneState] of Array.from(modalNodeProgressLanes.entries())) {
    if (laneState?.promptId === promptId) {
      modalNodeProgressLanes.delete(nodeIdValue);
    }
  }
  for (const [nodeIdValue, batchState] of Array.from(modalNodeBatchProgress.entries())) {
    if (batchState?.promptId === promptId) {
      modalNodeBatchProgress.delete(nodeIdValue);
    }
  }
  for (const [nodeIdValue, cachedState] of Array.from(modalNodeCachedStates.entries())) {
    if (cachedState?.promptId === promptId) {
      modalNodeCachedStates.delete(nodeIdValue);
    }
  }
  stopAnimationLoopIfIdle();
}

/**
 * Return whether an incoming prompt update should replace the current node state.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @returns {boolean}
 */
function shouldApplyPromptState(nodeIdValue, promptId) {
  if (isPromptTerminal(promptId)) {
    return false;
  }
  const incomingPromptState = modalPromptStates.get(promptId);
  if (!incomingPromptState) {
    return true;
  }

  const currentState = modalNodeStates.get(nodeIdValue);
  if (!currentState?.promptId) {
    return true;
  }

  const currentPromptState = modalPromptStates.get(currentState.promptId);
  if (!currentPromptState) {
    return true;
  }

  return incomingPromptState.startedAt >= currentPromptState.startedAt;
}

/**
 * Return whether a remote phase changes visually over time.
 * @param {string | undefined} phase
 * @returns {boolean}
 */
function isPulsingNodePhase(phase) {
  return [
    STATE_SETUP,
    STATE_STARTING,
    STATE_READY,
    STATE_ACTIVE,
    STATE_FINALIZING,
    STATE_CANCELLING,
    STATE_ERROR,
  ].includes(phase);
}

/**
 * Return whether a progress payload has an active time-based fade.
 * @param {{ fadingStartedAt?: number | null } | null | undefined} progressState
 * @returns {boolean}
 */
function progressFadeNeedsAnimation(progressState) {
  return Boolean(progressState?.fadingStartedAt && progressVisualOpacity(progressState) > 0);
}

/**
 * Return whether any progress overlay is currently changing without websocket input.
 * @returns {boolean}
 */
function progressNeedsAnimation() {
  for (const progressState of modalNodeProgress.values()) {
    if (progressFadeNeedsAnimation(progressState)) {
      return true;
    }
  }
  for (const batchState of modalNodeBatchProgress.values()) {
    if (progressFadeNeedsAnimation(batchState)) {
      return true;
    }
  }
  for (const laneState of modalNodeProgressLanes.values()) {
    for (const laneProgress of laneState.lanes.values()) {
      if (laneProgress.setupOnly || progressFadeNeedsAnimation(laneProgress)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Return whether any Modal canvas decoration needs a scheduled redraw.
 * @returns {boolean}
 */
function shouldAnimateModalVisuals() {
  const hasPulsingState = Array.from(modalNodeStates.values()).some((state) =>
    isPulsingNodePhase(state.phase),
  );
  return hasPulsingState || progressNeedsAnimation();
}

/**
 * Stop the scheduled redraw loop when no Modal visuals are moving.
 */
function stopAnimationLoopIfIdle() {
  if (shouldAnimateModalVisuals()) {
    return;
  }
  if (animationFrameHandle !== null && typeof cancelAnimationFrame === "function") {
    cancelAnimationFrame(animationFrameHandle);
  }
  animationFrameHandle = null;
}

/**
 * Mark the canvas dirty at a bounded cadence while visual states are active.
 * @param {number | undefined} timestamp
 */
function refreshCanvasAnimation(timestamp = performance.now()) {
  animationFrameHandle = null;
  if (!shouldAnimateModalVisuals()) {
    return;
  }
  if (timestamp - modalLastAnimationRedrawAt >= MODAL_ANIMATION_FRAME_INTERVAL_MS) {
    refreshNodeDecorations();
    modalLastAnimationRedrawAt = timestamp;
  }
  animationFrameHandle = requestAnimationFrame(refreshCanvasAnimation);
}

/**
 * Ensure the redraw loop is running while remote visual effects are active.
 */
function ensureAnimationLoop() {
  if (!shouldAnimateModalVisuals()) {
    stopAnimationLoopIfIdle();
    return;
  }
  if (animationFrameHandle !== null) {
    return;
  }
  animationFrameHandle = requestAnimationFrame(refreshCanvasAnimation);
}

/**
 * Clear any pending visual-state timeout for a node.
 * @param {string} nodeIdValue
 */
function clearNodeTimer(nodeIdValue) {
  const timerId = modalNodeClearTimers.get(nodeIdValue);
  if (timerId !== undefined) {
    clearTimeout(timerId);
    modalNodeClearTimers.delete(nodeIdValue);
  }
}

/**
 * Schedule a node visual state to clear after a delay.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @param {number} delayMs
 */
function scheduleNodeClear(nodeIdValue, promptId, delayMs) {
  clearNodeTimer(nodeIdValue);
  const timerId = setTimeout(() => {
    const state = modalNodeStates.get(nodeIdValue);
    if (state?.promptId !== promptId) {
      return;
    }
    modalNodeStates.delete(nodeIdValue);
    modalNodeClearTimers.delete(nodeIdValue);
    stopAnimationLoopIfIdle();
    reconcilePromptGlobalStatus(promptId);
    refreshNodeDecorations();
  }, delayMs);
  modalNodeClearTimers.set(nodeIdValue, timerId);
}

/**
 * Set the current visual phase for a list of remote nodes.
 * @param {string[]} nodeIds
 * @param {string} phase
 * @param {string} promptId
 * @param {string | undefined} errorMessage
 */
function setNodesPhase(nodeIds, phase, promptId, errorMessage) {
  const affectedAncestorNodeIds = new Set();
  const promptState = promptId ? modalPromptStates.get(promptId) : null;
  for (const currentNodeId of nodeIds) {
    if (!shouldApplyPromptState(currentNodeId, promptId)) {
      continue;
    }
    const existingState = modalNodeStates.get(currentNodeId);
    if (
      phase === STATE_READY &&
      existingState?.promptId === promptId &&
      [STATE_COMPLETE, STATE_FINALIZING, STATE_ERROR].includes(existingState.phase)
    ) {
      continue;
    }
    clearNodeTimer(currentNodeId);
    modalNodeStates.set(currentNodeId, {
      phase,
      promptId,
      errorMessage,
      updatedAt: nowMs(),
    });
    if (phase === STATE_ERROR) {
      scheduleNodeClear(currentNodeId, promptId, ERROR_CLEAR_DELAY_MS);
    }
    if ([STATE_SETUP, STATE_STARTING, STATE_CANCELLING, STATE_ERROR].includes(phase)) {
      clearNodeCached(currentNodeId, promptId);
    }
    if (phase === STATE_COMPLETE) {
      fadeNodeProgress(currentNodeId, promptId);
      if (promptState?.activeNodeId === String(currentNodeId)) {
        promptState.activeNodeId = null;
      }
    }
    if (phase === STATE_ERROR || phase === STATE_CANCELLING) {
      clearNodeProgress(currentNodeId, promptId);
    }
    for (const ancestorNodeId of ancestorNodeIds(currentNodeId)) {
      affectedAncestorNodeIds.add(ancestorNodeId);
    }
  }
  for (const ancestorNodeId of affectedAncestorNodeIds) {
    refreshAncestorNodePhase(promptId, ancestorNodeId, errorMessage);
  }
  ensureAnimationLoop();
  if ([STATE_COMPLETE, STATE_FINALIZING, STATE_ERROR].includes(phase)) {
    reconcilePromptGlobalStatus(promptId);
  }
  refreshNodeDecorations();
}

/**
 * Register remote component membership for a prompt.
 * @param {string} promptId
 * @param {string[]} remoteNodeIds
 * @param {{ representative_node_id: string, node_ids: string[] }[]} components
 */
function registerPromptComponents(promptId, remoteNodeIds, components) {
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  if (components.length > 0) {
    const mergedRemoteNodeIds = new Set(remoteNodeIds.map((nodeIdValue) => String(nodeIdValue)));
    promptState.componentsByRepresentative.clear();
    promptState.componentNodeIdsByMember.clear();
    promptState.representativeNodeIdByMember.clear();
    promptState.componentLabelByMember.clear();
    promptState.activeNodeId = null;
    promptState.hasStreamedProgress = false;
    for (const [componentIndex, component] of components.entries()) {
      const componentNodeIds = Array.from(
        new Set(component.node_ids.map((nodeIdValue) => String(nodeIdValue))),
      );
      const representativeNodeId = String(component.representative_node_id);
      const componentLabel = String(componentIndex + 1);
      promptState.componentsByRepresentative.set(
        representativeNodeId,
        componentNodeIds,
      );
      promptState.componentNodeIdsByMember.set(representativeNodeId, componentNodeIds);
      promptState.representativeNodeIdByMember.set(representativeNodeId, representativeNodeId);
      promptState.componentLabelByMember.set(representativeNodeId, componentLabel);
      for (const componentNodeId of componentNodeIds) {
        promptState.componentNodeIdsByMember.set(componentNodeId, componentNodeIds);
        promptState.representativeNodeIdByMember.set(componentNodeId, representativeNodeId);
        promptState.componentLabelByMember.set(componentNodeId, componentLabel);
        mergedRemoteNodeIds.add(componentNodeId);
      }
    }
    if (mergedRemoteNodeIds.size > 0) {
      promptState.remoteNodeIds = Array.from(mergedRemoteNodeIds);
    }
    rebuildPromptAncestorMap(promptState);
    return;
  }
  if (remoteNodeIds.length > 0) {
    if (promptState.remoteNodeIds.length === 0) {
      promptState.remoteNodeIds = [...remoteNodeIds];
    } else {
      const mergedRemoteNodeIds = new Set(promptState.remoteNodeIds);
      for (const remoteNodeId of remoteNodeIds) {
        mergedRemoteNodeIds.add(String(remoteNodeId));
      }
      promptState.remoteNodeIds = Array.from(mergedRemoteNodeIds);
    }
    rebuildPromptAncestorMap(promptState);
  }
}

/**
 * Store planner assignments by both representative and visible member node.
 * @param {string} promptId
 * @param {Record<string, Record<string, any>>} assignments
 */
function registerPromptExecutionAssignments(promptId, assignments) {
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  promptState.executionAssignmentsByRepresentative.clear();
  promptState.executionAssignmentByMember.clear();
  const environmentIds = new Set();

  for (const [representativeNodeIdValue, rawAssignment] of Object.entries(assignments ?? {})) {
    if (!rawAssignment || typeof rawAssignment !== "object") {
      continue;
    }
    const representativeNodeId = String(representativeNodeIdValue);
    const environmentId = String(rawAssignment.environment_id ?? "").trim();
    const assignment = {
      provider: String(rawAssignment.provider ?? "modal"),
      environmentId,
      location: String(rawAssignment.execution_location ?? "").trim() || null,
    };
    if (environmentId) {
      environmentIds.add(environmentId);
    }
    promptState.executionAssignmentsByRepresentative.set(representativeNodeId, assignment);
    const assignedNodeIds = Array.isArray(rawAssignment.node_ids) && rawAssignment.node_ids.length > 0
      ? rawAssignment.node_ids.map((nodeIdValue) => String(nodeIdValue))
      : (promptState.componentNodeIdsByMember.get(representativeNodeId) ?? [representativeNodeId]);
    for (const assignedNodeId of assignedNodeIds) {
      promptState.executionAssignmentByMember.set(assignedNodeId, assignment);
    }
    promptState.executionAssignmentByMember.set(representativeNodeId, assignment);
  }
  promptState.scheduledEnvironmentCount = environmentIds.size;
}

/**
 * Update one component's runtime identity from its streamed progress metadata.
 * @param {string} promptId
 * @param {string} nodeIdValue
 * @param {Record<string, any>} metadata
 */
function updateNodeExecutionLocation(promptId, nodeIdValue, metadata) {
  const promptState = modalPromptStates.get(promptId);
  const location = String(metadata.execution_location ?? "").trim();
  if (!promptState || !location) {
    return;
  }
  const safeNodeIdValue = String(nodeIdValue);
  const representativeNodeId =
    promptState.representativeNodeIdByMember.get(safeNodeIdValue) ?? safeNodeIdValue;
  const existingAssignment =
    promptState.executionAssignmentByMember.get(safeNodeIdValue) ??
    promptState.executionAssignmentsByRepresentative.get(representativeNodeId) ??
    {};
  const assignment = {
    ...existingAssignment,
    provider: String(metadata.execution_provider ?? existingAssignment.provider ?? "modal"),
    environmentId: String(
      metadata.execution_environment_id ?? existingAssignment.environmentId ?? "",
    ),
    location,
  };
  promptState.executionAssignmentsByRepresentative.set(representativeNodeId, assignment);
  const componentNodeIds =
    promptState.componentNodeIdsByMember.get(safeNodeIdValue) ?? [safeNodeIdValue];
  for (const componentNodeId of componentNodeIds) {
    promptState.executionAssignmentByMember.set(String(componentNodeId), assignment);
  }
  promptState.executionAssignmentByMember.set(representativeNodeId, assignment);
}

/**
 * Return ancestor node ids for a composed subgraph prompt id like `24:23`.
 * @param {string} nodeIdValue
 * @returns {string[]}
 */
function ancestorNodeIds(nodeIdValue) {
  const segments = String(nodeIdValue).split(":");
  const ancestorNodeIds = [];
  for (let index = 1; index < segments.length; index += 1) {
    ancestorNodeIds.push(segments.slice(0, index).join(":"));
  }
  return ancestorNodeIds;
}

/**
 * Rebuild descendant-to-ancestor mappings for one prompt state.
 * @param {{ remoteNodeIds: string[], componentsByRepresentative: Map<string, string[]>, descendantNodeIdsByAncestor: Map<string, Set<string>> }} promptState
 */
function rebuildPromptAncestorMap(promptState) {
  promptState.descendantNodeIdsByAncestor.clear();
  const candidateNodeIds = new Set(promptState.remoteNodeIds);
  for (const componentNodeIds of promptState.componentsByRepresentative.values()) {
    for (const componentNodeId of componentNodeIds) {
      candidateNodeIds.add(String(componentNodeId));
    }
  }

  for (const candidateNodeId of candidateNodeIds) {
    for (const ancestorNodeId of ancestorNodeIds(candidateNodeId)) {
      if (!promptState.descendantNodeIdsByAncestor.has(ancestorNodeId)) {
        promptState.descendantNodeIdsByAncestor.set(ancestorNodeId, new Set());
      }
      promptState.descendantNodeIdsByAncestor.get(ancestorNodeId).add(candidateNodeId);
    }
  }
}

/**
 * Return a descendant's effective phase, including live streamed progress.
 * @param {string} promptId
 * @param {string} descendantNodeId
 * @returns {{ phase: string, state: Record<string, any> } | null}
 */
function remoteDescendantPhase(promptId, descendantNodeId) {
  const state = modalNodeStates.get(descendantNodeId);
  if (state?.promptId !== promptId) {
    return null;
  }
  return {
    phase: deriveRemoteNodePhase(
      state.phase,
      hasLiveNodeProgress(descendantNodeId, promptId),
    ),
    state,
  };
}

/**
 * Select the most useful visible phase for a subgraph containing mixed remote work.
 * @param {Record<string, number>} phaseCounts
 * @returns {string}
 */
function dominantRemoteContainerPhase(phaseCounts) {
  const phasePriority = [
    STATE_ERROR,
    STATE_CANCELLING,
    STATE_ACTIVE,
    STATE_STARTING,
    STATE_SETUP,
    STATE_READY,
    STATE_FINALIZING,
    STATE_COMPLETE,
  ];
  return phasePriority.find((phase) => (phaseCounts[phase] ?? 0) > 0) ?? STATE_SETUP;
}

/**
 * Aggregate all remote descendants beneath one subgraph into a container visual state.
 * @param {string} promptId
 * @param {string} ancestorNodeId
 * @param {string | undefined} errorMessage
 * @returns {Record<string, any> | null}
 */
function remoteContainerVisualState(promptId, ancestorNodeId, errorMessage) {
  const promptState = modalPromptStates.get(promptId);
  const descendantNodeIds = promptState?.descendantNodeIdsByAncestor.get(ancestorNodeId);
  if (!promptState || !descendantNodeIds || descendantNodeIds.size === 0) {
    return null;
  }

  const phaseCounts = {};
  const descendantStates = [];
  for (const descendantNodeId of descendantNodeIds) {
    const descendantState = remoteDescendantPhase(promptId, descendantNodeId);
    if (!descendantState) {
      continue;
    }
    phaseCounts[descendantState.phase] = (phaseCounts[descendantState.phase] ?? 0) + 1;
    descendantStates.push(descendantState.state);
  }
  if (descendantStates.length === 0) {
    return null;
  }

  const phases = Object.keys(phaseCounts);
  const descendantErrorMessage = descendantStates.find(
    (state) => state.phase === STATE_ERROR && state.errorMessage,
  )?.errorMessage;
  return {
    phase: dominantRemoteContainerPhase(phaseCounts),
    promptId,
    errorMessage: descendantErrorMessage ?? errorMessage,
    updatedAt: Math.max(...descendantStates.map((state) => Number(state.updatedAt ?? 0))),
    isRemoteContainer: true,
    isMixedRemoteContainer: phases.length > 1,
    phaseCounts,
    remoteDescendantCount: descendantNodeIds.size,
  };
}

/**
 * Recompute one visible ancestor node's phase from its descendant remote prompt nodes.
 * @param {string} promptId
 * @param {string} ancestorNodeId
 * @param {string | undefined} errorMessage
 */
function refreshAncestorNodePhase(promptId, ancestorNodeId, errorMessage) {
  const ancestorState = remoteContainerVisualState(promptId, ancestorNodeId, errorMessage);
  if (!ancestorState) {
    return;
  }
  if (!shouldApplyPromptState(ancestorNodeId, promptId)) {
    return;
  }
  clearNodeTimer(ancestorNodeId);
  modalNodeStates.set(ancestorNodeId, ancestorState);
  if (ancestorState.phase === STATE_ERROR) {
    scheduleNodeClear(ancestorNodeId, promptId, ERROR_CLEAR_DELAY_MS);
  }
}

/**
 * Record the currently active remote node inside one prompt.
 * @param {string} promptId
 * @param {string | null} activeNodeId
 */
function setPromptActiveNode(promptId, activeNodeId) {
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  promptState.activeNodeId = activeNodeId ? String(activeNodeId) : null;
}

/**
 * Mark known upstream remote nodes complete when a downstream node is executing.
 * @param {string} promptId
 * @param {any[] | undefined} ancestorNodeIds
 * @param {string | null} activeNodeId
 */
function completeRemoteAncestorsBeforeActiveNode(promptId, ancestorNodeIds, activeNodeId) {
  if (!Array.isArray(ancestorNodeIds) || ancestorNodeIds.length === 0) {
    return;
  }
  const completedNodeIds = Array.from(
    new Set(
      ancestorNodeIds
        .map((nodeIdValue) => String(nodeIdValue))
        .filter((nodeIdValue) => nodeIdValue && nodeIdValue !== activeNodeId),
    ),
  );
  if (completedNodeIds.length === 0) {
    return;
  }
  for (const completedNodeId of completedNodeIds) {
    fadeNodeProgress(completedNodeId, promptId);
  }
  setNodesPhase(completedNodeIds, STATE_COMPLETE, promptId);
}

/**
 * Return whether two nodes belong to the same known remote component.
 * @param {string} promptId
 * @param {string | null | undefined} leftNodeId
 * @param {string | null | undefined} rightNodeId
 * @returns {boolean}
 */
function nodesShareRemoteComponent(promptId, leftNodeId, rightNodeId) {
  if (!leftNodeId || !rightNodeId) {
    return false;
  }
  const leftComponentNodeIds = resolveComponentNodeIds(promptId, leftNodeId);
  if (!leftComponentNodeIds?.length) {
    return String(leftNodeId) === String(rightNodeId);
  }
  return leftComponentNodeIds.map((nodeIdValue) => String(nodeIdValue)).includes(String(rightNodeId));
}

/**
 * Return the numeric progress payload for one node when it belongs to the prompt.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @returns {{ promptId: string, value: number, max: number, updatedAt: number, iterationRate: number | null } | null}
 */
function nodeProgressState(nodeIdValue, promptId) {
  const progressState = modalNodeProgress.get(String(nodeIdValue)) ?? null;
  return progressState?.promptId === promptId ? progressState : null;
}

/**
 * Return sorted per-lane progress payloads for one node when they belong to the prompt.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @returns {{ laneId: string, value: number, max: number, itemIndex: number | null, updatedAt: number, setupOnly: boolean, iterationRate: number | null }[]}
 */
function nodeProgressLanes(nodeIdValue, promptId) {
  const progressLaneState = modalNodeProgressLanes.get(String(nodeIdValue)) ?? null;
  if (progressLaneState?.promptId !== promptId) {
    return [];
  }
  return Array.from(progressLaneState.lanes.values()).sort(
    (left, right) => Number(left.laneId) - Number(right.laneId),
  );
}

/**
 * Return whether one node currently has live numeric or per-lane progress.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @returns {boolean}
 */
function hasLiveNodeProgress(nodeIdValue, promptId) {
  return (
    Boolean(nodeProgressState(nodeIdValue, promptId)) ||
    nodeProgressLanes(nodeIdValue, promptId).length > 0
  );
}

/**
 * Return opacity for one progress payload while it fades after completion.
 * @param {{ fadingStartedAt?: number | null } | null | undefined} progressState
 * @returns {number}
 */
function progressVisualOpacity(progressState) {
  if (!progressState) {
    return 0;
  }
  const fadingStartedAt = Number(progressState?.fadingStartedAt ?? 0);
  if (!fadingStartedAt) {
    return 1;
  }
  const fadeRatio = (nowMs() - fadingStartedAt) / PROGRESS_FADE_MS;
  return Math.max(0, Math.min(1, 1 - fadeRatio));
}

/**
 * Derive the displayed node phase from the stored phase plus live progress.
 * @param {string | undefined} phase
 * @param {boolean} hasLiveProgress
 * @returns {string | undefined}
 */
function deriveRemoteNodePhase(phase, hasLiveProgress) {
  if ([STATE_ERROR, STATE_SETUP, STATE_STARTING].includes(phase ?? "")) {
    return phase;
  }
  if (phase === STATE_COMPLETE || phase === STATE_FINALIZING) {
    return phase;
  }
  if (hasLiveProgress) {
    return STATE_ACTIVE;
  }
  return phase;
}

/**
 * Remove one worker lane from a node and its visible ancestors without triggering a redraw.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @param {string} laneId
 */
function deleteNodeProgressLane(nodeIdValue, promptId, laneId) {
  const progressNodeIds = [String(nodeIdValue), ...ancestorNodeIds(nodeIdValue)];

  for (const progressNodeId of progressNodeIds) {
    const laneState = modalNodeProgressLanes.get(progressNodeId);
    if (!laneState || laneState.promptId !== promptId) {
      continue;
    }
    laneState.lanes.delete(laneId);
    if (laneState.lanes.size === 0) {
      modalNodeProgressLanes.delete(progressNodeId);
    }
  }
}

/**
 * Remove stored numeric progress for one node and its visible ancestors.
 * @param {string} nodeIdValue
 * @param {string | undefined} promptId
 */
function clearNodeProgress(nodeIdValue, promptId) {
  const progressNodeIds = [String(nodeIdValue), ...ancestorNodeIds(nodeIdValue)];
  for (const progressNodeId of progressNodeIds) {
    const progressState = modalNodeProgress.get(progressNodeId);
    if (!progressState) {
      modalNodeProgress.delete(progressNodeId);
    } else if (!promptId || progressState.promptId === promptId) {
      modalNodeProgress.delete(progressNodeId);
    }
    const batchState = modalNodeBatchProgress.get(progressNodeId);
    if (batchState && (!promptId || batchState.promptId === promptId)) {
      modalNodeBatchProgress.delete(progressNodeId);
    }
    const laneState = modalNodeProgressLanes.get(progressNodeId);
    if (!laneState) {
      continue;
    }
    if (promptId && laneState.promptId !== promptId) {
      continue;
    }
    modalNodeProgressLanes.delete(progressNodeId);
  }
  stopAnimationLoopIfIdle();
}

/**
 * Remove one faded progress entry if it still belongs to the same prompt.
 * @param {string} progressNodeId
 * @param {string} promptId
 * @param {number} fadingStartedAt
 */
function clearFadedNodeProgress(progressNodeId, promptId, fadingStartedAt) {
  const progressState = modalNodeProgress.get(progressNodeId);
  if (progressState?.promptId === promptId && progressState.fadingStartedAt === fadingStartedAt) {
    modalNodeProgress.delete(progressNodeId);
  }

  const batchState = modalNodeBatchProgress.get(progressNodeId);
  if (batchState?.promptId === promptId && batchState.fadingStartedAt === fadingStartedAt) {
    modalNodeBatchProgress.delete(progressNodeId);
  }

  const laneState = modalNodeProgressLanes.get(progressNodeId);
  if (laneState?.promptId === promptId) {
    for (const [laneId, laneProgress] of Array.from(laneState.lanes.entries())) {
      if (laneProgress.fadingStartedAt === fadingStartedAt) {
        laneState.lanes.delete(laneId);
      }
    }
    if (laneState.lanes.size === 0) {
      modalNodeProgressLanes.delete(progressNodeId);
    }
  }

  stopAnimationLoopIfIdle();
  reconcilePromptGlobalStatus(promptId);
  refreshNodeDecorations();
}

/**
 * Fade stored progress for one completed node and its visible ancestors.
 * @param {string} nodeIdValue
 * @param {string | undefined} promptId
 */
function fadeNodeProgress(nodeIdValue, promptId) {
  const fadingStartedAt = nowMs();
  const progressNodeIds = [String(nodeIdValue), ...ancestorNodeIds(nodeIdValue)];
  let fadedAnyProgress = false;

  for (const progressNodeId of progressNodeIds) {
    let fadedNodeProgress = false;
    const progressState = modalNodeProgress.get(progressNodeId);
    if (progressState && (!promptId || progressState.promptId === promptId)) {
      progressState.fadingStartedAt = fadingStartedAt;
      progressState.value = progressState.max;
      fadedAnyProgress = true;
      fadedNodeProgress = true;
    }

    const batchState = modalNodeBatchProgress.get(progressNodeId);
    if (batchState && (!promptId || batchState.promptId === promptId)) {
      batchState.fadingStartedAt = fadingStartedAt;
      batchState.value = batchState.max;
      fadedAnyProgress = true;
      fadedNodeProgress = true;
    }

    const laneState = modalNodeProgressLanes.get(progressNodeId);
    if (laneState && (!promptId || laneState.promptId === promptId)) {
      for (const laneProgress of laneState.lanes.values()) {
        laneProgress.fadingStartedAt = fadingStartedAt;
        laneProgress.value = laneProgress.max;
        fadedAnyProgress = true;
        fadedNodeProgress = true;
      }
    }

    if (fadedNodeProgress && promptId) {
      setTimeout(
        () => clearFadedNodeProgress(progressNodeId, promptId, fadingStartedAt),
        PROGRESS_FADE_MS,
      );
    }
  }

  if (fadedAnyProgress) {
    ensureAnimationLoop();
    refreshNodeDecorations();
  }
}

/**
 * Return one cached-node marker when it belongs to the prompt.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @returns {{ promptId: string, cachedAt: number } | null}
 */
function nodeCachedState(nodeIdValue, promptId) {
  const cachedState = modalNodeCachedStates.get(String(nodeIdValue)) ?? null;
  return cachedState?.promptId === promptId ? cachedState : null;
}

/**
 * Mark one node and its visible ancestors as restored from the remote Modal node-output cache.
 * @param {string} nodeIdValue
 * @param {string} promptId
 */
function markNodeCached(nodeIdValue, promptId) {
  const progressNodeIds = [String(nodeIdValue), ...ancestorNodeIds(nodeIdValue)];
  for (const progressNodeId of progressNodeIds) {
    if (!shouldApplyPromptState(progressNodeId, promptId)) {
      continue;
    }
    modalNodeCachedStates.set(progressNodeId, {
      promptId,
      cachedAt: nowMs(),
    });
  }
  ensureAnimationLoop();
  refreshNodeDecorations();
}

/**
 * Remove one cached-node marker from a node and its visible ancestors.
 * @param {string} nodeIdValue
 * @param {string | undefined} promptId
 */
function clearNodeCached(nodeIdValue, promptId) {
  const progressNodeIds = [String(nodeIdValue), ...ancestorNodeIds(nodeIdValue)];
  for (const progressNodeId of progressNodeIds) {
    const cachedState = modalNodeCachedStates.get(progressNodeId);
    if (!cachedState) {
      continue;
    }
    if (promptId && cachedState.promptId !== promptId) {
      continue;
    }
    modalNodeCachedStates.delete(progressNodeId);
  }
  stopAnimationLoopIfIdle();
}

/**
 * Record aggregate mapped-batch progress on the representative remote node only.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @param {number} value
 * @param {number} maxValue
 */
function setNodeBatchProgress(nodeIdValue, promptId, value, maxValue) {
  const safeMaxValue = Math.max(1, Number(maxValue) || 1);
  const safeValue = Math.max(0, Math.min(safeMaxValue, Number(value) || 0));
  const progressNodeId = String(nodeIdValue);
  if (!shouldApplyPromptState(progressNodeId, promptId)) {
    return;
  }
  modalNodeBatchProgress.set(progressNodeId, {
    promptId,
    value: safeValue,
    max: safeMaxValue,
    updatedAt: nowMs(),
  });
  ensureAnimationLoop();
  refreshNodeDecorations();
}

/**
 * Record numeric progress for one node and its visible ancestors.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @param {number} value
 * @param {number} maxValue
 * @param {Record<string, unknown>} metadata
 */
function setNodeProgress(nodeIdValue, promptId, value, maxValue, metadata = {}) {
  const safeMaxValue = Math.max(1, Number(maxValue) || 1);
  const safeValue = Math.max(0, Math.min(safeMaxValue, Number(value) || 0));
  const progressNodeIds = [String(nodeIdValue), ...ancestorNodeIds(nodeIdValue)];
  const updatedAt = progressEventTimestampMs();

  for (const progressNodeId of progressNodeIds) {
    if (!shouldApplyPromptState(progressNodeId, promptId)) {
      continue;
    }
    const existingProgress = modalNodeProgress.get(progressNodeId);
    const reportedRate = Number(metadata.tokens_per_second);
    modalNodeProgress.set(progressNodeId, {
      promptId,
      value: safeValue,
      max: safeMaxValue,
      updatedAt,
      stage: String(metadata.stage ?? ""),
      message: String(metadata.message ?? ""),
      unit: String(metadata.unit ?? ""),
      indeterminate: Boolean(metadata.indeterminate),
      preGpu: Boolean(metadata.pre_gpu),
      elapsedSeconds: Number.isFinite(Number(metadata.elapsed_seconds))
        ? Number(metadata.elapsed_seconds)
        : null,
      timeToFirstTokenSeconds: Number.isFinite(Number(metadata.time_to_first_token_seconds))
        ? Number(metadata.time_to_first_token_seconds)
        : null,
      iterationRate: Number.isFinite(reportedRate) && reportedRate >= 0
        ? reportedRate
        : progressIterationRate(
            existingProgress?.promptId === promptId ? existingProgress : null,
            safeValue,
            safeMaxValue,
            updatedAt,
          ),
    });
  }

  ensureAnimationLoop();
  refreshNodeDecorations();
}

/**
 * Record numeric progress for one worker lane on a node and its visible ancestors.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @param {string} laneId
 * @param {number} value
 * @param {number} maxValue
 * @param {number | null | undefined} itemIndex
 * @param {boolean} setupOnly
 */
function setNodeProgressLane(nodeIdValue, promptId, laneId, value, maxValue, itemIndex, setupOnly = false) {
  const safeLaneId = String(laneId ?? "");
  if (!safeLaneId) {
    return;
  }
  const safeNodeIdValue = String(nodeIdValue);
  const safeLaneKey = laneOwnerKey(promptId, safeNodeIdValue, safeLaneId);
  const safeMaxValue = Math.max(1, Number(maxValue) || 1);
  const safeValue = Math.max(0, Math.min(safeMaxValue, Number(value) || 0));
  const updatedAt = progressEventTimestampMs();
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  const previousNodeId = promptState.laneNodeIdsByLane.get(safeLaneKey);
  if (previousNodeId && previousNodeId !== safeNodeIdValue) {
    deleteNodeProgressLane(previousNodeId, promptId, safeLaneId);
  }
  promptState.laneNodeIdsByLane.set(safeLaneKey, safeNodeIdValue);
  const progressNodeIds = [safeNodeIdValue, ...ancestorNodeIds(safeNodeIdValue)];

  for (const progressNodeId of progressNodeIds) {
    if (!shouldApplyPromptState(progressNodeId, promptId)) {
      continue;
    }
    const existingLaneState = modalNodeProgressLanes.get(progressNodeId);
    const laneState =
      existingLaneState?.promptId === promptId
        ? existingLaneState
        : {
            promptId,
            lanes: new Map(),
          };
    const existingLaneProgress = laneState.lanes.get(safeLaneId);
    laneState.lanes.set(safeLaneId, {
      laneId: safeLaneId,
      value: safeValue,
      max: safeMaxValue,
      itemIndex: Number.isFinite(Number(itemIndex)) ? Number(itemIndex) : null,
      updatedAt,
      setupOnly: Boolean(setupOnly),
      iterationRate: setupOnly
        ? null
        : progressIterationRate(
            existingLaneProgress?.setupOnly ? null : existingLaneProgress,
            safeValue,
            safeMaxValue,
            updatedAt,
          ),
    });
    modalNodeProgressLanes.set(progressNodeId, laneState);
  }

  ensureAnimationLoop();
  refreshNodeDecorations();
}

/**
 * Remove one worker lane progress bar from a node and its visible ancestors.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @param {string} laneId
 */
function clearNodeProgressLane(nodeIdValue, promptId, laneId) {
  const safeLaneId = String(laneId ?? "");
  if (!safeLaneId) {
    return;
  }
  const safeNodeIdValue = String(nodeIdValue);
  const safeLaneKey = laneOwnerKey(promptId, safeNodeIdValue, safeLaneId);
  const promptState = modalPromptStates.get(promptId);
  if (promptState?.laneNodeIdsByLane.get(safeLaneKey) === safeNodeIdValue) {
    promptState.laneNodeIdsByLane.delete(safeLaneKey);
  }
  deleteNodeProgressLane(safeNodeIdValue, promptId, safeLaneId);

  stopAnimationLoopIfIdle();
  refreshNodeDecorations();
}

/**
 * Return remote component node ids for an executing proxy node event.
 * @param {string} promptId
 * @param {string} representativeNodeId
 * @returns {string[] | null}
 */
function resolveComponentNodeIds(promptId, representativeNodeId) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    return null;
  }
  const candidateNodeId = String(representativeNodeId);
  if (promptState.componentNodeIdsByMember.has(candidateNodeId)) {
    return promptState.componentNodeIdsByMember.get(candidateNodeId) ?? null;
  }
  if (promptState.remoteNodeIds.includes(candidateNodeId)) {
    return [candidateNodeId];
  }
  return null;
}

/**
 * Return the representative node id for one remote component member when known.
 * @param {string} promptId
 * @param {string} nodeIdValue
 * @returns {string | null}
 */
function resolveComponentRepresentativeNodeId(promptId, nodeIdValue) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    return null;
  }
  return promptState.representativeNodeIdByMember.get(String(nodeIdValue)) ?? null;
}

/**
 * Return the stable prompt-local lane owner key for one component worker lane.
 * @param {string} promptId
 * @param {string} nodeIdValue
 * @param {string} laneId
 * @returns {string}
 */
function laneOwnerKey(promptId, nodeIdValue, laneId) {
  const ownerNodeId = resolveComponentRepresentativeNodeId(promptId, nodeIdValue) ?? String(nodeIdValue);
  return `${ownerNodeId}:${String(laneId)}`;
}

/**
 * Extract remote node ids from a workflow snapshot.
 * @param {object | undefined} workflow
 * @returns {string[]}
 */
function extractRemoteNodeIds(workflow) {
  const remoteNodeIds = [];
  for (const node of workflow?.nodes ?? []) {
    if (serializedRemoteFlag(node)) {
      remoteNodeIds.push(String(node.id));
    }
  }
  return remoteNodeIds;
}

/** Return remote ids used for immediate UI state, including automatic placement. */
function effectiveRemoteNodeIds(prompt, workflow) {
  const preferences = selectedExecutionPreferences(workflow);
  if (preferences.autoPlace) {
    return Object.keys(prompt ?? {});
  }
  return extractRemoteNodeIds(workflow);
}

/**
 * Return the user-visible Modal toggle value from a serialized workflow node.
 * @param {object | undefined} node
 * @returns {boolean}
 */
function serializedRemoteFlag(node) {
  const namedWidgetValue =
    node?.widgets_values_named?.[REMOTE_WIDGET_NAME]
    ?? node?.widgets_values_named?.[LEGACY_REMOTE_WIDGET_NAME];
  if (typeof namedWidgetValue === "boolean") {
    return namedWidgetValue;
  }
  return Boolean(node?.properties?.[REMOTE_PROPERTY]);
}

/** Return whether the live workflow uses connected configuration capacity. */
function usesRemoteExecutionConfigurator() {
  return Boolean(
    findSomethingInAllSubgraphs((graph) =>
      (graph?.nodes ?? []).find(
        (node) => String(node?.comfyClass ?? node?.type ?? "") === "RemoteExecutionConfigurator",
      ),
    ),
  );
}

/**
 * Return the root workflow graph, including subgraphs when available.
 * @returns {LGraph | null}
 */
function rootGraph() {
  return app.rootGraph ?? app.graph?.rootGraph ?? app.graph ?? null;
}

/** Return generic execution preferences persisted in the workflow graph. */
function selectedExecutionPreferences(workflow = null) {
  const workflowContainer = workflow ?? rootGraph();
  const remoteConfig = workflowContainer?.extra?.[WORKFLOW_REMOTE_CONFIG_KEY] ?? {};
  const policy = EXECUTION_POLICIES.some(([value]) => value === remoteConfig[WORKFLOW_REMOTE_POLICY_KEY])
    ? remoteConfig[WORKFLOW_REMOTE_POLICY_KEY]
    : "modal";
  return {
    policy,
    autoPlace: Boolean(remoteConfig[WORKFLOW_REMOTE_AUTO_PLACE_KEY]),
  };
}

/** Persist one provider policy or automatic-placement change on the root graph. */
function setExecutionPreferences(updates) {
  const graph = rootGraph();
  if (!graph) {
    throw new Error("The workflow graph is unavailable.");
  }
  const current = selectedExecutionPreferences(graph);
  const next = { ...current, ...updates };
  graph.extra ||= {};
  graph.extra[WORKFLOW_REMOTE_CONFIG_KEY] ||= {};
  graph.extra[WORKFLOW_REMOTE_CONFIG_KEY][WORKFLOW_REMOTE_POLICY_KEY] = next.policy;
  graph.extra[WORKFLOW_REMOTE_CONFIG_KEY][WORKFLOW_REMOTE_AUTO_PLACE_KEY] = next.autoPlace;
  graph.change?.();
  app.graph?.setDirtyCanvas?.(true, true);
  const label = EXECUTION_POLICIES.find(([value]) => value === next.policy)?.[1] ?? next.policy;
  notifyModal(`Remote execution policy: ${label}; automatic placement ${next.autoPlace ? "on" : "off"}.`);
}

/**
 * Return the workflow-level Modal GPU selection, installing the default when absent.
 * @param {object | null | undefined} workflow
 * @returns {string}
 */
function selectedModalGpu(workflow = null) {
  const workflowContainer = workflow ?? rootGraph();
  const savedGpu = workflowContainer?.extra?.[WORKFLOW_MODAL_CONFIG_KEY]?.[WORKFLOW_MODAL_GPU_KEY];
  if (MODAL_GPU_TYPES.includes(savedGpu)) {
    return savedGpu;
  }
  if (workflowContainer) {
    workflowContainer.extra ||= {};
    workflowContainer.extra[WORKFLOW_MODAL_CONFIG_KEY] ||= {};
    workflowContainer.extra[WORKFLOW_MODAL_CONFIG_KEY][WORKFLOW_MODAL_GPU_KEY] = DEFAULT_MODAL_GPU;
  }
  return DEFAULT_MODAL_GPU;
}

/**
 * Save one Modal GPU selection on the root graph so normal workflow serialization preserves it.
 * @param {string} modalGpu
 */
function setSelectedModalGpu(modalGpu) {
  if (!MODAL_GPU_TYPES.includes(modalGpu)) {
    throw new Error(`Unsupported Modal GPU selection: ${String(modalGpu)}`);
  }
  const graph = rootGraph();
  if (!graph) {
    throw new Error("The workflow graph is unavailable.");
  }
  const previousGpu = selectedModalGpu(graph);
  graph.extra[WORKFLOW_MODAL_CONFIG_KEY][WORKFLOW_MODAL_GPU_KEY] = modalGpu;
  if (previousGpu !== modalGpu) {
    graph.change?.();
    app.graph?.setDirtyCanvas?.(true, true);
  }
  notifyModal(
    `Modal GPU set to ${modalGpu}. The next remote run will rebuild the Modal app if needed.`,
  );
}

/**
 * Copy the live graph GPU selection onto one serialized workflow snapshot.
 * @param {object | null | undefined} workflow
 */
function stampModalGpuOnWorkflow(workflow) {
  if (!workflow || typeof workflow !== "object") {
    return;
  }
  workflow.extra ||= {};
  workflow.extra[WORKFLOW_MODAL_CONFIG_KEY] ||= {};
  workflow.extra[WORKFLOW_MODAL_CONFIG_KEY][WORKFLOW_MODAL_GPU_KEY] = selectedModalGpu();
}

/**
 * Look up a graph-local node id without assuming whether ids are numeric or strings.
 * @param {LGraph | null | undefined} graph
 * @param {string} id
 * @returns {LGraphNode | null}
 */
function getGraphNodeById(graph, id) {
  if (!graph || id == null) {
    return null;
  }
  const directMatch = graph.getNodeById?.(id) ?? null;
  if (directMatch) {
    return directMatch;
  }
  const numericId = Number(id);
  if (Number.isFinite(numericId)) {
    return graph.getNodeById?.(numericId) ?? null;
  }
  return null;
}

/**
 * Search every live subgraph for one matching value.
 * @param {(graph: LGraph) => any} matcher
 * @returns {any}
 */
function findSomethingInAllSubgraphs(matcher) {
  const graph = rootGraph();
  if (!graph) {
    return null;
  }

  const visitedGraphs = new Set();
  const visitGraph = (candidateGraph) => {
    if (!candidateGraph || visitedGraphs.has(candidateGraph)) {
      return null;
    }
    visitedGraphs.add(candidateGraph);
    const match = matcher(candidateGraph);
    if (match) {
      return match;
    }
    for (const node of candidateGraph.nodes ?? []) {
      const nestedMatch = visitGraph(node?.subgraph);
      if (nestedMatch) {
        return nestedMatch;
      }
    }
    if (candidateGraph.subgraphs?.values) {
      for (const subgraph of candidateGraph.subgraphs.values()) {
        const nestedMatch = visitGraph(subgraph);
        if (nestedMatch) {
          return nestedMatch;
        }
      }
    }
    return null;
  };
  return visitGraph(graph);
}

/**
 * Return all live workflow nodes across the root graph and any nested subgraphs.
 * @returns {LGraphNode[]}
 */
function allWorkflowNodes() {
  const graph = rootGraph();
  if (!graph) {
    return [];
  }

  const visitedGraphs = new Set();
  const nodes = [];
  const visitGraph = (candidateGraph) => {
    if (!candidateGraph || visitedGraphs.has(candidateGraph)) {
      return;
    }
    visitedGraphs.add(candidateGraph);
    for (const node of candidateGraph.nodes ?? []) {
      nodes.push(node);
      visitGraph(node?.subgraph);
    }
    if (candidateGraph.subgraphs?.values) {
      for (const subgraph of candidateGraph.subgraphs.values()) {
        visitGraph(subgraph);
      }
    }
  };
  visitGraph(graph);
  return nodes;
}

/**
 * Return whether one live graph node is the workflow capacity configurator.
 * @param {LGraphNode | null | undefined} node
 * @returns {boolean}
 */
function isRemoteExecutionConfiguratorNode(node) {
  return String(node?.comfyClass ?? node?.type ?? "") === "RemoteExecutionConfigurator";
}

/**
 * Return the configurator node id from one serialized API prompt.
 * @param {Record<string, any>} prompt
 * @returns {string | null}
 */
function serializedRemoteConfiguratorNodeId(prompt) {
  const matches = Object.entries(prompt ?? {}).filter(
    ([, promptNode]) => promptNode?.class_type === "RemoteExecutionConfigurator",
  );
  return matches.length === 1 ? String(matches[0][0]) : null;
}

/**
 * Resolve and lazily mount the configurator panel for one serialized node id.
 * @param {string | null | undefined} configuratorNodeId
 * @returns {Record<string, any> | null}
 */
function remoteConfiguratorPanel(configuratorNodeId) {
  const safeNodeId = String(configuratorNodeId ?? "");
  if (!safeNodeId) {
    return null;
  }
  const existing = remoteConfiguratorPanels.get(safeNodeId);
  if (existing) {
    return existing;
  }
  const node = allWorkflowNodes().find(
    (candidate) => nodeId(candidate) === safeNodeId && isRemoteExecutionConfiguratorNode(candidate),
  );
  if (!node) {
    return null;
  }
  mountRemoteExecutionConfiguratorPanel(node);
  return remoteConfiguratorPanels.get(safeNodeId) ?? null;
}

/**
 * Return a compact status sentence for the configurator panel.
 * @param {Record<string, any>} state
 * @returns {string}
 */
function remoteConfiguratorStatusMessage(state) {
  if (state?.statusMessage) {
    return String(state.statusMessage);
  }
  if (state?.phase === STATE_SETUP) {
    return "Preparing remote workflow";
  }
  if (state?.phase === STATE_STARTING) {
    return "Starting remote capacity";
  }
  if (state?.phase === STATE_WAITING) {
    return "Waiting for remote capacity";
  }
  if (state?.phase === EXECUTION_PHASE) {
    return "Remote workflow is running";
  }
  if (state?.phase === STATE_FINALIZING) {
    return "Receiving remote outputs";
  }
  if (state?.phase === STATE_CANCELLING) {
    return "Cancelling remote workflow";
  }
  if (state?.phase === STATE_ERROR) {
    return "Remote workflow failed";
  }
  if (state?.phase === STATE_COMPLETE) {
    return "Remote workflow complete";
  }
  return "Ready to plan remote execution";
}

/**
 * Update one mounted configurator's status without changing its placement table.
 * @param {Record<string, any>} panel
 * @param {Record<string, any>} state
 */
function renderRemoteConfiguratorStatus(panel, state) {
  if (!panel) {
    return;
  }
  const phase = String(state?.phase ?? "idle");
  panel.root.dataset.phase = phase;
  panel.statusText.textContent = remoteConfiguratorStatusMessage(state);
  const total = Math.max(0, Number(state?.statusTotal ?? state?.batchMax ?? 0));
  const current = Math.max(
    0,
    Math.min(total, Number(state?.statusCurrent ?? state?.batchValue ?? 0)),
  );
  panel.progress.hidden = total <= 1;
  panel.progressFill.style.width = total > 1 ? `${(current / total) * 100}%` : "0%";
  panel.progressValue.textContent = total > 1 ? `${Math.round(current)}/${Math.round(total)}` : "";
}

/**
 * Render the completed provider-neutral placement plan as a node-local table.
 * @param {Record<string, any>} panel
 * @param {Record<string, Record<string, any>>} assignments
 * @param {Array<Record<string, any>>} configurations
 */
function renderRemoteConfiguratorPlan(panel, assignments, configurations) {
  if (!panel) {
    return;
  }
  const configurationById = new Map(
    (configurations ?? []).map((configuration) => [
      String(configuration?.configuration_id ?? ""),
      configuration,
    ]),
  );
  const rows = Object.entries(assignments ?? {}).sort(([left], [right]) =>
    String(left).localeCompare(String(right), undefined, { numeric: true }),
  );
  panel.tableBody.replaceChildren();
  rows.forEach(([representativeNodeId, assignment], index) => {
    const row = document.createElement("tr");
    const componentCell = document.createElement("td");
    const targetCell = document.createElement("td");
    const estimateCell = document.createElement("td");
    const nodeIds = Array.isArray(assignment?.node_ids)
      ? assignment.node_ids.map((value) => String(value))
      : [String(representativeNodeId)];
    componentCell.textContent = `Component ${index + 1}`;
    componentCell.title = `Nodes ${nodeIds.join(", ")}`;
    const configurationId = String(assignment?.configuration_id ?? "");
    const configuration = configurationById.get(configurationId);
    const configurationName = String(configuration?.display_name ?? configurationId).trim();
    const environmentLabel = remoteExecutionEnvironmentLabel(
      assignment,
      configurationById,
      configuration?.gpu_type ?? null,
    );
    targetCell.textContent = configurationName
      ? `${configurationName} · ${environmentLabel}`
      : environmentLabel;
    const predictedCost = assignment?.predicted_cost_usd == null
      ? Number.NaN
      : Number(assignment.predicted_cost_usd);
    const predictedSeconds = Number(assignment?.predicted_completion_seconds);
    const estimateParts = [];
    if (Number.isFinite(predictedSeconds) && predictedSeconds > 0) {
      estimateParts.push(`${Math.round(predictedSeconds)}s`);
    }
    if (Number.isFinite(predictedCost) && predictedCost >= 0) {
      estimateParts.push(`$${predictedCost.toFixed(4)}`);
    }
    estimateCell.textContent = estimateParts.join(" · ") || "—";
    row.append(componentCell, targetCell, estimateCell);
    panel.tableBody.appendChild(row);
  });
  panel.emptyText.hidden = rows.length > 0;
  panel.table.hidden = rows.length === 0;
  panel.minHeight = 118 + Math.max(1, rows.length) * 31;
  const currentWidth = Number(panel.node.size?.[0]) || 0;
  const currentHeight = Number(panel.node.size?.[1]) || 0;
  panel.node.setSize?.([
    Math.max(currentWidth, 500),
    Math.max(currentHeight, panel.minHeight),
  ]);
  panel.node.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Associate one prompt with its configurator and reset the out-of-band display.
 * @param {string} promptId
 * @param {string | null | undefined} configuratorNodeId
 */
function registerPromptConfigurator(promptId, configuratorNodeId) {
  const safeNodeId = String(configuratorNodeId ?? "");
  const promptState = ensurePromptState(promptId);
  if (!promptState || !safeNodeId) {
    return;
  }
  promptState.configuratorNodeId = safeNodeId;
  const panel = remoteConfiguratorPanel(safeNodeId);
  if (!panel) {
    return;
  }
  const isNewPrompt = panel.promptId !== promptId;
  panel.promptId = promptId;
  if (isNewPrompt) {
    renderRemoteConfiguratorPlan(
      panel,
      promptState.remoteExecutionPlanAssignments,
      promptState.remoteExecutionConfigurations,
    );
    renderRemoteConfiguratorStatus(panel, {
      phase: STATE_SETUP,
      statusMessage: "Preparing remote workflow",
    });
  }
}

/**
 * Retain and render one provider-neutral plan independently of queue completion.
 * @param {string} promptId
 * @param {Record<string, Record<string, any>>} assignments
 * @param {Array<Record<string, any>>} configurations
 */
function registerRemoteConfiguratorPlan(promptId, assignments, configurations) {
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  promptState.remoteExecutionPlanAssignments = { ...(assignments ?? {}) };
  promptState.remoteExecutionConfigurations = [...(configurations ?? [])];
  registerPromptExecutionAssignments(promptId, assignments ?? {});
  const panel = remoteConfiguratorPanel(promptState.configuratorNodeId);
  if (!panel) {
    return;
  }
  panel.promptId = promptId;
  renderRemoteConfiguratorPlan(
    panel,
    promptState.remoteExecutionPlanAssignments,
    promptState.remoteExecutionConfigurations,
  );
}

/** Restore the newest retained prompt onto a newly mounted configurator panel. */
function restoreRemoteConfiguratorPanel(panel) {
  const nodeIdentifier = nodeId(panel?.node);
  const candidates = Array.from(modalPromptStates.entries())
    .filter(([, state]) => state?.configuratorNodeId === nodeIdentifier)
    .sort(([, left], [, right]) => Number(right?.startedAt ?? 0) - Number(left?.startedAt ?? 0));
  const newest = candidates[0];
  if (!newest) {
    return;
  }
  const [promptId, promptState] = newest;
  panel.promptId = promptId;
  renderRemoteConfiguratorPlan(
    panel,
    promptState.remoteExecutionPlanAssignments,
    promptState.remoteExecutionConfigurations,
  );
  renderRemoteConfiguratorStatus(panel, modalGlobalStatusStates.get(promptId) ?? {});
}

/**
 * Refresh the node-local status from the same durable state used by the global pill.
 * @param {string} promptId
 */
function refreshRemoteConfiguratorPanelForPrompt(promptId) {
  const promptState = modalPromptStates.get(promptId);
  const panel = remoteConfiguratorPanel(promptState?.configuratorNodeId);
  if (!panel || panel.promptId !== promptId) {
    return;
  }
  renderRemoteConfiguratorStatus(panel, modalGlobalStatusStates.get(promptId) ?? {});
}

/**
 * Preserve a terminal outcome on the configurator after prompt state is cleaned up.
 * @param {string} promptId
 * @param {string} phase
 * @param {string | null} message
 */
function setRemoteConfiguratorTerminalStatus(promptId, phase, message = null) {
  const promptState = modalPromptStates.get(promptId);
  const panel = remoteConfiguratorPanel(promptState?.configuratorNodeId);
  if (!panel || panel.promptId !== promptId) {
    return;
  }
  renderRemoteConfiguratorStatus(panel, { phase, statusMessage: message });
}

/**
 * Remove Configurator DOM widgets left behind by an older extension module.
 * @param {LGraphNode} node
 */
function removeStaleRemoteConfiguratorWidgets(node) {
  const staleWidgets = (node.widgets ?? []).filter(
    (widget) => String(widget?.name ?? "") === "remote_execution_plan",
  );
  for (const widget of staleWidgets) {
    widget?.element?.remove?.();
    if (typeof node.removeWidget === "function") {
      node.removeWidget(widget);
      continue;
    }
    widget?.onRemove?.();
    const widgetIndex = node.widgets?.indexOf(widget) ?? -1;
    if (widgetIndex >= 0) {
      node.widgets.splice(widgetIndex, 1);
    }
  }
}

/**
 * Mount the non-serialized planning and status surface on the configurator node.
 * @param {LGraphNode} node
 */
function mountRemoteExecutionConfiguratorPanel(node) {
  const safeNodeId = nodeId(node);
  if (
    !isRemoteExecutionConfiguratorNode(node) ||
    (node.__remoteConfiguratorPanelMounted && remoteConfiguratorPanels.has(safeNodeId)) ||
    typeof node.addDOMWidget !== "function" ||
    typeof document === "undefined"
  ) {
    return;
  }
  removeStaleRemoteConfiguratorWidgets(node);
  const root = document.createElement("div");
  root.className = "comfy-remote-configurator-panel";
  root.innerHTML = `
    <div class="remote-configurator-status">
      <span class="remote-configurator-dot"></span>
      <span class="remote-configurator-status-text">Ready to plan remote execution</span>
    </div>
    <div class="remote-configurator-progress" hidden>
      <span class="remote-configurator-progress-fill"></span>
      <span class="remote-configurator-progress-value"></span>
    </div>
    <div class="remote-configurator-plan-title">Execution plan</div>
    <div class="remote-configurator-empty">The selected environments will appear here after planning.</div>
    <table class="remote-configurator-table" hidden>
      <thead><tr><th>Component</th><th>Target</th><th>Estimate</th></tr></thead>
      <tbody></tbody>
    </table>`;
  const panel = {
    node,
    root,
    promptId: null,
    minHeight: 149,
    statusText: root.querySelector(".remote-configurator-status-text"),
    progress: root.querySelector(".remote-configurator-progress"),
    progressFill: root.querySelector(".remote-configurator-progress-fill"),
    progressValue: root.querySelector(".remote-configurator-progress-value"),
    emptyText: root.querySelector(".remote-configurator-empty"),
    table: root.querySelector(".remote-configurator-table"),
    tableBody: root.querySelector("tbody"),
  };
  let widget;
  try {
    widget = node.addDOMWidget(
      "remote_execution_plan",
      "remote-execution-plan",
      root,
      {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => panel.minHeight,
      },
    );
  } catch (error) {
    node.__remoteConfiguratorPanelMounted = false;
    console.warn("Unable to mount the remote execution configurator panel.", error);
    return;
  }
  if (widget) {
    widget.serialize = false;
  }
  node.setSize?.([
    Math.max(Number(node.size?.[0]) || 0, 500),
    Math.max(Number(node.size?.[1]) || 0, panel.minHeight),
  ]);
  remoteConfiguratorPanels.set(safeNodeId, panel);
  node.__remoteConfiguratorPanelMounted = true;
  restoreRemoteConfiguratorPanel(panel);
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function onRemoved() {
    remoteConfiguratorPanels.delete(nodeId(this));
    return originalOnRemoved?.apply(this, arguments);
  };
}

/**
 * Return the workflow node that owns one nested subgraph graph id.
 * @param {number | string | undefined} subgraphId
 * @returns {LGraphNode | null}
 */
function findContainingSubgraphNode(subgraphId) {
  if (subgraphId == null) {
    return null;
  }
  return (
    findSomethingInAllSubgraphs((graph) =>
      (graph?.nodes ?? []).find(
        (candidate) =>
          typeof candidate?.isSubgraphNode === "function" &&
          candidate.isSubgraphNode() &&
          candidate.subgraph?.id === subgraphId,
      ),
    ) ?? null
  );
}

/**
 * Return one node's composed workflow path, including any subgraph ancestors.
 * @param {LGraphNode} node
 * @returns {string}
 */
function workflowNodePath(node) {
  const pathSegments = [String(node?.id ?? "")];
  let currentGraph = node?.graph ?? null;
  const currentRootGraph = rootGraph();

  while (currentGraph && currentRootGraph && currentGraph !== currentRootGraph) {
    const parentNode = findContainingSubgraphNode(currentGraph.id);
    if (!parentNode) {
      break;
    }
    pathSegments.unshift(String(parentNode.id));
    currentGraph = parentNode.graph ?? null;
  }

  return pathSegments.filter(Boolean).join(":");
}

/**
 * Rebuild the workflow-level index of Modal-enabled leaves beneath each subgraph node.
 */
function rebuildRemoteDescendantIndex() {
  modalRemoteDescendantNodeIdsByAncestor.clear();
  for (const node of allWorkflowNodes()) {
    if (!isRemoteNode(node)) {
      continue;
    }
    const remoteNodePath = workflowNodePath(node);
    for (const ancestorNodeId of ancestorNodeIds(remoteNodePath)) {
      if (!modalRemoteDescendantNodeIdsByAncestor.has(ancestorNodeId)) {
        modalRemoteDescendantNodeIdsByAncestor.set(ancestorNodeId, new Set());
      }
      modalRemoteDescendantNodeIdsByAncestor.get(ancestorNodeId).add(remoteNodePath);
    }
  }
}

/**
 * Rebuild descendant containment after ComfyUI completes a graph mutation.
 */
function scheduleRemoteDescendantIndexRebuild() {
  if (remoteDescendantIndexRebuildScheduled) {
    return;
  }
  remoteDescendantIndexRebuildScheduled = true;
  setTimeout(() => {
    remoteDescendantIndexRebuildScheduled = false;
    rebuildRemoteDescendantIndex();
    refreshNodeDecorations();
  }, 0);
}

/**
 * Resolve a composed workflow path like `24:23` to a live node instance.
 * @param {string} workflowPath
 * @returns {LGraphNode | null}
 */
function findNodeByWorkflowPath(workflowPath) {
  const pathSegments = String(workflowPath)
    .split(":")
    .filter(Boolean);
  if (pathSegments.length === 0) {
    return null;
  }

  let currentGraph = rootGraph();
  let currentNode = null;
  for (const pathSegment of pathSegments) {
    currentNode = getGraphNodeById(currentGraph, pathSegment);
    if (!currentNode) {
      return null;
    }
    currentGraph = currentNode.subgraph ?? null;
  }
  return currentNode;
}

/**
 * Return the workflow-node paths that a context-menu action should expand from.
 * @param {LGraphNode} node
 * @returns {string[]}
 */
function selectedWorkflowNodePaths(node) {
  const selectedNodes = Object.values(app.canvas?.selected_nodes ?? {}).filter(
    (candidate) => candidate?.graph === node?.graph && isEligibleNode(candidate),
  );
  if (selectedNodes.some((candidate) => candidate === node) && selectedNodes.length > 1) {
    return selectedNodes.map((candidate) => workflowNodePath(candidate));
  }
  return [workflowNodePath(node)];
}

/**
 * Return the graph snapshot shape ComfyUI already uses for queue submission.
 * @returns {Promise<{ output: object, workflow: object }>}
 */
async function serializeCurrentGraphForModal() {
  if (typeof app.graphToPrompt !== "function") {
    throw new Error("ComfyUI graph serialization is unavailable.");
  }
  const prompt = await app.graphToPrompt();
  if (!prompt?.output || !prompt?.workflow) {
    throw new Error("ComfyUI did not return prompt and workflow data.");
  }
  return {
    output: prompt.output,
    workflow: prompt.workflow,
  };
}

/**
 * Show a short frontend notification without taking over the whole UI.
 * @param {string} value
 */
function notifyModal(value) {
  dispatchSyntheticApiEvent("notification", {
    id: `modal-analysis-${Date.now()}`,
    value,
  });
}

/**
 * Request one local Modal maintenance action from the backend.
 * @param {string} route
 * @param {string} successMessage
 */
async function requestModalMaintenance(route, successMessage) {
  if (typeof api.fetchApi !== "function") {
    throw new Error("ComfyUI API fetch is unavailable.");
  }
  const response = await api.fetchApi(route, { method: "POST" });
  if (response.status !== 200) {
    throw new PromptExecutionError(await response.json());
  }
  notifyModal(successMessage);
}

/**
 * Apply the remote marker value to the workflow nodes named by composed workflow paths.
 * @param {string[]} workflowNodePaths
 * @param {boolean} value
 * @returns {number}
 */
function setWorkflowNodePathsRemote(workflowNodePaths, value) {
  let appliedCount = 0;
  for (const workflowPath of workflowNodePaths) {
    const node = findNodeByWorkflowPath(workflowPath);
    if (!node) {
      console.warn("Unable to find Modal workflow node path in the live graph.", workflowPath);
      continue;
    }
    setRemoteFlag(node, value);
    appliedCount += 1;
  }
  return appliedCount;
}

/**
 * Set Modal on every currently eligible node in the live workflow.
 * @param {boolean} value
 * @returns {number}
 */
function setAllEligibleWorkflowNodesRemote(value) {
  let appliedCount = 0;
  for (const node of allWorkflowNodes()) {
    if (!isEligibleNode(node)) {
      continue;
    }
    setRemoteFlag(node, value);
    appliedCount += 1;
  }
  refreshNodeDecorations();
  const actionLabel = value ? "Enabled" : "Disabled";
  notifyModal(
    appliedCount > 0
      ? `${actionLabel} Modal on ${appliedCount} node${appliedCount === 1 ? "" : "s"}.`
      : "No Modal-eligible nodes were found in the workflow.",
  );
  return appliedCount;
}

/**
 * Request required upstream nodes from the backend and set their Modal state in the UI.
 * @param {LGraphNode} node
 * @param {boolean} value
 */
async function analyzeAndSetUpstreamRemoteNodes(node, value) {
  const seedNodeIds = selectedWorkflowNodePaths(node);
  const graphSnapshot = await serializeCurrentGraphForModal();
  const response = await api.fetchApi(MODAL_ANALYZE_ROUTE, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: graphSnapshot.output,
      workflow: graphSnapshot.workflow,
      seed_node_ids: seedNodeIds,
    }),
  });
  if (response.status !== 200) {
    throw new PromptExecutionError(await response.json());
  }

  const result = await response.json();
  const resolvedWorkflowNodePaths = result.resolved_workflow_node_paths ?? [];
  const addedWorkflowNodePaths = result.added_workflow_node_paths ?? [];
  const appliedCount = setWorkflowNodePathsRemote(resolvedWorkflowNodePaths, value);
  setSandwichedLocalNodeIds(value ? (result.sandwiched_local_node_ids ?? []) : []);
  refreshNodeDecorations();

  if (!value) {
    notifyModal(
      appliedCount > 0
        ? `Disabled Modal on ${appliedCount} upstream node${appliedCount === 1 ? "" : "s"}.`
        : "Modal analysis finished, but no matching live nodes were found to update.",
    );
    return;
  }
  if (addedWorkflowNodePaths.length > 0) {
    notifyModal(`Enabled Modal on ${addedWorkflowNodePaths.length} upstream node${addedWorkflowNodePaths.length === 1 ? "" : "s"}.`);
    return;
  }
  notifyModal(
    appliedCount > 0
      ? "No extra upstream Modal nodes were required."
      : "Modal analysis finished, but no matching live nodes were found to update.",
  );
}

/**
 * Return whether one node definition should expose Modal UI affordances.
 * @param {object | undefined} nodeData
 * @returns {boolean}
 */
function isEligibleNodeDef(nodeData) {
  return (
    Boolean(nodeData?.name) &&
    !String(nodeData.name).startsWith(INTERNAL_NODE_PREFIX) &&
    !LOCAL_MODAL_NODE_IDS.has(String(nodeData.name))
  );
}

/**
 * Inject the Modal context-menu entry on a node type prototype.
 * @param {typeof LGraphNode} nodeType
 * @param {object | undefined} nodeData
 */
function installModalContextMenu(nodeType, nodeData) {
  if (!isEligibleNodeDef(nodeData) || nodeType?.prototype?.__modalContextMenuInjected) {
    return;
  }

  const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function getExtraMenuOptionsWithModalAnalysis(
    canvas,
    options,
  ) {
    const menuOptions = originalGetExtraMenuOptions?.call(this, canvas, options) ?? options ?? [];
    const targetOptions = options ?? menuOptions;
    if (!Array.isArray(targetOptions)) {
      return menuOptions;
    }

    const selectedNodePaths = selectedWorkflowNodePaths(this);
    const enableUpstreamMenuItemLabel =
      selectedNodePaths.length > 1
        ? "Enable on Upstream Nodes for Selection"
        : "Enable on Upstream Nodes";
    const disableUpstreamMenuItemLabel =
      selectedNodePaths.length > 1
        ? "Disable on Upstream Nodes for Selection"
        : "Disable on Upstream Nodes";
    if (!targetOptions.some((option) => option?.content === "Remote Execution")) {
      const executionPreferences = selectedExecutionPreferences();
      const configuredCapacity = usesRemoteExecutionConfigurator();
      const executionPolicyLabel =
        EXECUTION_POLICIES.find(([value]) => value === executionPreferences.policy)?.[1]
        ?? executionPreferences.policy;
      targetOptions.push(null, {
        content: "Remote Execution",
        has_submenu: true,
        submenu: {
          options: [
            ...(configuredCapacity
              ? [{
                content: "Providers and capacity come from Remote Execution Configurator",
                disabled: true,
              }]
              : [
                {
                  content: `Legacy provider policy (${executionPolicyLabel})`,
                  disabled: true,
                },
                ...EXECUTION_POLICIES.map(([value, label]) => ({
                  content: label,
                  checked: executionPreferences.policy === value,
                  callback: () => setExecutionPreferences({ policy: value }),
                })),
              ]),
            null,
            {
              content: "Automatically place eligible nodes",
              checked: executionPreferences.autoPlace,
              callback: () => setExecutionPreferences({
                autoPlace: !selectedExecutionPreferences().autoPlace,
              }),
            },
            null,
            {
              content: "Manage SSH hosts…",
              callback: () => {
                void openRemoteEnvironmentManager().catch((error) => {
                  console.error("Remote environment manager failed.", error);
                  notifyModal(`Remote environment manager failed: ${String(error?.message ?? error)}`);
                });
              },
            },
          ],
        },
      });
    }
    if (!targetOptions.some((option) => option?.content === "Remote Execution Tools")) {
      const currentModalGpu = selectedModalGpu();
      const configuredCapacity = usesRemoteExecutionConfigurator();
      targetOptions.push(null, {
        content: "Remote Execution Tools",
        has_submenu: true,
        submenu: {
          options: [
            ...(configuredCapacity
              ? []
              : [
                {
                  content: `Legacy Modal GPU (${currentModalGpu})`,
                  disabled: true,
                },
                ...MODAL_GPU_TYPES.map((modalGpu) => ({
                  content: modalGpu,
                  checked: modalGpu === currentModalGpu,
                  callback: () => setSelectedModalGpu(modalGpu),
                })),
                null,
              ]),
            {
              content: enableUpstreamMenuItemLabel,
              callback: () => {
                void analyzeAndSetUpstreamRemoteNodes(this, true).catch((error) => {
                  console.error("Modal remote-node analysis failed.", error);
                  notifyModal(`Modal remote-node analysis failed: ${String(error?.message ?? error)}`);
                });
              },
            },
            {
              content: disableUpstreamMenuItemLabel,
              callback: () => {
                void analyzeAndSetUpstreamRemoteNodes(this, false).catch((error) => {
                  console.error("Modal remote-node analysis failed.", error);
                  notifyModal(`Modal remote-node analysis failed: ${String(error?.message ?? error)}`);
                });
              },
            },
            null,
            {
              content: "Enable All Nodes",
              callback: () => {
                setAllEligibleWorkflowNodesRemote(true);
              },
            },
            {
              content: "Disable All Nodes",
              callback: () => {
                setAllEligibleWorkflowNodesRemote(false);
              },
            },
            {
              content: "Delete Modal Caches",
              callback: () => {
                void requestModalMaintenance(
                  MODAL_DELETE_CACHES_ROUTE,
                  "Deleted Modal caches.",
                ).catch((error) => {
                  console.error("Modal cache deletion failed.", error);
                  notifyModal(`Modal cache deletion failed: ${String(error?.message ?? error)}`);
                });
              },
            },
            {
              content: "Delete Modal Volume",
              callback: () => {
                if (typeof window !== "undefined" && typeof window.confirm === "function") {
                  const confirmed = window.confirm("Delete the configured Modal volume?");
                  if (!confirmed) {
                    return;
                  }
                }
                void requestModalMaintenance(
                  MODAL_DELETE_VOLUME_ROUTE,
                  "Deleted Modal volume.",
                ).catch((error) => {
                  console.error("Modal volume deletion failed.", error);
                  notifyModal(`Modal volume deletion failed: ${String(error?.message ?? error)}`);
                });
              },
            },
          ],
        },
      });
    }

    return menuOptions;
  };
  nodeType.prototype.__modalContextMenuInjected = true;
}

/**
 * Update the node state and redraw the canvas.
 * @param {LGraphNode} node
 * @param {boolean} value
 */
function setRemoteFlag(node, value) {
  node.properties ||= {};
  const enabled = Boolean(value);
  node.properties[REMOTE_PROPERTY] = enabled;
  if (node.__modalToggleWidget) {
    node.__modalToggleWidget.value = enabled;
  }
  rebuildRemoteDescendantIndex();
  clearSandwichedLocalNodeWarnings();
  refreshNodeDecorations();
}

/**
 * Reconcile a restored node property with the toggle value displayed to the user.
 * @param {LGraphNode} node
 */
function synchronizeRemoteFlagFromWidget(node) {
  if (!isEligibleNode(node) || !node.__modalToggleWidget) {
    return;
  }
  const enabled = Boolean(node.__modalToggleWidget.value);
  node.properties ||= {};
  if (node.properties[REMOTE_PROPERTY] === enabled) {
    return;
  }
  node.properties[REMOTE_PROPERTY] = enabled;
  rebuildRemoteDescendantIndex();
  clearSandwichedLocalNodeWarnings();
  refreshNodeDecorations();
}

/**
 * Return the current remote visual state for a node.
 * @param {LGraphNode} node
 * @returns {{ phase: string, promptId: string } | null}
 */
function getRemoteVisualState(node) {
  const visualNodeId = workflowNodePath(node) || nodeId(node);
  const storedState = modalNodeStates.get(visualNodeId) ?? null;
  const promptContainerState = storedState?.promptId
    ? remoteContainerVisualState(storedState.promptId, visualNodeId, storedState.errorMessage)
    : null;
  const staticRemoteDescendantCount = hasRemoteDescendants(node)
    ? (modalRemoteDescendantNodeIdsByAncestor.get(visualNodeId)?.size ?? 0)
    : 0;
  const state =
    promptContainerState ??
    storedState ??
    (staticRemoteDescendantCount > 0
      ? {
          phase: "idle",
          promptId: null,
          isRemoteContainer: true,
          isMixedRemoteContainer: false,
          phaseCounts: {},
          remoteDescendantCount: staticRemoteDescendantCount,
        }
      : null);
  if (!state?.promptId) {
    return state;
  }
  const promptState = modalPromptStates.get(state.promptId);
  const progressState = nodeProgressState(visualNodeId, state.promptId);
  const progressLanes = nodeProgressLanes(visualNodeId, state.promptId);
  const cachedState = nodeCachedState(visualNodeId, state.promptId);
  const batchProgressState = modalNodeBatchProgress.get(visualNodeId) ?? null;
  const hasLiveProgress = hasLiveNodeProgress(visualNodeId, state.promptId);
  const executionAssignment =
    promptState?.executionAssignmentByMember.get(visualNodeId) ?? null;
  return {
    ...state,
    phase: deriveRemoteNodePhase(state.phase, hasLiveProgress),
    isActiveRemoteNode: hasLiveProgress || promptState?.activeNodeId === visualNodeId,
    isActiveComponentMember: isNodeInActiveComponent(state.promptId, visualNodeId),
    isCachedRemoteNode: Boolean(cachedState),
    componentLabel: state.isRemoteContainer
      ? null
      : (promptState?.componentLabelByMember.get(visualNodeId) ?? null),
    cachedAt: cachedState?.cachedAt ?? null,
    progress: progressState,
    batchProgress: batchProgressState?.promptId === state.promptId ? batchProgressState : null,
    progressLanes,
    scheduledEnvironmentCount: promptState?.scheduledEnvironmentCount ?? 0,
    executionLocation: executionAssignment?.location
      ? {
          provider: executionAssignment.provider,
          label: executionAssignment.location,
        }
      : null,
  };
}

/**
 * Return the remote node ids that currently have live numeric or lane progress.
 * @param {string} promptId
 * @returns {string[]}
 */
function activeProgressNodeIds(promptId) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    return [];
  }
  return promptState.remoteNodeIds.filter((candidateNodeId) =>
    hasLiveNodeProgress(candidateNodeId, promptId),
  );
}

/**
 * Return whether one visible node belongs to any component that is currently executing.
 * @param {string} promptId
 * @param {string} nodeIdValue
 * @returns {boolean}
 */
function isNodeInActiveComponent(promptId, nodeIdValue) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    return false;
  }

  const activeComponentNodeIds = new Set();
  const promptActiveNodeId = promptState.activeNodeId;
  if (promptActiveNodeId) {
    const promptActiveComponentNodeIds = resolveComponentNodeIds(promptId, promptActiveNodeId);
    if (promptActiveComponentNodeIds?.length) {
      for (const componentNodeId of promptActiveComponentNodeIds) {
        activeComponentNodeIds.add(String(componentNodeId));
      }
    } else {
      activeComponentNodeIds.add(String(promptActiveNodeId));
    }
  }

  for (const liveProgressNodeId of activeProgressNodeIds(promptId)) {
    const liveProgressComponentNodeIds = resolveComponentNodeIds(promptId, liveProgressNodeId);
    if (liveProgressComponentNodeIds?.length) {
      for (const componentNodeId of liveProgressComponentNodeIds) {
        activeComponentNodeIds.add(String(componentNodeId));
      }
    } else {
      activeComponentNodeIds.add(String(liveProgressNodeId));
    }
  }

  if (activeComponentNodeIds.size === 0) {
    return false;
  }

  const safeNodeIdValue = String(nodeIdValue);
  if (activeComponentNodeIds.has(safeNodeIdValue)) {
    return true;
  }

  const descendantNodeIds = promptState.descendantNodeIdsByAncestor.get(safeNodeIdValue);
  if (!descendantNodeIds || descendantNodeIds.size === 0) {
    return false;
  }
  return Array.from(activeComponentNodeIds).some((componentNodeId) => descendantNodeIds.has(componentNodeId));
}

/**
 * Return preparation-phase colors for one remote visual state.
 * @param {Record<string, any> | null} state
 * @param {number} elapsed
 * @returns {{ borderColor: string, shadowColor: string, fillColor: string | null } | null}
 */
function remotePreparationPalette(state, elapsed) {
  let borderColor;
  let shadowColor;
  let fillColor = null;

  if (state?.phase === STATE_SETUP) {
    const pulse = (Math.sin(elapsed * 5) + 1) / 2;
    borderColor = `${SETUP_BORDER_COLOR}${Math.round((0.65 + pulse * 0.35) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(245, 158, 11, ${0.25 + pulse * 0.35})`;
  } else if (state?.phase === STATE_STARTING) {
    const pulse = (Math.sin(elapsed * 8) + 1) / 2;
    borderColor = `${STARTING_BORDER_COLOR}${Math.round((0.58 + pulse * 0.42) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(234, 179, 8, ${0.2 + pulse * 0.42})`;
    fillColor = `rgba(250, 204, 21, ${0.08 + pulse * 0.08})`;
  } else if (state?.phase === STATE_READY) {
    const pulseRate = state?.isCachedRemoteNode ? 2 : 6;
    const pulse = (Math.sin(elapsed * pulseRate) + 1) / 2;
    borderColor = state?.isActiveComponentMember
      ? READY_ACTIVE_COMPONENT_BORDER_COLOR
      : READY_INACTIVE_COMPONENT_BORDER_COLOR;
    shadowColor = state?.isActiveComponentMember
      ? `rgba(34, 197, 94, ${0.24 + pulse * 0.18})`
      : `rgba(22, 101, 52, ${0.18 + pulse * 0.12})`;
    fillColor = state?.isActiveComponentMember
      ? `rgba(134, 239, 172, ${0.1 + pulse * 0.07})`
      : `rgba(74, 222, 128, ${0.06 + pulse * 0.04})`;
  } else {
    return null;
  }

  return { borderColor, shadowColor, fillColor };
}

/**
 * Return execution-phase colors for one remote visual state.
 * @param {Record<string, any> | null} state
 * @param {number} elapsed
 * @returns {{ borderColor: string, shadowColor: string, fillColor: string | null } | null}
 */
function remoteExecutionPalette(state, elapsed) {
  let borderColor;
  let shadowColor;
  let fillColor = null;

  if (state?.phase === STATE_ACTIVE) {
    const pulse = (Math.sin(elapsed * 7) + 1) / 2;
    borderColor = `${ACTIVE_BORDER_COLOR}${Math.round((0.7 + pulse * 0.3) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(168, 85, 247, ${0.28 + pulse * 0.32})`;
    fillColor = `rgba(216, 180, 254, ${0.16 + pulse * 0.1})`;
  } else if (state?.phase === STATE_COMPLETE) {
    borderColor = COMPLETE_BORDER_COLOR;
    shadowColor = "rgba(0, 79, 164, 0.28)";
    fillColor = `${COMPLETE_FILL_COLOR}33`;
  } else if (state?.phase === STATE_FINALIZING) {
    const pulse = (Math.sin(elapsed * 5) + 1) / 2;
    borderColor = FINALIZING_NODE_BORDER_COLOR;
    shadowColor = `rgba(0, 53, 138, ${0.2 + pulse * 0.26})`;
    fillColor = `${COMPLETE_FILL_COLOR}${Math.round((0.16 + pulse * 0.1) * 255)
      .toString(16)
      .padStart(2, "0")}`;
  } else if (state?.phase === STATE_CANCELLING) {
    const pulse = (Math.sin(elapsed * 9) + 1) / 2;
    borderColor = `${CANCELLING_BORDER_COLOR}${Math.round((0.64 + pulse * 0.36) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(251, 113, 133, ${0.24 + pulse * 0.36})`;
    fillColor = `rgba(251, 113, 133, ${0.1 + pulse * 0.08})`;
  } else if (state?.phase === STATE_ERROR) {
    const pulse = (Math.sin(elapsed * 6) + 1) / 2;
    borderColor = `${ERROR_BORDER_COLOR}${Math.round((0.7 + pulse * 0.3) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(239, 68, 68, ${0.22 + pulse * 0.28})`;
  } else {
    return null;
  }

  return { borderColor, shadowColor, fillColor };
}

/**
 * Return the border, glow, and fill colors for one remote visual state.
 * @param {Record<string, any> | null} state
 * @param {number} elapsed
 * @returns {{ borderColor: string, shadowColor: string, fillColor: string | null }}
 */
function remoteDecorationPalette(state, elapsed) {
  return (
    remotePreparationPalette(state, elapsed) ??
    remoteExecutionPalette(state, elapsed) ?? {
      borderColor: IDLE_BORDER_COLOR,
      shadowColor: "rgba(29, 155, 240, 0.35)",
      fillColor: null,
    }
  );
}

/**
 * Return the visible LiteGraph node matching a Nodes 2.0 DOM element.
 * @param {HTMLElement} nodeElement
 * @returns {LGraphNode | null}
 */
function vueNodeForElement(nodeElement) {
  const graph = app.canvas?.graph;
  const graphNodes = graph?.nodes ?? graph?._nodes ?? [];
  const elementNodeId = String(nodeElement.dataset.nodeId ?? "");
  return graphNodes.find((node) => nodeId(node) === elementNodeId) ?? null;
}

/**
 * Remove Modal's Nodes 2.0 decoration from one DOM node.
 * @param {HTMLElement} nodeElement
 */
function clearVueNodeDecoration(nodeElement) {
  nodeElement.classList.remove("comfy-modal-vue-node");
  delete nodeElement.dataset.modalPhase;
  delete nodeElement.dataset.modalContainer;
  nodeElement.querySelector(":scope > .comfy-modal-vue-node-decoration")?.remove();
}

/**
 * Describe the remote work summarized by a subgraph container.
 * @param {Record<string, any> | null | undefined} state
 * @returns {string}
 */
function remoteContainerTooltip(state) {
  const descendantCount = Number(state?.remoteDescendantCount ?? 0);
  const phaseCounts = state?.phaseCounts ?? {};
  const phaseSummary = [
    STATE_ERROR,
    STATE_CANCELLING,
    STATE_ACTIVE,
    STATE_STARTING,
    STATE_SETUP,
    STATE_READY,
    STATE_FINALIZING,
    STATE_COMPLETE,
  ]
    .filter((phase) => (phaseCounts[phase] ?? 0) > 0)
    .map((phase) => `${phaseCounts[phase]} ${phase}`)
    .join(", ");
  const nodeLabel = descendantCount === 1 ? "node" : "nodes";
  if (!phaseSummary) {
    return `${descendantCount} descendant ${nodeLabel} set to execute on Modal.`;
  }
  return `${descendantCount} Modal descendant ${nodeLabel}: ${phaseSummary}.`;
}

/**
 * Return the decoration element for one Nodes 2.0 node, creating it when needed.
 * @param {HTMLElement} nodeElement
 * @returns {HTMLDivElement}
 */
function ensureVueNodeDecoration(nodeElement) {
  let decoration = nodeElement.querySelector(":scope > .comfy-modal-vue-node-decoration");
  if (!decoration) {
    decoration = document.createElement("div");
    decoration.className = "comfy-modal-vue-node-decoration";
    decoration.setAttribute("aria-hidden", "true");
    const badge = document.createElement("span");
    badge.className = "comfy-modal-vue-node-badge";
    decoration.appendChild(badge);
    nodeElement.appendChild(decoration);
  }
  if (decoration.querySelector(".comfy-modal-vue-execution-location")) {
    return decoration;
  }
  const executionLocation = document.createElement("div");
  executionLocation.className = "comfy-modal-vue-execution-location";
  executionLocation.hidden = true;
  const executionLocationIcon = document.createElement("img");
  executionLocationIcon.className = "comfy-modal-vue-execution-location-icon";
  executionLocationIcon.alt = "";
  executionLocationIcon.setAttribute("aria-hidden", "true");
  const executionLocationLabel = document.createElement("span");
  executionLocationLabel.className = "comfy-modal-vue-execution-location-label";
  executionLocation.append(executionLocationIcon, executionLocationLabel);
  decoration.appendChild(executionLocation);
  return decoration;
}

/**
 * Update the execution-location badge rendered by the Nodes 2.0 DOM layer.
 * @param {HTMLDivElement} decoration
 * @param {Record<string, any> | null} state
 */
function updateVueExecutionLocation(decoration, state) {
  const executionLocation = decoration.querySelector(
    ".comfy-modal-vue-execution-location",
  );
  if (!executionLocation) {
    return;
  }
  const location =
    Number(state?.scheduledEnvironmentCount ?? 0) > 1 &&
    (state?.isActiveRemoteNode || state?.isActiveComponentMember)
      ? state?.executionLocation
      : null;
  const label = String(location?.label ?? "");
  const provider = String(location?.provider ?? "");
  const iconSource = REMOTE_LOCATION_ICON_SOURCES[provider] ?? "";
  executionLocation.hidden = !label;
  if (!label) {
    executionLocation.removeAttribute("title");
    return;
  }
  executionLocation.dataset.provider = provider;
  executionLocation.title = label;
  const icon = executionLocation.querySelector(".comfy-modal-vue-execution-location-icon");
  const labelElement = executionLocation.querySelector(".comfy-modal-vue-execution-location-label");
  if (icon) {
    icon.hidden = !iconSource;
    if (iconSource && icon.getAttribute("src") !== iconSource) {
      icon.setAttribute("src", iconSource);
    }
  }
  if (labelElement) {
    labelElement.textContent = label;
  }
}

/**
 * Apply Modal's current visual state to one Nodes 2.0 DOM node.
 * @param {HTMLElement} nodeElement
 * @param {LGraphNode} node
 * @param {number} timestamp
 */
function updateVueNodeDecoration(nodeElement, node, timestamp) {
  const localBottleneck = isSandwichedLocalNode(node);
  const state = getRemoteVisualState(node);
  if (!isRemoteNode(node) && !state?.isRemoteContainer && !localBottleneck) {
    clearVueNodeDecoration(nodeElement);
    return;
  }
  const palette = localBottleneck ? null : remoteDecorationPalette(state, timestamp / 1000);
  const decoration = ensureVueNodeDecoration(nodeElement);
  const innerWrapper = nodeElement.querySelector(':scope > [data-testid="node-inner-wrapper"]');
  const borderRadius =
    innerWrapper && typeof getComputedStyle === "function"
      ? getComputedStyle(innerWrapper).borderRadius
      : "12px";
  decoration.style.borderColor = palette?.borderColor ?? "transparent";
  decoration.style.backgroundColor = palette?.fillColor ?? "transparent";
  decoration.style.borderRadius = borderRadius || "12px";
  decoration.style.boxShadow = palette ? `0 0 8px ${palette.shadowColor}` : "none";
  const phase = localBottleneck ? "local-bottleneck" : (state?.phase ?? "idle");
  nodeElement.classList.add("comfy-modal-vue-node");
  nodeElement.dataset.modalPhase = phase;
  if (state?.isRemoteContainer) {
    nodeElement.dataset.modalContainer = "true";
  } else {
    delete nodeElement.dataset.modalContainer;
  }
  const badge = decoration.querySelector(".comfy-modal-vue-node-badge");
  const badgeText = localBottleneck
    ? "!"
    : (state?.isRemoteContainer ? "Σ" : String(state?.componentLabel ?? ""));
  badge.textContent = badgeText;
  badge.hidden = !badgeText;
  badge.title = localBottleneck
    ? LOCAL_BOTTLENECK_TOOLTIP
    : (state?.isRemoteContainer ? remoteContainerTooltip(state) : "");
  badge.style.borderColor = localBottleneck
    ? LOCAL_BOTTLENECK_BADGE_BORDER_COLOR
    : palette.borderColor;
  badge.style.boxShadow = localBottleneck ? "none" : `0 0 8px ${palette.shadowColor}`;
  updateVueExecutionLocation(decoration, state);
}

/**
 * Synchronize Modal decorations onto all currently rendered Nodes 2.0 nodes.
 * @param {number | undefined} timestamp
 */
function syncVueNodeDecorations(timestamp = performance.now()) {
  if (typeof document === "undefined") {
    return;
  }
  for (const nodeElement of document.querySelectorAll(".lg-node[data-node-id]")) {
    const node = vueNodeForElement(nodeElement);
    if (node) {
      updateVueNodeDecoration(nodeElement, node, timestamp);
    } else {
      clearVueNodeDecoration(nodeElement);
    }
  }
}

/**
 * Schedule one Nodes 2.0 DOM synchronization after Vue has rendered.
 */
function queueVueNodeDecorationSync() {
  if (typeof document === "undefined" || vueNodeSyncScheduled) {
    return;
  }
  vueNodeSyncScheduled = true;
  const callback = (timestamp) => {
    vueNodeSyncScheduled = false;
    syncVueNodeDecorations(timestamp);
  };
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(callback);
  } else {
    setTimeout(() => callback(performance.now()), 0);
  }
}

/**
 * Redraw both the legacy canvas and the Nodes 2.0 DOM layer.
 */
function refreshNodeDecorations() {
  app.graph?.setDirtyCanvas(true, true);
  queueVueNodeDecorationSync();
}

/**
 * Watch for Vue nodes mounted after workflow loads or renderer switches.
 */
function installVueNodeDecorationObserver() {
  if (vueNodeObserver || typeof MutationObserver === "undefined" || !document.body) {
    return;
  }
  vueNodeObserver = new MutationObserver((records) => {
    const addedVueNode = records.some((record) =>
      Array.from(record.addedNodes).some(
        (addedNode) =>
          addedNode.nodeType === 1 &&
          (addedNode.matches?.(".lg-node[data-node-id]") ||
            addedNode.querySelector?.(".lg-node[data-node-id]")),
      ),
    );
    if (addedVueNode) {
      queueVueNodeDecorationSync();
    }
  });
  vueNodeObserver.observe(document.body, { childList: true, subtree: true });
  queueVueNodeDecorationSync();
}

/**
 * Trace one rounded-rectangle path, using the native canvas helper when available.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} width
 * @param {number} height
 * @param {number} radius
 */
function traceRoundedRectPath(ctx, x, y, width, height, radius) {
  const safeRadius = Math.max(0, Math.min(radius, width / 2, height / 2));
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, width, height, safeRadius);
    return;
  }
  ctx.moveTo(x + safeRadius, y);
  ctx.lineTo(x + width - safeRadius, y);
  ctx.arcTo(x + width, y, x + width, y + safeRadius, safeRadius);
  ctx.lineTo(x + width, y + height - safeRadius);
  ctx.arcTo(x + width, y + height, x + width - safeRadius, y + height, safeRadius);
  ctx.lineTo(x + safeRadius, y + height);
  ctx.arcTo(x, y + height, x, y + height - safeRadius, safeRadius);
  ctx.lineTo(x, y + safeRadius);
  ctx.arcTo(x, y, x + safeRadius, y, safeRadius);
  ctx.closePath();
}

/**
 * Return whether a node-local pointer position is over the local-warning badge.
 * @param {LGraphNode} node
 * @param {number[] | undefined} localPosition
 * @param {number} scale
 * @returns {boolean}
 */
function localBottleneckBadgeContainsPoint(node, localPosition, scale = 1) {
  if (!Array.isArray(localPosition) || localPosition.length < 2) {
    return false;
  }
  const safeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  const titleHeight = node.constructor?.title_height ?? LiteGraph.NODE_TITLE_HEIGHT ?? 24;
  const badgeRadius = 10 / safeScale;
  const badgeX = 10 / safeScale;
  const badgeY = -titleHeight + 10 / safeScale;
  const deltaX = Number(localPosition[0]) - badgeX;
  const deltaY = Number(localPosition[1]) - badgeY;
  return deltaX * deltaX + deltaY * deltaY <= badgeRadius * badgeRadius;
}

/**
 * Set or clear native hover text for a legacy canvas Modal badge.
 * @param {LGraphCanvas | undefined} graphCanvas
 * @param {string} tooltip
 */
function updateLegacyModalTooltip(graphCanvas, tooltip) {
  const canvasElement = graphCanvas?.canvas ?? app.canvas?.canvas;
  if (!canvasElement) {
    return;
  }
  if (tooltip) {
    canvasElement.title = tooltip;
  } else if (
    canvasElement.title === LOCAL_BOTTLENECK_TOOLTIP ||
    canvasElement.title?.includes("Modal descendant") ||
    canvasElement.title?.includes("descendant node")
  ) {
    canvasElement.removeAttribute("title");
  }
}

/**
 * Draw a clipped provider icon and runtime identity beneath node progress bars.
 * @param {CanvasRenderingContext2D} ctx
 * @param {{ provider?: string, label?: string }} location
 * @param {number} rowY
 * @param {number} rowHeight
 * @param {number} nodeWidth
 * @param {number} paddingX
 * @param {number} scale
 */
function drawRemoteExecutionLocation(
  ctx,
  location,
  rowY,
  rowHeight,
  nodeWidth,
  paddingX,
  scale,
) {
  const provider = String(location?.provider ?? "");
  const label = String(location?.label ?? "");
  if (!label) {
    return;
  }
  const icon = remoteLocationIcon(provider);
  const iconHeight = 10 / scale;
  const iconWidth = (provider === "modal" ? 19 : 10) / scale;
  const iconX = paddingX;
  const iconY = rowY + (rowHeight - iconHeight) / 2;
  if (icon?.complete && Number(icon.naturalWidth) > 0 && typeof ctx.drawImage === "function") {
    ctx.drawImage(icon, iconX, iconY, iconWidth, iconHeight);
  }
  const textX = iconX + iconWidth + 5 / scale;
  const availableTextWidth = Math.max(0, nodeWidth - paddingX - textX);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = `${Math.max(8 / scale, 6.5)}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(
    fitCanvasText(ctx, label, availableTextWidth),
    textX,
    rowY + rowHeight / 2,
  );
}

/**
 * Draw the Modal execution decoration or neutral local-bottleneck badge for a node.
 * @param {LGraphNode} node
 * @param {CanvasRenderingContext2D} ctx
 */
function drawModalNodeDecoration(node, ctx) {
  const localBottleneck = isSandwichedLocalNode(node);
  const state = getRemoteVisualState(node);
  if (!isRemoteNode(node) && !state?.isRemoteContainer && !localBottleneck) {
    return;
  }

  const titleHeight = node.constructor?.title_height ?? LiteGraph.NODE_TITLE_HEIGHT ?? 24;
  const scale = app.canvas?.ds?.scale ?? 1;
  const borderWidth = 3 / scale;
  const cornerRadius = 12 / scale;
  const elapsed = performance.now() / 1000;
  const palette = localBottleneck ? null : remoteDecorationPalette(state, elapsed);

  if (palette) {
    ctx.save();
    if (palette.fillColor) {
      ctx.fillStyle = palette.fillColor;
      traceRoundedRectPath(
        ctx,
        borderWidth,
        -titleHeight + borderWidth,
        Math.max(0, node.size[0] - borderWidth * 2),
        Math.max(0, node.size[1] + titleHeight - borderWidth * 2),
        Math.max(0, cornerRadius - borderWidth),
      );
      ctx.fill();
    }
    ctx.strokeStyle = palette.borderColor;
    ctx.lineWidth = borderWidth;
    ctx.shadowColor = palette.shadowColor;
    ctx.shadowBlur = 8 / scale;
    traceRoundedRectPath(
      ctx,
      -borderWidth,
      -titleHeight,
      node.size[0] + borderWidth * 2,
      node.size[1] + titleHeight + borderWidth,
      cornerRadius,
    );
    ctx.stroke();
    ctx.restore();
  }

  const nodeBadgeText = localBottleneck
    ? "!"
    : (state?.isRemoteContainer ? "Σ" : state?.componentLabel);
  if (nodeBadgeText) {
    ctx.save();
    const badgeRadius = 10 / scale;
    const badgeX = 10 / scale;
    const badgeY = -titleHeight + 10 / scale;
    ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
    ctx.strokeStyle = localBottleneck
      ? LOCAL_BOTTLENECK_BADGE_BORDER_COLOR
      : palette.borderColor;
    ctx.lineWidth = 1.5 / scale;
    ctx.shadowColor = localBottleneck ? "transparent" : palette.shadowColor;
    ctx.shadowBlur = localBottleneck ? 0 : 8 / scale;
    ctx.beginPath();
    ctx.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#f8fafc";
    ctx.font = `${Math.max(10 / scale, 8)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(nodeBadgeText), badgeX, badgeY + 0.5 / scale);
    ctx.restore();
  }

  const progressLanes = Array.isArray(state?.progressLanes) ? state.progressLanes : [];
  const setupProgressLanes = progressLanes.filter((laneProgress) => laneProgress.setupOnly);
  const activeProgressLanes = progressLanes.filter((laneProgress) => !laneProgress.setupOnly);
  const visibleActiveProgressLanes = activeProgressLanes.filter(
    (laneProgress) => progressVisualOpacity(laneProgress) > 0,
  );
  const batchProgress = state?.batchProgress ?? null;
  const progressOpacity = progressVisualOpacity(state?.progress);
  const batchOpacity = progressVisualOpacity(batchProgress);
  const laneOpacity = Math.max(
    0,
    ...visibleActiveProgressLanes.map((laneProgress) => progressVisualOpacity(laneProgress)),
    ...setupProgressLanes.map((laneProgress) => progressVisualOpacity(laneProgress)),
  );
  const progressPanelOpacity = Math.max(progressOpacity, batchOpacity, laneOpacity);
  const hasAggregateProgress = Boolean(
    state?.progress &&
      progressOpacity > 0 &&
      (
        [STATE_ACTIVE, STATE_COMPLETE].includes(state.phase) ||
        (state.phase === STATE_STARTING && state.progress.preGpu)
      ),
  );
  const hasSetupLaneProgress = setupProgressLanes.length > 0;
  const hasLaneProgress =
    visibleActiveProgressLanes.length > 0 && [STATE_ACTIVE, STATE_COMPLETE].includes(state?.phase);
  const hasBatchBadge = Boolean(
    batchProgress &&
      batchOpacity > 0 &&
      [STATE_ACTIVE, STATE_COMPLETE].includes(state?.phase) &&
      !hasAggregateProgress,
  );
  const executionLocation =
    Number(state?.scheduledEnvironmentCount ?? 0) > 1 &&
    (state?.isActiveRemoteNode || state?.isActiveComponentMember)
      ? state?.executionLocation
      : null;
  const hasExecutionLocation = Boolean(executionLocation?.label);
  if (
    !hasAggregateProgress &&
    !hasLaneProgress &&
    !hasSetupLaneProgress &&
    !hasBatchBadge &&
    !hasExecutionLocation
  ) {
    return;
  }

  ctx.save();
  ctx.globalAlpha *= Math.max(progressPanelOpacity, hasExecutionLocation ? 1 : 0);
  const barWidth = node.size[0] + borderWidth * 2;
  const aggregateHeight = 12 / scale;
  const laneHeight = 12 / scale;
  const laneGap = 3 / scale;
  const panelY = node.size[1] + 6 / scale;
  const panelPaddingX = 6 / scale;
  const panelPaddingY = 6 / scale;
  const visibleLaneProgress = hasLaneProgress ? visibleActiveProgressLanes : setupProgressLanes;
  const hasVisibleLaneProgress = visibleLaneProgress.length > 0;
  const hasProgressContent =
    hasAggregateProgress || hasVisibleLaneProgress || hasBatchBadge;
  const headerHeight = hasProgressContent ? 16 / scale : 0;
  const hasIterationRateLabels = hasAggregateProgress || hasLaneProgress;
  const progressBarWidth = barWidth;
  const laneBlockHeight = hasVisibleLaneProgress
    ? visibleLaneProgress.length * laneHeight + (visibleLaneProgress.length - 1) * laneGap
    : 0;
  const progressBodyHeight =
    (hasVisibleLaneProgress ? laneBlockHeight + laneGap : 0) + (hasAggregateProgress ? aggregateHeight : 0);
  const progressTopGap = progressBodyHeight > 0 ? laneGap : 0;
  const locationRowHeight = 13 / scale;
  const locationTopGap = hasExecutionLocation && hasProgressContent ? 5 / scale : 0;
  const panelHeight =
    panelPaddingY * 2 +
    headerHeight +
    progressTopGap +
    progressBodyHeight +
    locationTopGap +
    (hasExecutionLocation ? locationRowHeight : 0);
  const laneColors = [
    "rgba(196, 181, 253, 0.94)",
    "rgba(147, 197, 253, 0.94)",
    "rgba(110, 231, 183, 0.94)",
    "rgba(253, 224, 71, 0.94)",
    "rgba(251, 146, 60, 0.94)",
    "rgba(244, 114, 182, 0.94)",
  ];
  const badgeText = hasBatchBadge
    ? `${Math.round(batchProgress.value)}/${Math.round(batchProgress.max)}`
    : null;

  ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
  ctx.beginPath();
  ctx.roundRect(-borderWidth, panelY, barWidth, panelHeight, 10 / scale);
  ctx.fill();
  ctx.strokeStyle = palette?.shadowColor ?? LOCAL_BOTTLENECK_BADGE_BORDER_COLOR;
  ctx.lineWidth = 1 / scale;
  ctx.stroke();

  const aggregateProgress = hasAggregateProgress ? state.progress : null;
  const progressPercent = aggregateProgress
    ? Math.round((aggregateProgress.value / aggregateProgress.max) * 100)
    : null;
  const llmTimingLabel = aggregateProgress?.timeToFirstTokenSeconds != null
    ? `${aggregateProgress.stage === "complete" ? "Done" : "Gen"} • TTFT ${aggregateProgress.timeToFirstTokenSeconds.toFixed(1)}s`
    : null;
  const headerText = llmTimingLabel
    ? llmTimingLabel
    : aggregateProgress?.message
      ? aggregateProgress.message
      : aggregateProgress
        ? `${progressPercent}%`
        : hasVisibleLaneProgress
          ? `${visibleLaneProgress.length}x`
          : null;
  const progressUnit = aggregateProgress?.unit === "tokens"
    ? "tok"
    : aggregateProgress?.unit || "";
  const headerMetric = aggregateProgress && !aggregateProgress.indeterminate
    ? `${Math.round(aggregateProgress.value)}/${Math.round(aggregateProgress.max)}${progressUnit ? ` ${progressUnit}` : ""}`
    : null;
  const headerBaselineY = panelY + panelPaddingY + headerHeight / 2;
  if (headerText) {
    ctx.fillStyle = "#f8fafc";
    ctx.font = `${Math.max(10 / scale, 8)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const metricWidth = headerMetric ? ctx.measureText(headerMetric).width + panelPaddingX : 0;
    const availableLabelWidth = Math.max(0, barWidth - panelPaddingX * 2 - metricWidth);
    ctx.fillText(
      fitCanvasText(ctx, headerText, availableLabelWidth),
      panelPaddingX,
      headerBaselineY,
    );
    if (headerMetric) {
      ctx.textAlign = "right";
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(headerMetric, node.size[0] - panelPaddingX, headerBaselineY);
    }
  }

  if (badgeText) {
    ctx.font = `${Math.max(10 / scale, 8)}px ui-sans-serif, system-ui, sans-serif`;
    const badgePaddingX = 6 / scale;
    const badgeWidth = ctx.measureText(badgeText).width + badgePaddingX * 2;
    const badgeHeight = 16 / scale;
    const badgeX = Math.max(0, node.size[0] - badgeWidth - 4 / scale);
    const badgeY = panelY + panelPaddingY;
    ctx.fillStyle = "rgba(2, 6, 23, 0.82)";
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, 8 / scale);
    ctx.fill();
    ctx.strokeStyle = "rgba(34, 197, 94, 0.55)";
    ctx.lineWidth = 1 / scale;
    ctx.stroke();
    ctx.fillStyle = "#f8fafc";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badgeText, badgeX + badgeWidth / 2, badgeY + badgeHeight / 2);
  }

  const barY = panelY + panelPaddingY + headerHeight + progressTopGap;

  if (hasIterationRateLabels) {
    ctx.font = `${Math.max(9 / scale, 7)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
  }

  if (hasVisibleLaneProgress) {
    let laneY = barY;
    const elapsedPulse = (Math.sin(elapsed * 6) + 1) / 2;
    for (const [laneIndex, laneProgress] of visibleLaneProgress.entries()) {
      const laneRatio = laneProgress.setupOnly
        ? 1
        : Math.max(0, Math.min(1, laneProgress.value / laneProgress.max));
      const laneWidth = Math.max(0, progressBarWidth * laneRatio);
      ctx.fillStyle = "rgba(15, 23, 42, 0.66)";
      ctx.fillRect(-borderWidth, laneY, progressBarWidth, laneHeight);
      const laneColor = laneColors[laneIndex % laneColors.length];
      if (laneProgress.setupOnly) {
        ctx.fillStyle = laneColor.replace("0.94)", `${0.28 + elapsedPulse * 0.22})`);
      } else {
        ctx.fillStyle = laneColor;
      }
      ctx.fillRect(-borderWidth, laneY, laneWidth, laneHeight);
      if (!laneProgress.setupOnly) {
        drawIterationRateOverlay(
          ctx,
          laneProgress.iterationRate,
          node.size[0] + borderWidth,
          laneY + laneHeight / 2,
          laneHeight,
          scale,
          laneProgress.unit,
        );
      }
      laneY += laneHeight + laneGap;
    }
  }

  if (hasAggregateProgress) {
    const progressRatio = Math.max(0, Math.min(1, state.progress.value / state.progress.max));
    const progressWidth = Math.max(0, progressBarWidth * progressRatio);
    const aggregateY = hasVisibleLaneProgress ? barY + laneBlockHeight + laneGap : barY;
    ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
    ctx.fillRect(-borderWidth, aggregateY, progressBarWidth, aggregateHeight);
    ctx.fillStyle = "rgba(216, 180, 254, 0.92)";
    if (state.progress.indeterminate) {
      const pulseWidth = Math.max(progressBarWidth * 0.18, 12 / scale);
      const travelWidth = Math.max(0, progressBarWidth - pulseWidth);
      const pulseX = -borderWidth + ((elapsed * 0.65) % 1) * travelWidth;
      ctx.fillRect(pulseX, aggregateY, pulseWidth, aggregateHeight);
    } else {
      ctx.fillRect(-borderWidth, aggregateY, progressWidth, aggregateHeight);
    }
    drawIterationRateOverlay(
      ctx,
      state.progress.iterationRate,
      node.size[0] + borderWidth,
      aggregateY + aggregateHeight / 2,
      aggregateHeight,
      scale,
      state.progress.unit,
    );
  }
  if (hasExecutionLocation) {
    const locationY =
      panelY +
      panelPaddingY +
      headerHeight +
      progressTopGap +
      progressBodyHeight +
      locationTopGap;
    drawRemoteExecutionLocation(
      ctx,
      executionLocation,
      locationY,
      locationRowHeight,
      node.size[0],
      panelPaddingX,
      scale,
    );
  }
  ctx.restore();
}

/**
 * Install renderer and lifecycle hooks needed by Modal decorations on any graph node.
 * @param {LGraphNode} node
 */
function installNodeDecorationHooks(node) {
  if (node.__modalDecorationInjected) {
    return;
  }
  const originalDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function onDrawForeground(ctx) {
    originalDrawForeground?.apply(this, arguments);
    drawModalNodeDecoration(this, ctx);
  };
  const originalOnMouseMove = node.onMouseMove;
  node.onMouseMove = function onMouseMove(event, localPosition, graphCanvas) {
    const result = originalOnMouseMove?.apply(this, arguments);
    const scale = graphCanvas?.ds?.scale ?? app.canvas?.ds?.scale ?? 1;
    const state = getRemoteVisualState(this);
    const badgeHovered = localBottleneckBadgeContainsPoint(this, localPosition, scale);
    const tooltip = !badgeHovered
      ? ""
      : isSandwichedLocalNode(this)
        ? LOCAL_BOTTLENECK_TOOLTIP
        : (state?.isRemoteContainer ? remoteContainerTooltip(state) : "");
    updateLegacyModalTooltip(graphCanvas, tooltip);
    return result;
  };
  const originalOnMouseLeave = node.onMouseLeave;
  node.onMouseLeave = function onMouseLeave(event, graphCanvas) {
    const result = originalOnMouseLeave?.apply(this, arguments);
    updateLegacyModalTooltip(graphCanvas, "");
    return result;
  };
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function onRemoved() {
    const result = originalOnRemoved?.apply(this, arguments);
    scheduleRemoteDescendantIndexRebuild();
    return result;
  };
  node.__modalDecorationInjected = true;
}

/**
 * Inject the Modal decoration hooks and, where eligible, the remote toggle widget.
 * @param {LGraphNode} node
 */
function decorateNode(node) {
  installNodeDecorationHooks(node);
  if (!isEligibleNode(node) || node.__modalToggleInjected) {
    scheduleRemoteDescendantIndexRebuild();
    return;
  }

  node.properties ||= {};
  node.properties[REMOTE_PROPERTY] = Boolean(node.properties[REMOTE_PROPERTY]);

  const widget = node.addWidget(
    "toggle",
    REMOTE_WIDGET_NAME,
    node.properties[REMOTE_PROPERTY],
    (value) => setRemoteFlag(node, value),
    {
      on: "Enabled",
      off: "Disabled",
      serialize: false,
    },
  );

  widget.value = node.properties[REMOTE_PROPERTY];
  node.__modalToggleInjected = true;
  node.__modalToggleWidget = widget;
  scheduleRemoteDescendantIndexRebuild();
  queueVueNodeDecorationSync();
}

/**
 * Apply an incoming Modal websocket status event.
 * @param {CustomEvent} event
 */
function handleModalStatus(event) {
  const detail = eventDetail(event);
  const promptId = String(detail.prompt_id ?? "");
  if (!promptId) {
    return;
  }
  if (isPromptCancelling(promptId) && ![STATE_ERROR, "execution_interrupted"].includes(detail.phase)) {
    return;
  }
  if (isPromptTerminal(promptId) && detail.phase !== STATE_SETUP) {
    return;
  }
  const nodeIds = (detail.node_ids ?? []).map((value) => String(value));
  const components = detail.components ?? [];
  if (nodeIds.length === 0 && components.length === 0) {
    return;
  }
  if (components.length > 0 || nodeIds.length > 0) {
    registerPromptComponents(promptId, nodeIds, components);
  }
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  if (detail.configurator_node_id) {
    registerPromptConfigurator(promptId, detail.configurator_node_id);
  }
  if (detail.remote_execution_assignments) {
    registerRemoteConfiguratorPlan(
      promptId,
      detail.remote_execution_assignments,
      detail.remote_execution_configurations ?? [],
    );
  }
  const executionProvider = String(detail.execution_provider ?? "").trim();
  const modalGpu =
    (!executionProvider || executionProvider === "modal") &&
    MODAL_GPU_TYPES.includes(detail.modal_gpu)
      ? detail.modal_gpu
      : null;
  if (modalGpu) {
    promptState.modalGpu = modalGpu;
  }

  if (detail.phase === STATE_SETUP) {
    if (isPromptQueuedBehindActiveModal(promptId)) {
      return;
    }
    beginSyntheticExecutionUi(promptId, nodeIds, modalGpu);
    setGlobalStatusPhase(promptId, STATE_SETUP, nodeIds.length, {
      message: detail.status_message ?? null,
      current: detail.status_current ?? null,
      total: detail.status_total ?? null,
      modalGpu,
    });
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_SETUP, promptId);
    return;
  }

  if (["llm_staging", "llm_staged"].includes(detail.phase)) {
    if (isPromptQueuedBehindActiveModal(promptId)) {
      return;
    }
    setGlobalStatusPhase(promptId, STATE_STARTING, nodeIds.length, {
      message: detail.status_message ?? "Staging LLM on CPU",
      current: detail.status_current ?? null,
      total: detail.status_total ?? null,
      modalGpu,
    });
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_STARTING, promptId);
    return;
  }

  if (detail.phase === STATE_WAITING) {
    if (isPromptQueuedBehindActiveModal(promptId)) {
      return;
    }
    setGlobalStatusPhase(promptId, STATE_WAITING, nodeIds.length, {
      message: detail.status_message ?? null,
      current: detail.status_current ?? null,
      total: detail.status_total ?? null,
      modalGpu,
    });
    setPromptActiveNode(promptId, null);
    return;
  }

  if (detail.phase === STATE_STARTING) {
    if (isPromptQueuedBehindActiveModal(promptId)) {
      return;
    }
    setGlobalStatusPhase(promptId, STATE_STARTING, nodeIds.length, {
      message:
        detail.status_message ??
        (executionProvider === "vast"
          ? "Starting Vast.ai component"
          : executionProvider === "ssh_docker"
            ? "Starting self-hosted component"
            : executionProvider === "modal"
              ? "Starting Modal component"
              : "Starting remote component"),
      current: detail.status_current ?? null,
      total: detail.status_total ?? null,
      modalGpu,
    });
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_STARTING, promptId);
    return;
  }

  if (detail.phase === STATE_FINALIZING) {
    clearPromptQueued(promptId);
    setGlobalStatusPhase(promptId, STATE_FINALIZING, nodeIds.length, {
      message: detail.status_message ?? null,
      current: detail.status_current ?? null,
      total: detail.status_total ?? null,
      modalGpu,
    });
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_FINALIZING, promptId);
    return;
  }

  if (detail.phase === STATE_ERROR) {
    const errorMessage = String(
      detail.error_message ?? detail.status_message ?? "Remote workflow failed",
    );
    clearPromptQueued(promptId);
    endSyntheticExecutionUi(promptId, true, errorMessage);
    setGlobalStatusPhase(promptId, STATE_ERROR, nodeIds.length, {
      message: errorMessage,
      modalGpu,
    });
    setRemoteConfiguratorTerminalStatus(promptId, STATE_ERROR, errorMessage);
    setTimeout(() => clearGlobalStatusPhase(promptId), ERROR_CLEAR_DELAY_MS);
    clearPromptRemoteNodeVisuals(promptId);
    markPromptTerminal(promptId, STATE_ERROR);
    return;
  }

  if (detail.phase === "execution_interrupted") {
    markPromptTerminal(promptId, "execution_interrupted");
    clearPromptQueued(promptId);
    endSyntheticExecutionUi(promptId);
    handlePromptInterruption(promptId);
    return;
  }

  if (detail.phase === EXECUTION_PHASE) {
    clearPromptQueued(promptId);
    endSyntheticExecutionUi(promptId);
    const nextActiveNodeId =
      detail.active_node_id != null ? String(detail.active_node_id) : null;
    const previousActiveNodeId = promptState.activeNodeId;
    promptState.hasStreamedProgress = true;
    if (nextActiveNodeId) {
      promptState.hasRemoteExecutionStarted = true;
      setGlobalStatusPhase(promptId, EXECUTION_PHASE, nodeIds.length, { modalGpu });
    } else {
      setGlobalStatusPhase(promptId, STATE_WAITING, nodeIds.length, {
        message:
          detail.status_message ??
          (executionProvider === "vast"
            ? "Waiting for Vast.ai capacity"
            : executionProvider === "ssh_docker"
              ? "Waiting for self-hosted capacity"
              : executionProvider === "modal"
                ? "Waiting for Modal container"
                : "Waiting for remote capacity"),
        modalGpu,
      });
    }
    setNodesPhase(nodeIds, STATE_READY, promptId);
    if (
      previousActiveNodeId &&
      previousActiveNodeId !== nextActiveNodeId &&
      nodesShareRemoteComponent(promptId, previousActiveNodeId, nextActiveNodeId)
    ) {
      fadeNodeProgress(previousActiveNodeId, promptId);
      setNodesPhase([previousActiveNodeId], STATE_COMPLETE, promptId);
    }
    completeRemoteAncestorsBeforeActiveNode(
      promptId,
      detail.completed_ancestor_node_ids,
      nextActiveNodeId,
    );
    if (nextActiveNodeId) {
      updateNodeExecutionLocation(promptId, nextActiveNodeId, detail);
      setNodesPhase([nextActiveNodeId], STATE_ACTIVE, promptId);
    }
    setPromptActiveNode(promptId, nextActiveNodeId);
    return;
  }

  if (detail.phase === "execution_success") {
    clearPromptQueued(promptId);
    promptState.hasStreamedProgress = true;
    promptState.hasRemoteExecutionStarted = true;
    if (promptState.activeNodeId && nodeIds.includes(promptState.activeNodeId)) {
      fadeNodeProgress(promptState.activeNodeId, promptId);
      setNodesPhase([promptState.activeNodeId], STATE_COMPLETE, promptId);
      setPromptActiveNode(promptId, null);
    }
    setNodesPhase(nodeIds, STATE_COMPLETE, promptId);
    return;
  }
}

/**
 * Clear one interrupted prompt's temporary Modal UI state.
 * @param {string} promptId
 */
function handlePromptInterruption(promptId) {
  if (!promptId) {
    return;
  }
  modalCancellingPromptIds.delete(promptId);
  markPromptTerminal(promptId, "execution_interrupted");
  clearGlobalStatusPhase(promptId);
  clearPromptRemoteStates(promptId);
}

/**
 * Apply a streamed numeric Modal node-progress event.
 * @param {CustomEvent} event
 */
function handleModalProgress(event) {
  const detail = eventDetail(event);
  const promptId = String(detail.prompt_id ?? "");
  const progressNodeId = String(detail.real_node_id ?? detail.display_node_id ?? detail.node_id ?? "");
  if (!promptId || !progressNodeId) {
    return;
  }
  if (isPromptCancelling(promptId)) {
    return;
  }
  if (isPromptTerminal(promptId)) {
    return;
  }
  clearPromptQueued(promptId);

  endSyntheticExecutionUi(promptId);
  const promptState = ensurePromptState(promptId);
  if (!promptState) {
    return;
  }
  if (!detail.cached_hit && !detail.pre_gpu) {
    promptState.hasRemoteExecutionStarted = true;
  }
  updateNodeExecutionLocation(promptId, progressNodeId, detail);
  const componentNodeIds = resolveComponentNodeIds(promptId, progressNodeId);
  const readyNodeIds = (componentNodeIds ?? []).filter((nodeIdValue) => nodeIdValue !== progressNodeId);
  promptState.hasStreamedProgress = true;
  if (detail.pre_gpu) {
    setGlobalStatusPhase(promptId, STATE_STARTING, promptState.remoteNodeIds.length || 1, {
      message: detail.message ? String(detail.message) : "Staging LLM on CPU",
    });
    setPromptActiveNode(promptId, null);
    setNodesPhase(componentNodeIds ?? [progressNodeId], STATE_STARTING, promptId);
    setNodeProgress(
      progressNodeId,
      promptId,
      Number(detail.value ?? 0),
      Number(detail.max ?? 1),
      detail,
    );
    return;
  }
  if (detail.aggregate_only) {
    if (componentNodeIds?.length) {
      setNodesPhase(componentNodeIds, STATE_READY, promptId);
    }
    setGlobalStatusBatchProgress(promptId, Number(detail.value ?? 0), Number(detail.max ?? 1));
    setNodeBatchProgress(
      progressNodeId,
      promptId,
      Number(detail.value ?? 0),
      Number(detail.max ?? 1),
    );
    setGlobalStatusPhase(promptId, EXECUTION_PHASE, promptState.remoteNodeIds.length || 1);
    return;
  }
  if (detail.cached_hit) {
    const cachedNodeIds = new Set(
      [detail.node_id, detail.display_node_id, detail.real_node_id]
        .filter((nodeIdValue) => nodeIdValue != null)
        .map((nodeIdValue) => String(nodeIdValue)),
    );
    if (cachedNodeIds.size === 0) {
      cachedNodeIds.add(progressNodeId);
    }
    for (const cachedNodeId of cachedNodeIds) {
      markNodeCached(cachedNodeId, promptId);
    }
    setGlobalStatusPhase(promptId, STATE_WAITING, promptState.remoteNodeIds.length || 1, {
      message: "Restoring Modal cache",
    });
    return;
  }
  setGlobalStatusPhase(promptId, EXECUTION_PHASE, promptState.remoteNodeIds.length || 1, {
    message: detail.message ? String(detail.message) : undefined,
  });
  if (detail.lane_id != null) {
    setPromptActiveNode(promptId, null);
    if (componentNodeIds?.length) {
      setNodesPhase(componentNodeIds, STATE_READY, promptId);
    } else {
      setNodesPhase([progressNodeId], STATE_READY, promptId);
    }
    if (detail.clear) {
      clearNodeProgressLane(progressNodeId, promptId, String(detail.lane_id));
      return;
    }
    setNodeProgressLane(
      progressNodeId,
      promptId,
      String(detail.lane_id),
      Number(detail.value ?? 0),
      Number(detail.max ?? 1),
      detail.item_index,
      Boolean(detail.setup_only),
    );
    return;
  }
  if (detail.clear) {
    clearNodeProgress(progressNodeId, promptId);
    return;
  }
  if (readyNodeIds.length > 0) {
    setNodesPhase(readyNodeIds, STATE_READY, promptId);
  }
  const previousActiveNodeId = promptState.activeNodeId;
  if (
    previousActiveNodeId &&
    previousActiveNodeId !== progressNodeId &&
    nodesShareRemoteComponent(promptId, previousActiveNodeId, progressNodeId)
  ) {
    fadeNodeProgress(previousActiveNodeId, promptId);
    setNodesPhase([previousActiveNodeId], STATE_COMPLETE, promptId);
  }
  completeRemoteAncestorsBeforeActiveNode(
    promptId,
    detail.completed_ancestor_node_ids,
    progressNodeId,
  );
  setPromptActiveNode(promptId, progressNodeId);
  setNodesPhase([progressNodeId], STATE_ACTIVE, promptId);
  setNodeProgress(
    progressNodeId,
    promptId,
    Number(detail.value ?? 0),
    Number(detail.max ?? 1),
    detail,
  );
}

/**
 * Update remote component visuals from a native ComfyUI execution event.
 * @param {CustomEvent} event
 * @param {string} phase
 */
function handleExecutionPhase(event, phase) {
  const detail = eventDetail(event);
  const promptId = String(detail.prompt_id ?? "");
  const representativeNodeId = String(detail.display_node ?? detail.node ?? detail.node_id ?? "");
  if (!promptId || !representativeNodeId) {
    return;
  }
  if (isPromptCancelling(promptId) && phase !== STATE_ERROR) {
    return;
  }
  if (isPromptTerminal(promptId)) {
    return;
  }
  clearPromptQueued(promptId);

  const componentNodeIds = resolveComponentNodeIds(promptId, representativeNodeId);
  if (!componentNodeIds) {
    return;
  }
  const promptState = ensurePromptState(promptId);
  if (promptState.hasStreamedProgress && phase === EXECUTION_PHASE) {
    return;
  }
  if (phase === EXECUTION_PHASE) {
    const plannedProviders = Array.from(
      promptState.executionAssignmentsByRepresentative.values(),
      (assignment) => assignment?.provider,
    ).filter(Boolean);
    setGlobalStatusPhase(
      promptId,
      promptState.hasRemoteExecutionStarted ? EXECUTION_PHASE : STATE_WAITING,
      componentNodeIds.length,
      promptState.hasRemoteExecutionStarted
        ? null
        : { message: remoteCapacityWaitingMessage(plannedProviders) },
    );
    setNodesPhase(componentNodeIds, STATE_READY, promptId, detail.exception_message);
    return;
  }
  if (phase === STATE_ERROR) {
    setGlobalStatusPhase(promptId, STATE_ERROR, componentNodeIds.length);
    setTimeout(() => clearGlobalStatusPhase(promptId), ERROR_CLEAR_DELAY_MS);
    clearPromptRemoteNodeVisuals(promptId);
    return;
  }
  if (phase === STATE_COMPLETE) {
    setPromptActiveNode(promptId, null);
    for (const nodeIdValue of componentNodeIds) {
      fadeNodeProgress(nodeIdValue, promptId);
    }
    setNodesPhase(componentNodeIds, STATE_COMPLETE, promptId, detail.exception_message);
    reconcilePromptGlobalStatus(promptId);
  }
}

/**
 * Clear all temporary remote execution visuals for a completed prompt.
 * @param {string} promptId
 */
function clearPromptRemoteStates(promptId) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    clearPromptProgressStates(promptId);
    for (const [nodeIdValue, state] of Array.from(modalNodeStates.entries())) {
      if (state?.promptId === promptId) {
        clearNodeTimer(nodeIdValue);
        modalNodeStates.delete(nodeIdValue);
      }
    }
    pruneGlobalStatusStates();
    refreshGlobalStatusElement();
    stopAnimationLoopIfIdle();
    refreshNodeDecorations();
    return;
  }
  clearPromptProgressStates(promptId);
  for (const remoteNodeId of promptState.remoteNodeIds) {
    clearNodeTimer(remoteNodeId);
    clearNodeProgress(remoteNodeId, promptId);
    clearNodeCached(remoteNodeId, promptId);
    const currentState = modalNodeStates.get(remoteNodeId);
    if (currentState?.promptId === promptId) {
      modalNodeStates.delete(remoteNodeId);
    }
  }
  for (const ancestorNodeId of promptState.descendantNodeIdsByAncestor.keys()) {
    clearNodeTimer(ancestorNodeId);
    clearNodeProgress(ancestorNodeId, promptId);
    clearNodeCached(ancestorNodeId, promptId);
    const currentState = modalNodeStates.get(ancestorNodeId);
    if (currentState?.promptId === promptId) {
      modalNodeStates.delete(ancestorNodeId);
    }
  }
  modalPromptStates.delete(promptId);
  clearPromptQueued(promptId);
  pruneGlobalStatusStates();
  refreshGlobalStatusElement();
  stopAnimationLoopIfIdle();
  refreshNodeDecorations();
}

/**
 * Clear prompt-scoped remote node visuals while leaving the global status entry intact.
 * @param {string} promptId
 */
function clearPromptRemoteNodeVisuals(promptId) {
  const promptState = modalPromptStates.get(promptId);
  const visualNodeIds = new Set();
  if (promptState) {
    for (const remoteNodeId of promptState.remoteNodeIds) {
      visualNodeIds.add(String(remoteNodeId));
    }
    for (const ancestorNodeId of promptState.descendantNodeIdsByAncestor.keys()) {
      visualNodeIds.add(String(ancestorNodeId));
    }
  }
  for (const [nodeIdValue, state] of modalNodeStates.entries()) {
    if (state?.promptId === promptId) {
      visualNodeIds.add(String(nodeIdValue));
    }
  }
  for (const nodeIdValue of visualNodeIds) {
    clearNodeTimer(nodeIdValue);
    clearNodeProgress(nodeIdValue, promptId);
    clearNodeCached(nodeIdValue, promptId);
    const currentState = modalNodeStates.get(nodeIdValue);
    if (currentState?.promptId === promptId) {
      modalNodeStates.delete(nodeIdValue);
    }
  }
  setPromptActiveNode(promptId, null);
  stopAnimationLoopIfIdle();
  refreshNodeDecorations();
}

/**
 * Apply a queue-time failure to all remote nodes in the just-submitted workflow.
 * @param {string[]} remoteNodeIds
 * @param {string} promptId
 * @param {Error} error
 */
function markQueueFailure(remoteNodeIds, promptId, error) {
  if (remoteNodeIds.length === 0) {
    return;
  }
  setNodesPhase(remoteNodeIds, STATE_ERROR, promptId, queueErrorMessage(error));
}

/**
 * Preserve an actionable server queue failure when PromptExecutionError is generic.
 * @param {Error | any} error
 * @returns {string}
 */
function queueErrorMessage(error) {
  const responsePayload = error?.modalQueueResponse;
  const serverError = responsePayload?.error;
  if (typeof serverError === "string" && serverError.trim()) {
    return serverError.trim();
  }
  if (typeof serverError?.message === "string" && serverError.message.trim()) {
    return serverError.message.trim();
  }
  return String(error?.message ?? error);
}

/**
 * Dispatch a synthetic frontend API event when the underlying API supports EventTarget semantics.
 * @param {string} eventType
 * @param {any} detail
 */
function dispatchSyntheticApiEvent(eventType, detail) {
  if (typeof api.dispatchEvent !== "function") {
    return;
  }
  api.dispatchEvent(new CustomEvent(eventType, { detail }));
}

/**
 * Return the minimal queue status payload expected by ComfyUI's status listeners.
 * @param {number} queueRemaining
 * @returns {{ exec_info: { queue_remaining: number } }}
 */
function statusPayload(queueRemaining) {
  return {
    exec_info: {
      queue_remaining: queueRemaining,
    },
  };
}

/**
 * Start a synthetic queue state while the Modal route is still preparing.
 *
 * Preparation is not node execution. Native `execution_start` and `executing`
 * events start ComfyUI's execution watchdog and can make a long asset download
 * look like a stalled or failed node. The backend queue bridge exposes the
 * prompt through `/queue`; the status event asks ComfyUI to refresh that view.
 * @param {string} promptId
 * @param {string[]} remoteNodeIds
 * @param {string | null} modalGpu
 */
function beginSyntheticExecutionUi(promptId, remoteNodeIds, modalGpu = null) {
  if (remoteNodeIds.length === 0 || syntheticPromptUiStates.has(promptId)) {
    return;
  }
  clearPromptTerminal(promptId);

  syntheticPromptUiStates.set(promptId, {});
  setGlobalStatusPhase(promptId, STATE_SETUP, remoteNodeIds.length, { modalGpu });
  dispatchSyntheticApiEvent("status", statusPayload(1));
  dispatchSyntheticApiEvent("notification", {
    id: promptId,
    value: "Waiting for remote capacity.",
  });
}

/**
 * End a synthetic running state after real queue/execution events take over or the request fails.
 * @param {string} promptId
 * @param {boolean} failed
 * @param {string | null} failureMessage
 */
function endSyntheticExecutionUi(promptId, failed = false, failureMessage = null) {
  if (!syntheticPromptUiStates.has(promptId)) {
    return;
  }

  syntheticPromptUiStates.delete(promptId);
  if (failed) {
    setGlobalStatusPhase(promptId, STATE_ERROR, 1);
    setTimeout(() => clearGlobalStatusPhase(promptId), ERROR_CLEAR_DELAY_MS);
  } else {
    clearGlobalStatusPhase(promptId);
  }
  dispatchSyntheticApiEvent("notification", {
    id: promptId,
    value:
      failed && failureMessage
        ? failureMessage
        : failed
          ? "Remote setup failed."
          : "Remote setup finished.",
  });
  dispatchSyntheticApiEvent("status", statusPayload(0));
}

/**
 * Register websocket listeners for Modal and execution status updates.
 */
function registerExecutionListeners() {
  if (api.__modalExecutionListenersRegistered) {
    return;
  }

  api.addEventListener("modal_status", handleModalStatus);
  api.addEventListener("modal_progress", handleModalProgress);
  api.addEventListener("executing", (event) => handleExecutionPhase(event, EXECUTION_PHASE));
  api.addEventListener("executed", (event) => {
    handleExecutionPhase(event, STATE_COMPLETE);
  });
  api.addEventListener("execution_error", (event) => {
    const detail = eventDetail(event);
    const promptId = String(detail.prompt_id ?? "");
    setRemoteConfiguratorTerminalStatus(
      promptId,
      STATE_ERROR,
      detail.exception_message ?? "Remote workflow failed",
    );
    endSyntheticExecutionUi(promptId, true);
    handleExecutionPhase(event, STATE_ERROR);
    markPromptTerminal(promptId, STATE_ERROR);
  });
  api.addEventListener("execution_interrupted", (event) => {
    const promptId = String(eventDetail(event).prompt_id ?? "");
    setRemoteConfiguratorTerminalStatus(
      promptId,
      STATE_ERROR,
      "Remote workflow interrupted",
    );
    markPromptTerminal(promptId, "execution_interrupted");
    endSyntheticExecutionUi(promptId);
    handlePromptInterruption(promptId);
  });
  api.addEventListener("execution_success", (event) => {
    const detail = eventDetail(event);
    const promptId = String(detail.prompt_id ?? "");
    if (!promptId) {
      return;
    }
    setRemoteConfiguratorTerminalStatus(promptId, STATE_COMPLETE);
    markPromptTerminal(promptId, "execution_success");
    endSyntheticExecutionUi(promptId);
    clearGlobalStatusPhase(promptId);
    clearPromptRemoteStates(promptId);
  });
  if (typeof document !== "undefined") {
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        refreshModalUiAfterVisibilityChange();
      }
    });
  }
  if (typeof window !== "undefined") {
    window.addEventListener("focus", refreshModalUiAfterVisibilityChange);
  }
  api.__modalExecutionListenersRegistered = true;
}

/**
 * Patch the queue API so prompt submission goes through the Modal route.
 */
function patchQueuePrompt() {
  if (api.__modalQueuePromptPatched) {
    return;
  }

  api.queuePrompt = async function modalQueuePrompt(number, data, options) {
    const { output: prompt, workflow } = data;
    stampModalGpuOnWorkflow(workflow);
    const modalGpu = selectedModalGpu(workflow);
    const configuratorNodeId = serializedRemoteConfiguratorNodeId(prompt);
    const usesConfiguredCapacity = Boolean(configuratorNodeId);
    const initialModalGpu = usesConfiguredCapacity ? null : modalGpu;
    const promptId = createPromptId();
    clearPromptTerminal(promptId);
    clearSupersededCancellingPrompts(promptId);
    const remoteNodeIds = effectiveRemoteNodeIds(prompt, workflow);
    registerPromptComponents(promptId, remoteNodeIds, []);
    if (configuratorNodeId) {
      registerPromptConfigurator(promptId, configuratorNodeId);
    }
    const queuedBehindActiveModal =
      remoteNodeIds.length > 0 && markPromptQueuedBehindActiveModal(promptId);
    if (remoteNodeIds.length > 0) {
      if (!queuedBehindActiveModal) {
        setNodesPhase(remoteNodeIds, STATE_SETUP, promptId);
        beginSyntheticExecutionUi(promptId, remoteNodeIds, initialModalGpu);
      }
    }

    const body = {
      client_id: this.clientId ?? "",
      prompt_id: promptId,
      prompt,
      ...(options?.partialExecutionTargets && {
        partial_execution_targets: options.partialExecutionTargets,
      }),
      extra_data: {
        auth_token_comfy_org: this.authToken,
        api_key_comfy_org: this.apiKey,
        extra_pnginfo: { workflow },
      },
    };

    if (number === -1) {
      body.front = true;
    } else if (number !== 0) {
      body.number = number;
    }

    try {
      const response = await this.fetchApi(MODAL_ROUTE, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (response.status !== 200) {
        const responsePayload = await response.json();
        const promptError = new PromptExecutionError(responsePayload);
        promptError.modalQueueResponse = responsePayload;
        throw promptError;
      }

      const responsePayload = await response.json();
      const acceptedConfiguratorNodeId = String(
        responsePayload.remote_execution_configurator_node_id ?? configuratorNodeId ?? "",
      );
      if (acceptedConfiguratorNodeId) {
        registerPromptConfigurator(promptId, acceptedConfiguratorNodeId);
      }
      const sandwichedLocalNodeIds = Array.isArray(
        responsePayload.modal_sandwiched_local_node_ids,
      )
        ? responsePayload.modal_sandwiched_local_node_ids
        : [];
      setSandwichedLocalNodeIds(sandwichedLocalNodeIds);
      if (remoteNodeIds.length > 0) {
        const executionAssignmentsByRepresentative =
          responsePayload.remote_execution_assignments ?? {};
        const executionAssignments = Object.values(executionAssignmentsByRepresentative);
        const selectedModalGpus = Array.isArray(
          responsePayload.remote_execution_modal_gpus,
        )
          ? responsePayload.remote_execution_modal_gpus.filter((value) =>
              MODAL_GPU_TYPES.includes(value),
            )
          : [];
        const configurationsById = new Map(
          (responsePayload.remote_execution_configurations ?? []).map(
            (configuration) => [String(configuration?.configuration_id ?? ""), configuration],
          ),
        );
        const usesModalExecution = executionAssignments.some(
          (assignment) => assignment?.provider === "modal",
        );
        const acceptedModalGpu = usesModalExecution
          ? selectedModalGpus.length === 1
            ? selectedModalGpus[0]
            : MODAL_GPU_TYPES.includes(responsePayload.modal_gpu)
              ? responsePayload.modal_gpu
              : usesConfiguredCapacity
                ? null
                : modalGpu
          : null;
        const selectedProviders = new Set(
          executionAssignments
            .map((assignment) => String(assignment?.provider ?? ""))
            .filter(Boolean),
        );
        const executionEnvironmentLabels = Array.from(
          new Set(
            executionAssignments.map((assignment) =>
              remoteExecutionEnvironmentLabel(
                assignment,
                configurationsById,
                acceptedModalGpu,
              ),
            ),
          ),
        );
        const resolvedRemoteNodeIds = (responsePayload.modal_remote_node_ids ?? []).map((nodeIdValue) =>
          String(nodeIdValue),
        );
        const resolvedComponents = Array.isArray(responsePayload.modal_components)
          ? responsePayload.modal_components
          : [];
        if (resolvedRemoteNodeIds.length > 0 || resolvedComponents.length > 0) {
          registerPromptComponents(promptId, resolvedRemoteNodeIds, resolvedComponents);
        }
        registerRemoteConfiguratorPlan(
          promptId,
          executionAssignmentsByRepresentative,
          responsePayload.remote_execution_configurations ?? [],
        );
        const promptState = ensurePromptState(promptId);
        if (!promptState) {
          return responsePayload;
        }
        promptState.modalGpu = acceptedModalGpu;
        promptState.executionLabel = executionEnvironmentLabels.join(" + ") || null;
        const acceptedRemoteNodeIds =
          promptState.remoteNodeIds.length > 0 ? promptState.remoteNodeIds : remoteNodeIds;
        if (!isPromptQueuedBehindActiveModal(promptId)) {
          endSyntheticExecutionUi(promptId);
          setGlobalStatusPhase(promptId, STATE_WAITING, acceptedRemoteNodeIds.length, {
            message: remoteCapacityWaitingMessage(selectedProviders),
            modalGpu: usesModalExecution ? acceptedModalGpu : null,
          });
          setNodesPhase(acceptedRemoteNodeIds, STATE_READY, promptId);
        }
      }

      return responsePayload;
    } catch (error) {
      clearPromptQueued(promptId);
      endSyntheticExecutionUi(promptId, true, queueErrorMessage(error));
      markQueueFailure(remoteNodeIds, promptId, error);
      setRemoteConfiguratorTerminalStatus(
        promptId,
        STATE_ERROR,
        queueErrorMessage(error),
      );
      throw error;
    }
  };

  api.__modalQueuePromptPatched = true;
}

/**
 * Install CSS keyframes used by the global Modal status badge.
 */
function installGlobalStatusStyles() {
  if (typeof document === "undefined" || document.getElementById("comfy-modal-status-styles")) {
    return;
  }

  const style = document.createElement("style");
  style.id = "comfy-modal-status-styles";
  style.textContent = `
    @keyframes modal-status-pulse {
      0% { transform: scale(0.9); opacity: 0.7; }
      50% { transform: scale(1.08); opacity: 1; }
      100% { transform: scale(0.9); opacity: 0.7; }
    }

    #comfy-modal-global-status .modal-status-copy {
      display: flex;
      min-width: 0;
      flex-direction: column;
      align-items: flex-start;
      line-height: 1.15;
    }

    #comfy-modal-global-status .modal-status-gpu {
      margin-top: 3px;
      color: rgba(226, 232, 240, 0.76);
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.04em;
      line-height: 1;
    }

    #comfy-modal-global-status .modal-status-gpu[hidden] {
      display: none;
    }

    #comfy-modal-global-status .modal-status-cost {
      margin-top: 4px;
      color: rgba(191, 219, 254, 0.9);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.01em;
      line-height: 1.1;
      white-space: nowrap;
    }

    #comfy-modal-global-status .modal-status-cost[hidden] {
      display: none;
    }

    #comfy-modal-global-status .modal-status-billing {
      margin-top: 3px;
      color: rgba(196, 181, 253, 0.92);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.01em;
      line-height: 1.1;
      white-space: nowrap;
    }

    #comfy-modal-global-status .modal-status-billing[hidden] {
      display: none;
    }

    #comfy-modal-global-status .modal-status-containers {
      display: flex;
      min-width: 210px;
      margin-top: 7px;
      padding-top: 6px;
      flex-direction: column;
      gap: 4px;
      border-top: 1px solid rgba(148, 163, 184, 0.18);
    }

    #comfy-modal-global-status .modal-status-containers[hidden] {
      display: none;
    }

    #comfy-modal-global-status .modal-status-container {
      color: rgba(226, 232, 240, 0.82);
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.01em;
      line-height: 1.2;
      white-space: nowrap;
    }

    #comfy-modal-global-status .modal-status-container-running {
      color: rgba(187, 247, 208, 0.92);
    }

    #comfy-modal-global-status .modal-status-container-starting {
      color: rgba(254, 240, 138, 0.92);
    }

    #comfy-modal-global-status .modal-status-container-error {
      color: rgba(254, 202, 202, 0.88);
    }

    .comfy-modal-vue-node-decoration {
      position: absolute;
      inset: -3px;
      z-index: 30;
      box-sizing: border-box;
      border: 3px solid transparent;
      pointer-events: none;
    }

    .comfy-modal-vue-node-badge {
      position: absolute;
      top: 7px;
      left: 7px;
      display: grid;
      width: 20px;
      height: 20px;
      place-items: center;
      box-sizing: border-box;
      border: 1.5px solid transparent;
      border-radius: 9999px;
      background: rgba(15, 23, 42, 0.92);
      color: #f8fafc;
      font: 10px/1 ui-sans-serif, system-ui, sans-serif;
    }

    .comfy-modal-vue-node[data-modal-phase="local-bottleneck"] .comfy-modal-vue-node-badge,
    .comfy-modal-vue-node[data-modal-container="true"] .comfy-modal-vue-node-badge {
      pointer-events: auto;
      cursor: help;
    }

    .comfy-modal-vue-node-badge[hidden] {
      display: none;
    }

    .comfy-modal-vue-execution-location {
      position: absolute;
      top: calc(100% + 6px);
      left: 6px;
      right: 6px;
      display: flex;
      min-width: 0;
      height: 24px;
      align-items: center;
      gap: 6px;
      box-sizing: border-box;
      overflow: hidden;
      padding: 5px 7px;
      border: 1px solid rgba(148, 163, 184, 0.34);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.94);
      box-shadow: 0 3px 10px rgba(2, 6, 23, 0.42);
      color: #cbd5e1;
      font: 500 10px/1 ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0.01em;
      white-space: nowrap;
    }

    .comfy-modal-vue-execution-location[hidden] {
      display: none;
    }

    .comfy-modal-vue-execution-location-icon {
      width: 12px;
      height: 12px;
      flex: 0 0 auto;
      object-fit: contain;
    }

    .comfy-modal-vue-execution-location[data-provider="modal"]
      .comfy-modal-vue-execution-location-icon {
      width: 22px;
    }

    .comfy-modal-vue-execution-location-icon[hidden] {
      display: none;
    }

    .comfy-modal-vue-execution-location-label {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .comfy-remote-configurator-panel {
      display: flex;
      width: 100%;
      min-width: 0;
      flex-direction: column;
      gap: 8px;
      box-sizing: border-box;
      padding: 9px 10px 11px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      border-radius: 9px;
      background: rgba(15, 23, 42, 0.9);
      color: #e2e8f0;
      font: 11px/1.3 ui-sans-serif, system-ui, sans-serif;
      pointer-events: none;
    }

    .remote-configurator-status {
      display: flex;
      min-width: 0;
      align-items: center;
      gap: 7px;
      font-weight: 650;
    }

    .remote-configurator-dot {
      width: 8px;
      height: 8px;
      flex: 0 0 auto;
      border-radius: 999px;
      background: #1d9bf0;
      box-shadow: 0 0 0 4px rgba(29, 155, 240, 0.14);
    }

    .comfy-remote-configurator-panel[data-phase="setup"] .remote-configurator-dot,
    .comfy-remote-configurator-panel[data-phase="waiting"] .remote-configurator-dot {
      background: #f59e0b;
      box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.16);
    }

    .comfy-remote-configurator-panel[data-phase="starting"] .remote-configurator-dot {
      background: #eab308;
      box-shadow: 0 0 0 4px rgba(234, 179, 8, 0.16);
    }

    .comfy-remote-configurator-panel[data-phase="executing"] .remote-configurator-dot {
      background: #a855f7;
      box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.16);
    }

    .comfy-remote-configurator-panel[data-phase="complete"] .remote-configurator-dot {
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.16);
    }

    .comfy-remote-configurator-panel[data-phase="error"] .remote-configurator-dot,
    .comfy-remote-configurator-panel[data-phase="cancelling"] .remote-configurator-dot {
      background: #ef4444;
      box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.16);
    }

    .remote-configurator-status-text {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .remote-configurator-progress {
      position: relative;
      height: 12px;
      overflow: hidden;
      border-radius: 6px;
      background: rgba(2, 6, 23, 0.78);
    }

    .remote-configurator-progress-fill {
      position: absolute;
      inset: 0 auto 0 0;
      width: 0;
      background: linear-gradient(90deg, #f59e0b, #eab308);
      transition: width 160ms ease-out;
    }

    .remote-configurator-progress-value {
      position: absolute;
      inset: 0 5px 0 auto;
      color: #f8fafc;
      font: 650 9px/12px ui-monospace, SFMono-Regular, monospace;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
    }

    .remote-configurator-plan-title {
      color: #94a3b8;
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .remote-configurator-empty {
      color: #94a3b8;
      font-size: 10px;
    }

    .remote-configurator-table {
      width: 100%;
      table-layout: fixed;
      border-collapse: collapse;
    }

    .remote-configurator-table th,
    .remote-configurator-table td {
      overflow: hidden;
      padding: 5px 6px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.16);
      text-align: left;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .remote-configurator-table th {
      color: #94a3b8;
      font-size: 9px;
      font-weight: 650;
    }

    .remote-configurator-table th:first-child,
    .remote-configurator-table td:first-child {
      width: 22%;
    }

    .remote-configurator-table th:last-child,
    .remote-configurator-table td:last-child {
      width: 20%;
      text-align: right;
    }
  `;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "Comfy.ModalSync.Toggle",

  async init() {
    installGlobalStatusStyles();
    installVueNodeDecorationObserver();
    patchInterruptFeedback();
    patchQueuePrompt();
    registerExecutionListeners();
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    installModalContextMenu(nodeType, nodeData);
  },

  async nodeCreated(node) {
    decorateNode(node);
    mountRemoteExecutionConfiguratorPanel(node);
  },

  async loadedGraphNode(node) {
    synchronizeRemoteFlagFromWidget(node);
  },

  async afterConfigureGraph() {
    for (const node of allWorkflowNodes()) {
      synchronizeRemoteFlagFromWidget(node);
      mountRemoteExecutionConfiguratorPanel(node);
    }
    rebuildRemoteDescendantIndex();
    refreshNodeDecorations();
    selectedModalGpu();
  },
});
