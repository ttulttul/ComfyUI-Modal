import { app } from "../../scripts/app.js";
import { PromptExecutionError, api } from "../../scripts/api.js";

const REMOTE_PROPERTY = "is_modal_remote";
const MODAL_ROUTE = "/modal/queue_prompt";
const INTERNAL_NODE_PREFIX = "ModalUniversalExecutor";

const IDLE_BORDER_COLOR = "#1d9bf0";
const SETUP_BORDER_COLOR = "#f59e0b";
const READY_BORDER_COLOR = "#22c55e";
const ACTIVE_BORDER_COLOR = "#a855f7";
const COMPLETE_BORDER_COLOR = "#16a34a";
const ERROR_BORDER_COLOR = "#ef4444";

const STATE_SETUP = "setup";
const EXECUTION_PHASE = "executing";
const STATE_READY = "ready";
const STATE_ACTIVE = "active";
const STATE_COMPLETE = "complete";
const STATE_ERROR = "error";
const ERROR_CLEAR_DELAY_MS = 5000;

const modalNodeStates = new Map();
const modalNodeClearTimers = new Map();
const modalPromptStates = new Map();
const syntheticPromptUiStates = new Map();
const modalGlobalStatusStates = new Map();

let animationFrameHandle = null;
let modalGlobalStatusElement = null;

/**
 * Return whether a node should show the Modal toggle.
 * @param {LGraphNode} node
 * @returns {boolean}
 */
function isEligibleNode(node) {
  return Boolean(node?.comfyClass) && !String(node.comfyClass).startsWith(INTERNAL_NODE_PREFIX);
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
  element.style.alignItems = "center";
  element.style.gap = "10px";
  element.style.padding = "10px 14px";
  element.style.borderRadius = "999px";
  element.style.border = "1px solid rgba(255, 255, 255, 0.16)";
  element.style.background = "rgba(15, 23, 42, 0.94)";
  element.style.boxShadow = "0 10px 30px rgba(0, 0, 0, 0.28)";
  element.style.color = "#f8fafc";
  element.style.fontFamily = "ui-sans-serif, system-ui, sans-serif";
  element.style.fontSize = "13px";
  element.style.fontWeight = "600";
  element.style.pointerEvents = "none";
  element.innerHTML =
    '<span class="modal-status-dot"></span><span class="modal-status-text"></span>';
  document.body.appendChild(element);
  modalGlobalStatusElement = element;
  return element;
}

/**
 * Return the most important active global Modal state.
 * @returns {{ phase: string, promptId: string, nodeCount: number } | null}
 */
function currentGlobalStatus() {
  if (modalGlobalStatusStates.size === 0) {
    return null;
  }

  const phases = Array.from(modalGlobalStatusStates.entries()).map(([promptId, state]) => ({
    promptId,
    phase: state.phase,
    nodeCount: state.nodeCount,
    updatedAt: state.updatedAt,
  }));
  phases.sort((left, right) => right.updatedAt - left.updatedAt);

  return (
    phases.find((state) => state.phase === STATE_ERROR) ??
    phases.find((state) => state.phase === STATE_SETUP) ??
    phases.find((state) => state.phase === EXECUTION_PHASE) ??
    phases[0]
  );
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
    return;
  }

  const dot = element.querySelector(".modal-status-dot");
  const text = element.querySelector(".modal-status-text");
  const nodeLabel = activeState.nodeCount === 1 ? "node" : "nodes";

  element.style.display = "inline-flex";
  element.dataset.phase = activeState.phase;

  if (activeState.phase === STATE_SETUP) {
    element.style.borderColor = "rgba(245, 158, 11, 0.55)";
    element.style.background = "rgba(61, 42, 9, 0.94)";
    dot.style.background = SETUP_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(245, 158, 11, 0.18)";
    dot.style.animation = "modal-status-pulse 1.1s ease-in-out infinite";
    text.textContent = `Modal setup running for ${activeState.nodeCount} ${nodeLabel}`;
  } else if (activeState.phase === EXECUTION_PHASE) {
    element.style.borderColor = "rgba(34, 197, 94, 0.55)";
    element.style.background = "rgba(8, 49, 28, 0.94)";
    dot.style.background = ACTIVE_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(34, 197, 94, 0.18)";
    dot.style.animation = "modal-status-pulse 1.1s ease-in-out infinite";
    text.textContent = `Modal workflow running on ${activeState.nodeCount} ${nodeLabel}`;
  } else if (activeState.phase === STATE_ERROR) {
    element.style.borderColor = "rgba(239, 68, 68, 0.55)";
    element.style.background = "rgba(69, 10, 10, 0.94)";
    dot.style.background = ERROR_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(239, 68, 68, 0.18)";
    dot.style.animation = "none";
    text.textContent = "Modal workflow failed";
  } else {
    element.style.borderColor = "rgba(29, 155, 240, 0.5)";
    element.style.background = "rgba(15, 23, 42, 0.94)";
    dot.style.background = IDLE_BORDER_COLOR;
    dot.style.boxShadow = "0 0 0 6px rgba(29, 155, 240, 0.18)";
    dot.style.animation = "none";
    text.textContent = "Modal workflow active";
  }

  dot.style.width = "10px";
  dot.style.height = "10px";
  dot.style.borderRadius = "999px";
  dot.style.display = "inline-block";
}

/**
 * Record one prompt's global Modal execution phase.
 * @param {string} promptId
 * @param {string} phase
 * @param {number} nodeCount
 */
function setGlobalStatusPhase(promptId, phase, nodeCount) {
  if (!promptId) {
    return;
  }
  modalGlobalStatusStates.set(promptId, {
    phase,
    nodeCount: Math.max(1, Number(nodeCount) || 1),
    updatedAt: nowMs(),
  });
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
 * Return the prompt metadata bucket, creating it if needed.
 * @param {string} promptId
 * @returns {{ startedAt: number, remoteNodeIds: string[], componentsByRepresentative: Map<string, string[]> }}
 */
function ensurePromptState(promptId) {
  if (!modalPromptStates.has(promptId)) {
    modalPromptStates.set(promptId, {
      startedAt: nowMs(),
      remoteNodeIds: [],
      componentsByRepresentative: new Map(),
      activeNodeId: null,
      hasStreamedProgress: false,
    });
  }
  return modalPromptStates.get(promptId);
}

/**
 * Return whether an incoming prompt update should replace the current node state.
 * @param {string} nodeIdValue
 * @param {string} promptId
 * @returns {boolean}
 */
function shouldApplyPromptState(nodeIdValue, promptId) {
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
 * Mark the canvas dirty and keep animation alive while visual states are active.
 */
function refreshCanvasAnimation() {
  app.graph?.setDirtyCanvas(true, true);
  const hasAnimatedState = Array.from(modalNodeStates.values()).some((state) =>
    [STATE_SETUP, STATE_READY, STATE_ACTIVE, STATE_ERROR].includes(state.phase),
  );
  if (!hasAnimatedState) {
    animationFrameHandle = null;
    return;
  }
  animationFrameHandle = requestAnimationFrame(refreshCanvasAnimation);
}

/**
 * Ensure the redraw loop is running while remote visual effects are active.
 */
function ensureAnimationLoop() {
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
    ensureAnimationLoop();
    app.graph?.setDirtyCanvas(true, true);
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
  for (const currentNodeId of nodeIds) {
    if (!shouldApplyPromptState(currentNodeId, promptId)) {
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
  }
  ensureAnimationLoop();
  app.graph?.setDirtyCanvas(true, true);
}

/**
 * Register remote component membership for a prompt.
 * @param {string} promptId
 * @param {string[]} remoteNodeIds
 * @param {{ representative_node_id: string, node_ids: string[] }[]} components
 */
function registerPromptComponents(promptId, remoteNodeIds, components) {
  const promptState = ensurePromptState(promptId);
  if (remoteNodeIds.length > 0) {
    promptState.remoteNodeIds = [...remoteNodeIds];
  }
  if (components.length > 0) {
    promptState.componentsByRepresentative.clear();
    promptState.activeNodeId = null;
    promptState.hasStreamedProgress = false;
    for (const component of components) {
      promptState.componentsByRepresentative.set(
        String(component.representative_node_id),
        component.node_ids.map((nodeIdValue) => String(nodeIdValue)),
      );
    }
  }
}

/**
 * Record the currently active remote node inside one prompt.
 * @param {string} promptId
 * @param {string | null} activeNodeId
 */
function setPromptActiveNode(promptId, activeNodeId) {
  const promptState = ensurePromptState(promptId);
  promptState.activeNodeId = activeNodeId ? String(activeNodeId) : null;
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
  return promptState.componentsByRepresentative.get(representativeNodeId) ?? null;
}

/**
 * Extract remote node ids from a workflow snapshot.
 * @param {object | undefined} workflow
 * @returns {string[]}
 */
function extractRemoteNodeIds(workflow) {
  const remoteNodeIds = [];
  for (const node of workflow?.nodes ?? []) {
    if (node?.properties?.[REMOTE_PROPERTY]) {
      remoteNodeIds.push(String(node.id));
    }
  }
  return remoteNodeIds;
}

/**
 * Update the node state and redraw the canvas.
 * @param {LGraphNode} node
 * @param {boolean} value
 */
function setRemoteFlag(node, value) {
  node.properties ||= {};
  node.properties[REMOTE_PROPERTY] = Boolean(value);
  app.graph.setDirtyCanvas(true, true);
}

/**
 * Return the current remote visual state for a node.
 * @param {LGraphNode} node
 * @returns {{ phase: string, promptId: string } | null}
 */
function getRemoteVisualState(node) {
  const state = modalNodeStates.get(nodeId(node)) ?? null;
  if (!state?.promptId) {
    return state;
  }
  const promptState = modalPromptStates.get(state.promptId);
  return {
    ...state,
    isActiveRemoteNode: promptState?.activeNodeId === nodeId(node),
  };
}

/**
 * Draw the remote execution border and shading for a node.
 * @param {LGraphNode} node
 * @param {CanvasRenderingContext2D} ctx
 */
function drawRemoteNodeDecoration(node, ctx) {
  if (!isRemoteNode(node)) {
    return;
  }

  const state = getRemoteVisualState(node);
  const titleHeight = node.constructor?.title_height ?? LiteGraph.NODE_TITLE_HEIGHT ?? 24;
  const scale = app.canvas?.ds?.scale ?? 1;
  const borderWidth = 3 / scale;
  const elapsed = performance.now() / 1000;

  let borderColor = IDLE_BORDER_COLOR;
  let shadowColor = "rgba(29, 155, 240, 0.35)";
  let fillColor = null;

  if (state?.phase === STATE_SETUP) {
    const pulse = (Math.sin(elapsed * 5) + 1) / 2;
    borderColor = `${SETUP_BORDER_COLOR}${Math.round((0.65 + pulse * 0.35) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(245, 158, 11, ${0.25 + pulse * 0.35})`;
  } else if (state?.phase === STATE_READY) {
    const pulse = (Math.sin(elapsed * 6) + 1) / 2;
    borderColor = READY_BORDER_COLOR;
    shadowColor = "rgba(34, 197, 94, 0.35)";
    fillColor = `rgba(134, 239, 172, ${0.12 + pulse * 0.08})`;
  } else if (state?.phase === STATE_ACTIVE) {
    const pulse = (Math.sin(elapsed * 7) + 1) / 2;
    borderColor = `${ACTIVE_BORDER_COLOR}${Math.round((0.7 + pulse * 0.3) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(168, 85, 247, ${0.28 + pulse * 0.32})`;
    fillColor = `rgba(216, 180, 254, ${0.16 + pulse * 0.1})`;
  } else if (state?.phase === STATE_COMPLETE) {
    borderColor = COMPLETE_BORDER_COLOR;
    shadowColor = "rgba(34, 197, 94, 0.28)";
    fillColor = "rgba(134, 239, 172, 0.14)";
  } else if (state?.phase === STATE_ERROR) {
    const pulse = (Math.sin(elapsed * 6) + 1) / 2;
    borderColor = `${ERROR_BORDER_COLOR}${Math.round((0.7 + pulse * 0.3) * 255)
      .toString(16)
      .padStart(2, "0")}`;
    shadowColor = `rgba(239, 68, 68, ${0.22 + pulse * 0.28})`;
  }

  ctx.save();
  if (fillColor) {
    ctx.fillStyle = fillColor;
    ctx.fillRect(
      borderWidth,
      -titleHeight + borderWidth,
      Math.max(0, node.size[0] - borderWidth * 2),
      Math.max(0, node.size[1] + titleHeight - borderWidth * 2),
    );
  }
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = borderWidth;
  ctx.shadowColor = shadowColor;
  ctx.shadowBlur = 8 / scale;
  ctx.strokeRect(
    -borderWidth,
    -titleHeight,
    node.size[0] + borderWidth * 2,
    node.size[1] + titleHeight + borderWidth,
  );
  ctx.restore();
}

/**
 * Inject the toggle widget and remote border behavior into a node.
 * @param {LGraphNode} node
 */
function decorateNode(node) {
  if (!isEligibleNode(node) || node.__modalToggleInjected) {
    return;
  }

  node.properties ||= {};
  node.properties[REMOTE_PROPERTY] = Boolean(node.properties[REMOTE_PROPERTY]);

  const widget = node.addWidget(
    "toggle",
    "Run on Modal",
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

  const originalDrawForeground = node.onDrawForeground;
  node.onDrawForeground = function onDrawForeground(ctx) {
    originalDrawForeground?.apply(this, arguments);
    drawRemoteNodeDecoration(this, ctx);
  };
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
  const nodeIds = (detail.node_ids ?? []).map((value) => String(value));
  const components = detail.components ?? [];
  if (components.length > 0 || nodeIds.length > 0) {
    registerPromptComponents(promptId, nodeIds, components);
  }
  const promptState = ensurePromptState(promptId);

  if (detail.phase === STATE_SETUP) {
    beginSyntheticExecutionUi(promptId, nodeIds);
    setGlobalStatusPhase(promptId, STATE_SETUP, nodeIds.length);
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_SETUP, promptId);
    return;
  }

  if (detail.phase === STATE_ERROR) {
    endSyntheticExecutionUi(promptId, true);
    setGlobalStatusPhase(promptId, STATE_ERROR, nodeIds.length);
    setTimeout(() => clearGlobalStatusPhase(promptId), ERROR_CLEAR_DELAY_MS);
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_ERROR, promptId, detail.error_message);
    return;
  }

  if (detail.phase === EXECUTION_PHASE) {
    const nextActiveNodeId =
      detail.active_node_id != null ? String(detail.active_node_id) : null;
    const previousActiveNodeId = promptState.activeNodeId;
    promptState.hasStreamedProgress = true;
    setGlobalStatusPhase(promptId, EXECUTION_PHASE, nodeIds.length);
    setNodesPhase(nodeIds, STATE_READY, promptId);
    if (previousActiveNodeId && previousActiveNodeId !== nextActiveNodeId) {
      setNodesPhase([previousActiveNodeId], STATE_COMPLETE, promptId);
    }
    if (nextActiveNodeId) {
      setNodesPhase([nextActiveNodeId], STATE_ACTIVE, promptId);
    }
    setPromptActiveNode(promptId, nextActiveNodeId);
    return;
  }

  if (detail.phase === "execution_success") {
    promptState.hasStreamedProgress = true;
    if (promptState.activeNodeId) {
      setNodesPhase([promptState.activeNodeId], STATE_COMPLETE, promptId);
    }
    setPromptActiveNode(promptId, null);
    setNodesPhase(nodeIds, STATE_COMPLETE, promptId);
    return;
  }
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

  const componentNodeIds = resolveComponentNodeIds(promptId, representativeNodeId);
  if (!componentNodeIds) {
    return;
  }
  const promptState = ensurePromptState(promptId);
  if (promptState.hasStreamedProgress && phase !== STATE_ERROR) {
    return;
  }
  if (phase === EXECUTION_PHASE) {
    setGlobalStatusPhase(promptId, EXECUTION_PHASE, componentNodeIds.length);
    setNodesPhase(componentNodeIds, STATE_READY, promptId, detail.exception_message);
    return;
  }
  if (phase === STATE_ERROR) {
    setGlobalStatusPhase(promptId, STATE_ERROR, componentNodeIds.length);
    setTimeout(() => clearGlobalStatusPhase(promptId), ERROR_CLEAR_DELAY_MS);
    setPromptActiveNode(promptId, null);
    setNodesPhase(componentNodeIds, STATE_ERROR, promptId, detail.exception_message);
    return;
  }
  if (phase === STATE_COMPLETE) {
    setPromptActiveNode(promptId, null);
    setNodesPhase(componentNodeIds, STATE_COMPLETE, promptId, detail.exception_message);
  }
}

/**
 * Clear all temporary remote execution visuals for a completed prompt.
 * @param {string} promptId
 */
function clearPromptRemoteStates(promptId) {
  const promptState = modalPromptStates.get(promptId);
  if (!promptState) {
    return;
  }
  for (const remoteNodeId of promptState.remoteNodeIds) {
    clearNodeTimer(remoteNodeId);
    const currentState = modalNodeStates.get(remoteNodeId);
    if (currentState?.promptId === promptId) {
      modalNodeStates.delete(remoteNodeId);
    }
  }
  modalPromptStates.delete(promptId);
  app.graph?.setDirtyCanvas(true, true);
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
  setNodesPhase(remoteNodeIds, STATE_ERROR, promptId, String(error?.message ?? error));
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
 * Start a synthetic running state so ComfyUI shows active execution while the Modal route is still preparing.
 * @param {string} promptId
 * @param {string[]} remoteNodeIds
 */
function beginSyntheticExecutionUi(promptId, remoteNodeIds) {
  if (remoteNodeIds.length === 0 || syntheticPromptUiStates.has(promptId)) {
    return;
  }

  const displayNode = remoteNodeIds[0];
  syntheticPromptUiStates.set(promptId, { displayNode });
  setGlobalStatusPhase(promptId, STATE_SETUP, remoteNodeIds.length);
  dispatchSyntheticApiEvent("status", statusPayload(1));
  dispatchSyntheticApiEvent("notification", {
    id: promptId,
    value: "Waiting for a machine on Modal.",
  });
  dispatchSyntheticApiEvent("execution_start", {
    prompt_id: promptId,
    timestamp: nowMs(),
  });
  dispatchSyntheticApiEvent("executing", displayNode);
}

/**
 * End a synthetic running state after real queue/execution events take over or the request fails.
 * @param {string} promptId
 * @param {boolean} failed
 */
function endSyntheticExecutionUi(promptId, failed = false) {
  const syntheticState = syntheticPromptUiStates.get(promptId);
  if (!syntheticState) {
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
    value: "Modal setup finished.",
  });
  dispatchSyntheticApiEvent("status", statusPayload(0));
  if (failed) {
    dispatchSyntheticApiEvent("execution_error", {
      prompt_id: promptId,
      node_id: syntheticState.displayNode,
      node_type: "ModalRemoteComponent",
      executed: [],
      exception_message: "Modal queue request failed before prompt execution started.",
      exception_type: "ModalQueueError",
      traceback: [],
      current_inputs: [],
      current_outputs: [],
    });
  }
}

/**
 * Register websocket listeners for Modal and execution status updates.
 */
function registerExecutionListeners() {
  if (api.__modalExecutionListenersRegistered) {
    return;
  }

  api.addEventListener("modal_status", handleModalStatus);
  api.addEventListener("executing", (event) => handleExecutionPhase(event, EXECUTION_PHASE));
  api.addEventListener("executed", (event) => {
    endSyntheticExecutionUi(String(eventDetail(event).prompt_id ?? ""));
    handleExecutionPhase(event, STATE_COMPLETE);
  });
  api.addEventListener("execution_error", (event) => {
    endSyntheticExecutionUi(String(eventDetail(event).prompt_id ?? ""), true);
    handleExecutionPhase(event, STATE_ERROR);
  });
  api.addEventListener("execution_interrupted", (event) => {
    endSyntheticExecutionUi(String(eventDetail(event).prompt_id ?? ""), true);
    handleExecutionPhase(event, STATE_ERROR);
  });
  api.addEventListener("execution_success", (event) => {
    const detail = eventDetail(event);
    const promptId = String(detail.prompt_id ?? "");
    if (!promptId) {
      return;
    }
    endSyntheticExecutionUi(promptId);
    clearGlobalStatusPhase(promptId);
    clearPromptRemoteStates(promptId);
  });
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
    const promptId = createPromptId();
    const remoteNodeIds = extractRemoteNodeIds(workflow);
    registerPromptComponents(promptId, remoteNodeIds, []);
    if (remoteNodeIds.length > 0) {
      setNodesPhase(remoteNodeIds, STATE_SETUP, promptId);
      beginSyntheticExecutionUi(promptId, remoteNodeIds);
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
        throw new PromptExecutionError(await response.json());
      }

      return await response.json();
    } catch (error) {
      endSyntheticExecutionUi(promptId, true);
      markQueueFailure(remoteNodeIds, promptId, error);
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
  `;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "Comfy.ModalSync.Toggle",

  async init() {
    installGlobalStatusStyles();
    patchQueuePrompt();
    registerExecutionListeners();
  },

  async nodeCreated(node) {
    decorateNode(node);
  },
});
