import { app } from "../../scripts/app.js";
import { PromptExecutionError, api } from "../../scripts/api.js";

const REMOTE_PROPERTY = "is_modal_remote";
const MODAL_ROUTE = "/modal/queue_prompt";
const INTERNAL_NODE_PREFIX = "ModalUniversalExecutor";

const IDLE_BORDER_COLOR = "#1d9bf0";
const SETUP_BORDER_COLOR = "#f59e0b";
const ACTIVE_BORDER_COLOR = "#22c55e";
const ERROR_BORDER_COLOR = "#ef4444";

const STATE_SETUP = "setup";
const STATE_EXECUTING = "executing";
const STATE_COMPLETE = "complete";
const STATE_ERROR = "error";

const COMPLETION_CLEAR_DELAY_MS = 1800;
const ERROR_CLEAR_DELAY_MS = 5000;

const modalNodeStates = new Map();
const modalNodeClearTimers = new Map();
const modalPromptStates = new Map();
const syntheticPromptUiStates = new Map();

let animationFrameHandle = null;

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
    [STATE_SETUP, STATE_EXECUTING, STATE_COMPLETE, STATE_ERROR].includes(state.phase),
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
    if (phase === STATE_COMPLETE) {
      scheduleNodeClear(currentNodeId, promptId, COMPLETION_CLEAR_DELAY_MS);
    } else if (phase === STATE_ERROR) {
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
  promptState.remoteNodeIds = [...remoteNodeIds];
  promptState.componentsByRepresentative.clear();
  for (const component of components) {
    promptState.componentsByRepresentative.set(
      String(component.representative_node_id),
      component.node_ids.map((nodeIdValue) => String(nodeIdValue)),
    );
  }
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
  return modalNodeStates.get(nodeId(node)) ?? null;
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
  } else if (state?.phase === STATE_EXECUTING) {
    const pulse = (Math.sin(elapsed * 6) + 1) / 2;
    borderColor = ACTIVE_BORDER_COLOR;
    shadowColor = "rgba(34, 197, 94, 0.35)";
    fillColor = `rgba(134, 239, 172, ${0.12 + pulse * 0.08})`;
  } else if (state?.phase === STATE_COMPLETE) {
    borderColor = ACTIVE_BORDER_COLOR;
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
  if (detail.phase === STATE_SETUP) {
    beginSyntheticExecutionUi(promptId, (detail.node_ids ?? []).map((value) => String(value)));
  } else if (detail.phase === STATE_ERROR) {
    endSyntheticExecutionUi(promptId, true);
  }

  const nodeIds = (detail.node_ids ?? []).map((value) => String(value));
  const components = detail.components ?? [];
  if (components.length > 0 || nodeIds.length > 0) {
    registerPromptComponents(promptId, nodeIds, components);
  }

  if (detail.phase === STATE_SETUP) {
    setNodesPhase(nodeIds, STATE_SETUP, promptId);
    return;
  }

  if (detail.phase === STATE_ERROR) {
    setNodesPhase(nodeIds, STATE_ERROR, promptId, detail.error_message);
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
  setNodesPhase(componentNodeIds, phase, promptId, detail.exception_message);
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
  api.addEventListener("executing", (event) => handleExecutionPhase(event, STATE_EXECUTING));
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
    const promptState = modalPromptStates.get(promptId);
    if (!promptState) {
      return;
    }
    setNodesPhase(promptState.remoteNodeIds, STATE_COMPLETE, promptId);
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

app.registerExtension({
  name: "Comfy.ModalSync.Toggle",

  async init() {
    patchQueuePrompt();
    registerExecutionListeners();
  },

  async nodeCreated(node) {
    decorateNode(node);
  },
});
