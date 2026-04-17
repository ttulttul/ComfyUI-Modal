import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const REMOTE_PROPERTY = "is_modal_remote";
const MODAL_ROUTE = "/modal/queue_prompt";
const REMOTE_BORDER_COLOR = "#1d9bf0";
const INTERNAL_NODE_PREFIX = "ModalUniversalExecutor";

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
    if (!isRemoteNode(this)) {
      return;
    }

    const titleHeight = this.constructor?.title_height ?? LiteGraph.NODE_TITLE_HEIGHT ?? 24;
    const borderWidth = 3 / app.canvas.ds.scale;
    ctx.save();
    ctx.strokeStyle = REMOTE_BORDER_COLOR;
    ctx.lineWidth = borderWidth;
    ctx.shadowColor = "rgba(29, 155, 240, 0.35)";
    ctx.shadowBlur = 8 / app.canvas.ds.scale;
    ctx.strokeRect(
      -borderWidth,
      -titleHeight,
      this.size[0] + borderWidth * 2,
      this.size[1] + titleHeight + borderWidth,
    );
    ctx.restore();
  };
}

/**
 * Patch the queue API so prompt submission goes through the Modal route.
 */
function patchQueuePrompt() {
  if (api.__modalQueuePromptPatched) {
    return;
  }

  const originalQueuePrompt = api.queuePrompt.bind(api);
  api.queuePrompt = async function modalQueuePrompt(number, data, options) {
    const { output: prompt, workflow } = data;
    const body = {
      client_id: this.clientId ?? "",
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

    const response = await this.fetchApi(MODAL_ROUTE, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (response.status !== 200) {
      return originalQueuePrompt(number, data, options);
    }

    return await response.json();
  };

  api.__modalQueuePromptPatched = true;
}

app.registerExtension({
  name: "Comfy.ModalSync.Toggle",

  async init() {
    patchQueuePrompt();
  },

  async nodeCreated(node) {
    decorateNode(node);
  },
});
