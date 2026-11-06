import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PROGRESSIVE_PREVIEW_NODE_CLASS = "RemoteProgressivePreview";
const PROGRESSIVE_PREVIEW_EVENT = "modal_progressive_preview";
const PROGRESSIVE_PREVIEW_WIDGET_NAME = "progressive_preview";
const PROGRESSIVE_PREVIEW_STYLE_ID = "comfy-progressive-preview-style";
const PROGRESSIVE_PREVIEW_MIN_HEIGHT = 160;
const PROGRESSIVE_PREVIEW_MIN_WIDTH = 280;

/** @type {WeakMap<object, ProgressivePreviewPanel>} */
const progressivePreviewPanels = new WeakMap();

/**
 * @typedef {{filename: string, subfolder?: string, type?: string}} PreviewImage
 * @typedef {{promptId: string|null, slotCount: number, slots: Map<number, PreviewImage[]>}} SlotState
 * @typedef {{node: object, root: HTMLElement, header: HTMLElement, grid: HTMLElement,
 *   cells: HTMLElement[], state: SlotState, visible: boolean, widget: object|null}} ProgressivePreviewPanel
 */

/**
 * Create an empty progressive preview slot state.
 * @returns {SlotState}
 */
function createSlotState() {
  return { promptId: null, slotCount: 0, slots: new Map() };
}

/**
 * Apply one progressive preview websocket payload to a slot state.
 * @param {SlotState} state
 * @param {Record<string, any>} detail
 * @returns {boolean} Whether the payload was applied.
 */
function applyProgressivePreviewEvent(state, detail) {
  if (!detail || typeof detail !== "object") {
    return false;
  }
  const promptId = detail.prompt_id == null ? null : String(detail.prompt_id);
  if (promptId !== state.promptId) {
    state.promptId = promptId;
    state.slotCount = 0;
    state.slots = new Map();
  }
  const declaredCount = Math.max(0, Number(detail.slot_count) || 0);
  if (detail.phase === "start") {
    state.slotCount = declaredCount;
    state.slots = new Map();
    return true;
  }
  if (detail.phase !== "slot") {
    return false;
  }
  const slotIndex = Math.max(0, Number(detail.slot_index) || 0);
  const images = Array.isArray(detail.images) ? detail.images : [];
  state.slots.set(slotIndex, images);
  state.slotCount = Math.max(state.slotCount, declaredCount, slotIndex + 1);
  return true;
}

/**
 * Return a near-square column count for a slot grid.
 * @param {number} slotCount
 * @returns {number}
 */
function slotGridColumns(slotCount) {
  return Math.max(1, Math.ceil(Math.sqrt(Math.max(1, slotCount))));
}

/**
 * Return how many slots have received images.
 * @param {SlotState} state
 * @returns {number}
 */
function filledSlotCount(state) {
  let filled = 0;
  for (let index = 0; index < state.slotCount; index += 1) {
    if ((state.slots.get(index) ?? []).length > 0) {
      filled += 1;
    }
  }
  return filled;
}

/**
 * Return the header text for a slot state.
 * @param {SlotState} state
 * @returns {string}
 */
function progressivePreviewHeaderText(state) {
  if (state.slotCount <= 0) {
    return "Waiting for remote images…";
  }
  const filled = filledSlotCount(state);
  if (filled >= state.slotCount) {
    return `Remote preview · all ${state.slotCount} received`;
  }
  return `Remote preview · ${filled}/${state.slotCount} received`;
}

/**
 * Build the ComfyUI /view URL for one saved preview image.
 * @param {PreviewImage} image
 * @returns {string}
 */
function progressivePreviewImageUrl(image) {
  const params = new URLSearchParams({
    filename: String(image.filename ?? ""),
    subfolder: String(image.subfolder ?? ""),
    type: String(image.type ?? "temp"),
  });
  const path = `/view?${params.toString()}`;
  return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

/**
 * Look up one node in a graph by string or numeric id.
 * @param {object|null|undefined} graph
 * @param {string} id
 * @returns {object|null}
 */
function graphNodeById(graph, id) {
  if (!graph) {
    return null;
  }
  const direct = graph.getNodeById?.(id) ?? null;
  if (direct) {
    return direct;
  }
  const numericId = Number(id);
  return Number.isFinite(numericId) ? graph.getNodeById?.(numericId) ?? null : null;
}

/**
 * Resolve a prompt node id, including "subgraph:node" paths, to a live graph node.
 * @param {string|number} nodeIdValue
 * @returns {object|null}
 */
function resolveProgressivePreviewNode(nodeIdValue) {
  const segments = String(nodeIdValue ?? "").split(":").filter(Boolean);
  let graph = app.rootGraph ?? app.graph;
  let node = null;
  for (let index = 0; index < segments.length; index += 1) {
    node = graphNodeById(graph, segments[index]);
    if (!node) {
      return null;
    }
    if (index < segments.length - 1) {
      graph = node.subgraph ?? null;
    }
  }
  return node;
}

/**
 * Return whether a graph node is a progressive preview node.
 * @param {object|null} node
 * @returns {boolean}
 */
function isProgressivePreviewNode(node) {
  return (node?.comfyClass ?? node?.type) === PROGRESSIVE_PREVIEW_NODE_CLASS;
}

/** Install the panel stylesheet once. */
function installProgressivePreviewStyles() {
  if (typeof document === "undefined" || document.getElementById?.(PROGRESSIVE_PREVIEW_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = PROGRESSIVE_PREVIEW_STYLE_ID;
  style.textContent = `
    .comfy-progressive-preview { display: flex; flex-direction: column; gap: 6px;
      width: 100%; height: 100%; box-sizing: border-box; padding: 4px; overflow: hidden; }
    .comfy-progressive-preview[hidden] { display: none; }
    .comfy-progressive-preview-header { font-size: 11px; opacity: 0.75;
      color: var(--input-text, #ddd); white-space: nowrap; }
    .comfy-progressive-preview-grid { display: grid; gap: 4px; flex: 1; min-height: 0; }
    .comfy-progressive-preview-cell { position: relative; aspect-ratio: 1; min-height: 0;
      border-radius: 4px; overflow: hidden; background: var(--comfy-input-bg, #222);
      display: flex; align-items: center; justify-content: center; }
    .comfy-progressive-preview-cell.is-empty { border: 1px dashed var(--border-color, #555);
      animation: comfy-progressive-preview-pulse 1.6s ease-in-out infinite; }
    .comfy-progressive-preview-cell img { width: 100%; height: 100%; object-fit: contain; }
    .comfy-progressive-preview-label { position: absolute; left: 4px; bottom: 2px;
      font-size: 10px; color: var(--input-text, #ddd); opacity: 0.7; }
    .comfy-progressive-preview-badge { position: absolute; right: 4px; top: 3px; font-size: 10px;
      padding: 0 4px; border-radius: 6px; background: rgba(0, 0, 0, 0.6); color: #fff; }
    @keyframes comfy-progressive-preview-pulse { 50% { opacity: 0.45; } }
  `;
  document.head?.appendChild(style);
}

/**
 * Create the DOM panel and widget for one progressive preview node.
 * @param {object} node
 * @returns {ProgressivePreviewPanel|null}
 */
function ensureProgressivePreviewPanel(node) {
  const existing = progressivePreviewPanels.get(node);
  if (existing) {
    return existing;
  }
  if (typeof document === "undefined" || typeof node?.addDOMWidget !== "function") {
    return null;
  }
  const root = document.createElement("div");
  root.className = "comfy-progressive-preview";
  root.hidden = true;
  const header = document.createElement("div");
  header.className = "comfy-progressive-preview-header";
  const grid = document.createElement("div");
  grid.className = "comfy-progressive-preview-grid";
  root.appendChild(header);
  root.appendChild(grid);
  /** @type {ProgressivePreviewPanel} */
  const panel = { node, root, header, grid, cells: [], state: createSlotState(), visible: false, widget: null };
  try {
    panel.widget = node.addDOMWidget(PROGRESSIVE_PREVIEW_WIDGET_NAME, "progressive-preview", root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => (panel.visible ? PROGRESSIVE_PREVIEW_MIN_HEIGHT : 0),
    });
  } catch (error) {
    console.warn("Unable to mount the remote progressive preview panel.", error);
    return null;
  }
  if (panel.widget) {
    panel.widget.serialize = false;
  }
  progressivePreviewPanels.set(node, panel);
  return panel;
}

/**
 * Return the cell element for one slot, creating it when needed.
 * @param {ProgressivePreviewPanel} panel
 * @param {number} index
 * @returns {HTMLElement}
 */
function progressivePreviewCell(panel, index) {
  let cell = panel.cells[index];
  if (!cell) {
    cell = document.createElement("div");
    panel.cells[index] = cell;
  }
  return cell;
}

/**
 * Render one slot cell as an empty placeholder or its first image.
 * @param {HTMLElement} cell
 * @param {number} index
 * @param {PreviewImage[]} images
 */
function renderProgressivePreviewCell(cell, index, images) {
  const firstImage = images[0];
  const imageKey = firstImage ? progressivePreviewImageUrl(firstImage) : "";
  if (cell.dataset.imageKey === imageKey && cell.dataset.imageCount === String(images.length)) {
    return;
  }
  cell.dataset.imageKey = imageKey;
  cell.dataset.imageCount = String(images.length);
  cell.replaceChildren?.();
  cell.className = `comfy-progressive-preview-cell${firstImage ? "" : " is-empty"}`;
  cell.title = firstImage ? `Item ${index + 1}` : `Item ${index + 1} · waiting for remote result`;
  if (firstImage) {
    const image = document.createElement("img");
    image.src = imageKey;
    image.alt = `Remote preview item ${index + 1}`;
    cell.appendChild(image);
  }
  if (images.length > 1) {
    const badge = document.createElement("span");
    badge.className = "comfy-progressive-preview-badge";
    badge.textContent = `×${images.length}`;
    cell.appendChild(badge);
  }
  const label = document.createElement("span");
  label.className = "comfy-progressive-preview-label";
  label.textContent = String(index + 1);
  cell.appendChild(label);
}

/**
 * Render a panel's header and slot grid from its state.
 * @param {ProgressivePreviewPanel} panel
 */
function renderProgressivePreviewPanel(panel) {
  const { state } = panel;
  panel.header.textContent = progressivePreviewHeaderText(state);
  panel.grid.style.gridTemplateColumns = `repeat(${slotGridColumns(state.slotCount)}, minmax(0, 1fr))`;
  const cells = [];
  for (let index = 0; index < state.slotCount; index += 1) {
    const cell = progressivePreviewCell(panel, index);
    renderProgressivePreviewCell(cell, index, state.slots.get(index) ?? []);
    cells.push(cell);
  }
  panel.cells.length = state.slotCount;
  panel.grid.replaceChildren?.(...cells);
}

/**
 * Show or hide the progressive panel, handing the node back to the native preview when hidden.
 * @param {ProgressivePreviewPanel} panel
 * @param {boolean} visible
 */
function setProgressivePreviewVisible(panel, visible) {
  if (panel.visible === visible) {
    return;
  }
  panel.visible = visible;
  panel.root.hidden = !visible;
  const node = panel.node;
  if (visible) {
    node.imgs = undefined;
    node.imageIndex = null;
    node.setSize?.([
      Math.max(Number(node.size?.[0]) || 0, PROGRESSIVE_PREVIEW_MIN_WIDTH),
      Math.max(Number(node.size?.[1]) || 0, PROGRESSIVE_PREVIEW_MIN_HEIGHT + 40),
    ]);
  }
  node.setDirtyCanvas?.(true, true);
}

/**
 * Handle one progressive preview websocket event.
 * @param {{detail?: Record<string, any>}} event
 * @returns {boolean} Whether a node panel was updated.
 */
function handleProgressivePreviewEvent(event) {
  const detail = event?.detail;
  const node = resolveProgressivePreviewNode(detail?.node);
  if (!isProgressivePreviewNode(node)) {
    return false;
  }
  const panel = ensureProgressivePreviewPanel(node);
  if (!panel || !applyProgressivePreviewEvent(panel.state, detail)) {
    return false;
  }
  renderProgressivePreviewPanel(panel);
  setProgressivePreviewVisible(panel, true);
  app.graph?.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * Hide a node's progressive panel once ComfyUI delivers its final images.
 * @param {object} node
 */
function finishProgressivePreview(node) {
  const panel = progressivePreviewPanels.get(node);
  if (panel) {
    setProgressivePreviewVisible(panel, false);
  }
}

app.registerExtension({
  name: "comfy.modal.progressivePreview",
  setup() {
    installProgressivePreviewStyles();
    api.addEventListener(PROGRESSIVE_PREVIEW_EVENT, handleProgressivePreviewEvent);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== PROGRESSIVE_PREVIEW_NODE_CLASS) {
      return;
    }
    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function onProgressivePreviewExecuted(...args) {
      const result = originalOnExecuted?.apply(this, args);
      finishProgressivePreview(this);
      return result;
    };
  },
});
