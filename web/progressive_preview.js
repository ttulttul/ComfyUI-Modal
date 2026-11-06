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
 * @typedef {{stage: HTMLElement, image: HTMLImageElement, waiting: HTMLElement,
 *   counter: HTMLElement, root: HTMLElement,
 *   actionButtons: Record<string, HTMLButtonElement>}} ProgressivePreviewViewer
 * @typedef {{node: object, root: HTMLElement, header: HTMLElement, grid: HTMLElement,
 *   cells: HTMLElement[], viewer: ProgressivePreviewViewer, state: SlotState,
 *   viewerIndex: number|null, focusSlotIndex: number|null, visible: boolean,
 *   widget: object|null, finishedPromptId: string|null,
 *   imageElements: Map<string, HTMLImageElement>}} ProgressivePreviewPanel
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
 * Replace a slot state with the final images ComfyUI delivered, one image per slot.
 * @param {SlotState} state
 * @param {PreviewImage[]} images
 * @returns {boolean} Whether there were images to show.
 */
function applyFinalPreviewImages(state, images) {
  const finalImages = Array.isArray(images) ? images.filter((image) => image?.filename) : [];
  if (finalImages.length === 0) {
    return false;
  }
  state.slotCount = finalImages.length;
  state.slots = new Map(finalImages.map((image, index) => [index, [image]]));
  return true;
}

/**
 * Return the slot index reached by stepping through all slots with wrap-around.
 * @param {SlotState} state
 * @param {number} index
 * @param {number} delta
 * @returns {number}
 */
function stepSlotIndex(state, index, delta) {
  const count = Math.max(1, state.slotCount);
  return (((index + delta) % count) + count) % count;
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
    .comfy-progressive-preview { position: relative; display: flex; flex-direction: column;
      gap: 6px; width: 100%; height: 100%; box-sizing: border-box; padding: 4px;
      overflow: hidden; outline: none; }
    .comfy-progressive-preview[hidden] { display: none; }
    .comfy-progressive-preview-header { font-size: 11px; opacity: 0.75;
      color: var(--input-text, #ddd); white-space: nowrap; }
    .comfy-progressive-preview-grid { display: grid; gap: 4px; flex: 1; min-height: 0;
      overflow: auto; align-content: start; }
    .comfy-progressive-preview-cell { position: relative; aspect-ratio: 1; min-height: 0;
      padding: 0; border: 0; border-radius: 4px; overflow: hidden; cursor: zoom-in;
      background: var(--comfy-input-bg, #222); display: flex; align-items: center;
      justify-content: center; pointer-events: auto; touch-action: manipulation; }
    .comfy-progressive-preview-cell.is-empty { border: 1px dashed var(--border-color, #555);
      animation: comfy-progressive-preview-pulse 1.6s ease-in-out infinite; }
    .comfy-progressive-preview-cell img { width: 100%; height: 100%; object-fit: contain; }
    .comfy-progressive-preview-label { position: absolute; left: 4px; bottom: 2px;
      font-size: 10px; color: var(--input-text, #ddd); opacity: 0.7; }
    .comfy-progressive-preview-badge { position: absolute; right: 4px; top: 3px; font-size: 10px;
      padding: 0 4px; border-radius: 6px; background: rgba(0, 0, 0, 0.6); color: #fff; }
    .comfy-progressive-preview-viewer { position: absolute; inset: 0; display: flex;
      align-items: center; justify-content: center; border-radius: 4px;
      background: var(--comfy-menu-bg, #1a1a1a); }
    .comfy-progressive-preview-viewer[hidden] { display: none; }
    .comfy-progressive-preview-stage { position: absolute; inset: 0; display: flex;
      align-items: center; justify-content: center; cursor: zoom-out; pointer-events: auto; }
    .comfy-progressive-preview-stage img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .comfy-progressive-preview-waiting { font-size: 12px; opacity: 0.7;
      color: var(--input-text, #ddd); }
    .comfy-progressive-preview-nav, .comfy-progressive-preview-close {
      position: absolute; display: flex; align-items: center; justify-content: center;
      border: 0; border-radius: 50%; background: rgba(0, 0, 0, 0.55); color: #fff;
      cursor: pointer; pointer-events: auto; touch-action: manipulation; opacity: 0.8; }
    .comfy-progressive-preview-nav:hover, .comfy-progressive-preview-close:hover { opacity: 1; }
    .comfy-progressive-preview-nav { top: 50%; width: 32px; height: 32px; margin-top: -16px;
      font-size: 20px; }
    .comfy-progressive-preview-nav.is-prev { left: 6px; }
    .comfy-progressive-preview-nav.is-next { right: 6px; }
    .comfy-progressive-preview-close { top: 6px; right: 6px; width: 24px; height: 24px;
      font-size: 13px; }
    .comfy-progressive-preview-actions { position: absolute; top: 6px; left: 6px; display: flex;
      gap: 4px; }
    .comfy-progressive-preview-action { width: 26px; height: 24px; border: 0; border-radius: 5px;
      background: rgba(0, 0, 0, 0.55); color: #fff; cursor: pointer; font-size: 13px;
      pointer-events: auto; touch-action: manipulation; opacity: 0.8; }
    .comfy-progressive-preview-action:hover { opacity: 1; }
    .comfy-progressive-preview-action:disabled { opacity: 0.3; cursor: default; }
    .comfy-progressive-preview-counter { position: absolute; bottom: 6px; left: 50%;
      transform: translateX(-50%); padding: 1px 8px; border-radius: 8px; font-size: 11px;
      background: rgba(0, 0, 0, 0.55); color: #fff; }
    @keyframes comfy-progressive-preview-pulse { 50% { opacity: 0.45; } }
  `;
  document.head?.appendChild(style);
}

/**
 * Create an element with a class name and optional text.
 * @param {string} tagName
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function createPreviewElement(tagName, className, text) {
  const element = document.createElement(tagName);
  element.className = className;
  if (text !== undefined) {
    element.textContent = text;
  }
  if (tagName === "button") {
    element.type = "button";
  }
  return element;
}

/**
 * Stop pointer gestures on interactive controls from dragging the node.
 * @param {HTMLElement} root
 */
function isolateProgressivePreviewPointerEvents(root) {
  for (const eventName of ["pointerdown", "pointermove", "pointerup", "wheel"]) {
    root.addEventListener(eventName, (event) => {
      if (event.target?.closest?.("button, .comfy-progressive-preview-stage")) {
        event.stopPropagation();
      }
    });
  }
}

/**
 * Build the enlarged single-image viewer that overlays the slot grid.
 * @param {ProgressivePreviewPanel} panel
 * @returns {ProgressivePreviewViewer}
 */
function createProgressivePreviewViewer(panel) {
  const root = createPreviewElement("div", "comfy-progressive-preview-viewer");
  root.hidden = true;
  root.style.display = "none";
  const stage = createPreviewElement("div", "comfy-progressive-preview-stage");
  const image = createPreviewElement("img", "");
  const waiting = createPreviewElement("div", "comfy-progressive-preview-waiting");
  stage.appendChild(image);
  stage.appendChild(waiting);
  const prev = createPreviewElement("button", "comfy-progressive-preview-nav is-prev", "‹");
  const next = createPreviewElement("button", "comfy-progressive-preview-nav is-next", "›");
  const close = createPreviewElement("button", "comfy-progressive-preview-close", "✕");
  const counter = createPreviewElement("div", "comfy-progressive-preview-counter");
  const actions = createPreviewElement("div", "comfy-progressive-preview-actions");
  const actionButtons = {
    open: createPreviewElement("button", "comfy-progressive-preview-action", "↗"),
    save: createPreviewElement("button", "comfy-progressive-preview-action", "⤓"),
    copy: createPreviewElement("button", "comfy-progressive-preview-action", "⧉"),
  };
  actionButtons.open.title = "Open image in a new tab";
  actionButtons.save.title = "Save image";
  actionButtons.copy.title = "Copy image";
  for (const [name, button] of Object.entries(actionButtons)) {
    actions.appendChild(button);
    button.addEventListener("click", () => void runProgressivePreviewAction(panel, name, button));
  }
  prev.title = "Previous image (←)";
  next.title = "Next image (→)";
  close.title = "Back to all images (Esc)";
  for (const child of [stage, prev, next, close, actions, counter]) {
    root.appendChild(child);
  }
  stage.addEventListener("click", () => closeProgressivePreviewViewer(panel));
  close.addEventListener("click", () => closeProgressivePreviewViewer(panel));
  prev.addEventListener("click", () => stepProgressivePreviewViewer(panel, -1));
  next.addEventListener("click", () => stepProgressivePreviewViewer(panel, 1));
  return { root, stage, image, waiting, counter, actionButtons };
}

/**
 * Handle keyboard navigation while the viewer is open.
 * @param {ProgressivePreviewPanel} panel
 * @param {KeyboardEvent} event
 */
function handleProgressivePreviewKeydown(panel, event) {
  if (panel.viewerIndex === null) {
    return;
  }
  const actions = {
    ArrowLeft: () => stepProgressivePreviewViewer(panel, -1),
    ArrowRight: () => stepProgressivePreviewViewer(panel, 1),
    Escape: () => closeProgressivePreviewViewer(panel),
  };
  const action = actions[event.key];
  if (!action) {
    return;
  }
  event.preventDefault?.();
  event.stopPropagation?.();
  action();
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
  suppressNativeImagePreview(node);
  const root = createPreviewElement("div", "comfy-progressive-preview");
  root.tabIndex = 0;
  root.hidden = true;
  root.style.display = "none";
  const header = createPreviewElement("div", "comfy-progressive-preview-header");
  const grid = createPreviewElement("div", "comfy-progressive-preview-grid");
  root.appendChild(header);
  root.appendChild(grid);
  /** @type {ProgressivePreviewPanel} */
  const panel = {
    node,
    root,
    header,
    grid,
    cells: [],
    viewer: /** @type {any} */ (null),
    state: createSlotState(),
    viewerIndex: null,
    focusSlotIndex: null,
    visible: false,
    widget: null,
    finishedPromptId: null,
    imageElements: new Map(),
  };
  panel.viewer = createProgressivePreviewViewer(panel);
  root.appendChild(panel.viewer.root);
  isolateProgressivePreviewPointerEvents(root);
  root.addEventListener("keydown", (event) => handleProgressivePreviewKeydown(panel, event));
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
    cell = createPreviewElement("button", "comfy-progressive-preview-cell");
    cell.addEventListener("click", () => openProgressivePreviewViewer(panel, index));
    cell.addEventListener("pointerenter", () => focusProgressivePreviewSlot(panel, index));
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
  cell.title = firstImage ? `Item ${index + 1} · click to enlarge` : `Item ${index + 1} · waiting for remote result`;
  if (firstImage) {
    const image = document.createElement("img");
    image.src = imageKey;
    image.alt = `Remote preview item ${index + 1}`;
    cell.appendChild(image);
  }
  if (images.length > 1) {
    cell.appendChild(createPreviewElement("span", "comfy-progressive-preview-badge", `×${images.length}`));
  }
  cell.appendChild(createPreviewElement("span", "comfy-progressive-preview-label", String(index + 1)));
}

/**
 * Render the enlarged viewer for the currently selected slot.
 * @param {ProgressivePreviewPanel} panel
 */
function renderProgressivePreviewViewer(panel) {
  const { viewer, state, viewerIndex } = panel;
  const open = viewerIndex !== null && state.slotCount > 0;
  viewer.root.hidden = !open;
  viewer.root.style.display = open ? "" : "none";
  if (!open) {
    return;
  }
  const image = (state.slots.get(viewerIndex) ?? [])[0];
  const url = image ? progressivePreviewImageUrl(image) : "";
  if (viewer.image.src !== url) {
    viewer.image.src = url;
  }
  viewer.image.alt = `Remote preview item ${viewerIndex + 1}`;
  viewer.image.style.display = image ? "" : "none";
  viewer.waiting.textContent = image ? "" : `Waiting for item ${viewerIndex + 1}…`;
  viewer.waiting.style.display = image ? "none" : "";
  viewer.counter.textContent = `${viewerIndex + 1} / ${state.slotCount}`;
  for (const button of Object.values(viewer.actionButtons)) {
    button.disabled = !image;
  }
}

/**
 * Render a panel's header, slot grid, and viewer from its state.
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
  if (panel.viewerIndex !== null && panel.viewerIndex >= state.slotCount) {
    panel.viewerIndex = null;
  }
  renderProgressivePreviewViewer(panel);
  syncProgressivePreviewMenuImages(panel);
}

/**
 * Enlarge one slot to fill the panel.
 * @param {ProgressivePreviewPanel} panel
 * @param {number} index
 */
function openProgressivePreviewViewer(panel, index) {
  if (index < 0 || index >= panel.state.slotCount) {
    return;
  }
  panel.viewerIndex = index;
  renderProgressivePreviewViewer(panel);
  syncProgressivePreviewMenuImages(panel);
  panel.root.focus?.({ preventScroll: true });
  panel.node.setDirtyCanvas?.(true, true);
}

/**
 * Move the enlarged viewer to a neighboring slot, wrapping at either end.
 * @param {ProgressivePreviewPanel} panel
 * @param {number} delta
 */
function stepProgressivePreviewViewer(panel, delta) {
  if (panel.viewerIndex === null) {
    return;
  }
  panel.viewerIndex = stepSlotIndex(panel.state, panel.viewerIndex, delta);
  renderProgressivePreviewViewer(panel);
  syncProgressivePreviewMenuImages(panel);
}

/**
 * Return from the enlarged viewer to the slot grid.
 * @param {ProgressivePreviewPanel} panel
 */
function closeProgressivePreviewViewer(panel) {
  panel.viewerIndex = null;
  renderProgressivePreviewViewer(panel);
  syncProgressivePreviewMenuImages(panel);
  panel.node.setDirtyCanvas?.(true, true);
}

/**
 * Show or hide the progressive panel.
 * @param {ProgressivePreviewPanel} panel
 * @param {boolean} visible
 */
function setProgressivePreviewVisible(panel, visible) {
  if (panel.visible === visible) {
    return;
  }
  panel.visible = visible;
  panel.root.hidden = !visible;
  panel.root.style.display = visible ? "" : "none";
  const node = panel.node;
  if (visible) {
    node.setSize?.([
      Math.max(Number(node.size?.[0]) || 0, PROGRESSIVE_PREVIEW_MIN_WIDTH),
      Math.max(Number(node.size?.[1]) || 0, PROGRESSIVE_PREVIEW_MIN_HEIGHT + 40),
    ]);
  }
  node.setDirtyCanvas?.(true, true);
}

/**
 * Suppress ComfyUI's native image preview so the panel is the only display.
 * @param {object} node
 */
function suppressNativeImagePreview(node) {
  node.hideOutputImages = true;
  if (!node.__progressivePreviewDrawSuppressed) {
    node.onDrawBackground = function skipNativeImagePreview() {};
    node.__progressivePreviewDrawSuppressed = true;
  }
}

/**
 * Return the first image of one slot, if it has arrived.
 * @param {ProgressivePreviewPanel} panel
 * @param {number|null} index
 * @returns {PreviewImage|null}
 */
function progressivePreviewSlotImage(panel, index) {
  return index === null ? null : (panel.state.slots.get(index) ?? [])[0] ?? null;
}

/**
 * Return a cached <img> element for one preview URL.
 * @param {ProgressivePreviewPanel} panel
 * @param {string} url
 * @returns {HTMLImageElement}
 */
function progressivePreviewImageElement(panel, url) {
  let element = panel.imageElements.get(url);
  if (!element) {
    element = new Image();
    element.src = url;
    panel.imageElements.set(url, element);
  }
  return element;
}

/**
 * Expose the slot images through node.imgs so ComfyUI's Open/Copy/Save Image menu works.
 * The selected image is the enlarged one, else the last hovered slot, else the first.
 * @param {ProgressivePreviewPanel} panel
 */
function syncProgressivePreviewMenuImages(panel) {
  const slotIndices = [];
  const elements = [];
  for (let index = 0; index < panel.state.slotCount; index += 1) {
    const image = progressivePreviewSlotImage(panel, index);
    if (image) {
      slotIndices.push(index);
      elements.push(progressivePreviewImageElement(panel, progressivePreviewImageUrl(image)));
    }
  }
  const node = panel.node;
  node.imgs = elements.length ? elements : undefined;
  const selectedSlot = panel.viewerIndex ?? panel.focusSlotIndex;
  const position = slotIndices.indexOf(selectedSlot);
  node.imageIndex = elements.length ? Math.max(0, position) : null;
}

/**
 * Remember the hovered slot as the context-menu target.
 * @param {ProgressivePreviewPanel} panel
 * @param {number} index
 */
function focusProgressivePreviewSlot(panel, index) {
  panel.focusSlotIndex = index;
  syncProgressivePreviewMenuImages(panel);
}

/**
 * Return a PNG blob for an image URL, converting other formats through a canvas.
 * @param {string} url
 * @returns {Promise<Blob>}
 */
async function fetchProgressivePreviewPng(url) {
  const response = await fetch(url);
  const blob = await response.blob();
  if (blob.type === "image/png") {
    return blob;
  }
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  return await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

/**
 * Open, save, or copy one preview image.
 * @param {PreviewImage} image
 * @param {"open"|"save"|"copy"} action
 * @returns {Promise<void>}
 */
async function performProgressivePreviewImageAction(image, action) {
  const url = progressivePreviewImageUrl(image);
  if (action === "open") {
    window.open(url, "_blank");
    return;
  }
  if (action === "save") {
    const link = document.createElement("a");
    link.href = url;
    link.download = String(image.filename ?? "image.png");
    document.body.appendChild(link);
    link.click();
    link.remove();
    return;
  }
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
    throw new Error("Image clipboard access is unavailable in this browser context.");
  }
  const png = await fetchProgressivePreviewPng(url);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
}

/**
 * Run one viewer toolbar action on the enlarged image and flash the result.
 * @param {ProgressivePreviewPanel} panel
 * @param {"open"|"save"|"copy"} action
 * @param {HTMLElement} button
 * @returns {Promise<boolean>} Whether the action succeeded.
 */
async function runProgressivePreviewAction(panel, action, button) {
  const image = progressivePreviewSlotImage(panel, panel.viewerIndex);
  if (!image) {
    return false;
  }
  const label = button.textContent;
  try {
    await performProgressivePreviewImageAction(image, action);
    button.textContent = "✓";
    return true;
  } catch (error) {
    console.error(`Remote progressive preview could not ${action} the image.`, error);
    button.textContent = "!";
    return false;
  } finally {
    setTimeout(() => {
      button.textContent = label;
    }, 1200);
  }
}

/**
 * Return whether an event belongs to a prompt whose final images were already shown.
 * Replayed events after a browser refocus must not overwrite the final images.
 * @param {ProgressivePreviewPanel} panel
 * @param {Record<string, any>} detail
 * @returns {boolean}
 */
function isFinishedProgressivePreviewEvent(panel, detail) {
  const promptId = detail?.prompt_id == null ? null : String(detail.prompt_id);
  return promptId !== null && promptId === panel.finishedPromptId;
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
  if (!panel || isFinishedProgressivePreviewEvent(panel, detail)) {
    return false;
  }
  const previousPromptId = panel.state.promptId;
  if (!applyProgressivePreviewEvent(panel.state, detail)) {
    return false;
  }
  if (detail.phase === "start" || panel.state.promptId !== previousPromptId) {
    panel.viewerIndex = null;
  }
  suppressNativeImagePreview(node);
  renderProgressivePreviewPanel(panel);
  setProgressivePreviewVisible(panel, true);
  app.graph?.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * Show ComfyUI's final images in the slot grid, which stays the node's only display.
 * @param {object} node
 * @param {Record<string, any>|undefined} output
 * @returns {boolean} Whether final images were shown.
 */
function finishProgressivePreview(node, output) {
  const panel = ensureProgressivePreviewPanel(node);
  if (!panel) {
    return false;
  }
  suppressNativeImagePreview(node);
  panel.finishedPromptId = panel.state.promptId;
  if (!applyFinalPreviewImages(panel.state, output?.images)) {
    return false;
  }
  renderProgressivePreviewPanel(panel);
  setProgressivePreviewVisible(panel, true);
  return true;
}

/**
 * Show outputs ComfyUI restored without an executed event, such as after a page reload.
 * @param {object} node
 */
function restoreStoredProgressivePreview(node) {
  const panel = progressivePreviewPanels.get(node);
  if (panel?.visible) {
    return;
  }
  const stored = app.nodeOutputs?.[String(node.id)] ?? app.nodeOutputs?.[node.id];
  if (stored?.images?.length) {
    finishProgressivePreview(node, stored);
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
      finishProgressivePreview(this, args[0]);
      return result;
    };
  },
  async nodeCreated(node) {
    if (!isProgressivePreviewNode(node)) {
      return;
    }
    ensureProgressivePreviewPanel(node);
    setTimeout(() => restoreStoredProgressivePreview(node), 0);
  },
});
