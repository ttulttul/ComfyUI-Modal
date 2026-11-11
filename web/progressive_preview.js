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

/** @type {Set<ProgressivePreviewPanel>} Panels mounted on live nodes, for tab-refocus repaints. */
const liveProgressivePreviewPanels = new Set();

/** @type {WeakMap<Element, ProgressivePreviewPanel>} */
const progressivePreviewPanelsByRoot = new WeakMap();

/** @type {IntersectionObserver|null} */
let progressivePreviewObserver = null;

/** Toolbar actions each preview kind supports in the enlarged viewer. */
const PREVIEW_ACTIONS_BY_KIND = {
  image: ["open", "save", "copy"],
  audio: ["open", "save"],
  text: ["save", "copy"],
};

/**
 * @typedef {{kind: "image"|"audio"|"text", filename?: string, subfolder?: string,
 *   type?: string, text?: string}} PreviewEntry
 * @typedef {{promptId: string|null, slotCount: number, slots: Map<number, PreviewEntry[]>}} SlotState
 * @typedef {{stage: HTMLElement, image: HTMLImageElement, audio: HTMLAudioElement,
 *   text: HTMLElement, waiting: HTMLElement, counter: HTMLElement, root: HTMLElement,
 *   actionButtons: Record<string, HTMLButtonElement>}} ProgressivePreviewViewer
 * @typedef {{node: object, root: HTMLElement, header: HTMLElement, grid: HTMLElement,
 *   cells: HTMLElement[], viewer: ProgressivePreviewViewer, state: SlotState,
 *   viewerIndex: number|null, focusSlotIndex: number|null, visible: boolean,
 *   widget: object|null, finishedPromptId: string|null,
 *   imageElements: Map<string, HTMLImageElement>, onScreen: boolean,
 *   stale: boolean}} ProgressivePreviewPanel
 */

/**
 * Create an empty progressive preview slot state.
 * @returns {SlotState}
 */
function createSlotState() {
  return { promptId: null, slotCount: 0, slots: new Map() };
}

/**
 * Return one preview entry in canonical form, or null when it cannot be shown.
 * Entries without a kind are images, as sent by earlier versions of the node.
 * @param {Record<string, any>|null|undefined} entry
 * @returns {PreviewEntry|null}
 */
function normalizePreviewEntry(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  if (entry.kind === "text") {
    return { kind: "text", text: String(entry.text ?? "") };
  }
  if (!entry.filename) {
    return null;
  }
  return {
    kind: entry.kind === "audio" ? "audio" : "image",
    filename: String(entry.filename),
    subfolder: String(entry.subfolder ?? ""),
    type: String(entry.type ?? "temp"),
  };
}

/**
 * Return the showable entries of a list of raw preview entries.
 * @param {unknown} entries
 * @returns {PreviewEntry[]}
 */
function normalizePreviewEntries(entries) {
  return Array.isArray(entries) ? entries.map(normalizePreviewEntry).filter(Boolean) : [];
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
  state.slots.set(slotIndex, normalizePreviewEntries(detail.items ?? detail.images));
  state.slotCount = Math.max(state.slotCount, declaredCount, slotIndex + 1);
  return true;
}

/**
 * Replace a slot state with the final entries ComfyUI delivered, one entry per slot.
 * Outputs saved before the node took other types carry only an ``images`` list.
 * @param {SlotState} state
 * @param {Record<string, any>|undefined} output
 * @returns {boolean} Whether there were entries to show.
 */
function applyFinalPreviewOutput(state, output) {
  const entries = normalizePreviewEntries(output?.modal_preview ?? output?.images);
  if (entries.length === 0) {
    return false;
  }
  state.slotCount = entries.length;
  state.slots = new Map(entries.map((entry, index) => [index, [entry]]));
  return true;
}

/**
 * Return whether a stored node output has anything for the panel to show.
 * @param {Record<string, any>|undefined} output
 * @returns {boolean}
 */
function hasStoredPreviewOutput(output) {
  return Boolean(output?.modal_preview?.length || output?.images?.length);
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
    return "Waiting for remote results…";
  }
  const filled = filledSlotCount(state);
  if (filled >= state.slotCount) {
    return `Remote preview · all ${state.slotCount} received`;
  }
  return `Remote preview · ${filled}/${state.slotCount} received`;
}

/**
 * Build the ComfyUI /view URL for one saved preview image or audio file.
 * @param {PreviewEntry} entry
 * @returns {string}
 */
function progressivePreviewFileUrl(entry) {
  const params = new URLSearchParams({
    filename: String(entry.filename ?? ""),
    subfolder: String(entry.subfolder ?? ""),
    type: String(entry.type ?? "temp"),
  });
  const path = `/view?${params.toString()}`;
  return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

/**
 * Return a key that changes whenever an entry's displayed content changes.
 * @param {PreviewEntry} entry
 * @returns {string}
 */
function previewEntryKey(entry) {
  return entry.kind === "text" ? `text:${entry.text}` : `${entry.kind}:${progressivePreviewFileUrl(entry)}`;
}

/**
 * Return the word used for an entry in titles and tooltips.
 * @param {PreviewEntry|null} entry
 * @returns {string}
 */
function previewEntryNoun(entry) {
  return { audio: "audio", text: "text" }[entry?.kind] ?? "image";
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
    .comfy-progressive-preview-text-thumb { width: 100%; height: 100%; box-sizing: border-box;
      padding: 4px 4px 14px; overflow: hidden; font-size: 10px; line-height: 1.3; text-align: left;
      white-space: pre-wrap; word-break: break-word; color: var(--input-text, #ddd); }
    .comfy-progressive-preview-audio-thumb { font-size: 28px; opacity: 0.8;
      color: var(--input-text, #ddd); }
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
    .comfy-progressive-preview-stage audio { width: calc(100% - 96px); }
    .comfy-progressive-preview-text { position: absolute; inset: 36px 44px 30px; overflow: auto;
      font-size: 12px; line-height: 1.4; text-align: left; white-space: pre-wrap;
      word-break: break-word; user-select: text; cursor: text; color: var(--input-text, #ddd); }
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
 * Return whether a click inside the viewer stage landed on content the user interacts with.
 * @param {Event} event
 * @returns {boolean}
 */
function isInteractiveViewerTarget(event) {
  return Boolean(event?.target?.closest?.("audio, .comfy-progressive-preview-text"));
}

/**
 * Build the enlarged single-entry viewer that overlays the slot grid.
 * @param {ProgressivePreviewPanel} panel
 * @returns {ProgressivePreviewViewer}
 */
function createProgressivePreviewViewer(panel) {
  const root = createPreviewElement("div", "comfy-progressive-preview-viewer");
  root.hidden = true;
  root.style.display = "none";
  const stage = createPreviewElement("div", "comfy-progressive-preview-stage");
  const image = createPreviewElement("img", "");
  const audio = createPreviewElement("audio", "");
  audio.controls = true;
  audio.preload = "metadata";
  const text = createPreviewElement("div", "comfy-progressive-preview-text");
  const waiting = createPreviewElement("div", "comfy-progressive-preview-waiting");
  for (const child of [image, audio, text, waiting]) {
    child.style.display = "none";
    stage.appendChild(child);
  }
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
  for (const [name, button] of Object.entries(actionButtons)) {
    actions.appendChild(button);
    button.addEventListener("click", () => void runProgressivePreviewAction(panel, name, button));
  }
  prev.title = "Previous item (←)";
  next.title = "Next item (→)";
  close.title = "Back to all items (Esc)";
  for (const child of [stage, prev, next, close, actions, counter]) {
    root.appendChild(child);
  }
  stage.addEventListener("click", (event) => {
    if (!isInteractiveViewerTarget(event)) {
      closeProgressivePreviewViewer(panel);
    }
  });
  close.addEventListener("click", () => closeProgressivePreviewViewer(panel));
  prev.addEventListener("click", () => stepProgressivePreviewViewer(panel, -1));
  next.addEventListener("click", () => stepProgressivePreviewViewer(panel, 1));
  return { root, stage, image, audio, text, waiting, counter, actionButtons };
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
    onScreen: typeof IntersectionObserver !== "function",
    stale: false,
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
  trackProgressivePreviewVisibility(panel);
  return panel;
}

/**
 * Return the shared observer that reports when panels scroll into or out of view.
 * @returns {IntersectionObserver|null}
 */
function progressivePreviewVisibilityObserver() {
  if (!progressivePreviewObserver && typeof IntersectionObserver === "function") {
    progressivePreviewObserver = new IntersectionObserver(handleProgressivePreviewIntersections);
  }
  return progressivePreviewObserver;
}

/**
 * Record panel on-screen changes and repaint panels that changed while off-screen.
 * @param {IntersectionObserverEntry[]} entries
 */
function handleProgressivePreviewIntersections(entries) {
  for (const entry of entries) {
    const panel = progressivePreviewPanelsByRoot.get(entry.target);
    if (!panel) {
      continue;
    }
    panel.onScreen = entry.isIntersecting;
    if (panel.onScreen) {
      repaintStaleProgressivePreview(panel);
    }
  }
}

/**
 * Watch a panel's visibility and stop watching once its node is removed.
 * @param {ProgressivePreviewPanel} panel
 */
function trackProgressivePreviewVisibility(panel) {
  liveProgressivePreviewPanels.add(panel);
  progressivePreviewPanelsByRoot.set(panel.root, panel);
  progressivePreviewVisibilityObserver()?.observe(panel.root);
  const node = panel.node;
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function onProgressivePreviewRemoved(...args) {
    liveProgressivePreviewPanels.delete(panel);
    progressivePreviewObserver?.unobserve(panel.root);
    return originalOnRemoved?.apply(this, args);
  };
}

/**
 * Return whether the panel is currently painted: on screen in a visible browser tab.
 * @param {ProgressivePreviewPanel} panel
 * @returns {boolean}
 */
function isProgressivePreviewOnScreen(panel) {
  const tabHidden = typeof document !== "undefined" && document.visibilityState === "hidden";
  return panel.onScreen && !tabHidden;
}

/**
 * Rebuild a panel's images if they changed while it was out of view.
 * Browsers do not reliably paint image content that arrived while its layer was hidden,
 * so every image element is recreated rather than trusting the existing DOM.
 * @param {ProgressivePreviewPanel} panel
 * @returns {boolean} Whether the panel was repainted.
 */
function repaintStaleProgressivePreview(panel) {
  if (!panel.stale || !isProgressivePreviewOnScreen(panel)) {
    return false;
  }
  panel.stale = false;
  for (const cell of panel.cells) {
    if (cell) {
      delete cell.dataset.entryKey;
    }
  }
  clearProgressivePreviewImageSource(panel.viewer.image);
  clearProgressivePreviewImageSource(panel.viewer.audio);
  renderProgressivePreviewPanel(panel);
  panel.node.setDirtyCanvas?.(true, true);
  return true;
}

/**
 * Drop a media element's source so the next render assigns it afresh and the browser repaints it.
 * @param {HTMLImageElement|HTMLAudioElement} image
 */
function clearProgressivePreviewImageSource(image) {
  if (typeof image.removeAttribute === "function") {
    image.removeAttribute("src");
  } else {
    image.src = "";
  }
}

/** Repaint every stale on-screen panel, such as after the browser tab becomes visible again. */
function repaintStaleProgressivePreviews() {
  for (const panel of liveProgressivePreviewPanels) {
    repaintStaleProgressivePreview(panel);
  }
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
 * Build the small grid thumbnail for one entry.
 * @param {PreviewEntry} entry
 * @param {number} index
 * @param {HTMLElement} cell
 * @returns {HTMLElement}
 */
function createPreviewEntryThumbnail(entry, index, cell) {
  if (entry.kind === "text") {
    return createPreviewElement("div", "comfy-progressive-preview-text-thumb", entry.text);
  }
  if (entry.kind === "audio") {
    return createPreviewElement("div", "comfy-progressive-preview-audio-thumb", "♪");
  }
  const image = document.createElement("img");
  image.addEventListener?.("error", () => {
    delete cell.dataset.entryKey;
  });
  image.src = progressivePreviewFileUrl(entry);
  image.alt = `Remote preview item ${index + 1}`;
  return image;
}

/**
 * Render one slot cell as an empty placeholder or a thumbnail of its first entry.
 * @param {HTMLElement} cell
 * @param {number} index
 * @param {PreviewEntry[]} entries
 */
function renderProgressivePreviewCell(cell, index, entries) {
  const firstEntry = entries[0];
  const entryKey = firstEntry ? previewEntryKey(firstEntry) : "";
  if (cell.dataset.entryKey === entryKey && cell.dataset.entryCount === String(entries.length)) {
    return;
  }
  cell.dataset.entryKey = entryKey;
  cell.dataset.entryCount = String(entries.length);
  cell.replaceChildren?.();
  cell.className = `comfy-progressive-preview-cell ${firstEntry ? `is-${firstEntry.kind}` : "is-empty"}`;
  cell.title = firstEntry
    ? `Item ${index + 1} · ${previewEntryNoun(firstEntry)} · click to enlarge`
    : `Item ${index + 1} · waiting for remote result`;
  if (firstEntry) {
    cell.appendChild(createPreviewEntryThumbnail(firstEntry, index, cell));
  }
  if (entries.length > 1) {
    cell.appendChild(createPreviewElement("span", "comfy-progressive-preview-badge", `×${entries.length}`));
  }
  cell.appendChild(createPreviewElement("span", "comfy-progressive-preview-label", String(index + 1)));
}

/**
 * Show one entry, or the waiting message, on the viewer stage.
 * @param {ProgressivePreviewViewer} viewer
 * @param {PreviewEntry|null} entry
 * @param {number} index
 */
function showViewerEntry(viewer, entry, index) {
  const kind = entry?.kind ?? null;
  const url = entry && kind !== "text" ? progressivePreviewFileUrl(entry) : "";
  if (kind === "image" && viewer.image.src !== url) {
    viewer.image.src = url;
  }
  if (kind !== "audio") {
    viewer.audio.pause?.();
  } else if (viewer.audio.src !== url) {
    viewer.audio.src = url;
  }
  viewer.text.textContent = kind === "text" ? entry.text : "";
  viewer.image.alt = `Remote preview item ${index + 1}`;
  viewer.image.style.display = kind === "image" ? "" : "none";
  viewer.audio.style.display = kind === "audio" ? "" : "none";
  viewer.text.style.display = kind === "text" ? "" : "none";
  viewer.waiting.textContent = entry ? "" : `Waiting for item ${index + 1}…`;
  viewer.waiting.style.display = entry ? "none" : "";
}

/**
 * Enable and label the toolbar actions the shown entry supports.
 * @param {ProgressivePreviewViewer} viewer
 * @param {PreviewEntry|null} entry
 */
function syncViewerActions(viewer, entry) {
  const allowed = entry ? PREVIEW_ACTIONS_BY_KIND[entry.kind] ?? [] : [];
  const noun = previewEntryNoun(entry);
  const titles = { open: `Open ${noun} in a new tab`, save: `Save ${noun}`, copy: `Copy ${noun}` };
  for (const [name, button] of Object.entries(viewer.actionButtons)) {
    button.disabled = !allowed.includes(name);
    button.title = titles[name];
  }
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
    viewer.audio.pause?.();
    return;
  }
  const entry = progressivePreviewSlotEntry(panel, viewerIndex);
  showViewerEntry(viewer, entry, viewerIndex);
  viewer.counter.textContent = `${viewerIndex + 1} / ${state.slotCount}`;
  syncViewerActions(viewer, entry);
}

/**
 * Render a panel's header, slot grid, and viewer from its state.
 * @param {ProgressivePreviewPanel} panel
 */
function renderProgressivePreviewPanel(panel) {
  const { state } = panel;
  if (!isProgressivePreviewOnScreen(panel)) {
    panel.stale = true;
  }
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
 * Return the first entry of one slot, if it has arrived.
 * @param {ProgressivePreviewPanel} panel
 * @param {number|null} index
 * @returns {PreviewEntry|null}
 */
function progressivePreviewSlotEntry(panel, index) {
  return index === null ? null : (panel.state.slots.get(index) ?? [])[0] ?? null;
}

/**
 * Return the first entry of one slot when it is an image.
 * @param {ProgressivePreviewPanel} panel
 * @param {number|null} index
 * @returns {PreviewEntry|null}
 */
function progressivePreviewSlotImage(panel, index) {
  const entry = progressivePreviewSlotEntry(panel, index);
  return entry?.kind === "image" ? entry : null;
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
 * Expose the image slots through node.imgs so ComfyUI's Open/Copy/Save Image menu works.
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
      elements.push(progressivePreviewImageElement(panel, progressivePreviewFileUrl(image)));
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
 * Download a URL under a file name through a temporary link.
 * @param {string} url
 * @param {string} filename
 */
function downloadProgressivePreviewUrl(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/**
 * Save or copy one text entry.
 * @param {PreviewEntry} entry
 * @param {"save"|"copy"} action
 * @param {number} index
 * @returns {Promise<void>}
 */
async function performProgressivePreviewTextAction(entry, action, index) {
  if (action === "copy") {
    if (!navigator.clipboard?.writeText) {
      throw new Error("Text clipboard access is unavailable in this browser context.");
    }
    await navigator.clipboard.writeText(entry.text);
    return;
  }
  const url = URL.createObjectURL(new Blob([entry.text], { type: "text/plain;charset=utf-8" }));
  try {
    downloadProgressivePreviewUrl(url, `remote-preview-item-${index + 1}.txt`);
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

/**
 * Open, save, or copy one preview entry.
 * @param {PreviewEntry} entry
 * @param {"open"|"save"|"copy"} action
 * @param {number} index
 * @returns {Promise<void>}
 */
async function performProgressivePreviewEntryAction(entry, action, index) {
  if (!(PREVIEW_ACTIONS_BY_KIND[entry.kind] ?? []).includes(action)) {
    throw new Error(`Cannot ${action} a ${entry.kind} preview.`);
  }
  if (entry.kind === "text") {
    await performProgressivePreviewTextAction(entry, action, index);
    return;
  }
  const url = progressivePreviewFileUrl(entry);
  if (action === "open") {
    window.open(url, "_blank");
    return;
  }
  if (action === "save") {
    downloadProgressivePreviewUrl(url, String(entry.filename ?? "preview"));
    return;
  }
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
    throw new Error("Image clipboard access is unavailable in this browser context.");
  }
  const png = await fetchProgressivePreviewPng(url);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
}

/**
 * Run one viewer toolbar action on the enlarged entry and flash the result.
 * @param {ProgressivePreviewPanel} panel
 * @param {"open"|"save"|"copy"} action
 * @param {HTMLElement} button
 * @returns {Promise<boolean>} Whether the action succeeded.
 */
async function runProgressivePreviewAction(panel, action, button) {
  const entry = progressivePreviewSlotEntry(panel, panel.viewerIndex);
  if (!entry) {
    return false;
  }
  const label = button.textContent;
  try {
    await performProgressivePreviewEntryAction(entry, action, panel.viewerIndex);
    button.textContent = "✓";
    return true;
  } catch (error) {
    console.error(`Remote progressive preview could not ${action} the ${entry.kind}.`, error);
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
 * Show ComfyUI's final entries in the slot grid, which stays the node's only display.
 * @param {object} node
 * @param {Record<string, any>|undefined} output
 * @returns {boolean} Whether final entries were shown.
 */
function finishProgressivePreview(node, output) {
  const panel = ensureProgressivePreviewPanel(node);
  if (!panel) {
    return false;
  }
  suppressNativeImagePreview(node);
  panel.finishedPromptId = panel.state.promptId;
  if (!applyFinalPreviewOutput(panel.state, output)) {
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
  if (hasStoredPreviewOutput(stored)) {
    finishProgressivePreview(node, stored);
  }
}

app.registerExtension({
  name: "comfy.modal.progressivePreview",
  setup() {
    installProgressivePreviewStyles();
    api.addEventListener(PROGRESSIVE_PREVIEW_EVENT, handleProgressivePreviewEvent);
    if (typeof document !== "undefined") {
      document.addEventListener?.("visibilitychange", repaintStaleProgressivePreviews);
    }
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
