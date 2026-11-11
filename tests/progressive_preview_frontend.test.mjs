import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runInThisContext } from "node:vm";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.className = "";
    this.textContent = "";
    this.title = "";
    this.src = "";
    this.listeners = new Map();
    this.focused = false;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = [...children];
  }

  addEventListener(name, handler) {
    const handlers = this.listeners.get(name) ?? [];
    handlers.push(handler);
    this.listeners.set(name, handlers);
  }

  dispatch(name, event = {}) {
    let stopped = false;
    let prevented = false;
    const fullEvent = {
      target: this,
      stopPropagation: () => {
        stopped = true;
      },
      preventDefault: () => {
        prevented = true;
      },
      ...event,
    };
    for (const handler of this.listeners.get(name) ?? []) {
      handler(fullEvent);
    }
    return { stopped, prevented };
  }

  focus() {
    this.focused = true;
  }
}

const createdElements = [];
globalThis.document = {
  head: new FakeElement("head"),
  body: new FakeElement("body"),
  createElement: (tagName) => {
    const element = new FakeElement(tagName);
    createdElements.push(element);
    element.click = () => {
      element.clicked = true;
    };
    element.remove = () => {
      element.removed = true;
    };
    return element;
  },
  getElementById: () => null,
  visibilityState: "visible",
  listeners: new Map(),
  addEventListener(name, handler) {
    this.listeners.set(name, handler);
  },
};
const observedRoots = new Set();
let intersectionCallback = null;
globalThis.IntersectionObserver = class IntersectionObserver {
  constructor(callback) {
    intersectionCallback = callback;
  }

  observe(element) {
    observedRoots.add(element);
  }

  unobserve(element) {
    observedRoots.delete(element);
  }
};
const setOnScreen = (element, isIntersecting) => {
  intersectionCallback([{ target: element, isIntersecting }]);
};
globalThis.Image = class Image {
  constructor() {
    this.src = "";
  }
};
const openedUrls = [];
globalThis.window = { open: (url) => openedUrls.push(url) };
const clipboardWrites = [];
globalThis.ClipboardItem = class ClipboardItem {
  constructor(items) {
    this.items = items;
  }
};
Object.defineProperty(globalThis, "navigator", {
  value: {
    clipboard: {
      write: async (items) => clipboardWrites.push(items),
      writeText: async (text) => clipboardWrites.push(text),
    },
  },
  configurable: true,
});
const objectUrlBlobs = [];
URL.createObjectURL = (blob) => {
  objectUrlBlobs.push(blob);
  return `blob:preview-${objectUrlBlobs.length}`;
};
URL.revokeObjectURL = () => {};
globalThis.fetch = async (url) => ({ blob: async () => ({ type: "image/png", url }) });

const listeners = new Map();
const extensions = [];
const graphNodes = new Map();
globalThis.__progressiveAppStub = {
  registerExtension: (extension) => extensions.push(extension),
  rootGraph: { getNodeById: (id) => graphNodes.get(String(id)) ?? null },
  graph: { setDirtyCanvas() {} },
  nodeOutputs: {},
};
globalThis.__progressiveApiStub = {
  addEventListener: (name, handler) => listeners.set(name, handler),
  apiURL: (route) => `/api${route}`,
};

const sourcePath = path.join(repoRoot, "web", "progressive_preview.js");
const originalSource = await readFile(sourcePath, "utf8");
runInThisContext(
  [
    "const app = globalThis.__progressiveAppStub;",
    "const api = globalThis.__progressiveApiStub;",
    originalSource.replace(/^import .*?;\n/gm, ""),
    "globalThis.__progressiveExports = {",
    "  applyProgressivePreviewEvent, applyFinalPreviewOutput, createSlotState, stepSlotIndex,",
    "  slotGridColumns, filledSlotCount, progressivePreviewHeaderText, progressivePreviewFileUrl,",
    "  normalizePreviewEntry, fitSlotGrid, applyProgressivePreviewGridLayout,",
    "  adoptProgressivePreviewImageAspect,",
    "  resolveProgressivePreviewNode, handleProgressivePreviewEvent, finishProgressivePreview,",
    "  restoreStoredProgressivePreview, progressivePreviewPanels, runProgressivePreviewAction,",
    "};",
  ].join("\n"),
  { filename: sourcePath },
);
const preview = globalThis.__progressiveExports;
const viewUrl = (name) => `/api/view?filename=${name}&subfolder=&type=temp`;

// Slot state: start lays out empty slots, slots fill out of order, a new prompt resets.
const state = preview.createSlotState();
assert.equal(preview.applyProgressivePreviewEvent(state, { phase: "start", prompt_id: "p1", slot_count: 4 }), true);
assert.equal(state.slotCount, 4);
assert.equal(preview.progressivePreviewHeaderText(state), "Remote preview · 0/4 received");
preview.applyProgressivePreviewEvent(state, {
  phase: "slot", prompt_id: "p1", slot_index: 2, slot_count: 4, images: [{ filename: "c.png" }],
});
assert.equal(preview.filledSlotCount(state), 1);
preview.applyProgressivePreviewEvent(state, {
  phase: "slot", prompt_id: "p2", slot_index: 0, images: [{ filename: "a.png" }],
});
assert.equal(state.promptId, "p2");
assert.equal(state.slotCount, 1);
assert.equal(preview.progressivePreviewHeaderText(state), "Remote preview · all 1 received");
assert.equal(preview.applyProgressivePreviewEvent(state, { phase: "bogus", prompt_id: "p2" }), false);

// Final entries replace the slots one entry per slot; empty finals are ignored.
assert.equal(preview.applyFinalPreviewOutput(state, { images: [] }), false);
assert.equal(preview.applyFinalPreviewOutput(state, { images: [{ filename: "x.png" }, { filename: "y.png" }] }), true);
assert.equal(state.slotCount, 2);
assert.equal(state.slots.get(1)[0].filename, "y.png");
assert.equal(state.slots.get(1)[0].kind, "image");
assert.equal(
  preview.applyFinalPreviewOutput(state, {
    modal_preview: [{ kind: "text", text: "hi" }, { kind: "audio", filename: "a.flac" }, { kind: "bogus" }],
    images: [],
  }),
  true,
);
assert.equal(state.slotCount, 2);
assert.deepEqual(state.slots.get(0)[0], { kind: "text", text: "hi" });
assert.equal(state.slots.get(1)[0].kind, "audio");

// Entries normalize by kind; untyped file entries are images from earlier versions.
assert.deepEqual(preview.normalizePreviewEntry({ filename: "a.png" }), {
  kind: "image", filename: "a.png", subfolder: "", type: "temp",
});
assert.deepEqual(preview.normalizePreviewEntry({ kind: "text", text: 3 }), { kind: "text", text: "3" });
assert.equal(preview.normalizePreviewEntry({ kind: "audio" }), null);
assert.equal(preview.normalizePreviewEntry(null), null);
assert.equal(preview.progressivePreviewHeaderText(preview.createSlotState()), "Waiting for remote results…");

// Navigation wraps in both directions.
assert.equal(preview.stepSlotIndex({ slotCount: 3 }, 2, 1), 0);
assert.equal(preview.stepSlotIndex({ slotCount: 3 }, 0, -1), 2);

// Layout and URLs.
assert.equal(preview.slotGridColumns(0), 1);
assert.equal(preview.slotGridColumns(4), 2);
assert.equal(preview.slotGridColumns(5), 3);
assert.equal(
  preview.progressivePreviewFileUrl({ filename: "a b.png", subfolder: "", type: "temp" }),
  "/api/view?filename=a+b.png&subfolder=&type=temp",
);

// The grid picks the column count that makes slots largest while all of them fit.
assert.deepEqual(preview.fitSlotGrid(12, 400, 300), { columns: 4, cellWidth: 97, cellHeight: 97 });
assert.deepEqual(preview.fitSlotGrid(12, 900, 120), { columns: 12, cellWidth: 71, cellHeight: 71 });
assert.deepEqual(preview.fitSlotGrid(12, 120, 900), { columns: 1, cellWidth: 71, cellHeight: 71 });
// Portrait items get portrait cells.
assert.deepEqual(preview.fitSlotGrid(4, 400, 400, 0.75), { columns: 2, cellWidth: 148, cellHeight: 197 });
// Too many slots for the space keeps a minimum cell size and lets the grid scroll.
assert.equal(preview.fitSlotGrid(400, 100, 100).cellWidth, 32);
assert.equal(preview.fitSlotGrid(0, 100, 100).columns, 1);

// Subgraph node ids resolve through each subgraph node.
const innerNode = { id: 18, comfyClass: "RemoteProgressivePreview" };
graphNodes.set("12", { id: 12, subgraph: { getNodeById: (id) => (String(id) === "18" ? innerNode : null) } });
assert.equal(preview.resolveProgressivePreviewNode("12:18"), innerNode);
assert.equal(preview.resolveProgressivePreviewNode("99"), null);

// End to end: events mount a panel with placeholders and fill it.
const widgets = [];
const makeNode = (id) => ({
  id,
  comfyClass: "RemoteProgressivePreview",
  size: [100, 100],
  imgs: ["stale"],
  addDOMWidget: (name, type, element, options) => {
    const widget = { name, type, element, options };
    widgets.push(widget);
    return widget;
  },
  setSize(size) {
    this.size = size;
  },
  setDirtyCanvas() {},
});
const node = makeNode(7);
graphNodes.set("7", node);
for (const extension of extensions) {
  extension.setup?.();
}
const handler = listeners.get("modal_progressive_preview");
assert.equal(handler({ detail: { phase: "start", prompt_id: "p3", node: "7", slot_count: 3 } }), true);
const panel = preview.progressivePreviewPanels.get(node);
assert.equal(node.hideOutputImages, true);
assert.equal(node.imgs, undefined);
assert.equal(node.imageIndex, null);
assert.equal(typeof node.onDrawBackground, "function");
assert.equal(node.onDrawBackground(), undefined);
assert.equal(panel.visible, true);
assert.equal(widgets[0].options.getMinHeight() > 0, true);
assert.equal(panel.grid.children.length, 3);
assert.equal(panel.grid.children.every((cell) => cell.className.includes("is-empty")), true);
handler({
  detail: {
    phase: "slot", prompt_id: "p3", node: "7", slot_index: 1, slot_count: 3,
    images: [{ filename: "b.png", type: "temp" }, { filename: "b2.png", type: "temp" }],
  },
});
const filledCell = panel.grid.children[1];
assert.equal(filledCell.tagName, "button");
assert.equal(filledCell.children[0].src, viewUrl("b.png"));
assert.equal(filledCell.children[1].textContent, "×2");
assert.equal(panel.header.textContent, "Remote preview · 1/3 received");
// ComfyUI's Open/Copy/Save Image menu reads node.imgs[node.imageIndex].
assert.equal(node.imgs.length, 1);
assert.equal(node.imgs[0].src, viewUrl("b.png"));
assert.equal(node.imageIndex, 0);
assert.equal(handler({ detail: { phase: "start", prompt_id: "p3", node: "99", slot_count: 3 } }), false);

// Clicking a slot enlarges it; buttons and keys navigate with wrap-around; Esc returns to the grid.
assert.equal(panel.viewer.root.style.display, "none");
filledCell.dispatch("click");
assert.equal(panel.viewerIndex, 1);
assert.equal(panel.viewer.root.style.display, "");
assert.equal(panel.viewer.image.src, viewUrl("b.png"));
assert.equal(panel.viewer.counter.textContent, "2 / 3");
assert.equal(panel.root.focused, true);
const rightKey = panel.root.dispatch("keydown", { key: "ArrowRight" });
assert.deepEqual(rightKey, { stopped: true, prevented: true });
assert.equal(panel.viewerIndex, 2);
assert.equal(panel.viewer.image.style.display, "none");
assert.equal(panel.viewer.waiting.textContent, "Waiting for item 3…");
panel.root.dispatch("keydown", { key: "ArrowRight" });
assert.equal(panel.viewerIndex, 0);
panel.root.dispatch("keydown", { key: "ArrowLeft" });
assert.equal(panel.viewerIndex, 2);
const prevButton = panel.viewer.root.children.find((child) => child.className.includes("is-prev"));
prevButton.dispatch("click");
assert.equal(panel.viewerIndex, 1);
assert.equal(panel.root.dispatch("keydown", { key: "a" }).stopped, false);
// A slot that fills while it is enlarged appears without reopening.
panel.root.dispatch("keydown", { key: "ArrowRight" });
handler({ detail: { phase: "slot", prompt_id: "p3", node: "7", slot_index: 2, slot_count: 3, images: [{ filename: "c.png" }] } });
assert.equal(panel.viewer.image.src, viewUrl("c.png"));
assert.equal(panel.viewer.image.style.display, "");
// The context menu targets the enlarged image.
assert.equal(node.imgs[node.imageIndex].src, viewUrl("c.png"));
// Viewer toolbar: open, save, and copy act on the enlarged image.
const actions = panel.viewer.actionButtons;
assert.equal(actions.save.disabled, false);
assert.equal(await preview.runProgressivePreviewAction(panel, "open", actions.open), true);
assert.deepEqual(openedUrls, [viewUrl("c.png")]);
assert.equal(await preview.runProgressivePreviewAction(panel, "save", actions.save), true);
const downloadLink = createdElements.filter((element) => element.tagName === "a").at(-1);
assert.equal(downloadLink.href, viewUrl("c.png"));
assert.equal(downloadLink.download, "c.png");
assert.equal(downloadLink.clicked, true);
assert.equal(await preview.runProgressivePreviewAction(panel, "copy", actions.copy), true);
assert.equal(clipboardWrites.length, 1);
assert.equal(clipboardWrites[0][0].items["image/png"].url, viewUrl("c.png"));
assert.equal(actions.copy.textContent, "✓");
panel.root.dispatch("keydown", { key: "ArrowLeft" });
assert.equal(actions.save.disabled, false);
panel.root.dispatch("keydown", { key: "ArrowLeft" });
assert.equal(actions.save.disabled, true);
assert.equal(await preview.runProgressivePreviewAction(panel, "save", actions.save), false);
panel.root.dispatch("keydown", { key: "Escape" });
assert.equal(panel.viewerIndex, null);
// Hovering a grid slot makes it the context-menu target.
panel.grid.children[2].dispatch("pointerenter");
assert.equal(node.imgs[node.imageIndex].src, viewUrl("c.png"));
panel.grid.children[1].dispatch("pointerenter");
assert.equal(node.imgs[node.imageIndex].src, viewUrl("b.png"));
assert.equal(panel.viewer.root.style.display, "none");
assert.equal(panel.root.dispatch("keydown", { key: "ArrowRight" }).stopped, false);
// Pointer gestures on interactive controls do not drag the node.
assert.equal(panel.root.dispatch("pointerdown", { target: { closest: () => filledCell } }).stopped, true);
assert.equal(panel.root.dispatch("pointerdown", { target: { closest: () => null } }).stopped, false);

// The final executed event fills the grid, which stays the only display.
const nodeType = { prototype: {} };
let originalCalled = false;
nodeType.prototype.onExecuted = () => {
  originalCalled = true;
};
await extensions[0].beforeRegisterNodeDef(nodeType, { name: "RemoteProgressivePreview" });
node.imgs = ["native"];
nodeType.prototype.onExecuted.call(node, {
  images: [{ filename: "f1.png", type: "temp" }, { filename: "f2.png", type: "temp" }, { filename: "f3.png", type: "temp" }],
});
assert.equal(originalCalled, true);
assert.equal(node.imgs.length, 3);
assert.equal(node.imgs[2].src, viewUrl("f3.png"));
assert.equal(panel.visible, true);
assert.equal(panel.root.style.display, "");
assert.equal(panel.header.textContent, "Remote preview · all 3 received");
assert.equal(panel.grid.children[0].children[0].src, viewUrl("f1.png"));

// Replayed events for the finished prompt (browser refocus) must not overwrite the final images.
assert.equal(handler({ detail: { phase: "start", prompt_id: "p3", node: "7", slot_count: 3 } }), false);
assert.equal(panel.header.textContent, "Remote preview · all 3 received");

// The next prompt lays out fresh slots and closes any open viewer.
panel.grid.children[0].dispatch("click");
assert.equal(handler({ detail: { phase: "start", prompt_id: "p4", node: "7", slot_count: 2 } }), true);
assert.equal(panel.viewerIndex, null);
assert.equal(panel.grid.children.length, 2);

// Outputs restored without an executed event (page reload) still render in the panel.
const reloadedNode = makeNode(8);
graphNodes.set("8", reloadedNode);
globalThis.__progressiveAppStub.nodeOutputs["8"] = { images: [{ filename: "r.png", type: "temp" }] };
await extensions[0].nodeCreated(reloadedNode);
preview.restoreStoredProgressivePreview(reloadedNode);
const reloadedPanel = preview.progressivePreviewPanels.get(reloadedNode);
assert.equal(reloadedPanel.visible, true);
assert.equal(reloadedPanel.grid.children[0].children[0].src, viewUrl("r.png"));

// Slots that arrive while the panel is off-screen are repainted when it scrolls back into view.
const offscreenNode = makeNode(9);
graphNodes.set("9", offscreenNode);
handler({ detail: { phase: "start", prompt_id: "p5", node: "9", slot_count: 2 } });
const offscreenPanel = preview.progressivePreviewPanels.get(offscreenNode);
assert.equal(observedRoots.has(offscreenPanel.root), true);
setOnScreen(offscreenPanel.root, true);
assert.equal(offscreenPanel.stale, false);
setOnScreen(offscreenPanel.root, false);
handler({ detail: { phase: "slot", prompt_id: "p5", node: "9", slot_index: 0, slot_count: 2, images: [{ filename: "o.png" }] } });
assert.equal(offscreenPanel.stale, true);
const offscreenImage = offscreenPanel.grid.children[0].children[0];
assert.equal(offscreenImage.src, viewUrl("o.png"));
setOnScreen(offscreenPanel.root, true);
assert.equal(offscreenPanel.stale, false);
const repaintedImage = offscreenPanel.grid.children[0].children[0];
assert.notEqual(repaintedImage, offscreenImage);
assert.equal(repaintedImage.src, viewUrl("o.png"));
assert.equal(offscreenPanel.header.textContent, "Remote preview · 1/2 received");
// An on-screen update does not need a repaint.
handler({ detail: { phase: "slot", prompt_id: "p5", node: "9", slot_index: 1, slot_count: 2, images: [{ filename: "o2.png" }] } });
assert.equal(offscreenPanel.stale, false);

// Updates while the browser tab is hidden are repainted when the tab becomes visible.
document.visibilityState = "hidden";
handler({ detail: { phase: "start", prompt_id: "p6", node: "9", slot_count: 1 } });
handler({ detail: { phase: "slot", prompt_id: "p6", node: "9", slot_index: 0, slot_count: 1, images: [{ filename: "h.png" }] } });
assert.equal(offscreenPanel.stale, true);
const hiddenTabImage = offscreenPanel.grid.children[0].children[0];
document.listeners.get("visibilitychange")();
assert.equal(offscreenPanel.stale, true);
document.visibilityState = "visible";
document.listeners.get("visibilitychange")();
assert.equal(offscreenPanel.stale, false);
assert.notEqual(offscreenPanel.grid.children[0].children[0], hiddenTabImage);
assert.equal(offscreenPanel.grid.children[0].children[0].src, viewUrl("h.png"));

// The enlarged viewer image is reassigned on repaint too.
offscreenPanel.grid.children[0].dispatch("click");
setOnScreen(offscreenPanel.root, false);
handler({ detail: { phase: "slot", prompt_id: "p6", node: "9", slot_index: 0, slot_count: 1, images: [{ filename: "h2.png" }] } });
setOnScreen(offscreenPanel.root, true);
assert.equal(offscreenPanel.viewer.image.src, viewUrl("h2.png"));

// Text and audio slots render by kind, and the viewer and toolbar adapt to each.
const mixedNode = makeNode(10);
graphNodes.set("10", mixedNode);
handler({ detail: { phase: "start", prompt_id: "p7", node: "10", slot_count: 3 } });
handler({ detail: { phase: "slot", prompt_id: "p7", node: "10", slot_index: 0, slot_count: 3,
  items: [{ kind: "text", text: "a red fox" }] } });
handler({ detail: { phase: "slot", prompt_id: "p7", node: "10", slot_index: 1, slot_count: 3,
  items: [{ kind: "audio", filename: "clip.flac", type: "temp" }] } });
handler({ detail: { phase: "slot", prompt_id: "p7", node: "10", slot_index: 2, slot_count: 3,
  items: [{ kind: "image", filename: "m.png", type: "temp" }] } });
const mixedPanel = preview.progressivePreviewPanels.get(mixedNode);
const [textCell, audioCell, imageCell] = mixedPanel.grid.children;
assert.equal(textCell.className.includes("is-text"), true);
assert.equal(textCell.children[0].textContent, "a red fox");
assert.equal(audioCell.className.includes("is-audio"), true);
assert.equal(audioCell.children[0].textContent, "♪");
assert.equal(imageCell.children[0].src, viewUrl("m.png"));
assert.equal(mixedPanel.header.textContent, "Remote preview · all 3 received");
// ComfyUI's image menu sees only the image slot.
assert.equal(mixedNode.imgs.length, 1);
assert.equal(mixedNode.imgs[0].src, viewUrl("m.png"));
const mixedViewer = mixedPanel.viewer;
textCell.dispatch("click");
assert.equal(mixedViewer.text.textContent, "a red fox");
assert.equal(mixedViewer.text.style.display, "");
assert.equal(mixedViewer.image.style.display, "none");
assert.equal(mixedViewer.audio.style.display, "none");
assert.deepEqual(
  Object.fromEntries(Object.entries(mixedViewer.actionButtons).map(([name, button]) => [name, button.disabled])),
  { open: true, save: false, copy: false },
);
assert.equal(mixedViewer.actionButtons.copy.title, "Copy text");
assert.equal(await preview.runProgressivePreviewAction(mixedPanel, "copy", mixedViewer.actionButtons.copy), true);
assert.equal(clipboardWrites.at(-1), "a red fox");
assert.equal(await preview.runProgressivePreviewAction(mixedPanel, "save", mixedViewer.actionButtons.save), true);
const textDownload = createdElements.filter((element) => element.tagName === "a").at(-1);
assert.equal(textDownload.download, "remote-preview-item-1.txt");
assert.equal(textDownload.href.startsWith("blob:"), true);
assert.equal(await objectUrlBlobs.at(-1).text(), "a red fox");
assert.equal(await preview.runProgressivePreviewAction(mixedPanel, "open", mixedViewer.actionButtons.open), false);
// Clicking the text keeps the viewer open so it can be selected; clicking elsewhere closes it.
mixedViewer.stage.dispatch("click", { target: { closest: () => mixedViewer.text } });
assert.equal(mixedPanel.viewerIndex, 0);
mixedPanel.root.dispatch("keydown", { key: "ArrowRight" });
assert.equal(mixedViewer.audio.src, viewUrl("clip.flac"));
assert.equal(mixedViewer.audio.style.display, "");
assert.equal(mixedViewer.text.style.display, "none");
assert.equal(mixedViewer.actionButtons.copy.disabled, true);
assert.equal(mixedViewer.actionButtons.open.title, "Open audio in a new tab");
assert.equal(await preview.runProgressivePreviewAction(mixedPanel, "open", mixedViewer.actionButtons.open), true);
assert.equal(openedUrls.at(-1), viewUrl("clip.flac"));
let audioPaused = false;
mixedViewer.audio.pause = () => {
  audioPaused = true;
};
mixedPanel.root.dispatch("keydown", { key: "ArrowRight" });
assert.equal(audioPaused, true);
assert.equal(mixedViewer.image.src, viewUrl("m.png"));
assert.equal(mixedViewer.actionButtons.copy.disabled, false);
mixedViewer.stage.dispatch("click");
assert.equal(mixedPanel.viewerIndex, null);
// Final outputs keep each kind, and a reload restores them from the stored output.
nodeType.prototype.onExecuted.call(mixedNode, {
  modal_preview: [{ kind: "text", text: "done" }, { kind: "image", filename: "f.png", type: "temp" }],
  images: [{ filename: "f.png", type: "temp" }],
});
assert.equal(mixedPanel.grid.children.length, 2);
assert.equal(mixedPanel.grid.children[0].children[0].textContent, "done");
const restoredTextNode = makeNode(11);
graphNodes.set("11", restoredTextNode);
globalThis.__progressiveAppStub.nodeOutputs["11"] = { modal_preview: [{ kind: "text", text: "saved" }] };
await extensions[0].nodeCreated(restoredTextNode);
preview.restoreStoredProgressivePreview(restoredTextNode);
const restoredTextPanel = preview.progressivePreviewPanels.get(restoredTextNode);
assert.equal(restoredTextPanel.visible, true);
assert.equal(restoredTextPanel.grid.children[0].children[0].textContent, "saved");

// Resizing the node refits the grid to the panel; the first loaded image sets the cell shape.
const fitPanel = preview.progressivePreviewPanels.get(mixedNode);
fitPanel.grid.clientWidth = 400;
fitPanel.grid.clientHeight = 300;
handler({ detail: { phase: "start", prompt_id: "p8", node: "10", slot_count: 12 } });
assert.equal(fitPanel.grid.style.gridTemplateColumns, "repeat(4, 97px)");
assert.equal(fitPanel.grid.style.gridAutoRows, "97px");
preview.adoptProgressivePreviewImageAspect(fitPanel, { naturalWidth: 768, naturalHeight: 1024 });
assert.equal(fitPanel.grid.style.gridTemplateColumns, "repeat(4, 73px)");
assert.equal(fitPanel.grid.style.gridAutoRows, "97px");
fitPanel.grid.clientWidth = 0;
preview.applyProgressivePreviewGridLayout(fitPanel);
assert.equal(fitPanel.grid.style.gridTemplateColumns, "repeat(4, minmax(0, 1fr))");
// Waiting slots are drawn without a running animation, which would repaint the whole Nodes 2.0 graph every frame.
const panelStyles = document.head.children.map((child) => child.textContent).join("\n");
assert.equal(/animation|@keyframes/.test(panelStyles), false);

// Removing the node stops observing its panel.
offscreenNode.onRemoved();
assert.equal(observedRoots.has(offscreenPanel.root), false);

console.log("progressive preview frontend checks passed");
