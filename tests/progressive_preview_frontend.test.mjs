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

globalThis.document = {
  head: new FakeElement("head"),
  createElement: (tagName) => new FakeElement(tagName),
  getElementById: () => null,
};

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
    "  applyProgressivePreviewEvent, applyFinalPreviewImages, createSlotState, stepSlotIndex,",
    "  slotGridColumns, filledSlotCount, progressivePreviewHeaderText, progressivePreviewImageUrl,",
    "  resolveProgressivePreviewNode, handleProgressivePreviewEvent, finishProgressivePreview,",
    "  restoreStoredProgressivePreview, progressivePreviewPanels,",
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

// Final images replace the slots one image per slot; empty finals are ignored.
assert.equal(preview.applyFinalPreviewImages(state, []), false);
assert.equal(preview.applyFinalPreviewImages(state, [{ filename: "x.png" }, { filename: "y.png" }]), true);
assert.equal(state.slotCount, 2);
assert.equal(state.slots.get(1)[0].filename, "y.png");

// Navigation wraps in both directions.
assert.equal(preview.stepSlotIndex({ slotCount: 3 }, 2, 1), 0);
assert.equal(preview.stepSlotIndex({ slotCount: 3 }, 0, -1), 2);

// Layout and URLs.
assert.equal(preview.slotGridColumns(0), 1);
assert.equal(preview.slotGridColumns(4), 2);
assert.equal(preview.slotGridColumns(5), 3);
assert.equal(
  preview.progressivePreviewImageUrl({ filename: "a b.png", subfolder: "", type: "temp" }),
  "/api/view?filename=a+b.png&subfolder=&type=temp",
);

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
panel.root.dispatch("keydown", { key: "Escape" });
assert.equal(panel.viewerIndex, null);
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
assert.equal(node.imgs, undefined);
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

console.log("progressive preview frontend checks passed");
