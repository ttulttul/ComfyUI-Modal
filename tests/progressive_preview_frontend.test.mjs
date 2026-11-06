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
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = [...children];
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
    "  applyProgressivePreviewEvent, createSlotState, slotGridColumns, filledSlotCount,",
    "  progressivePreviewHeaderText, progressivePreviewImageUrl, resolveProgressivePreviewNode,",
    "  handleProgressivePreviewEvent, finishProgressivePreview, progressivePreviewPanels,",
    "};",
  ].join("\n"),
  { filename: sourcePath },
);
const preview = globalThis.__progressiveExports;

// Slot state: start lays out empty slots, slots fill out of order, a new prompt resets.
const state = preview.createSlotState();
assert.equal(preview.applyProgressivePreviewEvent(state, { phase: "start", prompt_id: "p1", slot_count: 4 }), true);
assert.equal(state.slotCount, 4);
assert.equal(preview.progressivePreviewHeaderText(state), "Remote preview · 0/4 received");
preview.applyProgressivePreviewEvent(state, {
  phase: "slot", prompt_id: "p1", slot_index: 2, slot_count: 4, images: [{ filename: "c.png" }],
});
assert.equal(preview.filledSlotCount(state), 1);
assert.equal(preview.progressivePreviewHeaderText(state), "Remote preview · 1/4 received");
preview.applyProgressivePreviewEvent(state, {
  phase: "slot", prompt_id: "p2", slot_index: 0, images: [{ filename: "a.png" }],
});
assert.equal(state.promptId, "p2");
assert.equal(state.slotCount, 1);
assert.equal(preview.progressivePreviewHeaderText(state), "Remote preview · all 1 received");
assert.equal(preview.applyProgressivePreviewEvent(state, { phase: "bogus", prompt_id: "p2" }), false);

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

// End to end: events mount a panel with placeholders, fill it, and hand back on executed.
const widgets = [];
const node = {
  id: 7,
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
};
graphNodes.set("7", node);
for (const extension of extensions) {
  extension.setup?.();
}
const handler = listeners.get("modal_progressive_preview");
assert.equal(typeof handler, "function");
assert.equal(handler({ detail: { phase: "start", prompt_id: "p3", node: "7", slot_count: 3 } }), true);
const panel = preview.progressivePreviewPanels.get(node);
assert.equal(widgets.length, 1);
assert.equal(panel.visible, true);
assert.equal(node.imgs, undefined);
assert.equal(widgets[0].options.getMinHeight() > 0, true);
assert.equal(panel.grid.children.length, 3);
assert.equal(panel.grid.children.every((cell) => cell.className.includes("is-empty")), true);
handler({
  detail: { phase: "slot", prompt_id: "p3", node: "7", slot_index: 1, slot_count: 3, images: [{ filename: "b.png", type: "temp" }, { filename: "b2.png", type: "temp" }] },
});
const filledCell = panel.grid.children[1];
assert.equal(filledCell.className.includes("is-empty"), false);
assert.equal(filledCell.children[0].src, "/api/view?filename=b.png&subfolder=&type=temp");
assert.equal(filledCell.children[1].textContent, "×2");
assert.equal(panel.header.textContent, "Remote preview · 1/3 received");
assert.equal(handler({ detail: { phase: "start", prompt_id: "p3", node: "99", slot_count: 3 } }), false);

const nodeType = { prototype: {} };
let originalCalled = false;
nodeType.prototype.onExecuted = () => {
  originalCalled = true;
};
await extensions[0].beforeRegisterNodeDef(nodeType, { name: "RemoteProgressivePreview" });
nodeType.prototype.onExecuted.call(node, { images: [] });
assert.equal(originalCalled, true);
assert.equal(panel.visible, false);
assert.equal(widgets[0].options.getMinHeight(), 0);

console.log("progressive preview frontend checks passed");
