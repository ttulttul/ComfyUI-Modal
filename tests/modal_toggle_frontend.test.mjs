import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

globalThis.__modalAppStub = {
  registerExtension() {},
  graph: {
    rootGraph: null,
    setDirtyCanvas() {},
  },
  canvas: {
    selected_nodes: {},
  },
  rootGraph: null,
};
globalThis.__modalApiStub = {
  addEventListener() {},
  dispatchEvent() {},
};
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};
globalThis.performance = { now: () => 0 };
globalThis.CustomEvent = class CustomEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.detail = init.detail;
  }
};

const sourcePath = path.join(repoRoot, "web", "modal_toggle.js");
const originalSource = await readFile(sourcePath, "utf8");
const transformedSource = `${[
  "const app = globalThis.__modalAppStub;",
  "const api = globalThis.__modalApiStub;",
  "class PromptExecutionError extends Error {}",
  originalSource.replace(/^import .*?;\n/gm, ""),
  "globalThis.__modalToggleExports = {",
  "  ensurePromptState,",
  "  registerPromptComponents,",
  "  resolveComponentNodeIds,",
  "  handleModalProgress,",
  "  handleExecutionPhase,",
  "  clearPromptRemoteStates,",
  "  modalNodeStates,",
  "  modalNodeProgress,",
  "  modalNodeProgressLanes,",
  "  modalNodeBatchProgress,",
  "  modalPromptStates,",
  "  STATE_READY,",
  "  STATE_ACTIVE,",
  "  EXECUTION_PHASE,",
  "};",
].join("\n")}`;
await import(`data:text/javascript;base64,${Buffer.from(transformedSource).toString("base64")}`);

const modalToggle = globalThis.__modalToggleExports;

function resetFrontendState() {
  modalToggle.modalNodeStates.clear();
  modalToggle.modalNodeProgress.clear();
  modalToggle.modalNodeProgressLanes.clear();
  modalToggle.modalNodeBatchProgress.clear();
  modalToggle.modalPromptStates.clear();
}

resetFrontendState();
modalToggle.registerPromptComponents("prompt-a", ["10", "11", "12"], [
  {
    representative_node_id: "11",
    node_ids: ["10", "11", "12"],
  },
]);
assert.deepEqual(modalToggle.resolveComponentNodeIds("prompt-a", "11"), ["10", "11", "12"]);
assert.deepEqual(modalToggle.resolveComponentNodeIds("prompt-a", "10"), ["10", "11", "12"]);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-b", ["10", "11", "12"], [
  {
    representative_node_id: "11",
    node_ids: ["10", "11", "12"],
  },
]);
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-b",
      node: "10",
    },
  },
  modalToggle.EXECUTION_PHASE,
);
assert.equal(modalToggle.modalNodeStates.get("10")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("11")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("12")?.phase, modalToggle.STATE_READY);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-c", ["10", "11", "12"], [
  {
    representative_node_id: "11",
    node_ids: ["10", "11", "12"],
  },
]);
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-c",
    node_id: "11",
    display_node_id: "11",
    real_node_id: "12",
    value: 3,
    max: 9,
    lane_id: "0",
  },
});
assert.equal(modalToggle.modalPromptStates.get("prompt-c")?.activeNodeId, "12");
assert.equal(modalToggle.modalNodeStates.get("10")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("11")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("12")?.phase, modalToggle.STATE_ACTIVE);
assert.equal(modalToggle.modalNodeProgressLanes.has("11"), false);
assert.equal(modalToggle.modalNodeProgressLanes.get("12")?.lanes.get("0")?.value, 3);
