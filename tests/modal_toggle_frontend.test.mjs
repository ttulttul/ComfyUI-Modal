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
  "  handleModalStatus,",
  "  handleExecutionPhase,",
  "  handlePromptInterruption,",
  "  setRemoteFlag,",
  "  setAllEligibleWorkflowNodesRemote,",
  "  clearPromptRemoteStates,",
  "  getRemoteVisualState,",
  "  currentGlobalStatus,",
  "  fadeNodeProgress,",
  "  markPromptQueuedBehindActiveModal,",
  "  isPromptQueuedBehindActiveModal,",
  "  modalGlobalStatusStates,",
  "  modalNodeStates,",
  "  modalNodeProgress,",
  "  modalNodeProgressLanes,",
  "  modalNodeBatchProgress,",
  "  modalPromptStates,",
  "  modalTerminalPromptStates,",
  "  modalQueuedPromptIds,",
  "  STATE_READY,",
  "  STATE_ACTIVE,",
  "  STATE_COMPLETE,",
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
  modalToggle.modalTerminalPromptStates.clear();
  modalToggle.modalQueuedPromptIds.clear();
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
assert.equal(modalToggle.getRemoteVisualState({ id: "10" })?.componentLabel ?? null, null);

modalToggle.modalNodeStates.set("10", {
  phase: modalToggle.STATE_READY,
  promptId: "prompt-a",
  updatedAt: 1,
});
assert.equal(modalToggle.getRemoteVisualState({ id: "10" })?.componentLabel, "1");

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
assert.equal(modalToggle.currentGlobalStatus()?.phase, "waiting");

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
assert.equal(modalToggle.modalPromptStates.get("prompt-c")?.activeNodeId ?? null, null);
assert.equal(modalToggle.currentGlobalStatus()?.phase, modalToggle.EXECUTION_PHASE);
assert.equal(modalToggle.modalNodeStates.get("10")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("11")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("12")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeProgressLanes.has("11"), false);
assert.equal(modalToggle.modalNodeProgressLanes.get("12")?.lanes.get("0")?.value, 3);
assert.equal(modalToggle.getRemoteVisualState({ id: "12" })?.phase, modalToggle.STATE_ACTIVE);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-d", ["10", "11"], [
  {
    representative_node_id: "10",
    node_ids: ["10", "11"],
  },
]);
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-d",
    node_id: "10",
    display_node_id: "10",
    real_node_id: "10",
    value: 2,
    max: 8,
    lane_id: "0",
  },
});
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-d",
    node_id: "10",
    display_node_id: "10",
    real_node_id: "11",
    value: 5,
    max: 9,
    lane_id: "1",
  },
});
assert.equal(modalToggle.getRemoteVisualState({ id: "10" })?.phase, modalToggle.STATE_ACTIVE);
assert.equal(modalToggle.getRemoteVisualState({ id: "11" })?.phase, modalToggle.STATE_ACTIVE);
assert.equal(modalToggle.modalNodeStates.get("10")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("11")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeProgressLanes.get("10")?.lanes.get("0")?.value, 2);
assert.equal(modalToggle.modalNodeProgressLanes.get("11")?.lanes.get("1")?.value, 5);

modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-d",
    node_id: "10",
    display_node_id: "10",
    real_node_id: "11",
    value: 6,
    max: 9,
    lane_id: "0",
  },
});
assert.equal(modalToggle.modalNodeProgressLanes.get("10")?.lanes.has("0") ?? false, false);
assert.equal(modalToggle.modalNodeProgressLanes.get("11")?.lanes.get("0")?.value, 6);

modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-d",
    node_id: "10",
    display_node_id: "10",
    real_node_id: "11",
    value: 0,
    max: 1,
    lane_id: "1",
    clear: true,
  },
});
assert.equal(modalToggle.getRemoteVisualState({ id: "10" })?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.getRemoteVisualState({ id: "11" })?.phase, modalToggle.STATE_ACTIVE);

modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-d",
    node_id: "10",
    display_node_id: "10",
    real_node_id: "11",
    value: 0,
    max: 1,
    lane_id: "0",
    clear: true,
  },
});
assert.equal(modalToggle.getRemoteVisualState({ id: "11" })?.phase, modalToggle.STATE_READY);

const toggleSyncNode = {
  properties: {},
  __modalToggleWidget: { value: false },
};
modalToggle.setRemoteFlag(toggleSyncNode, true);
assert.equal(toggleSyncNode.properties.is_modal_remote, true);
assert.equal(toggleSyncNode.__modalToggleWidget.value, true);
modalToggle.setRemoteFlag(toggleSyncNode, false);
assert.equal(toggleSyncNode.properties.is_modal_remote, false);
assert.equal(toggleSyncNode.__modalToggleWidget.value, false);

const eligibleNode = {
  id: "eligible",
  comfyClass: "KSampler",
  properties: {},
  __modalToggleWidget: { value: false },
};
const internalNode = {
  id: "internal",
  comfyClass: "ModalUniversalExecutor_deadbeef",
  properties: {},
  __modalToggleWidget: { value: false },
};
const nestedEligibleNode = {
  id: "nested",
  comfyClass: "CheckpointLoaderSimple",
  properties: {},
  __modalToggleWidget: { value: false },
};
globalThis.__modalAppStub.rootGraph = {
  nodes: [
    eligibleNode,
    {
      id: "subgraph-owner",
      comfyClass: "Subgraph",
      properties: {},
      __modalToggleWidget: { value: false },
      subgraph: {
        nodes: [nestedEligibleNode, internalNode],
      },
    },
  ],
};
assert.equal(modalToggle.setAllEligibleWorkflowNodesRemote(true), 3);
assert.equal(eligibleNode.properties.is_modal_remote, true);
assert.equal(eligibleNode.__modalToggleWidget.value, true);
assert.equal(nestedEligibleNode.properties.is_modal_remote, true);
assert.equal(nestedEligibleNode.__modalToggleWidget.value, true);
assert.equal(internalNode.properties.is_modal_remote, undefined);
assert.equal(internalNode.__modalToggleWidget.value, false);
assert.equal(modalToggle.setAllEligibleWorkflowNodesRemote(false), 3);
assert.equal(eligibleNode.properties.is_modal_remote, false);
assert.equal(eligibleNode.__modalToggleWidget.value, false);
assert.equal(nestedEligibleNode.properties.is_modal_remote, false);
assert.equal(nestedEligibleNode.__modalToggleWidget.value, false);
assert.equal(internalNode.properties.is_modal_remote, undefined);
assert.equal(internalNode.__modalToggleWidget.value, false);
globalThis.__modalAppStub.rootGraph = null;

resetFrontendState();
modalToggle.registerPromptComponents("prompt-e", ["10", "11"], [
  {
    representative_node_id: "10",
    node_ids: ["10", "11"],
  },
]);
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-e",
      node: "10",
    },
  },
  modalToggle.EXECUTION_PHASE,
);
assert.equal(modalToggle.modalPromptStates.has("prompt-e"), true);
assert.equal(modalToggle.modalNodeStates.get("10")?.phase, modalToggle.STATE_READY);
modalToggle.handlePromptInterruption("prompt-e");
assert.equal(modalToggle.modalPromptStates.has("prompt-e"), false);
assert.equal(modalToggle.modalNodeStates.has("10"), false);
assert.equal(modalToggle.modalNodeStates.has("11"), false);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-f", ["20", "21"], [
  {
    representative_node_id: "20",
    node_ids: ["20", "21"],
  },
]);
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-f",
      node: "20",
    },
  },
  modalToggle.EXECUTION_PHASE,
);
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-f",
    phase: "execution_interrupted",
    node_ids: ["20", "21"],
  },
});
assert.equal(modalToggle.modalPromptStates.has("prompt-f"), false);
assert.equal(modalToggle.modalNodeStates.has("20"), false);
assert.equal(modalToggle.modalNodeStates.has("21"), false);
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-f",
    node_id: "20",
    display_node_id: "20",
    real_node_id: "21",
    value: 1,
    max: 4,
  },
});
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-f",
    phase: "executing",
    node_ids: ["20", "21"],
    active_node_id: "21",
  },
});
assert.equal(modalToggle.modalPromptStates.has("prompt-f"), false);
assert.equal(modalToggle.modalGlobalStatusStates.has("prompt-f"), false);
assert.equal(modalToggle.modalNodeProgress.has("21"), false);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-container-wait", ["24"], [
  {
    representative_node_id: "24",
    node_ids: ["24"],
  },
]);
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-container-wait",
    phase: "executing",
    node_ids: ["24"],
  },
});
assert.equal(modalToggle.currentGlobalStatus()?.phase, "waiting");
assert.equal(modalToggle.currentGlobalStatus()?.statusMessage, "Waiting for Modal container");

modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-container-wait",
    phase: "executing",
    node_ids: ["24"],
    active_node_id: "24",
  },
});
assert.equal(modalToggle.currentGlobalStatus()?.phase, modalToggle.EXECUTION_PHASE);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-g", ["30", "31"], [
  {
    representative_node_id: "30",
    node_ids: ["30", "31"],
  },
]);
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-g",
      node: "30",
    },
  },
  modalToggle.EXECUTION_PHASE,
);
assert.equal(modalToggle.modalGlobalStatusStates.has("prompt-g"), true);
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-g",
      node: "30",
    },
  },
  modalToggle.STATE_COMPLETE,
);
assert.equal(modalToggle.modalNodeStates.get("30")?.phase, modalToggle.STATE_COMPLETE);
assert.equal(modalToggle.modalNodeStates.get("31")?.phase, modalToggle.STATE_COMPLETE);
assert.equal(modalToggle.modalGlobalStatusStates.has("prompt-g"), false);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-g-fade", ["32"], [
  {
    representative_node_id: "32",
    node_ids: ["32"],
  },
]);
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-g-fade",
    node_id: "32",
    display_node_id: "32",
    real_node_id: "32",
    value: 10,
    max: 10,
  },
});
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-g-fade",
      node: "32",
    },
  },
  modalToggle.STATE_COMPLETE,
);
assert.equal(modalToggle.modalNodeStates.get("32")?.phase, modalToggle.STATE_COMPLETE);
assert.equal(modalToggle.getRemoteVisualState({ id: "32" })?.phase, modalToggle.STATE_COMPLETE);
assert.equal(modalToggle.modalNodeProgress.get("32")?.fadingStartedAt > 0, true);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-h", ["40", "41", "42"], [
  {
    representative_node_id: "40",
    node_ids: ["40"],
  },
  {
    representative_node_id: "41",
    node_ids: ["41", "42"],
  },
]);
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-h",
    phase: "executing",
    node_ids: ["41"],
    active_node_id: "41",
  },
});
assert.equal(modalToggle.currentGlobalStatus()?.nodeCount, 3);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-active", ["50", "51"], [
  {
    representative_node_id: "50",
    node_ids: ["50", "51"],
  },
]);
modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-active",
      node: "50",
    },
  },
  modalToggle.EXECUTION_PHASE,
);
modalToggle.registerPromptComponents("prompt-queued", ["60", "61"], [
  {
    representative_node_id: "60",
    node_ids: ["60", "61"],
  },
]);
assert.equal(modalToggle.markPromptQueuedBehindActiveModal("prompt-queued"), true);
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-queued",
    phase: "setup",
    node_ids: ["60", "61"],
  },
});
assert.equal(modalToggle.isPromptQueuedBehindActiveModal("prompt-queued"), true);
assert.equal(modalToggle.modalNodeStates.has("60"), false);
assert.equal(modalToggle.modalNodeStates.has("61"), false);
assert.equal(modalToggle.currentGlobalStatus()?.promptId, "prompt-active");

modalToggle.handleExecutionPhase(
  {
    detail: {
      prompt_id: "prompt-queued",
      node: "60",
    },
  },
  modalToggle.EXECUTION_PHASE,
);
assert.equal(modalToggle.isPromptQueuedBehindActiveModal("prompt-queued"), false);
assert.equal(modalToggle.modalNodeStates.get("60")?.phase, modalToggle.STATE_READY);
assert.equal(modalToggle.modalNodeStates.get("61")?.phase, modalToggle.STATE_READY);
