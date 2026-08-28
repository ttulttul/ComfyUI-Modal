import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runInThisContext } from "node:vm";

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
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 24 };
globalThis.CustomEvent = class CustomEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.detail = init.detail;
  }
};
globalThis.Image = class Image {
  constructor() {
    this.complete = true;
    this.naturalWidth = 1;
    this.src = "";
  }

  addEventListener() {}
};
class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.textContent = "";
    this.title = "";
    this.isConnected = false;
    this.elementsBySelector = new Map();
    this.listeners = new Map();
  }

  set innerHTML(_value) {
    for (const selector of [
      ".modal-status-dot",
      ".modal-status-text",
      ".modal-status-gpu",
      ".modal-status-cost",
      ".modal-status-billing",
      ".modal-status-containers",
      ".remote-configurator-status-text",
      ".remote-configurator-progress",
      ".remote-configurator-progress-fill",
      ".remote-configurator-progress-value",
      ".remote-configurator-environments",
      ".remote-configurator-empty",
      ".remote-configurator-targets",
      ".remote-configurator-capacity",
      ".remote-configurator-capacity-list",
      ".remote-configurator-capacity-empty",
      ".remote-configurator-capacity-reload",
      ".remote-configurator-capacity-refresh-status",
      ".remote-configurator-storage",
      ".remote-configurator-storage-list",
      ".remote-configurator-storage-reload",
      ".remote-configurator-storage-refresh-status",
    ]) {
      this.elementsBySelector.set(selector, new FakeElement(selector));
    }
  }

  querySelector(selector) {
    return this.elementsBySelector.get(selector) ?? null;
  }

  append(...children) {
    this.children.push(...children);
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...children) {
    this.children = [...children];
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  remove() {
    this.removed = true;
    this.isConnected = false;
  }
}
const fakeBody = new FakeElement("body");
fakeBody.appendChild = (child) => {
  child.isConnected = true;
  fakeBody.children.push(child);
  return child;
};
globalThis.document = {
  body: fakeBody,
  head: new FakeElement("head"),
  visibilityState: "visible",
  createElement(tagName) {
    return new FakeElement(tagName);
  },
  addEventListener() {},
  getElementById() {
    return null;
  },
  querySelectorAll() {
    return [];
  },
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
  "  registerPromptExecutionAssignments,",
  "  registerPromptConfigurator,",
  "  registerRemoteConfiguratorPlan,",
  "  updateRemoteConfiguratorEnvironmentStatus,",
  "  historyPromptTerminalOutcome,",
  "  clearRefocusCompletedPrompt,",
  "  mountRemoteExecutionConfiguratorPanel,",
  "  remoteConfiguratorStorageEntries,",
  "  refreshRemoteConfiguratorStorage,",
  "  remoteManagedCapacityEntries,",
  "  refreshRemoteManagedCapacity,",
  "  remoteStorageSizeLabel,",
  "  remoteStorageUsageLabel,",
  "  resolveComponentNodeIds,",
  "  handleModalProgress,",
  "  handleModalStatus,",
  "  handleExecutionPhase,",
  "  handlePromptInterruption,",
  "  markPromptTerminal,",
  "  patchQueuePrompt,",
  "  queueErrorMessage,",
  "  setRemoteFlag,",
  "  synchronizeRemoteFlagFromWidget,",
  "  extractRemoteNodeIds,",
  "  setSandwichedLocalNodeIds,",
  "  isSandwichedLocalNode,",
  "  localBottleneckBadgeContainsPoint,",
  "  decorateNode,",
  "  setAllEligibleWorkflowNodesRemote,",
  "  selectedModalGpu,",
  "  setSelectedModalGpu,",
  "  stampModalGpuOnWorkflow,",
  "  installModalContextMenu,",
  "  MODAL_GPU_TYPES,",
  "  REMOTE_LOCATION_ICON_SOURCES,",
  "  clearPromptRemoteStates,",
  "  getRemoteVisualState,",
  "  rebuildRemoteDescendantIndex,",
  "  hasRemoteDescendants,",
  "  remoteContainerTooltip,",
  "  drawModalNodeDecoration,",
  "  updateVueExecutionLocation,",
  "  currentGlobalStatus,",
  "  shouldPollModalContainerStatus,",
  "  remoteCapacityWaitingMessage,",
  "  formatIterationRate,",
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
  "  modalSandwichedLocalNodeIds,",
  "  modalRemoteDescendantNodeIdsByAncestor,",
  "  remoteConfiguratorPanels,",
  "  STATE_SETUP,",
  "  STATE_STARTING,",
  "  STATE_READY,",
  "  STATE_ACTIVE,",
  "  STATE_COMPLETE,",
  "  EXECUTION_PHASE,",
  "};",
].join("\n")}`;
runInThisContext(transformedSource, { filename: sourcePath });

const modalToggle = globalThis.__modalToggleExports;

assert.equal(modalToggle.remoteStorageSizeLabel(0), "0 B");
assert.equal(modalToggle.remoteStorageSizeLabel(1536), "1.50 KiB");
assert.equal(modalToggle.remoteStorageSizeLabel(undefined), "Unavailable");
assert.equal(
  modalToggle.remoteStorageUsageLabel({
    storage_usage_bytes: 1024,
    storage_object_count: 1,
  }),
  "1.00 KiB · 1 object",
);

const managedCapacityEntries = modalToggle.remoteManagedCapacityEntries(
  {
    leases: [{
      instance_id: 49047119,
      profile_name: "vast-small",
      actual_status: "running",
      active_invocations: 0,
      gpu_name: "RTX PRO 6000 WS",
      gpu_count: 1,
      gpu_ram_mb: 97894,
      cpu_ram_mb: 148275,
      hourly_cost_usd: 0.056,
    }],
  },
  {
    containers: [{
      container_id: "ta-managed",
      app_name: "comfy-modal-sync-B300",
      modal_gpu: "B300",
      state: "running",
      estimated_gpu_cost_per_second: 0.001972,
    }],
  },
);
assert.equal(managedCapacityEntries.length, 2);
assert.equal(managedCapacityEntries[0].provider, "vast");
assert.equal(managedCapacityEntries[0].resourceId, "49047119");
assert.equal(managedCapacityEntries[0].details.at(-1), "$0.056/hr");
assert.equal(managedCapacityEntries[1].provider, "modal");
assert.equal(managedCapacityEntries[1].resourceId, "ta-managed");

assert.equal(
  modalToggle.queueErrorMessage({
    message: "Prompt validation failed",
    modalQueueResponse: {
      error: "lambda: insufficient GPU VRAM (11.40 GiB available, 16.00 GiB required)",
    },
  }),
  "lambda: insufficient GPU VRAM (11.40 GiB available, 16.00 GiB required)",
);

function resetFrontendState() {
  modalToggle.modalNodeStates.clear();
  modalToggle.modalNodeProgress.clear();
  modalToggle.modalNodeProgressLanes.clear();
  modalToggle.modalNodeBatchProgress.clear();
  modalToggle.modalPromptStates.clear();
  modalToggle.modalTerminalPromptStates.clear();
  modalToggle.modalQueuedPromptIds.clear();
  modalToggle.modalSandwichedLocalNodeIds.clear();
  modalToggle.modalRemoteDescendantNodeIdsByAncestor.clear();
  modalToggle.modalGlobalStatusStates.clear();
  modalToggle.remoteConfiguratorPanels.clear();
}

resetFrontendState();
const localBottleneckNode = { id: 174, properties: {} };
modalToggle.setSandwichedLocalNodeIds([174]);
assert.equal(modalToggle.isSandwichedLocalNode(localBottleneckNode), true);
localBottleneckNode.properties.is_modal_remote = true;
assert.equal(modalToggle.isSandwichedLocalNode(localBottleneckNode), false);
modalToggle.setRemoteFlag(localBottleneckNode, false);
assert.equal(modalToggle.modalSandwichedLocalNodeIds.size, 0);

resetFrontendState();
const legacyCanvasElement = {
  title: "",
  removeAttribute(name) {
    if (name === "title") {
      this.title = "";
    }
  },
};
const legacyGraphCanvas = {
  canvas: legacyCanvasElement,
  ds: { scale: 1 },
};
const legacyBottleneckNode = {
  id: 175,
  comfyClass: "ModalLLM",
  properties: {},
  addWidget() {
    return {};
  },
};
modalToggle.setSandwichedLocalNodeIds([175]);
assert.equal(
  modalToggle.localBottleneckBadgeContainsPoint(legacyBottleneckNode, [10, -14], 1),
  true,
);
assert.equal(
  modalToggle.localBottleneckBadgeContainsPoint(legacyBottleneckNode, [40, 40], 1),
  false,
);
modalToggle.decorateNode(legacyBottleneckNode);
legacyBottleneckNode.onMouseMove({}, [10, -14], legacyGraphCanvas);
assert.equal(
  legacyCanvasElement.title,
  "Did you mean to make this node execute on Modal?",
);
legacyBottleneckNode.onMouseMove({}, [40, 40], legacyGraphCanvas);
assert.equal(legacyCanvasElement.title, "");

resetFrontendState();
const remoteProgressNode = {
  id: 176,
  comfyClass: "KSampler",
  properties: { is_modal_remote: true },
  size: [160, 80],
  addWidget() {
    return {};
  },
};
modalToggle.registerPromptComponents("prompt-decoration", ["176"], [
  {
    representative_node_id: "176",
    node_ids: ["176"],
  },
]);
modalToggle.registerPromptExecutionAssignments("prompt-decoration", {
  "176": {
    provider: "modal",
    environment_id: "modal:H100",
    node_ids: ["176"],
  },
});
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-decoration",
    node_id: "176",
    value: 2,
    max: 10,
    execution_provider: "modal",
  },
});
modalToggle.decorateNode(remoteProgressNode);
const progressPanelStrokeStyles = [];
const progressProviderIcons = [];
const progressCanvasContext = {
  globalAlpha: 1,
  save() {},
  restore() {},
  beginPath() {},
  roundRect() {},
  arc() {},
  fill() {},
  fillRect() {},
  drawImage(image, x, y, width, height) {
    progressProviderIcons.push({ src: image.src, x, y, width, height });
  },
  fillText() {},
  measureText(text) {
    return { width: String(text).length * 6 };
  },
  stroke() {
    progressPanelStrokeStyles.push(this.strokeStyle);
  },
};
assert.doesNotThrow(() => remoteProgressNode.onDrawForeground(progressCanvasContext));
assert.match(progressPanelStrokeStyles.at(-1), /^rgba\(168, 85, 247, 0\.44/);
assert.equal(progressProviderIcons.length, 1);
assert.match(decodeURIComponent(progressProviderIcons[0].src), /viewBox="0 0 368 192"/);
assert.match(
  decodeURIComponent(modalToggle.REMOTE_LOCATION_ICON_SOURCES.vast),
  /viewBox="0 0 54\.3 46\.28"/,
);
assert.doesNotMatch(
  decodeURIComponent(modalToggle.REMOTE_LOCATION_ICON_SOURCES.vast),
  /viewBox="0 0 173\.64 46\.28"/,
);

resetFrontendState();
const mixedLocationNode = {
  id: 177,
  comfyClass: "KSampler",
  properties: { is_modal_remote: true },
  size: [132, 80],
};
modalToggle.registerPromptComponents("prompt-mixed-location", ["177", "178"], [
  { representative_node_id: "177", node_ids: ["177"] },
  { representative_node_id: "178", node_ids: ["178"] },
]);
modalToggle.registerPromptExecutionAssignments("prompt-mixed-location", {
  "177": {
    provider: "modal",
    environment_id: "modal:B300",
    node_ids: ["177"],
  },
  "178": {
    provider: "ssh_docker",
    environment_id: "spark-one",
    execution_location: "spark-one.internal.example",
    node_ids: ["178"],
  },
});
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-mixed-location",
    node_id: "177",
    value: 2,
    max: 10,
    execution_provider: "modal",
    execution_environment_id: "modal:B300",
    execution_location: "ta-01K3M0DALCONTAINERIDENTITY",
  },
});
const mixedLocationState = modalToggle.getRemoteVisualState(mixedLocationNode);
assert.equal(mixedLocationState?.scheduledEnvironmentCount, 2);
assert.deepEqual(mixedLocationState?.executionLocation, {
  provider: "modal",
  label: "ta-01K3M0DALCONTAINERIDENTITY",
});
const vueLocationIcon = {
  hidden: true,
  attributes: new Map(),
  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  },
  setAttribute(name, value) {
    this.attributes.set(name, value);
  },
};
const vueLocationLabel = { textContent: "" };
const vueLocationElement = {
  hidden: true,
  dataset: {},
  title: "",
  removeAttribute(name) {
    if (name === "title") {
      this.title = "";
    }
  },
  querySelector(selector) {
    if (selector === ".comfy-modal-vue-execution-location-icon") {
      return vueLocationIcon;
    }
    if (selector === ".comfy-modal-vue-execution-location-label") {
      return vueLocationLabel;
    }
    return null;
  },
};
const vueLocationDecoration = {
  querySelector(selector) {
    return selector === ".comfy-modal-vue-execution-location"
      ? vueLocationElement
      : null;
  },
};
modalToggle.updateVueExecutionLocation(vueLocationDecoration, mixedLocationState);
assert.equal(vueLocationElement.hidden, false);
assert.equal(vueLocationElement.dataset.provider, "modal");
assert.equal(vueLocationElement.title, "ta-01K3M0DALCONTAINERIDENTITY");
assert.equal(vueLocationLabel.textContent, "ta-01K3M0DALCONTAINERIDENTITY");
assert.match(vueLocationIcon.getAttribute("src"), /^data:image\/svg\+xml/);
const mixedLocationLabels = [];
const mixedLocationCanvasContext = {
  globalAlpha: 1,
  save() {},
  restore() {},
  beginPath() {},
  roundRect() {},
  arc() {},
  fill() {},
  fillRect() {},
  drawImage() {},
  stroke() {},
  fillText(text) {
    mixedLocationLabels.push(String(text));
  },
  measureText(text) {
    return { width: String(text).length * 6 };
  },
};
modalToggle.drawModalNodeDecoration(mixedLocationNode, mixedLocationCanvasContext);
assert.equal(mixedLocationLabels.some((label) => label.endsWith("…")), true);

modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-mixed-location",
    node_id: "178",
    value: 1,
    max: 4,
    execution_provider: "ssh_docker",
    execution_environment_id: "spark-one",
    execution_location: "spark-one.internal.example",
  },
});
assert.deepEqual(modalToggle.getRemoteVisualState({ id: 178 })?.executionLocation, {
  provider: "ssh_docker",
  label: "spark-one.internal.example",
});
modalToggle.updateVueExecutionLocation(
  vueLocationDecoration,
  modalToggle.getRemoteVisualState({ id: 178 }),
);
assert.equal(vueLocationElement.dataset.provider, "ssh_docker");
assert.equal(vueLocationLabel.textContent, "spark-one.internal.example");
assert.match(vueLocationIcon.getAttribute("src"), /^data:image\/svg\+xml/);
assert.match(decodeURIComponent(vueLocationIcon.getAttribute("src")), /viewBox="0 0 73 73"/);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-active-location", ["180", "181"], [
  { representative_node_id: "180", node_ids: ["180"] },
  { representative_node_id: "181", node_ids: ["181"] },
]);
modalToggle.registerPromptExecutionAssignments("prompt-active-location", {
  "180": {
    provider: "modal",
    environment_id: "modal:B300",
    execution_location: "ta-01K3ACTIVE",
    node_ids: ["180"],
  },
  "181": {
    provider: "ssh_docker",
    environment_id: "spark-one",
    execution_location: "spark-one.internal.example",
    node_ids: ["181"],
  },
});
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-active-location",
    phase: "executing",
    node_ids: ["180"],
    active_node_id: "180",
  },
});
const activeLocationLabels = [];
modalToggle.drawModalNodeDecoration(
  { id: 180, properties: { is_modal_remote: true }, size: [132, 80] },
  {
    ...mixedLocationCanvasContext,
    fillText(text) {
      activeLocationLabels.push(String(text));
    },
  },
);
assert.equal(activeLocationLabels.includes("ta-01K3ACTIVE"), true);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-single-location", ["179"], [
  { representative_node_id: "179", node_ids: ["179"] },
]);
modalToggle.registerPromptExecutionAssignments("prompt-single-location", {
  "179": {
    provider: "ssh_docker",
    environment_id: "spark-one",
    execution_location: "spark-one.internal.example",
    node_ids: ["179"],
  },
});
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-single-location",
    node_id: "179",
    value: 1,
    max: 4,
  },
});
const singleLocationLabels = [];
modalToggle.drawModalNodeDecoration(
  { id: 179, size: [132, 80] },
  {
    ...mixedLocationCanvasContext,
    fillText(text) {
      singleLocationLabels.push(String(text));
    },
  },
);
assert.equal(
  singleLocationLabels.some((label) => label.includes("spark-one")),
  false,
);

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

const originalDateNow = Date.now;
let progressClockMs = 1_000;
Date.now = () => progressClockMs;
resetFrontendState();
modalToggle.registerPromptComponents("prompt-rate", ["20"], [
  {
    representative_node_id: "20",
    node_ids: ["20"],
  },
]);
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-rate",
    node_id: "20",
    value: 1,
    max: 10,
  },
});
assert.equal(modalToggle.modalNodeProgress.get("20")?.iterationRate, null);
progressClockMs = 2_000;
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-rate",
    node_id: "20",
    value: 3,
    max: 10,
  },
});
assert.equal(modalToggle.modalNodeProgress.get("20")?.iterationRate, 2);
progressClockMs = 3_000;
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-rate",
    node_id: "20",
    value: 7,
    max: 10,
  },
});
assert.equal(modalToggle.modalNodeProgress.get("20")?.iterationRate, 2.7);
assert.equal(modalToggle.formatIterationRate(2.7), "2.70 it/s");
assert.equal(modalToggle.formatIterationRate(42.34), "42.3 it/s");
assert.equal(modalToggle.formatIterationRate(null), "— it/s");
assert.equal(modalToggle.formatIterationRate(875, "B"), "875 B/s");
assert.equal(modalToggle.formatIterationRate(1_500, "bytes"), "1.50 KB/s");
assert.equal(modalToggle.formatIterationRate(12_345_678, "byte"), "12.3 MB/s");
assert.equal(modalToggle.formatIterationRate(1_500_000_000, "B/s"), "1.50 GB/s");
assert.equal(
  modalToggle.formatIterationRate(2_500_000_000_000, "bytes/s"),
  "2.50 TB/s",
);
assert.equal(modalToggle.formatIterationRate(null, "bytes"), "— B/s");

progressClockMs = 4_000;
resetFrontendState();
modalToggle.registerPromptComponents("prompt-lane-rate", ["21"], [
  {
    representative_node_id: "21",
    node_ids: ["21"],
  },
]);
modalToggle.registerPromptExecutionAssignments("prompt-lane-rate", {
  "21": {
    provider: "ssh_docker",
    environment_id: "ssh-one",
    node_ids: ["21"],
  },
});
for (const [laneId, value] of [["0", 1], ["1", 2]]) {
  modalToggle.handleModalProgress({
    detail: {
      prompt_id: "prompt-lane-rate",
      node_id: "21",
      value,
      max: 10,
      lane_id: laneId,
      execution_provider: "ssh_docker",
    },
  });
}
progressClockMs = 5_000;
for (const [laneId, value] of [["0", 3], ["1", 6]]) {
  modalToggle.handleModalProgress({
    detail: {
      prompt_id: "prompt-lane-rate",
      node_id: "21",
      value,
      max: 10,
      lane_id: laneId,
      execution_provider: "ssh_docker",
    },
  });
}
assert.equal(modalToggle.modalNodeProgressLanes.get("21")?.lanes.get("0")?.iterationRate, 2);
assert.equal(modalToggle.modalNodeProgressLanes.get("21")?.lanes.get("1")?.iterationRate, 4);
const laneProviderIcons = [];
modalToggle.drawModalNodeDecoration(
  { id: 21, properties: { is_modal_remote: true }, size: [160, 80] },
  {
    ...progressCanvasContext,
    drawImage(image, x, y, width, height) {
      laneProviderIcons.push({ src: image.src, x, y, width, height });
    },
  },
);
assert.equal(laneProviderIcons.length, 2);
assert.equal(
  laneProviderIcons.every(({ src }) =>
    decodeURIComponent(src).includes('viewBox="0 0 73 73"')),
  true,
);
Date.now = originalDateNow;

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

const restoredToggleNode = {
  comfyClass: "ModalLLM",
  properties: { is_modal_remote: true },
  __modalToggleWidget: { value: false },
};
modalToggle.synchronizeRemoteFlagFromWidget(restoredToggleNode);
assert.equal(restoredToggleNode.properties.is_modal_remote, false);
assert.deepEqual(
  modalToggle.extractRemoteNodeIds({
    nodes: [
      {
        id: 9,
        properties: { is_modal_remote: true },
        widgets_values_named: { "Run on Modal": false },
      },
    ],
  }),
  [],
);
assert.deepEqual(
  modalToggle.extractRemoteNodeIds({
    nodes: [
      {
        id: 9,
        properties: { is_modal_remote: false },
        widgets_values_named: { "Run on Modal": true },
      },
    ],
  }),
  ["9"],
);

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
    execution_provider: "modal",
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

resetFrontendState();
const nestedLeafGraph = {
  id: "nested-leaf-graph",
  nodes: [],
  getNodeById(id) {
    return this.nodes.find((node) => String(node.id) === String(id)) ?? null;
  },
};
const nestedMiddleGraph = {
  id: "nested-middle-graph",
  nodes: [],
  getNodeById(id) {
    return this.nodes.find((node) => String(node.id) === String(id)) ?? null;
  },
};
const nestedRootGraph = {
  id: "nested-root-graph",
  extra: {},
  nodes: [],
  getNodeById(id) {
    return this.nodes.find((node) => String(node.id) === String(id)) ?? null;
  },
};
const outerSubgraphNode = {
  id: "100",
  comfyClass: "Subgraph",
  graph: nestedRootGraph,
  subgraph: nestedMiddleGraph,
  properties: {},
  size: [180, 90],
  isSubgraphNode() {
    return true;
  },
  addWidget() {
    return {};
  },
};
const innerSubgraphNode = {
  id: "200",
  comfyClass: "Subgraph",
  graph: nestedMiddleGraph,
  subgraph: nestedLeafGraph,
  properties: {},
  size: [170, 80],
  isSubgraphNode() {
    return true;
  },
  addWidget() {
    return {};
  },
};
const activeNestedLeaf = {
  id: "1",
  comfyClass: "KSampler",
  graph: nestedLeafGraph,
  properties: { is_modal_remote: true },
};
const startingNestedLeaf = {
  id: "2",
  comfyClass: "VAEDecode",
  graph: nestedLeafGraph,
  properties: { is_modal_remote: true },
};
nestedRootGraph.nodes.push(outerSubgraphNode);
nestedMiddleGraph.nodes.push(innerSubgraphNode);
nestedLeafGraph.nodes.push(activeNestedLeaf, startingNestedLeaf);
globalThis.__modalAppStub.rootGraph = nestedRootGraph;
globalThis.__modalAppStub.graph.rootGraph = nestedRootGraph;
modalToggle.rebuildRemoteDescendantIndex();

assert.equal(modalToggle.hasRemoteDescendants(outerSubgraphNode), true);
assert.equal(modalToggle.hasRemoteDescendants(innerSubgraphNode), true);
assert.deepEqual(
  Array.from(modalToggle.modalRemoteDescendantNodeIdsByAncestor.get("100")),
  ["100:200:1", "100:200:2"],
);
assert.deepEqual(
  Array.from(modalToggle.modalRemoteDescendantNodeIdsByAncestor.get("100:200")),
  ["100:200:1", "100:200:2"],
);
assert.equal(modalToggle.getRemoteVisualState(outerSubgraphNode)?.phase, "idle");
assert.equal(modalToggle.getRemoteVisualState(outerSubgraphNode)?.remoteDescendantCount, 2);

modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-nested-container",
    phase: "setup",
    node_ids: ["100:200:1", "100:200:2"],
  },
});
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-nested-container",
    phase: "executing",
    node_ids: ["100:200:1", "100:200:2"],
    active_node_id: "100:200:1",
  },
});
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-nested-container",
    phase: "starting",
    node_ids: ["100:200:2"],
  },
});

const outerMixedState = modalToggle.getRemoteVisualState(outerSubgraphNode);
const innerMixedState = modalToggle.getRemoteVisualState(innerSubgraphNode);
assert.equal(outerMixedState?.phase, modalToggle.STATE_ACTIVE);
assert.equal(innerMixedState?.phase, modalToggle.STATE_ACTIVE);
assert.equal(outerMixedState?.isRemoteContainer, true);
assert.equal(outerMixedState?.isMixedRemoteContainer, true);
assert.equal(outerMixedState?.phaseCounts.active, 1);
assert.equal(outerMixedState?.phaseCounts.starting, 1);
assert.equal(
  modalToggle.remoteContainerTooltip(outerMixedState),
  "2 Modal descendant nodes: 1 active, 1 starting.",
);

const containerStrokeStyles = [];
const containerCanvasContext = {
  save() {},
  restore() {},
  beginPath() {},
  roundRect() {},
  arc() {},
  fill() {},
  fillText() {},
  stroke() {
    containerStrokeStyles.push(this.strokeStyle);
  },
};
modalToggle.drawModalNodeDecoration(outerSubgraphNode, containerCanvasContext);
assert.equal(containerStrokeStyles.some((color) => String(color).startsWith("#a855f7")), true);

modalToggle.clearPromptRemoteStates("prompt-nested-container");
assert.equal(modalToggle.getRemoteVisualState(outerSubgraphNode)?.phase, "idle");
activeNestedLeaf.properties.is_modal_remote = false;
startingNestedLeaf.properties.is_modal_remote = false;
modalToggle.rebuildRemoteDescendantIndex();
assert.equal(modalToggle.hasRemoteDescendants(outerSubgraphNode), false);
assert.equal(modalToggle.getRemoteVisualState(outerSubgraphNode), null);

const workflowGraph = {
  extra: {},
  nodes: [],
  changeCount: 0,
  change() {
    this.changeCount += 1;
  },
  getNodeById(id) {
    return this.nodes.find((node) => String(node.id) === String(id)) ?? null;
  },
};
globalThis.__modalAppStub.rootGraph = workflowGraph;
globalThis.__modalAppStub.graph.rootGraph = workflowGraph;
assert.equal(modalToggle.selectedModalGpu(), "RTX-PRO-6000");
assert.equal(workflowGraph.extra.comfy_modal.gpu, "RTX-PRO-6000");

modalToggle.setSelectedModalGpu("B300");
assert.equal(modalToggle.selectedModalGpu(), "B300");
assert.equal(workflowGraph.extra.comfy_modal.gpu, "B300");
assert.equal(workflowGraph.changeCount, 1);

const serializedWorkflow = { nodes: [] };
modalToggle.stampModalGpuOnWorkflow(serializedWorkflow);
assert.equal(serializedWorkflow.extra.comfy_modal.gpu, "B300");

resetFrontendState();
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-gpu-status",
    phase: "setup",
    node_ids: ["70"],
    modal_gpu: "B300",
    status_message: "Rebuilding Modal app",
  },
});
assert.equal(modalToggle.currentGlobalStatus()?.statusMessage, "Rebuilding Modal app");
assert.equal(modalToggle.currentGlobalStatus()?.modalGpu, "B300");
modalToggle.handleModalProgress({
  detail: {
    prompt_id: "prompt-gpu-status",
    node_id: "70",
    display_node_id: "70",
    real_node_id: "70",
    value: 1,
    max: 4,
  },
});
assert.equal(modalToggle.currentGlobalStatus()?.modalGpu, "B300");

resetFrontendState();
const syntheticErrorEvents = [];
globalThis.__modalApiStub.dispatchEvent = (event) => syntheticErrorEvents.push(event);
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-setup-error",
    phase: "setup",
    node_ids: ["11"],
    status_message: "Starting Vast.ai capacity",
  },
});
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-setup-error",
    phase: "error",
    node_ids: ["11"],
    error_message:
      "Vast instance 48597015 did not become SSH-ready within 900s; " +
      "last status was 'unknown'.",
  },
});
const syntheticSetupFailure = syntheticErrorEvents
  .filter(
    (event) =>
      event.type === "notification" &&
      event.detail?.id === "prompt-setup-error",
  )
  .at(-1);
assert.equal(
  syntheticSetupFailure?.detail?.value,
  "Vast instance 48597015 did not become SSH-ready within 900s; " +
    "last status was 'unknown'.",
);
assert.equal(
  syntheticErrorEvents.some((event) => event.type === "execution_error"),
  false,
);
globalThis.__modalApiStub.dispatchEvent = () => {};

resetFrontendState();
globalThis.__modalApiStub.__modalQueuePromptPatched = false;
globalThis.__modalApiStub.clientId = "client-fast";
globalThis.__modalApiStub.fetchApi = async (_route, options) => {
  const requestBody = JSON.parse(options.body);
  modalToggle.markPromptTerminal(requestBody.prompt_id, "execution_success");
  modalToggle.clearPromptRemoteStates(requestBody.prompt_id);
  return {
    status: 200,
    async json() {
      return {
        prompt_id: requestBody.prompt_id,
        modal_remote_node_ids: ["80"],
        modal_sandwiched_local_node_ids: ["174"],
        modal_components: [],
        modal_gpu: "B300",
      };
    },
  };
};
modalToggle.patchQueuePrompt();
const fastPromptResponse = await globalThis.__modalApiStub.queuePrompt(0, {
  output: {},
  workflow: {
    extra: { comfy_modal: { gpu: "B300" } },
    nodes: [{ id: 80, properties: { is_modal_remote: true } }],
  },
});
assert.equal(typeof fastPromptResponse.prompt_id, "string");
assert.deepEqual(fastPromptResponse.node_errors, {});
assert.equal(fastPromptResponse.modal_gpu, undefined);
await Promise.resolve();
await Promise.resolve();
assert.equal(modalToggle.modalNodeStates.has("80"), false);
assert.deepEqual(Array.from(modalToggle.modalSandwichedLocalNodeIds), ["174"]);

resetFrontendState();
globalThis.__modalApiStub.__modalQueuePromptPatched = false;
globalThis.__modalApiStub.clientId = "client-vast-only";
let releaseVastOnlyResponse;
globalThis.__modalApiStub.fetchApi = async (_route, options) => {
  const requestBody = JSON.parse(options.body);
  return await new Promise((resolve) => {
    releaseVastOnlyResponse = () =>
      resolve({
        status: 200,
        async json() {
          return {
            prompt_id: requestBody.prompt_id,
            modal_remote_node_ids: ["80"],
            modal_components: [
              { representative_node_id: "80", node_ids: ["80"] },
            ],
            modal_gpu: null,
            remote_execution_modal_gpus: [],
            remote_execution_assignments: {
              "80": {
                provider: "vast",
                environment_id: "vast:instance-1",
                execution_location: "vast.example",
                configuration_id: "90",
                node_ids: ["80"],
              },
            },
          };
        },
      });
  });
};
modalToggle.patchQueuePrompt();
const vastOnlyPrompt = globalThis.__modalApiStub.queuePrompt(0, {
  output: {
    "80": { class_type: "KSampler", inputs: {} },
    "90": { class_type: "VastRemoteConfiguration", inputs: {} },
    "99": {
      class_type: "RemoteExecutionConfigurator",
      inputs: { "configurations.configuration_0": ["90", 0] },
    },
  },
  workflow: {
    nodes: [{ id: 80, properties: { is_modal_remote: true } }],
  },
});
await Promise.resolve();
assert.equal(modalToggle.currentGlobalStatus()?.modalGpu, null);
assert.equal(
  modalToggle.shouldPollModalContainerStatus(modalToggle.currentGlobalStatus()),
  false,
);
assert.equal(
  modalToggle.remoteCapacityWaitingMessage(["vast"]),
  "Waiting for Vast.ai instance",
);
assert.equal(
  modalToggle.remoteCapacityWaitingMessage(["ssh_docker"]),
  "Waiting for self-hosted worker",
);
assert.equal(
  modalToggle.remoteCapacityWaitingMessage(["vast", "ssh_docker"]),
  "Waiting for remote capacity",
);
releaseVastOnlyResponse();
await vastOnlyPrompt;
await Promise.resolve();
await Promise.resolve();
assert.equal(modalToggle.currentGlobalStatus()?.statusMessage, "Waiting for Vast.ai instance");
assert.equal(modalToggle.currentGlobalStatus()?.modalGpu, null);
assert.equal(
  modalToggle.shouldPollModalContainerStatus(modalToggle.currentGlobalStatus()),
  false,
);

resetFrontendState();
modalToggle.registerPromptComponents("prompt-vast-runtime", ["81"], [
  { representative_node_id: "81", node_ids: ["81"] },
]);
modalToggle.handleModalStatus({
  detail: {
    prompt_id: "prompt-vast-runtime",
    phase: "starting",
    node_ids: ["81"],
    execution_provider: "vast",
    modal_gpu: "RTX-PRO-6000",
  },
});
assert.equal(
  modalToggle.currentGlobalStatus()?.statusMessage,
  "Starting Vast.ai component",
);
assert.equal(modalToggle.currentGlobalStatus()?.modalGpu, null);
assert.equal(
  modalToggle.shouldPollModalContainerStatus(modalToggle.currentGlobalStatus()),
  false,
);

class MenuNode {}
const menuNode = new MenuNode();
menuNode.id = 101;
menuNode.graph = workflowGraph;
workflowGraph.nodes.push(menuNode);
modalToggle.installModalContextMenu(MenuNode, { name: "KSampler" });
const menuOptions = [];
menuNode.getExtraMenuOptions(null, menuOptions);
const remoteExecutionMenu = menuOptions.find(
  (option) => option?.content === "Remote Execution",
);
const providerPolicyHeading = remoteExecutionMenu.submenu.options.find(
  (option) => option?.content === "Legacy provider policy (Modal only)",
);
assert.equal(providerPolicyHeading.disabled, true);
assert.equal(providerPolicyHeading.has_submenu, undefined);
const providerPolicyHeadingIndex = remoteExecutionMenu.submenu.options.indexOf(
  providerPolicyHeading,
);
assert.deepEqual(
  remoteExecutionMenu.submenu.options
    .slice(providerPolicyHeadingIndex + 1, providerPolicyHeadingIndex + 5)
    .map((option) => option.content),
  ["Modal only", "Self-hosted only", "Vast.ai only", "Automatic (lowest cost compatible)"],
);
const vastPolicyOption = remoteExecutionMenu.submenu.options.find(
  (option) => option?.content === "Vast.ai only",
);
assert.equal(vastPolicyOption.checked, false);
assert.equal(vastPolicyOption.submenu, undefined);
vastPolicyOption.callback();
assert.equal(workflowGraph.extra.remote_execution.policy, "vast");
assert.equal(workflowGraph.changeCount, 2);

const modalMenu = menuOptions.find(
  (option) => option?.content === "Remote Execution Tools",
);
const gpuHeading = modalMenu.submenu.options.find(
  (option) => option?.content === "Legacy Modal GPU (B300)",
);
assert.equal(gpuHeading.disabled, true);
assert.equal(gpuHeading.has_submenu, undefined);
assert.deepEqual(
  modalMenu.submenu.options
    .filter((option) => modalToggle.MODAL_GPU_TYPES.includes(option?.content))
    .map((option) => option.content),
  modalToggle.MODAL_GPU_TYPES,
);
assert.equal(
  modalMenu.submenu.options.find((option) => option?.content === "B300")?.checked,
  true,
);
modalMenu.submenu.options.find((option) => option?.content === "L40S").callback();
assert.equal(workflowGraph.extra.comfy_modal.gpu, "L40S");
assert.equal(workflowGraph.changeCount, 3);

const configuratorNode = { id: 102, comfyClass: "RemoteExecutionConfigurator" };
workflowGraph.nodes.push(configuratorNode);
class ConfiguredMenuNode {}
const configuredMenuNode = new ConfiguredMenuNode();
configuredMenuNode.id = 103;
configuredMenuNode.graph = workflowGraph;
workflowGraph.nodes.push(configuredMenuNode);
modalToggle.installModalContextMenu(ConfiguredMenuNode, { name: "KSampler" });
const configuredMenuOptions = [];
configuredMenuNode.getExtraMenuOptions(null, configuredMenuOptions);
const configuredRemoteMenu = configuredMenuOptions.find(
  (option) => option?.content === "Remote Execution",
);
assert.equal(
  configuredRemoteMenu.submenu.options[0].content,
  "Providers and capacity come from Remote Execution Configurator",
);
assert.equal(configuredRemoteMenu.submenu.options[0].disabled, true);
assert.equal(
  configuredRemoteMenu.submenu.options.some(
    (option) => option?.content === "Modal only",
  ),
  false,
);
const configuredToolsMenu = configuredMenuOptions.find(
  (option) => option?.content === "Remote Execution Tools",
);
assert.equal(
  configuredToolsMenu.submenu.options.some(
    (option) => modalToggle.MODAL_GPU_TYPES.includes(option?.content),
  ),
  false,
);

resetFrontendState();
const configuratorGraph = {
  nodes: [],
  setDirtyCanvas() {},
};
globalThis.__modalAppStub.rootGraph = configuratorGraph;
const r2StorageNode = {
  id: 385,
  comfyClass: "R2StorageBackingConfiguration",
  widgets: [{ name: "credential_id", value: "opaque-reference" }],
};
configuratorGraph.nodes.push(r2StorageNode);
modalToggle.registerPromptConfigurator("plan-before-capacity", "381");
modalToggle.registerRemoteConfiguratorPlan(
  "plan-before-capacity",
  {
    "14": {
      provider: "vast",
      environment_id: "vast:vast-big",
      configuration_id: "vast-big",
      node_ids: ["14", "15"],
      predicted_cost_usd: 0.031,
      predicted_completion_seconds: 120,
      hardware: {
        machine_type: "RTX 4090",
        gpu_count: 1,
        gpu_memory_kind: "VRAM",
        gpu_memory_bytes_per_device: 24 * 1024 ** 3,
        gpu_memory_bytes_total: 24 * 1024 ** 3,
        ram_bytes: 64 * 1024 ** 3,
      },
    },
    "24": {
      provider: "modal",
      environment_id: "modal:lambda",
      configuration_id: "lambda",
      node_ids: ["24"],
      predicted_cost_usd: 0,
      predicted_completion_seconds: 60,
      hardware: {
        machine_type: "L40S",
        gpu_count: 1,
        gpu_memory_kind: "VRAM",
        gpu_memory_bytes_per_device: 48 * 1024 ** 3,
        gpu_memory_bytes_total: 48 * 1024 ** 3,
        ram_capacity_label: "Provider managed",
      },
    },
  },
  [
    { configuration_id: "vast-big", display_name: "Vast Big" },
    { configuration_id: "lambda", display_name: "Lambda", gpu_type: "L40S" },
    {
      configuration_id: "385",
      configuration_kind: "storage",
      display_name: "Shared model cache",
      storage_provider: "cloudflare_r2",
      bucket: "comfy-models",
      storage_usage_bytes: 5 * 1024 ** 3 + 256 * 1024 ** 2,
      storage_object_count: 42,
      jurisdiction: "eu",
      key_prefix: "comfy-modal-cache/v1/blobs/sha256",
      write_back_mode: "async",
    },
  ],
);
const stalePanelRoot = new FakeElement("div");
const stalePanelWidget = {
  name: "remote_execution_plan",
  element: stalePanelRoot,
  onRemove() {
    this.wasUnregistered = true;
  },
};
const lateConfiguratorNode = {
  id: 381,
  comfyClass: "RemoteExecutionConfigurator",
  __remoteConfiguratorPanelMounted: true,
  size: [300, 100],
  graph: configuratorGraph,
  widgets: [stalePanelWidget],
  addDOMWidget(name, _type, root) {
    this.panelRoot = root;
    const widget = { name, element: root };
    this.widgets.push(widget);
    return widget;
  },
  removeWidget(widget) {
    widget.onRemove?.();
    this.widgets.splice(this.widgets.indexOf(widget), 1);
  },
  setSize(size) {
    this.size = size;
  },
};
configuratorGraph.nodes.push(lateConfiguratorNode);
modalToggle.mountRemoteExecutionConfiguratorPanel(lateConfiguratorNode);
const retainedPanel = modalToggle.remoteConfiguratorPanels.get("381");
assert.ok(retainedPanel, "a stale mount flag must not prevent a retry");
assert.equal(stalePanelRoot.removed, true);
assert.equal(stalePanelWidget.wasUnregistered, true);
assert.equal(lateConfiguratorNode.widgets.length, 1);
assert.equal(lateConfiguratorNode.widgets[0].element, retainedPanel.root);
assert.equal(
  lateConfiguratorNode.widgets[0].width,
  undefined,
  "the DOM widget must use the node's current layout width",
);
lateConfiguratorNode.widgets[0].width = 275;
assert.equal(
  lateConfiguratorNode.widgets[0].width,
  undefined,
  "legacy frontend width writes must not freeze the Configurator panel",
);
globalThis.LiteGraph.vueNodesMode = true;
lateConfiguratorNode.widgets[0].width = 275;
assert.equal(
  lateConfiguratorNode.widgets[0].width,
  275,
  "Vue-node mode must retain its managed widget width",
);
globalThis.LiteGraph.vueNodesMode = false;
assert.equal(retainedPanel.promptId, "plan-before-capacity");
assert.equal(retainedPanel.targets.children.length, 2);
assert.equal(retainedPanel.environmentRows.size, 2);
assert.equal(retainedPanel.targets.hidden, false);
assert.equal(retainedPanel.emptyText.hidden, true);
const [lambdaTarget, vastTarget] = retainedPanel.targets.children;
assert.deepEqual(
  lambdaTarget.children[0].children.map((child) => child.textContent),
  ["Lambda", "L40S"],
);
assert.deepEqual(
  lambdaTarget.children[2].children.map((child) => child.textContent),
  ["48.0 GiB VRAM", "RAM Provider managed"],
);
assert.deepEqual(
  lambdaTarget.children[3].children[0].children[0].children.map(
    (child) => child.textContent,
  ),
  ["Component 2", "Nodes #24"],
);
assert.deepEqual(
  vastTarget.children[0].children.map((child) => child.textContent),
  ["Vast Big", "RTX 4090"],
);
assert.deepEqual(
  vastTarget.children[2].children.map((child) => child.textContent),
  ["24.0 GiB VRAM", "64.0 GiB RAM"],
);
assert.deepEqual(
  vastTarget.children[3].children[0].children[0].children.map(
    (child) => child.textContent,
  ),
  ["Component 1", "Nodes #14, #15"],
);
assert.equal(retainedPanel.storage.hidden, false);
assert.equal(retainedPanel.storageList.children.length, 1);
assert.equal(typeof retainedPanel.storageReload.listeners.get("click"), "function");
const storageCard = retainedPanel.storageList.children[0];
assert.deepEqual(
  storageCard.children[0].children.map((child) => child.textContent),
  ["Shared model cache", "Cloudflare R2"],
);
assert.deepEqual(
  storageCard.children[1].children.map((child) => child.textContent),
  [
    "Bucket",
    "comfy-models",
    "Storage used",
    "5.25 GiB · 42 objects",
    "Jurisdiction",
    "European Union (EU)",
    "Cache policy",
    "Async write-back",
    "Key prefix",
    "comfy-modal-cache/v1/blobs/sha256",
  ],
);
assert.deepEqual(modalToggle.remoteConfiguratorStorageEntries([]), []);
const storageRefreshRequests = [];
globalThis.__modalApiStub.fetchApi = async (route, options) => {
  storageRefreshRequests.push({ route, options });
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    async json() {
      return {
        configuration_id: "385",
        storage_usage_bytes: 6 * 1024 ** 3,
        storage_object_count: 43,
      };
    },
  };
};
await modalToggle.refreshRemoteConfiguratorStorage(retainedPanel);
assert.equal(storageRefreshRequests.length, 1);
assert.equal(storageRefreshRequests[0].route, "/remote/storage/r2/usage");
assert.equal(
  JSON.parse(storageRefreshRequests[0].options.body).credential_id,
  "opaque-reference",
);
assert.equal(retainedPanel.storageRefreshStatus.textContent, "Updated");
assert.equal(retainedPanel.storageReload.textContent, "Reload");
assert.equal(
  retainedPanel.storageList.children[0].children[1].children[3].textContent,
  "6.00 GiB · 43 objects",
);
globalThis.__modalApiStub.fetchApi = async () => ({
  ok: false,
  status: 502,
  statusText: "Bad Gateway",
  async json() {
    return { error: "R2 is temporarily unavailable." };
  },
});
await modalToggle.refreshRemoteConfiguratorStorage(retainedPanel);
assert.equal(retainedPanel.storageRefreshStatus.textContent, "Refresh failed");
assert.equal(
  retainedPanel.storageRefreshStatus.title,
  "R2 is temporarily unavailable.",
);
assert.equal(
  retainedPanel.storageList.children[0].children[1].children[3].textContent,
  "6.00 GiB · 43 objects",
);
assert.equal(retainedPanel.root.isConnected, false);

modalToggle.handleModalStatus({
  detail: {
    phase: "setup",
    prompt_id: "plan-before-capacity",
    node_ids: ["14", "15"],
    configurator_node_id: "381",
    status_message: "Acquiring Vast capacity",
  },
});
assert.equal(retainedPanel.statusText.textContent, "Acquiring Vast capacity");
assert.equal(
  modalToggle.remoteConfiguratorPanels.get("381"),
  retainedPanel,
  "a detached panel should retain prompt state instead of being remounted",
);

modalToggle.handleModalStatus({
  detail: {
    phase: "setup",
    prompt_id: "plan-before-capacity",
    node_ids: ["14", "15"],
    configurator_node_id: "381",
    execution_environment_id: "vast:vast-big",
    status_message: "Vast.ai worker is ready",
  },
});
modalToggle.handleModalStatus({
  detail: {
    phase: "setup",
    prompt_id: "plan-before-capacity",
    node_ids: ["14", "15"],
    configurator_node_id: "381",
    execution_environment_id: "vast:vast-big:49052528",
    status_message: "Uploading Vast assets",
    status_current: 3,
    status_total: 10,
  },
});
modalToggle.handleModalStatus({
  detail: {
    phase: "starting",
    prompt_id: "plan-before-capacity",
    node_ids: ["24"],
    configurator_node_id: "381",
    execution_environment_id: "modal:lambda",
    status_message: "Starting Lambda",
    status_current: 1,
    status_total: 4,
  },
});
const vastEnvironmentRow = retainedPanel.environmentRows.get("vast:vast-big:49052528");
const modalEnvironmentRow = retainedPanel.environmentRows.get("modal:lambda");
assert.equal(retainedPanel.environmentRows.size, 2);
assert.equal(retainedPanel.environmentRows.has("vast:vast-big"), false);
assert.equal(
  vastEnvironmentRow.label.textContent,
  "Vast Big · Vast.ai vast:vast-big:49052528",
);
assert.equal(
  retainedPanel.assignments["14"].environment_id,
  "vast:vast-big:49052528",
);
assert.equal(
  modalToggle.modalPromptStates
    .get("plan-before-capacity")
    .remoteEnvironmentStatuses.has("vast:vast-big"),
  false,
);
assert.equal(vastEnvironmentRow.statusText.textContent, "Uploading Vast assets");
assert.equal(vastEnvironmentRow.progressValue.textContent, "3/10");
assert.equal(vastEnvironmentRow.progressFill.style.width, "30%");
assert.equal(modalEnvironmentRow.statusText.textContent, "Starting Lambda");
assert.equal(modalEnvironmentRow.progressValue.textContent, "1/4");
assert.equal(modalEnvironmentRow.progressFill.style.width, "25%");

modalToggle.handleModalStatus({
  detail: {
    phase: "error",
    prompt_id: "plan-before-capacity",
    node_ids: ["14", "15", "24"],
    configurator_node_id: "381",
    error_message: "Hugging Face could not be reached",
  },
});
const failedPromptState = modalToggle.modalPromptStates.get("plan-before-capacity");
assert.equal(
  failedPromptState.remoteEnvironmentStatuses.get("vast:vast-big:49052528")?.phase,
  "error",
);
assert.equal(
  failedPromptState.remoteEnvironmentStatuses.get("modal:lambda")?.statusMessage,
  "Hugging Face could not be reached",
);
assert.equal(vastEnvironmentRow.progress.hidden, true);
assert.equal(modalEnvironmentRow.progress.hidden, true);
modalToggle.registerRemoteConfiguratorPlan(
  "plan-before-capacity",
  failedPromptState.remoteExecutionPlanAssignments,
  failedPromptState.remoteExecutionConfigurations,
);
assert.equal(
  retainedPanel.environmentRows.get("vast:vast-big:49052528").root.dataset.phase,
  "error",
);
assert.equal(
  retainedPanel.environmentRows.get("modal:lambda").statusText.textContent,
  "Hugging Face could not be reached",
);

modalToggle.registerPromptConfigurator("newer-prompt", "381");
modalToggle.modalPromptStates.get("plan-before-capacity").startedAt = 1;
modalToggle.modalPromptStates.get("newer-prompt").startedAt = 2;
modalToggle.registerPromptConfigurator("newer-prompt", "381");
assert.equal(retainedPanel.promptId, "newer-prompt");
modalToggle.registerPromptConfigurator("plan-before-capacity", "381");
assert.equal(retainedPanel.promptId, "newer-prompt");

const failedHistoryOutcome = modalToggle.historyPromptTerminalOutcome(
  {
    "newer-prompt": {
      status: {
        status_str: "error",
        completed: false,
        messages: [[
          "execution_error",
          { exception_message: "SSH worker has no GPU" },
        ]],
      },
    },
  },
  "newer-prompt",
);
assert.deepEqual(failedHistoryOutcome, {
  terminalPhase: "execution_error",
  displayPhase: "error",
  message: "SSH worker has no GPU",
});
assert.equal(
  modalToggle.modalPromptStates.get("newer-prompt").configuratorNodeId,
  "381",
);
assert.equal(retainedPanel.promptId, "newer-prompt");
modalToggle.clearRefocusCompletedPrompt("newer-prompt", failedHistoryOutcome);
assert.equal(retainedPanel.statusText.textContent, "SSH worker has no GPU");
assert.equal(retainedPanel.root.dataset.phase, "error");
assert.equal(modalToggle.modalPromptStates.has("newer-prompt"), false);
