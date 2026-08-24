"""Regression tests for the frontend Modal queue shim."""

from __future__ import annotations

from pathlib import Path


def _modal_toggle_source() -> str:
    """Return the current frontend extension source."""
    return (Path(__file__).resolve().parents[1] / "web" / "modal_toggle.js").read_text(encoding="utf-8")


def test_synthetic_status_event_matches_comfyui_status_shape() -> None:
    """Synthetic status events should use the same detail shape as websocket status events."""
    source = _modal_toggle_source()

    assert 'dispatchSyntheticApiEvent("status", statusPayload(1));' in source
    assert 'dispatchSyntheticApiEvent("status", statusPayload(0));' in source


def test_synthetic_execution_events_match_comfyui_execution_shapes() -> None:
    """Synthetic execution events should mirror ComfyUI's websocket adapter payloads."""
    source = _modal_toggle_source()

    assert 'dispatchSyntheticApiEvent("execution_start", {' in source
    assert "timestamp: nowMs()," in source
    assert 'dispatchSyntheticApiEvent("executing", displayNode);' in source
    assert 'dispatchSyntheticApiEvent("notification", {' in source
    assert "Waiting for remote capacity." in source


def test_global_modal_status_badge_is_installed() -> None:
    """The frontend should expose a dedicated global Modal activity indicator."""
    source = _modal_toggle_source()

    assert 'element.id = "comfy-modal-global-status";' in source
    assert 'class="modal-status-gpu" hidden' in source
    assert "gpuText.textContent = activeState.modalGpu ?? \"\";" in source
    assert "font-size: 10px;" in source
    assert "Preparing remote workflow" in source
    assert "Waiting for Modal startup" in source
    assert "Receiving remote outputs" in source
    assert "Remote workflow running on" in source
    assert "setGlobalStatusBatchProgress(promptId, value, maxValue)" in source
    assert "batchValue: state.batchValue ?? null," in source
    assert "batchMax: state.batchMax ?? null," in source
    assert "statusMessage: state.statusMessage ?? null," in source
    assert "statusCurrent: state.statusCurrent ?? null," in source
    assert "statusTotal: state.statusTotal ?? null," in source
    assert "modalGpu: state.modalGpu ?? null," in source
    assert "linear-gradient(90deg" in source
    assert "installGlobalStatusStyles()" in source
    assert "function pruneGlobalStatusStates()" in source
    assert "function effectiveGlobalStatusPhase(promptId, phase)" in source
    assert "function promptHasLiveRemoteWork(promptId)" in source
    assert "function reconcilePromptGlobalStatus(promptId)" in source
    assert "function promptRemoteNodeCount(promptId, fallbackCount = 1)" in source
    assert "nodeCount: promptRemoteNodeCount(promptId, nodeCount)," in source
    assert "hasRemoteExecutionStarted: false" in source
    assert "Waiting for Modal container" in source
    assert "Starting remote component" in source
    assert "Cancelling remote workflow" in source
    assert "function promptActiveNodeIsLive(promptId)" in source


def test_configurator_renders_out_of_band_plan_and_status() -> None:
    """Configured workflows should expose planning before normal node execution."""
    source = _modal_toggle_source()

    assert "function serializedRemoteConfiguratorNodeId(prompt)" in source
    assert "function mountRemoteExecutionConfiguratorPanel(node)" in source
    assert "function removeStaleRemoteConfiguratorWidgets(node)" in source
    assert "function registerPromptConfigurator(promptId, configuratorNodeId)" in source
    assert "function registerRemoteConfiguratorPlan(promptId, assignments, configurations)" in source
    assert "function renderRemoteConfiguratorStatus(panel, state)" in source
    assert "function renderRemoteConfiguratorPlan(panel, assignments, configurations)" in source
    assert 'widget = node.addDOMWidget(\n      "remote_execution_plan"' in source
    assert "serialize: false" in source
    assert "getMinHeight: () => panel.minHeight" in source
    assert 'class="remote-configurator-table"' in source
    assert "responsePayload.remote_execution_assignments" in source
    assert "responsePayload.remote_execution_configurations" in source
    assert "responsePayload.remote_execution_configurator_node_id" in source
    assert "detail.remote_execution_assignments" in source
    assert "refreshRemoteConfiguratorPanelForPrompt(promptId);" in source
    assert "setRemoteConfiguratorTerminalStatus(promptId, STATE_COMPLETE);" in source
    assert "const configuratorPanel = remoteConfiguratorPanel(" in source
    assert 'element.style.display = "none";' in source


def test_frontend_defaults_modal_gpu_to_rtx_pro_6000() -> None:
    """Unsaved workflows should receive the requested RTX GPU default."""
    source = _modal_toggle_source()

    assert 'const DEFAULT_MODAL_GPU = "RTX-PRO-6000";' in source


def test_endpoint_chat_node_is_not_offered_nested_modal_execution() -> None:
    """The local endpoint client should not be wrapped in another Modal execution island."""
    source = _modal_toggle_source()

    assert '"ModalEndpointChat"' in source
    assert '"VastAILeaseConfiguration"' in source
    assert '"ModalRemoteConfiguration"' in source
    assert '"VastRemoteConfiguration"' in source
    assert '"SshRemoteConfiguration"' in source
    assert '"RemoteExecutionConfigurator"' in source
    assert "!LOCAL_MODAL_NODE_IDS.has(String(node.comfyClass))" in source
    assert "!LOCAL_MODAL_NODE_IDS.has(String(nodeData.name))" in source


def test_empty_modal_status_events_do_not_show_global_pill() -> None:
    """Queue requests without Modal-enabled nodes should not create global Modal UI state."""
    source = _modal_toggle_source()

    assert "if (nodeIds.length === 0 && components.length === 0) {\n    return;\n  }" in source


def test_remote_modal_status_tracks_active_node_ids() -> None:
    """The frontend should track and highlight the currently active remote node."""
    source = _modal_toggle_source()

    assert "activeNodeId: null" in source
    assert "hasStreamedProgress: false" in source
    assert "descendantNodeIdsByAncestor: new Map()" in source
    assert "function setPromptActiveNode(promptId, activeNodeId)" in source
    assert "detail.active_node_id" in source
    assert "clearPromptRemoteStates(promptId)" in source


def test_interrupted_prompts_clear_modal_ui_by_prompt_id() -> None:
    """Interrupt cleanup should not depend on native execution events naming a specific node."""
    source = _modal_toggle_source()

    assert "const modalTerminalPromptStates = new Map();" in source
    assert "const modalCancellingPromptIds = new Set();" in source
    assert "function isPromptTerminal(promptId)" in source
    assert "function isPromptCancelling(promptId)" in source
    assert "function markPromptTerminal(promptId, phase)" in source
    assert 'if (detail.phase === "execution_interrupted") {' in source
    assert "function handlePromptInterruption(promptId)" in source
    assert "modalCancellingPromptIds.delete(promptId);" in source
    assert 'markPromptTerminal(promptId, "execution_interrupted");' in source
    assert "clearGlobalStatusPhase(promptId);" in source
    assert "clearPromptRemoteStates(promptId);" in source
    assert 'api.addEventListener("execution_interrupted", (event) => {' in source
    assert "const promptId = String(eventDetail(event).prompt_id ?? \"\");" in source
    assert "handlePromptInterruption(promptId);" in source


def test_error_prompts_clear_remote_node_visual_state() -> None:
    """Prompt errors should not leave stale remote node colors behind for the next submission."""
    source = _modal_toggle_source()

    assert "function clearPromptRemoteNodeVisuals(promptId)" in source
    assert "for (const [nodeIdValue, state] of modalNodeStates.entries()) {" in source
    assert "clearNodeTimer(nodeIdValue);" in source
    assert "clearNodeProgress(nodeIdValue, promptId);" in source
    assert "clearNodeCached(nodeIdValue, promptId);" in source
    assert "modalNodeStates.delete(nodeIdValue);" in source
    assert "clearPromptRemoteNodeVisuals(promptId);" in source
    assert "setNodesPhase(componentNodeIds, STATE_ERROR, promptId, detail.exception_message);" not in source


def test_remote_modal_uses_distinct_ready_active_and_complete_colors() -> None:
    """The frontend should distinguish ready, active, and completed remote nodes visually."""
    source = _modal_toggle_source()

    assert 'const READY_ACTIVE_COMPONENT_BORDER_COLOR = "#22c55e";' in source
    assert 'const READY_INACTIVE_COMPONENT_BORDER_COLOR = "#166534";' in source
    assert 'const STARTING_BORDER_COLOR = "#eab308";' in source
    assert 'const ACTIVE_BORDER_COLOR = "#a855f7";' in source
    assert 'const COMPLETE_BORDER_COLOR = "#004FA4";' in source
    assert 'const COMPLETE_FILL_COLOR = "#001C71";' in source
    assert 'const FINALIZING_NODE_BORDER_COLOR = "#00358A";' in source
    assert 'const CANCELLING_BORDER_COLOR = "#fb7185";' in source
    assert 'const STATE_STARTING = "starting";' in source
    assert 'const STATE_WAITING = "waiting";' in source
    assert 'const STATE_FINALIZING = "finalizing";' in source
    assert 'const STATE_CANCELLING = "cancelling";' in source
    assert 'const STATE_READY = "ready";' in source
    assert 'const STATE_ACTIVE = "active";' in source
    assert 'detail.phase === "execution_success"' in source
    assert "phase === STATE_COMPLETE || phase === STATE_FINALIZING" in source
    assert "state?.phase === STATE_STARTING" in source
    assert "dot.style.background = STARTING_BORDER_COLOR;" in source
    assert "state?.phase === STATE_FINALIZING" in source
    assert "borderColor = FINALIZING_NODE_BORDER_COLOR;" in source
    assert "setNodesPhase(nodeIds, STATE_FINALIZING, promptId);" in source


def test_nodes_2_0_dom_nodes_receive_remote_state_decorations() -> None:
    """Nodes 2.0 should receive the same Modal palette as legacy nodes."""
    source = _modal_toggle_source()

    assert "function remoteDecorationPalette(state, elapsed)" in source
    assert "function syncVueNodeDecorations(timestamp = performance.now())" in source
    assert 'document.querySelectorAll(".lg-node[data-node-id]")' in source
    assert 'nodeElement.dataset.modalPhase = phase;' in source
    assert 'decoration.style.borderColor = palette?.borderColor ?? "transparent";' in source
    assert (
        'decoration.style.backgroundColor = palette?.fillColor ?? "transparent";'
        in source
    )
    assert 'decoration.style.boxShadow = palette ?' in source
    assert 'decoration.className = "comfy-modal-vue-node-decoration";' in source
    assert 'badge.className = "comfy-modal-vue-node-badge";' in source


def test_nodes_2_0_decorations_follow_vue_mounts_and_visual_refreshes() -> None:
    """Late Vue mounts and visual refreshes should resynchronize DOM nodes."""
    source = _modal_toggle_source()

    assert "function installVueNodeDecorationObserver()" in source
    assert "vueNodeObserver = new MutationObserver((records) => {" in source
    assert 'addedNode.matches?.(".lg-node[data-node-id]")' in source
    assert (
        "vueNodeObserver.observe(document.body, { childList: true, subtree: true });"
        in source
    )
    assert "function refreshNodeDecorations()" in source
    assert "queueVueNodeDecorationSync();" in source
    assert "installVueNodeDecorationObserver();" in source
    assert source.count("app.graph?.setDirtyCanvas(true, true);") == 1


def test_cancel_click_shows_immediate_modal_cancelling_feedback() -> None:
    """Cancel should paint Modal prompts as cancelling before backend cleanup finishes."""
    source = _modal_toggle_source()

    assert "function patchInterruptFeedback()" in source
    assert "api.fetchApi = async function modalFetchApi(resource, options) {" in source
    assert "MODAL_CANCEL_PREPARATION_ROUTE" in source
    assert 'body: JSON.stringify({ prompt_id: promptId })' in source
    assert "function promptIdsFromInterruptRequest(resource, options)" in source
    assert 'route.includes("/interrupt")' in source
    assert "function markPromptCancellationRequested(promptId)" in source
    assert "modalCancellingPromptIds.add(promptId);" in source
    assert "setGlobalStatusPhase(promptId, STATE_CANCELLING" in source
    assert "setNodesPhase(remoteNodeIds, STATE_CANCELLING, promptId);" in source
    assert "function clearSupersededCancellingPrompts(activePromptId)" in source
    assert 'markPromptTerminal(promptId, "superseded_by_new_prompt");' in source
    assert "clearSupersededCancellingPrompts(promptId);" in source
    assert "if (isPromptCancelling(promptId)) {\n    return;\n  }" in source
    assert "patchInterruptFeedback();" in source


def test_starting_modal_status_marks_component_before_remote_progress() -> None:
    """Dispatch-time Modal status should mark the component while remote progress is still pending."""
    source = _modal_toggle_source()

    assert "if (detail.phase === STATE_STARTING) {" in source
    assert "setGlobalStatusPhase(promptId, STATE_STARTING, nodeIds.length, {" in source
    assert "setNodesPhase(nodeIds, STATE_STARTING, promptId);" in source
    assert "phases.find((state) => state.phase === STATE_STARTING)" in source
    assert "function isPulsingNodePhase(phase)" in source
    assert "STATE_FINALIZING" in source
    assert "activeState.phase === STATE_STARTING" in source
    assert "STARTING_BORDER_COLOR" in source


def test_remote_modal_nodes_show_component_badges() -> None:
    """Remote nodes should render a compact component label badge."""
    source = _modal_toggle_source()

    assert "componentLabelByMember: new Map()," in source
    assert "promptState.componentLabelByMember.set(componentNodeId, componentLabel);" in source
    assert "componentLabel: state.isRemoteContainer" in source
    assert '(state?.isRemoteContainer ? "Σ" : state?.componentLabel);' in source
    assert "if (nodeBadgeText) {" in source
    assert "ctx.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);" in source
    assert "ctx.fillText(String(nodeBadgeText), badgeX, badgeY + 0.5 / scale);" in source


def test_planner_marks_sandwiched_local_nodes_without_error_styling() -> None:
    """Local re-entry advice should preserve default node colors and explain itself."""
    source = _modal_toggle_source()

    assert "const modalSandwichedLocalNodeIds = new Set();" in source
    assert "function isSandwichedLocalNode(node)" in source
    assert "function localBottleneckDecorationPalette()" not in source
    assert "LOCAL_BOTTLENECK_BORDER_COLOR" not in source
    assert "LOCAL_BOTTLENECK_FILL_COLOR" not in source
    assert "LOCAL_BOTTLENECK_SHADOW_COLOR" not in source
    assert 'const palette = localBottleneck ? null :' in source
    assert 'decoration.style.borderColor = palette?.borderColor ?? "transparent";' in source
    assert 'decoration.style.boxShadow = palette ?' in source
    assert 'const phase = localBottleneck ? "local-bottleneck"' in source
    assert "const nodeBadgeText = localBottleneck" in source
    assert (
        'const LOCAL_BOTTLENECK_TOOLTIP = "Did you mean to make this node execute '
        'on Modal?";'
        in source
    )
    assert "badge.title = localBottleneck" in source
    assert "function localBottleneckBadgeContainsPoint(" in source
    assert "updateLegacyModalTooltip(graphCanvas, tooltip);" in source
    assert 'data-modal-phase="local-bottleneck"' in source
    assert "pointer-events: auto;" in source
    assert "cursor: help;" in source
    assert "responsePayload.modal_sandwiched_local_node_ids" in source
    assert "setSandwichedLocalNodeIds(sandwichedLocalNodeIds);" in source


def test_global_modal_status_badge_supports_setup_and_finalizing_details() -> None:
    """The frontend should surface detailed setup and result-receive messages in the global pill."""
    source = _modal_toggle_source()

    assert 'const FINALIZING_BORDER_COLOR = "#3b82f6";' in source
    assert "detail.status_message ?? null" in source
    assert "detail.status_current ?? null" in source
    assert "detail.status_total ?? null" in source
    assert "setGlobalStatusPhase(promptId, STATE_SETUP, nodeIds.length, {" in source
    assert "if (detail.phase === STATE_FINALIZING) {" in source
    assert "dot.style.background = FINALIZING_BORDER_COLOR;" in source
    assert "READY_BORDER_COLOR" not in source
    assert 'text.textContent = activeState.statusMessage ?? "Receiving remote outputs";' in source


def test_global_modal_status_badge_polls_and_renders_active_containers() -> None:
    """The global pill should adaptively poll and show every active Modal container."""
    source = _modal_toggle_source()

    assert 'const MODAL_CONTAINER_STATUS_ROUTE = MODAL_ROUTE.replace(' in source
    assert '"/container_status"' in source
    assert "function pollModalContainerStatus()" in source
    assert "function modalContainerStatusPollDelay()" in source
    assert "CONTAINER_STATUS_FAST_POLL_MS = 1500" in source
    assert "CONTAINER_STATUS_STABLE_POLL_MS = 5000" in source
    assert "CONTAINER_STATUS_HIDDEN_POLL_MS = 15000" in source
    assert "CONTAINER_STATUS_MAX_BACKOFF_MS = 30000" in source
    assert "document.visibilityState !== \"visible\"" in source
    assert "modalContainerStatusUnchangedPolls >= 2" in source
    assert "modalContainerStatusFailureCount > 0" in source
    assert "function renderModalContainerStatuses(containerElement)" in source
    assert "modalContainerStatuses.forEach((container, index) =>" in source
    assert "Container ${index + 1} · ${state}" in source
    assert 'class="modal-status-containers" hidden' in source
    assert "requestImmediateModalContainerStatusPoll();" in source
    assert "?modal_gpu=${encodeURIComponent(requestedModalGpu)}" in source


def test_global_modal_status_badge_estimates_prompt_gpu_cost() -> None:
    """The global pill should integrate active GPU-container seconds per prompt."""
    source = _modal_toggle_source()

    assert 'class="modal-status-cost" hidden' in source
    assert "estimatedGpuCostPerSecond:" in source
    assert "function modalPromptCostContainers(promptId, containers)" in source
    assert "container.modalGpu === selectedModalGpu" in source
    assert "function modalContainerGpuBurnRate(" in source
    assert "function updateModalContainerCostEstimate(" in source
    assert "function liveModalContainerCostEstimate()" in source
    assert "previousIntervalSeconds *" in source
    assert "modalContainerGpuBurnRate(previousPromptStatuses, promptId)" in source
    assert "modalContainerCostUpdatedAtSeconds" in source
    assert "Estimated GPU cost ${formatEstimatedModalUsd(estimatedCostUsd)}" in source
    assert "formatEstimatedModalUsd(burnRatePerMinuteUsd)}/min" in source
    assert "modalContainerStatusPromptId !== requestedPromptId" in source


def test_global_modal_status_badge_renders_hourly_app_billing() -> None:
    """The global pill should show cached actual billing for the selected GPU app."""
    source = _modal_toggle_source()

    assert 'class="modal-status-billing" hidden' in source
    assert "function normalizedModalHourlyBillingStatus(payload)" in source
    assert "appCostUsdBeforeCredits:" in source
    assert "function renderModalHourlyBilling(billingElement, activeState)" in source
    assert "Reported app cost ${reportedCost} · hour ending ${intervalEnd}" in source
    assert "actual Modal metered cost before credits and reservations" in source
    assert "Hourly reports exclude the partial current hour" in source
    assert 'element.querySelector(".modal-status-billing")' in source
    assert "renderModalHourlyBilling(billingText, activeState);" in source


def test_queue_success_marks_all_remote_nodes_ready_before_component_execution() -> None:
    """Once the Modal route accepts the prompt, all remote nodes should flip from setup to ready."""
    source = _modal_toggle_source()

    assert "const modalQueuedPromptIds = new Set();" in source
    assert "function markPromptQueuedBehindActiveModal(promptId)" in source
    assert "function isPromptQueuedBehindActiveModal(promptId)" in source
    assert "const queuedBehindActiveModal =" in source
    assert "if (!queuedBehindActiveModal) {" in source
    assert "const resolvedRemoteNodeIds = (responsePayload.modal_remote_node_ids ?? []).map((nodeIdValue) =>" in source
    assert "const resolvedComponents = Array.isArray(responsePayload.modal_components)" in source
    assert "registerPromptComponents(promptId, resolvedRemoteNodeIds, resolvedComponents);" in source
    assert "if (!promptState) {\n          return responsePayload;\n        }" in source
    assert "endSyntheticExecutionUi(promptId);" in source
    assert 'setGlobalStatusPhase(promptId, STATE_WAITING, acceptedRemoteNodeIds.length, {' in source
    assert "message: remoteCapacityWaitingMessage(selectedProviders)," in source
    assert "setNodesPhase(acceptedRemoteNodeIds, STATE_READY, promptId);" in source


def test_queue_failure_preserves_server_validation_detail() -> None:
    """Synthetic ComfyUI errors should show the planner's actionable rejection."""
    source = _modal_toggle_source()

    assert "function queueErrorMessage(error)" in source
    assert "promptError.modalQueueResponse = responsePayload;" in source
    assert (
        "endSyntheticExecutionUi(promptId, true, queueErrorMessage(error));"
        in source
    )
    assert (
        'failureMessage || "Modal queue request failed before prompt execution started."'
        in source
    )
    assert "endSyntheticExecutionUi(promptId, true, errorMessage);" in source
    assert (
        "setRemoteConfiguratorTerminalStatus(promptId, STATE_ERROR, errorMessage);"
        in source
    )


def test_completed_remote_nodes_clear_stale_global_status_entries() -> None:
    """The global pill should clear once a prompt has no active remote work left."""
    source = _modal_toggle_source()

    assert "promptState?.hasRemoteExecutionStarted &&\n    nodeStates.some((state) => state.phase === STATE_READY)" in source
    assert "nodeStates.some((state) => state.phase === STATE_READY || state.phase === STATE_COMPLETE)" not in source
    assert "modalGlobalStatusStates.delete(promptId);" in source
    assert "reconcilePromptGlobalStatus(promptId);" in source
    assert "promptActiveNodeIsLive(promptId)" in source
    assert "activeNodeState?.promptId === promptId && activeNodeState.phase === STATE_ACTIVE" in source
    assert "if (promptState?.activeNodeId === String(currentNodeId))" in source


def test_streamed_modal_progress_takes_precedence_over_proxy_events() -> None:
    """Once streamed node progress starts, coarse proxy execution events should stop overriding it."""
    source = _modal_toggle_source()

    assert "promptState.hasStreamedProgress = true;" in source
    assert "promptState.hasRemoteExecutionStarted = true;" in source
    assert "if (promptState.hasStreamedProgress && phase === EXECUTION_PHASE)" in source


def test_streamed_modal_execution_ends_synthetic_setup_without_waiting_for_final_executed() -> None:
    """Real streamed execution should end synthetic setup on progress, not on the first executed node."""
    source = _modal_toggle_source()

    assert "if (detail.phase === EXECUTION_PHASE) {\n    clearPromptQueued(promptId);\n    endSyntheticExecutionUi(promptId);" in source
    assert "function handleModalProgress(event)" in source
    assert "  endSyntheticExecutionUi(promptId);" in source
    assert 'endSyntheticExecutionUi(String(eventDetail(event).prompt_id ?? ""));' not in source


def test_streamed_modal_node_progress_updates_active_overlay() -> None:
    """The frontend should listen for numeric Modal node progress and render it on the node."""
    source = _modal_toggle_source()

    assert 'const PROGRESS_FADE_MS = 900;' in source
    assert "function fadeNodeProgress(nodeIdValue, promptId)" in source
    assert "function progressVisualOpacity(progressState)" in source
    assert 'api.addEventListener("modal_progress", handleModalProgress);' in source


def test_mixed_remote_environments_render_clipped_runtime_locations() -> None:
    """Mixed planner assignments should add provider identity beneath progress bars."""
    source = _modal_toggle_source()

    assert "function registerPromptExecutionAssignments(promptId, assignments)" in source
    assert "promptState.scheduledEnvironmentCount = environmentIds.size;" in source
    assert "updateNodeExecutionLocation(promptId, progressNodeId, detail);" in source
    assert "updateNodeExecutionLocation(promptId, nextActiveNodeId, detail);" in source
    assert "function drawRemoteExecutionLocation(" in source
    assert "function updateVueExecutionLocation(decoration, state)" in source
    assert 'text-overflow: ellipsis;' in source
    assert "Number(state?.scheduledEnvironmentCount ?? 0) > 1" in source
    assert "fitCanvasText(ctx, label, availableTextWidth)" in source
    assert "REMOTE_LOCATION_ICON_SOURCES" in source
    assert "function handleModalProgress(event)" in source
    assert "if (detail.aggregate_only) {" in source
    assert "setNodeBatchProgress(" in source
    assert (
        "function setNodeProgress(nodeIdValue, promptId, value, maxValue, metadata = {})"
        in source
    )
    assert 'unit === "tokens" ? "tok/s" : "it/s"' in source
    assert "timeToFirstTokenSeconds" in source
    assert "if (state.progress.indeterminate) {" in source
    assert "preGpu: Boolean(metadata.pre_gpu)" in source
    assert "state.phase === STATE_STARTING && state.progress.preGpu" in source
    assert "function setNodeProgressLane(nodeIdValue, promptId, laneId, value, maxValue, itemIndex, setupOnly = false)" in source
    assert "function clearNodeProgressLane(nodeIdValue, promptId, laneId)" in source
    assert "function clearNodeProgress(nodeIdValue, promptId)" in source
    assert "fadeNodeProgress(nodeIdValue, promptId);" in source
    assert "function progressIterationRate(previousState, value, maxValue, updatedAt)" in source
    assert "function formatIterationRate(iterationRate, unit = null)" in source
    assert "function drawIterationRateOverlay(" in source
    assert 'ctx.fillStyle = "rgba(0, 0, 0, 0.9)";' in source
    assert "const progressBarWidth = barWidth;" in source
    assert "iterationRateColumnWidth" not in source
    assert ": progressIterationRate(" in source
    assert "laneProgress.iterationRate," in source
    assert "state.progress.iterationRate," in source
    assert "state?.progress" in source
    assert "state?.progressLanes" in source
    assert "detail.lane_id != null" in source
    assert "clearNodeProgressLane(progressNodeId, promptId, String(detail.lane_id));" in source
    assert "setNodeProgressLane(" in source
    assert "function deriveRemoteNodePhase(phase, hasLiveProgress)" in source
    assert "isActiveRemoteNode: hasLiveProgress || promptState?.activeNodeId === visualNodeId," in source
    assert "isActiveComponentMember: isNodeInActiveComponent(state.promptId, visualNodeId)," in source
    assert "isCachedRemoteNode: Boolean(cachedState)," in source
    assert 'const panelY = node.size[1] + 6 / scale;' in source
    assert 'ctx.roundRect(-borderWidth, panelY, barWidth, panelHeight, 10 / scale);' in source
    assert "const llmTimingLabel = aggregateProgress?.timeToFirstTokenSeconds" in source


def test_modal_canvas_animation_loop_is_selective_and_throttled() -> None:
    """Static progress data should not keep an unbounded canvas rAF loop alive."""
    source = _modal_toggle_source()

    assert "const MODAL_ANIMATION_FRAME_INTERVAL_MS = 100;" in source
    assert "let modalLastAnimationRedrawAt = 0;" in source
    assert "function shouldAnimateModalVisuals()" in source
    assert "function progressNeedsAnimation()" in source
    assert "function stopAnimationLoopIfIdle()" in source
    assert "function isPulsingNodePhase(phase)" in source
    assert "progressFadeNeedsAnimation(progressState)" in source
    assert "if (laneProgress.setupOnly || progressFadeNeedsAnimation(laneProgress))" in source
    assert "timestamp - modalLastAnimationRedrawAt >= MODAL_ANIMATION_FRAME_INTERVAL_MS" in source
    assert "cancelAnimationFrame(animationFrameHandle);" in source
    assert "const hasProgressState =" not in source
    assert "const hasCachedPulse =" not in source


def test_streamed_modal_node_progress_fades_previous_active_node() -> None:
    """Only same-component progress should complete the previous active node."""
    source = _modal_toggle_source()

    assert "function nodesShareRemoteComponent(promptId, leftNodeId, rightNodeId)" in source
    assert "const previousActiveNodeId = promptState.activeNodeId;" in source
    assert "previousActiveNodeId !== progressNodeId" in source
    assert "nodesShareRemoteComponent(promptId, previousActiveNodeId, progressNodeId)" in source
    assert "fadeNodeProgress(previousActiveNodeId, promptId);" in source
    assert "setNodesPhase([previousActiveNodeId], STATE_COMPLETE, promptId);" in source
    assert "setPromptActiveNode(promptId, progressNodeId);" in source


def test_parallel_component_progress_does_not_complete_other_active_nodes() -> None:
    """Parallel component progress should not treat the prompt-wide previous active node as complete."""
    source = _modal_toggle_source()

    assert "nodesShareRemoteComponent(promptId, previousActiveNodeId, nextActiveNodeId)" in source
    assert "nodesShareRemoteComponent(promptId, previousActiveNodeId, progressNodeId)" in source
    assert "promptState.activeNodeId && nodeIds.includes(promptState.activeNodeId)" in source


def test_streamed_modal_node_progress_completes_reported_ancestors() -> None:
    """When a remote node becomes active, reported upstream nodes should be marked complete."""
    source = _modal_toggle_source()

    assert "function completeRemoteAncestorsBeforeActiveNode(promptId, ancestorNodeIds, activeNodeId)" in source
    assert "detail.completed_ancestor_node_ids" in source
    assert "setNodesPhase(completedNodeIds, STATE_COMPLETE, promptId);" in source
    assert "completeRemoteAncestorsBeforeActiveNode(\n    promptId,\n    detail.completed_ancestor_node_ids,\n    progressNodeId,\n  );" in source
    assert "completeRemoteAncestorsBeforeActiveNode(\n      promptId,\n      detail.completed_ancestor_node_ids,\n      nextActiveNodeId,\n    );" in source


def test_ready_updates_do_not_downgrade_completed_remote_nodes() -> None:
    """Coarse execution status should not turn already completed remote nodes green again."""
    source = _modal_toggle_source()

    assert "const existingState = modalNodeStates.get(currentNodeId);" in source
    assert "phase === STATE_READY &&\n      existingState?.promptId === promptId" in source
    assert "[STATE_COMPLETE, STATE_FINALIZING, STATE_ERROR].includes(existingState.phase)" in source


def test_mapped_parallel_modal_progress_renders_multiple_lane_bars() -> None:
    """Parallel mapped Modal runs should render one local progress lane per active worker."""
    source = _modal_toggle_source()

    assert "const modalNodeProgressLanes = new Map();" in source
    assert "const modalNodeBatchProgress = new Map();" in source
    assert "visibleLaneProgress.length > 0" in source
    assert "const laneColors = [" in source
    assert 'const badgeText = hasBatchBadge' in source
    assert 'const badgeY = panelY + panelPaddingY;' in source
    assert "if (detail.clear) {" in source
    assert "clearNodeProgress(progressNodeId, promptId);" in source
    assert 'let laneY = barY;' in source
    assert "laneNodeIdsByLane: new Map()," in source
    assert "representativeNodeIdByMember: new Map()," in source
    assert "function deleteNodeProgressLane(nodeIdValue, promptId, laneId)" in source
    assert "function laneOwnerKey(promptId, nodeIdValue, laneId)" in source
    assert "promptState.laneNodeIdsByLane.set(safeLaneKey, safeNodeIdValue);" in source
    assert "const setupProgressLanes = progressLanes.filter((laneProgress) => laneProgress.setupOnly);" in source
    assert "const activeProgressLanes = progressLanes.filter((laneProgress) => !laneProgress.setupOnly);" in source
    assert "laneProgress.setupOnly" in source
    assert 'laneColor.replace("0.94)", `${0.28 + elapsedPulse * 0.22})`)' in source


def test_mapped_lane_setup_events_render_placeholder_lane_bars() -> None:
    """Lane setup events should render translucent placeholder bars before real node progress arrives."""
    source = _modal_toggle_source()

    assert "function setNodeProgressLane(nodeIdValue, promptId, laneId, value, maxValue, itemIndex, setupOnly = false)" in source
    assert "setupOnly: Boolean(setupOnly)," in source
    assert "Boolean(detail.setup_only)," in source
    assert "const hasSetupLaneProgress = setupProgressLanes.length > 0;" in source
    assert "const hasVisibleLaneProgress = visibleLaneProgress.length > 0;" in source


def test_ready_nodes_distinguish_active_component_members_and_cached_nodes() -> None:
    """Ready nodes should visually distinguish active-component membership and cache hits."""
    source = _modal_toggle_source()

    assert "function traceRoundedRectPath(ctx, x, y, width, height, radius)" in source
    assert "function activeProgressNodeIds(promptId)" in source
    assert "function isNodeInActiveComponent(promptId, nodeIdValue)" in source
    assert "const promptActiveNodeId = promptState.activeNodeId;" in source
    assert "for (const liveProgressNodeId of activeProgressNodeIds(promptId)) {" in source
    assert "const liveProgressComponentNodeIds = resolveComponentNodeIds(promptId, liveProgressNodeId);" in source
    assert "const descendantNodeIds = promptState.descendantNodeIdsByAncestor.get(safeNodeIdValue);" in source
    assert "const pulseRate = state?.isCachedRemoteNode ? 2 : 6;" in source
    assert "borderColor = state?.isActiveComponentMember" in source
    assert "READY_INACTIVE_COMPONENT_BORDER_COLOR" in source
    assert "READY_ACTIVE_COMPONENT_BORDER_COLOR" in source
    assert "const cornerRadius = 12 / scale;" in source
    assert "traceRoundedRectPath(" in source
    assert "ctx.strokeRect(" not in source


def test_cached_node_hits_are_marked_without_fake_progress() -> None:
    """Cache-hit markers should be tracked separately from numeric progress."""
    source = _modal_toggle_source()

    assert "const modalNodeCachedStates = new Map();" in source
    assert "function markNodeCached(nodeIdValue, promptId)" in source
    assert "function clearNodeCached(nodeIdValue, promptId)" in source
    assert "function nodeCachedState(nodeIdValue, promptId)" in source
    assert "if (detail.cached_hit) {" in source
    assert "const cachedNodeIds = new Set(" in source
    assert "[detail.node_id, detail.display_node_id, detail.real_node_id]" in source
    assert "for (const cachedNodeId of cachedNodeIds) {" in source
    assert "markNodeCached(cachedNodeId, promptId);" in source
    assert "function shouldAnimateModalVisuals()" in source
    assert "const hasCachedPulse =" not in source


def test_modal_context_menu_can_expand_required_upstream_nodes() -> None:
    """Right-clicking a node should offer a dry-run expansion action backed by the backend."""
    source = _modal_toggle_source()

    assert 'const MODAL_ANALYZE_ROUTE = MODAL_ROUTE.replace(/\\/queue_prompt$/, "/analyze_remote_nodes");' in source
    assert "function workflowNodePath(node)" in source
    assert "function findNodeByWorkflowPath(workflowPath)" in source
    assert "function selectedWorkflowNodePaths(node)" in source
    assert "function analyzeAndSetUpstreamRemoteNodes(node, value)" in source
    assert "function requestModalMaintenance(route, successMessage)" in source
    assert "function installModalContextMenu(nodeType, nodeData)" in source
    assert 'api.fetchApi(MODAL_ANALYZE_ROUTE, {' in source
    assert 'const MODAL_DELETE_CACHES_ROUTE = MODAL_ROUTE.replace(/\\/queue_prompt$/, "/delete_caches");' in source
    assert 'const MODAL_DELETE_VOLUME_ROUTE = MODAL_ROUTE.replace(/\\/queue_prompt$/, "/delete_volume");' in source
    assert "async beforeRegisterNodeDef(nodeType, nodeData)" in source
    assert "installModalContextMenu(nodeType, nodeData);" in source
    assert 'content: "Remote Execution Tools"' in source
    assert 'const REMOTE_WIDGET_NAME = "Run Remotely";' in source
    assert 'const LEGACY_REMOTE_WIDGET_NAME = "Run on Modal";' in source
    assert "has_submenu: true" in source
    assert "submenu: {" in source
    assert '"Enable on Upstream Nodes"' in source
    assert '"Enable on Upstream Nodes for Selection"' in source
    assert '"Disable on Upstream Nodes"' in source
    assert '"Disable on Upstream Nodes for Selection"' in source
    assert '"Enable All Nodes"' in source
    assert '"Disable All Nodes"' in source
    assert '"Delete Modal Caches"' in source
    assert '"Delete Modal Volume"' in source
    assert '"Modal: Enable on Upstream Nodes"' not in source
    assert '"Modal: Disable on Upstream Nodes"' not in source
    assert '"Modal: Enable All Nodes"' not in source
    assert "analyzeAndSetUpstreamRemoteNodes(this, true)" in source
    assert "analyzeAndSetUpstreamRemoteNodes(this, false)" in source
    assert "setAllEligibleWorkflowNodesRemote(true);" in source
    assert "setAllEligibleWorkflowNodesRemote(false);" in source
    assert "requestModalMaintenance(" in source
    assert "MODAL_DELETE_CACHES_ROUTE" in source
    assert "MODAL_DELETE_VOLUME_ROUTE" in source


def test_modal_context_menu_marks_nodes_across_subgraphs() -> None:
    """The UI action should be able to resolve and mark nested workflow-node paths."""
    source = _modal_toggle_source()

    assert "function rootGraph()" in source
    assert "function findSomethingInAllSubgraphs(matcher)" in source
    assert "function allWorkflowNodes()" in source
    assert "function findContainingSubgraphNode(subgraphId)" in source
    assert "candidate.subgraph?.id === subgraphId" in source
    assert "function setWorkflowNodePathsRemote(workflowNodePaths, value)" in source
    assert "setRemoteFlag(node, value);" in source
    assert "node.__modalToggleWidget.value = enabled;" in source


def test_modal_context_menu_can_set_all_eligible_nodes() -> None:
    """The graph-wide menu actions should toggle every eligible live workflow node."""
    source = _modal_toggle_source()

    assert "function setAllEligibleWorkflowNodesRemote(value)" in source
    assert "for (const node of allWorkflowNodes()) {" in source
    assert "if (!isEligibleNode(node)) {" in source
    assert "setRemoteFlag(node, value);" in source
    assert 'const actionLabel = value ? "Enabled" : "Disabled";' in source
    assert "${actionLabel} Modal on ${appliedCount} node" in source


def test_prompt_component_registration_does_not_shrink_remote_node_count() -> None:
    """Per-component status updates should not overwrite the prompt-wide remote node list."""
    source = _modal_toggle_source()

    assert "const mergedRemoteNodeIds = new Set(remoteNodeIds.map((nodeIdValue) => String(nodeIdValue)));" in source
    assert "if (promptState.remoteNodeIds.length === 0) {" in source
    assert "const mergedRemoteNodeIds = new Set(promptState.remoteNodeIds);" in source


def test_prompt_cleanup_prunes_orphaned_global_status_entries() -> None:
    """Workflow cleanup should remove stale global badge states once prompt state is gone."""
    source = _modal_toggle_source()

    assert "pruneGlobalStatusStates();" in source
    assert "refreshGlobalStatusElement();" in source


def test_subgraph_descendant_states_percolate_to_visible_ancestor_nodes() -> None:
    """Subgraph-expanded remote prompt ids should aggregate their phase onto visible ancestor nodes."""
    source = _modal_toggle_source()

    assert "function ancestorNodeIds(nodeIdValue)" in source
    assert "function rebuildPromptAncestorMap(promptState)" in source
    assert "function refreshAncestorNodePhase(promptId, ancestorNodeId, errorMessage)" in source
    assert "promptState.descendantNodeIdsByAncestor.get(ancestorNodeId)" in source
    assert "function remoteContainerVisualState(promptId, ancestorNodeId, errorMessage)" in source
    assert "function dominantRemoteContainerPhase(phaseCounts)" in source
    assert "isMixedRemoteContainer: phases.length > 1" in source
    assert "remoteDescendantCount: descendantNodeIds.size" in source


def test_subgraph_containers_receive_recursive_idle_and_runtime_decorations() -> None:
    """Every containing subgraph should summarize marked descendants without becoming remote."""
    source = _modal_toggle_source()

    assert "const modalRemoteDescendantNodeIdsByAncestor = new Map();" in source
    assert "function rebuildRemoteDescendantIndex()" in source
    assert "for (const ancestorNodeId of ancestorNodeIds(remoteNodePath)) {" in source
    assert "function hasRemoteDescendants(node)" in source
    assert "state?.isRemoteContainer" in source
    assert 'state?.isRemoteContainer ? "Σ"' in source
    assert "function remoteContainerTooltip(state)" in source
    assert 'nodeElement.dataset.modalContainer = "true";' in source
    assert '[data-modal-container="true"]' in source
    assert "function installNodeDecorationHooks(node)" in source
    assert "installNodeDecorationHooks(node);" in source


def test_subgraph_nodes_resolve_visual_state_by_composed_workflow_path() -> None:
    """Inner subgraph nodes should read status stored under composed prompt ids."""
    source = _modal_toggle_source()

    assert "const visualNodeId = workflowNodePath(node) || nodeId(node);" in source
    assert "const storedState = modalNodeStates.get(visualNodeId) ?? null;" in source
    assert "const promptContainerState = storedState?.promptId" in source
    assert "const progressState = nodeProgressState(visualNodeId, state.promptId);" in source
    assert "const progressLanes = nodeProgressLanes(visualNodeId, state.promptId);" in source
    assert "const cachedState = nodeCachedState(visualNodeId, state.promptId);" in source
    assert "const batchProgressState = modalNodeBatchProgress.get(visualNodeId) ?? null;" in source
    assert "const hasLiveProgress = hasLiveNodeProgress(visualNodeId, state.promptId);" in source


def test_modal_ui_refreshes_after_visibility_or_focus_returns() -> None:
    """Background-tab throttling should not leave the status pill stale after refocus."""
    source = _modal_toggle_source()

    assert 'const MODAL_PROGRESS_STATE_ROUTE = MODAL_ROUTE.replace(/\\/queue_prompt$/, "/progress_state");' in source
    assert 'const COMFY_QUEUE_ROUTE = "/queue";' in source
    assert 'const COMFY_HISTORY_ROUTE = "/history";' in source
    assert "const REFOCUS_STALE_PROMPT_GRACE_MS = 30000;" in source
    assert "function refreshModalUiAfterVisibilityChange()" in source
    assert "function replayModalUiEventsAfterVisibilityChange()" in source
    assert "function replayModalUiEvent(eventRecord)" in source
    assert "modalReplayedEventUpdatedAtMs = Number.isFinite(updatedAtSeconds)" in source
    assert "function reconcileModalUiAfterVisibilityChange()" in source
    assert "function activeModalUiPromptIds()" in source
    assert "for (const laneState of modalNodeProgressLanes.values())" in source
    assert "for (const batchState of modalNodeBatchProgress.values())" in source
    assert "function promptIdsFromQueuePayload(queuePayload)" in source
    assert "function historyPayloadHasPrompt(historyPayload, promptId)" in source
    assert "function clearRefocusCompletedPrompt(promptId, phase)" in source
    assert "function clearPromptProgressStates(promptId)" in source
    assert "clearPromptProgressStates(promptId);" in source
    assert "stopAnimationLoopIfIdle();" in source
    assert "for (const [nodeIdValue, state] of Array.from(modalNodeStates.entries()))" in source
    assert "const startedAt = modalReplayedEventUpdatedAtMs ?? nowMs();" in source
    assert "modalVisibilityRefreshInFlight" in source
    assert "api.fetchApi(" in source
    assert "fetchComfyJson(COMFY_QUEUE_ROUTE)" in source
    assert "queuedPromptIds.has(promptId)" in source
    assert "`${COMFY_HISTORY_ROUTE}/${encodeURIComponent(promptId)}`" in source
    assert 'clearRefocusCompletedPrompt(promptId, "execution_success");' in source
    assert 'clearRefocusCompletedPrompt(promptId, "stale_refocus_cleanup");' in source
    assert ".then(() => reconcileModalUiAfterVisibilityChange())" in source
    assert "handleModalStatus({ detail: payload });" in source
    assert "handleModalProgress({ detail: payload });" in source
    assert 'document.addEventListener("visibilitychange"' in source
    assert 'window.addEventListener("focus", refreshModalUiAfterVisibilityChange);' in source
