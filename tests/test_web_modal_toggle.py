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
    assert "Waiting for a machine on Modal." in source


def test_global_modal_status_badge_is_installed() -> None:
    """The frontend should expose a dedicated global Modal activity indicator."""
    source = _modal_toggle_source()

    assert 'element.id = "comfy-modal-global-status";' in source
    assert "Syncing graph with Modal" in source
    assert "Waiting for Modal startup" in source
    assert "Receiving Modal outputs" in source
    assert "Modal workflow running on" in source
    assert "setGlobalStatusBatchProgress(promptId, value, maxValue)" in source
    assert "batchValue: state.batchValue ?? null," in source
    assert "batchMax: state.batchMax ?? null," in source
    assert "statusMessage: state.statusMessage ?? null," in source
    assert "statusCurrent: state.statusCurrent ?? null," in source
    assert "statusTotal: state.statusTotal ?? null," in source
    assert "linear-gradient(90deg" in source
    assert "installGlobalStatusStyles()" in source
    assert "function pruneGlobalStatusStates()" in source
    assert "function effectiveGlobalStatusPhase(promptId, phase)" in source
    assert "function promptHasLiveRemoteWork(promptId)" in source
    assert "function reconcilePromptGlobalStatus(promptId)" in source


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

    assert 'if (detail.phase === "execution_interrupted") {' in source
    assert "function handlePromptInterruption(promptId)" in source
    assert "clearGlobalStatusPhase(promptId);" in source
    assert "clearPromptRemoteStates(promptId);" in source
    assert 'api.addEventListener("execution_interrupted", (event) => {' in source
    assert "const promptId = String(eventDetail(event).prompt_id ?? \"\");" in source
    assert "handlePromptInterruption(promptId);" in source


def test_remote_modal_uses_distinct_ready_active_and_complete_colors() -> None:
    """The frontend should distinguish ready, active, and completed remote nodes visually."""
    source = _modal_toggle_source()

    assert 'const READY_ACTIVE_COMPONENT_BORDER_COLOR = "#22c55e";' in source
    assert 'const READY_INACTIVE_COMPONENT_BORDER_COLOR = "#166534";' in source
    assert 'const ACTIVE_BORDER_COLOR = "#a855f7";' in source
    assert 'const COMPLETE_BORDER_COLOR = "#16a34a";' in source
    assert 'const STATE_WAITING = "waiting";' in source
    assert 'const STATE_FINALIZING = "finalizing";' in source
    assert 'const STATE_READY = "ready";' in source
    assert 'const STATE_ACTIVE = "active";' in source
    assert 'detail.phase === "execution_success"' in source


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
    assert 'text.textContent = activeState.statusMessage ?? "Receiving Modal outputs";' in source


def test_queue_success_marks_all_remote_nodes_ready_before_component_execution() -> None:
    """Once the Modal route accepts the prompt, all remote nodes should flip from setup to ready."""
    source = _modal_toggle_source()

    assert "const resolvedRemoteNodeIds = (responsePayload.modal_remote_node_ids ?? []).map((nodeIdValue) =>" in source
    assert "const resolvedComponents = Array.isArray(responsePayload.modal_components)" in source
    assert "registerPromptComponents(promptId, resolvedRemoteNodeIds, resolvedComponents);" in source
    assert "endSyntheticExecutionUi(promptId);" in source
    assert 'setGlobalStatusPhase(promptId, STATE_WAITING, acceptedRemoteNodeIds.length, {' in source
    assert 'message: "Waiting for Modal startup",' in source
    assert "setNodesPhase(acceptedRemoteNodeIds, STATE_READY, promptId);" in source


def test_completed_remote_nodes_clear_stale_global_status_entries() -> None:
    """The global pill should clear once a prompt has no active remote work left."""
    source = _modal_toggle_source()

    assert "if (nodeStates.some((state) => state.phase === STATE_READY)) {" in source
    assert "nodeStates.some((state) => state.phase === STATE_READY || state.phase === STATE_COMPLETE)" not in source
    assert "modalGlobalStatusStates.delete(promptId);" in source
    assert "reconcilePromptGlobalStatus(promptId);" in source


def test_streamed_modal_progress_takes_precedence_over_proxy_events() -> None:
    """Once streamed node progress starts, coarse proxy execution events should stop overriding it."""
    source = _modal_toggle_source()

    assert "promptState.hasStreamedProgress = true;" in source
    assert "if (promptState.hasStreamedProgress && phase === EXECUTION_PHASE)" in source


def test_streamed_modal_execution_ends_synthetic_setup_without_waiting_for_final_executed() -> None:
    """Real streamed execution should end synthetic setup on progress, not on the first executed node."""
    source = _modal_toggle_source()

    assert "if (detail.phase === EXECUTION_PHASE) {\n    endSyntheticExecutionUi(promptId);" in source
    assert "function handleModalProgress(event)" in source
    assert "  endSyntheticExecutionUi(promptId);" in source
    assert 'endSyntheticExecutionUi(String(eventDetail(event).prompt_id ?? ""));' not in source


def test_streamed_modal_node_progress_updates_active_overlay() -> None:
    """The frontend should listen for numeric Modal node progress and render it on the node."""
    source = _modal_toggle_source()

    assert 'api.addEventListener("modal_progress", handleModalProgress);' in source
    assert "function handleModalProgress(event)" in source
    assert "if (detail.aggregate_only) {" in source
    assert "setNodeBatchProgress(" in source
    assert "function setNodeProgress(nodeIdValue, promptId, value, maxValue)" in source
    assert "function setNodeProgressLane(nodeIdValue, promptId, laneId, value, maxValue, itemIndex, setupOnly = false)" in source
    assert "function clearNodeProgressLane(nodeIdValue, promptId, laneId)" in source
    assert "function clearNodeProgress(nodeIdValue, promptId)" in source
    assert "state?.progress" in source
    assert "state?.progressLanes" in source
    assert "detail.lane_id != null" in source
    assert "clearNodeProgressLane(progressNodeId, promptId, String(detail.lane_id));" in source
    assert "setNodeProgressLane(" in source
    assert "function deriveRemoteNodePhase(phase, hasLiveProgress)" in source
    assert "isActiveRemoteNode: hasLiveProgress || promptState?.activeNodeId === nodeId(node)," in source
    assert "isActiveComponentMember: isNodeInActiveComponent(state.promptId, nodeId(node))," in source
    assert "isCachedRemoteNode: Boolean(cachedState)," in source
    assert 'const panelY = node.size[1] + 6 / scale;' in source
    assert 'ctx.roundRect(-borderWidth, panelY, barWidth, panelHeight, 10 / scale);' in source
    assert 'const headerText = hasAggregateProgress' in source


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
    assert "const hasCachedPulse = Array.from(modalNodeCachedStates.values()).length > 0;" in source


def test_modal_context_menu_can_expand_required_upstream_nodes() -> None:
    """Right-clicking a node should offer a dry-run expansion action backed by the backend."""
    source = _modal_toggle_source()

    assert 'const MODAL_ANALYZE_ROUTE = MODAL_ROUTE.replace(/\\/queue_prompt$/, "/analyze_remote_nodes");' in source
    assert "function workflowNodePath(node)" in source
    assert "function findNodeByWorkflowPath(workflowPath)" in source
    assert "function selectedWorkflowNodePaths(node)" in source
    assert "function analyzeAndMarkRequiredRemoteNodes(node)" in source
    assert "function installModalContextMenu(nodeType, nodeData)" in source
    assert 'api.fetchApi(MODAL_ANALYZE_ROUTE, {' in source
    assert "async beforeRegisterNodeDef(nodeType, nodeData)" in source
    assert "installModalContextMenu(nodeType, nodeData);" in source
    assert '"Modal: Include Required Upstream Nodes"' in source
    assert '"Modal: Include Required Upstream Nodes for Selection"' in source


def test_modal_context_menu_marks_nodes_across_subgraphs() -> None:
    """The UI action should be able to resolve and mark nested workflow-node paths."""
    source = _modal_toggle_source()

    assert "function rootGraph()" in source
    assert "function findSomethingInAllSubgraphs(matcher)" in source
    assert "function findContainingSubgraphNode(subgraphId)" in source
    assert "candidate.subgraph?.id === subgraphId" in source
    assert "function markWorkflowNodePathsRemote(workflowNodePaths)" in source
    assert "setRemoteFlag(node, true);" in source


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
    assert "descendantStates.every((state) => state.phase === STATE_COMPLETE)" in source


def test_modal_ui_refreshes_after_visibility_or_focus_returns() -> None:
    """Background-tab throttling should not leave the status pill stale after refocus."""
    source = _modal_toggle_source()

    assert "function refreshModalUiAfterVisibilityChange()" in source
    assert 'document.addEventListener("visibilitychange"' in source
    assert 'window.addEventListener("focus", refreshModalUiAfterVisibilityChange);' in source
