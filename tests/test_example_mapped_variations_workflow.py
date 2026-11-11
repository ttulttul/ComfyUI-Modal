"""Tests for the checked-in mapped-variations example workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "Mapped Variations Progressive Preview.json"
)
SEED_COUNT = 4
PROMPT_COUNT = 3
ITEM_COUNT = SEED_COUNT * PROMPT_COUNT


@pytest.fixture(scope="module")
def example_workflow() -> dict[str, Any]:
    """Return the example workflow as saved by the ComfyUI frontend."""
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def _nodes_by_id(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the workflow's nodes keyed by string node id."""
    return {str(node["id"]): node for node in workflow["nodes"]}


def _node_ids_of_type(workflow: dict[str, Any], node_type: str) -> list[str]:
    """Return the ids of every workflow node of one type, in id order."""
    return sorted(
        (str(node["id"]) for node in workflow["nodes"] if node["type"] == node_type),
        key=int,
    )


def _workflow_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the API prompt links of a saved workflow; widget values are not needed."""
    origins = {link[0]: [str(link[1]), int(link[2])] for link in workflow["links"]}
    prompt: dict[str, Any] = {}
    for node in workflow["nodes"]:
        inputs = {
            node_input["name"]: origins[node_input["link"]]
            for node_input in node.get("inputs", [])
            if node_input.get("link") is not None
        }
        if node["type"] == "ModalInputChannel":
            inputs["name"] = node["widgets_values"][0]
        prompt[str(node["id"])] = {
            "class_type": node["type"],
            "inputs": inputs,
            "_meta": {"title": node.get("title", node["type"])},
        }
    return prompt


def _fake_node_class(node: dict[str, Any]) -> type:
    """Return a stand-in node class that declares one workflow node's outputs."""
    outputs = node.get("outputs", [])
    return type(
        f"_Fake{node['type']}",
        (),
        {
            "RETURN_TYPES": tuple(str(output["type"]) for output in outputs),
            "RETURN_NAMES": tuple(str(output["name"]) for output in outputs),
            "OUTPUT_IS_LIST": tuple(node["type"] == "CreateList" for _ in outputs),
        },
    )


def _fake_nodes_module(workflow: dict[str, Any]) -> Any:
    """Return a nodes module holding a stand-in class for every workflow node type."""
    mappings = {node["type"]: _fake_node_class(node) for node in workflow["nodes"]}
    return type(
        "FakeNodesModule",
        (),
        {"NODE_CLASS_MAPPINGS": mappings, "NODE_DISPLAY_NAME_MAPPINGS": {}},
    )()


def _expand_list(prompt: dict[str, Any], node_id: str) -> list[str]:
    """Return the source node ids a chain of Create List nodes emits, in order."""
    if prompt[node_id]["class_type"] != "CreateList":
        return [node_id]
    items: list[str] = []
    for _, origin in sorted(prompt[node_id]["inputs"].items()):
        items.extend(_expand_list(prompt, origin[0]))
    return items


def test_example_marks_the_sampler_cascade_remote_and_previews_local(
    example_workflow: dict[str, Any],
) -> None:
    """Loaders, encoders, both samplers and both decodes run remotely; the rest stays local."""
    nodes = _nodes_by_id(example_workflow)
    remote_types = sorted(
        node["type"] for node in nodes.values() if node["properties"].get("is_modal_remote")
    )

    assert remote_types == [
        "CLIPLoader",
        "CLIPTextEncode",
        "CLIPTextEncode",
        "EmptySD3LatentImage",
        "KSampler",
        "KSampler",
        "LatentUpscaleBy",
        "UNETLoader",
        "VAEDecode",
        "VAEDecode",
        "VAELoader",
    ]
    assert len(_node_ids_of_type(example_workflow, "RemoteProgressivePreview")) == 2


def test_example_previews_read_each_sampler_stage(example_workflow: dict[str, Any]) -> None:
    """Each progressive preview is wired straight to the VAE Decode of one sampler stage."""
    prompt = _workflow_prompt(example_workflow)
    sampler_one, sampler_two = _node_ids_of_type(example_workflow, "KSampler")
    previewed_samplers = []
    for preview_id in _node_ids_of_type(example_workflow, "RemoteProgressivePreview"):
        decode_id = prompt[preview_id]["inputs"]["images"][0]
        assert prompt[decode_id]["class_type"] == "VAEDecode"
        previewed_samplers.append(prompt[decode_id]["inputs"]["samples"][0])

    assert previewed_samplers == [sampler_one, sampler_two]
    upscale_id = prompt[sampler_two]["inputs"]["latent_image"][0]
    assert prompt[upscale_id]["class_type"] == "LatentUpscaleBy"
    assert prompt[upscale_id]["inputs"]["samples"][0] == sampler_one


def test_example_lists_pair_every_seed_with_every_prompt(
    example_workflow: dict[str, Any],
) -> None:
    """Repeating 4 seeds 3 times and 3 prompts 4 times zips into all 12 combinations."""
    prompt = _workflow_prompt(example_workflow)
    seed_marker, prompt_marker = _node_ids_of_type(example_workflow, "ModalMapInput")
    seeds = _expand_list(prompt, prompt[seed_marker]["inputs"]["value"][0])
    prompts = _expand_list(prompt, prompt[prompt_marker]["inputs"]["value"][0])

    assert len(seeds) == len(prompts) == ITEM_COUNT
    assert len(set(seeds)) == SEED_COUNT
    assert len(set(prompts)) == PROMPT_COUNT
    assert len(set(zip(seeds, prompts, strict=True))) == ITEM_COUNT
    assert {prompt[node_id]["class_type"] for node_id in seeds} == {"PrimitiveInt"}
    assert {prompt[node_id]["class_type"] for node_id in prompts} == {
        "PrimitiveStringMultiline"
    }


def test_example_rewrites_into_one_zipped_mapped_component(
    example_workflow: dict[str, Any],
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """The planner accepts the example and zips both markers through the named channel."""
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    channel_id = _node_ids_of_type(example_workflow, "ModalInputChannel")[0]
    seed_marker, prompt_marker = _node_ids_of_type(example_workflow, "ModalMapInput")

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=_workflow_prompt(example_workflow),
        workflow=example_workflow,
        sync_engine=sync_engine_module.ModalAssetSyncEngine.from_environment(settings),
        settings=settings,
        nodes_module=_fake_nodes_module(example_workflow),
    )

    assert channel_id not in rewritten_prompt
    mapped_labels = {
        mapped_input["label"]
        for component_id in summary.remote_component_ids
        for mapped_input in rewritten_prompt[component_id]["inputs"][
            "original_node_data"
        ].get("mapped_inputs", [])
    }
    assert mapped_labels == {
        f'Modal Map Input {seed_marker} on "variations" (channel {channel_id})',
        f'Modal Map Input {prompt_marker} on "variations" (channel {channel_id})',
    }
    for preview_id in _node_ids_of_type(example_workflow, "RemoteProgressivePreview"):
        source_id = rewritten_prompt[preview_id]["inputs"]["images"][0]
        assert rewritten_prompt[source_id]["class_type"].startswith("ModalUniversalExecutor")
