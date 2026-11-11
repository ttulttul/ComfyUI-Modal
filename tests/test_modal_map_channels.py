"""Tests for zipping several Modal Map Input markers through a Modal Input Channel."""

from __future__ import annotations

import importlib

from api_intercept_test_support import *  # noqa: F401,F403
from conftest import PACKAGE_NAME


@pytest.fixture
def channels_module(extension_package: object) -> Any:
    """Return the Modal Map Input channel helpers module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_map_channels")


class _FakeSeedListNode:
    """Fake upstream node that represents a seed-list producer."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    OUTPUT_IS_LIST = (False,)


class _FakeRemoteTextSeedNode:
    """Fake remote node that consumes one prompt and one seed."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)


class _FakeModalInputChannelNode:
    """Fake channel node whose output only groups markers."""

    RETURN_TYPES = ("MODAL_INPUT_CHANNEL",)
    RETURN_NAMES = ("channel",)
    OUTPUT_IS_LIST = (False,)


def _channel_prompt(*, with_channel: bool = True) -> dict[str, Any]:
    """Return a prompt where two local markers feed one remote node."""
    prompt: dict[str, Any] = {
        "1": {"class_type": "PromptList", "inputs": {}, "_meta": {"title": "Prompts"}},
        "5": {"class_type": "SeedList", "inputs": {}, "_meta": {"title": "Seeds"}},
        "2": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["1", 0]},
            "_meta": {"title": "Map prompts"},
        },
        "6": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["5", 0]},
            "_meta": {"title": "Map seeds"},
        },
        "3": {
            "class_type": "RemoteTextSeed",
            "inputs": {"text": ["2", 0], "seed": ["6", 0]},
            "_meta": {"title": "Remote"},
        },
        "4": {
            "class_type": "LocalStringSink",
            "inputs": {"text": ["3", 0]},
            "_meta": {"title": "Local Sink"},
        },
    }
    if with_channel:
        prompt["7"] = {
            "class_type": "ModalInputChannel",
            "inputs": {"name": " portraits "},
            "_meta": {},
        }
        prompt["2"]["inputs"]["channel"] = ["7", 0]
        prompt["6"]["inputs"]["channel"] = ["7", 0]
    return prompt


def _rewrite(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
    prompt: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Rewrite one prompt with only node 3 marked remote."""
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
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "PromptList": _FakePromptListNode,
                "SeedList": _FakeSeedListNode,
                "ModalMapInput": _FakeModalMapInputNode,
                "ModalInputChannel": _FakeModalInputChannelNode,
                "RemoteTextSeed": _FakeRemoteTextSeedNode,
                "LocalStringSink": _FakeLocalStringSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": int(node_id), "properties": {"is_modal_remote": node_id == "3"}}
            for node_id in prompt
        ]
    }
    return api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine_module.ModalAssetSyncEngine.from_environment(settings),
        settings=settings,
        nodes_module=fake_nodes_module,
    )


def test_resolve_channels_folds_wiring_into_marker_metadata(channels_module: Any) -> None:
    """Channel links and channel nodes disappear; markers remember their channel."""
    prompt = _channel_prompt()

    resolved = channels_module.resolve_modal_map_channels(prompt)

    assert "7" not in resolved
    assert "channel" not in resolved["2"]["inputs"]
    assert channels_module.modal_map_channel(resolved["2"]) == "7"
    assert channels_module.modal_map_channel(resolved["6"]) == "7"
    assert channels_module.modal_map_channel_members(resolved, "7") == ["2", "6"]
    assert prompt["2"]["inputs"]["channel"] == ["7", 0]


def test_resolve_channels_rejects_channel_feeding_other_inputs(channels_module: Any) -> None:
    """A channel may only feed the channel socket of Modal Map Input nodes."""
    prompt = _channel_prompt()
    prompt["4"]["inputs"]["text"] = ["7", 0]

    with pytest.raises(ValueError, match="may only connect"):
        channels_module.resolve_modal_map_channels(prompt)


def test_resolve_channels_rejects_non_channel_source(channels_module: Any) -> None:
    """A marker channel input linked to an ordinary node is a wiring mistake."""
    prompt = _channel_prompt(with_channel=False)
    prompt["2"]["inputs"]["channel"] = ["5", 0]

    with pytest.raises(ValueError, match="must come from a Modal Input Channel"):
        channels_module.resolve_modal_map_channels(prompt)


def test_marker_label_uses_shared_channel_name(channels_module: Any) -> None:
    """Every marker on a channel is labelled with the channel's shared name."""
    resolved = channels_module.resolve_modal_map_channels(_channel_prompt())

    assert channels_module.modal_map_channel_name(resolved["2"]) == "portraits"
    assert channels_module.modal_map_input_label(resolved, "2") == (
        'Modal Map Input 2 on "portraits" (channel 7)'
    )
    assert channels_module.modal_map_input_label(resolved, "6") == (
        'Modal Map Input 6 on "portraits" (channel 7)'
    )


def test_marker_label_without_channel_name(channels_module: Any) -> None:
    """Unnamed channels and unchanneled markers fall back to node ids."""
    prompt = _channel_prompt()
    prompt["7"]["inputs"]["name"] = "   "
    resolved = channels_module.resolve_modal_map_channels(prompt)
    unchanneled = _channel_prompt(with_channel=False)

    assert channels_module.modal_map_input_label(resolved, "2") == (
        "Modal Map Input 2 on channel 7"
    )
    assert channels_module.modal_map_input_label(unchanneled, "2") == "Modal Map Input 2"


def test_rewrite_zips_channel_markers_into_one_mapped_payload(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Two markers on one channel map the same component and both appear as mapped inputs."""
    rewritten_prompt, summary = _rewrite(
        api_intercept_module,
        settings_module,
        sync_engine_module,
        tmp_path,
        _channel_prompt(),
    )

    assert "7" not in rewritten_prompt
    assert summary.remote_component_ids == ["3"]
    payload = rewritten_prompt["3"]["inputs"]["original_node_data"]
    assert payload["payload_kind"] == "mapped_subgraph"
    mapped_inputs = {
        mapped_input["label"]: mapped_input for mapped_input in payload["mapped_inputs"]
    }
    prompts_label = 'Modal Map Input 2 on "portraits" (channel 7)'
    seeds_label = 'Modal Map Input 6 on "portraits" (channel 7)'
    assert set(mapped_inputs) == {prompts_label, seeds_label}
    assert mapped_inputs[prompts_label]["io_type"] == "STRING"
    assert mapped_inputs[seeds_label]["io_type"] == "INT"
    assert {mapped_input["channel_id"] for mapped_input in mapped_inputs.values()} == {
        "7"
    }
    assert payload["mapped_input"]["proxy_input_name"] in {
        mapped_input["proxy_input_name"] for mapped_input in mapped_inputs.values()
    }


def test_rewrite_rejects_two_markers_without_shared_channel(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Two unchanneled markers into one component stay ambiguous and are refused."""
    with pytest.raises(ValueError, match="same Modal Input Channel"):
        _rewrite(
            api_intercept_module,
            settings_module,
            sync_engine_module,
            tmp_path,
            _channel_prompt(with_channel=False),
        )


def test_split_zipped_mapped_inputs_pairs_items_and_shares_singletons(
    serialization_module: Any,
) -> None:
    """Item i gets item i of each list; a one-item input is shared by every item."""
    specs = [
        {"proxy_input_name": "a", "io_type": "STRING", "label": "prompts"},
        {"proxy_input_name": "b", "io_type": "INT", "label": "seeds"},
        {"proxy_input_name": "c", "io_type": "FLOAT", "label": "cfg"},
    ]

    items = serialization_module.split_zipped_mapped_inputs(
        {"a": ["p0", "p1"], "b": [10, 11], "c": 7.5}, specs
    )

    assert items == [
        {"a": "p0", "b": 10, "c": 7.5},
        {"a": "p1", "b": 11, "c": 7.5},
    ]


def test_split_zipped_mapped_inputs_names_mismatched_lists(
    serialization_module: Any,
) -> None:
    """Different list lengths fail with every marker's label and count."""
    specs = [
        {"proxy_input_name": "a", "io_type": "STRING", "label": "prompts"},
        {"proxy_input_name": "b", "io_type": "INT", "label": "seeds"},
    ]

    with pytest.raises(
        serialization_module.MappedInputLengthError,
        match="prompts has 3 items, seeds has 2 items",
    ):
        serialization_module.split_zipped_mapped_inputs(
            {"a": ["p0", "p1", "p2"], "b": [10, 11]}, specs
        )


def test_payload_mapped_input_specs_falls_back_to_single_input(
    serialization_module: Any,
) -> None:
    """Payloads without a channel keep using their single mapped input."""
    single = {"proxy_input_name": "remote_input_0", "io_type": "STRING"}

    assert serialization_module.payload_mapped_input_specs(
        {"mapped_input": single}
    ) == [single]
    assert serialization_module.payload_mapped_input_specs({"mapped_input": None}) == []


def test_implicit_split_names_channel_markers_on_mismatch(
    remote_modal_app_module: Any,
) -> None:
    """Ordered phases report mismatched channel lists with marker labels and targets."""
    payload = {
        "payload_kind": "subgraph",
        "component_id": "141",
        "subgraph_prompt": {
            "147:5": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
            "136": {"class_type": "KSampler", "inputs": {"seed": 0}},
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_3",
                "io_type": "STRING",
                "targets": [{"node_id": "147:5", "input_name": "text"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "136", "input_name": "seed"}],
            },
        ],
        "mapped_inputs": [
            {"proxy_input_name": "remote_input_3", "label": '"prompts" (Modal Map Input 2)'},
            {"proxy_input_name": "remote_input_1", "label": '"seeds" (Modal Map Input 6)'},
        ],
    }

    with pytest.raises(Exception, match=r'"prompts" \(Modal Map Input 2\) -> 147:5.text: 4 items'):
        remote_modal_app_module._split_batch_boundary_inputs(
            payload,
            {"remote_input_3": ["a", "b", "c", "d"], "remote_input_1": [1, 2]},
        )


def test_implicit_split_always_splits_channel_inputs_of_any_type(
    remote_modal_app_module: Any,
) -> None:
    """A channel input list is split even for types implicit batching would broadcast."""
    payload = {
        "payload_kind": "subgraph",
        "component_id": "141",
        "subgraph_prompt": {
            "136": {"class_type": "KSampler", "inputs": {"model": ["9", 0]}},
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "136", "input_name": "model"}],
            }
        ],
        "mapped_inputs": [{"proxy_input_name": "remote_input_0", "label": "models"}],
    }

    split = remote_modal_app_module._split_batch_boundary_inputs(
        payload, {"remote_input_0": ["model-a", "model-b"]}
    )

    assert split == ({"remote_input_0": ["model-a", "model-b"]}, 2)


def test_channel_anchor_nodes_cover_local_and_remote_markers(
    remote_graph_analysis_module: Any,
    channels_module: Any,
) -> None:
    """Remote markers anchor themselves; local markers anchor the remote nodes they feed."""
    prompt = channels_module.resolve_modal_map_channels(_channel_prompt())
    consumers = remote_graph_analysis_module._build_consumer_map(prompt)

    anchors = remote_graph_analysis_module._modal_map_channel_anchor_node_ids(
        prompt, {"3", "6"}, consumers
    )

    assert anchors == {"7": ["3", "6"]}


def test_schemas_put_shared_name_on_channel_node(
    modal_executor_module: Any,
) -> None:
    """The marker gains an optional channel input; the channel node owns the shared name."""
    marker_inputs = {
        item.id: item for item in modal_executor_module.ModalMapInput.define_schema().inputs
    }
    channel_outputs = modal_executor_module.ModalInputChannel.define_schema().outputs

    channel_inputs = {
        item.id: item
        for item in modal_executor_module.ModalInputChannel.define_schema().inputs
    }

    assert set(marker_inputs) == {"value", "channel"}
    assert marker_inputs["channel"].optional
    assert set(channel_inputs) == {"name"}
    assert channel_inputs["name"].optional
    assert channel_outputs[0].io_type == "MODAL_INPUT_CHANNEL"


def test_cloud_mapped_execution_zips_channel_inputs_per_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-container mapped loop hands item i of every channel input to item i."""
    from modal_executor_test_support import _cloud_mapped_execution_owner

    owner = _cloud_mapped_execution_owner()
    observed_inputs: list[dict[str, Any]] = []

    def fake_execute_prompt_subgraph(
        payload: dict[str, Any], hydrated_inputs: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> tuple[str, ...]:
        """Record each item's inputs and echo them as its output."""
        observed_inputs.append(dict(hydrated_inputs))
        return (f"{hydrated_inputs['remote_input_0']}@{hydrated_inputs['remote_input_1']}",)

    monkeypatch.setattr(owner, "_execute_prompt_subgraph", fake_execute_prompt_subgraph)
    boundary_output = {
        "proxy_output_name": "3_text",
        "node_id": "3",
        "output_index": 0,
        "io_type": "STRING",
        "is_list": False,
        "mapped_output": True,
        "scheduler_is_list": True,
    }
    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "3",
        "mapped_input": {"proxy_input_name": "remote_input_0", "io_type": "STRING"},
        "mapped_inputs": [
            {"proxy_input_name": "remote_input_0", "io_type": "STRING", "label": "prompts"},
            {"proxy_input_name": "remote_input_1", "io_type": "INT", "label": "seeds"},
        ],
        "static_to_mapped_boundaries": [],
        "mapped_phase": {
            "component_node_ids": ["3"],
            "subgraph_prompt": {"3": {"class_type": "RemoteTextSeed", "inputs": {}}},
            "boundary_inputs": [],
            "boundary_outputs": [boundary_output],
            "execute_node_ids": ["3"],
        },
        "boundary_outputs": [boundary_output],
    }

    outputs = owner._execute_mapped_subgraph_payload(
        payload,
        {"remote_input_0": ["cat", "dog"], "remote_input_1": [1, 2], "shared": "vae"},
        None,
    )

    assert [(item["remote_input_0"], item["remote_input_1"]) for item in observed_inputs] == [
        ("cat", 1),
        ("dog", 2),
    ]
    assert all(item["shared"] == "vae" for item in observed_inputs)
    assert outputs == (["cat@1", "dog@2"],)
