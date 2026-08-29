"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_modal_proxy_waits_for_active_prompt_before_starting_next_prompt(
    modal_executor_module: Any,
) -> None:
    """A Modal proxy from a later prompt should wait until the active prompt's remote work drains."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("STRING",),
        output_names=("value",),
        output_is_list=(False,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    observed_events: list[str] = []

    class FakeClient:
        """Fake async remote client that records when each prompt reaches execution."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str]:
            """Block the first prompt so the second prompt has to wait at the gate."""
            component_id = str(payload["component_id"])
            observed_events.append(f"start:{component_id}")
            if component_id == "component-1":
                first_started.set()
                await release_first.wait()
            if component_id == "component-2":
                second_started.set()
            observed_events.append(f"finish:{component_id}")
            return (component_id,)

    async def run_scenario() -> tuple[Any, Any]:
        """Run two prompts concurrently and verify prompt-level serialization."""
        first_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-1",
                    "component_id": "component-1",
                },
                unique_id="component-1",
            )
        )
        await first_started.wait()
        second_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-2",
                    "component_id": "component-2",
                },
                unique_id="component-2",
            )
        )
        await asyncio.sleep(0.05)
        assert not second_started.is_set()
        release_first.set()
        return await asyncio.gather(first_task, second_task)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        first_output, second_output = asyncio.run(run_scenario())
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert first_output.result == ("component-1",)
    assert second_output.result == ("component-2",)
    assert observed_events == [
        "start:component-1",
        "finish:component-1",
        "start:component-2",
        "finish:component-2",
    ]

def test_modal_proxy_allows_same_prompt_components_to_overlap(
    modal_executor_module: Any,
) -> None:
    """Modal proxies from the same prompt should be allowed to execute concurrently."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("STRING",),
        output_names=("value",),
        output_is_list=(False,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    started_components: set[str] = set()
    both_started = asyncio.Event()
    release_components = asyncio.Event()
    observed_events: list[str] = []

    class FakeClient:
        """Fake async remote client that waits until both same-prompt components start."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str]:
            """Record concurrent same-prompt starts before returning."""
            component_id = str(payload["component_id"])
            observed_events.append(f"start:{component_id}")
            started_components.add(component_id)
            if started_components == {"component-1", "component-2"}:
                both_started.set()
            await release_components.wait()
            observed_events.append(f"finish:{component_id}")
            return (component_id,)

    async def run_scenario() -> tuple[Any, Any, Any]:
        """Run two same-prompt proxy components concurrently."""
        first_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-1",
                    "component_id": "component-1",
                },
                unique_id="component-1",
            )
        )
        second_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-1",
                    "component_id": "component-2",
                },
                unique_id="component-2",
            )
        )
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        release_components.set()
        return await asyncio.gather(first_task, second_task)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        first_output, second_output = asyncio.run(run_scenario())
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert first_output.result == ("component-1",)
    assert second_output.result == ("component-2",)
    assert observed_events[:2] == ["start:component-1", "start:component-2"]

def test_modal_proxy_cancellation_while_waiting_does_not_leak_workflow_gate(
    modal_executor_module: Any,
) -> None:
    """Cancelling a proxy before it gets the prompt gate should not block later prompts."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("STRING",),
        output_names=("value",),
        output_is_list=(False,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    observed_events: list[str] = []

    class FakeClient:
        """Fake async remote client that lets one prompt hold the workflow gate."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str]:
            """Record remote dispatch and optionally block the first prompt."""
            component_id = str(payload["component_id"])
            observed_events.append(f"start:{component_id}")
            if component_id == "component-1":
                first_started.set()
                await release_first.wait()
            observed_events.append(f"finish:{component_id}")
            return (component_id,)

    async def run_scenario() -> tuple[Any, Any]:
        """Cancel a waiting prompt, then prove the next prompt can still dispatch."""
        first_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-1",
                    "component_id": "component-1",
                },
                unique_id="component-1",
            )
        )
        await first_started.wait()
        waiting_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-2",
                    "component_id": "component-2",
                },
                unique_id="component-2",
            )
        )
        await asyncio.sleep(0.05)
        waiting_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting_task
        release_first.set()
        first_output = await first_task
        third_output = await asyncio.wait_for(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-3",
                    "component_id": "component-3",
                },
                unique_id="component-3",
            ),
            timeout=1.0,
        )
        return first_output, third_output

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        first_output, third_output = asyncio.run(run_scenario())
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert first_output.result == ("component-1",)
    assert third_output.result == ("component-3",)
    assert observed_events == [
        "start:component-1",
        "finish:component-1",
        "start:component-3",
        "finish:component-3",
    ]

def test_abandoned_modal_prompt_gate_unblocks_next_prompt(
    modal_executor_module: Any,
) -> None:
    """Cancelling an active Modal prompt should not leave later prompts stuck behind its gate."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("STRING",),
        output_names=("value",),
        output_is_list=(False,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    started_first_components: set[str] = set()
    first_components_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first_components = asyncio.Event()
    observed_events: list[str] = []

    class FakeClient:
        """Fake async remote client that lets cancelled prompt components stay stuck."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str]:
            """Record remote dispatch while prompt one waits for cleanup."""
            del kwargs
            component_id = str(payload["component_id"])
            observed_events.append(f"start:{component_id}")
            if component_id in {"component-1a", "component-1b"}:
                started_first_components.add(component_id)
                if started_first_components == {"component-1a", "component-1b"}:
                    first_components_started.set()
                await release_first_components.wait()
            if component_id == "component-2":
                second_started.set()
            observed_events.append(f"finish:{component_id}")
            return (component_id,)

    async def run_scenario() -> tuple[Any, Any]:
        """Abandon prompt one, then prove prompt two can start before prompt one exits."""
        first_task_a = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-1",
                    "component_id": "component-1a",
                },
                unique_id="component-1a",
            )
        )
        first_task_b = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-1",
                    "component_id": "component-1b",
                },
                unique_id="component-1b",
            )
        )
        await first_components_started.wait()

        second_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "prompt-2",
                    "component_id": "component-2",
                },
                unique_id="component-2",
            )
        )
        await asyncio.sleep(0.05)
        assert not second_started.is_set()

        modal_executor_module.abandon_modal_workflow_execution_prompt(
            "prompt-1",
            "test cancellation",
        )
        await asyncio.wait_for(second_started.wait(), timeout=1.0)
        second_output = await second_task

        release_first_components.set()
        first_output_a, first_output_b = await asyncio.gather(first_task_a, first_task_b)
        return first_output_a, first_output_b, second_output

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        first_output_a, first_output_b, second_output = asyncio.run(run_scenario())
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert first_output_a.result == ("component-1a",)
    assert first_output_b.result == ("component-1b",)
    assert second_output.result == ("component-2",)
    assert set(observed_events[:2]) == {"start:component-1a", "start:component-1b"}
    assert observed_events[2:4] == ["start:component-2", "finish:component-2"]
    assert set(observed_events[4:]) == {"finish:component-1a", "finish:component-1b"}

@pytest.mark.parametrize(
    ("registration_function_name", "node_class_name", "node_id_constant_name"),
    [
        (
            "ensure_modal_parallel_local_passthrough_registered",
            "ModalParallelLocalPassthrough",
            "MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID",
        ),
        (
            "ensure_modal_local_bridge_materializer_registered",
            "ModalLocalBridgeMaterializer",
            "MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID",
        ),
        (
            "ensure_modal_artifact_finalizer_registered",
            "ModalArtifactFinalizer",
            "MODAL_ARTIFACT_FINALIZER_NODE_ID",
        ),
    ],
)
def test_internal_node_registration_sets_module_identity(
    modal_executor_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    registration_function_name: str,
    node_class_name: str,
    node_id_constant_name: str,
) -> None:
    """Static internal nodes should serialize the startup node's module identity."""
    expected_module = "custom_nodes.ComfyUI-Modal"
    monkeypatch.setattr(
        modal_executor_module.ModalUniversalExecutor,
        "RELATIVE_PYTHON_MODULE",
        expected_module,
    )
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    registration_function = getattr(modal_executor_module, registration_function_name)
    node_class = getattr(modal_executor_module, node_class_name)
    monkeypatch.setattr(
        node_class,
        "RELATIVE_PYTHON_MODULE",
        node_class.RELATIVE_PYTHON_MODULE,
    )
    registration_function(fake_nodes_module)

    node_id = getattr(modal_executor_module, node_id_constant_name)
    assert fake_nodes_module.NODE_CLASS_MAPPINGS[node_id] is node_class
    _assert_node_module_identity(node_class, expected_module)

def test_dynamic_proxy_node_preserves_output_signature(
    modal_executor_module: Any,
) -> None:
    """Dynamic Modal proxies should mirror the original output count and names."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )

    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]
    schema = proxy_class.GET_SCHEMA()

    assert schema.node_id == proxy_id
    assert [output.display_name for output in schema.outputs] == ["image", "count"]
    assert [output.io_type for output in schema.outputs] == ["IMAGE", "INT"]
    assert proxy_class.INPUT_IS_LIST is True

def test_scheduler_list_proxy_wraps_singleton_image_tensor(
    modal_executor_module: Any,
) -> None:
    """A singleton mapped IMAGE must retain its batch dimension in ComfyUI."""
    torch = pytest.importorskip("torch")

    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("IMAGE",),
        output_names=("image",),
        output_is_list=(True,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    image = torch.zeros((1, 8, 6, 3), dtype=torch.float32)

    class FakeClient:
        """Return one ordinary IMAGE tensor from a singleton remote execution."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[Any, ...]:
            """Return the unwrapped tensor produced by the ordinary subgraph path."""
            assert payload["boundary_outputs"][0]["scheduler_is_list"] is True
            assert kwargs == {}
            return (image,)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = asyncio.run(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "singleton-image-prompt",
                    "component_id": "singleton-image-component",
                    "boundary_outputs": [
                        {
                            "proxy_output_name": "image",
                            "node_id": "11",
                            "output_index": 0,
                            "io_type": "IMAGE",
                            "is_list": False,
                            "scheduler_is_list": True,
                        }
                    ],
                },
                unique_id="singleton-image-component",
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert isinstance(result.result[0], list)
    assert len(result.result[0]) == 1
    assert result.result[0][0] is image
    scheduler_items: list[Any] = []
    scheduler_items.extend(result.result[0])
    assert len(scheduler_items) == 1
    assert scheduler_items[0] is image
    assert tuple(scheduler_items[0].shape) == (1, 8, 6, 3)

def test_transportable_singleton_list_reaches_scalar_remote_consumer_as_item(
    modal_executor_module: Any,
) -> None:
    """A cached singleton list output should enter the next remote proxy as a scalar."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    producer_proxy_id = (
        modal_executor_module.ensure_modal_component_proxy_node_registered(
            output_types=("INT",),
            output_names=("seed",),
            output_is_list=(True,),
            nodes_module=fake_nodes_module,
            is_output_node=False,
        )
    )
    consumer_proxy_id = (
        modal_executor_module.ensure_modal_component_proxy_node_registered(
            output_types=("STRING",),
            output_names=("result",),
            output_is_list=(False,),
            nodes_module=fake_nodes_module,
            is_output_node=False,
        )
    )
    producer_proxy = fake_nodes_module.NODE_CLASS_MAPPINGS[producer_proxy_id]
    consumer_proxy = fake_nodes_module.NODE_CLASS_MAPPINGS[consumer_proxy_id]
    observed_consumer_inputs: list[Any] = []

    class FakeClient:
        """Return a singleton seed list and record the downstream normalized input."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[Any, ...]:
            """Model the two remote components around ComfyUI's output cache."""
            if payload["component_id"] == "producer":
                assert kwargs == {}
                return ([123],)
            observed_consumer_inputs.append(kwargs["remote_input_0"])
            return (f"seed:{int(kwargs['remote_input_0'])}",)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        producer_result = asyncio.run(
            producer_proxy.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "component_id": "producer",
                    "boundary_outputs": [
                        {
                            "proxy_output_name": "seed",
                            "node_id": "325",
                            "output_index": 0,
                            "io_type": "INT",
                            "is_list": True,
                        }
                    ],
                },
                unique_id="producer",
            )
        )
        cached_scheduler_items: list[Any] = []
        cached_scheduler_items.extend(producer_result.result[0])
        consumer_result = asyncio.run(
            consumer_proxy.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "component_id": "consumer",
                    "boundary_outputs": [
                        {
                            "proxy_output_name": "result",
                            "node_id": "372",
                            "output_index": 0,
                            "io_type": "STRING",
                            "is_list": False,
                        }
                    ],
                },
                remote_input_0=cached_scheduler_items,
                unique_id="consumer",
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert cached_scheduler_items == [123]
    assert observed_consumer_inputs == [123]
    assert consumer_result.result == ("seed:123",)

def test_component_proxy_emits_completion_token_after_remote_execution(
    modal_executor_module: Any,
) -> None:
    """Artifact-only component proxies should expose completion after remote work returns."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=(),
        output_names=(),
        output_is_list=(),
        nodes_module=fake_nodes_module,
        is_output_node=False,
        include_completion_output=True,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    schema = proxy_class.GET_SCHEMA()

    assert [output.io_type for output in schema.outputs] == ["BOOLEAN"]
    assert [output.display_name for output in schema.outputs] == [
        modal_executor_module.MODAL_COMPONENT_COMPLETION_OUTPUT_NAME
    ]

    class FakeClient:
        """Return no remote boundary values for an artifact-only component."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[()]:
            """Confirm the remote component ran before returning no boundary outputs."""
            assert payload["component_id"] == "artifact-component"
            assert kwargs == {}
            return ()

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = asyncio.run(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": "artifact-prompt",
                    "component_id": "artifact-component",
                },
                unique_id="artifact-component",
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == (True,)

def test_modal_artifact_finalizer_requires_completed_components(
    modal_executor_module: Any,
) -> None:
    """The internal output sink should reject any missing completion signal."""
    schema = modal_executor_module.ModalArtifactFinalizer.GET_SCHEMA()

    assert schema.is_output_node is True
    assert schema.outputs == []
    assert modal_executor_module.ModalArtifactFinalizer.execute(
        {"component_0": True, "component_1": True}
    ).result is None
    with pytest.raises(RuntimeError, match="component_1"):
        modal_executor_module.ModalArtifactFinalizer.execute(
            {"component_0": True, "component_1": False}
        )

def test_modal_parallel_local_passthrough_releases_after_remote_dispatch(
    modal_executor_module: Any,
) -> None:
    """A local preview should unblock when its remote continuation starts."""
    schema = modal_executor_module.ModalParallelLocalPassthrough.GET_SCHEMA()
    dispatch_context = {
        "dispatch_group_id": "prompt-parallel",
        "component_ids": ["component-b"],
    }

    async def run_scenario() -> tuple[Any, ...]:
        """Prove the passthrough remains pending until remote dispatch is signalled."""
        passthrough_task = asyncio.create_task(
            modal_executor_module.ModalParallelLocalPassthrough.execute(
                "image-value",
                dispatch_context,
            )
        )
        await asyncio.sleep(0)
        assert passthrough_task.done() is False
        assert modal_executor_module._signal_parallel_local_dispatch(
            {
                "parallel_local_dispatch_group_id": "prompt-parallel",
                "component_id": "component-b",
                "signal_parallel_local_dispatch": True,
            }
        ) is True
        result = await asyncio.wait_for(passthrough_task, timeout=1.0)
        return result.result

    try:
        result = asyncio.run(run_scenario())
    finally:
        modal_executor_module._clear_parallel_dispatch_events("prompt-parallel")

    assert schema.is_output_node is False
    assert result == ("image-value",)

def test_modal_parallel_local_passthrough_runs_before_remote_result_returns(
    modal_executor_module: Any,
) -> None:
    """A local preview should run while its downstream remote call is in flight."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    proxy_node_id = modal_executor_module.ensure_modal_component_proxy_node_registered(
        output_types=("STRING",),
        output_names=("value",),
        output_is_list=(False,),
        nodes_module=fake_nodes_module,
        is_output_node=False,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_node_id]
    remote_started = asyncio.Event()
    release_remote = asyncio.Event()

    class FakeClient:
        """Fake async remote client that remains in flight until explicitly released."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str]:
            """Record dispatch, then keep the remote result pending."""
            remote_started.set()
            await release_remote.wait()
            return (str(payload["component_id"]),)

    async def run_scenario() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Prove local passthrough completion precedes the remote result."""
        dispatch_group_id = "prompt-parallel-in-flight"
        local_task = asyncio.create_task(
            modal_executor_module.ModalParallelLocalPassthrough.execute(
                "preview-image",
                {
                    "dispatch_group_id": dispatch_group_id,
                    "component_ids": ["component-b"],
                },
            )
        )
        remote_task = asyncio.create_task(
            proxy_class.execute(
                original_node_data={
                    "payload_kind": "subgraph",
                    "prompt_id": dispatch_group_id,
                    "component_id": "component-b",
                    "parallel_local_dispatch_group_id": dispatch_group_id,
                    "signal_parallel_local_dispatch": True,
                },
                unique_id="component-b",
            )
        )
        await asyncio.wait_for(remote_started.wait(), timeout=1.0)
        local_result = await asyncio.wait_for(local_task, timeout=1.0)
        assert remote_task.done() is False
        release_remote.set()
        remote_result = await asyncio.wait_for(remote_task, timeout=1.0)
        return local_result.result, remote_result.result

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        local_result, remote_result = asyncio.run(run_scenario())
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)
        modal_executor_module._clear_parallel_dispatch_events(
            "prompt-parallel-in-flight"
        )

    assert local_result == ("preview-image",)
    assert remote_result == ("component-b",)

def test_parallel_local_dispatch_frontier_stops_at_nearest_remote_component(
    api_intercept_module: Any,
    prompt_affinity_planning_module: Any,
    remote_graph_analysis_module: Any,
) -> None:
    """A preview gate should not wait for remote descendants of a remote frontier."""
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteProxy": _FakeRewriteRemoteSamplerNode,
            "LocalFeedback": _FakeRewriteLocalFeedbackNode,
        },
    )
    rewritten_prompt = {
        "source": {"class_type": "RemoteProxy", "inputs": {}},
        "feedback": {
            "class_type": "LocalFeedback",
            "inputs": {"value": ["source", 0]},
        },
        "nearest": {
            "class_type": "RemoteProxy",
            "inputs": {"value": ["feedback", 0]},
        },
        "later": {
            "class_type": "RemoteProxy",
            "inputs": {"value": ["nearest", 0]},
        },
    }
    consumers = remote_graph_analysis_module._build_consumer_map(rewritten_prompt)

    nearest_component_ids = (
        prompt_affinity_planning_module._nearest_downstream_remote_component_ids(
            rewritten_prompt=rewritten_prompt,
            consumers=consumers,
            seed_targets=consumers[
                api_intercept_module.LinkedOutputRef("source", 0)
            ],
            remote_component_id_set={"source", "nearest", "later"},
            nodes_module=fake_nodes_module,
        )
    )

    assert nearest_component_ids == ["nearest"]

def test_proxy_execution_uses_injected_remote_client(
    modal_executor_module: Any,
) -> None:
    """Proxy execution should delegate to the configured remote client asynchronously."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]

    class FakeClient:
        """Test client that returns deterministic outputs."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str, int]:
            """Return values derived from the proxied node payload."""
            return (f"{payload['class_type']}::{kwargs['value']}", 3)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = asyncio.run(
            proxy_class.execute(original_node_data={"class_type": "OriginalNode"}, value="payload")
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("OriginalNode::payload", 3)

def test_proxy_execution_normalizes_input_is_list_kwargs(
    modal_executor_module: Any,
) -> None:
    """Dynamic Modal proxies should unwrap singleton INPUT_IS_LIST wrappers before remote execution."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]
    observed_kwargs: dict[str, Any] = {}

    class FakeClient:
        """Test client that records normalized proxy kwargs."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str, int]:
            """Capture the kwargs forwarded by the proxy."""
            observed_kwargs["payload_kind"] = payload["payload_kind"]
            observed_kwargs.update(kwargs)
            return ("ok", 1)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = asyncio.run(
            proxy_class.execute(
                original_node_data=[{"payload_kind": "subgraph", "class_type": "OriginalNode"}],
                scalar_value=[3],
                mapped_value=["a", "b", "c"],
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("ok", 1)
    assert observed_kwargs == {
        "payload_kind": "subgraph",
        "scalar_value": 3,
        "mapped_value": ["a", "b", "c"],
    }

def test_proxy_execution_wraps_sync_remote_clients(
    modal_executor_module: Any,
) -> None:
    """Async proxy execution should still support legacy sync remote clients."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]

    class FakeSyncClient:
        """Legacy client that only exposes the blocking execution method."""

        def execute_payload(self, payload: dict[str, Any], kwargs: dict[str, Any]) -> tuple[str, int]:
            """Return values derived from the proxied node payload."""
            return (f"sync::{payload['class_type']}::{kwargs['value']}", 4)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeSyncClient())
    try:
        result = asyncio.run(
            proxy_class.execute(original_node_data={"class_type": "OriginalNode"}, value="payload")
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("sync::OriginalNode::payload", 4)

def test_local_remote_app_executes_original_node(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback remote app should execute a mapped legacy node."""
    payload = remote_modal_app_module.execute_node_locally(
        node_data={"class_type": "OriginalNode"},
        kwargs_payload='{"value": "hello"}',
        node_mapping={"OriginalNode": _FakeOriginalNode},
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == ("hello", 1)

def test_local_gap_keepalive_is_bounded_and_stopped_by_next_remote_component(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """A completed producer should retain its slot only until downstream remote work starts."""
    submitted_tasks: list[tuple[Any, tuple[Any, ...], Future[Any]]] = []

    class FakeExecutor:
        """Executor double that captures but does not run the keepalive loop."""

        def submit(self, fn: Any, *args: Any) -> Future[Any]:
            """Record one keepalive job and return its pending future."""
            future: Future[Any] = Future()
            submitted_tasks.append((fn, args, future))
            return future

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(modal_warmup_module, "modal", object())
    monkeypatch.setattr(
        modal_warmup_module,
        "_REMOTE_MODAL_KEEPALIVE_EXECUTOR",
        FakeExecutor(),
    )
    remote_modal_app_module.get_settings.cache_clear()
    with modal_warmup_module._LOCAL_GAP_KEEPALIVES_LOCK:
        modal_warmup_module._LOCAL_GAP_KEEPALIVES.clear()
        producer_payload = {
            "prompt_id": "prompt-gap",
            "component_id": "component-a",
            "execution_provider": "modal",
            "execution_environment_id": "modal:RTX-PRO-6000",
            "keepalive_after_remote_component": True,
        }
        consumer_payload = {
            "prompt_id": "prompt-gap",
            "component_id": "component-b",
            "execution_provider": "modal",
            "execution_environment_id": "modal:RTX-PRO-6000",
            "stop_local_gap_keepalive_before_remote_component": True,
    }

    try:
        assert modal_warmup_module._start_local_gap_keepalive(producer_payload) is True
        assert len(submitted_tasks) == 1
        _fn, args, _future = submitted_tasks[0]
        stop_event = args[1]
        assert args[2:] == (900.0, 15.0)
        assert stop_event.is_set() is False

        assert modal_warmup_module._stop_local_gap_keepalive(
            consumer_payload,
            reason="test_next_component",
        ) is True
        assert stop_event.is_set() is True
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        with modal_warmup_module._LOCAL_GAP_KEEPALIVES_LOCK:
            modal_warmup_module._LOCAL_GAP_KEEPALIVES.clear()

def test_register_exact_component_parallelism_refines_prompt_target(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped fan-out should raise the prompt-wide warmup target once exact item count is known."""
    monkeypatch.setenv("COMFY_MODAL_MAX_CONTAINERS", "6")
    remote_modal_app_module.get_settings.cache_clear()
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()
        modal_warmup_module._PROMPT_WARMUP_STATE_ORDER = None

    try:
        payload = {
            "prompt_id": "prompt-2",
            "component_id": "component-a",
            "extra_data": {
                "modal": {
                    "component_execution_stages": [["component-a", "component-b"], ["component-c"]],
                    "mapped_component_ids": ["component-a"],
                    "estimated_max_parallel_requests": 2,
                }
            },
        }
        refined_target = modal_warmup_module._register_exact_component_parallelism(payload, 5)
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert refined_target == 6

def test_workflow_gpu_changes_expected_remote_runtime_fingerprint(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
) -> None:
    """Each workflow GPU target should select a distinct deploy-time runtime identity."""
    a100_fingerprint = modal_deployment_module._expected_remote_runtime_fingerprint(
        {"modal_gpu": "A100"}
    )
    b300_fingerprint = modal_deployment_module._expected_remote_runtime_fingerprint(
        {"modal_gpu": "B300"}
    )

    assert a100_fingerprint != b300_fingerprint
    assert modal_deployment_module._settings_for_payload(
        {"modal_gpu": "L40S"}
    ).modal_gpu == "L40S"
    configured_settings = modal_deployment_module._settings_for_payload(
        {"modal_gpu": "H200", "modal_max_containers": 3}
    )
    assert configured_settings.modal_gpu == "H200"
    assert configured_settings.max_containers == 3
    assert modal_deployment_module._modal_deploy_cache_key(
        {"modal_gpu": "A100"}
    )[0] == "comfy-modal-sync"
    assert modal_deployment_module._modal_deploy_cache_key(
        {"modal_gpu": "B300"}
    )[0] == "comfy-modal-sync-gpu-b300"

def test_local_gap_components_share_one_affinity_slot(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
) -> None:
    """Sequential remote phases around local work must address the same worker pool."""
    parallelism_metadata = {
        "modal": {
            "component_execution_stages": [
                ["component-a", "independent-component"],
                ["component-b"],
            ],
            "estimated_max_parallel_requests": 2,
        }
    }
    first_payload = {
        "component_id": "component-a",
        "remote_local_gap_pool": True,
        "extra_data": parallelism_metadata,
    }
    second_payload = {
        "component_id": "component-b",
        "remote_local_gap_pool": True,
        "extra_data": parallelism_metadata,
    }

    assert modal_deployment_module._component_pool_slot_index(first_payload) == 0
    assert modal_deployment_module._component_pool_slot_index(second_payload) == 0
    assert modal_deployment_module._remote_worker_affinity_key(first_payload) == (
        modal_deployment_module._remote_worker_affinity_key(second_payload)
    )

def test_emit_local_remote_dispatch_status_marks_component_starting(
    remote_modal_app_module: Any,
    local_ui_events_module: Any,
    monkeypatch: Any,
) -> None:
    """Dispatching a remote component should immediately tell the local UI it is starting."""

    class FakePromptServer:
        """Capture websocket events emitted by local dispatch status."""

        def __init__(self) -> None:
            """Initialize the event sink."""
            self.messages: list[tuple[str, dict[str, Any], str | None]] = []

        def send_sync(self, event: str, data: dict[str, Any], sid: str | None) -> None:
            """Record one emitted websocket message."""
            self.messages.append((event, data, sid))

    prompt_server = FakePromptServer()
    monkeypatch.setattr(local_ui_events_module, "_lookup_local_prompt_server", lambda: prompt_server)

    remote_modal_app_module._emit_local_remote_dispatch_status(
        {
            "prompt_id": "prompt-1",
            "component_id": "component-1",
            "component_node_ids": ["7", "8"],
            "modal_gpu": "B300",
            "extra_data": {"client_id": "client-1"},
        }
    )

    assert prompt_server.messages == [
        (
            "modal_status",
            {
                "phase": "starting",
                "prompt_id": "prompt-1",
                "node_ids": ["7", "8"],
                "modal_gpu": "B300",
                    "status_message": "Starting remote component on Modal",
            },
            "client-1",
        )
    ]

def test_modal_gpu_estimated_rates_cover_supported_aliases(
    modal_container_logs_module: Any,
) -> None:
    """Published GPU estimates should cover every selectable billed GPU identity."""
    rates = modal_container_logs_module.MODAL_GPU_ESTIMATED_USD_PER_SECOND

    assert set(rates) == set(modal_container_logs_module.MODAL_GPU_TYPES)
    assert rates["A100"] == rates["A100-40GB"]
    assert rates["H100!"] == rates["H100"]
    assert rates["B200+"] == rates["B200"]
    assert modal_container_logs_module.MODAL_GPU_PRICING_EFFECTIVE_DATE == "2026-08-13"

def test_request_remote_interrupt_async_uses_shared_control_async_put(
    remote_modal_app_module: Any,
    modal_interrupts_module: Any,
    monkeypatch: Any,
) -> None:
    """Async local cancellation should use Modal Dict.put.aio instead of blocking put."""

    class FakePut:
        """Modal Dict put double that records sync and async writes separately."""

        def __init__(self) -> None:
            """Initialize captured writes."""
            self.sync_calls: list[tuple[str, Any]] = []
            self.async_calls: list[tuple[str, Any]] = []

        def __call__(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
            """Record one blocking write."""
            del skip_if_exists
            self.sync_calls.append((key, value))
            return True

        async def aio(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
            """Record one async write."""
            del skip_if_exists
            self.async_calls.append((key, value))
            return True

    class FakeInterruptStore:
        """Simple Modal Dict double exposing put.aio."""

        def __init__(self) -> None:
            """Initialize the fake async put handle."""
            self.put = FakePut()

    interrupt_store = FakeInterruptStore()

    class FakeModalDict:
        """Minimal modal.Dict shim that returns the fake interrupt store."""

        @staticmethod
        def from_name(
            name: str,
            *,
            environment_name: str | None = None,
            create_if_missing: bool = False,
            client: Any | None = None,
        ) -> FakeInterruptStore:
            return interrupt_store

    monkeypatch.setattr(
        modal_interrupts_module,
        "modal",
        types.SimpleNamespace(Dict=FakeModalDict),
    )
    monkeypatch.setenv("COMFY_MODAL_INTERRUPT_DICT_NAME", "shared-interrupts")
    remote_modal_app_module.get_settings.cache_clear()
    modal_interrupts_module._MODAL_INTERRUPT_DICTS.clear()
    try:
        wrote_interrupt = asyncio.run(
            remote_modal_app_module._request_remote_interrupt_async(
                {"prompt_id": "prompt-1", "component_id": "component-2"}
            )
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_interrupts_module._MODAL_INTERRUPT_DICTS.clear()

    assert wrote_interrupt is True
    assert interrupt_store.put.sync_calls == []
    assert len(interrupt_store.put.async_calls) == 1
    interrupt_key, interrupt_value = interrupt_store.put.async_calls[0]
    assert interrupt_key == "prompt-1:component-2"
    assert isinstance(interrupt_value["requested_at"], float)

def test_local_remote_app_normalizes_wrapped_scalar_prompt_inputs(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback runner should unwrap singleton-list scalar prompt inputs."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "BoundarySource",
                    "inputs": {"value": [4]},
                    "_meta": {},
                }
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "remote_1_value",
                    "node_id": "remote_1",
                    "output_index": 0,
                    "io_type": "INT",
                    "is_list": False,
                }
            ],
            "execute_node_ids": [["remote_1"]],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload="{}",
        node_mapping={
            "BoundarySource": _BoundarySourceNode,
        },
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (5,)

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_apply_boundary_inputs_normalizes_wrapped_scalar_values(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Boundary input hydration should unwrap singleton-list scalar wrappers before PromptExecutor sees them."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "BoundarySource",
            "inputs": {"value": 0},
            "_meta": {},
        }
    }

    target_module._apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=[
            {
                "proxy_input_name": "remote_input_0",
                "targets": [{"node_id": "remote_1", "input_name": "value"}],
            }
        ],
        hydrated_inputs={"remote_input_0": [4]},
    )

    assert prompt["remote_1"]["inputs"]["value"] == 4

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_apply_boundary_inputs_preserves_singleton_conditioning_lists(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Boundary input hydration must not flatten singleton CONDITIONING payloads."""
    target_module = request.getfixturevalue(module_fixture_name)
    conditioning = [["cond", {"pooled_output": None}]]
    prompt = {
        "remote_1": {
            "class_type": "BoundarySource",
            "inputs": {"value": 0},
            "_meta": {},
        }
    }

    target_module._apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=[
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "CONDITIONING",
                "targets": [{"node_id": "remote_1", "input_name": "value"}],
            }
        ],
        hydrated_inputs={"remote_input_0": conditioning},
    )

    assert prompt["remote_1"]["inputs"]["value"] == conditioning

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_validate_required_prompt_inputs_reports_missing_latent_image(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Remote execution should fail before PromptExecutor calls samplers with missing required inputs."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "3": {
            "class_type": "KSamplerLoraSigmaInverse",
            "inputs": {
                "seed": 1,
                "steps": 24,
                "positive": [["cond", {"reference_latents": ["reference"]}]],
            },
        }
    }

    with pytest.raises(target_module.RemoteSubgraphExecutionError) as exc_info:
        target_module._validate_required_prompt_inputs(
            prompt,
            {"KSamplerLoraSigmaInverse": _FakeImplicitBatchKSamplerNode},
        )

    message = str(exc_info.value)
    assert "missing required node inputs" in message
    assert "latent_image" in message
    assert "available_inputs=['positive', 'seed', 'steps']" in message

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_validate_required_prompt_inputs_expands_v3_autogrow_inputs(
    request: Any,
    module_fixture_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V3 Autogrow prompts should validate their expanded `group.name` sockets."""
    target_module = request.getfixturevalue(module_fixture_name)
    _install_fake_v3_input_finalizer(monkeypatch)
    batch_images_node = _v3_batch_images_node_class()
    live_inputs = {
        "images.image0": ["101", 0],
        "images.image1": ["102", 0],
    }
    prompt = {
        "153": {
            "class_type": "ModalTestV3BatchImagesNode",
            "inputs": live_inputs,
        }
    }

    target_module._validate_required_prompt_inputs(
        prompt,
        {"ModalTestV3BatchImagesNode": batch_images_node},
    )

    assert target_module._node_required_input_names(
        batch_images_node,
        live_inputs,
    ) == {"images.image0"}
    input_type_map = target_module._node_input_type_map(batch_images_node, live_inputs)
    assert input_type_map["images.image0"] == "IMAGE"
    assert input_type_map["images.image1"] == "IMAGE"
    assert "images" not in input_type_map

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_validate_prompt_input_shapes_rejects_list_on_primitive_socket(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Prepared remote prompts should fail early when primitive widget inputs still carry raw lists."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "BoundarySource",
            "inputs": {"value": [4, 5]},
            "_meta": {},
        }
    }

    with pytest.raises(target_module.RemoteSubgraphExecutionError, match="input_name='value'"):
        target_module._validate_prompt_input_shapes(
            prompt,
            {"BoundarySource": _BoundarySourceNode},
        )

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_validate_prompt_input_shapes_allows_boundary_supplied_list_on_primitive_socket(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Boundary-fed primitive lists should pass through so ComfyUI can map over them normally."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "BoundarySource",
            "inputs": {"value": [4, 5]},
            "_meta": {},
        }
    }

    target_module._validate_prompt_input_shapes(
        prompt,
        {"BoundarySource": _BoundarySourceNode},
        [
            {
                "proxy_input_name": "remote_input_0",
                "targets": [{"node_id": "remote_1", "input_name": "value"}],
            }
        ],
    )

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_coerce_prompt_primitive_input_values_matches_comfyui_semantics(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Remote runtimes should coerce primitive prompt literals the same way ComfyUI does."""
    target_module = request.getfixturevalue(module_fixture_name)
    prompt = {
        "remote_1": {
            "class_type": "PrimitiveEcho",
            "inputs": {
                "steps": 18.0,
                "cfg": 5,
                "enabled": 1,
                "label": 7,
            },
            "_meta": {},
        }
    }

    target_module._coerce_prompt_primitive_input_values(
        prompt,
        {"PrimitiveEcho": _PrimitiveEchoNode},
    )

    assert prompt["remote_1"]["inputs"] == {
        "steps": 18,
        "cfg": 5.0,
        "enabled": True,
        "label": "7",
    }
    assert isinstance(prompt["remote_1"]["inputs"]["steps"], int)
    assert isinstance(prompt["remote_1"]["inputs"]["cfg"], float)
    assert isinstance(prompt["remote_1"]["inputs"]["enabled"], bool)
    assert isinstance(prompt["remote_1"]["inputs"]["label"], str)

def test_local_remote_app_coerces_primitive_widget_literals_before_execution(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback runner should coerce primitive widget literals before executing nodes."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-primitive-coercion",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "PrimitiveEcho",
                    "inputs": {
                        "steps": 18.0,
                        "cfg": 5,
                        "enabled": 1,
                        "label": 7,
                    },
                    "_meta": {},
                }
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "steps",
                    "node_id": "remote_1",
                    "output_index": 0,
                    "io_type": "INT",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "cfg",
                    "node_id": "remote_1",
                    "output_index": 1,
                    "io_type": "FLOAT",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "enabled",
                    "node_id": "remote_1",
                    "output_index": 2,
                    "io_type": "BOOLEAN",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "label",
                    "node_id": "remote_1",
                    "output_index": 3,
                    "io_type": "STRING",
                    "is_list": False,
                },
            ],
            "execute_node_ids": ["remote_1"],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload="{}",
        node_mapping={"PrimitiveEcho": _PrimitiveEchoNode},
    )

    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (18, 5.0, True, "7")
