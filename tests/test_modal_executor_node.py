"""Tests for dynamic Modal proxy nodes and local execution fallback."""

from __future__ import annotations

from typing import Any


class _FakeOriginalNode:
    """Simple fake legacy node for proxy signature mirroring."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "count")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "run"

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        """Return a tuple that exposes the inputs for verification."""
        return (kwargs["value"], 1)


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


def test_proxy_execution_uses_injected_remote_client(
    modal_executor_module: Any,
) -> None:
    """Proxy execution should delegate to the configured remote client."""
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

        def execute_node(self, node_data: dict[str, Any], kwargs: dict[str, Any]) -> tuple[str, int]:
            """Return values derived from the proxied node payload."""
            return (f"{node_data['class_type']}::{kwargs['value']}", 3)

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        result = proxy_class.execute({"class_type": "OriginalNode"}, value="payload")
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert result.result == ("OriginalNode::payload", 3)


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
