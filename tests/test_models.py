"""Unit tests for the vLLM model / LoRA lifecycle layer.

Proves: /v1/models normalises + flags LoRA adapters, lora_load posts to the
right endpoint, the write tools carry correct risk tiers, and the high-risk
lora_unload dry-run previews without touching the engine. No real vLLM needed —
the connection is a MagicMock.
"""

import pathlib
from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_model_list_normalizes_and_flags_lora():
    from inference_aiops.ops import models as ops

    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {
        "object": "list",
        "data": [
            {"id": "meta-llama/Llama-3-8B", "object": "model",
             "owned_by": "vllm", "root": "meta-llama/Llama-3-8B", "parent": None},
            {"id": "my-lora", "object": "model", "owned_by": "vllm",
             "root": "meta-llama/Llama-3-8B", "parent": "meta-llama/Llama-3-8B"},
        ],
    }
    rows = ops.list_models(conn)
    assert rows[0]["id"] == "meta-llama/Llama-3-8B"
    assert rows[0]["isLora"] is False
    assert rows[1]["id"] == "my-lora"
    assert rows[1]["isLora"] is True


@pytest.mark.unit
def test_lora_load_posts_to_load_endpoint():
    from inference_aiops.ops import models as ops

    conn = MagicMock(name="conn")
    conn.post_vllm.return_value = {}
    out = ops.load_lora_adapter(conn, "my-lora", "/models/my-lora")
    conn.post_vllm.assert_called_once_with(
        "/v1/load_lora_adapter",
        json={"lora_name": "my-lora", "lora_path": "/models/my-lora"},
    )
    assert out["action"] == "lora_load"
    assert out["loraName"] == "my-lora" and out["loraPath"] == "/models/my-lora"


@pytest.mark.unit
def test_model_write_tools_have_correct_risk_tiers():
    from mcp_server.tools import models as md

    assert md.lora_load._risk_level == "medium"
    assert md.lora_unload._risk_level == "high"
    assert md.model_sleep._risk_level == "high"
    assert md.model_wake._risk_level == "medium"
    assert md.model_is_sleeping._risk_level == "low"


@pytest.mark.unit
def test_lora_unload_dry_run_does_not_mutate(monkeypatch):
    from mcp_server.tools import models as md

    conn = MagicMock(name="conn")
    monkeypatch.setattr(md, "_get_connection", lambda target=None: conn)

    result = md.lora_unload(lora_name="my-lora", dry_run=True)
    assert result["dryRun"] is True
    assert result["wouldUnload"] == {"loraName": "my-lora"}
    conn.post_vllm.assert_not_called()


@pytest.mark.unit
def test_hot_swap_tool_is_gone_from_the_registry():
    """model_hot_swap POSTed to /v1/hot_swap, which vLLM has never served.

    It could not be repaired by renaming a path, and being [WRITE][high] it burnt
    an approval before 404ing. It must be absent from the module AND from the MCP
    registry — a tool a weak model can still see is a tool it will still call.
    """
    from inference_aiops.ops import models as ops
    from mcp_server.server import mcp
    from mcp_server.tools import models as md

    assert not hasattr(md, "model_hot_swap")
    assert not hasattr(ops, "hot_swap_model")
    assert "model_hot_swap" not in mcp._tool_manager._tools


@pytest.mark.unit
def test_no_module_references_the_invented_hot_swap_path():
    """/v1/hot_swap must not survive anywhere in the shipped source."""
    import inference_aiops.ops.models as models_mod

    assert "hot_swap" not in pathlib.Path(models_mod.__file__).read_text()


@pytest.mark.unit
def test_model_sleep_dry_run_does_not_mutate(monkeypatch):
    from mcp_server.tools import models as md

    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": False}
    monkeypatch.setattr(md, "_get_connection", lambda target=None: conn)

    result = md.model_sleep(level=2, dry_run=True)
    assert result["dryRun"] is True
    assert result["wouldSleep"] == {"level": 2}
    assert result["currentlySleeping"] is False
    conn.post_vllm.assert_not_called()


@pytest.mark.unit
def test_model_wake_dry_run_does_not_mutate(monkeypatch):
    from mcp_server.tools import models as md

    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": True}
    monkeypatch.setattr(md, "_get_connection", lambda target=None: conn)

    result = md.model_wake(dry_run=True)
    assert result["dryRun"] is True and result["wouldWake"] is True
    assert result["currentlySleeping"] is True
    conn.post_vllm.assert_not_called()


@pytest.mark.unit
def test_sleep_undo_wakes_only_what_it_put_to_sleep():
    from mcp_server.tools import models as md

    put_to_sleep = md._sleep_undo({"target": "gpu1"},
                                  {"priorState": {"wasSleeping": False}})
    assert put_to_sleep["tool"] == "model_wake"
    assert put_to_sleep["params"] == {"target": "gpu1"}


@pytest.mark.unit
@pytest.mark.parametrize("prior", [True, None])
def test_sleep_undo_declines_when_it_did_not_cause_the_sleep(prior):
    """Already-asleep or unknown => no descriptor; an undo must not invent a state."""
    from mcp_server.tools import models as md

    assert md._sleep_undo({"target": None}, {"priorState": {"wasSleeping": prior}}) is None
