"""Unit tests for the vLLM model / LoRA lifecycle layer.

Proves: /v1/models normalises + flags LoRA adapters, lora_load posts to the
right endpoint, the write tools carry correct risk tiers, and the high-risk
lora_unload dry-run previews without touching the engine. No real vLLM needed —
the connection is a MagicMock.
"""

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
    assert md.model_hot_swap._risk_level == "high"


@pytest.mark.unit
def test_lora_unload_dry_run_does_not_mutate(monkeypatch):
    from mcp_server.tools import models as md

    conn = MagicMock(name="conn")
    monkeypatch.setattr(md, "_get_connection", lambda target=None: conn)

    result = md.lora_unload(lora_name="my-lora", dry_run=True)
    assert result["dryRun"] is True
    assert result["wouldUnload"] == {"loraName": "my-lora"}
    conn.post_vllm.assert_not_called()
