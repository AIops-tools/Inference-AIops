"""vLLM model-info read + LoRA-unload / hot-swap writes (with prior-state undo).

Complements test_models: proves get_model_info resolves one row (and its
not-found teaching path), unload posts to the right endpoint, and hot_swap
captures the prior base model into ``priorState`` for a faithful undo.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.ops import models as ops

_MODELS = {"data": [
    {"id": "meta/llama3", "object": "model", "owned_by": "vllm",
     "max_model_len": 8192, "root": "meta/llama3", "parent": None,
     "permission": [{"id": "perm-1"}]},
    {"id": "my-lora", "object": "model", "owned_by": "vllm",
     "root": "meta/llama3", "parent": "meta/llama3"},
]}


@pytest.mark.unit
def test_get_model_info_returns_resolved_row():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = _MODELS
    info = ops.get_model_info(conn, "meta/llama3")
    assert info["id"] == "meta/llama3"
    assert info["maxModelLen"] == 8192
    assert info["parent"] is None
    assert info["permission"] == [{"id": "perm-1"}]


@pytest.mark.unit
def test_get_model_info_not_found_is_teaching():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = _MODELS
    out = ops.get_model_info(conn, "does/not-exist")
    assert "error" in out and "not found" in out["error"]


@pytest.mark.unit
def test_get_model_info_read_failure_degrades():
    conn = MagicMock(name="conn")
    conn.get_vllm.side_effect = RuntimeError("models 500")
    assert "error" in ops.get_model_info(conn, "x")


@pytest.mark.unit
def test_unload_lora_posts_to_unload_endpoint():
    conn = MagicMock(name="conn")
    out = ops.unload_lora_adapter(conn, "my-lora")
    conn.post_vllm.assert_called_once_with("/v1/unload_lora_adapter",
                                           json={"lora_name": "my-lora"})
    assert out == {"action": "lora_unload", "loraName": "my-lora"}


@pytest.mark.unit
def test_hot_swap_captures_prior_base_model():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = _MODELS
    out = ops.hot_swap_model(conn, "mistral/mixtral")
    assert out["action"] == "model_hot_swap"
    assert out["newModel"] == "mistral/mixtral"
    assert out["priorState"] == {"model": "meta/llama3"}  # first served id
    conn.post_vllm.assert_called_once_with("/v1/hot_swap", json={"model": "mistral/mixtral"})


@pytest.mark.unit
def test_hot_swap_with_no_models_has_null_prior():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"data": []}
    out = ops.hot_swap_model(conn, "new/model")
    assert out["priorState"] == {"model": None}
