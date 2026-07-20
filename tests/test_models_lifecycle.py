"""vLLM model-info read + LoRA-unload / Sleep-Mode writes (with prior-state undo).

Complements test_models: proves get_model_info resolves one row (and its
not-found teaching path), unload posts to the right endpoint, and the Sleep-Mode
pair captures whether the engine was ALREADY asleep into ``priorState`` so the
undo never wakes an engine it did not suspend.

The Sleep-Mode routes exist only under VLLM_SERVER_DEV_MODE=1, so the 404 path is
tested too: it must teach that the server was not started in dev mode rather than
degrade into the generic stale-id 404 message.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.connection import EngineCapabilityError, InferenceApiError
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
def test_sleep_captures_prior_awake_state():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": False}
    out = ops.sleep_model(conn, 1)
    assert out["action"] == "model_sleep"
    assert out["level"] == 1
    assert out["priorState"] == {"wasSleeping": False}
    conn.post_vllm.assert_called_once_with("/sleep", params={"level": 1})


@pytest.mark.unit
def test_sleep_records_already_sleeping_so_undo_can_decline():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": True}
    out = ops.sleep_model(conn, 2)
    assert out["priorState"] == {"wasSleeping": True}
    conn.post_vllm.assert_called_once_with("/sleep", params={"level": 2})


@pytest.mark.unit
def test_sleep_probe_failure_leaves_prior_state_unknown():
    """A failed probe must read as UNKNOWN (None), never as 'it was awake'."""
    conn = MagicMock(name="conn")
    conn.get_vllm.side_effect = RuntimeError("probe blew up")
    out = ops.sleep_model(conn, 1)
    assert out["priorState"] == {"wasSleeping": None}
    conn.post_vllm.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("level", [0, 3, -1, "1"])
def test_sleep_rejects_invalid_level_before_calling_the_engine(level):
    conn = MagicMock(name="conn")
    with pytest.raises(ValueError, match="level=1"):
        ops.sleep_model(conn, level)
    conn.post_vllm.assert_not_called()


@pytest.mark.unit
def test_wake_posts_to_wake_up_and_captures_prior_state():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": True}
    out = ops.wake_model(conn)
    assert out["action"] == "model_wake"
    assert out["priorState"] == {"wasSleeping": True}
    conn.post_vllm.assert_called_once_with("/wake_up")


@pytest.mark.unit
def test_is_sleeping_reports_the_flag():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": True}
    assert ops.is_sleeping(conn) == {"isSleeping": True}
    conn.get_vllm.assert_called_once_with("/is_sleeping")


@pytest.mark.unit
def test_is_sleeping_missing_flag_is_unknown_not_false():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {}
    assert ops.is_sleeping(conn) == {"isSleeping": None}


@pytest.mark.unit
def test_sleep_404_teaches_dev_mode_not_a_stale_id():
    """Bug class 7: a route that only exists under a flag must say so."""
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": False}
    conn.post_vllm.side_effect = InferenceApiError("nope", status_code=404, path="/sleep")
    with pytest.raises(EngineCapabilityError, match="VLLM_SERVER_DEV_MODE=1"):
        ops.sleep_model(conn, 1)


@pytest.mark.unit
def test_wake_404_teaches_dev_mode():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": True}
    conn.post_vllm.side_effect = InferenceApiError("nope", status_code=404, path="/wake_up")
    with pytest.raises(EngineCapabilityError, match="VLLM_SERVER_DEV_MODE=1"):
        ops.wake_model(conn)


@pytest.mark.unit
def test_is_sleeping_404_teaches_dev_mode_in_the_error_field():
    conn = MagicMock(name="conn")
    conn.get_vllm.side_effect = InferenceApiError("nope", status_code=404,
                                                  path="/is_sleeping")
    out = ops.is_sleeping(conn)
    assert "VLLM_SERVER_DEV_MODE=1" in out["error"]


@pytest.mark.unit
def test_sleep_non_404_error_is_not_recast_as_dev_mode():
    """Only 404 means 'route absent'; a 500 must surface as the server error."""
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": False}
    conn.post_vllm.side_effect = InferenceApiError("boom", status_code=500, path="/sleep")
    with pytest.raises(InferenceApiError) as ei:
        ops.sleep_model(conn, 1)
    assert not isinstance(ei.value, EngineCapabilityError)


# ── the dry-run preview must reach the same guard as the write ────────────


@pytest.mark.unit
def test_sleep_preview_hits_the_dev_mode_guard_the_write_would():
    """A preview that reads 'all clear' on a server the write cannot use is a trap.

    model_sleep is [WRITE][high], so a caller pays an approval to run it. If the
    dry run reported currentlySleeping=null on a server with no Sleep-Mode
    routes, that approval would buy nothing but the 404 the preview already had
    in hand.
    """
    conn = MagicMock(name="conn")
    conn.get_vllm.side_effect = InferenceApiError("nope", status_code=404,
                                                  path="/is_sleeping")
    with pytest.raises(EngineCapabilityError, match="VLLM_SERVER_DEV_MODE=1"):
        ops.preview_sleep_state(conn)


@pytest.mark.unit
def test_preview_reports_the_state_on_a_dev_mode_server():
    conn = MagicMock(name="conn")
    conn.get_vllm.return_value = {"is_sleeping": True}
    assert ops.preview_sleep_state(conn) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome",
    [
        {"side_effect": InferenceApiError("boom", status_code=500, path="/is_sleeping")},
        {"side_effect": RuntimeError("socket died")},
        {"return_value": {}},
        {"return_value": "not a dict"},
    ],
)
def test_preview_degrades_to_unknown_rather_than_inventing_a_blocker(outcome):
    """Only a 404 proves the route is absent; anything else is 'unknown'.

    A transient read failure is not evidence the write would fail, so the
    preview must not manufacture a blocker out of it — null means unknown.
    """
    conn = MagicMock(name="conn")
    conn.get_vllm.configure_mock(**outcome)
    assert ops.preview_sleep_state(conn) is None


@pytest.mark.unit
@pytest.mark.parametrize("tool,kwargs", [("model_sleep", {"level": 1}), ("model_wake", {})])
def test_both_previews_refuse_to_look_clear_without_sleep_mode(monkeypatch, tool, kwargs):
    """Preview and real call agree about capability, for sleep and wake alike."""
    from mcp_server.tools import models as md

    conn = MagicMock(name="conn")
    conn.get_vllm.side_effect = InferenceApiError("nope", status_code=404,
                                                  path="/is_sleeping")
    monkeypatch.setattr(md, "_get_connection", lambda target=None: conn)

    out = getattr(md, tool)(dry_run=True, **kwargs)
    assert "VLLM_SERVER_DEV_MODE=1" in out["error"]
    conn.post_vllm.assert_not_called()
