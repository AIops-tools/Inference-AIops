"""Absent fields come back as null, not as an empty string.

An empty string reads as "the engine reported this field and it was blank"; a
missing field is a different fact. Collapsing the two hides information from any
consumer, and a smaller local model will confidently invent the difference.
These tests pin the contract end-to-end: helper, ops layer, and the truncation
envelope that has to survive a null field in a row.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.governance import opt_str
from inference_aiops.ops import models as model_ops
from inference_aiops.ops import ray_cluster as ray_ops
from inference_aiops.ops import serve as serve_ops
from inference_aiops.ops._util import opt_s, s


@pytest.mark.unit
def test_opt_str_distinguishes_absent_from_empty():
    assert opt_str(None) is None, "absent must stay absent"
    assert opt_str("") == "", "a genuinely empty value is not the same as absent"
    assert opt_str("meta-llama/Llama-3-8B", 64) == "meta-llama/Llama-3-8B"


@pytest.mark.unit
def test_opt_str_still_sanitizes_and_truncates():
    assert opt_str("a\x00b") == "ab"  # control character stripped
    # A cut announces itself: the ellipsis is the only signal a reader gets
    # that what they are looking at is not the whole value.
    assert opt_str("abcdef", 3) == "ab\u2026"
    assert opt_str("abc", 3) == "abc"  # exactly at the cap is not truncated


@pytest.mark.unit
def test_opt_str_accepts_non_string_values():
    assert opt_str(42) == "42"


@pytest.mark.unit
def test_opt_s_and_s_differ_only_on_absence():
    assert s(None) == "" and opt_s(None) is None
    assert s("RUNNING") == opt_s("RUNNING") == "RUNNING"


@pytest.mark.unit
def test_model_rows_report_absent_fields_as_none():
    """A /v1/models row with no object/owned_by reports null, not ''."""
    conn = MagicMock()
    conn.get_vllm.return_value = [{"id": "llama-3"}]
    row = model_ops.list_models(conn)[0]
    assert row["id"] == "llama-3"
    assert row["object"] is None and row["ownedBy"] is None
    assert row["isLora"] is False


@pytest.mark.unit
def test_model_rows_keep_empty_string_when_source_is_empty():
    """An explicitly empty upstream value is preserved as '' — not turned into null."""
    conn = MagicMock()
    conn.get_vllm.return_value = [{"id": "llama-3", "owned_by": ""}]
    assert model_ops.list_models(conn)[0]["ownedBy"] == ""


@pytest.mark.unit
def test_ops_never_drop_the_key_itself():
    """Keys are always present; only their value may be null."""
    conn = MagicMock()
    conn.get_vllm.return_value = [{}]
    row = model_ops.list_models(conn)[0]
    for key in ("id", "object", "ownedBy", "isLora"):
        assert key in row, f"{key} must be present even when the source omitted it"


@pytest.mark.unit
def test_lora_detection_survives_null_ids():
    """A row with neither id nor root is not silently classified as a LoRA adapter.

    Both fold to None, and ``None != None`` is False — so the comparison that
    detects an adapter must not fire on two absences.
    """
    conn = MagicMock()
    conn.get_vllm.return_value = [{}]
    assert model_ops.list_models(conn)[0]["isLora"] is False


@pytest.mark.unit
def test_model_info_reports_absent_root_and_parent_as_none():
    conn = MagicMock()
    conn.get_vllm.return_value = [{"id": "llama-3", "max_model_len": 8192}]
    out = model_ops.get_model_info(conn, "llama-3")
    assert out["maxModelLen"] == 8192
    assert out["root"] is None and out["parent"] is None


@pytest.mark.unit
def test_serve_deployment_status_is_none_when_absent():
    conn = MagicMock()
    conn.get_ray.return_value = {"applications": {"app": {"deployments": {"d": {}}}}}
    row = serve_ops.list_serve_deployments(conn)[0]
    assert row["application"] == "app" and row["deployment"] == "d"
    assert row["status"] is None


@pytest.mark.unit
def test_dashboard_status_treats_absent_app_status_as_not_healthy():
    """An app whose status the dashboard omitted must not read as RUNNING.

    This is the consumer a naive conversion gets wrong: ``all(st == "RUNNING")``
    over a list containing None must be False, not silently healthy.
    """
    conn = MagicMock()
    conn.get_ray.return_value = {"applications": {"app": {}}}
    assert ray_ops.get_dashboard_status(conn)["serveController"] == "DEGRADED"


@pytest.mark.unit
def test_job_rows_report_absent_fields_as_none():
    conn = MagicMock()
    conn.get_ray.return_value = [{"job_id": "01"}]
    row = ray_ops.list_jobs(conn)["jobs"][0]
    assert row["jobId"] == "01"
    assert row["status"] is None and row["entrypoint"] is None


# ── truncation announces itself ──────────────────────────────────────────


def _many_jobs(n: int) -> list[dict]:
    return [{"job_id": f"{i:02d}", "status": "SUCCEEDED"} for i in range(n)]


@pytest.mark.unit
def test_list_jobs_returns_a_truncation_envelope():
    conn = MagicMock()
    conn.get_ray.return_value = _many_jobs(5)
    out = ray_ops.list_jobs(conn, limit=2)
    assert out["returned"] == 2 and out["limit"] == 2
    assert out["truncated"] is True, "more jobs existed than were returned"
    assert len(out["jobs"]) == 2


@pytest.mark.unit
def test_list_jobs_is_not_truncated_at_exactly_the_limit():
    """The boundary case a length-comparison heuristic gets wrong.

    Exactly ``limit`` rows is NOT truncation. Measuring against the full fetch is
    what makes this answerable instead of guessed.
    """
    conn = MagicMock()
    conn.get_ray.return_value = _many_jobs(2)
    out = ray_ops.list_jobs(conn, limit=2)
    assert out["returned"] == 2 and out["truncated"] is False


@pytest.mark.unit
def test_list_jobs_reports_a_failure_as_an_error_envelope():
    conn = MagicMock()
    conn.get_ray.side_effect = RuntimeError("dashboard down")
    out = ray_ops.list_jobs(conn)
    assert "error" in out and "dashboard down" in out["error"]


@pytest.mark.unit
def test_undo_list_envelope_measures_truncation(monkeypatch):
    from mcp_server.tools import undo as undo_tools

    rows = [
        {
            "undo_id": f"u{i}",
            "ts": "2026-07-18T00:00:00Z",
            "tool": "some_tool",
            "undo_tool": "some_inverse_tool",
            "note": "",
        }
        for i in range(4)
    ]
    captured = {}

    class _Store:
        def list(self, *, status=None, limit=50):
            captured["limit"] = limit
            return rows[:limit]

    monkeypatch.setattr(undo_tools, "get_undo_store", lambda: _Store())
    result = undo_tools.undo_list(limit=3)
    assert captured["limit"] == 4, "one extra row is fetched to measure truncation"
    assert result["returned"] == 3
    assert result["limit"] == 3
    assert result["truncated"] is True
    assert len(result["undos"]) == 3
