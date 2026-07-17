"""Ray Serve deployment reads + guarded scale/drain writes (ops + MCP).

Proves: the app→deployment map is flattened into normalised rows, reads degrade
to an ``error`` field, scale writes capture the BEFORE replica count into
``priorState`` for a faithful undo, and the MCP layer's dry_run previews never
touch the backend while the undo builder inverts a scale back to the prior count.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.ops import serve as ops

# A representative Ray Serve /api/serve/applications/ payload.
_APPS = {
    "applications": {
        "llm": {
            "deployments": {
                "VLLMDeployment": {
                    "status": "HEALTHY",
                    "replicas": [{"state": "RUNNING"}, {"state": "RUNNING"},
                                 {"state": "STARTING"}],
                    "deployment_config": {
                        "num_replicas": 3,
                        "autoscaling_config": {"min_replicas": 1, "max_replicas": 6,
                                               "target_ongoing_requests": 8},
                    },
                }
            }
        }
    }
}


def _conn_reading_apps():
    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _APPS
    return conn


@pytest.mark.unit
def test_list_serve_deployments_flattens_rows():
    rows = ops.list_serve_deployments(_conn_reading_apps())
    assert len(rows) == 1
    row = rows[0]
    assert row["application"] == "llm" and row["deployment"] == "VLLMDeployment"
    assert row["status"] == "HEALTHY"
    assert row["numReplicas"] == 3
    assert row["replicaStates"] == ["RUNNING", "RUNNING", "STARTING"]


@pytest.mark.unit
def test_deployment_status_and_list_replicas_and_autoscale():
    conn = _conn_reading_apps()
    st = ops.get_deployment_status(conn, "llm", "VLLMDeployment")
    assert st["numReplicas"] == 3
    reps = ops.list_replicas(conn, "llm", "VLLMDeployment")
    assert reps["replicaStates"] == ["RUNNING", "RUNNING", "STARTING"]
    auto = ops.get_autoscale_config(conn, "llm", "VLLMDeployment")
    assert auto["minReplicas"] == 1 and auto["maxReplicas"] == 6
    assert auto["targetOngoingRequests"] == 8


@pytest.mark.unit
def test_deployment_status_not_found_degrades_to_error():
    conn = _conn_reading_apps()
    out = ops.get_deployment_status(conn, "llm", "ghost")
    assert "error" in out and "not found" in out["error"]


@pytest.mark.unit
def test_list_serve_deployments_read_failure_degrades():
    conn = MagicMock(name="conn")
    conn.get_ray.side_effect = RuntimeError("dashboard down")
    rows = ops.list_serve_deployments(conn)
    assert rows == [{"error": rows[0]["error"]}] and "dashboard down" in rows[0]["error"]


@pytest.mark.unit
def test_scale_up_captures_prior_replica_count():
    conn = _conn_reading_apps()
    out = ops.scale_replicas_up(conn, "llm", "VLLMDeployment", 5)
    assert out["action"] == "scale_replicas_up"
    assert out["numReplicas"] == 5
    assert out["priorState"] == {"numReplicas": 3}
    (path,) = conn.put_ray.call_args.args
    assert path == "/api/serve/applications/llm/deployments/VLLMDeployment"
    assert conn.put_ray.call_args.kwargs["json"] == {"num_replicas": 5}


@pytest.mark.unit
def test_scale_to_zero_targets_zero_and_captures_prior():
    conn = _conn_reading_apps()
    out = ops.scale_to_zero(conn, "llm", "VLLMDeployment")
    assert out["action"] == "scale_to_zero" and out["numReplicas"] == 0
    assert out["priorState"] == {"numReplicas": 3}


@pytest.mark.unit
def test_update_autoscale_config_builds_partial_body_and_prior():
    conn = _conn_reading_apps()
    out = ops.update_autoscale_config(conn, "llm", "VLLMDeployment", max_replicas=10)
    assert out["applied"] == {"max_replicas": 10}
    assert out["priorState"]["maxReplicas"] == 6
    (path,) = conn.put_ray.call_args.args
    assert path.endswith("/autoscale")


@pytest.mark.unit
def test_scale_write_tools_risk_tiers():
    from mcp_server.tools import serve as sv

    assert sv.scale_replicas_up._risk_level == "medium"
    assert sv.scale_replicas_down._risk_level == "high"
    assert sv.scale_to_zero._risk_level == "high"
    assert sv.drain_replica._risk_level == "high"
    assert sv.autoscale_config_update._risk_level == "medium"


@pytest.mark.unit
def test_mcp_scale_to_zero_dry_run_does_not_write(monkeypatch):
    from mcp_server.tools import serve as sv

    conn = _conn_reading_apps()
    monkeypatch.setattr(sv, "_get_connection", lambda target=None: conn)
    out = sv.scale_to_zero(application="llm", deployment="VLLMDeployment", dry_run=True)
    assert out == {"dryRun": True, "from": 3, "to": 0}
    conn.put_ray.assert_not_called()


@pytest.mark.unit
def test_mcp_drain_replica_dry_run_previews(monkeypatch):
    from mcp_server.tools import serve as sv

    conn = MagicMock(name="conn")
    monkeypatch.setattr(sv, "_get_connection", lambda target=None: conn)
    out = sv.drain_replica(application="llm", deployment="d", replica_id="r-1", dry_run=True)
    assert out["dryRun"] is True
    assert out["wouldDrain"] == {"application": "llm", "deployment": "d", "replicaId": "r-1"}
    conn.post_ray.assert_not_called()


@pytest.mark.unit
def test_replica_undo_inverts_scale_to_prior_count():
    from mcp_server.tools import serve as sv

    undo = sv._replica_undo(
        {"application": "llm", "deployment": "d"},
        {"action": "scale_to_zero", "priorState": {"numReplicas": 4}},
    )
    assert undo["tool"] == "scale_replicas_up"
    assert undo["params"]["num_replicas"] == 4
    # No prior captured → no undo record.
    assert sv._replica_undo({}, {"priorState": {}}) is None


@pytest.mark.unit
def test_autoscale_undo_restores_prior_bounds():
    from mcp_server.tools import serve as sv

    undo = sv._autoscale_undo(
        {"application": "llm", "deployment": "d"},
        {"priorState": {"minReplicas": 1, "maxReplicas": 6, "targetOngoingRequests": 8}},
    )
    assert undo["tool"] == "update_autoscale_config"
    assert undo["params"]["max_replicas"] == 6
    # All-None prior → nothing to restore.
    assert sv._autoscale_undo({}, {"priorState": {"minReplicas": None}}) is None
