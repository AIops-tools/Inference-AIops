"""Ray cluster / jobs / GPU tests.

Proves: job listing normalises the /api/jobs/ payload, cluster-resource parsing
pulls GPU totals and degrades to an ``error`` field on a dashboard failure, the
write tools carry the correct risk tiers, and replica_restart's dry-run preview
never touches the backend.
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_ray_job_list_normalizes_payload():
    from inference_aiops.ops import ray_cluster as ops

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = [
        {"job_id": "01", "status": "RUNNING",
         "entrypoint": "python serve.py", "start_time": 1700000000},
        {"submission_id": "raysubmit_02", "status": "SUCCEEDED",
         "entrypoint": "python train.py", "startTime": 1700000100},
    ]
    out = ops.list_jobs(conn)
    assert out["returned"] == 2 and out["limit"] == 100 and out["truncated"] is False
    rows = out["jobs"]
    assert rows[0] == {"jobId": "01", "status": "RUNNING",
                       "entrypoint": "python serve.py", "startTime": 1700000000}
    assert rows[1]["jobId"] == "raysubmit_02"
    assert rows[1]["status"] == "SUCCEEDED" and rows[1]["startTime"] == 1700000100


@pytest.mark.unit
def test_get_cluster_resources_parses_gpu_and_is_resilient():
    from inference_aiops.ops import ray_cluster as ops

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {
        "clusterResources": {"CPU": 64.0, "GPU": 8.0},
        "availableResources": {"CPU": 32.0, "GPU": 3.0},
        "pendingPlacementGroups": [{"id": "pg1"}, {"id": "pg2"}],
    }
    out = ops.get_cluster_resources(conn)
    assert out["totalGpu"] == 8.0 and out["availableGpu"] == 3.0
    assert out["totalCpu"] == 64.0 and out["pendingPlacementGroups"] == 2

    conn.get_ray.side_effect = RuntimeError("dashboard down")
    assert "error" in ops.get_cluster_resources(conn)


@pytest.mark.unit
def test_write_tools_have_correct_risk_tiers():
    from mcp_server.tools import ray_cluster as rc

    assert rc.replica_restart._risk_level == "high"
    assert rc.ray_job_cancel._risk_level == "medium"
    assert rc.ray_cluster_resources._risk_level == "low"
    assert rc.ray_dashboard_status._risk_level == "low"
    assert rc.ray_job_list._risk_level == "low"
    assert rc.gpu_utilization._risk_level == "low"


@pytest.mark.unit
def test_replica_restart_dry_run_does_not_call_backend(monkeypatch):
    from mcp_server.tools import ray_cluster as rc

    conn = MagicMock(name="conn")
    monkeypatch.setattr(rc, "_get_connection", lambda target=None: conn)

    result = rc.replica_restart(application="app1", deployment="dep1",
                                replica_id="r-1", dry_run=True)
    assert result["dryRun"] is True
    assert result["wouldRestart"] == {"application": "app1", "deployment": "dep1",
                                      "replicaId": "r-1"}
    conn.post_ray.assert_not_called()
