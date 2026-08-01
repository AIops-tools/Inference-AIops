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
    """Current-Ray shape (verified against 2.56): totals + usage live under
    clusterStatus.loadMetricsReport.usageByNode as {res: [used, total]}, summed
    across nodes. availableCpu = total - used."""
    from inference_aiops.ops import ray_cluster as ops

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"data": {"clusterStatus": {"loadMetricsReport": {
        "usageByNode": {
            "n1": {"CPU": [16.0, 32.0], "GPU": [2.0, 4.0]},
            "n2": {"CPU": [16.0, 32.0], "GPU": [3.0, 4.0]},
        },
        "pgDemand": [{"bundle": 1}, {"bundle": 2}],
    }}}}
    out = ops.get_cluster_resources(conn)
    assert out["totalCpu"] == 64.0 and out["availableCpu"] == 32.0  # 64 total - 32 used
    assert out["totalGpu"] == 8.0 and out["availableGpu"] == 3.0    # 8 total - 5 used
    assert out["pendingPlacementGroups"] == 2

    conn.get_ray.side_effect = RuntimeError("dashboard down")
    assert "error" in ops.get_cluster_resources(conn)


@pytest.mark.unit
def test_get_cluster_resources_gpuless_cluster_is_null_not_zero():
    """A CPU-only cluster reports no GPU key at all — GPU must be null (unknown),
    never a fabricated 0, while CPU is real."""
    from inference_aiops.ops import ray_cluster as ops

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"data": {"clusterStatus": {"loadMetricsReport": {
        "usageByNode": {"n1": {"CPU": [0.4, 2.0]}}}}}}
    out = ops.get_cluster_resources(conn)
    assert out["totalCpu"] == 2.0 and out["availableCpu"] == 1.6
    assert out["totalGpu"] is None and out["availableGpu"] is None


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
    assert result["available"] is False
    assert "per-replica" in result["reason"]
    conn.post_ray.assert_not_called()
