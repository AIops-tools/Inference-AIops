"""Ray dashboard-status + per-node GPU utilisation + replica-restart write.

Complements test_ray_cluster: proves the Serve controller health rollup, the
/api/nodes GPU normalisation (including nested shapes), and that restart_replica
posts to the correctly-encoded restart path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.ops import ray_cluster as ops


@pytest.mark.unit
def test_dashboard_status_healthy_when_all_running():
    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {
        "applications": {
            "a": {"status": "RUNNING", "deployments": {"d1": {}, "d2": {}}},
            "b": {"status": "RUNNING", "deployments": {"d3": {}}},
        }
    }
    out = ops.get_dashboard_status(conn)
    assert out["serveController"] == "HEALTHY"
    assert out["appCount"] == 2 and out["deploymentCount"] == 3


@pytest.mark.unit
def test_dashboard_status_degraded_and_empty():
    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"applications": {"a": {"status": "DEPLOY_FAILED",
                                                        "deployments": {}}}}
    assert ops.get_dashboard_status(conn)["serveController"] == "DEGRADED"

    conn.get_ray.return_value = {"applications": {}}
    assert ops.get_dashboard_status(conn)["serveController"] == "NO_APPLICATIONS"


@pytest.mark.unit
def test_dashboard_status_read_failure_degrades():
    conn = MagicMock(name="conn")
    conn.get_ray.side_effect = RuntimeError("boom")
    assert "error" in ops.get_dashboard_status(conn)


@pytest.mark.unit
def test_gpu_utilization_normalises_nodes_and_averages():
    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {
        "data": {"summary": [
            {"nodeId": "node-1", "gpus": [
                {"utilizationGpu": 80, "memoryUsed": 10, "memoryTotal": 40},
                {"utilizationGpu": 60, "memoryUsed": 20, "memoryTotal": 40},
            ]},
            {"ip": "10.0.0.9", "gpus": []},
        ]}
    }
    rows = ops.get_gpu_utilization(conn)
    assert rows[0]["nodeId"] == "node-1"
    assert rows[0]["gpuCount"] == 2
    assert rows[0]["gpuUtilPercent"] == 70.0  # (80 + 60) / 2
    assert rows[0]["gpuMemUsedBytes"] == 30 and rows[0]["gpuMemTotalBytes"] == 80
    assert rows[1]["gpuCount"] == 0 and rows[1]["gpuUtilPercent"] is None


@pytest.mark.unit
def test_gpu_utilization_read_failure_degrades():
    conn = MagicMock(name="conn")
    conn.get_ray.side_effect = RuntimeError("nodes 500")
    rows = ops.get_gpu_utilization(conn)
    assert "error" in rows[0]


@pytest.mark.unit
def test_cancel_job_posts_stop_and_returns_action():
    conn = MagicMock(name="conn")
    out = ops.cancel_job(conn, "raysubmit_9")
    assert out == {"action": "ray_job_cancel", "jobId": "raysubmit_9"}
    (path,) = conn.post_ray.call_args.args
    assert path == "/api/jobs/raysubmit_9/stop"


@pytest.mark.unit
def test_restart_replica_posts_encoded_restart_path():
    conn = MagicMock(name="conn")
    out = ops.restart_replica(conn, "llm", "VLLMDeployment", "r-7")
    assert out["action"] == "replica_restart" and out["replicaId"] == "r-7"
    (path,) = conn.post_ray.call_args.args
    assert path == "/api/serve/applications/llm/deployments/VLLMDeployment/replicas/r-7/restart"
