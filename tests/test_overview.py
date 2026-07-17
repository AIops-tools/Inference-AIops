"""One-shot fleet overview: folds Serve deployments + vLLM queue signal.

Proves the summary sums replica counts and surfaces backpressure, and that a
failing sub-call degrades to a partial summary with an ``errors`` list rather
than raising.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.ops import overview as ops

_APPS = {
    "applications": {
        "llm": {"deployments": {
            "d1": {"status": "HEALTHY", "replicas": [{"state": "RUNNING"},
                                                     {"state": "RUNNING"}],
                   "deployment_config": {"num_replicas": 2}},
            "d2": {"status": "HEALTHY", "replicas": [{"state": "RUNNING"}],
                   "deployment_config": {"num_replicas": 1}},
        }}
    }
}


@pytest.mark.unit
def test_fleet_overview_sums_replicas_and_reads_queue():
    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _APPS
    conn.vllm_metrics.return_value = {
        "vllm:num_requests_waiting": [{"labels": {}, "value": 4.0}],
        "vllm:num_requests_running": [{"labels": {}, "value": 3.0}],
    }
    out = ops.fleet_overview(conn)
    assert out["deployments"] == 2
    assert out["totalReplicas"] == 3
    assert out["numWaiting"] == 4.0 and out["numRunning"] == 3.0
    assert out["backpressure"] is True
    assert out["errors"] == []


@pytest.mark.unit
def test_fleet_overview_degrades_on_serve_and_metrics_failure():
    conn = MagicMock(name="conn")
    conn.get_ray.side_effect = RuntimeError("dashboard down")
    conn.vllm_metrics.side_effect = RuntimeError("metrics down")
    out = ops.fleet_overview(conn)
    assert out["deployments"] == 0 and out["totalReplicas"] == 0
    assert any("serve:" in e for e in out["errors"])
    assert any("metrics:" in e for e in out["errors"])
