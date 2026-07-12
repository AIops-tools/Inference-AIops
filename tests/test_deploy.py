"""Unit tests for the Ray Serve DEPLOY lifecycle + routing module.

Proves: deploy_model PUTs the right body; routing_policy_update captures the
BEFORE policy and records an undo descriptor that restores it; the write tools
carry the correct risk tiers; and the model_undeploy dry-run gate never issues
a DELETE. No real Ray is needed — the connection is a MagicMock.
"""

from unittest.mock import MagicMock

import pytest


def _apps_routing(policy="round_robin"):
    return {"applications": {"app1": {"deployments": {"dep1": {
        "deployment_config": {"routing_policy": policy},
    }}}}}


@pytest.mark.unit
def test_deploy_model_puts_expected_body():
    from inference_aiops.ops import deploy as ops

    conn = MagicMock(name="conn")
    conn.put_ray.return_value = {}
    out = ops.deploy_model(conn, "app1", "module:app", num_replicas=2)

    conn.put_ray.assert_called_once_with(
        "/api/serve/applications/",
        json={"name": "app1", "import_path": "module:app", "num_replicas": 2},
    )
    assert out["action"] == "model_deploy"
    assert out["application"] == "app1"
    assert out["importPath"] == "module:app"


@pytest.mark.unit
def test_routing_policy_update_captures_prior_and_records_undo(monkeypatch):
    import inference_aiops.governance.undo as undo_mod
    from mcp_server.tools import deploy as dp

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps_routing("round_robin")
    conn.put_ray.return_value = {}
    monkeypatch.setattr(dp, "_get_connection", lambda target=None: conn)

    recorded = {}

    class _Store:
        def record(self, *, skill, tool, undo_descriptor, orig_params):
            recorded["d"] = undo_descriptor
            return "undo-r"

    monkeypatch.setattr(undo_mod, "get_undo_store", lambda: _Store())

    result = dp.routing_policy_update(
        application="app1", deployment="dep1", policy="prefix_aware"
    )
    assert result["priorState"]["policy"] == "round_robin"
    assert recorded["d"]["tool"] == "routing_policy_update"
    assert recorded["d"]["params"]["policy"] == "round_robin"  # restore prior
    assert result.get("_undo_id") == "undo-r"


@pytest.mark.unit
def test_deploy_write_tools_have_correct_risk_tiers():
    from mcp_server.tools import deploy as dp

    assert dp.model_undeploy._risk_level == "high"
    assert dp.deployment_redeploy._risk_level == "high"
    assert dp.model_deploy._risk_level == "medium"
    assert dp.routing_policy_update._risk_level == "medium"


@pytest.mark.unit
def test_model_undeploy_dry_run_does_not_delete(monkeypatch):
    from mcp_server.tools import deploy as dp

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"applications": {"app1": {}}}
    monkeypatch.setattr(dp, "_get_connection", lambda target=None: conn)

    result = dp.model_undeploy(application="app1", dry_run=True)
    assert result["dryRun"] is True
    assert result["wouldUndeploy"]["application"] == "app1"
    conn.delete_ray.assert_not_called()
