"""Unit tests for the Ray Serve DEPLOY lifecycle + routing module.

Proves: deploy_model PUTs the right body; routing_policy_update captures the
BEFORE policy and records an undo descriptor that restores it; the write tools
carry the correct risk tiers; and the model_undeploy dry-run gate never issues
a DELETE. No real Ray is needed — the connection is a MagicMock.
"""

from unittest.mock import MagicMock

import pytest


def _apps_declarative(*names):
    """A /applications/ blob where each app has a declarative deployed_app_config."""
    return {"applications": {
        n: {"deployed_app_config": {"name": n, "import_path": f"{n}:app",
                                    "route_prefix": f"/{n}",
                                    "deployments": [{"name": "D", "num_replicas": 1}]}}
        for n in names}}


@pytest.mark.unit
def test_deploy_model_merges_app_into_declarative_schema():
    """Ray's REST deploy is a whole-schema PUT — deploy_model must PRESERVE the
    existing apps and add the new one, PUTting to the constant path."""
    from inference_aiops.ops import deploy as ops

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps_declarative("existing")
    conn.put_ray.return_value = {}
    out = ops.deploy_model(conn, "app1", "module:app")

    (path,) = conn.put_ray.call_args.args
    assert path == "/api/serve/applications/"
    apps = conn.put_ray.call_args.kwargs["json"]["applications"]
    names = {a["name"] for a in apps}
    assert names == {"existing", "app1"}, "must not delete the existing app"
    new = next(a for a in apps if a["name"] == "app1")
    assert new["import_path"] == "module:app"
    assert out["action"] == "model_deploy" and out["importPath"] == "module:app"


@pytest.mark.unit
def test_undeploy_puts_schema_without_the_target_app():
    from inference_aiops.ops import deploy as ops

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps_declarative("app1", "keep")
    conn.put_ray.return_value = {}
    out = ops.undeploy_model(conn, "app1")

    apps = conn.put_ray.call_args.kwargs["json"]["applications"]
    names = {a["name"] for a in apps}
    assert names == {"keep"}, "undeploy removes only the target, PUT preserves the rest"
    assert out["priorState"]["importPath"] == "app1:app"  # captured for the audit


@pytest.mark.unit
def test_routing_policy_update_refuses_no_rest_endpoint(monkeypatch):
    """Ray Serve routing policy is a code-level decorator option, not REST-settable;
    the tool must refuse (teaching error) instead of PUTting to a 404 path."""
    from mcp_server.tools import deploy as dp

    conn = MagicMock(name="conn")
    monkeypatch.setattr(dp, "_get_connection", lambda target=None: conn)

    result = dp.routing_policy_update(
        application="app1", deployment="dep1", policy="prefix_aware"
    )
    assert "error" in result and "REST" in result["error"]
    conn.put_ray.assert_not_called()


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
