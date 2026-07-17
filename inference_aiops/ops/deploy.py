"""Ray Serve DEPLOY lifecycle + request routing (guarded writes).

These are the app-level lifecycle ops: deploying a Serve application from an
import path, tearing it down, forcing a redeploy to apply new config, and
switching a deployment's request-routing policy (round-robin vs prefix-aware /
session-affinity — the knob that decides prefix-cache locality across replicas).

The destructive ones — undeploy (removes a whole app) and redeploy (re-applies
config and can drop unfinished requests) — are risk=high with a dry-run preview
at the MCP layer. ``update_routing_policy`` captures the BEFORE policy into
``priorState`` for a faithful undo. Prior-state capture is best-effort: a
dashboard hiccup must not block the write.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops._util import _seg, as_obj, s
from inference_aiops.ops.engine import require_control_plane

_APPS = "/api/serve/applications/"


def deploy_model(conn: Any, application: str, import_path: str, num_replicas: int = 1) -> dict:
    """[WRITE] Deploy a Serve application from an import path (create/replace)."""
    require_control_plane(conn, "model_deploy")
    conn.put_ray(
        _APPS,
        json={"name": application, "import_path": import_path, "num_replicas": num_replicas},
    )
    return {"action": "model_deploy", "application": s(application), "importPath": s(import_path)}


def undeploy_model(conn: Any, application: str) -> dict:
    """[WRITE][high] Tear down a whole Serve application (best-effort prior capture)."""
    require_control_plane(conn, "model_undeploy")
    prior: dict[str, Any] = {"application": s(application)}
    try:
        apps = as_obj(conn.get_ray(_APPS)).get("applications", {}) or {}
        app = apps.get(application) or {}
        if isinstance(app, dict):
            prior["importPath"] = s(app.get("import_path")) if app.get("import_path") else None
            prior["routePrefix"] = s(app.get("route_prefix")) if app.get("route_prefix") else None
    except Exception:  # noqa: BLE001 — prior capture is best-effort, never block the write
        pass
    conn.delete_ray(f"{_APPS}{_seg(application)}")
    return {"action": "model_undeploy", "application": s(application), "priorState": prior}


def redeploy_deployment(conn: Any, application: str, deployment: str) -> dict:
    """[WRITE][high] Force a deployment to re-apply new config (can drop in-flight)."""
    require_control_plane(conn, "deployment_redeploy")
    conn.put_ray(f"{_APPS}{_seg(application)}/deployments/{_seg(deployment)}/redeploy", json={})
    return {"action": "deployment_redeploy", "application": s(application),
            "deployment": s(deployment)}


def _current_routing_policy(conn: Any, application: str, deployment: str) -> Any:
    """Read a deployment's current routing policy from the Serve app map."""
    apps = as_obj(conn.get_ray(_APPS)).get("applications", {}) or {}
    dep = ((apps.get(application) or {}).get("deployments", {}) or {}).get(deployment, {})
    cfg = (dep or {}).get("deployment_config", {}) or {}
    return cfg.get("routing_policy")


def update_routing_policy(conn: Any, application: str, deployment: str, policy: str) -> dict:
    """[WRITE] Switch a deployment's routing policy (reversible → prior policy).

    ``policy`` is e.g. ``prefix_aware`` / ``round_robin`` / ``session_affinity``.
    """
    require_control_plane(conn, "routing_policy_update")
    try:
        prior = _current_routing_policy(conn, application, deployment)
    except Exception:  # noqa: BLE001 — prior capture is best-effort
        prior = None
    conn.put_ray(f"{_APPS}{_seg(application)}/deployments/{_seg(deployment)}/routing",
                 json={"policy": policy})
    return {"action": "routing_policy_update", "application": s(application),
            "deployment": s(deployment), "policy": s(policy),
            "priorState": {"policy": s(prior) if prior is not None else None}}
