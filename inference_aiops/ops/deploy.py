"""Ray Serve DEPLOY lifecycle (guarded writes).

These are the app-level lifecycle ops: deploying a Serve application from an
import path, tearing it down, and forcing a re-apply of its config.

Ray Serve's REST control plane is DECLARATIVE — the only mutating endpoint is
``PUT /api/serve/applications/`` (the whole ServeDeploySchema) and the only
delete is ``DELETE /api/serve/applications/`` (ALL apps). There are NO per-app or
per-deployment REST endpoints; an earlier version of this module DELETEd
``.../applications/{app}`` and PUT ``.../deployments/{dep}/redeploy`` and
``.../deployments/{dep}/routing`` — all 404 on every real Ray (confirmed against
Ray 2.56). So every op here works by fetching all app configs, changing the set,
and PUTting it back. ``request-routing policy`` is not a REST-settable field at
all, so ``update_routing_policy`` refuses honestly.

The destructive ones — undeploy (removes a whole app) — are risk=high with a
dry-run preview at the MCP layer.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.connection import EngineCapabilityError
from inference_aiops.ops._util import opt_s, s
from inference_aiops.ops.engine import require_control_plane
from inference_aiops.ops.serve import _all_app_configs

_APPS = "/api/serve/applications/"


def deploy_model(conn: Any, application: str, import_path: str) -> dict:
    """[WRITE] Deploy a Serve application from an import path (create/replace).

    Merges the app into the cluster's declarative schema and PUTs the whole set,
    so existing apps are preserved. Ray derives replica counts from the app's own
    config; adjust them afterward with ``scale`` (the deployment names are known
    only once the app materialises, so they cannot be set at deploy time here).
    """
    require_control_plane(conn, "model_deploy")
    others = [cfg for name, cfg in _all_app_configs(conn).items() if name != application]
    new_app = {"name": application, "import_path": import_path,
               "route_prefix": f"/{application}"}
    conn.put_ray(_APPS, json={"applications": [*others, new_app]})
    return {"action": "model_deploy", "application": s(application),
            "importPath": s(import_path)}


def undeploy_model(conn: Any, application: str) -> dict:
    """[WRITE][high] Tear down a whole Serve application (best-effort prior capture).

    Ray has no per-app delete, so we PUT the full schema WITHOUT this app. The
    prior import path/route prefix are captured for the audit trail; the inverse
    (redeploy from that import path) is recorded as the undo descriptor.
    """
    require_control_plane(conn, "model_undeploy")
    prior: dict[str, Any] = {"application": s(application)}
    configs = _all_app_configs(conn)
    target = configs.get(application)
    if isinstance(target, dict):
        prior["importPath"] = opt_s(target.get("import_path"))
        prior["routePrefix"] = opt_s(target.get("route_prefix"))
    conn.put_ray(_APPS, json={"applications": [
        cfg for name, cfg in configs.items() if name != application]})
    return {"action": "model_undeploy", "application": s(application), "priorState": prior}


def redeploy_deployment(conn: Any, application: str, deployment: str) -> dict:
    """[WRITE][high] Re-apply an application's declarative config (force reconcile).

    Ray exposes no per-deployment redeploy endpoint; re-applying config is an
    app-level operation — we PUT the app's current config back unchanged, which
    makes the controller reconcile it. The ``deployment`` argument is retained
    for the audit message but the reconcile is app-scoped.
    """
    require_control_plane(conn, "deployment_redeploy")
    configs = _all_app_configs(conn)
    if application not in configs:
        raise EngineCapabilityError(
            f"Application '{application}' is not deployed on this Ray cluster, "
            f"so there is nothing to redeploy.")
    conn.put_ray(_APPS, json={"applications": list(configs.values())})
    return {"action": "deployment_redeploy", "application": s(application),
            "deployment": s(deployment)}


def update_routing_policy(conn: Any, application: str, deployment: str, policy: str) -> dict:
    """[WRITE] Switch a deployment's request-routing policy — NOT available via REST.

    Ray Serve's request router is not a per-deployment field settable over the
    Dashboard REST API (the old ``.../deployments/{dep}/routing`` PUT 404s). It is
    configured in code on the ``@serve.deployment`` decorator. Rather than invent
    an endpoint, refuse with a teaching error.
    """
    require_control_plane(conn, "routing_policy_update")
    raise EngineCapabilityError(
        "Ray Serve's request-routing policy is not settable over the Dashboard "
        "REST API — it is a code-level option on the @serve.deployment decorator "
        f"(there is no REST endpoint for it). Cannot switch '{application}/"
        f"{deployment}' to '{policy}' from here; change it in the deployment's "
        "source and redeploy.")
