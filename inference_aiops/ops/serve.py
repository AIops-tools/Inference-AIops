"""Ray Serve deployment inventory + guarded lifecycle (read + writes).

The write ops here are exactly the ones the community reports as fragile in
production — scaling, scale-to-zero, and graceful drain — so each captures the
deployment's BEFORE replica count into ``priorState`` for a faithful undo, and
the destructive ones (drain, scale-to-zero) are risk=high with a dry-run preview
at the MCP layer.

Reads are resilient (a dashboard hiccup degrades to an ``error`` field).
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops._util import _seg, as_obj, s

_APPS = "/api/serve/applications/"


def _iter_deployments(apps: dict) -> list[dict]:
    """Flatten Ray Serve's app→deployment map into a list of normalised rows."""
    rows: list[dict] = []
    applications = apps.get("applications", apps) if isinstance(apps, dict) else {}
    if not isinstance(applications, dict):
        return rows
    for app_name, app in applications.items():
        deployments = (app or {}).get("deployments", {}) if isinstance(app, dict) else {}
        for dep_name, dep in (deployments or {}).items():
            dep = dep if isinstance(dep, dict) else {}
            cfg = dep.get("deployment_config", {}) or {}
            replicas = dep.get("replicas", []) or []
            rows.append({
                "application": s(app_name),
                "deployment": s(dep_name),
                "status": s(dep.get("status")),
                "numReplicas": len(replicas) if replicas else cfg.get("num_replicas"),
                "targetReplicas": cfg.get("num_replicas"),
                "replicaStates": [s((r or {}).get("state")) for r in replicas
                                  if isinstance(r, dict)],
            })
    return rows


def list_serve_deployments(conn: Any) -> list[dict]:
    """[READ] All Ray Serve deployments: status, replica count, target."""
    try:
        return _iter_deployments(as_obj(conn.get_ray(_APPS)))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return [{"error": s(exc, 200)}]


def _find(conn: Any, application: str, deployment: str) -> dict:
    for row in _iter_deployments(as_obj(conn.get_ray(_APPS))):
        if row.get("application") == application and row.get("deployment") == deployment:
            return row
    raise KeyError(f"Deployment '{application}/{deployment}' not found.")


def get_deployment_status(conn: Any, application: str, deployment: str) -> dict:
    """[READ] One deployment's status + current/target replica count."""
    try:
        return _find(conn, application, deployment)
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}


def list_replicas(conn: Any, application: str, deployment: str) -> dict:
    """[READ] Replica states for one deployment (running/starting/draining)."""
    try:
        row = _find(conn, application, deployment)
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    return {"application": application, "deployment": deployment,
            "replicaStates": row.get("replicaStates", [])}


def get_autoscale_config(conn: Any, application: str, deployment: str) -> dict:
    """[READ] Autoscale bounds (min/max replicas, target ongoing requests)."""
    try:
        apps = as_obj(conn.get_ray(_APPS))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    applications = apps.get("applications", {})
    dep = ((applications.get(application) or {}).get("deployments", {}) or {}).get(deployment, {})
    cfg = (dep or {}).get("deployment_config", {}) or {}
    auto = cfg.get("autoscaling_config", {}) or {}
    return {
        "application": s(application), "deployment": s(deployment),
        "minReplicas": auto.get("min_replicas"),
        "maxReplicas": auto.get("max_replicas"),
        "targetOngoingRequests": auto.get("target_ongoing_requests"),
    }


# ── writes ───────────────────────────────────────────────────────────────


def _set_replicas(conn: Any, application: str, deployment: str, num: int) -> dict:
    """PUT a new replica count, capturing the prior count for undo/audit."""
    prior = _find(conn, application, deployment).get("numReplicas")
    conn.put_ray(f"{_APPS}{_seg(application)}/deployments/{_seg(deployment)}",
                 json={"num_replicas": num})
    return {"application": s(application), "deployment": s(deployment),
            "numReplicas": num, "priorState": {"numReplicas": prior}}


def scale_replicas_up(conn: Any, application: str, deployment: str, num_replicas: int) -> dict:
    """[WRITE] Raise a deployment's replica count (reversible → prior count)."""
    return {"action": "scale_replicas_up",
            **_set_replicas(conn, application, deployment, num_replicas)}


def scale_replicas_down(conn: Any, application: str, deployment: str, num_replicas: int) -> dict:
    """[WRITE] Lower a deployment's replica count (reversible → prior count)."""
    return {"action": "scale_replicas_down",
            **_set_replicas(conn, application, deployment, num_replicas)}


def scale_to_zero(conn: Any, application: str, deployment: str) -> dict:
    """[WRITE][high] Park a deployment at 0 replicas (reversible → prior count)."""
    return {"action": "scale_to_zero", **_set_replicas(conn, application, deployment, 0)}


def update_autoscale_config(
    conn: Any, application: str, deployment: str,
    min_replicas: int | None = None, max_replicas: int | None = None,
    target_ongoing_requests: float | None = None,
) -> dict:
    """[WRITE] Live-tune autoscale bounds (reversible → prior config)."""
    prior = get_autoscale_config(conn, application, deployment)
    body: dict[str, Any] = {}
    if min_replicas is not None:
        body["min_replicas"] = min_replicas
    if max_replicas is not None:
        body["max_replicas"] = max_replicas
    if target_ongoing_requests is not None:
        body["target_ongoing_requests"] = target_ongoing_requests
    conn.put_ray(f"{_APPS}{_seg(application)}/deployments/{_seg(deployment)}/autoscale", json=body)
    return {"action": "update_autoscale_config", "application": s(application),
            "deployment": s(deployment), "applied": body,
            "priorState": {"minReplicas": prior.get("minReplicas"),
                           "maxReplicas": prior.get("maxReplicas"),
                           "targetOngoingRequests": prior.get("targetOngoingRequests")}}


def drain_replica(conn: Any, application: str, deployment: str, replica_id: str) -> dict:
    """[WRITE][high] Gracefully drain one replica (finish in-flight, take no new)."""
    conn.post_ray(
        f"{_APPS}{_seg(application)}/deployments/{_seg(deployment)}"
        f"/replicas/{_seg(replica_id)}/drain",
        json={},
    )
    return {"action": "drain_replica", "application": s(application),
            "deployment": s(deployment), "replicaId": s(replica_id)}
