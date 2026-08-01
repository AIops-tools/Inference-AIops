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

from inference_aiops.connection import EngineCapabilityError, InferenceApiError
from inference_aiops.ops._util import as_obj, opt_s, s
from inference_aiops.ops.engine import require_control_plane

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
                "status": opt_s(dep.get("status")),
                "numReplicas": len(replicas) if replicas else cfg.get("num_replicas"),
                "targetReplicas": cfg.get("num_replicas"),
                "replicaStates": [opt_s((r or {}).get("state")) for r in replicas
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
#
# Ray Serve's REST control plane is DECLARATIVE: the only mutating endpoint is
# ``PUT /api/serve/applications/``, which replaces the whole ServeDeploySchema.
# There are NO per-deployment or per-replica REST endpoints (an earlier version
# of this module PUT to ``.../deployments/{name}`` and POSTed to
# ``.../replicas/{id}/drain`` — both 404 on every real Ray, confirmed against
# Ray 2.56). To change one deployment we fetch every app's declarative config,
# patch the target, and PUT them all back.


def _all_app_configs(conn: Any) -> dict[str, dict]:
    """Return ``{app_name: declarative_config_copy}`` for every deployed app.

    The whole-schema PUT (the only mutating Serve endpoint) replaces EVERY app,
    so any write built on it must be able to reconstruct every app or it would
    silently delete the ones it dropped. An app deployed imperatively
    (``serve.run``) has no ``deployed_app_config`` and cannot be reconstructed —
    so we refuse the whole operation rather than delete a bystander. Copies are
    returned so the fetched status blob is never mutated.
    """
    apps = as_obj(conn.get_ray(_APPS)).get("applications", {})
    out: dict[str, dict] = {}
    for name, app in (apps if isinstance(apps, dict) else {}).items():
        cfg = (app or {}).get("deployed_app_config")
        if not isinstance(cfg, dict):
            raise EngineCapabilityError(
                f"Application '{name}' has no Serve config — it was deployed "
                f"imperatively (serve.run), not from a Serve config. Ray's REST "
                f"control plane cannot modify it, and any single-app change means "
                f"re-submitting all apps, which would delete '{name}'. Refusing. "
                f"Redeploy it declaratively to manage it here.")
        cfg = dict(cfg)
        cfg["deployments"] = [dict(d) for d in (cfg.get("deployments") or [])]
        out[name] = cfg
    return out


def _app_configs_for_put(conn: Any, application: str) -> tuple[list[dict], dict]:
    """Full app-config list plus the target's config (the same dict in the list).

    The caller patches ``target_cfg`` in place, then PUTs the returned list.
    """
    configs = _all_app_configs(conn)
    if application not in configs:
        raise InferenceApiError(
            f"Application '{application}' not found on this Ray cluster.",
            status_code=404, path=_APPS)
    return list(configs.values()), configs[application]


def _deployment_entry(target_cfg: dict, deployment: str) -> dict:
    """Return the deployment's entry in an app config, creating it if implicit.

    Ray omits deployments that run fully on defaults, so a first-time scale of
    such a deployment has no entry to patch — add one keyed by name.
    """
    for entry in target_cfg["deployments"]:
        if entry.get("name") == deployment:
            return entry
    entry = {"name": deployment}
    target_cfg["deployments"].append(entry)
    return entry


def _set_replicas(conn: Any, application: str, deployment: str, num: int) -> dict:
    """Set a fixed replica count via the declarative config, capturing prior."""
    require_control_plane(conn, "scale_replicas")
    prior = _find(conn, application, deployment).get("numReplicas")
    configs, target_cfg = _app_configs_for_put(conn, application)
    entry = _deployment_entry(target_cfg, deployment)
    # num_replicas and autoscaling_config are mutually exclusive in Ray Serve;
    # an explicit scale pins a fixed count, so drop any autoscaling on this dep.
    entry.pop("autoscaling_config", None)
    entry["num_replicas"] = num
    conn.put_ray(_APPS, json={"applications": configs})
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
    require_control_plane(conn, "update_autoscale_config")
    prior = get_autoscale_config(conn, application, deployment)
    body: dict[str, Any] = {}
    if min_replicas is not None:
        body["min_replicas"] = min_replicas
    if max_replicas is not None:
        body["max_replicas"] = max_replicas
    if target_ongoing_requests is not None:
        body["target_ongoing_requests"] = target_ongoing_requests
    configs, target_cfg = _app_configs_for_put(conn, application)
    entry = _deployment_entry(target_cfg, deployment)
    merged = dict(entry.get("autoscaling_config") or {})
    merged.update(body)
    # autoscaling_config and a fixed num_replicas are mutually exclusive in Ray.
    entry.pop("num_replicas", None)
    entry["autoscaling_config"] = merged
    conn.put_ray(_APPS, json={"applications": configs})
    return {"action": "update_autoscale_config", "application": s(application),
            "deployment": s(deployment), "applied": body,
            "priorState": {"minReplicas": prior.get("minReplicas"),
                           "maxReplicas": prior.get("maxReplicas"),
                           "targetOngoingRequests": prior.get("targetOngoingRequests")}}


def drain_replica(conn: Any, application: str, deployment: str, replica_id: str) -> dict:
    """[WRITE][high] Gracefully drain one replica (finish in-flight, take no new).

    Ray Serve's REST control plane exposes NO per-replica drain — there is no
    endpoint to reach one replica over HTTP (the old ``.../replicas/{id}/drain``
    POST 404s on every real Ray). Individual-replica draining is a Python-API
    capability only. Rather than invent an endpoint, refuse with a teaching
    error: scaling the deployment down lets the controller drain surplus
    replicas gracefully (``graceful_shutdown_timeout_s``).
    """
    require_control_plane(conn, "drain_replica")
    raise EngineCapabilityError(
        "Ray Serve's REST API cannot drain an individual replica — it has no "
        "per-replica endpoint (only the whole-cluster declarative config). To "
        f"retire replica '{replica_id}' of '{application}/{deployment}', scale "
        "the deployment down: the controller drains the surplus replicas "
        "gracefully. Per-replica drain is available only via Ray's Python API.")
