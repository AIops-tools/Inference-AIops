"""Ray cluster / jobs / GPU layer (read + guarded writes).

These reads answer the "is the fleet healthy and fed" questions that sit one
level below Ray Serve: cluster-wide CPU/GPU capacity + headroom, the Serve
controller's health, submitted jobs, and per-node GPU utilisation. The writes
are the two blunt recovery levers — cancel a runaway job (risk=medium) and
restart a wedged replica (risk=high, with a dry-run preview at the MCP layer).

All reads are resilient: a dashboard hiccup degrades to an ``error`` field.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops._util import as_list, as_obj, s

_CLUSTER = "/api/cluster_status"
_APPS = "/api/serve/applications/"
_JOBS = "/api/jobs/"
_NODES = "/api/nodes"


def _num(source: Any, *keys: str) -> float | None:
    """First numeric value found under any of ``keys`` in ``source`` (None if absent)."""
    if not isinstance(source, dict):
        return None
    for key in keys:
        val = source.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def get_cluster_resources(conn: Any) -> dict:
    """[READ] Cluster-wide CPU/GPU capacity + headroom (best-effort from cluster_status)."""
    try:
        status = as_obj(conn.get_ray(_CLUSTER))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    data = as_obj(status.get("data", status))
    total = as_obj(data.get("clusterResources") or data.get("totalResources"))
    avail = as_obj(data.get("availableResources"))
    pending = data.get("pendingPlacementGroups")
    pending_count = (len(pending) if isinstance(pending, list)
                     else _num(data, "pendingPlacementGroups"))
    return {
        "totalCpu": _num(total, "CPU"),
        "availableCpu": _num(avail, "CPU"),
        "totalGpu": _num(total, "GPU"),
        "availableGpu": _num(avail, "GPU"),
        "pendingPlacementGroups": pending_count,
    }


def get_dashboard_status(conn: Any) -> dict:
    """[READ] Serve controller health + app/deployment counts from the applications map."""
    try:
        apps = as_obj(conn.get_ray(_APPS))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    applications = apps.get("applications", {})
    applications = applications if isinstance(applications, dict) else {}
    statuses = [s((app or {}).get("status")) for app in applications.values()
                if isinstance(app, dict)]
    deployment_count = sum(
        len((app or {}).get("deployments", {}) or {})
        for app in applications.values() if isinstance(app, dict)
    )
    if not statuses:
        controller = "NO_APPLICATIONS"
    elif all(st == "RUNNING" for st in statuses):
        controller = "HEALTHY"
    else:
        controller = "DEGRADED"
    return {
        "serveController": controller,
        "appCount": len(applications),
        "deploymentCount": deployment_count,
    }


def _job_row(job: dict) -> dict:
    return {
        "jobId": s(job.get("job_id") or job.get("submission_id") or job.get("jobId")),
        "status": s(job.get("status")),
        "entrypoint": s(job.get("entrypoint")),
        "startTime": job.get("start_time") or job.get("startTime"),
    }


def list_jobs(conn: Any) -> list[dict]:
    """[READ] Submitted Ray jobs: id, status, entrypoint, start time."""
    try:
        return [_job_row(job) for job in as_list(conn.get_ray(_JOBS))]
    except Exception as exc:  # noqa: BLE001 — report as partial
        return [{"error": s(exc, 200)}]


def _gpu_row(node: dict) -> dict:
    gpus = node.get("gpus") or []
    gpus = [g for g in gpus if isinstance(g, dict)]
    utils = [g.get("utilizationGpu") for g in gpus
             if isinstance(g.get("utilizationGpu"), (int, float))]
    util = round(sum(utils) / len(utils), 2) if utils else None
    mem_used = sum(g.get("memoryUsed", 0) for g in gpus
                   if isinstance(g.get("memoryUsed"), (int, float))) or None
    mem_total = sum(g.get("memoryTotal", 0) for g in gpus
                    if isinstance(g.get("memoryTotal"), (int, float))) or None
    raylet = as_obj(node.get("raylet"))
    return {
        "nodeId": s(node.get("nodeId") or raylet.get("nodeId") or node.get("ip")),
        "gpuCount": len(gpus),
        "gpuUtilPercent": util,
        "gpuMemUsedBytes": mem_used,
        "gpuMemTotalBytes": mem_total,
    }


def _node_rows(payload: Any) -> list[dict]:
    """Extract the node list from /api/nodes' (nested) shapes."""
    if isinstance(payload, list):
        return [n for n in payload if isinstance(n, dict)]
    obj = as_obj(payload)
    data = as_obj(obj.get("data", obj))
    nodes = data.get("summary") or data.get("nodes") or obj.get("nodes") or []
    return [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []


def get_gpu_utilization(conn: Any) -> list[dict]:
    """[READ] Per-node GPU count, utilisation %, and memory (best-effort from /api/nodes)."""
    try:
        return [_gpu_row(node) for node in _node_rows(conn.get_ray(_NODES))]
    except Exception as exc:  # noqa: BLE001 — report as partial
        return [{"error": s(exc, 200)}]


# ── writes ───────────────────────────────────────────────────────────────


def cancel_job(conn: Any, job_id: str) -> dict:
    """[WRITE] Stop a submitted/running Ray job."""
    conn.post_ray(f"{_JOBS}{job_id}/stop", json={})
    return {"action": "ray_job_cancel", "jobId": s(job_id)}


def restart_replica(conn: Any, application: str, deployment: str, replica_id: str) -> dict:
    """[WRITE][high] Restart one wedged Serve replica (kills + respawns the actor)."""
    conn.post_ray(
        f"{_APPS}{application}/deployments/{deployment}/replicas/{replica_id}/restart",
        json={},
    )
    return {"action": "replica_restart", "application": s(application),
            "deployment": s(deployment), "replicaId": s(replica_id)}
