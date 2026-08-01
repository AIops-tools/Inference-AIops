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

from inference_aiops.connection import EngineCapabilityError
from inference_aiops.ops._util import _seg, as_list, as_obj, opt_s, s
from inference_aiops.ops.engine import require_control_plane

_CLUSTER = "/api/cluster_status"
_APPS = "/api/serve/applications/"
_JOBS = "/api/jobs/"
_NODES = "/nodes"  # Ray dropped the /api prefix on this route; /api/nodes 404s


def _agg_resource(usage_by_node: dict, resource: str) -> tuple[float | None, float | None]:
    """Sum (used, total) of one resource across nodes, or (None, None) if absent.

    Ray reports per-node usage as ``{node: {ResourceName: [used, total]}}``. A
    resource no node reports (e.g. GPU on a CPU-only cluster) yields (None, None)
    — unknown, not a fabricated zero.
    """
    used = total = 0.0
    seen = False
    for node in usage_by_node.values():
        pair = node.get(resource) if isinstance(node, dict) else None
        if (isinstance(pair, list) and len(pair) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in pair)):
            used += pair[0]
            total += pair[1]
            seen = True
    return (round(used, 3), round(total, 3)) if seen else (None, None)


def get_cluster_resources(conn: Any) -> dict:
    """[READ] Cluster-wide CPU/GPU capacity + headroom (from cluster_status).

    Current Ray exposes resource totals/usage only under
    ``clusterStatus.loadMetricsReport.usageByNode``. Older Ray put
    ``clusterResources``/``availableResources`` at the ``data`` level — that shape
    is gone, which is why reading it returned all-null on every live cluster.
    """
    try:
        status = as_obj(conn.get_ray(_CLUSTER))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    data = as_obj(status.get("data", status))
    lmr = as_obj(as_obj(data.get("clusterStatus")).get("loadMetricsReport"))
    usage_by_node = as_obj(lmr.get("usageByNode"))
    used_cpu, total_cpu = _agg_resource(usage_by_node, "CPU")
    used_gpu, total_gpu = _agg_resource(usage_by_node, "GPU")
    pg = lmr.get("pgDemand")
    return {
        "totalCpu": total_cpu,
        "availableCpu": round(total_cpu - used_cpu, 3) if total_cpu is not None else None,
        "totalGpu": total_gpu,
        "availableGpu": round(total_gpu - used_gpu, 3) if total_gpu is not None else None,
        "pendingPlacementGroups": len(pg) if isinstance(pg, list) else None,
    }


def get_dashboard_status(conn: Any) -> dict:
    """[READ] Serve controller health + app/deployment counts from the applications map."""
    try:
        apps = as_obj(conn.get_ray(_APPS))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    applications = apps.get("applications", {})
    applications = applications if isinstance(applications, dict) else {}
    statuses = [opt_s((app or {}).get("status")) for app in applications.values()
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
        "jobId": opt_s(job.get("job_id") or job.get("submission_id") or job.get("jobId")),
        "status": opt_s(job.get("status")),
        "entrypoint": opt_s(job.get("entrypoint")),
        "startTime": job.get("start_time") or job.get("startTime"),
    }


def list_jobs(conn: Any, limit: int = 100) -> dict:
    """[READ] Submitted Ray jobs: id, status, entrypoint, start time.

    A long-lived Ray cluster accumulates thousands of finished jobs, so the list
    is capped by ``limit`` and the cap announces itself::

        {"jobs": [...], "returned": 100, "limit": 100, "truncated": true}

    A bare list cannot say "there is more" — the consumer has to infer it from
    the length happening to equal the limit, and a smaller local model reads that
    coincidence as "that is every job". One row past the limit is kept while
    slicing so ``truncated`` is *measured* rather than guessed.
    """
    requested = int(limit)
    try:
        raw = as_list(conn.get_ray(_JOBS))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    truncated = len(raw) > requested
    jobs = [_job_row(job) for job in raw[:requested]]
    return {
        "jobs": jobs,
        "returned": len(jobs),
        "limit": requested,
        "truncated": truncated,
    }


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
        "nodeId": opt_s(node.get("nodeId") or raylet.get("nodeId") or node.get("ip")),
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
    """[READ] Per-node GPU count, utilisation %, and memory (from the nodes summary)."""
    try:
        return [_gpu_row(node) for node in
                _node_rows(conn.get_ray(_NODES, params={"view": "summary"}))]
    except Exception as exc:  # noqa: BLE001 — report as partial
        return [{"error": s(exc, 200)}]


# ── writes ───────────────────────────────────────────────────────────────


def cancel_job(conn: Any, job_id: str) -> dict:
    """[WRITE] Stop a submitted/running Ray job."""
    require_control_plane(conn, "ray_job_cancel")
    conn.post_ray(f"{_JOBS}{_seg(job_id)}/stop", json={})
    return {"action": "ray_job_cancel", "jobId": s(job_id)}


def restart_replica(conn: Any, application: str, deployment: str, replica_id: str) -> dict:
    """[WRITE][high] Restart one wedged Serve replica — NOT available over Ray's REST API.

    Like per-replica drain, Ray Serve exposes no per-replica restart endpoint over
    REST (the old ``.../replicas/{id}/restart`` POST 404s on every real Ray). The
    controller restarts unhealthy replicas on its own; to force a cycle, scale the
    deployment down and back up, or redeploy. Refuse rather than invent an endpoint.
    """
    require_control_plane(conn, "replica_restart")
    raise EngineCapabilityError(
        "Ray Serve's REST API cannot restart an individual replica — it has no "
        "per-replica endpoint. The controller already respawns unhealthy replicas; "
        f"to force-cycle replica '{replica_id}' of '{application}/{deployment}', "
        "scale the deployment down then up, or redeploy.")
