"""Ray cluster / jobs / GPU MCP tools (read + guarded writes).

Cancel-job is risk=medium; replica-restart is the destructive recovery lever —
risk=high with a dry_run preview.
"""

from typing import Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import ray_cluster as ops
from mcp_server._shared import _get_connection, mcp, tool_errors

# ── reads ────────────────────────────────────────────────────────────────


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def ray_cluster_resources(target: Optional[str] = None) -> dict:
    """[READ] Cluster-wide CPU/GPU capacity + headroom + pending placement groups.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_cluster_resources(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def ray_dashboard_status(target: Optional[str] = None) -> dict:
    """[READ] Serve controller health + app/deployment counts.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_dashboard_status(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def ray_job_list(target: Optional[str] = None, limit: int = 100) -> dict:
    """[READ] Submitted Ray jobs: id, status, entrypoint, start time.

    Returns ``{"jobs": [...], "returned": N, "limit": L, "truncated": bool}``.
    When ``truncated`` is true there are more jobs than were returned — re-run
    with a higher limit rather than treating the result as the whole history.

    Args:
        target: Inference target name from config; omit for the default.
        limit: Maximum job rows to return. Default 100.
    """
    return ops.list_jobs(_get_connection(target), limit=limit)


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def gpu_utilization(target: Optional[str] = None) -> list:
    """[READ] Per-node GPU count, utilisation %, and memory used/total.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_gpu_utilization(_get_connection(target))


# ── writes ───────────────────────────────────────────────────────────────


@mcp.tool()
@governed_tool(risk_level="medium")
@tool_errors("dict")
def ray_job_cancel(job_id: str, target: Optional[str] = None) -> dict:
    """[WRITE][risk=medium] Stop a submitted/running Ray job.

    Args:
        job_id: Job id (from ray_job_list).
        target: Inference target name from config; omit for the default.
    """
    return ops.cancel_job(_get_connection(target), job_id)


@mcp.tool()
@governed_tool(risk_level="high")
@tool_errors("dict")
def replica_restart(
    application: str, deployment: str, replica_id: str,
    dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Restart one wedged Serve replica (kills + respawns the actor).

    Drops the replica's in-flight requests — pass dry_run=True to preview.
    Requires an approver (INFERENCE_AUDIT_APPROVED_BY).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        replica_id: Replica id (from replica_list).
        dry_run: If True, preview without restarting.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True, "wouldRestart": {"application": application,
                                                 "deployment": deployment,
                                                 "replicaId": replica_id}}
    return ops.restart_replica(conn, application, deployment, replica_id)
