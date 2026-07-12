"""Ray Serve deployment MCP tools (read + guarded writes).

Scale-down / scale-to-zero / drain are the fragile prod ops — risk=high with a
dry_run preview. Reversible writes record an undo capturing the prior replica
count / autoscale config.
"""

from typing import Any, Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import serve as ops
from mcp_server._shared import _get_connection, mcp, tool_errors


def _replica_undo(params: dict[str, Any], result: Any) -> Optional[dict]:
    """Inverse of a scale op: restore the captured prior replica count."""
    if not isinstance(result, dict):
        return None
    prior = (result.get("priorState") or {}).get("numReplicas")
    if prior is None:
        return None
    return {"tool": "scale_replicas_up",
            "params": {"application": params.get("application"),
                       "deployment": params.get("deployment"), "num_replicas": prior},
            "note": "Restore the deployment's prior replica count."}


def _autoscale_undo(params: dict[str, Any], result: Any) -> Optional[dict]:
    """Inverse of update_autoscale_config: restore the captured prior bounds."""
    if not isinstance(result, dict):
        return None
    prior = result.get("priorState") or {}
    if not any(v is not None for v in prior.values()):
        return None
    return {"tool": "update_autoscale_config",
            "params": {"application": params.get("application"),
                       "deployment": params.get("deployment"),
                       "min_replicas": prior.get("minReplicas"),
                       "max_replicas": prior.get("maxReplicas"),
                       "target_ongoing_requests": prior.get("targetOngoingRequests")},
            "note": "Restore the deployment's prior autoscale bounds."}


# ── reads ────────────────────────────────────────────────────────────────


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def serve_deployment_list(target: Optional[str] = None) -> list:
    """[READ] All Ray Serve deployments: status, replica count, target.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.list_serve_deployments(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def deployment_status(application: str, deployment: str, target: Optional[str] = None) -> dict:
    """[READ] One deployment's status + current/target replica count.

    Args:
        application: Serve application name (from serve_deployment_list).
        deployment: Deployment name within the application.
        target: Inference target name from config; omit for the default.
    """
    return ops.get_deployment_status(_get_connection(target), application, deployment)


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def replica_list(application: str, deployment: str, target: Optional[str] = None) -> dict:
    """[READ] Replica states for one deployment (running/starting/draining).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        target: Inference target name from config; omit for the default.
    """
    return ops.list_replicas(_get_connection(target), application, deployment)


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def autoscale_config_get(application: str, deployment: str, target: Optional[str] = None) -> dict:
    """[READ] Autoscale bounds (min/max replicas, target ongoing requests).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        target: Inference target name from config; omit for the default.
    """
    return ops.get_autoscale_config(_get_connection(target), application, deployment)


# ── writes ───────────────────────────────────────────────────────────────


@mcp.tool()
@governed_tool(risk_level="medium", undo=_replica_undo)
@tool_errors("dict")
def scale_replicas_up(
    application: str, deployment: str, num_replicas: int, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=medium] Raise a deployment's replica count (reversible → prior).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        num_replicas: New (higher) replica count.
        target: Inference target name from config; omit for the default.
    """
    return ops.scale_replicas_up(_get_connection(target), application, deployment, num_replicas)


@mcp.tool()
@governed_tool(risk_level="high", undo=_replica_undo)
@tool_errors("dict")
def scale_replicas_down(
    application: str, deployment: str, num_replicas: int,
    dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Lower a deployment's replica count on prod (reversible → prior).

    Fewer replicas can strand in-flight requests — pass dry_run=True to preview.

    Args:
        application: Serve application name.
        deployment: Deployment name.
        num_replicas: New (lower) replica count.
        dry_run: If True, preview without scaling.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        cur = ops.get_deployment_status(conn, application, deployment)
        return {"dryRun": True, "from": cur.get("numReplicas"), "to": num_replicas}
    return ops.scale_replicas_down(conn, application, deployment, num_replicas)


@mcp.tool()
@governed_tool(risk_level="high", undo=_replica_undo)
@tool_errors("dict")
def scale_to_zero(
    application: str, deployment: str, dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Park a deployment at 0 replicas (reversible → prior count).

    Stops the cost bleed but adds cold-start latency and can strand the ingress —
    pass dry_run=True to preview. Requires an approver (INFERENCE_AUDIT_APPROVED_BY).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        dry_run: If True, preview without scaling to zero.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        cur = ops.get_deployment_status(conn, application, deployment)
        return {"dryRun": True, "from": cur.get("numReplicas"), "to": 0}
    return ops.scale_to_zero(conn, application, deployment)


@mcp.tool()
@governed_tool(risk_level="medium", undo=_autoscale_undo)
@tool_errors("dict")
def autoscale_config_update(
    application: str, deployment: str,
    min_replicas: Optional[int] = None, max_replicas: Optional[int] = None,
    target_ongoing_requests: Optional[float] = None, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=medium] Live-tune autoscale bounds without a redeploy (reversible).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        min_replicas: New floor (omit to leave unchanged).
        max_replicas: New ceiling (omit to leave unchanged).
        target_ongoing_requests: New per-replica concurrency target (omit to leave).
        target: Inference target name from config; omit for the default.
    """
    return ops.update_autoscale_config(
        _get_connection(target), application, deployment,
        min_replicas=min_replicas, max_replicas=max_replicas,
        target_ongoing_requests=target_ongoing_requests)


@mcp.tool()
@governed_tool(risk_level="high")
@tool_errors("dict")
def drain_replica(
    application: str, deployment: str, replica_id: str,
    dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Gracefully drain one replica (finish in-flight, take no new).

    Pass dry_run=True to preview. Requires an approver (INFERENCE_AUDIT_APPROVED_BY).

    Args:
        application: Serve application name.
        deployment: Deployment name.
        replica_id: Replica id (from replica_list).
        dry_run: If True, preview without draining.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True, "wouldDrain": {"application": application,
                                               "deployment": deployment, "replicaId": replica_id}}
    return ops.drain_replica(conn, application, deployment, replica_id)
