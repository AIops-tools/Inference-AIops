"""Ray Serve DEPLOY lifecycle + routing MCP tools (guarded writes).

Undeploy (removes a whole app) and redeploy (re-applies config, can drop
in-flight requests) are the fragile app-level ops — risk=high with a dry_run
preview. ``routing_policy_update`` records an undo capturing the prior routing
policy so a bad switch is reversible.
"""

from typing import Any, Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import deploy as ops
from mcp_server._shared import _get_connection, mcp, tool_errors


def _routing_undo(params: dict[str, Any], result: Any) -> Optional[dict]:
    """Inverse of routing_policy_update: restore the captured prior policy."""
    if not isinstance(result, dict):
        return None
    prior = (result.get("priorState") or {}).get("policy")
    if prior is None:
        return None
    return {"tool": "routing_policy_update",
            "params": {"application": params.get("application"),
                       "deployment": params.get("deployment"), "policy": prior},
            "note": "Restore the deployment's prior routing policy."}


@mcp.tool()
@governed_tool(risk_level="medium")
@tool_errors("dict")
def model_deploy(
    application: str, import_path: str, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=medium] Deploy a Serve application from an import path.

    The app is merged into the cluster's declarative Serve config (existing apps
    preserved). Replica counts come from the app's own config; set them afterward
    with serve scale (deployment names are known only once the app materialises).

    Args:
        application: Serve application name to create/replace.
        import_path: Python import path of the Serve app (e.g. 'module:app').
        target: Inference target name from config; omit for the default.
    """
    return ops.deploy_model(_get_connection(target), application, import_path)


@mcp.tool()
@governed_tool(risk_level="high")
@tool_errors("dict")
def model_undeploy(
    application: str, dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Tear down a whole Serve application (removes all deployments).

    Irreversible without the original import path — pass dry_run=True to preview.

    Args:
        application: Serve application name (from serve_deployment_list).
        dry_run: If True, preview without undeploying.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True, "wouldUndeploy": {"application": application}}
    return ops.undeploy_model(conn, application)


@mcp.tool()
@governed_tool(risk_level="high")
@tool_errors("dict")
def deployment_redeploy(
    application: str, deployment: str, dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Force a deployment to re-apply new config.

    Applies the new config immediately and can drop unfinished requests — pass
    dry_run=True to preview.

    Args:
        application: Serve application name.
        deployment: Deployment name within the application.
        dry_run: If True, preview without redeploying.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True,
                "wouldRedeploy": {"application": application, "deployment": deployment}}
    return ops.redeploy_deployment(conn, application, deployment)


@mcp.tool()
@governed_tool(risk_level="medium", undo=_routing_undo)
@tool_errors("dict")
def routing_policy_update(
    application: str, deployment: str, policy: str, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=medium] Switch a deployment's request-routing policy.

    NOTE: Ray Serve's request-routing policy is a code-level option on the
    @serve.deployment decorator, NOT a field settable over the Dashboard REST API
    — there is no REST endpoint for it. This tool therefore refuses with a
    teaching error rather than pretend; change the policy in the deployment's
    source and redeploy.

    Args:
        application: Serve application name.
        deployment: Deployment name.
        policy: Desired routing policy (prefix_aware / round_robin / session_affinity).
        target: Inference target name from config; omit for the default.
    """
    return ops.update_routing_policy(_get_connection(target), application, deployment, policy)
