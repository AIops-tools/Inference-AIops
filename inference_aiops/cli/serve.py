"""``inference-aiops serve`` — Ray Serve reads + guarded scaling writes."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from inference_aiops.cli._common import (
    DryRunOption,
    TargetOption,
    cli_errors,
    console,
    double_confirm,
    dry_run_preview,
    get_connection,
)

serve_app = typer.Typer(
    name="serve",
    help="Ray Serve: list/status, scale, scale-to-zero, drain.",
    no_args_is_help=True,
)

AppArg = Annotated[str, typer.Argument(help="Serve application name")]
DepArg = Annotated[str, typer.Argument(help="Deployment name")]


@serve_app.command("list")
@cli_errors
def serve_list(target: TargetOption = None) -> None:
    """List Ray Serve deployments."""
    from inference_aiops.ops import serve as ops

    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.list_serve_deployments(conn)))


@serve_app.command("status")
@cli_errors
def serve_status(application: AppArg, deployment: DepArg, target: TargetOption = None) -> None:
    """Show one deployment's status + replica count."""
    from inference_aiops.ops import serve as ops

    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.get_deployment_status(conn, application, deployment)))


@serve_app.command("scale")
@cli_errors
def serve_scale(
    application: AppArg,
    deployment: DepArg,
    num_replicas: Annotated[int, typer.Argument(help="Target replica count")],
    target: TargetOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Scale a deployment to a target replica count."""
    from mcp_server.tools import serve as gov

    if dry_run:
        preview = gov.scale_replicas_up(
            application=application, deployment=deployment,
            num_replicas=num_replicas, dry_run=True, target=target)
        dry_run_preview(
            preview,
            operation="scale_replicas",
            api_call=f"PUT /api/serve/applications/{application}/deployments/"
                     f"{deployment}",
            parameters={"from_replicas": preview.get("from"),
                        "to_replicas": preview.get("to")})
        return
    console.print_json(json.dumps(gov.scale_replicas_up(
        application=application, deployment=deployment,
        num_replicas=num_replicas, target=target)))


@serve_app.command("scale-to-zero")
@cli_errors
def serve_scale_zero(
    application: AppArg, deployment: DepArg,
    target: TargetOption = None, dry_run: DryRunOption = False,
) -> None:
    """Park a deployment at 0 replicas (dry-run + double confirm)."""
    from mcp_server.tools import serve as gov

    if dry_run:
        preview = gov.scale_to_zero(
            application=application, deployment=deployment, dry_run=True, target=target)
        dry_run_preview(
            preview,
            operation="scale_to_zero",
            api_call=f"PUT /api/serve/applications/{application}/deployments/"
                     f"{deployment}",
            parameters={"from_replicas": preview.get("from"),
                        "to_replicas": preview.get("to")})
        return
    double_confirm("scale to zero", f"{application}/{deployment}")
    console.print_json(json.dumps(gov.scale_to_zero(
        application=application, deployment=deployment, target=target)))
