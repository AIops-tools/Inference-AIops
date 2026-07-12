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
    dry_run_print,
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
    from inference_aiops.ops import serve as ops

    if dry_run:
        dry_run_print(operation="scale_replicas",
                      api_call=f"PUT /api/serve/applications/{application}/deployments/"
                               f"{deployment}",
                      parameters={"num_replicas": num_replicas})
        return
    conn, _ = get_connection(target)
    result = ops.scale_replicas_up(conn, application, deployment, num_replicas)
    console.print_json(json.dumps(result))


@serve_app.command("scale-to-zero")
@cli_errors
def serve_scale_zero(
    application: AppArg, deployment: DepArg,
    target: TargetOption = None, dry_run: DryRunOption = False,
) -> None:
    """Park a deployment at 0 replicas (dry-run + double confirm)."""
    from inference_aiops.ops import serve as ops

    if dry_run:
        dry_run_print(operation="scale_to_zero",
                      api_call=f"PUT /api/serve/applications/{application}/deployments/"
                               f"{deployment}",
                      parameters={"num_replicas": 0})
        return
    double_confirm("scale to zero", f"{application}/{deployment}")
    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.scale_to_zero(conn, application, deployment)))
