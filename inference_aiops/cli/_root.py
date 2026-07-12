"""Top-level Typer app: assembles sub-apps and top-level commands."""

from __future__ import annotations

import typer

from inference_aiops.cli._common import cli_errors
from inference_aiops.cli.doctor import doctor_cmd
from inference_aiops.cli.init import init_cmd
from inference_aiops.cli.metrics import metrics_app
from inference_aiops.cli.overview import overview_cmd
from inference_aiops.cli.secret import secret_app
from inference_aiops.cli.serve import serve_app

app = typer.Typer(
    name="inference-aiops",
    help="Governed AI-ops for GPU inference (vLLM + Ray Serve): metrics/RCA, "
    "scaling, drain, models, jobs.",
    no_args_is_help=True,
)

app.add_typer(serve_app, name="serve")
app.add_typer(metrics_app, name="metrics")
app.add_typer(secret_app, name="secret")
app.command("init")(init_cmd)
app.command("overview")(overview_cmd)
app.command("doctor")(doctor_cmd)


@app.command("mcp")
@cli_errors
def mcp_cmd() -> None:
    """Start the MCP server (stdio transport).

    Single-command entry point for MCP clients (does not go through uvx/PyPI
    resolution at launch):
        inference-aiops mcp
    """
    import sys

    if sys.version_info < (3, 11):
        typer.echo(
            f"ERROR: inference-aiops requires Python >= 3.11 "
            f"(got {sys.version_info.major}.{sys.version_info.minor}).\n"
            f"Fix: uv python install 3.12 && "
            f"uv tool install --python 3.12 --force inference-aiops",
            err=True,
        )
        raise typer.Exit(2)

    from mcp_server.server import main as _mcp_main

    _mcp_main()


if __name__ == "__main__":
    app()
