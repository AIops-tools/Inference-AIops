"""Shared helpers for inference-aiops CLI sub-modules."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console

if TYPE_CHECKING:
    from inference_aiops.config import AppConfig
    from inference_aiops.connection import InferenceConnection

console = Console()

# ─── Shared Option types ───────────────────────────────────────────────────

TargetOption = Annotated[
    str | None, typer.Option("--target", "-t", help="Target name from config")
]
DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Print the API call without executing")
]


def _cli_error_types() -> tuple[type[BaseException], ...]:
    """Exceptions translated to a one-line teaching error instead of a traceback.

    ``PolicyDenied`` is kept in the set defensively. The harness no longer raises
    it — there is no read-only switch or approval gate to deny a call — but were a
    future guard ever to raise it, it would reach the CLI as a live exception
    (it is raised by ``@governed_tool``, OUTSIDE ``@tool_errors``, so it is never
    flattened into an ``{"error": ...}`` dict). Catching it here means such a
    refusal would print its message and exit 1 rather than a blank screen.
    """
    from inference_aiops.connection import InferenceApiError
    from inference_aiops.governance import PolicyDenied

    return (InferenceApiError, KeyError, OSError, ValueError, PolicyDenied)


def cli_errors(fn: Callable) -> Callable:
    """Translate known exceptions into one red line + exit code 1."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except _cli_error_types() as e:
            message = str(e)
            if isinstance(e, KeyError):
                message = f"Missing required key or environment variable: {message}"
            console.print(f"[red]Error: {message}[/]")
            raise typer.Exit(1) from e

    return wrapper


def get_connection(
    target: str | None, config_path: Path | None = None
) -> tuple[InferenceConnection, AppConfig]:
    """Return a (conn, config) tuple for the given target."""
    from inference_aiops.config import load_config
    from inference_aiops.connection import ConnectionManager

    cfg = load_config(config_path)
    mgr = ConnectionManager(cfg)
    return mgr.connect(target), cfg


def dry_run_print(*, operation: str, api_call: str, parameters: dict | None = None) -> None:
    """Print a dry-run preview of the API call that would be made."""
    console.print("\n[bold magenta][DRY-RUN] No changes will be made.[/]")
    console.print(f"[magenta]  Operation: {operation}[/]")
    console.print(f"[magenta]  API Call:  {api_call}[/]")
    for k, v in (parameters or {}).items():
        console.print(f"[magenta]  Param:     {k} = {v}[/]")
    console.print("[magenta]  Run without --dry-run to execute.[/]\n")


def dry_run_preview(
    preview: Any, *, operation: str, api_call: str, parameters: dict | None = None
) -> None:
    """Render a GOVERNED dry-run result as the human-readable DRY-RUN banner.

    ``preview`` must come from calling the governed twin with ``dry_run=True``,
    so every guard that twin carries has already run against the real cluster
    and the same audit row lands as for a real call — the CLI silently not
    auditing previews was the outlier, since MCP previews have always been
    audited.

    A refusal arrives as ``{"error": ...}`` (``tool_errors`` flattens the
    exception into the dict) and is surfaced exactly like a refused real write:
    the teaching message in red, exit code 1. A green banner for a call the
    write is about to reject is the preview being *wrong*, not merely
    incomplete — and a caller that reads "here is what would happen" and then a
    refusal treats the refusal as transient and retries.

    Only the *serialization* stays CLI-shaped: the reader is a human, so the
    returned dict is rendered into the existing banner rather than dumped as
    JSON.

    Invariant: **a dry_run MAY read; it must never write.**
    """
    if isinstance(preview, dict) and preview.get("error"):
        console.print(f"[red]Error: {preview['error']}[/]")
        raise typer.Exit(1)
    dry_run_print(operation=operation, api_call=api_call, parameters=parameters)


def double_confirm(action: str, resource: str) -> None:
    """Require two confirmations for a destructive operation."""
    console.print(f"[bold yellow]⚠️  About to: {action} '{resource}'[/]")
    typer.confirm(f"Confirm 1/2: {action} '{resource}'?", abort=True)
    typer.confirm(
        f"Confirm 2/2: really {action} '{resource}'? This may be irreversible.",
        abort=True,
    )
