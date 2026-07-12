"""``inference-aiops metrics`` — vLLM metric reads + latency/util RCA."""

from __future__ import annotations

import json

import typer

from inference_aiops.cli._common import TargetOption, cli_errors, console, get_connection

metrics_app = typer.Typer(
    name="metrics",
    help="vLLM request metrics, queue depth, KV cache, and latency/util RCA.",
    no_args_is_help=True,
)


@metrics_app.command("requests")
@cli_errors
def metrics_requests(target: TargetOption = None) -> None:
    """TTFT / TPOT / e2e latency + generation-token totals."""
    from inference_aiops.ops import metrics as ops

    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.get_request_metrics(conn)))


@metrics_app.command("queue")
@cli_errors
def metrics_queue(target: TargetOption = None) -> None:
    """Running vs waiting requests (backpressure signal)."""
    from inference_aiops.ops import metrics as ops

    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.get_queue_depth(conn)))


@metrics_app.command("diagnose")
@cli_errors
def metrics_diagnose(target: TargetOption = None) -> None:
    """RCA: rank the probable cause of a latency spike + the knob to turn."""
    from inference_aiops.ops import metrics as ops

    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.diagnose_latency_spike(conn)))
