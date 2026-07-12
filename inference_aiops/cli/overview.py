"""``inference-aiops overview`` — one-shot fleet health."""

from __future__ import annotations

import json

from inference_aiops.cli._common import TargetOption, cli_errors, console, get_connection


@cli_errors
def overview_cmd(target: TargetOption = None) -> None:
    """One-shot stack summary: deployments, total replicas, queue backpressure."""
    from inference_aiops.ops import overview as ops

    conn, _ = get_connection(target)
    console.print_json(json.dumps(ops.fleet_overview(conn)))
