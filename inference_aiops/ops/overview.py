"""One-shot inference-stack overview (read-only).

Folds Ray Serve deployments + vLLM queue signal into a single summary an agent
can call first. Resilient — a failing sub-call degrades to a partial summary
with an ``errors`` list rather than a raised traceback.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops import metrics as m
from inference_aiops.ops import serve as sv


def fleet_overview(conn: Any) -> dict:
    """[READ] Deployment count + total replicas + queue backpressure signal."""
    errors: list[str] = []

    deployments = sv.list_serve_deployments(conn)
    if deployments and isinstance(deployments[0], dict) and "error" in deployments[0]:
        errors.append(f"serve: {deployments[0]['error']}")
        deployments = []

    queue = m.get_queue_depth(conn)
    if "error" in queue:
        errors.append(f"metrics: {queue['error']}")
        queue = {}

    total_replicas = sum((d.get("numReplicas") or 0) for d in deployments)
    return {
        "deployments": len(deployments),
        "totalReplicas": total_replicas,
        "numWaiting": queue.get("numWaiting"),
        "numRunning": queue.get("numRunning"),
        "backpressure": queue.get("backpressure"),
        "errors": errors,
    }
