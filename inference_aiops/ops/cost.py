"""Cost attribution for a GPU inference stack (read-only).

Turns live vLLM throughput into a $/token unit cost so an operator can reason
about the economics of a deployment without a separate FinOps pipeline. The read
is deterministic — it multiplies the current generation-throughput gauge by the
supplied GPU hourly cost; no clock or sampling is involved.

Reads are resilient: a scrape failure degrades to an ``error`` field.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops._util import metric_latest, s

# vLLM generation-throughput gauge (tokens/sec), stable across recent releases.
_THROUGHPUT = "vllm:avg_generation_throughput_toks_per_s"


def get_cost_per_token(conn: Any, gpu_hourly_cost: float, num_gpus: int = 1) -> dict:
    """[READ] Attribute a $/1M-token unit cost from live vLLM throughput.

    Reads the current generation-throughput gauge and multiplies the cluster
    GPU hourly cost across the hourly token volume. Returns an
    ``insufficient-data`` forecast when no throughput metric is present.
    """
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}

    throughput = metric_latest(m, _THROUGHPUT)
    if not throughput or throughput <= 0:
        return {
            "forecast": "insufficient-data",
            "reason": "no throughput metric",
            "gpuHourlyCost": gpu_hourly_cost,
            "numGpus": num_gpus,
        }

    tokens_per_hour = throughput * 3600
    cluster_hourly_cost = gpu_hourly_cost * num_gpus
    cost_per_1m = round(cluster_hourly_cost / (tokens_per_hour / 1_000_000), 4)
    return {
        "throughputTokPerSec": throughput,
        "tokensPerHour": tokens_per_hour,
        "clusterHourlyCost": cluster_hourly_cost,
        "costPer1MTokens": cost_per_1m,
        "gpuHourlyCost": gpu_hourly_cost,
        "numGpus": num_gpus,
    }
