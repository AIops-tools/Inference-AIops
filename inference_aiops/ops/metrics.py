"""vLLM request metrics + latency/utilization RCA (read-only).

These reads pull the signals operators actually alert on — queue depth (the
leading production alert), KV-cache utilisation + preemptions, and TTFT/TPOT —
straight from vLLM's Prometheus ``/metrics``. ``diagnose_latency_spike`` and
``diagnose_low_utilization`` are the flagship value-add: they correlate those
signals into a ranked probable-cause + suggested-action summary, so an agent
gets an answer instead of a wall of raw metrics.

All reads are resilient: a scrape failure degrades to an ``error`` field.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops._util import histogram_avg, metric_latest, metric_sum, s

# vLLM metric names (stable across recent vLLM releases).
_WAITING = "vllm:num_requests_waiting"
_RUNNING = "vllm:num_requests_running"
_KV_USAGE = "vllm:gpu_cache_usage_perc"
_PREEMPT = "vllm:num_preemptions_total"
_PREFIX_HITS = "vllm:prefix_cache_hits_total"
_PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
_TTFT = "vllm:time_to_first_token_seconds"
_TPOT = "vllm:time_per_output_token_seconds"
_E2E = "vllm:e2e_request_latency_seconds"
_GEN_TOKENS = "vllm:generation_tokens_total"


def get_request_metrics(conn: Any) -> dict:
    """[READ] TTFT / TPOT / e2e latency + token totals from vLLM /metrics."""
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    return {
        "ttftSeconds": histogram_avg(m, _TTFT),
        "tpotSeconds": histogram_avg(m, _TPOT),
        "e2eLatencySeconds": histogram_avg(m, _E2E),
        "generationTokensTotal": metric_sum(m, _GEN_TOKENS),
    }


def get_queue_depth(conn: Any) -> dict:
    """[READ] Running vs waiting requests — the leading backpressure signal."""
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    waiting = metric_latest(m, _WAITING) or 0.0
    running = metric_latest(m, _RUNNING) or 0.0
    return {
        "numWaiting": waiting,
        "numRunning": running,
        "backpressure": waiting > 0,
    }


def get_kv_cache_stats(conn: Any) -> dict:
    """[READ] KV-cache utilisation, prefix-cache hit rate, and preemption count."""
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    hits = metric_sum(m, _PREFIX_HITS)
    queries = metric_sum(m, _PREFIX_QUERIES)
    hit_rate = round(hits / queries, 4) if hits and queries else None
    return {
        "gpuCacheUsagePerc": metric_latest(m, _KV_USAGE),
        "prefixCacheHitRate": hit_rate,
        "preemptionsTotal": metric_sum(m, _PREEMPT),
    }


def diagnose_latency_spike(conn: Any) -> dict:
    """[READ][RCA] Correlate queue depth + KV eviction + prefix locality into a cause.

    Deterministic heuristic over the live signals — ranks the probable cause of a
    TTFT/latency spike and suggests the specific knob to turn.
    """
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}

    waiting = metric_latest(m, _WAITING) or 0.0
    kv = metric_latest(m, _KV_USAGE) or 0.0
    preempt = metric_sum(m, _PREEMPT) or 0.0
    hits = metric_sum(m, _PREFIX_HITS)
    queries = metric_sum(m, _PREFIX_QUERIES)
    hit_rate = (hits / queries) if hits and queries else None

    causes: list[dict] = []
    if kv >= 0.9 or preempt > 0:
        causes.append({
            "cause": "KV-cache pressure / preemption — the engine is evicting and "
                     "recomputing, spiking TTFT.",
            "action": "Lower --max-num-seqs or raise --gpu-memory-utilization; add a "
                      "replica if sustained.",
            "signal": {"kvUsage": kv, "preemptions": preempt},
        })
    if waiting > 0:
        causes.append({
            "cause": "Queue backpressure — requests are waiting for a running slot.",
            "action": "Scale replicas up (scale_replicas_up) or raise batch capacity.",
            "signal": {"numWaiting": waiting},
        })
    if hit_rate is not None and hit_rate < 0.3:
        causes.append({
            "cause": "Cold prefix cache — naive load balancing is destroying cache "
                     "locality across replicas.",
            "action": "Switch to prefix-aware / session-affinity routing "
                      "(update_routing_policy).",
            "signal": {"prefixCacheHitRate": round(hit_rate, 4)},
        })
    if not causes:
        causes.append({
            "cause": "No dominant bottleneck in queue / KV / prefix signals.",
            "action": "Check GPU throttling and per-replica skew (get_gpu_utilization, "
                      "list_replicas).",
            "signal": {},
        })
    return {"probableCauses": causes, "signalsChecked":
            {"numWaiting": waiting, "kvUsage": kv, "preemptions": preempt,
             "prefixCacheHitRate": round(hit_rate, 4) if hit_rate is not None else None}}


def diagnose_low_utilization(conn: Any) -> dict:
    """[READ][RCA] Explain an under-used GPU (batching / imbalance / overprovision)."""
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    waiting = metric_latest(m, _WAITING) or 0.0
    running = metric_latest(m, _RUNNING) or 0.0
    kv = metric_latest(m, _KV_USAGE) or 0.0

    if running == 0 and waiting == 0:
        finding = ("Idle — no traffic. If replicas are held warm for latency, "
                   "consider scale-to-zero to stop the cost bleed.")
        action = "scale_to_zero (if this deployment can tolerate cold starts)."
    elif running > 0 and kv < 0.3:
        finding = ("Low batching — few concurrent sequences and low KV usage means "
                   "the GPU is under-fed (utilisation < ~30%).")
        action = "Raise --max-num-seqs / consolidate replicas; route more traffic per replica."
    else:
        finding = "Utilisation looks reasonable for the current load."
        action = "No change indicated."
    return {"finding": finding, "suggestedAction": action,
            "signals": {"numRunning": running, "numWaiting": waiting, "kvUsage": kv}}
