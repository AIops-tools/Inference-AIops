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

    # Absent counters stay None: an engine that exposes none of these must not read as
    # "nothing is wrong". The comparisons below treat None as "not measured", and the
    # unreadable ones are named in the result.
    waiting_raw = metric_latest(m, _WAITING)
    kv_raw = metric_latest(m, _KV_USAGE)
    preempt_raw = metric_sum(m, _PREEMPT)
    unreadable = [n for n, v in (("numWaiting", waiting_raw), ("kvUsage", kv_raw),
                                 ("preemptions", preempt_raw)) if v is None]
    waiting = waiting_raw or 0.0
    kv = kv_raw or 0.0
    preempt = preempt_raw or 0.0
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
        # "No dominant bottleneck" is a measurement, and must not be what an engine that
        # reported none of these signals looks like.
        if len(unreadable) == 3:
            # Nothing was measurable, so there is no verdict to give. Withholding it only
            # here keeps a partial-but-benign reading ("queue and KV are clean") useful.
            causes.append({
                "cause": f"No bottleneck could be ranked: the engine did not report "
                         f"{', '.join(unreadable)}. An absent counter is not zero, so this "
                         f"is not evidence that the engine is healthy.",
                "action": "Confirm the engine exposes vLLM-compatible metrics before "
                          "reading this diagnosis as an all-clear.",
                "signal": {},
            })
        else:
            note = (f" Not measured: {', '.join(unreadable)}." if unreadable else "")
            causes.append({
                "cause": f"No dominant bottleneck in the queue / KV / prefix signals the "
                         f"engine reported.{note}",
                "action": "Check GPU throttling and per-replica skew (get_gpu_utilization, "
                          "list_replicas).",
                "signal": {},
            })
    return {"probableCauses": causes, "signalsChecked":
            {"numWaiting": waiting_raw, "kvUsage": kv_raw, "preemptions": preempt_raw,
             "prefixCacheHitRate": round(hit_rate, 4) if hit_rate is not None else None},
            "unreadableSignals": unreadable}


def diagnose_low_utilization(conn: Any) -> dict:
    """[READ][RCA] Explain an under-used GPU (batching / imbalance / overprovision)."""
    try:
        m = conn.vllm_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    # metric_latest already returns None for a counter the engine did not expose.
    # Coercing that to 0.0 made "not reported" identical to "genuinely zero", and the
    # idle branch below then recommended scaling serving capacity to zero.
    waiting = metric_latest(m, _WAITING)
    running = metric_latest(m, _RUNNING)
    kv = metric_latest(m, _KV_USAGE)
    unreadable = [n for n, v in (("numRunning", running), ("numWaiting", waiting),
                                 ("kvUsage", kv)) if v is None]

    # Refuse only the conclusion whose evidence is missing. `running > 0` already rules
    # out idle, so an unreported queue depth does not block the batching verdict.
    if running is None or (running == 0 and waiting is None):
        missing = "numRunning" if running is None else "numWaiting"
        finding = (f"Cannot tell whether this deployment is idle: the engine did not report "
                   f"{missing}. An absent counter is not zero.")
        action = ("Check the engine's /metrics endpoint and that it is a vLLM-compatible "
                  "exporter. Do not scale this deployment down on the strength of this result.")
    elif running == 0 and waiting == 0:
        finding = ("Idle — no traffic. If replicas are held warm for latency, "
                   "consider scale-to-zero to stop the cost bleed.")
        action = "scale_to_zero (if this deployment can tolerate cold starts)."
    elif running > 0 and kv is not None and kv < 0.3:
        finding = ("Low batching — few concurrent sequences and low KV usage means "
                   "the GPU is under-fed (utilisation < ~30%).")
        action = "Raise --max-num-seqs / consolidate replicas; route more traffic per replica."
    else:
        finding = "Utilisation looks reasonable for the current load."
        action = "No change indicated."
    return {"finding": finding, "suggestedAction": action,
            "signals": {"numRunning": running, "numWaiting": waiting, "kvUsage": kv},
            "unreadableSignals": unreadable}
