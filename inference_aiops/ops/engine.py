"""Engine-agnostic reads across vLLM / SGLang / TGI (+ control-plane guard).

These reads map every supported serving engine onto one canonical surface —
health, running-model inventory, request-latency metrics, queue depth, and a
latency root-cause correlation — by pulling the right paths and metric names
from the target's :class:`~inference_aiops.engines.EngineSpec`. A signal the
engine does not expose degrades to ``None`` instead of being guessed.

The write side is deliberately thin: multi-replica scale / drain are Ray Serve
control-plane actions that only vLLM has. ``require_control_plane`` is the guard
the Ray-shaped write ops call first; on a single-process engine (SGLang / TGI)
it raises :class:`~inference_aiops.connection.EngineCapabilityError` with a
teaching message instead of issuing a call that could never succeed.

All reads are resilient: a scrape/endpoint hiccup degrades to an ``error`` field.
"""

from __future__ import annotations

from typing import Any

from inference_aiops.connection import EngineCapabilityError
from inference_aiops.engines import EngineSpec, get_engine_spec
from inference_aiops.ops._util import (
    as_list,
    as_obj,
    histogram_avg,
    metric_latest,
    metric_sum,
    opt_s,
    s,
)

# Engine names known to lack a Ray Serve control plane (single-process servers).
_SINGLE_PROCESS_ENGINES = frozenset({"sglang", "tgi"})


def _engine_name(conn: Any) -> str:
    """Best-effort engine name for ``conn`` (defaults to ``vllm``).

    Tolerant of mock connections whose ``target.engine`` is not a real string,
    so control-plane guarding never mis-fires in unit tests or on a plain vLLM
    target — it only trips on an explicitly single-process engine.
    """
    engine = getattr(getattr(conn, "target", None), "engine", None)
    return engine if isinstance(engine, str) else "vllm"


def _spec(conn: Any) -> EngineSpec:
    return get_engine_spec(_engine_name(conn))


# ── control-plane guard (used by the Ray-shaped write ops) ─────────────────


def require_control_plane(conn: Any, operation: str) -> None:
    """Raise a teaching error when ``operation`` needs a control plane the engine lacks.

    No-op for vLLM (and for mock connections). For SGLang / TGI it raises
    :class:`EngineCapabilityError` explaining that horizontal scale/drain lives
    outside the single-process engine.
    """
    engine = _engine_name(conn)
    if engine not in _SINGLE_PROCESS_ENGINES:
        return
    label = get_engine_spec(engine).label
    raise EngineCapabilityError(
        f"'{operation}' is a Ray Serve control-plane action, but this target runs "
        f"the single-process {label} engine, which has no multi-replica "
        f"scale/drain/autoscale API. Scale {label} horizontally by fronting it with "
        f"a process manager, load balancer, Ray Serve, or Kubernetes; use "
        f"engine_health / engine_request_metrics / diagnose_engine_latency for "
        f"{label} observability."
    )


# ── reads ──────────────────────────────────────────────────────────────────


def engine_health(conn: Any) -> dict:
    """[READ] Liveness of the serving engine via its health probe."""
    spec = _spec(conn)
    try:
        conn.get_engine(spec.health_path)
    except Exception as exc:  # noqa: BLE001 — health is a status, not a crash
        return {"engine": spec.name, "label": spec.label, "healthy": False,
                "error": s(exc, 200)}
    return {"engine": spec.name, "label": spec.label, "healthy": True}


def _server_info(conn: Any, spec: EngineSpec) -> dict:
    """Best-effort server-info payload (SGLang ``/get_server_info``, TGI ``/info``)."""
    if not spec.info_path:
        return {}
    try:
        return as_obj(conn.get_engine(spec.info_path))
    except Exception:  # noqa: BLE001 — server info is optional metadata
        return {}


def engine_inventory(conn: Any) -> dict:
    """[READ] Running-model identity + engine server info (engine-agnostic).

    vLLM / SGLang report their served ids from ``/v1/models``; TGI serves a
    single model whose id comes from ``/info`` (``model_id``).
    """
    spec = _spec(conn)
    info = _server_info(conn, spec)
    try:
        if spec.models_path:
            models = [s(e.get("id")) for e in as_list(conn.get_engine(spec.models_path))
                      if e.get("id") is not None]
        else:  # TGI: identity from /info
            model_id = info.get("model_id") or info.get("model_path")
            models = [s(model_id)] if model_id else []
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"engine": spec.name, "label": spec.label, "error": s(exc, 200)}
    return {
        "engine": spec.name,
        "label": spec.label,
        "models": models,
        "serverInfo": {
            "model": opt_s(info.get("model_path") or info.get("model_id")),
            "version": opt_s(info.get("version")),
            "maxConcurrentRequests": info.get("max_concurrent_requests"),
        },
    }


def get_engine_request_metrics(conn: Any) -> dict:
    """[READ] TTFT / TPOT / e2e latency + generation-token totals (where exposed)."""
    spec = _spec(conn)
    try:
        m = conn.engine_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"engine": spec.name, "error": s(exc, 200)}
    return {
        "engine": spec.name,
        "ttftSeconds": histogram_avg(m, spec.m_ttft) if spec.m_ttft else None,
        "tpotSeconds": histogram_avg(m, spec.m_tpot) if spec.m_tpot else None,
        "e2eLatencySeconds": histogram_avg(m, spec.m_e2e) if spec.m_e2e else None,
        "generationTokensTotal": metric_sum(m, spec.m_gen_tokens) if spec.m_gen_tokens else None,
    }


def get_engine_queue_depth(conn: Any) -> dict:
    """[READ] Running vs waiting requests — the leading backpressure signal."""
    spec = _spec(conn)
    try:
        m = conn.engine_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"engine": spec.name, "error": s(exc, 200)}
    waiting = (metric_latest(m, spec.m_waiting) if spec.m_waiting else None) or 0.0
    running = (metric_latest(m, spec.m_running) if spec.m_running else None) or 0.0
    return {
        "engine": spec.name,
        "numWaiting": waiting,
        "numRunning": running,
        "backpressure": waiting > 0,
    }


def _cache_hit_rate(m: dict, spec: EngineSpec) -> float | None:
    """Prefix/KV cache hit rate: a direct gauge (SGLang) or a hits/queries pair (vLLM)."""
    if spec.m_cache_hit_rate:
        raw = metric_latest(m, spec.m_cache_hit_rate)
        if raw is None:
            return None
        # SGLang reports a percentage (0–100); normalise to a 0–1 fraction.
        return round(raw / 100 if raw > 1 else raw, 4)
    if spec.m_prefix_hits and spec.m_prefix_queries:
        hits = metric_sum(m, spec.m_prefix_hits)
        queries = metric_sum(m, spec.m_prefix_queries)
        return round(hits / queries, 4) if hits and queries else None
    return None


def diagnose_engine_latency(conn: Any) -> dict:
    """[READ][RCA] Rank the probable cause of a latency spike for any engine.

    Correlates whichever signals the engine exposes — queue backpressure,
    KV/token-cache pressure + preemption, and cache locality — into a ranked
    cause list plus the knob to turn. Signals the engine does not expose are
    simply skipped rather than fabricated.
    """
    spec = _spec(conn)
    try:
        m = conn.engine_metrics()
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"engine": spec.name, "error": s(exc, 200)}

    waiting = (metric_latest(m, spec.m_waiting) if spec.m_waiting else None) or 0.0
    kv = (metric_latest(m, spec.m_kv_usage) if spec.m_kv_usage else None) or 0.0
    preempt = (metric_sum(m, spec.m_preempt) if spec.m_preempt else None) or 0.0
    hit_rate = _cache_hit_rate(m, spec)

    causes: list[dict] = []
    if kv >= 0.9 or preempt > 0:
        causes.append({
            "cause": "KV/token-cache pressure — the engine is at capacity and "
                     "evicting/recomputing, spiking latency.",
            "action": "Lower concurrent sequences or raise the KV memory fraction; "
                      "add serving capacity if sustained.",
            "signal": {"cacheUsage": kv, "preemptions": preempt},
        })
    if waiting > 0:
        causes.append({
            "cause": "Queue backpressure — requests are waiting for a running slot.",
            "action": "Raise batch capacity or add another engine instance behind "
                      "your load balancer.",
            "signal": {"numWaiting": waiting},
        })
    if hit_rate is not None and hit_rate < 0.3:
        causes.append({
            "cause": "Cold prefix cache — low cache-hit rate means poor prompt "
                     "locality (or naive cross-instance load balancing).",
            "action": "Use prefix-aware / session-affinity routing to preserve "
                      "cache locality.",
            "signal": {"cacheHitRate": hit_rate},
        })
    if not causes:
        causes.append({
            "cause": "No dominant bottleneck in the queue / cache signals this "
                     "engine exposes.",
            "action": "Check GPU throttling and request-mix; compare with "
                      "get_engine_request_metrics.",
            "signal": {},
        })
    return {
        "engine": spec.name,
        "probableCauses": causes,
        "signalsChecked": {
            "numWaiting": waiting, "cacheUsage": kv, "preemptions": preempt,
            "cacheHitRate": hit_rate,
        },
    }
