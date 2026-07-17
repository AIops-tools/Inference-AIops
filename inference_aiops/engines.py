"""Serving-engine platform registry (vLLM, SGLang, TGI).

vLLM, SGLang, and TGI all front a GPU inference server over an HTTP surface:
a health probe, a running-model identity, and a Prometheus ``/metrics``
endpoint. They differ in the *paths* they expose and the *names* of their
metrics. An :class:`EngineSpec` captures those differences behind one small,
frozen record so the engine-agnostic ops (``engine_health`` /
``engine_inventory`` / ``get_engine_request_metrics`` / ``diagnose_engine_latency``)
read the same canonical signals everywhere.

Only **vLLM** additionally has a Ray Serve control plane (multi-replica scale /
drain / autoscale). **SGLang** and **TGI** are single-process servers with no
such API, so ``has_control_plane`` is ``False`` for them and the Ray-shaped
scale/drain writes raise the line's standard teaching error rather than issuing
a call that could never work (see ``inference_aiops.ops.engine``).

Metric names are drawn from each engine's stable Prometheus exposition:

  * vLLM   — ``vllm:*`` (num_requests_waiting/running, gpu_cache_usage_perc, …)
  * SGLang — ``sglang:*`` (num_queue_reqs, num_running_reqs, token_usage,
             cache_hit_rate, time_to_first_token_seconds, …)
  * TGI    — ``tgi_*`` (queue_size, batch_current_size, request_duration,
             request_mean_time_per_token_duration, …)

A ``None`` metric means the engine does not expose that signal; the reader
degrades that field to ``None`` instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical default port for each engine's HTTP server.
DEFAULT_ENGINE_PORTS: dict[str, int] = {"vllm": 8000, "sglang": 30000, "tgi": 8080}


@dataclass(frozen=True)
class EngineSpec:
    """Immutable description of one serving engine's HTTP + metrics surface.

    ``m_*`` fields are Prometheus metric names (or ``None`` when the engine does
    not expose that signal). Histogram signals (``m_ttft`` / ``m_tpot`` /
    ``m_e2e``) name the histogram *base* — the reader averages ``<base>_sum`` /
    ``<base>_count``. Counter/gauge signals name the metric directly.
    """

    name: str
    label: str
    default_port: int
    health_path: str
    info_path: str | None
    models_path: str | None
    has_control_plane: bool
    m_waiting: str | None
    m_running: str | None
    m_kv_usage: str | None
    m_ttft: str | None
    m_tpot: str | None
    m_e2e: str | None
    m_gen_tokens: str | None
    m_preempt: str | None
    m_cache_hit_rate: str | None
    m_prefix_hits: str | None
    m_prefix_queries: str | None


_VLLM = EngineSpec(
    name="vllm",
    label="vLLM",
    default_port=DEFAULT_ENGINE_PORTS["vllm"],
    health_path="/health",
    info_path=None,
    models_path="/v1/models",
    has_control_plane=True,
    m_waiting="vllm:num_requests_waiting",
    m_running="vllm:num_requests_running",
    m_kv_usage="vllm:gpu_cache_usage_perc",
    m_ttft="vllm:time_to_first_token_seconds",
    m_tpot="vllm:time_per_output_token_seconds",
    m_e2e="vllm:e2e_request_latency_seconds",
    m_gen_tokens="vllm:generation_tokens_total",
    m_preempt="vllm:num_preemptions_total",
    m_cache_hit_rate=None,
    m_prefix_hits="vllm:prefix_cache_hits_total",
    m_prefix_queries="vllm:prefix_cache_queries_total",
)

_SGLANG = EngineSpec(
    name="sglang",
    label="SGLang",
    default_port=DEFAULT_ENGINE_PORTS["sglang"],
    health_path="/health",
    info_path="/get_server_info",
    models_path="/v1/models",
    has_control_plane=False,
    m_waiting="sglang:num_queue_reqs",
    m_running="sglang:num_running_reqs",
    m_kv_usage="sglang:token_usage",
    m_ttft="sglang:time_to_first_token_seconds",
    m_tpot="sglang:inter_token_latency_seconds",
    m_e2e="sglang:e2e_request_latency_seconds",
    m_gen_tokens="sglang:generation_tokens_total",
    m_preempt=None,
    # SGLang exposes a ready-made hit-rate gauge (percent) rather than a pair.
    m_cache_hit_rate="sglang:cache_hit_rate",
    m_prefix_hits=None,
    m_prefix_queries=None,
)

_TGI = EngineSpec(
    name="tgi",
    label="TGI",
    default_port=DEFAULT_ENGINE_PORTS["tgi"],
    health_path="/health",
    info_path="/info",
    models_path=None,  # TGI serves a single model; identity comes from /info.
    has_control_plane=False,
    m_waiting="tgi_queue_size",
    m_running="tgi_batch_current_size",
    m_kv_usage=None,  # TGI does not expose KV-cache utilisation.
    m_ttft=None,  # TGI has no direct time-to-first-token metric.
    m_tpot="tgi_request_mean_time_per_token_duration",
    m_e2e="tgi_request_duration",
    m_gen_tokens=None,
    m_preempt=None,
    m_cache_hit_rate=None,
    m_prefix_hits=None,
    m_prefix_queries=None,
)


ENGINES: dict[str, EngineSpec] = {"vllm": _VLLM, "sglang": _SGLANG, "tgi": _TGI}

# Public tuple of engine names, in a stable order (used for validation + help).
SUPPORTED_ENGINES: tuple[str, ...] = tuple(ENGINES)


def get_engine_spec(name: str) -> EngineSpec:
    """Return the :class:`EngineSpec` for ``name`` (case-insensitive).

    Raises ``ValueError`` (a teaching message) for an unknown engine so a
    typo in config.yaml fails fast rather than silently defaulting.
    """
    key = str(name or "").strip().lower()
    spec = ENGINES.get(key)
    if spec is None:
        supported = ", ".join(SUPPORTED_ENGINES)
        raise ValueError(
            f"Unknown serving engine '{name}'. Supported engines: {supported}."
        )
    return spec
