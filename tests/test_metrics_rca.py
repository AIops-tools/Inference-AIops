"""vLLM request-metrics reads + latency/utilization RCA thresholds.

Drives ops/metrics with a canned parsed-metrics map (what the connection layer
returns after parsing /metrics). Proves the read shapes, and — the flagship
value-add — that diagnose_latency_spike / diagnose_low_utilization cross the
right thresholds into the right probable-cause + suggested-action.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from inference_aiops.ops import metrics as ops


def _series(value):
    return [{"labels": {}, "value": value}]


def _conn(metrics=None, error=None):
    conn = MagicMock(name="conn")
    if error is not None:
        conn.vllm_metrics.side_effect = error
    else:
        conn.vllm_metrics.return_value = metrics or {}
    return conn


@pytest.mark.unit
def test_get_request_metrics_averages_histograms():
    conn = _conn({
        "vllm:time_to_first_token_seconds_sum": _series(30.0),
        "vllm:time_to_first_token_seconds_count": _series(120.0),
        "vllm:generation_tokens_total": _series(9000.0),
    })
    out = ops.get_request_metrics(conn)
    assert out["ttftSeconds"] == 0.25  # 30 / 120
    assert out["generationTokensTotal"] == 9000.0
    assert out["tpotSeconds"] is None  # absent metric → None


@pytest.mark.unit
def test_get_queue_depth_and_kv_cache_hit_rate():
    conn = _conn({
        "vllm:num_requests_waiting": _series(5.0),
        "vllm:num_requests_running": _series(3.0),
        "vllm:gpu_cache_usage_perc": _series(0.7),
        "vllm:prefix_cache_hits_total": _series(80.0),
        "vllm:prefix_cache_queries_total": _series(100.0),
        "vllm:num_preemptions_total": _series(2.0),
    })
    q = ops.get_queue_depth(conn)
    assert q == {"numWaiting": 5.0, "numRunning": 3.0, "backpressure": True}
    kv = ops.get_kv_cache_stats(conn)
    assert kv["gpuCacheUsagePerc"] == 0.7
    assert kv["prefixCacheHitRate"] == 0.8  # 80 / 100
    assert kv["preemptionsTotal"] == 2.0


@pytest.mark.unit
def test_reads_degrade_to_error_on_scrape_failure():
    conn = _conn(error=RuntimeError("metrics endpoint 503"))
    assert "error" in ops.get_request_metrics(conn)
    assert "error" in ops.get_queue_depth(conn)
    assert "error" in ops.get_kv_cache_stats(conn)


@pytest.mark.unit
def test_diagnose_latency_spike_kv_pressure_ranked_first():
    conn = _conn({
        "vllm:gpu_cache_usage_perc": _series(0.92),
        "vllm:num_preemptions_total": _series(5.0),
        "vllm:num_requests_waiting": _series(9.0),
        "vllm:prefix_cache_hits_total": _series(10.0),
        "vllm:prefix_cache_queries_total": _series(100.0),  # 10% hit → cold
    })
    out = ops.diagnose_latency_spike(conn)
    causes = out["probableCauses"]
    assert "KV-cache pressure" in causes[0]["cause"]
    kinds = " ".join(c["cause"] for c in causes)
    assert "Queue backpressure" in kinds
    assert "Cold prefix cache" in kinds
    assert out["signalsChecked"]["prefixCacheHitRate"] == 0.1


@pytest.mark.unit
def test_diagnose_latency_spike_no_dominant_bottleneck():
    conn = _conn({
        "vllm:gpu_cache_usage_perc": _series(0.4),
        "vllm:num_requests_waiting": _series(0.0),
    })
    out = ops.diagnose_latency_spike(conn)
    assert len(out["probableCauses"]) == 1
    assert "No dominant bottleneck" in out["probableCauses"][0]["cause"]


@pytest.mark.unit
def test_diagnose_low_utilization_idle_recommends_scale_to_zero():
    conn = _conn({
        "vllm:num_requests_running": _series(0.0),
        "vllm:num_requests_waiting": _series(0.0),
    })
    out = ops.diagnose_low_utilization(conn)
    assert "Idle" in out["finding"]
    assert "scale_to_zero" in out["suggestedAction"]


@pytest.mark.unit
def test_diagnose_low_utilization_low_batching():
    conn = _conn({
        "vllm:num_requests_running": _series(2.0),
        "vllm:gpu_cache_usage_perc": _series(0.1),
    })
    out = ops.diagnose_low_utilization(conn)
    assert "Low batching" in out["finding"]
    assert "max-num-seqs" in out["suggestedAction"]


@pytest.mark.unit
def test_diagnose_low_utilization_reasonable():
    conn = _conn({
        "vllm:num_requests_running": _series(8.0),
        "vllm:gpu_cache_usage_perc": _series(0.7),
    })
    out = ops.diagnose_low_utilization(conn)
    assert "reasonable" in out["finding"]
    assert out["suggestedAction"] == "No change indicated."


@pytest.mark.unit
def test_diagnose_functions_degrade_on_scrape_failure():
    conn = _conn(error=RuntimeError("down"))
    assert "error" in ops.diagnose_latency_spike(conn)
    assert "error" in ops.diagnose_low_utilization(conn)


@pytest.mark.unit
def test_unreadable_counters_never_read_as_an_idle_deployment():
    """An engine exposing no vLLM counters used to produce "Idle — no traffic" and a
    recommendation to scale serving capacity to zero, because `or 0.0` made an absent
    counter identical to a measured zero."""
    out = ops.diagnose_low_utilization(_conn({}))
    assert "Cannot tell" in out["finding"]
    assert "scale_to_zero" not in out["suggestedAction"]
    assert out["signals"]["numRunning"] is None
    assert "numRunning" in out["unreadableSignals"]


@pytest.mark.unit
def test_a_measured_zero_still_reads_as_idle():
    """Positive control: the fix must not suppress the real idle verdict."""
    out = ops.diagnose_low_utilization(_conn({
        "vllm:num_requests_running": _series(0.0),
        "vllm:num_requests_waiting": _series(0.0),
    }))
    assert "Idle" in out["finding"] and "scale_to_zero" in out["suggestedAction"]
    assert out["unreadableSignals"] == ["kvUsage"]


@pytest.mark.unit
def test_an_unreported_queue_does_not_block_the_batching_verdict():
    """Refuse only the conclusion whose evidence is missing: running > 0 already rules
    out idle, so an absent queue depth must not turn into "cannot tell"."""
    out = ops.diagnose_low_utilization(_conn({
        "vllm:num_requests_running": _series(2.0),
        "vllm:gpu_cache_usage_perc": _series(0.1),
    }))
    assert "Low batching" in out["finding"] and out["unreadableSignals"] == ["numWaiting"]


@pytest.mark.unit
def test_latency_rca_does_not_report_an_all_clear_it_could_not_measure():
    out = ops.diagnose_latency_spike(_conn({}))
    assert out["unreadableSignals"] == ["numWaiting", "kvUsage", "preemptions"]
    assert "not evidence that the engine is healthy" in out["probableCauses"][0]["cause"]
    assert out["signalsChecked"]["kvUsage"] is None

    # Partial readings keep their verdict and name what was missing — withholding it
    # there would make the tool useless on engines that expose most of the set.
    partial = ops.diagnose_latency_spike(_conn({
        "vllm:gpu_cache_usage_perc": _series(0.4),
        "vllm:num_requests_waiting": _series(0.0),
    }))
    assert "No dominant bottleneck" in partial["probableCauses"][0]["cause"]
    assert "Not measured: preemptions" in partial["probableCauses"][0]["cause"]
