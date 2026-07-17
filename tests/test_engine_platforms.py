"""Engine-agnostic reads across vLLM / SGLang / TGI (+ control-plane guard).

Proves the ops/engine layer maps each serving engine onto one canonical surface:
health probe, running-model inventory (incl. TGI's single model from /info),
per-engine metric-name resolution, queue-depth backpressure, and the latency
RCA thresholds. Also proves the single-process engines (SGLang/TGI) raise the
teaching :class:`EngineCapabilityError` on Ray-shaped control-plane writes while
vLLM does not. No live engine — the connection is a canned fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_aiops.connection import EngineCapabilityError
from inference_aiops.ops import engine as ops


class FakeConn:
    """Canned engine connection: path-keyed JSON + a parsed metrics map."""

    def __init__(self, engine="vllm", json_by_path=None, metrics=None,
                 health_error=None, metrics_error=None):
        self.target = SimpleNamespace(engine=engine)
        self._json = json_by_path or {}
        self._metrics = metrics or {}
        self._health_error = health_error
        self._metrics_error = metrics_error

    def get_engine(self, path, **kwargs):
        if self._health_error is not None and path in ("/health",):
            raise self._health_error
        if path not in self._json:
            raise KeyError(f"no canned path {path}")
        return self._json[path]

    def engine_metrics(self):
        if self._metrics_error is not None:
            raise self._metrics_error
        return self._metrics


def _series(value):
    return [{"labels": {}, "value": value}]


@pytest.mark.unit
def test_engine_health_healthy_and_unhealthy():
    ok = ops.engine_health(FakeConn("vllm", json_by_path={"/health": {}}))
    assert ok == {"engine": "vllm", "label": "vLLM", "healthy": True}

    down = ops.engine_health(FakeConn("sglang", health_error=RuntimeError("connrefused")))
    assert down["engine"] == "sglang" and down["healthy"] is False
    assert "connrefused" in down["error"]


@pytest.mark.unit
def test_engine_inventory_vllm_lists_v1_models():
    conn = FakeConn("vllm", json_by_path={
        "/v1/models": {"data": [{"id": "meta/llama3"}, {"id": "my-lora"}]},
    })
    inv = ops.engine_inventory(conn)
    assert inv["engine"] == "vllm"
    assert inv["models"] == ["meta/llama3", "my-lora"]


@pytest.mark.unit
def test_engine_inventory_tgi_identity_from_info():
    # TGI has no /v1/models; identity comes from /info (model_id).
    conn = FakeConn("tgi", json_by_path={
        "/info": {"model_id": "bigscience/bloom", "version": "2.0", "max_concurrent_requests": 128},
    })
    inv = ops.engine_inventory(conn)
    assert inv["engine"] == "tgi" and inv["label"] == "TGI"
    assert inv["models"] == ["bigscience/bloom"]
    assert inv["serverInfo"]["model"] == "bigscience/bloom"
    assert inv["serverInfo"]["version"] == "2.0"
    assert inv["serverInfo"]["maxConcurrentRequests"] == 128


@pytest.mark.unit
def test_engine_inventory_sglang_uses_server_info_and_models():
    conn = FakeConn("sglang", json_by_path={
        "/get_server_info": {"model_path": "Qwen/Qwen2", "version": "0.4"},
        "/v1/models": {"data": [{"id": "Qwen/Qwen2"}]},
    })
    inv = ops.engine_inventory(conn)
    assert inv["models"] == ["Qwen/Qwen2"]
    assert inv["serverInfo"]["model"] == "Qwen/Qwen2"


@pytest.mark.unit
def test_engine_request_metrics_resolves_per_engine_names():
    # SGLang histogram bases differ from vLLM; the reader must use the spec's names.
    metrics = {
        "sglang:time_to_first_token_seconds_sum": _series(10.0),
        "sglang:time_to_first_token_seconds_count": _series(50.0),
        "sglang:generation_tokens_total": _series(4321.0),
    }
    out = ops.get_engine_request_metrics(FakeConn("sglang", metrics=metrics))
    assert out["engine"] == "sglang"
    assert out["ttftSeconds"] == 0.2  # 10 / 50
    assert out["generationTokensTotal"] == 4321.0
    # TGI exposes no TTFT metric → field degrades to None (not guessed).
    tgi = ops.get_engine_request_metrics(FakeConn("tgi", metrics={}))
    assert tgi["ttftSeconds"] is None


@pytest.mark.unit
def test_engine_queue_depth_backpressure_flag():
    metrics = {"tgi_queue_size": _series(6.0), "tgi_batch_current_size": _series(2.0)}
    out = ops.get_engine_queue_depth(FakeConn("tgi", metrics=metrics))
    assert out["numWaiting"] == 6.0 and out["numRunning"] == 2.0
    assert out["backpressure"] is True

    idle = ops.get_engine_queue_depth(FakeConn("vllm", metrics={}))
    assert idle["backpressure"] is False


@pytest.mark.unit
def test_diagnose_engine_latency_ranks_kv_and_queue():
    # vLLM: high KV usage + queue backpressure → both causes, KV ranked first.
    metrics = {
        "vllm:gpu_cache_usage_perc": _series(0.95),
        "vllm:num_preemptions_total": _series(3.0),
        "vllm:num_requests_waiting": _series(8.0),
    }
    out = ops.diagnose_engine_latency(FakeConn("vllm", metrics=metrics))
    causes = [c["cause"] for c in out["probableCauses"]]
    assert "KV/token-cache pressure" in causes[0]
    assert any("Queue backpressure" in c for c in causes)
    assert out["signalsChecked"]["cacheUsage"] == 0.95


@pytest.mark.unit
def test_diagnose_engine_latency_sglang_cache_hit_rate_normalised():
    # SGLang reports a percentage hit-rate gauge; <30% cold-cache cause fires.
    metrics = {"sglang:cache_hit_rate": _series(12.0)}  # 12% → 0.12 fraction
    out = ops.diagnose_engine_latency(FakeConn("sglang", metrics=metrics))
    assert out["signalsChecked"]["cacheHitRate"] == 0.12
    assert any("Cold prefix cache" in c["cause"] for c in out["probableCauses"])


@pytest.mark.unit
def test_diagnose_engine_latency_no_bottleneck_default():
    out = ops.diagnose_engine_latency(FakeConn("vllm", metrics={}))
    assert len(out["probableCauses"]) == 1
    assert "No dominant bottleneck" in out["probableCauses"][0]["cause"]


@pytest.mark.unit
def test_diagnose_engine_latency_scrape_failure_degrades():
    out = ops.diagnose_engine_latency(FakeConn("vllm", metrics_error=RuntimeError("scrape 500")))
    assert "scrape 500" in out["error"]


@pytest.mark.unit
@pytest.mark.parametrize("engine", ["sglang", "tgi"])
def test_control_plane_guard_raises_teaching_error_for_single_process(engine):
    with pytest.raises(EngineCapabilityError) as ei:
        ops.require_control_plane(FakeConn(engine), "scale_replicas")
    msg = str(ei.value)
    assert "single-process" in msg
    assert "scale_replicas" in msg


@pytest.mark.unit
def test_control_plane_guard_noop_for_vllm_and_unknown():
    # vLLM has a control plane → no raise.
    ops.require_control_plane(FakeConn("vllm"), "drain_replica")
    # A mock conn whose engine is not a string defaults to vllm → no raise.
    ops.require_control_plane(SimpleNamespace(target=SimpleNamespace(engine=object())), "x")
