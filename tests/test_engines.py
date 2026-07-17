"""Tests for the multi-engine surface: SGLang + TGI alongside vLLM.

Covers the engine registry, config parsing/validation of the ``engine`` field,
the connection layer's engine-agnostic HTTP surface (correct port + teaching
label), the engine-agnostic reads over canned SGLang/TGI Prometheus exposition
text, and the control-plane teaching error raised when a Ray-shaped write hits a
single-process engine. No live engine is needed — httpx is faked/parsed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from inference_aiops.connection import EngineCapabilityError, parse_prometheus
from inference_aiops.engines import SUPPORTED_ENGINES, get_engine_spec
from inference_aiops.ops import engine as eng

pytestmark = pytest.mark.unit


# ── canned Prometheus exposition text (real metric names) ────────────────────

SGLANG_METRICS = (
    "# HELP sglang:num_running_reqs Running requests\n"
    "# TYPE sglang:num_running_reqs gauge\n"
    'sglang:num_running_reqs{model_name="llama-3-8b"} 4.0\n'
    'sglang:num_queue_reqs{model_name="llama-3-8b"} 3.0\n'
    'sglang:token_usage{model_name="llama-3-8b"} 0.94\n'
    'sglang:cache_hit_rate{model_name="llama-3-8b"} 22.5\n'
    'sglang:time_to_first_token_seconds_sum{model_name="llama-3-8b"} 12.0\n'
    'sglang:time_to_first_token_seconds_count{model_name="llama-3-8b"} 40.0\n'
    "sglang:inter_token_latency_seconds_sum 3.0\n"
    "sglang:inter_token_latency_seconds_count 300.0\n"
    "sglang:e2e_request_latency_seconds_sum 80.0\n"
    "sglang:e2e_request_latency_seconds_count 40.0\n"
    "sglang:generation_tokens_total 5000\n"
)

TGI_METRICS = (
    "# HELP tgi_queue_size Queue size\n"
    "# TYPE tgi_queue_size gauge\n"
    "tgi_queue_size 6\n"
    "tgi_batch_current_size 2\n"
    "tgi_request_duration_sum 40.0\n"
    "tgi_request_duration_count 20.0\n"
    "tgi_request_mean_time_per_token_duration_sum 2.0\n"
    "tgi_request_mean_time_per_token_duration_count 200.0\n"
)


class _FakeEngineConn:
    """A connection stand-in: parses canned metrics text; dispatches get_engine by path."""

    def __init__(
        self,
        engine: str,
        metrics_text: str = "",
        responses: dict[str, Any] | None = None,
        health_exc: Exception | None = None,
        metrics_exc: Exception | None = None,
    ) -> None:
        self.target = SimpleNamespace(engine=engine, engine_url="http://host:0")
        self._metrics_text = metrics_text
        self._responses = responses or {}
        self._health_exc = health_exc
        self._metrics_exc = metrics_exc

    def get_engine(self, path: str, **_: Any) -> Any:
        spec = get_engine_spec(self.target.engine)
        if path == spec.health_path and self._health_exc is not None:
            raise self._health_exc
        val = self._responses.get(path, {})
        if isinstance(val, Exception):
            raise val
        return val

    def engine_metrics(self) -> dict:
        if self._metrics_exc is not None:
            raise self._metrics_exc
        return parse_prometheus(self._metrics_text)


# ── registry ─────────────────────────────────────────────────────────────────


def test_supported_engines_are_vllm_sglang_tgi():
    assert set(SUPPORTED_ENGINES) == {"vllm", "sglang", "tgi"}


def test_get_engine_spec_is_case_insensitive_and_typed():
    assert get_engine_spec("SGLang").name == "sglang"
    assert get_engine_spec("vllm").has_control_plane is True
    assert get_engine_spec("sglang").has_control_plane is False
    assert get_engine_spec("tgi").has_control_plane is False
    assert get_engine_spec("tgi").models_path is None  # single model → /info identity


def test_get_engine_spec_unknown_raises():
    with pytest.raises(ValueError, match="Unknown serving engine"):
        get_engine_spec("triton")


# ── config: the engine field ─────────────────────────────────────────────────


def test_target_defaults_to_vllm_with_control_plane():
    from inference_aiops.config import TargetConfig

    t = TargetConfig(name="t", host="h")
    assert t.engine == "vllm"
    assert t.has_control_plane is True
    assert t.engine_url == t.vllm_url


def test_load_config_parses_sglang_default_port(tmp_path):
    from inference_aiops.config import load_config

    (tmp_path / "c.yaml").write_text(
        "targets:\n  - name: sg\n    host: sg.local\n    engine: sglang\n", "utf-8"
    )
    cfg = load_config(tmp_path / "c.yaml")
    t = cfg.get_target("sg")
    assert t.engine == "sglang"
    assert t.engine_port == 30000  # SGLang default
    assert t.engine_url == "http://sg.local:30000"
    assert t.has_control_plane is False


def test_load_config_tgi_engine_port_key(tmp_path):
    from inference_aiops.config import load_config

    (tmp_path / "c.yaml").write_text(
        "targets:\n  - name: e\n    host: e.local\n    engine: tgi\n    engine_port: 8085\n",
        "utf-8",
    )
    t = load_config(tmp_path / "c.yaml").get_target("e")
    assert t.engine == "tgi" and t.engine_port == 8085


def test_load_config_legacy_vllm_port_still_wins(tmp_path):
    from inference_aiops.config import load_config

    (tmp_path / "c.yaml").write_text(
        "targets:\n  - name: p\n    host: p.local\n    vllm_port: 9000\n", "utf-8"
    )
    t = load_config(tmp_path / "c.yaml").get_target("p")
    assert t.engine == "vllm" and t.vllm_port == 9000


def test_load_config_rejects_unknown_engine(tmp_path):
    from inference_aiops.config import load_config

    (tmp_path / "c.yaml").write_text(
        "targets:\n  - name: x\n    host: x.local\n    engine: nope\n", "utf-8"
    )
    with pytest.raises(ValueError, match="unknown serving engine"):
        load_config(tmp_path / "c.yaml")


# ── connection: engine-agnostic HTTP surface ─────────────────────────────────


def test_get_engine_routes_to_engine_port_and_labels_errors():
    from inference_aiops.config import TargetConfig
    from inference_aiops.connection import InferenceApiError, InferenceConnection

    seen: dict[str, str] = {}

    class _Client:
        def __init__(self, status: int) -> None:
            self._status = status

        def request(self, method: str, url: str, **_: Any):
            seen["url"] = url
            return SimpleNamespace(
                status_code=self._status, content=b"{}", text="boom", json=lambda: {}
            )

        def close(self) -> None:
            pass

    target = TargetConfig(name="sg", host="sg.local", vllm_port=30000, engine="sglang")
    conn = InferenceConnection(target, client=_Client(200))
    conn.get_engine("/health")
    assert seen["url"] == "http://sg.local:30000/health"

    conn_err = InferenceConnection(target, client=_Client(503))
    with pytest.raises(InferenceApiError, match="SGLang"):
        conn_err.get_engine("/health")


def test_engine_metrics_parses_sglang_via_real_connection():
    from inference_aiops.config import TargetConfig
    from inference_aiops.connection import InferenceConnection

    class _Client:
        def request(self, method: str, url: str, **_: Any):
            assert url.endswith(":30000/metrics")
            return SimpleNamespace(status_code=200, content=b"x", text=SGLANG_METRICS,
                                   json=lambda: {})

        def close(self) -> None:
            pass

    target = TargetConfig(name="sg", host="sg.local", vllm_port=30000, engine="sglang")
    m = InferenceConnection(target, client=_Client()).engine_metrics()
    assert m["sglang:num_running_reqs"][0]["value"] == 4.0
    assert m["sglang:cache_hit_rate"][0]["value"] == 22.5


def test_prometheus_parser_handles_tgi_histograms():
    m = parse_prometheus(TGI_METRICS)
    assert m["tgi_queue_size"][0]["value"] == 6.0
    assert m["tgi_request_duration_sum"][0]["value"] == 40.0
    assert m["tgi_request_duration_count"][0]["value"] == 20.0


# ── engine-agnostic reads ────────────────────────────────────────────────────


def test_engine_health_healthy_and_unhealthy():
    ok = eng.engine_health(_FakeEngineConn("sglang"))
    assert ok == {"engine": "sglang", "label": "SGLang", "healthy": True}

    down = eng.engine_health(_FakeEngineConn("tgi", health_exc=ConnectionError("refused")))
    assert down["healthy"] is False and "refused" in down["error"]


def test_engine_inventory_sglang_from_v1_models():
    conn = _FakeEngineConn(
        "sglang",
        responses={
            "/v1/models": {"data": [{"id": "llama-3-8b"}, {"id": "adapter"}]},
            "/get_server_info": {"model_path": "meta/llama-3-8b", "version": "0.4.1"},
        },
    )
    inv = eng.engine_inventory(conn)
    assert inv["models"] == ["llama-3-8b", "adapter"]
    assert inv["serverInfo"]["model"] == "meta/llama-3-8b"
    assert inv["serverInfo"]["version"] == "0.4.1"


def test_engine_inventory_tgi_from_info():
    conn = _FakeEngineConn(
        "tgi",
        responses={"/info": {"model_id": "bigscience/bloom", "max_concurrent_requests": 128}},
    )
    inv = eng.engine_inventory(conn)
    assert inv["models"] == ["bigscience/bloom"]
    assert inv["serverInfo"]["maxConcurrentRequests"] == 128


def test_engine_request_metrics_sglang():
    out = eng.get_engine_request_metrics(_FakeEngineConn("sglang", SGLANG_METRICS))
    assert out["ttftSeconds"] == round(12.0 / 40.0, 4)
    assert out["tpotSeconds"] == round(3.0 / 300.0, 4)
    assert out["e2eLatencySeconds"] == round(80.0 / 40.0, 4)
    assert out["generationTokensTotal"] == 5000.0


def test_engine_request_metrics_tgi_omits_unexposed_ttft():
    out = eng.get_engine_request_metrics(_FakeEngineConn("tgi", TGI_METRICS))
    assert out["ttftSeconds"] is None  # TGI exposes no TTFT
    assert out["tpotSeconds"] == round(2.0 / 200.0, 4)
    assert out["e2eLatencySeconds"] == round(40.0 / 20.0, 4)
    assert out["generationTokensTotal"] is None


def test_engine_queue_depth_tgi():
    out = eng.get_engine_queue_depth(_FakeEngineConn("tgi", TGI_METRICS))
    assert out["numWaiting"] == 6.0 and out["numRunning"] == 2.0
    assert out["backpressure"] is True


def test_diagnose_engine_latency_sglang_ranks_cache_pressure():
    out = eng.diagnose_engine_latency(_FakeEngineConn("sglang", SGLANG_METRICS))
    causes = out["probableCauses"]
    # token_usage 0.94 ≥ 0.9 → cache pressure ranked first.
    assert "cache pressure" in causes[0]["cause"]
    # cache_hit_rate 22.5% normalises to 0.225 → cold-cache cause present.
    assert out["signalsChecked"]["cacheHitRate"] == 0.225
    assert any("Cold prefix cache" in c["cause"] for c in causes)


def test_diagnose_engine_latency_resilient_to_scrape_failure():
    conn = _FakeEngineConn("tgi", metrics_exc=RuntimeError("scrape failed"))
    out = eng.diagnose_engine_latency(conn)
    assert "error" in out


# ── control-plane teaching error ─────────────────────────────────────────────


def test_require_control_plane_raises_for_single_process_engines():
    for engine in ("sglang", "tgi"):
        with pytest.raises(EngineCapabilityError, match="control-plane"):
            eng.require_control_plane(_FakeEngineConn(engine), "scale_replicas")


def test_require_control_plane_noop_for_vllm_and_mock_conns():
    from unittest.mock import MagicMock

    eng.require_control_plane(_FakeEngineConn("vllm"), "scale_replicas")  # no raise
    eng.require_control_plane(MagicMock(name="conn"), "scale_replicas")  # no raise


def test_ray_scale_write_raises_teaching_error_on_sglang_target():
    """A Ray-shaped scale write against a real SGLang connection teaches, not 500s."""
    from inference_aiops.config import TargetConfig
    from inference_aiops.connection import InferenceConnection
    from inference_aiops.ops import serve as sv

    class _Client:
        def request(self, *a: Any, **k: Any):  # must never be called
            raise AssertionError("no HTTP call should be issued for a guarded engine")

        def close(self) -> None:
            pass

    target = TargetConfig(name="sg", host="sg.local", vllm_port=30000, engine="sglang")
    conn = InferenceConnection(target, client=_Client())
    with pytest.raises(EngineCapabilityError, match="SGLang"):
        sv.scale_to_zero(conn, "app", "dep")
