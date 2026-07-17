"""InferenceConnection: Prometheus parsing + teaching errors + manager.

Drives a real :class:`InferenceConnection`/`ConnectionManager` with an injected
fake httpx client (never a live vLLM/Ray stack). Proves: the ``/metrics``
exposition text is parsed into ``{name: [{labels, value}]}``, non-2xx statuses
become teaching :class:`InferenceApiError`s, transport errors are wrapped, and
the manager reuses + tears down sessions.
"""

from __future__ import annotations

import httpx
import pytest

from inference_aiops.config import AppConfig, TargetConfig
from inference_aiops.connection import (
    ConnectionManager,
    InferenceApiError,
    InferenceConnection,
    parse_prometheus,
)

# A slice of a real vLLM Prometheus /metrics exposition (HELP/TYPE + label sets).
_VLLM_METRICS = """\
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama3"} 7.0
# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama3"} 4.0
vllm:gpu_cache_usage_perc{model_name="llama3"} 0.93
vllm:num_preemptions_total{model_name="llama3"} 12.0
vllm:time_to_first_token_seconds_sum{model_name="llama3"} 20.0
vllm:time_to_first_token_seconds_count{model_name="llama3"} 100.0
malformed_line_without_value
another:metric_no_labels 3.5 169900000
"""


class _Resp:
    def __init__(self, status: int = 200, text: str = "", payload=None) -> None:
        self.status_code = status
        self.text = text
        self._payload = payload
        self.content = text.encode() if text else (b"{}" if payload is not None else b"")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Client:
    """Fake httpx client recording the last request; returns a queued response."""

    def __init__(self, resp=None, raise_exc=None) -> None:
        self._resp = resp or _Resp(200, payload={})
        self._raise = raise_exc
        self.seen: dict = {}
        self.closed = False

    def request(self, method: str, url: str, **kwargs):
        self.seen = {"method": method, "url": url, "kwargs": kwargs}
        if self._raise is not None:
            raise self._raise
        return self._resp

    def close(self) -> None:
        self.closed = True


def _conn(client, engine: str = "vllm") -> InferenceConnection:
    return InferenceConnection(
        TargetConfig(name="t", host="gpu.local", engine=engine), client=client
    )


@pytest.mark.unit
def test_parse_prometheus_extracts_labels_and_skips_malformed():
    out = parse_prometheus(_VLLM_METRICS)
    assert out["vllm:num_requests_waiting"][0]["value"] == 7.0
    assert out["vllm:num_requests_waiting"][0]["labels"] == {"model_name": "llama3"}
    assert out["vllm:gpu_cache_usage_perc"][0]["value"] == 0.93
    # label-less line with a trailing timestamp: value is the first token.
    assert out["another:metric_no_labels"][0]["value"] == 3.5
    assert out["another:metric_no_labels"][0]["labels"] == {}
    # A line with no parseable value must be dropped, not crash the parse.
    assert "malformed_line_without_value" not in out


@pytest.mark.unit
def test_vllm_metrics_fetches_and_parses_from_client():
    client = _Client(_Resp(200, text=_VLLM_METRICS))
    parsed = _conn(client).vllm_metrics()
    assert parsed["vllm:num_requests_running"][0]["value"] == 4.0
    assert client.seen["url"].endswith("/metrics")


@pytest.mark.unit
def test_engine_metrics_uses_engine_url_and_label():
    client = _Client(_Resp(200, text="tgi_queue_size 5.0\n"))
    parsed = _conn(client, engine="tgi").engine_metrics()
    assert parsed["tgi_queue_size"][0]["value"] == 5.0
    # engine_url == vllm_url here (same host/port), path is /metrics.
    assert client.seen["url"].endswith("/metrics")
    assert client.seen["method"] == "GET"


@pytest.mark.unit
@pytest.mark.parametrize(
    "status,needle",
    [
        (401, "Authentication failed"),
        (404, "Not found"),
        (503, "server error"),
        (418, "API error"),
    ],
)
def test_teaching_error_messages_by_status(status, needle):
    client = _Client(_Resp(status, text="boom detail"))
    with pytest.raises(InferenceApiError) as ei:
        _conn(client).get_ray("/api/serve/applications/")
    assert needle in str(ei.value)
    assert ei.value.status_code == status
    assert ei.value.path == "/api/serve/applications/"


@pytest.mark.unit
def test_transport_error_is_wrapped_with_reachability_hint():
    client = _Client(raise_exc=httpx.ConnectError("refused"))
    with pytest.raises(InferenceApiError) as ei:
        _conn(client).get_vllm("/v1/models")
    assert "Could not reach vLLM" in str(ei.value)


@pytest.mark.unit
def test_empty_and_nonjson_bodies_degrade_to_empty_dict():
    # 204-style empty body → {}
    assert _conn(_Client(_Resp(200, text=""))).get_ray("/api/jobs/") == {}
    # 200 with non-JSON body → {}
    assert _conn(_Client(_Resp(200, text="<html>nope</html>"))).get_vllm("/v1/models") == {}


@pytest.mark.unit
def test_bearer_header_set_only_when_token_present(monkeypatch):
    # No token configured → no Authorization header on the real httpx client.
    conn = InferenceConnection(TargetConfig(name="noauth", host="h"))
    assert "Authorization" not in conn._client.headers
    conn.close()


@pytest.mark.unit
def test_connection_manager_caches_and_disconnects():
    cfg = AppConfig(targets=(TargetConfig(name="a", host="h1"),
                             TargetConfig(name="b", host="h2")))
    mgr = ConnectionManager(cfg)
    assert set(mgr.list_targets()) == {"a", "b"}
    first = mgr.connect("a")
    assert mgr.connect("a") is first  # cached, same session
    assert mgr.list_connected() == ["a"]
    mgr.connect()  # default target → "a" (already cached)
    mgr.disconnect_all()
    assert mgr.list_connected() == []


@pytest.mark.unit
def test_connection_manager_from_config_uses_given_config():
    cfg = AppConfig(targets=(TargetConfig(name="only", host="h"),))
    mgr = ConnectionManager.from_config(cfg)
    assert mgr.connect().target.name == "only"
