"""Smoke + ops tests for inference-aiops.

Proves: every module imports, the CLI builds and --help works, the MCP server
exposes the expected tools and EVERY tool carries the harness marker
``_is_governed_tool``, the Prometheus parser + dual-backend connection work, the
flagship latency RCA correlates signals into a ranked cause, and the Ray Serve
writes capture BEFORE-state, record undo, gate dry-run, and carry correct risk
tiers. No real vLLM/Ray is needed — the connection is a fake/MagicMock.
"""

import asyncio
import importlib
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

# Kept in sync with mcp_server/server.py (the full registered tool surface).
EXPECTED_TOOLS = {
    # metrics
    "request_metrics", "queue_depth", "kv_cache_stats",
    "diagnose_latency_spike", "diagnose_low_utilization",
    # serve
    "serve_deployment_list", "deployment_status", "replica_list", "autoscale_config_get",
    "scale_replicas_up", "scale_replicas_down", "scale_to_zero",
    "autoscale_config_update", "drain_replica",
    # models
    "model_list", "model_info", "model_is_sleeping", "lora_load", "lora_unload",
    "model_sleep", "model_wake",
    # ray cluster / jobs / gpu
    "ray_cluster_resources", "ray_dashboard_status", "ray_job_list", "gpu_utilization",
    "ray_job_cancel", "replica_restart",
    # deploy lifecycle
    "model_deploy", "model_undeploy", "deployment_redeploy", "routing_policy_update",
    # cost
    "cost_per_token",
    # engine-agnostic (vLLM / SGLang / TGI)
    "engine_health", "engine_inventory", "engine_request_metrics",
    "engine_queue_depth", "diagnose_engine_latency",
}


@pytest.mark.unit
def test_all_modules_import():
    for name in (
        "inference_aiops", "inference_aiops.config", "inference_aiops.connection",
        "inference_aiops.doctor", "inference_aiops.secretstore",
        "inference_aiops.ops.metrics", "inference_aiops.ops.serve",
        "inference_aiops.ops.overview", "inference_aiops.ops.engine",
        "inference_aiops.engines",
        "inference_aiops.cli", "inference_aiops.cli._root", "inference_aiops.cli._common",
        "inference_aiops.cli.init", "inference_aiops.cli.secret", "inference_aiops.cli.serve",
        "inference_aiops.cli.metrics", "inference_aiops.cli.overview",
        "inference_aiops.cli.doctor",
        "mcp_server.server", "mcp_server._shared",
        "mcp_server.tools.metrics", "mcp_server.tools.serve",
        "mcp_server.tools.engine",
    ):
        importlib.import_module(name)


@pytest.mark.unit
def test_version_matches_pyproject():
    """__version__ is single-sourced from package metadata; it must track
    pyproject.toml so a release bump can never ship a stale self-report."""
    import tomllib
    from pathlib import Path

    import inference_aiops

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text("utf-8"))["project"]["version"]
    assert inference_aiops.__version__ == expected


@pytest.mark.unit
def test_cli_app_builds_and_help_works():
    from inference_aiops.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("serve", "metrics", "secret", "init", "overview", "doctor", "mcp"):
        assert sub in result.output


@pytest.mark.unit
def test_cli_leaf_help_triggers_lazy_imports():
    from inference_aiops.cli import app

    runner = CliRunner()
    for cmd in (
        ["serve", "--help"], ["metrics", "--help"], ["secret", "--help"],
        ["doctor", "--help"], ["overview", "--help"], ["init", "--help"],
        ["serve", "list", "--help"], ["serve", "status", "--help"],
        ["serve", "scale", "--help"], ["serve", "scale-to-zero", "--help"],
        ["metrics", "requests", "--help"], ["metrics", "queue", "--help"],
        ["metrics", "diagnose", "--help"],
        ["secret", "list", "--help"], ["secret", "set", "--help"],
    ):
        result = runner.invoke(app, cmd)
        assert result.exit_code == 0, f"{cmd} failed: {result.output}"


@pytest.mark.unit
def test_mcp_list_tools_exposes_expected_tools():
    from mcp_server.server import mcp

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names, f"missing: {EXPECTED_TOOLS - names}"


@pytest.mark.unit
def test_every_mcp_tool_is_governed_by_harness():
    from mcp_server import _shared

    tool_objs = _shared.mcp._tool_manager._tools
    assert EXPECTED_TOOLS <= set(tool_objs), "tool registry incomplete"
    for name, tool in tool_objs.items():
        fn = getattr(tool, "fn", None)
        assert fn is not None, f"{name} has no fn"
        assert getattr(fn, "_is_governed_tool", False), f"{name} missing @governed_tool"


@pytest.mark.unit
def test_write_tools_have_correct_risk_tiers():
    from mcp_server.tools import serve as sv

    assert sv.scale_replicas_up._risk_level == "medium"
    assert sv.scale_replicas_down._risk_level == "high"
    assert sv.scale_to_zero._risk_level == "high"
    assert sv.drain_replica._risk_level == "high"
    assert sv.autoscale_config_update._risk_level == "medium"


# ── Prometheus parser + dual-backend connection ─────────────────────────


class _Resp:
    def __init__(self, status, payload=None, text="", content=b"{}"):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text
        self.content = content

    def json(self):
        return self._payload


@pytest.mark.unit
def test_prometheus_parser():
    from inference_aiops.connection import parse_prometheus

    text = (
        "# HELP vllm:num_requests_waiting x\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        'vllm:num_requests_waiting{model="m"} 5.0\n'
        "vllm:generation_tokens_total 1200\n"
    )
    m = parse_prometheus(text)
    assert m["vllm:num_requests_waiting"][0]["value"] == 5.0
    assert m["vllm:num_requests_waiting"][0]["labels"]["model"] == "m"
    assert m["vllm:generation_tokens_total"][0]["value"] == 1200.0


@pytest.mark.unit
def test_connection_routes_ray_vs_vllm_and_optional_auth(monkeypatch):
    from inference_aiops.config import TargetConfig
    from inference_aiops.connection import InferenceConnection

    target = TargetConfig(name="prod", host="gpu.local")  # no token → open
    seen = {}

    class _Client:
        def request(self, method, url, **k):
            seen["url"] = url
            if url.endswith("/metrics"):
                return _Resp(200, text="vllm:num_requests_running 2.0\n")
            return _Resp(200, {"applications": {}})

        def close(self):
            pass

    conn = InferenceConnection(target, client=_Client())
    conn.get_ray("/api/serve/applications/")
    assert "8265" in seen["url"]
    conn.get_vllm("/v1/models")
    assert "8000" in seen["url"]
    m = conn.vllm_metrics()
    assert m["vllm:num_requests_running"][0]["value"] == 2.0


# ── flagship latency RCA ─────────────────────────────────────────────────


@pytest.mark.unit
def test_diagnose_latency_spike_ranks_kv_pressure():
    from inference_aiops.ops import metrics as ops

    conn = MagicMock(name="conn")
    conn.vllm_metrics.return_value = {
        "vllm:num_requests_waiting": [{"labels": {}, "value": 0.0}],
        "vllm:gpu_cache_usage_perc": [{"labels": {}, "value": 0.95}],
        "vllm:num_preemptions_total": [{"labels": {}, "value": 12.0}],
    }
    out = ops.diagnose_latency_spike(conn)
    assert out["probableCauses"]
    assert "KV-cache" in out["probableCauses"][0]["cause"]


@pytest.mark.unit
def test_diagnose_latency_spike_resilient():
    from inference_aiops.ops import metrics as ops

    conn = MagicMock(name="conn")
    conn.vllm_metrics.side_effect = RuntimeError("scrape failed")
    out = ops.diagnose_latency_spike(conn)
    assert "error" in out


# ── Ray Serve writes: undo, before-state, dry-run ───────────────────────


def _apps(num=3):
    return {"applications": {"app1": {"deployments": {"dep1": {
        "status": "HEALTHY",
        "deployment_config": {"num_replicas": num,
                              "autoscaling_config": {"min_replicas": 1, "max_replicas": 8}},
        "replicas": [{"state": "RUNNING"}] * num,
    }}}}}


@pytest.mark.unit
def test_scale_to_zero_captures_prior_and_records_undo(monkeypatch):
    import inference_aiops.governance.undo as undo_mod
    from mcp_server.tools import serve as sv

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=3)
    conn.put_ray.return_value = {}
    monkeypatch.setattr(sv, "_get_connection", lambda target=None: conn)

    recorded = {}

    class _Store:
        def record(self, *, skill, tool, undo_descriptor, orig_params, effect_verified=True):
            recorded["d"] = undo_descriptor
            return "undo-1"

    monkeypatch.setattr(undo_mod, "get_undo_store", lambda: _Store())

    result = sv.scale_to_zero(application="app1", deployment="dep1")
    assert result["priorState"]["numReplicas"] == 3
    assert recorded["d"]["tool"] == "scale_replicas_up"
    assert recorded["d"]["params"]["num_replicas"] == 3  # restore prior
    assert result.get("_undo_id") == "undo-1"


@pytest.mark.unit
def test_scale_to_zero_dry_run_does_not_mutate(monkeypatch):
    from mcp_server.tools import serve as sv

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=4)
    monkeypatch.setattr(sv, "_get_connection", lambda target=None: conn)

    result = sv.scale_to_zero(application="app1", deployment="dep1", dry_run=True)
    assert result["dryRun"] is True and result["from"] == 4 and result["to"] == 0
    conn.put_ray.assert_not_called()


@pytest.mark.unit
def test_cli_serve_scale_zero_dry_run_gates():
    from inference_aiops.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "scale-to-zero", "app1", "dep1", "--dry-run"])
    assert result.exit_code == 0
    assert "DRY-RUN" in result.output


@pytest.mark.unit
def test_queue_depth_backpressure_flag():
    from inference_aiops.ops import metrics as ops

    conn = MagicMock(name="conn")
    conn.vllm_metrics.return_value = {
        "vllm:num_requests_waiting": [{"labels": {}, "value": 7.0}],
        "vllm:num_requests_running": [{"labels": {}, "value": 4.0}],
    }
    out = ops.get_queue_depth(conn)
    assert out["numWaiting"] == 7.0 and out["backpressure"] is True
