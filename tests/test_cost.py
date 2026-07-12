"""Cost-attribution ops + MCP tool tests for inference-aiops.

Proves: the $/token read multiplies live vLLM throughput into a unit cost,
degrades to an insufficient-data forecast when the throughput gauge is absent,
and the MCP tool carries the harness marker + a low risk tier. No real vLLM is
needed — the connection is a MagicMock.
"""

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_cost_per_token_computes_cost_per_1m():
    from inference_aiops.ops import cost as ops

    conn = MagicMock(name="conn")
    conn.vllm_metrics.return_value = {
        "vllm:avg_generation_throughput_toks_per_s": [{"labels": {}, "value": 1000.0}],
    }
    out = ops.get_cost_per_token(conn, gpu_hourly_cost=2.0, num_gpus=1)
    assert out["throughputTokPerSec"] == 1000.0
    assert out["tokensPerHour"] == 3_600_000.0
    assert out["clusterHourlyCost"] == 2.0
    # 2.0 / (3.6M / 1M) = 2.0 / 3.6 = 0.5556
    assert out["costPer1MTokens"] == 0.5556


@pytest.mark.unit
def test_cost_per_token_insufficient_data_when_no_throughput():
    from inference_aiops.ops import cost as ops

    conn = MagicMock(name="conn")
    conn.vllm_metrics.return_value = {}
    out = ops.get_cost_per_token(conn, gpu_hourly_cost=2.0, num_gpus=4)
    assert out["forecast"] == "insufficient-data"
    assert out["reason"] == "no throughput metric"
    assert out["numGpus"] == 4


@pytest.mark.unit
def test_cost_tool_is_governed_and_low_risk():
    from mcp_server.tools import cost

    assert cost.cost_per_token._risk_level == "low"
    assert getattr(cost.cost_per_token, "_is_governed_tool", False)
