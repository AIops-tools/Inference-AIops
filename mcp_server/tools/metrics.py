"""vLLM metrics + latency/utilization RCA MCP tools (read-only)."""

from typing import Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import metrics as ops
from mcp_server._shared import _get_connection, mcp, tool_errors


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def request_metrics(target: Optional[str] = None) -> dict:
    """[READ] vLLM TTFT / TPOT / e2e latency + generation-token totals.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_request_metrics(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def queue_depth(target: Optional[str] = None) -> dict:
    """[READ] Running vs waiting requests — the leading backpressure signal.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_queue_depth(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def kv_cache_stats(target: Optional[str] = None) -> dict:
    """[READ] KV-cache utilisation, prefix-cache hit rate, and preemption count.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_kv_cache_stats(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def diagnose_latency_spike(target: Optional[str] = None) -> dict:
    """[READ][RCA] Rank the probable cause of a TTFT/latency spike + the knob to turn.

    Correlates queue depth, KV-cache pressure/preemption, and prefix-cache
    locality into a ranked cause list — call this first on "why is inference slow".

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.diagnose_latency_spike(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def diagnose_low_utilization(target: Optional[str] = None) -> dict:
    """[READ][RCA] Explain an under-used GPU (batching / idle / overprovision).

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.diagnose_low_utilization(_get_connection(target))
