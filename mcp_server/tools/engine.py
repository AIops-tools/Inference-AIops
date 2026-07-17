"""Engine-agnostic serving-engine MCP tools (vLLM / SGLang / TGI, read-only).

These reads work across every supported serving engine — reading each engine's
own health path, running-model identity, and Prometheus ``/metrics`` (with the
engine's own metric names) behind one canonical surface. They complement the
vLLM/Ray-specific tools: use these for SGLang/TGI targets (which have no Ray
control plane) or for a uniform view across a mixed fleet.
"""

from typing import Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import engine as ops
from mcp_server._shared import _get_connection, mcp, tool_errors


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def engine_health(target: Optional[str] = None) -> dict:
    """[READ] Liveness of the serving engine (vLLM / SGLang / TGI) via its health probe.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.engine_health(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def engine_inventory(target: Optional[str] = None) -> dict:
    """[READ] Running-model identity + engine server info (engine-agnostic).

    vLLM / SGLang report served ids from /v1/models; TGI's single model id
    comes from /info.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.engine_inventory(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def engine_request_metrics(target: Optional[str] = None) -> dict:
    """[READ] TTFT / TPOT / e2e latency + generation-token totals (where the engine exposes them).

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_engine_request_metrics(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def engine_queue_depth(target: Optional[str] = None) -> dict:
    """[READ] Running vs waiting requests — the leading backpressure signal (any engine).

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.get_engine_queue_depth(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def diagnose_engine_latency(target: Optional[str] = None) -> dict:
    """[READ][RCA] Rank the probable cause of a latency spike for any serving engine.

    Correlates whichever signals the engine exposes (queue backpressure,
    KV/token-cache pressure, cache locality) into a ranked cause + the knob to
    turn — the engine-agnostic counterpart to diagnose_latency_spike (vLLM).

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.diagnose_engine_latency(_get_connection(target))
