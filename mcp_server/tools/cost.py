"""Cost attribution MCP tools (read-only)."""

from typing import Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import cost as ops
from mcp_server._shared import _get_connection, mcp, tool_errors


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def cost_per_token(
    gpu_hourly_cost: float, num_gpus: int = 1, target: Optional[str] = None
) -> dict:
    """[READ] Attribute a $/1M-token unit cost from live vLLM throughput.

    Multiplies the current generation-throughput gauge by the supplied GPU
    hourly cost to derive the cost of serving 1M tokens; degrades to an
    ``insufficient-data`` forecast when no throughput metric is present.

    Args:
        gpu_hourly_cost: Hourly cost of a single GPU (e.g. cloud on-demand rate).
        num_gpus: Number of GPUs backing the deployment; defaults to 1.
        target: Inference target name from config; omit for the default.
    """
    return ops.get_cost_per_token(_get_connection(target), gpu_hourly_cost, num_gpus)
