"""MCP server wrapping inference-aiops operations (stdio transport).

Thin adapter layer: each ``@mcp.tool()`` function (in ``mcp_server/tools/``)
delegates to the ``inference_aiops`` ops package and is wrapped with the
inference-aiops ``@governed_tool`` harness (audit / budget / undo / risk-tier).

Standalone, self-governed GPU inference-cluster operations (preview) over vLLM +
Ray Serve/Jobs: metrics/RCA, scaling, drain, model lifecycle, jobs, cost.

Source: https://github.com/AIops-tools/Inference-AIops
License: MIT
"""

import logging

from mcp_server._shared import _safe_error, mcp, tool_errors

# Importing the tool modules registers every @mcp.tool() onto the shared
# `mcp` instance. Order does not matter; each module is self-contained.
from mcp_server.tools import (  # noqa: F401 — side effects
    cost,
    deploy,
    engine,
    metrics,
    models,
    ray_cluster,
    serve,
    undo,
)

__all__ = ["mcp", "main", "_safe_error", "tool_errors"]


def main() -> None:
    """Run the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")
