"""Shared MCP server primitives: the FastMCP instance, connection helper,
error sanitisation, and the ``@tool_errors`` decorator.

Tool modules under ``mcp_server/tools/`` import ``mcp`` from here and register
their ``@mcp.tool()`` functions onto it. ``mcp_server/server.py`` then imports
those modules and runs the server.

Keep ``Optional[X]`` (never PEP 604 ``X | None``) in any FastMCP-reflected
tool signature — on older mcp/pydantic the union eval'd to ``types.UnionType``
crashes FastMCP's ``issubclass`` check.
"""

import functools
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from inference_aiops.config import load_config
from inference_aiops.connection import ConnectionManager, InferenceApiError
from inference_aiops.governance import sanitize

logger = logging.getLogger(__name__)

_DOCTOR_HINT = "Run 'inference-aiops doctor' to verify connectivity and credentials."


def _safe_error(exc: Exception, tool: str) -> str:
    """Return an agent-safe error string; log full detail server-side only."""
    logger.error("Tool %s failed", tool, exc_info=True)
    _passthrough = (
        ValueError,
        FileNotFoundError,
        KeyError,
        PermissionError,
        TimeoutError,
        ConnectionError,
        InferenceApiError,
    )
    if isinstance(exc, _passthrough):
        return sanitize(str(exc), 300)
    return f"{type(exc).__name__}: operation failed."


def tool_errors(shape: str = "dict") -> Callable:
    """Wrap a tool body in the canonical try/except → ``_safe_error`` pattern.

    Place this *between* ``@governed_tool`` and the function so the audit
    decorator and FastMCP still see the original signature.
    """

    def decorator(func: Callable) -> Callable:
        name = func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — sanitised below
                msg = _safe_error(e, name)
                if shape == "list":
                    return [{"error": msg, "hint": _DOCTOR_HINT}]
                if shape == "str":
                    return f"Error: {msg} {_DOCTOR_HINT}"
                return {"error": msg, "hint": _DOCTOR_HINT}

        return wrapper

    return decorator


mcp = FastMCP(
    "inference-aiops",
    instructions=(
        "GPU inference-cluster operations (preview) over vLLM + Ray Serve/Jobs, "
        "plus SGLang and TGI single-process engines: vLLM request metrics "
        "(TTFT/TPOT/queue depth/KV-cache) and the flagship 'diagnose_latency_spike' "
        "/ 'diagnose_low_utilization' RCA correlators; engine-agnostic reads "
        "(engine_health / engine_inventory / engine_request_metrics / "
        "engine_queue_depth / diagnose_engine_latency) that also cover SGLang/TGI; "
        "Ray Serve deployment/replica/autoscale reads; vLLM model list + LoRA "
        "load/unload and model hot-swap; Ray cluster/jobs and GPU utilisation; "
        "and cost-per-token. Guarded writes cover scaling, autoscale tuning, and "
        "the fragile prod ops — scale-down/scale-to-zero/drain/undeploy/hot-swap "
        "are risk=high with a dry_run preview and require an approver. Every tool "
        "runs through the inference-aiops governance harness (audit / budget / "
        "risk-tier / undo)."
    ),
)

_conn_mgr: Optional[ConnectionManager] = None


def _get_connection(target: Optional[str] = None) -> Any:
    """Return a Inference connection, lazily initialising the manager."""
    global _conn_mgr  # noqa: PLW0603
    if _conn_mgr is None:
        config_path_str = os.environ.get("INFERENCE_AIOPS_CONFIG")
        config_path = Path(config_path_str) if config_path_str else None
        _conn_mgr = ConnectionManager(load_config(config_path))
    return _conn_mgr.connect(target)
