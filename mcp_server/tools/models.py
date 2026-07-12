"""vLLM model inventory + LoRA / base-model lifecycle MCP tools (read + guarded writes).

LoRA unload and base-model hot-swap are traffic-affecting — risk=high with a
dry_run preview. Hot-swap records an undo capturing the prior base model.
"""

from typing import Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import models as ops
from mcp_server._shared import _get_connection, mcp, tool_errors

# ── reads ────────────────────────────────────────────────────────────────


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def model_list(target: Optional[str] = None) -> list:
    """[READ] All served vLLM models, LoRA adapters flagged.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.list_models(_get_connection(target))


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def model_info(model_id: str, target: Optional[str] = None) -> dict:
    """[READ] One model's config: max length, root/parent, permission (best-effort).

    Args:
        model_id: Model id (from model_list).
        target: Inference target name from config; omit for the default.
    """
    return ops.get_model_info(_get_connection(target), model_id)


# ── writes ───────────────────────────────────────────────────────────────


@mcp.tool()
@governed_tool(risk_level="medium")
@tool_errors("dict")
def lora_load(lora_name: str, lora_path: str, target: Optional[str] = None) -> dict:
    """[WRITE][risk=medium] Hot-load a LoRA adapter onto the running engine.

    Args:
        lora_name: Adapter name to register.
        lora_path: Local path or HF repo id of the adapter weights.
        target: Inference target name from config; omit for the default.
    """
    return ops.load_lora_adapter(_get_connection(target), lora_name, lora_path)


@mcp.tool()
@governed_tool(risk_level="high")
@tool_errors("dict")
def lora_unload(
    lora_name: str, dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Hot-unload a LoRA adapter (traffic on it starts failing).

    Pass dry_run=True to preview. Requires an approver (INFERENCE_AUDIT_APPROVED_BY).

    Args:
        lora_name: Adapter name to unload (from model_list).
        dry_run: If True, preview without unloading.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True, "wouldUnload": {"loraName": lora_name}}
    return ops.unload_lora_adapter(conn, lora_name)


@mcp.tool()
@governed_tool(risk_level="high")
@tool_errors("dict")
def model_hot_swap(
    new_model: str, dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Sleep-Mode base-model swap (reversible → prior model).

    Swaps the loaded base model in place; in-flight requests on the old model
    are affected. Pass dry_run=True to preview. Requires an approver
    (INFERENCE_AUDIT_APPROVED_BY).

    Args:
        new_model: Target base model to load.
        dry_run: If True, preview without swapping.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        cur = ops.list_models(conn)
        current = cur[0].get("id") if cur and isinstance(cur[0], dict) else None
        return {"dryRun": True, "from": current, "to": new_model}
    return ops.hot_swap_model(conn, new_model)
