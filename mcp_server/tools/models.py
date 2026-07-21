"""vLLM model inventory + LoRA / Sleep-Mode lifecycle MCP tools (read + guarded writes).

LoRA unload and Sleep-Mode suspend are traffic-affecting — risk=high with a
dry_run preview. ``model_sleep`` records an undo (``model_wake``) only when it
observed the engine awake beforehand, so an already-sleeping engine is never
woken by an undo that did not put it there.

``model_wake`` records NO undo on purpose: vLLM reports *whether* the engine is
sleeping but never at which level, so re-sleeping would have to guess between
level 1 (offload) and level 2 (discard) — two materially different operations.
Guessing a prior state is exactly what the undo contract forbids, so the inverse
is left to the operator, who knows which level they want.

Both dry runs probe through :func:`ops.preview_sleep_state`, which raises the
same dev-mode capability error the real call would. A preview reporting
``currentlySleeping: null`` on a server that has no Sleep-Mode routes would read
as "clear to proceed" and cost an approval to learn otherwise.
"""

from typing import Any, Optional

from inference_aiops.governance import governed_tool
from inference_aiops.ops import models as ops
from mcp_server._shared import _get_connection, mcp, tool_errors


def _sleep_undo(params: dict[str, Any], result: Any) -> Optional[dict]:
    """Inverse of a sleep: wake the engine — but only if WE put it to sleep.

    ``wasSleeping`` is True (already asleep, so this call changed nothing) or
    None (the probe could not tell) → no descriptor. Recording one in either
    case would offer to "restore" a state the engine was never in.
    """
    if not isinstance(result, dict):
        return None
    if (result.get("priorState") or {}).get("wasSleeping") is not False:
        return None
    return {"tool": "model_wake",
            "params": {"target": params.get("target")},
            "note": "Wake the engine, which was serving before this sleep."}


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


@mcp.tool()
@governed_tool(risk_level="low")
@tool_errors("dict")
def model_is_sleeping(target: Optional[str] = None) -> dict:
    """[READ] Whether the vLLM engine is suspended in Sleep Mode.

    Returns isSleeping: true (suspended, serving nothing), false (serving), or
    null when the engine did not report it — null means UNKNOWN, not awake.

    Sleep Mode exists only on servers started with VLLM_SERVER_DEV_MODE=1; on any
    other server this reports that the route is absent rather than failing vaguely.

    Args:
        target: Inference target name from config; omit for the default.
    """
    return ops.is_sleeping(_get_connection(target))


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

    Pass dry_run=True to preview.

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
@governed_tool(risk_level="high", undo=_sleep_undo)
@tool_errors("dict")
def model_sleep(
    level: int = 1, dry_run: bool = False, target: Optional[str] = None
) -> dict:
    """[WRITE][risk=high] Suspend the engine via Sleep Mode (it stops serving requests).

    Frees GPU memory between bursts. level=1 offloads the weights to CPU RAM and
    wakes fast; level=2 discards them, so waking reloads from disk. The engine
    serves nothing until model_wake. Pass dry_run=True to preview.

    Sleep Mode exists only on servers started with VLLM_SERVER_DEV_MODE=1; on any
    other server this reports that the route is absent rather than failing vaguely.

    Args:
        level: 1 to offload weights to CPU RAM, 2 to discard them (default 1).
        dry_run: If True, preview without suspending.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True, "wouldSleep": {"level": level},
                "currentlySleeping": ops.preview_sleep_state(conn)}
    return ops.sleep_model(conn, level)


@mcp.tool()
@governed_tool(risk_level="medium")
@tool_errors("dict")
def model_wake(dry_run: bool = False, target: Optional[str] = None) -> dict:
    """[WRITE][risk=medium] Resume serving after Sleep Mode (weights return to GPU).

    The inverse of model_sleep. Pass dry_run=True to preview. Records no undo:
    vLLM never reports which sleep level the engine was at, so re-sleeping would
    have to guess between level 1 and level 2 — call model_sleep with the level
    you want instead.

    Sleep Mode exists only on servers started with VLLM_SERVER_DEV_MODE=1; on any
    other server this reports that the route is absent rather than failing vaguely.

    Args:
        dry_run: If True, preview without waking.
        target: Inference target name from config; omit for the default.
    """
    conn = _get_connection(target)
    if dry_run:
        return {"dryRun": True, "wouldWake": True,
                "currentlySleeping": ops.preview_sleep_state(conn)}
    return ops.wake_model(conn)
