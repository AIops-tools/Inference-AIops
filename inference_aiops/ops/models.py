"""vLLM model inventory + LoRA adapter / Sleep-Mode lifecycle (read + writes).

The reads normalise vLLM's OpenAI-style ``/v1/models`` into flat rows — flagging
LoRA adapters (which share the served endpoint with their base) so an operator
can see what's actually loaded. The writes are the fragile ones the community
hits: hot-loading/unloading LoRA adapters and suspending/resuming the engine via
Sleep Mode. Unload and sleep are traffic-affecting → risk=high with a dry-run
preview, and sleep captures whether the engine was ALREADY asleep so its undo
never wakes an engine the caller did not put to sleep.

Sleep Mode is what an operator reaches for to free GPU memory between bursts:
``POST /sleep?level=1`` offloads the weights to CPU RAM, ``level=2`` discards
them, ``POST /wake_up`` restores serving, and ``GET /is_sleeping`` reports the
state. It suspends the SAME model — it does NOT swap base models. Serving a
different base model requires restarting vLLM with a different ``--model``, so
no tool here advertises an in-place base-model swap.

These three routes are registered ONLY when vLLM was started with
``VLLM_SERVER_DEV_MODE=1``. Most production deployments do not set it, so a 404
here means "the server was not started in dev mode", not "the id was stale" —
:func:`_dev_mode_error` turns that specific 404 into a teaching
:class:`EngineCapabilityError` instead of the generic stale-id 404 message.

Reads are resilient (a scrape/endpoint hiccup degrades to an ``error`` field).
"""

from __future__ import annotations

from typing import Any

from inference_aiops.connection import EngineCapabilityError, InferenceApiError
from inference_aiops.ops._util import as_list, opt_s, s

_MODELS = "/v1/models"
_LOAD_LORA = "/v1/load_lora_adapter"
_UNLOAD_LORA = "/v1/unload_lora_adapter"
_SLEEP = "/sleep"
_WAKE_UP = "/wake_up"
_IS_SLEEPING = "/is_sleeping"

# vLLM accepts exactly these Sleep-Mode levels; anything else is rejected before
# the request goes out, so the caller gets the reason instead of a raw 400.
_SLEEP_LEVELS = (1, 2)

_DEV_MODE_HINT = (
    "vLLM's Sleep-Mode endpoints (/sleep, /wake_up, /is_sleeping) are registered "
    "ONLY when the server is started with VLLM_SERVER_DEV_MODE=1. This server "
    "returned 404 for {path}, which means it was NOT started in dev mode — the "
    "tool is working, the route does not exist on this server. Restart vLLM with "
    "VLLM_SERVER_DEV_MODE=1 to enable Sleep Mode, or leave it off if this is a "
    "production deployment that should not expose it."
)


def _normalize_model(entry: dict) -> dict:
    """Flatten one ``/v1/models`` row, flagging LoRA adapters."""
    model_id = opt_s(entry.get("id"))
    root = entry.get("root")
    parent = entry.get("parent")
    is_lora = parent is not None or (root is not None and opt_s(root) != model_id)
    return {
        "id": model_id,
        "object": opt_s(entry.get("object")),
        "ownedBy": opt_s(entry.get("owned_by")),
        "isLora": bool(is_lora),
    }


def list_models(conn: Any) -> list[dict]:
    """[READ] All served vLLM models, LoRA adapters flagged."""
    try:
        return [_normalize_model(e) for e in as_list(conn.get_vllm(_MODELS))]
    except Exception as exc:  # noqa: BLE001 — report as partial
        return [{"error": s(exc, 200)}]


def get_model_info(conn: Any, model_id: str) -> dict:
    """[READ] One model's config: max length, root/parent, permission (best-effort)."""
    try:
        rows = as_list(conn.get_vllm(_MODELS))
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    for entry in rows:
        if opt_s(entry.get("id")) == s(model_id):
            return {
                "id": opt_s(entry.get("id")),
                "maxModelLen": entry.get("max_model_len"),
                "root": opt_s(entry.get("root")),
                "parent": opt_s(entry.get("parent")),
                "permission": entry.get("permission"),
            }
    return {"error": s(f"Model '{model_id}' not found — list_models first.", 200)}


# ── writes ───────────────────────────────────────────────────────────────


def load_lora_adapter(conn: Any, lora_name: str, lora_path: str) -> dict:
    """[WRITE] Hot-load a LoRA adapter onto the running engine."""
    conn.post_vllm(_LOAD_LORA, json={"lora_name": lora_name, "lora_path": lora_path})
    return {"action": "lora_load", "loraName": s(lora_name), "loraPath": s(lora_path)}


def unload_lora_adapter(conn: Any, lora_name: str) -> dict:
    """[WRITE][high] Hot-unload a LoRA adapter (traffic on it will start failing)."""
    conn.post_vllm(_UNLOAD_LORA, json={"lora_name": lora_name})
    return {"action": "lora_unload", "loraName": s(lora_name)}


# ── Sleep Mode ───────────────────────────────────────────────────────────


def _dev_mode_error(exc: InferenceApiError, path: str) -> EngineCapabilityError:
    """Recast a Sleep-Mode 404 as the capability error it actually is."""
    return EngineCapabilityError(
        _DEV_MODE_HINT.format(path=path), status_code=exc.status_code, path=path
    )


def _sleeping_flag(conn: Any) -> bool | None:
    """Current sleep state, or None when the engine will not say.

    None is load-bearing: it means "could not determine", which is why the sleep
    undo declines to act on it rather than assuming the engine was awake.
    """
    try:
        data = conn.get_vllm(_IS_SLEEPING)
    except Exception:  # noqa: BLE001 — a probe failure must not block the write
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("is_sleeping")
    return bool(value) if isinstance(value, bool) else None


def preview_sleep_state(conn: Any) -> bool | None:
    """Sleep state for a dry-run preview, raising the guard the real write would.

    A preview answers "what happens if I do this?", so it has to reach the same
    capability guard as the write it previews. Reporting a benign
    ``currentlySleeping: null`` on a server with no Sleep-Mode routes would send
    the caller on to a high-risk write that cannot succeed — burning an approval
    to discover what the preview already knew. So a 404 raises here exactly as
    it does on the real call.

    A non-404 probe failure still degrades to ``None`` (unknown): that is a
    transient read problem, not evidence the write would fail, and a preview
    must not invent a blocker either.
    """
    try:
        data = conn.get_vllm(_IS_SLEEPING)
    except InferenceApiError as exc:
        if exc.status_code == 404:
            raise _dev_mode_error(exc, _IS_SLEEPING) from exc
        return None
    except Exception:  # noqa: BLE001 — a probe hiccup must not block a preview
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("is_sleeping")
    return value if isinstance(value, bool) else None


def is_sleeping(conn: Any) -> dict:
    """[READ] Whether the vLLM engine is currently in Sleep Mode."""
    try:
        data = conn.get_vllm(_IS_SLEEPING)
    except InferenceApiError as exc:
        if exc.status_code == 404:
            return {"error": s(_DEV_MODE_HINT.format(path=_IS_SLEEPING), 600)}
        return {"error": s(exc, 200)}
    except Exception as exc:  # noqa: BLE001 — report as partial
        return {"error": s(exc, 200)}
    value = data.get("is_sleeping") if isinstance(data, dict) else None
    return {"isSleeping": value if isinstance(value, bool) else None}


def sleep_model(conn: Any, level: int = 1) -> dict:
    """[WRITE][high] Suspend the engine via Sleep Mode (requests stop being served).

    ``level=1`` offloads the weights to CPU RAM (fast to wake); ``level=2``
    discards them (wake reloads from disk). Captures whether the engine was
    ALREADY asleep so the undo can decline rather than wake it spuriously.
    """
    if level not in _SLEEP_LEVELS:
        raise ValueError(
            f"Sleep level {level!r} is not valid. vLLM accepts level=1 (offload "
            f"weights to CPU RAM, fast wake) or level=2 (discard weights, wake "
            f"reloads from disk)."
        )
    was_sleeping = _sleeping_flag(conn)
    try:
        conn.post_vllm(_SLEEP, params={"level": level})
    except InferenceApiError as exc:
        if exc.status_code == 404:
            raise _dev_mode_error(exc, _SLEEP) from exc
        raise
    return {"action": "model_sleep", "level": level,
            "priorState": {"wasSleeping": was_sleeping}}


def wake_model(conn: Any) -> dict:
    """[WRITE][med] Resume serving after Sleep Mode (weights are restored to GPU)."""
    was_sleeping = _sleeping_flag(conn)
    try:
        conn.post_vllm(_WAKE_UP)
    except InferenceApiError as exc:
        if exc.status_code == 404:
            raise _dev_mode_error(exc, _WAKE_UP) from exc
        raise
    return {"action": "model_wake", "priorState": {"wasSleeping": was_sleeping}}
