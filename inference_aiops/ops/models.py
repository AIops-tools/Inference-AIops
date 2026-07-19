"""vLLM model inventory + LoRA adapter / base-model lifecycle (read + writes).

The reads normalise vLLM's OpenAI-style ``/v1/models`` into flat rows — flagging
LoRA adapters (which share the served endpoint with their base) so an operator
can see what's actually loaded. The writes are the fragile ones the community
hits: hot-loading/unloading LoRA adapters and Sleep-Mode base-model hot-swaps.
Unload + hot-swap are traffic-affecting → risk=high with a dry-run preview, and
hot-swap captures the prior base model into ``priorState`` for a faithful undo.

Reads are resilient (a scrape/endpoint hiccup degrades to an ``error`` field).
"""

from __future__ import annotations

from typing import Any

from inference_aiops.ops._util import as_list, opt_s, s

_MODELS = "/v1/models"
_LOAD_LORA = "/v1/load_lora_adapter"
_UNLOAD_LORA = "/v1/unload_lora_adapter"
_HOT_SWAP = "/v1/hot_swap"


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


def hot_swap_model(conn: Any, new_model: str) -> dict:
    """[WRITE][high] Sleep-Mode base-model swap (reversible → prior model)."""
    prior = None
    rows = as_list(conn.get_vllm(_MODELS))
    if rows:
        prior = opt_s(rows[0].get("id"))
    conn.post_vllm(_HOT_SWAP, json={"model": new_model})
    return {"action": "model_hot_swap", "newModel": s(new_model),
            "priorState": {"model": prior}}
