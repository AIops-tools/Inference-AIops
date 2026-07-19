"""Shared helpers for Inference ops modules.

Reads come from two shapes: Ray dashboard JSON (dicts/arrays) and vLLM
Prometheus metrics (parsed into ``{name: [{labels, value}]}`` by the connection
layer). ``metric_sum`` / ``metric_latest`` / ``histogram_avg`` pull scalar
signals out of the parsed metric map. All server text reaches the caller only
after ``sanitize()`` (encoding-level output hygiene). ``_seg`` is the ONLY
sanctioned way to place an agent-supplied identifier into a REST URL path
segment — it percent-encodes everything (including ``/``) so an id like
``../admin`` cannot rewrite the request path.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from inference_aiops.governance import opt_str, sanitize


def _seg(value: Any) -> str:
    """Percent-encode one URL *path segment* (agent-supplied identifier).

    ``safe=""`` also encodes ``/``, so a hostile id (``../drain``, ``a/b``)
    stays a single segment instead of rewriting the request path. Query-string
    values are NOT routed through here — httpx ``params=`` handles those.
    """
    return quote(str(value), safe="")


def as_list(data: Any) -> list[dict]:
    """Normalise a list payload (bare array or ``{"data": [...]}``) to a list of dicts."""
    if isinstance(data, dict):
        items = data.get("data", data.get("applications", []))
    else:
        items = data
    if isinstance(items, dict):  # Ray Serve returns applications as a name→obj map
        items = list(items.values())
    return [i for i in (items or []) if isinstance(i, dict)]


def as_obj(data: Any) -> dict:
    """Return ``data`` as a dict (empty dict if it isn't one)."""
    return data if isinstance(data, dict) else {}


def s(value: Any, limit: int = 256) -> str:
    """Sanitize an arbitrary value to a bounded, injection-safe string."""
    return sanitize(str(value if value is not None else ""), limit)


def opt_s(value: Any, limit: int = 256) -> str | None:
    """Sanitize a value that may legitimately be absent, preserving that absence.

    Companion to :func:`s`, which folds ``None`` into ``""``. That conflation is
    invisible downstream: an empty string reads as "the engine reported this
    field and it was blank" when the truth may be "this engine/Ray version never
    reports the field". Neither a consumer nor a smaller local model can recover
    the difference, and both tend to invent one.

    Use this for any optional field (a job's ``entrypoint``, a model's ``parent``
    adapter, a replica ``state``, a server-info ``version``); keep :func:`s` for
    values that are always present, such as a map key already in hand.
    """
    return opt_str(value, limit)


def metric_sum(metrics: dict[str, list[dict]], name: str) -> float | None:
    """Sum all series values for a metric (None if absent)."""
    series = metrics.get(name)
    if not series:
        return None
    return round(sum(p.get("value", 0.0) for p in series), 4)


def metric_latest(metrics: dict[str, list[dict]], name: str) -> float | None:
    """Return the max series value for a gauge-like metric (None if absent)."""
    series = metrics.get(name)
    if not series:
        return None
    return max(p.get("value", 0.0) for p in series)


def histogram_avg(metrics: dict[str, list[dict]], base: str) -> float | None:
    """Average of a Prometheus histogram: ``<base>_sum`` / ``<base>_count``."""
    total = metric_sum(metrics, f"{base}_sum")
    count = metric_sum(metrics, f"{base}_count")
    if not total or not count:
        return None
    return round(total / count, 4)
