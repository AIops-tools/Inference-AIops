"""Connection management for a GPU inference stack (Ray dashboard + vLLM).

One :class:`InferenceConnection` fronts BOTH HTTP services of a target:

  * **Ray** dashboard API (Serve + Jobs) at ``ray_url`` — ``get_ray`` /
    ``post_ray`` / ``delete_ray`` return parsed JSON.
  * **vLLM** at ``vllm_url`` — ``get_vllm`` for the OpenAI-style JSON endpoints
    (``/v1/models``) and ``vllm_metrics`` which fetches the Prometheus
    ``/metrics`` text and parses it into ``{metric_name: [{labels, value}]}`` so
    ops code can read TTFT / queue depth / KV-cache stats without a Prometheus
    server.

Bearer auth is optional (many on-prem inference stacks run open); the token is
sent only when configured. All non-2xx responses become ``InferenceApiError``
with a teaching message.

The httpx client is injectable for tests: pass ``client=`` (an object with
``request`` / ``close``). Mock responses expose ``status_code``, ``content``,
``text``, and ``json()``.
"""

from __future__ import annotations

from typing import Any

import httpx

from inference_aiops.config import AppConfig, TargetConfig, load_config

_TIMEOUT = 30.0


class InferenceApiError(Exception):
    """A Ray/vLLM call failed; carries a teaching message + status code."""

    def __init__(self, message: str, *, status_code: int | None = None, path: str = "") -> None:
        self.status_code = status_code
        self.path = path
        super().__init__(message)


def _teaching_message(status: int, path: str, body: str, backend: str) -> str:
    """Map a non-2xx status to an actionable, teaching error message."""
    snippet = body[:200].strip()
    if status in (401, 403):
        return (
            f"Authentication failed ({status}) on {backend} {path}. If the "
            f"{backend} API requires a token, set it with 'inference-aiops secret "
            f"set <target>'. {snippet}"
        )
    if status == 404:
        return (
            f"Not found (404) on {backend} {path}. The deployment/replica/model id "
            f"may be stale — list them first. {snippet}"
        )
    if status in (500, 502, 503, 504):
        return (
            f"{backend} server error ({status}) on {path}. The dashboard/engine may "
            f"be starting or overloaded; retry shortly. {snippet}"
        )
    return f"{backend} API error ({status}) on {path}. {snippet}"


def parse_prometheus(text: str) -> dict[str, list[dict]]:
    """Parse Prometheus exposition text into ``{metric: [{labels, value}]}``.

    Ignores ``#`` comment/HELP/TYPE lines. Labels are parsed from the
    ``{k="v",...}`` block. Values that don't parse as float are skipped (a
    malformed line must not crash metric reads).
    """
    out: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                label_str, _, value_str = rest.rpartition("}")
                labels = _parse_labels(label_str)
            else:
                name, value_str = line.split(None, 1)
                labels = {}
            value = float(value_str.strip().split()[0])
        except (ValueError, IndexError):
            continue
        out.setdefault(name.strip(), []).append({"labels": labels, "value": value})
    return out


def _parse_labels(label_str: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for pair in label_str.split(","):
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        labels[k.strip()] = v.strip().strip('"')
    return labels


class InferenceConnection:
    """A session fronting one target's Ray dashboard + vLLM services."""

    def __init__(self, target: TargetConfig, client: Any | None = None) -> None:
        self._target = target
        headers = {"Accept": "application/json"}
        if target.token:
            headers["Authorization"] = f"Bearer {target.token}"
        self._client = client or httpx.Client(
            verify=target.verify_ssl, timeout=_TIMEOUT, headers=headers
        )

    @property
    def target(self) -> TargetConfig:
        return self._target

    def _request(self, method: str, url: str, path: str, backend: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise InferenceApiError(
                f"Could not reach {backend} at {url} ({method} {path}): {exc}. "
                f"Check the host/port and that the {backend} service is running.",
                path=path,
            ) from exc
        if not (200 <= resp.status_code < 300):
            raise InferenceApiError(
                _teaching_message(resp.status_code, path, resp.text, backend),
                status_code=resp.status_code, path=path,
            )
        return resp

    @staticmethod
    def _json(resp: Any) -> Any:
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    # ── Ray dashboard (Serve + Jobs) ─────────────────────────────────────
    def get_ray(self, path: str, **kwargs: Any) -> Any:
        url = f"{self._target.ray_url}{path}"
        return self._json(self._request("GET", url, path, "Ray", **kwargs))

    def post_ray(self, path: str, *, json: Any = None, **kwargs: Any) -> Any:
        url = f"{self._target.ray_url}{path}"
        return self._json(self._request("POST", url, path, "Ray", json=json, **kwargs))

    def put_ray(self, path: str, *, json: Any = None, **kwargs: Any) -> Any:
        url = f"{self._target.ray_url}{path}"
        return self._json(self._request("PUT", url, path, "Ray", json=json, **kwargs))

    def delete_ray(self, path: str, **kwargs: Any) -> Any:
        url = f"{self._target.ray_url}{path}"
        return self._json(self._request("DELETE", url, path, "Ray", **kwargs))

    # ── vLLM (OpenAI JSON + Prometheus metrics) ──────────────────────────
    def get_vllm(self, path: str, **kwargs: Any) -> Any:
        url = f"{self._target.vllm_url}{path}"
        return self._json(self._request("GET", url, path, "vLLM", **kwargs))

    def post_vllm(self, path: str, *, json: Any = None, **kwargs: Any) -> Any:
        url = f"{self._target.vllm_url}{path}"
        return self._json(self._request("POST", url, path, "vLLM", json=json, **kwargs))

    def vllm_metrics(self) -> dict[str, list[dict]]:
        """Fetch and parse the vLLM Prometheus ``/metrics`` endpoint."""
        resp = self._request("GET", f"{self._target.vllm_url}/metrics", "/metrics", "vLLM")
        return parse_prometheus(resp.text or "")

    def close(self) -> None:
        self._client.close()


class ConnectionManager:
    """Manages connections to multiple inference targets with session reuse."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._connections: dict[str, InferenceConnection] = {}

    @classmethod
    def from_config(cls, config: AppConfig | None = None) -> ConnectionManager:
        cfg = config or load_config()
        return cls(cfg)

    def connect(self, target_name: str | None = None) -> InferenceConnection:
        target = (
            self._config.get_target(target_name)
            if target_name
            else self._config.default_target
        )
        cached = self._connections.get(target.name)
        if cached is not None:
            return cached
        conn = InferenceConnection(target)
        self._connections[target.name] = conn
        return conn

    def disconnect(self, target_name: str) -> None:
        conn = self._connections.pop(target_name, None)
        if conn is not None:
            conn.close()

    def disconnect_all(self) -> None:
        for name in list(self._connections):
            self.disconnect(name)

    def list_targets(self) -> list[str]:
        return [t.name for t in self._config.targets]

    def list_connected(self) -> list[str]:
        return list(self._connections.keys())
