"""Configuration management for Inference AIops.

Loads GPU inference-cluster connection targets from a YAML config file. A
"target" is one inference stack reached over two HTTP endpoints:

  * a **Ray dashboard** (Ray Serve + Jobs API, default port ``8265``), and
  * a **vLLM** OpenAI-compatible + Prometheus ``/metrics`` server (default port
    ``8000``).

Inference endpoints frequently run with **no auth** on a trusted network, so the
bearer **token is optional**: if none is stored, requests are sent unauthenticated.
When a token *is* used it is NEVER stored in the config file or in plaintext on
disk — it lives in the encrypted store ``~/.inference-aiops/secrets.enc`` (see
:mod:`inference_aiops.secretstore`), with a legacy env var
(``INFERENCE_<TARGET>_TOKEN``) honoured as a fallback.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from inference_aiops.engines import DEFAULT_ENGINE_PORTS, SUPPORTED_ENGINES, get_engine_spec
from inference_aiops.governance.paths import ops_home
from inference_aiops.secretstore import SecretStoreError, get_secret, has_store

CONFIG_DIR = ops_home()
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"

DEFAULT_RAY_PORT = 8265
DEFAULT_VLLM_PORT = 8000
DEFAULT_ENGINE = "vllm"

# Legacy env-var prefix/suffix; also used by the migration helper.
SECRET_ENV_PREFIX = "INFERENCE_"  # nosec B105 — env-var name, not a secret
SECRET_ENV_SUFFIX = "_TOKEN"  # nosec B105 — env-var name, not a secret

_log = logging.getLogger("inference-aiops.config")


def _secret_env_key(name: str) -> str:
    """Legacy per-target token env var name, e.g. INFERENCE_PROD_TOKEN."""
    return f"{SECRET_ENV_PREFIX}{name.upper().replace('-', '_')}{SECRET_ENV_SUFFIX}"


def _resolve_secret(name: str) -> str:
    """Return a target's bearer token, or "" when none is configured (auth optional)."""
    if has_store():
        try:
            return get_secret(name)
        except SecretStoreError:
            pass  # fall through to legacy env var / no-auth
    legacy = os.environ.get(_secret_env_key(name))
    if legacy:
        _log.warning(
            "Using plaintext env var %s. Migrate to the encrypted store with "
            "'inference-aiops secret migrate'.",
            _secret_env_key(name),
        )
        return legacy
    return ""  # no token → unauthenticated (common for on-prem inference)


@dataclass(frozen=True)
class TargetConfig:
    """A connection target for one GPU inference stack.

    ``host`` is shared by every service on the target. ``engine`` selects the
    serving platform (``vllm`` / ``sglang`` / ``tgi``); ``vllm_port`` (kept for
    back-compat as the *engine* HTTP port) reaches that engine's OpenAI +
    Prometheus ``/metrics`` server, and — for the vLLM engine only — ``ray_port``
    (8265) reaches the Ray Serve/Jobs dashboard that provides its control plane.
    SGLang and TGI are single-process servers with no Ray dashboard. ``token``
    is optional (empty = unauthenticated).
    """

    name: str
    host: str
    ray_port: int = DEFAULT_RAY_PORT
    vllm_port: int = DEFAULT_VLLM_PORT
    scheme: str = "http"
    verify_ssl: bool = False
    engine: str = DEFAULT_ENGINE

    @property
    def token(self) -> str:
        return _resolve_secret(self.name)

    @property
    def ray_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.ray_port}"

    @property
    def vllm_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.vllm_port}"

    @property
    def engine_port(self) -> int:
        """The serving engine's HTTP port (stored in ``vllm_port`` field)."""
        return self.vllm_port

    @property
    def engine_url(self) -> str:
        """Base URL of the serving engine's HTTP surface (any engine)."""
        return f"{self.scheme}://{self.host}:{self.vllm_port}"

    @property
    def has_control_plane(self) -> bool:
        """True when the engine has a Ray Serve control plane (vLLM only)."""
        return get_engine_spec(self.engine).has_control_plane


@dataclass(frozen=True)
class AppConfig:
    """Top-level application config."""

    targets: tuple[TargetConfig, ...] = ()

    def get_target(self, name: str) -> TargetConfig:
        for t in self.targets:
            if t.name == name:
                return t
        available = ", ".join(t.name for t in self.targets) or "(none)"
        raise KeyError(f"Target '{name}' not found. Available: {available}")

    @property
    def default_target(self) -> TargetConfig:
        if not self.targets:
            raise ValueError("No targets configured. Check config.yaml")
        return self.targets[0]


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load config from YAML; any bearer token comes from the encrypted store."""
    path = config_path or CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Run 'inference-aiops init' to set up an inference target, or create "
            f"{CONFIG_FILE} with a 'targets' list."
        )

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    targets = tuple(_build_target(t) for t in raw.get("targets", []))
    return AppConfig(targets=targets)


def _resolve_engine(raw_engine: object, target_name: str) -> str:
    """Validate + normalise a target's engine (fail fast on a typo)."""
    engine = str(raw_engine or DEFAULT_ENGINE).strip().lower()
    if engine not in SUPPORTED_ENGINES:
        supported = ", ".join(SUPPORTED_ENGINES)
        raise ValueError(
            f"Target '{target_name}': unknown serving engine '{engine}'. "
            f"Supported engines: {supported}."
        )
    return engine


def _resolve_engine_port(t: dict, engine: str) -> int:
    """Resolve the engine's HTTP port from any accepted key, else its default.

    Accepts (in precedence order) ``vllm_port`` (legacy), ``engine_port``, and
    ``port``; falls back to the engine's canonical default.
    """
    for key in ("vllm_port", "engine_port", "port"):
        if t.get(key) is not None:
            return int(t[key])
    return DEFAULT_ENGINE_PORTS.get(engine, DEFAULT_VLLM_PORT)


def _build_target(t: dict) -> TargetConfig:
    name = t["name"]
    engine = _resolve_engine(t.get("engine"), name)
    return TargetConfig(
        name=name,
        host=t["host"],
        ray_port=t.get("ray_port", DEFAULT_RAY_PORT),
        vllm_port=_resolve_engine_port(t, engine),
        scheme=t.get("scheme", "http"),
        verify_ssl=t.get("verify_ssl", False),
        engine=engine,
    )
