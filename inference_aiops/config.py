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

from inference_aiops.governance.paths import ops_home
from inference_aiops.secretstore import SecretStoreError, get_secret, has_store

CONFIG_DIR = ops_home()
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ENV_FILE = CONFIG_DIR / ".env"

DEFAULT_RAY_PORT = 8265
DEFAULT_VLLM_PORT = 8000

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
    """A connection target for one GPU inference stack (Ray dashboard + vLLM).

    ``host`` is shared by both services; ``ray_port`` (8265) reaches the Ray
    Serve/Jobs dashboard API and ``vllm_port`` (8000) the vLLM OpenAI +
    ``/metrics`` server. ``token`` is optional (empty = unauthenticated).
    """

    name: str
    host: str
    ray_port: int = DEFAULT_RAY_PORT
    vllm_port: int = DEFAULT_VLLM_PORT
    scheme: str = "http"
    verify_ssl: bool = False

    @property
    def token(self) -> str:
        return _resolve_secret(self.name)

    @property
    def ray_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.ray_port}"

    @property
    def vllm_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.vllm_port}"


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

    targets = tuple(
        TargetConfig(
            name=t["name"],
            host=t["host"],
            ray_port=t.get("ray_port", DEFAULT_RAY_PORT),
            vllm_port=t.get("vllm_port", DEFAULT_VLLM_PORT),
            scheme=t.get("scheme", "http"),
            verify_ssl=t.get("verify_ssl", False),
        )
        for t in raw.get("targets", [])
    )

    return AppConfig(targets=targets)
