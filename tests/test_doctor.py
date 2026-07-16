"""Tests for ``inference_aiops.doctor.run_doctor``.

Everything runs against an ``isolated_home`` (see conftest) — no real
``~/.inference-aiops`` and no network: the Ray/vLLM probes are exercised by
patching ``ConnectionManager`` at the connection-module boundary.

Inference specifics: the bearer token is OPTIONAL (``target.token`` returns ""
when none is stored, never raising), and doctor probes the Ray dashboard and
the vLLM server independently — one can be up while the other is down.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
import yaml
from rich.console import Console

import inference_aiops.doctor as doc
import inference_aiops.secretstore as ss
from tests.conftest import MASTER_PW

pytestmark = pytest.mark.unit


@pytest.fixture
def doctor_out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Capture doctor output on a wide console (no line-wrapping surprises)."""
    buf = io.StringIO()
    monkeypatch.setattr(doc, "_console", Console(file=buf, width=200))
    return buf


def _write_config(home, targets: list[dict]) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"targets": targets}), "utf-8")


def _seed_secret(name: str, value: str) -> None:
    ss.SecretStore.unlock(MASTER_PW).set(name, value)


PROD = {"name": "prod", "host": "gpu.example.com", "ray_port": 8265, "vllm_port": 8000}


# ─── broken-environment paths ───────────────────────────────────────────────


def test_missing_config_file(isolated_home, doctor_out):
    assert doc.run_doctor() == 1
    out = doctor_out.getvalue()
    assert "✗ Config file missing" in out
    assert "inference-aiops init" in out


def test_config_load_failure(isolated_home, doctor_out):
    (isolated_home / "config.yaml").write_text("targets: [ {name: broken", "utf-8")
    assert doc.run_doctor() == 1
    assert "✗ Config load failed" in doctor_out.getvalue()


def test_no_targets_configured(isolated_home, doctor_out):
    _write_config(isolated_home, [])
    assert doc.run_doctor() == 1
    assert "✗ No targets configured" in doctor_out.getvalue()


def test_no_secret_store_is_the_only_problem(isolated_home, doctor_out):
    # Tokens are optional, so the missing store alone drives the exit code.
    _write_config(isolated_home, [PROD])
    assert doc.run_doctor(skip_auth=True) == 1
    out = doctor_out.getvalue()
    assert "! No secret store yet" in out
    assert "✓ Target 'prod' — auth: no-auth (open)" in out


def test_legacy_env_file_warns_but_works(isolated_home, doctor_out, monkeypatch):
    _write_config(isolated_home, [PROD])
    (isolated_home / ".env").write_text("INFERENCE_PROD_TOKEN=legacy\n", "utf-8")
    monkeypatch.setenv("INFERENCE_PROD_TOKEN", "legacy")
    assert doc.run_doctor(skip_auth=True) == 0
    out = doctor_out.getvalue()
    assert "legacy plaintext .env" in out
    assert "secret migrate" in out
    assert "✓ Target 'prod' — auth: token" in out


def test_world_readable_secrets_warns(isolated_home, doctor_out):
    _write_config(isolated_home, [PROD])
    _seed_secret("prod", "bearer-tok")
    (isolated_home / "secrets.enc").chmod(0o644)
    assert doc.run_doctor(skip_auth=True) == 0  # warning, not a failure
    assert "should be 600" in doctor_out.getvalue()


# ─── healthy paths ───────────────────────────────────────────────────────────


def test_healthy_skip_auth_with_token(isolated_home, doctor_out):
    _write_config(isolated_home, [PROD])
    _seed_secret("prod", "bearer-tok")
    assert doc.run_doctor(skip_auth=True) == 0
    out = doctor_out.getvalue()
    assert "✓ Config file present" in out
    assert "✓ 1 target(s) configured" in out
    assert "✓ Encrypted secret store present" in out
    assert "✓ Target 'prod' — auth: token" in out
    assert "Skipping connectivity check" in out


def test_open_target_displayed_as_no_auth(isolated_home, doctor_out):
    open_target = {**PROD, "name": "lab"}
    _write_config(isolated_home, [PROD, open_target])
    _seed_secret("prod", "bearer-tok")  # store exists; 'lab' has no token
    assert doc.run_doctor(skip_auth=True) == 0
    out = doctor_out.getvalue()
    assert "✓ Target 'prod' — auth: token" in out
    assert "✓ Target 'lab' — auth: no-auth (open)" in out


class _FakeConn:
    def __init__(self, ray: Any, vllm: Any) -> None:
        self._ray = ray
        self._vllm = vllm

    def get_ray(self, path: str, **_: Any) -> Any:
        if isinstance(self._ray, Exception):
            raise self._ray
        assert path == "/api/serve/applications/"
        return self._ray

    def get_vllm(self, path: str, **_: Any) -> Any:
        if isinstance(self._vllm, Exception):
            raise self._vllm
        assert path == "/v1/models"
        return self._vllm


class _FakeMgr:
    """Stands in for ConnectionManager; per-target canned (ray, vllm) results."""

    results: dict[str, tuple[Any, Any]] = {}

    def __init__(self, config: Any) -> None:
        self._config = config

    def connect(self, name: str) -> _FakeConn:
        ray, vllm = self.results[name]
        return _FakeConn(ray, vllm)


@pytest.fixture
def fake_mgr(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMgr]:
    import inference_aiops.connection as conn_mod

    _FakeMgr.results = {}
    monkeypatch.setattr(conn_mod, "ConnectionManager", _FakeMgr)
    return _FakeMgr


def test_healthy_end_to_end_both_backends(isolated_home, doctor_out, fake_mgr):
    _write_config(isolated_home, [PROD])
    _seed_secret("prod", "bearer-tok")
    fake_mgr.results["prod"] = (
        {"applications": {}},
        {"data": [{"id": "llama-3"}, {"id": "mistral"}]},
    )
    assert doc.run_doctor() == 0
    out = doctor_out.getvalue()
    assert "✓ Ray dashboard reachable (http://gpu.example.com:8265)" in out
    assert "✓ vLLM reachable (http://gpu.example.com:8000) — 2 model(s)" in out


def test_one_backend_down_fails_but_reports_the_other(isolated_home, doctor_out, fake_mgr):
    _write_config(isolated_home, [PROD])
    _seed_secret("prod", "bearer-tok")
    fake_mgr.results["prod"] = ({"applications": {}}, ConnectionError("connection refused"))
    assert doc.run_doctor() == 1
    out = doctor_out.getvalue()
    assert "✓ Ray dashboard reachable" in out
    assert "✗ vLLM 'prod'" in out
    assert "connection refused" in out


def test_both_backends_down(isolated_home, doctor_out, fake_mgr):
    _write_config(isolated_home, [PROD])
    _seed_secret("prod", "bearer-tok")
    fake_mgr.results["prod"] = (
        RuntimeError("ray dashboard unreachable"),
        RuntimeError("vllm unreachable"),
    )
    assert doc.run_doctor() == 1
    out = doctor_out.getvalue()
    assert "✗ Ray dashboard 'prod'" in out
    assert "✗ vLLM 'prod'" in out
