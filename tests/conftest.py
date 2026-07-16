"""Shared test fixtures (no live vLLM / Ray stack needed)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_approver(monkeypatch):
    """The policy layer is secure-by-default: with no rules.yaml, high/critical
    governed calls require a named approver. Tests exercising tool behavior
    are not about that gate, so record a synthetic approver globally; the
    governance-persistence tests remove it to test the gate itself."""
    monkeypatch.setenv("INFERENCE_AUDIT_APPROVED_BY", "pytest")


MASTER_PW = "test-master-pw"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect every config/secret/governance path to a throwaway home.

    The path constants are bound at import time in each module, so patch the
    names where they are *used* (config, secretstore, doctor, cli.init), plus
    the env vars for call-time resolution (governance ``ops_path`` and the
    secret-store master password).
    """
    import inference_aiops.cli.init as init_mod
    import inference_aiops.config as cfg
    import inference_aiops.doctor as doc
    import inference_aiops.secretstore as ss

    monkeypatch.setenv("INFERENCE_AIOPS_HOME", str(tmp_path))
    monkeypatch.setenv("INFERENCE_AIOPS_MASTER_PASSWORD", MASTER_PW)
    monkeypatch.setattr(ss, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(ss, "SECRETS_FILE", tmp_path / "secrets.enc")
    monkeypatch.setattr(ss, "LEGACY_ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(ss, "_cached", None)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(cfg, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(doc, "CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr(doc, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(doc, "SECRETS_FILE", tmp_path / "secrets.enc")
    monkeypatch.setattr(init_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(init_mod, "CONFIG_FILE", tmp_path / "config.yaml")
    return tmp_path
