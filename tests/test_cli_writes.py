"""CLI confirmed-write path — past dry-run, through governance, onto disk.

The CLI write commands delegate real execution to the ``@governed_tool``
functions in ``mcp_server.tools``. These tests drive a write command PAST the
dry-run branch and the double-confirm prompts and assert the call really went
through the governed path (audit row on disk) — the regression test for the
"CLI writes were unaudited" line-wide fix.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

import inference_aiops.governance.audit as audit_mod
import inference_aiops.governance.policy as policy_mod
import inference_aiops.governance.undo as undo_mod


@pytest.fixture
def gov_home(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_AIOPS_HOME", str(tmp_path))
    audit_mod.reset_engine()
    policy_mod.reset_policy_engine()
    undo_mod.reset_undo_store()
    yield tmp_path
    audit_mod.reset_engine()
    policy_mod.reset_policy_engine()
    undo_mod.reset_undo_store()


def _audit_tools(db_path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT tool FROM audit_log ORDER BY id")]
    finally:
        conn.close()


def _apps(num: int = 3) -> dict:
    return {"applications": {"app1": {"deployments": {"dep1": {
        "status": "HEALTHY",
        "deployment_config": {"num_replicas": num},
        "replicas": [{"state": "RUNNING"}] * num,
    }}}}}


@pytest.mark.unit
def test_cli_scale_to_zero_dry_run_makes_no_call_and_no_audit(gov_home, monkeypatch):
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale-to-zero", "app1", "dep1", "--dry-run"])
    assert result.exit_code == 0
    assert "DRY-RUN" in result.output
    conn.get_ray.assert_not_called()
    conn.put_ray.assert_not_called()
    assert not (gov_home / "audit.db").exists()


@pytest.mark.unit
def test_cli_scale_to_zero_confirmed_goes_through_governance(gov_home, monkeypatch):
    """Confirmed CLI write must execute via the governed twin: the API call runs
    AND an audit row lands in audit.db (this is what the reroute fix bought)."""
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=3)
    conn.put_ray.return_value = {}
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale-to-zero", "app1", "dep1"], input="y\ny\n")
    assert result.exit_code == 0, result.output
    conn.put_ray.assert_called_once()
    assert _audit_tools(gov_home / "audit.db") == ["scale_to_zero"]


@pytest.mark.unit
def test_cli_scale_to_zero_aborts_without_double_confirm(gov_home, monkeypatch):
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale-to-zero", "app1", "dep1"], input="y\nn\n")
    assert result.exit_code != 0
    conn.get_ray.assert_not_called()
    conn.put_ray.assert_not_called()
    assert not (gov_home / "audit.db").exists()
