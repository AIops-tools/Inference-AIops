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


def _no_mutating_call(conn) -> None:
    """No mutating verb reached the cluster, whatever else happened.

    These are InferenceConnection's real mutating methods (Ray Dashboard
    put/post/delete plus the vLLM POST). Asserting on methods the transport does
    not have would pass vacuously against a MagicMock.
    """
    conn.put_ray.assert_not_called()
    conn.post_ray.assert_not_called()
    conn.delete_ray.assert_not_called()
    conn.post_vllm.assert_not_called()


@pytest.mark.unit
def test_cli_scale_to_zero_dry_run_reads_and_audits_but_never_writes(gov_home, monkeypatch):
    """A dry_run MAY read; it must never write.

    The older "dry_run does zero I/O and leaves no trace" assumption was never a
    stated rule and is wrong on its face: a preview that cannot read cannot
    answer "would this be refused?", nor report the replica count it is about to
    park at zero. So the read is expected, the audit row is expected (MCP
    previews were always audited — the CLI silently not auditing was the
    outlier), and only the MUTATING call is forbidden.
    """
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=3)
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale-to-zero", "app1", "dep1", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "DRY-RUN" in result.output  # human banner preserved, not raw JSON
    conn.get_ray.assert_called()  # it DID read, to resolve the real from→to
    _no_mutating_call(conn)
    assert _audit_tools(gov_home / "audit.db") == ["scale_to_zero"]


@pytest.mark.unit
def test_cli_scale_to_zero_dry_run_reports_the_real_current_replica_count(
    gov_home, monkeypatch
):
    """The banner carries the count the tool read, not a hardcoded 'num_replicas: 0'."""
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=7)
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale-to-zero", "app1", "dep1", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "from_replicas = 7" in result.output
    assert "to_replicas = 0" in result.output


@pytest.mark.unit
def test_cli_scale_to_zero_dry_run_records_no_undo_token(gov_home, monkeypatch):
    """A preview changed nothing, so there is nothing to reverse.

    A phantom undo token is not inert: undo_apply would dispatch a REAL
    scale_replicas_up for a scale-to-zero that never happened.
    """
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=3)
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    CliRunner().invoke(app, ["serve", "scale-to-zero", "app1", "dep1", "--dry-run"])
    if (gov_home / "undo.db").exists():
        rows = sqlite3.connect(gov_home / "undo.db").execute(
            "SELECT undo_tool FROM undo_log"
        ).fetchall()
        assert rows == [], f"dry-run registered a phantom undo: {rows}"


@pytest.mark.unit
def test_cli_scale_to_zero_dry_run_of_an_unknown_deployment_refuses_nonzero(
    gov_home, monkeypatch
):
    """A preview that cannot resolve its target must refuse, not reassure.

    Naming a deployment that does not exist is the commonest way a caller gets
    this wrong. Before the reroute the preview happily described the scale
    anyway, and the failure only surfaced on the confirmed write.
    """
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = {"applications": {}}
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale-to-zero", "nope", "dep1", "--dry-run"])
    assert result.exit_code == 1
    assert "DRY-RUN" not in result.output  # no green banner for a refusal
    _no_mutating_call(conn)


@pytest.mark.unit
def test_cli_serve_scale_dry_run_reads_and_audits_but_never_writes(gov_home, monkeypatch):
    """A dry_run MAY read; it must never write."""
    from inference_aiops.cli import app

    conn = MagicMock(name="conn")
    conn.get_ray.return_value = _apps(num=2)
    import mcp_server.tools.serve as gov_serve

    monkeypatch.setattr(gov_serve, "_get_connection", lambda target=None: conn)
    result = CliRunner().invoke(app, ["serve", "scale", "app1", "dep1", "5", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "DRY-RUN" in result.output
    assert "from_replicas = 2" in result.output and "to_replicas = 5" in result.output
    _no_mutating_call(conn)
    assert _audit_tools(gov_home / "audit.db") == ["scale_replicas_up"]


@pytest.mark.unit
def test_cli_undo_apply_dry_run_of_an_unknown_token_refuses_nonzero(gov_home):
    """An unknown undo id is a refusal, not a preview of 'inverse: ?'.

    Before the reroute this printed a green banner naming the inverse tool as
    '?' — a preview of an operation that does not exist.
    """
    from inference_aiops.cli import app

    result = CliRunner().invoke(app, ["undo", "apply", "nope-not-a-token", "--dry-run"])
    assert result.exit_code == 1
    assert "Unknown undo id" in result.output
    assert "DRY-RUN" not in result.output


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
