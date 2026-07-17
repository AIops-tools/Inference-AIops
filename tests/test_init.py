"""Tests for the ``inference-aiops init`` wizard.

Driven through typer's CliRunner against an ``isolated_home`` (see conftest);
the master password comes from ``INFERENCE_AIOPS_MASTER_PASSWORD`` and the
hidden token prompt is fed by patching ``getpass``. The trailing doctor run is
either declined via stdin or patched out.

Inference specifics: no TLS confirm (plain-HTTP stacks), and the bearer token
is optional — the wizard only touches the secret store when the user says the
API requires one.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

import inference_aiops.cli.init as init_mod
import inference_aiops.secretstore as ss
from inference_aiops.cli._root import app
from tests.conftest import MASTER_PW

pytestmark = pytest.mark.unit

runner = CliRunner()

BEARER_TOKEN = "bearer-secret-123"  # noqa: S105 — test fixture value

# Prompt order (vLLM): name, engine(default vllm), host, vLLM port(default),
# ray_port(default), token confirm(default=False), add-another(No), doctor(No).
WIZARD_INPUT_NO_TOKEN = "prod\n\ngpu.example.com\n\n\n\n\nn\n"
# Same, but answer Yes at the token confirm ([getpass patched] supplies it).
WIZARD_INPUT_WITH_TOKEN = "prod\n\ngpu.example.com\n\n\ny\n\nn\n"
# SGLang: name, engine(sglang), host, SGLang port(default), token(No), add(No), doctor(No).
WIZARD_INPUT_SGLANG = "sg\nsglang\nsg.example.com\n\n\n\nn\n"

EXPECTED_ENTRY = {
    "name": "prod",
    "host": "gpu.example.com",
    "engine": "vllm",
    "scheme": "http",
    "ray_port": 8265,
    "vllm_port": 8000,
}


@pytest.fixture
def hidden_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """getpass reads the TTY, not CliRunner stdin — patch it."""
    monkeypatch.setattr(init_mod.getpass, "getpass", lambda prompt="": BEARER_TOKEN)


def test_init_with_token_writes_config_and_encrypted_secret(isolated_home, hidden_token):
    result = runner.invoke(app, ["init"], input=WIZARD_INPUT_WITH_TOKEN)
    assert result.exit_code == 0, result.output

    config_text = (isolated_home / "config.yaml").read_text("utf-8")
    raw = yaml.safe_load(config_text)
    assert raw["targets"] == [EXPECTED_ENTRY]

    # The token lands encrypted in secrets.enc, never in config.yaml.
    secrets_blob = (isolated_home / "secrets.enc").read_text("utf-8")
    assert BEARER_TOKEN not in config_text
    assert BEARER_TOKEN not in secrets_blob
    assert ss.SecretStore.unlock(MASTER_PW).get("prod") == BEARER_TOKEN


def test_init_without_token_stores_no_secret(isolated_home, hidden_token):
    result = runner.invoke(app, ["init"], input=WIZARD_INPUT_NO_TOKEN)
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((isolated_home / "config.yaml").read_text("utf-8"))
    assert raw["targets"] == [EXPECTED_ENTRY]
    # No token entered → the store was never persisted.
    assert not (isolated_home / "secrets.enc").exists()


def test_init_seeds_default_policy_rules(isolated_home, hidden_token):
    result = runner.invoke(app, ["init"], input=WIZARD_INPUT_NO_TOKEN)
    assert result.exit_code == 0, result.output
    rules = (isolated_home / "rules.yaml").read_text("utf-8")
    assert "high-risk-requires-approver" in rules
    assert "tier: dual" in rules


def test_init_does_not_clobber_existing_rules(isolated_home, hidden_token):
    sentinel = "# operator-authored rules — do not touch\nrisk_tiers: []\n"
    (isolated_home / "rules.yaml").write_text(sentinel, "utf-8")
    result = runner.invoke(app, ["init"], input=WIZARD_INPUT_NO_TOKEN)
    assert result.exit_code == 0, result.output
    assert (isolated_home / "rules.yaml").read_text("utf-8") == sentinel


def test_init_appends_to_existing_targets(isolated_home, hidden_token):
    assert runner.invoke(app, ["init"], input=WIZARD_INPUT_NO_TOKEN).exit_code == 0
    result = runner.invoke(app, ["init"], input="lab\n\ngpu2.example.com\n\n\n\n\nn\n")
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((isolated_home / "config.yaml").read_text("utf-8"))
    assert [t["name"] for t in raw["targets"]] == ["prod", "lab"]


def test_init_overwrites_target_on_confirm(isolated_home, hidden_token):
    assert runner.invoke(app, ["init"], input=WIZARD_INPUT_NO_TOKEN).exit_code == 0
    # Re-add 'prod': confirm the overwrite, accept vllm, change the host.
    result = runner.invoke(app, ["init"], input="prod\ny\n\nnew-gpu.example.com\n\n\n\n\nn\n")
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((isolated_home / "config.yaml").read_text("utf-8"))
    assert len(raw["targets"]) == 1
    assert raw["targets"][0]["host"] == "new-gpu.example.com"


def test_init_sglang_writes_engine_and_engine_port(isolated_home, hidden_token):
    """A single-process engine (SGLang) records engine + engine_port, no Ray port."""
    result = runner.invoke(app, ["init"], input=WIZARD_INPUT_SGLANG)
    assert result.exit_code == 0, result.output
    raw = yaml.safe_load((isolated_home / "config.yaml").read_text("utf-8"))
    entry = raw["targets"][0]
    assert entry["engine"] == "sglang"
    assert entry["engine_port"] == 30000
    assert entry["host"] == "sg.example.com"
    assert "ray_port" not in entry and "vllm_port" not in entry


def test_init_tgi_uses_default_port(isolated_home, hidden_token):
    """TGI default port is 8080 and no Ray dashboard is configured."""
    result = runner.invoke(app, ["init"], input="edge\ntgi\ntgi.example.com\n\n\n\nn\n")
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load((isolated_home / "config.yaml").read_text("utf-8"))["targets"][0]
    assert entry["engine"] == "tgi"
    assert entry["engine_port"] == 8080
    assert "ray_port" not in entry


def test_init_runs_doctor_when_accepted(isolated_home, hidden_token, monkeypatch):
    import inference_aiops.doctor as doc

    calls: list[bool] = []

    def fake_doctor(skip_auth: bool = False) -> int:
        calls.append(True)
        return 0

    monkeypatch.setattr(doc, "run_doctor", fake_doctor)
    # Accept the trailing doctor confirm (default=True) with a blank line.
    result = runner.invoke(app, ["init"], input="prod\n\ngpu.example.com\n\n\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert calls == [True]
