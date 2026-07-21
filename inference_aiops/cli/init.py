"""``inference-aiops init`` — a friendly, interactive onboarding wizard.

Walks a new user through connecting their first inference-cluster target
(vLLM + Ray Serve): collects the non-secret connection details into
``config.yaml`` and the API key into the *encrypted* store (never plaintext on
disk). Designed to be run on a terminal; everything it needs is prompted with
sensible defaults.
"""

from __future__ import annotations

import getpass

import typer
import yaml

from inference_aiops.cli._common import cli_errors, console
from inference_aiops.config import CONFIG_DIR, CONFIG_FILE, DEFAULT_RAY_PORT
from inference_aiops.engines import DEFAULT_ENGINE_PORTS, SUPPORTED_ENGINES, get_engine_spec
from inference_aiops.secretstore import SecretStore, resolve_master_password


def _load_existing_targets() -> list[dict]:
    if not CONFIG_FILE.exists():
        return []
    raw = yaml.safe_load(CONFIG_FILE.read_text("utf-8")) or {}
    return list(raw.get("targets", []))


def _write_targets(targets: list[dict]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CONFIG_DIR.chmod(0o700)
    except OSError:
        pass
    CONFIG_FILE.write_text(yaml.safe_dump({"targets": targets}, sort_keys=False), "utf-8")


@cli_errors
def init_cmd() -> None:
    """Interactively set up your first Inference connection."""
    console.print("[bold cyan]Inference AIops — setup wizard[/]")
    console.print(
        "This collects your serving engine's connection details (saved to "
        "config.yaml). Supported engines: vLLM (+ Ray Serve), SGLang, TGI. A "
        "bearer token is optional (many inference stacks run open); if given it "
        "is saved [bold]encrypted[/] to secrets.enc.\n"
    )

    console.print("[bold]Step 1 — master password[/]")
    console.print(
        "[dim]Encrypts secrets.enc. You'll set it via the "
        "INFERENCE_AIOPS_MASTER_PASSWORD env var for non-interactive/MCP use.[/]"
    )
    password = resolve_master_password(confirm_if_new=True)
    store = SecretStore.unlock(password)

    targets = _load_existing_targets()
    existing_names = {t.get("name") for t in targets}

    while True:
        console.print("\n[bold]Step 2 — add an inference target[/]")
        name = typer.prompt("Target name (e.g. prod)").strip()
        if name in existing_names:
            if not typer.confirm(f"'{name}' already exists — overwrite?", default=False):
                continue
            targets = [t for t in targets if t.get("name") != name]

        engine = typer.prompt(
            f"Serving engine ({'/'.join(SUPPORTED_ENGINES)})", default="vllm"
        ).strip().lower()
        while engine not in SUPPORTED_ENGINES:
            console.print(
                f"[yellow]Unknown engine — choose one of: {', '.join(SUPPORTED_ENGINES)}[/]"
            )
            engine = typer.prompt(
                f"Serving engine ({'/'.join(SUPPORTED_ENGINES)})", default="vllm"
            ).strip().lower()
        spec = get_engine_spec(engine)

        host = typer.prompt("Host (IP or FQDN)").strip()
        engine_port = typer.prompt(
            f"{spec.label} port", default=DEFAULT_ENGINE_PORTS[engine], type=int
        )

        entry: dict = {"name": name, "host": host, "engine": engine, "scheme": "http"}
        if engine == "vllm":
            # vLLM's control plane is the Ray Serve/Jobs dashboard.
            ray_port = typer.prompt("Ray dashboard port", default=DEFAULT_RAY_PORT, type=int)
            entry["ray_port"] = ray_port
            entry["vllm_port"] = engine_port
        else:
            # SGLang / TGI are single-process — no Ray dashboard.
            entry["engine_port"] = engine_port

        if typer.confirm("Does the API require a bearer token?", default=False):
            secret = getpass.getpass(f"Token for '{name}' (hidden): ")
            store = store.set(name, secret)
        targets.append(entry)
        existing_names.add(name)
        _write_targets(targets)
        console.print(f"[green]✓ Saved target '{name}'.[/]")

        if not typer.confirm("\nAdd another target?", default=False):
            break

    console.print(f"\n[green]✓ Setup complete.[/] Config: {CONFIG_FILE}")
    console.print(
        "[dim]Tip: export INFERENCE_AIOPS_MASTER_PASSWORD=... in your shell profile "
        "so the MCP server and CLI can unlock secrets non-interactively.[/]"
    )
    if typer.confirm("Run a connectivity check now (inference-aiops doctor)?", default=True):
        from inference_aiops.doctor import run_doctor

        raise typer.Exit(run_doctor())
