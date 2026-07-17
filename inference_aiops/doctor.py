"""Environment and connectivity diagnostics for Inference AIops."""

from __future__ import annotations

from rich.console import Console

from inference_aiops.config import CONFIG_FILE, ENV_FILE, load_config
from inference_aiops.secretstore import SECRETS_FILE, check_permissions, has_store

_console = Console()


def run_doctor(skip_auth: bool = False) -> int:
    """Check config, secrets, and (optionally) connectivity.

    Returns a process exit code: 0 healthy, 1 problems found. Connectivity
    failures are reported as status, never raised as tracebacks (a doctor must
    survive the thing it diagnoses being unhealthy).
    """
    problems = 0

    if not CONFIG_FILE.exists():
        _console.print(f"[red]✗ Config file missing: {CONFIG_FILE}[/]")
        _console.print("[yellow]  Run 'inference-aiops init' to set up your first target.[/]")
        return 1
    _console.print(f"[green]✓ Config file present: {CONFIG_FILE}[/]")

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001 — report, do not crash
        _console.print(f"[red]✗ Config load failed: {exc}[/]")
        return 1

    if not config.targets:
        _console.print("[red]✗ No targets configured[/]")
        return 1
    _console.print(f"[green]✓ {len(config.targets)} target(s) configured[/]")

    if has_store():
        _console.print(f"[green]✓ Encrypted secret store present: {SECRETS_FILE}[/]")
        perm_warning = check_permissions()
        if perm_warning:
            _console.print(f"[yellow]! {perm_warning}[/]")
    elif ENV_FILE.exists():
        _console.print(
            f"[yellow]! Using legacy plaintext .env ({ENV_FILE}). Migrate with "
            f"'inference-aiops secret migrate'.[/]"
        )
    else:
        _console.print(
            "[yellow]! No secret store yet. Run 'inference-aiops init' to set up "
            "credentials (stored encrypted).[/]"
        )
        problems += 1

    for target in config.targets:
        auth = "token" if target.token else "no-auth (open)"
        _console.print(f"[green]✓ Target '{target.name}' — auth: {auth}[/]")

    if skip_auth:
        _console.print("[dim]Skipping connectivity check (--skip-auth).[/]")
        return 1 if problems else 0

    from inference_aiops.connection import ConnectionManager

    mgr = ConnectionManager(config)
    for target in config.targets:
        conn = mgr.connect(target.name)
        if target.engine == "vllm":
            problems += _probe_vllm_target(conn, target)
        else:
            problems += _probe_engine_target(conn, target)

    return 1 if problems else 0


def _probe_vllm_target(conn, target) -> int:  # type: ignore[no-untyped-def]
    """Probe a vLLM+Ray target's two backends independently (one may be up alone)."""
    problems = 0
    try:
        conn.get_ray("/api/serve/applications/")
        _console.print(f"[green]✓ Ray dashboard reachable ({target.ray_url})[/]")
    except Exception as exc:  # noqa: BLE001 — connectivity is a status, not a crash
        _console.print(f"[red]✗ Ray dashboard '{target.name}' ({target.ray_url}): {exc}[/]")
        problems += 1
    try:
        models = conn.get_vllm("/v1/models")
        n = len(models.get("data", [])) if isinstance(models, dict) else 0
        _console.print(f"[green]✓ vLLM reachable ({target.vllm_url}) — {n} model(s)[/]")
    except Exception as exc:  # noqa: BLE001 — connectivity is a status, not a crash
        _console.print(f"[red]✗ vLLM '{target.name}' ({target.vllm_url}): {exc}[/]")
        problems += 1
    return problems


def _probe_engine_target(conn, target) -> int:  # type: ignore[no-untyped-def]
    """Probe a single-process engine (SGLang / TGI): health + running-model identity."""
    from inference_aiops.engines import get_engine_spec
    from inference_aiops.ops.engine import engine_inventory

    spec = get_engine_spec(target.engine)
    problems = 0
    try:
        conn.get_engine(spec.health_path)
        _console.print(f"[green]✓ {spec.label} reachable ({target.engine_url})[/]")
    except Exception as exc:  # noqa: BLE001 — connectivity is a status, not a crash
        _console.print(f"[red]✗ {spec.label} '{target.name}' ({target.engine_url}): {exc}[/]")
        return 1
    inv = engine_inventory(conn)
    if "error" in inv:
        _console.print(f"[yellow]! {spec.label} inventory unavailable: {inv['error']}[/]")
        problems += 1
    else:
        _console.print(f"[green]✓ {spec.label} model(s): {', '.join(inv['models']) or '(none)'}[/]")
    return problems
