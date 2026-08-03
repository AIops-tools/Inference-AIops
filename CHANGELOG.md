# Changelog

## v0.8.0 — 2026-08-03

### Fixed
- **`undo apply` replays against the target the original write ran on.** It dispatched the inverse against whatever target the *caller* named — in practice the config's first entry — while the write's own target sat unused in the undo record. On a multi-target config the inverse therefore ran against the wrong host; it only looks harmless because the resource usually is not there, but two hosts holding the same name and the inverse **succeeds on the wrong one, silently**. An explicitly named target still wins. Line-wide: all 24 copies had the identical defect. Caught live in container-host-aiops, where a stop recorded against a Podman target replayed against a Portainer one.

## v0.7.0 — 2026-08-02

### Changed (BREAKING)
- **Requires MCP SDK 2.0** (`mcp[cli]>=2.0,<3.0`). `mcp.server.fastmcp` no longer exists in 2.0; the server is now built with `MCPServer` and reports its package version in the stdio handshake.
- **`model_deploy` no longer takes `num_replicas`.** Ray derives replica counts from the application's own config, and the deployment names needed to set them are only known once the app materialises — use `serve scale` afterwards.
- **`drain_replica`, `restart_replica` and `update_routing_policy` now refuse with a teaching error** instead of calling endpoints that do not exist. Ray Serve's REST control plane has no per-replica or routing-policy surface; per-replica draining is a Python-API capability only. Scaling a deployment down lets the controller drain surplus replicas gracefully.

### Fixed
- **`undo apply` works from the CLI.** Every write tool is imported lazily inside its own CLI command, so a CLI-driven undo ran in a process where the inverse tool was never registered and failed with "inverse tool is not registered" — for every write tool. Only the MCP entry point, which imports the whole server, worked. Found while live-verifying against a real cluster.
- **An undetermined outcome is audited `unknown`, not `ok`.** The harness only classified a result as undetermined when the payload *also* carried an `error` key, so a write that looked successful but had not been confirmed was recorded as a success.
- **The entire Ray Serve control plane called endpoints Ray has never had.** Live-verified against Ray 2.56.1: the REST control plane is **declarative** — the only mutating endpoint is `PUT /api/serve/applications/` with the whole `ServeDeploySchema`. The per-deployment and per-replica paths this tool used 404 on every real cluster, so scale, scale-to-zero, autoscale, deploy, undeploy, redeploy, drain, restart and routing **could never have worked**. All rebuilt on the real API: fetch every app config, patch the target, PUT the whole set.
- A single-app change is now **refused** when another app was deployed imperatively (`serve.run`) rather than from a Serve config, because the whole-schema PUT would silently delete that bystander.
- `get_cluster_resources` read fields that no longer exist and returned all-null on every live cluster; it now reads `clusterStatus.loadMetricsReport.usageByNode`. A resource no node reports yields `null`, not a fabricated `0`.
- `get_gpu_utilization` used `/api/nodes`, which 404s; the route is `/nodes`.


## v0.6.0 — 2026-07-21

### Changed (BREAKING)
- **Removed the authorization layer** — read-only mode, the approver gate, and rules.yaml deny are gone. The skill no longer decides read vs write; that is the agent's judgement or the connecting account's permissions. `<PREFIX>_READ_ONLY` now has no effect (a startup warning is logged); `<PREFIX>_AUDIT_APPROVED_BY`/`_RATIONALE` are optional audit annotations.
- The retained guarantee is **unbypassable audit over MCP and CLI alike** — no unaudited entry point. Harness = audit + runaway safety guard + undo + sanitize; `risk_level` is a descriptive audit label, not a gate.

See RELEASE_NOTES.md for tool-specific changes.


## v0.5.0 — 2026-07-20

### Fixed
- **BREAKING: `hot_swap_model` is removed.** It POSTed to `/v1/hot_swap`, **an endpoint vLLM has never had.** The docstring named vLLM Sleep Mode, so the path appears to have been invented for it — and Sleep Mode suspends the same model rather than swapping base models, so the advertised capability does not exist on the platform at all.
- **Replaced by the real thing:** `model_sleep`, `model_wake` and `model_is_sleeping`, on vLLM's actual `/sleep`, `/wake_up` and `/is_sleeping`.
- Harness: a write whose response is lost is audited `status=unknown`, not `error` — it may have taken effect. Undo tokens gain `effectVerified` (undo.db migrated in place).
- Harness: a dry-run no longer records an undo token, and no longer requires a named approver. Guards now run on the preview path.
- Truncated strings end in an ellipsis instead of being cut silently; error messages are capped at 800 chars, not 300.

See RELEASE_NOTES.md for the full detail.

## v0.3.0 — 2026-07-17

### Added
- **New:** SGLang + TGI serving-engine platforms.
- **Undo executor**: `undo list` / `undo apply <id>` (CLI + MCP) — apply a recorded replayable inverse; the dispatched inverse is re-gated by its own risk tier; single-use, dry-run, double-confirm, both wrapper + inverse audited.

## v0.7.0 — 2026-08-02

### Added
- **SGLang and TGI serving engines** alongside vLLM + Ray. A target now declares an
  `engine` (`vllm` / `sglang` / `tgi`); `init` prompts for it, and `doctor` probes the
  right surface (vLLM: Ray + vLLM; SGLang/TGI: engine health + running-model inventory).
- **5 engine-agnostic MCP reads** (now 35 tools, 21 read / 14 write): `engine_health`,
  `engine_inventory`, `engine_request_metrics`, `engine_queue_depth`, and
  `diagnose_engine_latency` — each reads the target engine's own health path,
  running-model identity, and Prometheus `/metrics` (vLLM `vllm:*`, SGLang `sglang:*`,
  TGI `tgi_*`); signals an engine does not expose degrade to `null`.
- **Engine registry** (`inference_aiops.engines`) mapping each engine's paths + metric
  names; config accepts `engine` + `engine_port` (default ports vLLM 8000 / SGLang 30000
  / TGI 8080), validating the engine name at load.

### Changed
- The Ray-shaped write ops (scale / drain / autoscale / deploy / redeploy / routing /
  job-cancel / replica-restart) now raise `EngineCapabilityError` — a teaching error —
  when targeted at a single-process SGLang/TGI engine that has no control plane.

## v0.2.1 — 2026-07-16

### Fixed
- **`secrets.enc` now follows `INFERENCE_AIOPS_HOME`** (secretstore hardcoded the real
  home directory; config/audit/undo already relocated — found in live verification).
- **Audit fidelity**: failures sanitized into `{"error": ...}` results by the MCP error
  layer are now audited as `status=error` (they previously read as `ok`, hiding failed
  attempts from exception reports), and no undo is recorded for a call that failed.

### Tests
- `doctor` and the `init` wizard are now fully covered (previously ~10–20%); plus a
  regression test for the sanitized-failure audit status.

## v0.2.0 — 2026-07-13

Security-hardening release from a line-wide code review.

### Changed (behavior)
- **Secure by default**: with no `rules.yaml`, high/critical operations now require a
  named approver (`INFERENCE_AUDIT_APPROVED_BY`). A fresh install no longer allows
  destructive writes unattended; `init` seeds a starter `rules.yaml` you can edit,
  and an operator-authored rules file is honoured as-is.
- `__version__` is now single-sourced from package metadata (the previous release
  self-reported a stale version string).
- Sanitize docs no longer overstate scope: it strips control/format characters and
  truncates; semantic prompt-injection resistance must come from the consuming agent.

### Fixed
- Agent-supplied ids are percent-encoded in vLLM/Ray REST URL paths (path-traversal hardening).

### Tests
- Governance persistence is now tested against REAL `audit.db`/`undo.db` files
  (write → audit row + inverse undo row with captured prior state).
- The CLI confirmed-write path (dry-run / double-confirm / governed execution) is
  covered end-to-end.
- `pytest-cov` added to the dev dependencies.

## v0.1.1

- Fix: `INFERENCE_AIOPS_HOME` now also relocates `config.yaml` (was hardcoded to `~/.inference-aiops`).
- Fix: **CLI writes are now audited + undo-recorded** via the governance path — previously only the MCP tools recorded audit/undo; CLI `manage`/`remediate`/etc. writes now go through the same `@governed_tool` layer (they keep their dry-run + double-confirm). CLI write output is now the governed JSON result. No API/tool changes.


All notable changes to inference-aiops are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [0.1.0] — preview

Initial preview release: governed AI-ops for GPU inference clusters (vLLM + Ray
Serve / Ray Jobs) with a bundled governance harness. **Mock-validated only — not
yet verified against a live cluster.**

### Added

- **30 MCP tools** (16 read, 14 write), every one wrapped with the bundled
  `@governed_tool` harness (audit, policy, token/runaway budget, undo,
  risk-tiers):
  - **Metrics & RCA** — `request_metrics` (TTFT/TPOT/e2e latency + token
    totals), `queue_depth` (running vs waiting backpressure), `kv_cache_stats`
    (KV utilisation, prefix-cache hit rate, preemptions), and the flagship
    correlators `diagnose_latency_spike` and `diagnose_low_utilization` (fold
    queue depth + KV-cache pressure + prefix-cache locality into a ranked cause
    plus the specific knob to turn).
  - **Ray Serve** — reads `serve_deployment_list`, `deployment_status`,
    `replica_list`, `autoscale_config_get`; writes `scale_replicas_up` (med,
    undo), `scale_replicas_down` (high, dry-run, undo), `scale_to_zero` (high,
    dry-run, undo), `autoscale_config_update` (med, undo), `drain_replica`
    (high, dry-run, graceful — finishes in-flight requests).
  - **Models / vLLM** — `model_list`, `model_info` (read); `lora_load` (med),
    `lora_unload` (high, dry-run), `model_hot_swap` (high, dry-run — Sleep-Mode
    base swap, captures the prior model).
  - **Ray cluster / jobs / GPU** — `ray_cluster_resources` (CPU/GPU alloc),
    `ray_dashboard_status`, `ray_job_list`, `gpu_utilization` (per-node) as
    reads; `ray_job_cancel` (med) and `replica_restart` (high, dry-run) as writes.
  - **Deploy lifecycle** — `model_deploy` (med), `model_undeploy` (high,
    dry-run), `deployment_redeploy` (high, dry-run), `routing_policy_update`
    (med, undo — prefix-aware / session-affinity routing to fix cache locality).
  - **Cost** — `cost_per_token` (deterministic $/1M tokens from throughput ×
    GPU $/hr).
- **Prometheus-native metrics** — parses vLLM's `/metrics` endpoint directly; no
  Prometheus server required.
- **Optional encrypted secret store** — a bearer token is optional (many stacks
  run open); when required it is stored encrypted in
  `~/.inference-aiops/secrets.enc` (Fernet + scrypt), never plaintext on disk.
  Legacy `INFERENCE_<TARGET>_TOKEN` env var honoured as a fallback.
- **CLI** (`inference-aiops`) — `init` wizard, `overview`, `serve`
  (list/status/scale/scale-to-zero), `metrics` (requests/queue/diagnose),
  `secret` management, `mcp`, and a `doctor` that probes the Ray dashboard and
  vLLM independently.
- **Connection layer** over the Ray dashboard (Serve + Jobs) and the vLLM
  services with centralised teaching error translation.

### Known limitations

- Preview / mock-only: validated against mocked vLLM `/metrics`, vLLM OpenAI
  API, and Ray dashboard responses; needs live verification.
- Unverified against real hardware / topology: multi-GPU tensor-parallel /
  pipeline-parallel deployments, real GPU thermal/throttle telemetry, and
  multi-node drain / node-reboot orchestration.
