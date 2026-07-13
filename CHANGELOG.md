# Changelog

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
