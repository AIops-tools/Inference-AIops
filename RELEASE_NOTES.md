# Inference AIops v0.1.0 — preview

Governed AI-ops for **GPU inference clusters** — **vLLM** (OpenAI API +
Prometheus `/metrics`) and **Ray Serve / Ray Jobs** (Ray dashboard) — for AI
agents, with a built-in governance harness (audit, policy, token/runaway budget,
undo-token recording, graduated risk tiers) and an optional encrypted credential
store. Standalone — no external skill-family dependency.

> **Preview / mock-only.** All behaviour is validated against mocked vLLM
> `/metrics`, vLLM OpenAI API, and Ray dashboard responses; it has not been run
> against a live cluster. The fastest live check is `inference-aiops doctor`.

## Highlights

- **30 MCP tools** (16 read, 14 write), every one wrapped with `@governed_tool`.
  - **Metrics & RCA** (read): `request_metrics` (TTFT/TPOT/e2e + token totals),
    `queue_depth` (running vs waiting backpressure), `kv_cache_stats` (KV util,
    prefix-cache hit rate, preemptions), and the flagship correlators
    `diagnose_latency_spike` / `diagnose_low_utilization`.
  - **Ray Serve**: reads (`serve_deployment_list`, `deployment_status`,
    `replica_list`, `autoscale_config_get`) + guarded writes (`scale_replicas_up`,
    `scale_replicas_down`, `scale_to_zero`, `autoscale_config_update`,
    `drain_replica`).
  - **Models / vLLM**: `model_list`, `model_info`, `lora_load`, `lora_unload`,
    `model_hot_swap` (Sleep-Mode base swap, captures prior model).
  - **Ray cluster / jobs / GPU**: `ray_cluster_resources`, `ray_dashboard_status`,
    `ray_job_list`, `gpu_utilization`, `ray_job_cancel`, `replica_restart`.
  - **Deploy lifecycle**: `model_deploy`, `model_undeploy`, `deployment_redeploy`,
    `routing_policy_update` (prefix-aware / session-affinity routing).
  - **Cost**: `cost_per_token` (deterministic $/1M tokens from throughput × GPU $/hr).
- **Prometheus-native** — parses vLLM's `/metrics` directly; no Prometheus
  server required.
- **Safety on the fragile ops** — `scale_replicas_down`, `scale_to_zero`,
  `drain_replica`, `lora_unload`, `model_hot_swap`, `replica_restart`,
  `model_undeploy`, `deployment_redeploy` are high-risk with `dry_run` +
  double-confirm; reversible writes record an undo descriptor.
- **Optional-token auth** — a bearer token is optional (many stacks run open);
  when required it is stored **encrypted** (`~/.inference-aiops/secrets.enc`,
  Fernet + scrypt) — never plaintext on disk.
- **CLI** with an `init` onboarding wizard, `secret` management, and a `doctor`
  that probes the Ray dashboard and vLLM independently.

## Install

```bash
uv tool install inference-aiops
inference-aiops init
inference-aiops doctor
```

## Caveats

- Preview / mock-only: validated against mocked responses; needs live verification.
- Unverified against real hardware: multi-GPU tensor/pipeline-parallel
  deployments, real GPU thermal/throttle telemetry, and multi-node drain.
