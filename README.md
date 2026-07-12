<!-- mcp-name: io.github.AIops-tools/inference-aiops -->

# Inference AIops (preview)

> **Disclaimer**: Community-maintained open-source project. **Not affiliated with, endorsed by, or sponsored by the vLLM or Ray projects or any inference-serving vendor.** Product and trademark names belong to their owners. MIT licensed.

Governed AI-ops for **GPU inference clusters** — **vLLM** (OpenAI API + Prometheus
`/metrics`) and **Ray Serve / Ray Jobs** (Ray dashboard) — with a **built-in
governance harness**: unified audit log, policy engine, token/runaway budget
guard, undo-token recording, and graduated-autonomy risk tiers. It parses
vLLM's Prometheus `/metrics` directly (no Prometheus server required) and probes
the Ray dashboard independently. A bearer token is **optional** (many stacks run
open). **Preview — mock-validated only, not yet verified against a live cluster.**

## What it does

The flagship value is **root-cause analysis**, wrapped in guarded reads and writes:

- **`diagnose_latency_spike`** (flagship RCA) — when TTFT/TPOT/e2e latency
  climbs, it correlates **queue depth** (running vs waiting), **KV-cache
  pressure / preemptions**, and **prefix-cache locality** into a *ranked* cause
  plus the **specific knob to turn** (add replicas, raise `max-num-seqs`, fix
  routing, enlarge KV cache). Every flag is a number, not a black-box verdict.
- **`diagnose_low_utilization`** — the inverse: idle GPUs, over-provisioned
  replicas, or routing that strands a cache-warm replica → what to scale down.
- **Prometheus-native** — reads vLLM's `/metrics` endpoint directly; no
  Prometheus/Grafana deployment needed.
- **Governance-grade** — the **first governance-grade entrant** in this niche:
  audit + budget + risk-tier approval + undo-token + prompt-injection sanitize,
  with **dry-run + double-confirm** on the fragile prod ops (scale-down,
  scale-to-zero, drain, redeploy, hot-swap) the community reports as dangerous.
- **Laptop self-test** — ~80% of the tool self-tests free: vLLM on a single GPU
  or CPU-mock + Ray in one local container (`ray start --head`).

## Capability matrix (30 MCP tools)

| Group | Tools | Count | R/W (risk) |
|-------|-------|:-----:|:-----------|
| **Metrics & RCA** | `request_metrics`, `queue_depth`, `kv_cache_stats`, `diagnose_latency_spike`, `diagnose_low_utilization` | 5 | read |
| **Ray Serve (read)** | `serve_deployment_list`, `deployment_status`, `replica_list`, `autoscale_config_get` | 4 | read |
| **Ray Serve (write)** | `scale_replicas_up`, `scale_replicas_down`, `scale_to_zero`, `autoscale_config_update`, `drain_replica` | 5 | write (med / **high**) |
| **Models / vLLM** | `model_list`, `model_info`, `lora_load`, `lora_unload`, `model_hot_swap` | 5 | read + write (med / **high**) |
| **Ray cluster / jobs / GPU** | `ray_cluster_resources`, `ray_dashboard_status`, `ray_job_list`, `gpu_utilization`, `ray_job_cancel`, `replica_restart` | 6 | read + write (med / **high**) |
| **Deploy lifecycle** | `model_deploy`, `model_undeploy`, `deployment_redeploy`, `routing_policy_update` | 4 | write (med / **high**) |
| **Cost** | `cost_per_token` | 1 | read |

**16 read, 14 write.** High-risk writes (`scale_replicas_down`,
`scale_to_zero`, `drain_replica`, `lora_unload`, `model_hot_swap`,
`replica_restart`, `model_undeploy`, `deployment_redeploy`) all support
`dry_run` + double-confirm; reversible writes record an undo descriptor.

## Install

```bash
uv tool install inference-aiops          # or: pipx install inference-aiops
```

## Quick start

```bash
inference-aiops init                     # wizard: host + ray_port + vllm_port + scheme
inference-aiops doctor                   # probes BOTH the Ray dashboard and vLLM independently
inference-aiops overview                 # deployments + total replicas + queue backpressure
inference-aiops metrics diagnose         # why is inference slow? ranked RCA + the knob to turn
inference-aiops serve list               # Ray Serve deployments + replica counts
```

Run as an MCP server (stdio) for the full 30-tool surface:

```bash
export INFERENCE_AIOPS_MASTER_PASSWORD=...   # only if a bearer token is stored
inference-aiops mcp
```

The CLI is a convenience subset (`init`, `overview`, `serve …`, `metrics …`,
`secret …`, `doctor`, `mcp`); the full 30 tools are exposed via the MCP server.

## Governance

Every MCP tool passes through the bundled `@governed_tool` harness:

- **Audit** — every call (params, result, status, duration, risk tier,
  approver, rationale) logged to `~/.inference-aiops/audit.db` (relocatable via
  `INFERENCE_AIOPS_HOME`).
- **Budget / runaway guard** — token and call budgets trip a circuit breaker on
  tight poll/retry loops.
- **Risk tiers** — graduated autonomy; high-risk ops can require a named
  approver (`INFERENCE_AUDIT_APPROVED_BY` / `INFERENCE_AUDIT_RATIONALE`).
- **Undo recording** — reversible writes (scale, autoscale-config, routing,
  hot-swap, LoRA load) record an inverse descriptor.

## Supported scope + limitations

**Preview / mock-only.** All behaviour is validated against mocked vLLM
`/metrics`, vLLM OpenAI API, and Ray dashboard responses. **~80% of the tool
self-tests on a laptop** — vLLM on a single GPU or CPU-mock plus a local
one-node Ray head. Not yet verified against a live production cluster.

Unverified against real hardware / topology:

- multi-GPU **tensor-parallel / pipeline-parallel** deployments,
- real GPU **thermal / throttle** telemetry (utilisation is best-effort from
  the Ray dashboard's `/api/nodes`),
- **multi-node drain** and node-reboot orchestration.

The fastest live check is `inference-aiops doctor`.

## Missing a capability?

This is the GPU-inference member of the AIops-tools family (governed AI-ops with
audit + budget + undo + risk tiers). If a vLLM or Ray capability you need is
missing, or your stack speaks a dialect these tools don't yet handle — open an
issue or a PR. Contributions welcome.
