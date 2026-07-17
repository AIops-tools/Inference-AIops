---
name: inference-aiops
description: >
  Use this skill whenever the user needs to operate a GPU inference cluster — vLLM (OpenAI API + Prometheus /metrics) and Ray Serve / Ray Jobs (Ray dashboard), plus the single-process serving engines SGLang and TGI (Text Generation Inference): a one-shot cluster overview (deployments + total replicas + queue backpressure), request metrics (TTFT / TPOT / e2e latency + token totals), queue depth, KV-cache stats (utilisation, prefix-cache hit rate, preemptions), the flagship latency root-cause analysis (diagnose_latency_spike / diagnose_engine_latency) and low-utilisation RCA, engine-agnostic health + running-model inventory across vLLM/SGLang/TGI, Ray Serve autoscaling and scaling (scale up/down, scale-to-zero, drain a replica), LoRA load/unload, base-model hot-swap, deploy/undeploy/redeploy, prefix-aware routing, GPU utilisation, Ray jobs, and cost per million tokens.
  Always use this skill for "why is inference slow", "TTFT spike", "latency spike", "GPU underutilised", "scale down the deployment", "scale to zero", "drain a replica before a reboot", "hot-swap the base model", "load a LoRA adapter", "KV cache pressure", "prefix cache hit rate", "queue backpressure", "autoscale config", "SGLang health", "TGI metrics", or "cost per token" when the context is a vLLM / SGLang / TGI / Ray Serve inference cluster.
  Do NOT use for non-inference infrastructure (hypervisors, storage appliances, backup products, general container/cluster workloads, network devices, or OT/industrial equipment) — those belong to other AIops-tools; this skill is scoped to GPU inference serving (vLLM + Ray).
  Preview — governed vLLM + Ray inference operations with a built-in governance harness (audit, policy, token budget, undo, risk-tiers). Mock-validated only, not yet verified against a live cluster.
installer:
  kind: uv
  package: inference-aiops
argument-hint: "[deployment/model name or describe your inference-cluster task]"
allowed-tools:
  - Bash
metadata: {"openclaw":{"requires":{"env":["INFERENCE_AIOPS_CONFIG"],"bins":["inference-aiops"],"config":["~/.inference-aiops/config.yaml"]},"optional":{"env":["INFERENCE_AIOPS_MASTER_PASSWORD"],"config":["~/.inference-aiops/secrets.enc"]},"primaryEnv":"INFERENCE_AIOPS_CONFIG","homepage":"https://github.com/AIops-tools/Inference-AIops","emoji":"🚀","os":["macos","linux"]}}
compatibility: >
  Standalone, self-governed GPU-inference operations (preview). The governance harness (audit, policy, token/runaway budget, undo, risk-tiers) is bundled in the package — no external skill-family dependency.
  All write operations are audited to a local SQLite DB under ~/.inference-aiops/ (relocatable via INFERENCE_AIOPS_HOME).
  Auth: a bearer token is OPTIONAL — many vLLM / Ray stacks run open. When the API requires one it is stored ENCRYPTED in ~/.inference-aiops/secrets.enc (Fernet/AES-128 + scrypt-derived key) — never plaintext on disk. Run 'inference-aiops init' to onboard, or 'inference-aiops secret set <target>' to add one. The store is unlocked by a master password from INFERENCE_AIOPS_MASTER_PASSWORD (non-interactive/MCP/CI) or an interactive prompt (CLI on a TTY). A legacy plaintext env var INFERENCE_<TARGET_NAME_UPPER>_TOKEN is still honoured as a fallback (migrate with 'inference-aiops secret migrate'). The token is sent as an Authorization: Bearer header at request time and held only in memory; it is never logged or echoed.
  State-changing operations require double confirmation at the CLI layer and support --dry-run. All write tools pass through the @governed_tool decorator (pre-check + budget guard + audit + risk-tier gate). The fragile prod ops — scale_replicas_down, scale_to_zero, drain_replica, lora_unload, model_hot_swap, replica_restart, model_undeploy, deployment_redeploy — are high-risk with a dry_run preview; reversible writes (scale, autoscale-config, routing, hot-swap, LoRA load) record an undo descriptor.
  Engines: vLLM (with its Ray Serve control plane), SGLang, and TGI. SGLang/TGI are single-process servers with engine-agnostic observability (health, running-model inventory, request metrics, queue depth, latency RCA); Ray-shaped scale/drain writes are vLLM-only and raise a teaching error on a SGLang/TGI target.
  Metrics: each engine's Prometheus /metrics endpoint is parsed directly — no Prometheus server is required.
  Webhooks: none — no outbound calls beyond the configured Ray dashboard and vLLM services.
  SSL: verify_ssl defaults to true; disable only for self-signed lab certificates.
  Transitive dependencies: httpx (HTTP client) and the MCP SDK. No post-install scripts or background services.
  PREVIEW: mock-validated only; unverified against multi-GPU tensor/pipeline-parallel deployments, real GPU thermal/throttle telemetry, and multi-node drain.
---

# Inference AIops (preview)

> **Disclaimer**: Community-maintained open-source project, **not affiliated with, endorsed by, or sponsored by the vLLM or Ray projects or any inference-serving vendor.** Product and trademark names belong to their owners. Source at [github.com/AIops-tools/Inference-AIops](https://github.com/AIops-tools/Inference-AIops) under the MIT license.

Governed GPU-inference operations for **vLLM** (OpenAI API + Prometheus `/metrics`) and **Ray Serve / Ray Jobs** (Ray dashboard), plus the single-process serving engines **SGLang** and **TGI** — **35 MCP tools**, every one wrapped with the bundled `@governed_tool` harness: a local unified audit log under `~/.inference-aiops/`, policy engine, token/runaway budget guard, undo-token recording, and graduated-autonomy risk tiers. The flagship `diagnose_latency_spike` folds queue depth + KV-cache pressure + prefix-cache locality into a ranked cause and the specific knob to turn; the engine-agnostic `diagnose_engine_latency` does the same across whatever signals SGLang/TGI expose. Each engine's Prometheus `/metrics` is parsed directly — **no Prometheus server required**.

> **Standalone**: the governance harness is bundled in the package (`inference_aiops.governance`) — no external skill-family dependency. **Preview / mock-only**: not yet validated against a live cluster. A bearer token is **optional** (many stacks run open).

## What This Skill Does

| Group | Tools | Count | Read or Write |
|-------|-------|:-----:|:-------------:|
| **Metrics & RCA** (vLLM) | request metrics, queue depth, KV-cache stats, diagnose latency spike, diagnose low utilisation | 5 | 5 read |
| **Engine-agnostic** (vLLM/SGLang/TGI) | engine health, engine inventory, engine request metrics, engine queue depth, diagnose engine latency | 5 | 5 read |
| **Ray Serve (read)** | deployment list, deployment status, replica list, autoscale config get | 4 | 4 read |
| **Ray Serve (write)** | scale up (med), scale down (high), scale-to-zero (high), autoscale config update (med), drain replica (high) | 5 | 5 write |
| **Models / vLLM** | model list, model info, LoRA load (med), LoRA unload (high), base hot-swap (high) | 5 | 2 read / 3 write |
| **Ray cluster / jobs / GPU** | cluster resources, dashboard status, job list, GPU utilisation, job cancel (med), replica restart (high) | 6 | 4 read / 2 write |
| **Deploy lifecycle** | deploy (med), undeploy (high), redeploy (high), routing policy update (med) | 4 | 4 write |
| **Cost** | cost per token | 1 | 1 read |

**21 read, 14 write.** The high-risk writes support `dry_run` + double-confirm; reversible writes record an undo descriptor. The engine-agnostic reads cover any engine; the Ray Serve / cluster / deploy write groups are vLLM-only and teach-and-refuse on a SGLang/TGI target (single-process engines have no Ray control plane).

## Quick Install

```bash
uv tool install inference-aiops
inference-aiops init       # interactive wizard: engine (vllm/sglang/tgi) + host + port + scheme (token optional)
inference-aiops doctor     # vLLM: probes Ray + vLLM; SGLang/TGI: engine health + inventory
```

## When to Use This Skill

- Triage a cluster (`overview`): Serve deployments, total replicas, queue backpressure
- Diagnose slow inference (`metrics diagnose` / `diagnose_latency_spike`): rank the cause (queue depth vs KV-cache preemption vs prefix-cache locality) and get the knob to turn
- Find idle GPUs and over-provisioned replicas (`diagnose_low_utilization`)
- Scale a Ray Serve deployment up/down, **scale-to-zero** to stop cost bleed, or update autoscale bounds
- **Drain** a replica gracefully before a node reboot (finishes in-flight requests)
- Load/unload a **LoRA** adapter; **hot-swap** a base model (Sleep-Mode swap, captures the prior model)
- Inspect GPU utilisation per node, list/cancel Ray jobs, restart a stuck replica
- Compute **cost per million tokens** from throughput × GPU $/hr
- Observe an **SGLang** or **TGI** server (`engine_health`, `engine_inventory`, `engine_request_metrics`, `engine_queue_depth`, `diagnose_engine_latency`) — single-process engines with no Ray control plane

**Do NOT use for** non-inference infrastructure (hypervisors, storage appliances, backup products, general container workloads, network devices, or OT/industrial equipment) — those belong to other AIops-tools. This skill is scoped to GPU inference serving (vLLM + Ray).

## Related Skills — Skill Routing

| If the user wants… | Use |
|--------------------|-----|
| vLLM / Ray Serve inference: latency RCA, autoscale, drain, LoRA, cost/token | **inference-aiops** (this skill) |
| SGLang / TGI serving: health, running-model inventory, request metrics, queue depth, latency RCA | **inference-aiops** (this skill — engine-agnostic reads) |
| Any non-inference infrastructure (hypervisor, storage, backup, general clusters, network, OT) | the appropriate **other AIops-tools** line |

## Common Workflows

### "Why is inference slow?" (flagship RCA)

1. `inference-aiops metrics diagnose` (or the `diagnose_latency_spike` tool) → a ranked cause with numbers: is `waiting` queue depth high (backpressure)? Are there KV-cache **preemptions**? Has the **prefix-cache hit rate** dropped (routing lost locality)?
2. Act on the named knob: add replicas (`scale_replicas_up`), raise the batch cap via `autoscale_config_update`, or fix locality with `routing_policy_update` (prefix-aware / session-affinity).
3. Re-check with `request_metrics` (TTFT / TPOT / e2e) and `queue_depth`.

### Safely scale down / scale-to-zero a prod deployment

1. Set an approver: `export INFERENCE_AUDIT_APPROVED_BY=you INFERENCE_AUDIT_RATIONALE="off-peak cost save"`.
- **Secure by default (v0.2.0+)**: with no `~/.inference-aiops/rules.yaml`, high/critical operations are denied unless `INFERENCE_AUDIT_APPROVED_BY` names an approver (set `INFERENCE_AUDIT_RATIONALE` too). `inference-aiops init` seeds a starter rules.yaml; an operator-authored rules file is honoured as-is.
2. `inference-aiops serve scale-to-zero <app> <deployment> --dry-run` → preview the call (double-confirm required). `scale_to_zero` stops the cost bleed but **strands ingress** — confirm that's intended.
3. Re-run without `--dry-run`; the undo descriptor captures the prior replica count so you can restore it with `scale_replicas_up`.

### Graceful drain before a node reboot

1. `drain_replica <app> <deployment> <replica_id> --dry-run` (high risk) → previews; the drain finishes in-flight requests before removing the replica.
2. Confirm, drain, then reboot the node. (Multi-node drain is unverified in preview.)

### Hot-swap a base model

1. `model_hot_swap <new_model>` (high risk, dry-run first) → a Sleep-Mode base swap that **captures the prior model** into an undo descriptor.
2. Verify with `model_info` / `request_metrics`; replay the undo to roll back.

### Cost per million tokens

`cost_per_token` derives a deterministic $/1M tokens from measured throughput × your GPU $/hr — useful for sizing replicas or justifying a scale-to-zero.

## Governance & Safety

- Every tool is audited to `~/.inference-aiops/audit.db` (relocatable via `INFERENCE_AIOPS_HOME`).
- High-risk ops can require a named approver: set `INFERENCE_AUDIT_APPROVED_BY` and `INFERENCE_AUDIT_RATIONALE`.
- The fragile prod writes support `--dry-run` and double confirmation at the CLI.
- Reversible writes (scale, autoscale-config, routing, hot-swap, LoRA load) record an inverse descriptor.

## References

- `references/capabilities.md` — full tool → backend → endpoint → returns reference
- `references/cli-reference.md` — CLI command reference
- `references/setup-guide.md` — onboarding, optional token, and connectivity
