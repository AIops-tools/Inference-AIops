---
name: inference-aiops
slug: inference-aiops
displayName: "Inference AIops"
summary: "Governed GPU inference ops (vLLM + Ray Serve): latency RCA, scaling, drain, 37 tools."
license: MIT
homepage: https://github.com/AIops-tools/Inference-AIops
tags: [aiops, mcp, governance, inference]
description: >
  Use this skill whenever the user needs to operate a GPU inference cluster — vLLM (OpenAI API + Prometheus /metrics) and Ray Serve / Ray Jobs (Ray dashboard), plus the single-process serving engines SGLang and TGI (Text Generation Inference): a one-shot cluster overview (deployments + total replicas + queue backpressure), request metrics (TTFT / TPOT / e2e latency + token totals), queue depth, KV-cache stats (utilisation, prefix-cache hit rate, preemptions), the flagship latency root-cause analysis (diagnose_latency_spike / diagnose_engine_latency) and low-utilisation RCA, engine-agnostic health + running-model inventory across vLLM/SGLang/TGI, Ray Serve autoscaling and scaling (scale up/down, scale-to-zero, drain a replica), LoRA load/unload, base-model hot-swap, deploy/undeploy/redeploy, prefix-aware routing, GPU utilisation, Ray jobs, and cost per million tokens.
  Always use this skill for "why is inference slow", "TTFT spike", "latency spike", "GPU underutilised", "scale down the deployment", "scale to zero", "drain a replica before a reboot", "hot-swap the base model", "load a LoRA adapter", "KV cache pressure", "prefix cache hit rate", "queue backpressure", "autoscale config", "SGLang health", "TGI metrics", or "cost per token" when the context is a vLLM / SGLang / TGI / Ray Serve inference cluster.
  Do NOT use for non-inference infrastructure (hypervisors, storage appliances, backup products, general container/cluster workloads, network devices, or OT/industrial equipment) — those belong to other AIops-tools; this skill is scoped to GPU inference serving (vLLM + Ray).
  Governed vLLM + Ray inference operations with a built-in governance harness (audit, policy, token budget, undo, risk-tiers).
installer:
  kind: uv
  package: inference-aiops
argument-hint: "[deployment/model name or describe your inference-cluster task]"
allowed-tools:
  - Bash
metadata: {"openclaw":{"requires":{"env":["INFERENCE_AIOPS_CONFIG"],"bins":["inference-aiops"],"config":["~/.inference-aiops/config.yaml"]},"optional":{"env":["INFERENCE_AIOPS_MASTER_PASSWORD"],"config":["~/.inference-aiops/secrets.enc"]},"primaryEnv":"INFERENCE_AIOPS_CONFIG","homepage":"https://github.com/AIops-tools/Inference-AIops","emoji":"🚀","os":["macos","linux"]}}
compatibility: >
  Standalone, self-governed GPU-inference operations. The governance harness (audit, policy, token/runaway budget, undo, risk-tiers) is bundled in the package — no external skill-family dependency.
  All write operations are audited to a local SQLite DB under ~/.inference-aiops/ (relocatable via INFERENCE_AIOPS_HOME).
  Auth: a bearer token is OPTIONAL — many vLLM / Ray stacks run open. When the API requires one it is stored ENCRYPTED in ~/.inference-aiops/secrets.enc (Fernet/AES-128 + scrypt-derived key) — never plaintext on disk. Run 'inference-aiops init' to onboard, or 'inference-aiops secret set <target>' to add one. The store is unlocked by a master password from INFERENCE_AIOPS_MASTER_PASSWORD (non-interactive/MCP/CI) or an interactive prompt (CLI on a TTY). A legacy plaintext env var INFERENCE_<TARGET_NAME_UPPER>_TOKEN is still honoured as a fallback (migrate with 'inference-aiops secret migrate'). The token is sent as an Authorization: Bearer header at request time and held only in memory; it is never logged or echoed.
  State-changing operations require double confirmation at the CLI layer and support --dry-run. All write tools pass through the @governed_tool decorator (pre-check + budget guard + audit + risk-tier gate). The fragile prod ops — scale_replicas_down, scale_to_zero, drain_replica, lora_unload, model_hot_swap, replica_restart, model_undeploy, deployment_redeploy — are high-risk with a dry_run preview; reversible writes (scale, autoscale-config, routing, hot-swap, LoRA load) record an undo descriptor.
  Engines: vLLM (with its Ray Serve control plane), SGLang, and TGI. SGLang/TGI are single-process servers with engine-agnostic observability (health, running-model inventory, request metrics, queue depth, latency RCA); Ray-shaped scale/drain writes are vLLM-only and raise a teaching error on a SGLang/TGI target.
  Metrics: each engine's Prometheus /metrics endpoint is parsed directly — no Prometheus server is required.
  Webhooks: none — no outbound calls beyond the configured Ray dashboard and vLLM services.
  SSL: verify_ssl defaults to true; disable only for self-signed lab certificates.
  Transitive dependencies: httpx (HTTP client) and the MCP SDK. No post-install scripts or background services.
  Validation status: behaviour is exercised against mocked vLLM/Ray responses; unverified against multi-GPU tensor/pipeline-parallel deployments, real GPU thermal/throttle telemetry, and multi-node drain (see docs/VERIFICATION.md).
---

# Inference AIops

> **Disclaimer**: Community-maintained open-source project, **not affiliated with, endorsed by, or sponsored by the vLLM or Ray projects or any inference-serving vendor.** Product and trademark names belong to their owners. Source at [github.com/AIops-tools/Inference-AIops](https://github.com/AIops-tools/Inference-AIops) under the MIT license.

Governed GPU-inference operations for **vLLM** (OpenAI API + Prometheus `/metrics`) and **Ray Serve / Ray Jobs** (Ray dashboard), plus the single-process serving engines **SGLang** and **TGI** — **37 MCP tools**, every one wrapped with the bundled `@governed_tool` harness: a local unified audit log under `~/.inference-aiops/`, policy engine, token/runaway budget guard, undo-token recording, and graduated-autonomy risk tiers. The flagship `diagnose_latency_spike` folds queue depth + KV-cache pressure + prefix-cache locality into a ranked cause and the specific knob to turn; the engine-agnostic `diagnose_engine_latency` does the same across whatever signals SGLang/TGI expose. Each engine's Prometheus `/metrics` is parsed directly — **no Prometheus server required**.

> **Standalone**: the governance harness is bundled in the package (`inference_aiops.governance`) — no external skill-family dependency. A bearer token is **optional** (many stacks run open).

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

**21 read, 14 write**, plus `undo_list` / `undo_apply` — **37 MCP tools** in total. The high-risk writes support `dry_run` + double-confirm; reversible writes record an undo descriptor. The engine-agnostic reads cover any engine; the Ray Serve / cluster / deploy write groups are vLLM-only and teach-and-refuse on a SGLang/TGI target (single-process engines have no Ray control plane).

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

> **Secure by default (v0.2.0+)**: with no `~/.inference-aiops/rules.yaml`, high/critical operations are denied unless `INFERENCE_AUDIT_APPROVED_BY` names an approver (set `INFERENCE_AUDIT_RATIONALE` too). `inference-aiops init` seeds a starter rules.yaml; an operator-authored rules file is honoured as-is.

### 1. "Inference got slow this afternoon" (flagship RCA → the right knob)

1. `inference-aiops doctor` → confirm the vLLM endpoint and Ray dashboard are actually reachable before blaming the model
2. `inference-aiops overview` → Serve deployments, total replicas, and whether queue backpressure is cluster-wide or one deployment
3. `inference-aiops metrics diagnose` (MCP: `diagnose_latency_spike`) → a **ranked** cause with the measured numbers: is `waiting` queue depth high (backpressure)? Are there KV-cache **preemptions** (`kv_cache_stats`)? Has the **prefix-cache hit rate** dropped (routing lost locality)?
4. Turn the knob the RCA names, not a guess:
   - backpressure → `inference-aiops serve scale <app> <deployment> --replicas N` (`scale_replicas_up`, reversible, prior count captured)
   - KV-cache preemption → `autoscale_config_update` to lower the concurrent-request cap (reversible, prior config captured)
   - lost locality → `routing_policy_update` to prefix-aware / session-affinity (reversible)
5. Re-check `inference-aiops metrics requests` (TTFT / TPOT / e2e) and `inference-aiops metrics queue` to confirm the p99 actually moved
6. **Failure branch**: if the fix makes it worse, `inference-aiops undo list` → `inference-aiops undo apply <id>` restores the exact prior replica count / autoscale config / routing policy. If `diagnose_latency_spike` reports no clear cause, the bottleneck is likely upstream of serving — check `gpu_utilization` for a throttling or shared-GPU problem before scaling anything.

### 2. Off-peak cost save: scale a deployment down to zero and bring it back

1. `inference-aiops metrics requests` → confirm traffic really is idle, not just briefly quiet
2. `diagnose_low_utilization` → the deployments actually burning GPU for nothing, with the measured utilisation
3. `cost_per_token` → quantify the bleed ($/1M tokens at the current throughput) so the change is justifiable in the audit trail
4. `export INFERENCE_AUDIT_APPROVED_BY=you INFERENCE_AUDIT_RATIONALE="off-peak cost save"`
5. `inference-aiops serve scale-to-zero <app> <deployment> --dry-run`, then re-run without `--dry-run` → **high** risk, double confirmation. `scale_to_zero` stops the bleed but **strands ingress** — requests will queue or fail until replicas return
6. To restore: `inference-aiops undo apply <id>` (replays the captured prior replica count) or `inference-aiops serve scale <app> <deployment> --replicas N`
7. **Failure branch**: if traffic arrives while at zero, restore immediately via undo — do not wait for autoscale, since `scale_to_zero` may have been applied outside the autoscaler's floor. If the restore fails, `serve status` will show the deployment unhealthy; `deployment_redeploy` is the last resort (high risk, disruptive).

### 3. Drain a replica before a node reboot

1. `inference-aiops serve list` / `replica_list` → identify the replicas pinned to the node you are about to reboot
2. `queue_depth` → confirm the remaining replicas can absorb the load; if not, `scale_replicas_up` **first** so draining does not cause a brownout
3. `drain_replica <app> <deployment> <replica_id> --dry-run`, then confirm → **high** risk; the drain finishes in-flight requests before removing the replica
4. Watch `replica_list` until the replica is gone and `request_metrics` shows no error spike, then reboot the node
5. **Failure branch**: if the drain hangs on a long-running request, `replica_restart` forcibly cycles it — that **drops** in-flight requests, so only reach for it once you accept the loss. Multi-node drain has not been verified against a live cluster (see `docs/VERIFICATION.md`).

### 4. Hot-swap a base model, verify it, roll back if quality regresses

1. `model_list` / `model_info` → record what is serving now (the undo descriptor will capture this, but you want it in front of you)
2. `request_metrics` → capture a **baseline** TTFT/TPOT before the swap, so "it feels slower" can be checked against a number
3. `model_hot_swap <new_model> --dry-run`, then confirm → **high** risk; a Sleep-Mode base swap that captures the prior model into an undo descriptor
4. For adapter-level changes instead of a base swap: `lora_load` (reversible) and `lora_unload` (high) are the lighter path
5. Verify: `model_info` shows the new model, and `request_metrics` is compared against the step-2 baseline
6. **Failure branch**: if latency or quality regresses, `inference-aiops undo apply <id>` replays the swap back to the **captured** prior model — not a guessed one. If the swap left the deployment unhealthy rather than merely worse, `deployment_redeploy` restores a clean state.

## Governance & Safety

- Every tool is audited to `~/.inference-aiops/audit.db` (relocatable via `INFERENCE_AIOPS_HOME`).
- High-risk ops can require a named approver: set `INFERENCE_AUDIT_APPROVED_BY` and `INFERENCE_AUDIT_RATIONALE`.
- The fragile prod writes support `--dry-run` and double confirmation at the CLI.
- Reversible writes (scale, autoscale-config, routing, hot-swap, LoRA load) record an inverse descriptor.

## References

- `references/capabilities.md` — full tool → backend → endpoint → returns reference
- `references/cli-reference.md` — CLI command reference
- `references/setup-guide.md` — onboarding, optional token, and connectivity
