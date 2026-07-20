<!-- mcp-name: io.github.AIops-tools/inference-aiops -->

# Inference AIops

> **Disclaimer**: Community-maintained open-source project. **Not affiliated with, endorsed by, or sponsored by the vLLM or Ray projects or any inference-serving vendor.** Product and trademark names belong to their owners. MIT licensed.

Governed AI-ops for **GPU inference clusters** — **vLLM** (OpenAI API + Prometheus
`/metrics`) and **Ray Serve / Ray Jobs** (Ray dashboard), plus the single-process
serving engines **SGLang** and **TGI (Text Generation Inference)** — with a
**built-in governance harness**: unified audit log, policy engine, token/runaway
budget guard, undo-token recording, and graduated-autonomy risk tiers. It parses
each engine's Prometheus `/metrics` directly (no Prometheus server required) and
probes the Ray dashboard independently. A bearer token is **optional** (many
stacks run open).

**Serving engines.** vLLM is the flagship (full Ray Serve control plane: scale,
drain, autoscale, LoRA, hot-swap). **SGLang** and **TGI** are supported for
engine-agnostic observability — health, running-model identity, request-latency
metrics, queue depth, and latency RCA — read from each engine's own endpoints and
metric names. Being single-process servers, they have no Ray-shaped scale/drain
API: those writes return a teaching error pointing you at a real horizontal-scale
layer (Ray Serve / Kubernetes / a load balancer).

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

## Security: read-only mode

This tool is meant to be handed to an AI agent, so its safety story is enforced
by the server rather than requested in a prompt:

```bash
export INFERENCE_READ_ONLY=1
```

With that set, the **16 write tools are never registered**. An MCP client
lists **23 tools instead of 39** — the writes are not hidden, not
gated behind a flag, and not merely refused when called. They are absent from
the session. A model cannot invoke a tool it was never offered, and cannot be
argued into one.

That distinction is the whole point. A tool that exists but refuses still invites
retry loops and "I'll describe the call instead" behaviour from smaller models,
and it leaves a reviewer trusting a promise. An absent tool is a fact you can
check: connect, list the tools, and see that the writes are not there.

Enforcement is two layers deep, so the switch cannot be sidestepped by changing
entry point:

| Layer | What it does | Covers |
|---|---|---|
| `@governed_tool` harness | refuses every non-read operation outright | MCP, CLI, and in-process callers |
| MCP registration | write tools are removed from `list_tools()` | anything speaking MCP |

Read operations are unaffected, and every call is still audited to
`~/.inference-aiops/audit.db`.

> The read/write split is derived from each tool's declared `risk_level`, and a
> test asserts that this never disagrees with the `[READ]`/`[WRITE]` tag in the
> tool's own documentation — so a write can't quietly present itself as a read.

Running a smaller / local model? See
[agent-guardrails.md](skills/inference-aiops/references/agent-guardrails.md) — it lists
the guardrails this tool now enforces for you (so you don't spend prompt budget
restating them) and gives a ready-made system prompt for what's left.

## Capability matrix (39 MCP tools)

| Group | Tools | Count | R/W (risk) |
|-------|-------|:-----:|:-----------|
| **Metrics & RCA** (vLLM) | `request_metrics`, `queue_depth`, `kv_cache_stats`, `diagnose_latency_spike`, `diagnose_low_utilization` | 5 | read |
| **Engine-agnostic** (vLLM / SGLang / TGI) | `engine_health`, `engine_inventory`, `engine_request_metrics`, `engine_queue_depth`, `diagnose_engine_latency` | 5 | read |
| **Ray Serve (read)** | `serve_deployment_list`, `deployment_status`, `replica_list`, `autoscale_config_get` | 4 | read |
| **Ray Serve (write)** | `scale_replicas_up`, `scale_replicas_down`, `scale_to_zero`, `autoscale_config_update`, `drain_replica` | 5 | write (med / **high**) |
| **Models / vLLM** | `model_list`, `model_info`, `model_is_sleeping`, `lora_load`, `lora_unload` | 5 | read + write (med) |
| **Sleep Mode / vLLM** (needs `VLLM_SERVER_DEV_MODE=1`) | `model_sleep`, `model_wake` | 2 | write (**high** / med) |
| **Ray cluster / jobs / GPU** | `ray_cluster_resources`, `ray_dashboard_status`, `ray_job_list`, `gpu_utilization`, `ray_job_cancel`, `replica_restart` | 6 | read + write (med / **high**) |
| **Deploy lifecycle** | `model_deploy`, `model_undeploy`, `deployment_redeploy`, `routing_policy_update` | 4 | write (med / **high**) |
| **Cost** | `cost_per_token` | 1 | read |

The engine-agnostic group works against **any** supported engine (including
vLLM); use it for SGLang/TGI targets or a uniform view across a mixed fleet. The
Ray Serve / cluster / deploy write groups are vLLM-only (Ray control plane) — they
teach-and-refuse on a SGLang/TGI target.

**23 read, 16 write.** High-risk writes (`scale_replicas_down`,
`scale_to_zero`, `drain_replica`, `lora_unload`, `model_sleep`,
`replica_restart`, `model_undeploy`, `deployment_redeploy`) all support
`dry_run` + double-confirm; reversible writes record an undo descriptor.

> **Sleep Mode requires a dev-mode server.** vLLM registers `/sleep`,
> `/wake_up` and `/is_sleeping` **only** when started with
> `VLLM_SERVER_DEV_MODE=1`. Against any other server these three tools
> report that the route is absent and why, rather than failing vaguely.
> Sleep Mode suspends the **same** model; it does not swap base models —
> serving a different base model means restarting vLLM with a different
> `--model`.

## Install

```bash
uv tool install inference-aiops          # or: pipx install inference-aiops
```

## Quick start

```bash
inference-aiops init                     # wizard: engine (vllm/sglang/tgi) + host + port + scheme
inference-aiops doctor                   # vLLM: probes Ray + vLLM; SGLang/TGI: engine health + inventory
inference-aiops overview                 # deployments + total replicas + queue backpressure
inference-aiops metrics diagnose         # why is inference slow? ranked RCA + the knob to turn
inference-aiops serve list               # Ray Serve deployments + replica counts
```

Run as an MCP server (stdio) for the full 39-tool surface:

```bash
export INFERENCE_AIOPS_MASTER_PASSWORD=...   # only if a bearer token is stored
inference-aiops mcp
```

The CLI is a convenience subset (`init`, `overview`, `serve …`, `metrics …`,
`secret …`, `doctor`, `mcp`); the full 39 tools are exposed via the MCP server.

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

Behaviour is exercised by the test suite against mocked vLLM `/metrics`, vLLM
OpenAI API, and Ray dashboard responses. **~80% of the tool self-tests on a
laptop** — vLLM on a single GPU or CPU-mock plus a local one-node Ray head. It
has not been run against a live production cluster; see
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) for the live-verification
checklist.

Unverified against real hardware / topology:

- multi-GPU **tensor-parallel / pipeline-parallel** deployments,
- real GPU **thermal / throttle** telemetry (utilisation is best-effort from
  the Ray dashboard's `/api/nodes`),
- **multi-node drain** and node-reboot orchestration.

The fastest live check is `inference-aiops doctor`; the full checklist lives in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## Missing a capability?

This is the GPU-inference member of the AIops-tools family (governed AI-ops with
audit + budget + undo + risk tiers). If a vLLM or Ray capability you need is
missing, or your stack speaks a dialect these tools don't yet handle — open an
issue or a PR. Contributions welcome.
