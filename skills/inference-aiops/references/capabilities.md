# inference-aiops capabilities

> Preview / mock-only. 30 MCP tools (16 read, 14 write). Backends: **vLLM**
> (OpenAI API + Prometheus `/metrics`, default port 8000) and **Ray** dashboard
> (Serve + Jobs, default port 8265). Endpoints modelled against those APIs; need
> live verification.

## Metrics & RCA (read, 5)

| Tool | Backend | Endpoint | Returns |
|------|---------|----------|---------|
| `request_metrics` | vLLM | `GET /metrics` | TTFT, TPOT, e2e latency (avg/p50/p90/p99), prompt/generation token totals, request counts |
| `queue_depth` | vLLM | `GET /metrics` | running vs waiting requests (backpressure), scheduler state |
| `kv_cache_stats` | vLLM | `GET /metrics` | KV-cache utilisation %, prefix-cache hit rate, preemption count |
| `diagnose_latency_spike` | vLLM | `GET /metrics` (fold) | **ranked cause** (queue backpressure / KV-cache preemption / prefix-cache locality) + the specific knob to turn |
| `diagnose_low_utilization` | vLLM | `GET /metrics` (fold) | idle-GPU / over-provisioned / routing-stranded diagnosis + what to scale down |

## Ray Serve — read (4)

| Tool | Backend | Endpoint | Returns |
|------|---------|----------|---------|
| `serve_deployment_list` | Ray | `GET /api/serve/applications/` | all Serve deployments: status, replica count, target |
| `deployment_status` | Ray | `GET /api/serve/applications/` | one deployment's status + current/target replica count |
| `replica_list` | Ray | `GET /api/serve/applications/` | per-replica id, state, node |
| `autoscale_config_get` | Ray | `GET /api/serve/applications/` | min/max replicas, target ongoing requests |

## Ray Serve — write (5)

| Tool | Risk | Backend | Endpoint | Undo / safety |
|------|------|---------|----------|---------------|
| `scale_replicas_up` | med | Ray | `PUT /api/serve/applications/{app}/deployments/{dep}` | reversible (records prior count) |
| `scale_replicas_down` | **high** | Ray | `PUT …/deployments/{dep}` | dry-run; captures prior count → undo |
| `scale_to_zero` | **high** | Ray | `PUT …/deployments/{dep}` | dry-run; stops cost bleed but **strands ingress**; captures prior count → undo |
| `autoscale_config_update` | med | Ray | `PUT …/deployments/{dep}/autoscale` | reversible (records prior bounds) |
| `drain_replica` | **high** | Ray | `POST …/deployments/{dep}` (drain) | dry-run; graceful — finishes in-flight requests; no undo |

## Models / vLLM (5)

| Tool | R/W (risk) | Backend | Endpoint | Notes |
|------|-----------|---------|----------|-------|
| `model_list` | read | vLLM | `GET /v1/models` | served model ids |
| `model_info` | read | vLLM | `GET /v1/models` | one model's detail (normalised) |
| `lora_load` | write (med) | vLLM | `POST /v1/load_lora_adapter` | reversible (undo unloads it) |
| `lora_unload` | write (**high**) | vLLM | `POST /v1/unload_lora_adapter` | dry-run |
| `model_hot_swap` | write (**high**) | vLLM | `POST /v1/hot_swap` | dry-run; Sleep-Mode base swap; captures prior model → undo |

## Ray cluster / jobs / GPU (6)

| Tool | R/W (risk) | Backend | Endpoint | Returns / notes |
|------|-----------|---------|----------|-----------------|
| `ray_cluster_resources` | read | Ray | `GET /api/cluster_status` | CPU/GPU total vs allocated |
| `ray_dashboard_status` | read | Ray | Ray dashboard | dashboard reachability/version |
| `ray_job_list` | read | Ray | `GET /api/jobs/` | submitted jobs + status |
| `gpu_utilization` | read | Ray | `GET /api/nodes` | per-node GPU count, utilisation %, memory (best-effort) |
| `ray_job_cancel` | write (med) | Ray | `POST /api/jobs/{id}/stop` | cancel a running job |
| `replica_restart` | write (**high**) | Ray | `GET/PUT /api/serve/applications/` | dry-run; restart a stuck replica |

## Deploy lifecycle (4, write)

| Tool | Risk | Backend | Endpoint | Undo / safety |
|------|------|---------|----------|---------------|
| `model_deploy` | med | Ray | `PUT /api/serve/applications/` | deploy an application |
| `model_undeploy` | **high** | Ray | `DELETE/PUT /api/serve/applications/` | dry-run |
| `deployment_redeploy` | **high** | Ray | `PUT /api/serve/applications/` | dry-run |
| `routing_policy_update` | med | Ray | `PUT /api/serve/applications/` | reversible; prefix-aware / session-affinity routing to fix cache locality |

## Cost (read, 1)

| Tool | Backend | Endpoint | Returns |
|------|---------|----------|---------|
| `cost_per_token` | vLLM | `GET /metrics` + GPU $/hr | deterministic $/1M tokens from measured throughput × GPU hourly rate |

## Out of scope (by design)

- Cluster **provisioning** (spinning up GPU nodes, driver/CUDA install)
- vLLM/Ray **install or version upgrades**
- Multi-node **drain / reboot orchestration** (single-replica drain only, unverified at multi-node scale)
- Non-inference infrastructure (use the appropriate other AIops-tools line)

Want one of these? Open an issue or PR — feedback and contributions welcome.
