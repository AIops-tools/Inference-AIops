# Live verification — inference-aiops

`inference-aiops` has two planes: a **Ray control plane** (Serve + jobs + cluster,
port 8265) and a **vLLM engine plane** (OpenAI JSON + Prometheus `/metrics`, port
8000). As of 2026-08-01 the **Ray control plane is live-verified** against a real
Ray cluster; the vLLM engine plane still needs a GPU and remains mock-only.

## ✅ Ray control plane — live-verified against real Ray 2.56.1 (2026-08-01)

Verified against a real `rayproject/ray:2.56.1` head (native x86_64, no GPU),
driven through the real governed CLI + ops + MCP path with a declaratively
deployed Serve app. **This run found that essentially the ENTIRE Ray control
plane was built against wrong or nonexistent REST endpoints — the tool had never
worked against a real Ray cluster.** Ten distinct bugs, all fixed and
regression-tested; the write fixes verified end-to-end (change landed on the
cluster, then reverted):

**Serve writes — Ray Serve's REST control plane is DECLARATIVE (only
`PUT /api/serve/applications/`, the whole ServeDeploySchema); there are no
per-deployment or per-replica endpoints.** The old code targeted invented paths:

1. `scale_replicas_up/down`, `scale_to_zero` — PUT `.../deployments/{dep}` (404
   on every real Ray). Now: GET all app configs → patch the target deployment's
   `num_replicas` → PUT the whole schema. **Live: scaled 2→3 (cluster reconciled),
   audit row, `undo_apply` → back to 2, `effectVerified`.**
2. `update_autoscale_config` — PUT `.../deployments/{dep}/autoscale` (404). Now the
   same declarative round-trip on `autoscaling_config`.
3. `deploy_model` — PUT a bare `{name, import_path, num_replicas}` (400: not a
   ServeDeploySchema). Now merges the app into the full schema, PRESERVING other
   apps. **Live: deployed a 2nd app without deleting the first.** (Signature lost
   `num_replicas` — Ray takes replica counts per-deployment; set them via `scale`.)
4. `undeploy_model` — DELETE `.../applications/{app}` (404; Ray has no per-app
   delete). Now PUTs the schema WITHOUT that app. **Live: removed only the target.**
5. `redeploy_deployment` — PUT `.../deployments/{dep}/redeploy` (404). Now re-PUTs
   the app's current config (a real reconcile).
6. `drain_replica`, `restart_replica` — POST `.../replicas/{id}/drain|restart`
   (404; no per-replica REST endpoint exists). Now refuse with a teaching
   `EngineCapabilityError` (scale down / redeploy instead). **Live: both refuse.**
7. `update_routing_policy` — PUT `.../deployments/{dep}/routing` (404; routing is a
   code-level `@serve.deployment` option, not REST-settable). Now refuses honestly.

**Ray reads:**

8. `get_cluster_resources` — read `data.clusterResources`/`availableResources`,
   fields that **no longer exist** in current Ray → returned all-null on every live
   cluster (bug class #7). Now sums
   `clusterStatus.loadMetricsReport.usageByNode` (`{res: [used, total]}`). **Live:
   totalCpu 2.0 / availableCpu 1.6 matches the cluster; GPU null on a CPU-only
   node — unknown, not a fabricated 0.**
9. `get_gpu_utilization` — hit `/api/nodes` (404; the route is now `/nodes`). Fixed
   to `/nodes?view=summary`. **Live: returns the real node, `gpuCount: 0`.**
10. A guardrail added along the way: a single-app change refuses if ANY app is
    imperatively deployed (no `deployed_app_config`), because the whole-schema PUT
    would otherwise silently delete that bystander app.

Not covered by this run: the **vLLM engine plane** (metrics RCA, KV-cache, TTFT/
TPOT, engine writes) — needs a real GPU-backed vLLM, still mock-only.

> The engine plane is exercised by a mock-only test suite (no real vLLM). Until it
> is run against a GPU-backed engine, we do not claim it works against real
> serving endpoints. The checklist below still stands for that half.

## What the mock suite already guarantees

- Every module imports; the CLI builds; every MCP tool carries the
  `@governed_tool` harness marker (`tests/test_smoke.py`).
- The Prometheus `/metrics` text parser extracts TTFT/TPOT/queue/KV-cache series
  correctly from captured exposition-format samples — including missing and
  renamed series.
- `diagnose_latency_spike` / `diagnose_low_utilization` / `diagnose_engine_latency`
  rank the expected cause from synthetic metric fixtures, and name the knob.
- Write tools carry the correct risk tier and record the correct inverse undo
  descriptor; the undo executor replays it.
- Ray-shaped writes teach-and-refuse on an SGLang/TGI target.
- Governance genuinely persists: audit rows and undo tokens land in a real
  SQLite DB.

What it does **not** guarantee: that real vLLM/Ray metric names, Ray dashboard
JSON shapes, and Serve scaling semantics match what the analyses assume — nor
that the GPU-utilisation reads mean what we think on real hardware.

### Sleep Mode paths are modelled from documentation — NOT live-verified

`model_sleep`, `model_wake` and `model_is_sleeping` call `POST /sleep?level=N`,
`POST /wake_up` and `GET /is_sleeping`. Those paths, the `level` semantics
(1 = offload weights to CPU RAM, 2 = discard them), the `is_sleeping` response
shape, and the `VLLM_SERVER_DEV_MODE=1` gate are all taken from vLLM's
documentation. **None of it has been exercised against a running vLLM server**,
and this repo cannot be verified on the current development machine: the
`rayproject/ray` image is amd64-only and collapses under QEMU on arm64 (its
dashboard dies as soon as it is queried). Status: **UNKNOWN — pending live**.

This is stated plainly because the same class of defect has already shipped from
this repo once. The tool these three replace, `model_hot_swap`, POSTed to
`/v1/hot_swap` — an endpoint vLLM has never served. The mock suite was green
because the fixtures asserted the same invented path the code called, so the
tests proved only that the code agreed with itself. Doc-modelled paths are a
weaker claim than mock-tested ones, not a stronger one; treat everything in this
section as unconfirmed until the checklist below is run.

## Prerequisites for a live run

**Cheap path (~80% of the checklist)**: vLLM on a single GPU — or a CPU-only
mock model — plus a one-node Ray head (`ray start --head`) on the same box, with
one throwaway Ray Serve deployment. This covers sections 1–3 and 5.

**Full path (sections 4)**: a multi-node Ray cluster with ≥2 GPU nodes and a
tensor-parallel deployment. Never verify against a deployment serving real
traffic — scale-to-zero and drain strand ingress by design.

```bash
uv tool install inference-aiops
inference-aiops init       # engine + host + port; bearer token optional
```

## Verification checklist

Tick every box. A box that cannot be ticked is a verification gap — record it,
do not silently pass.

### 1. Connectivity (the fastest live gate)
- [ ] `inference-aiops doctor` → green for both the vLLM endpoint and the Ray
      dashboard (and, on an SGLang/TGI target, engine health + inventory).

### 2. Reads return real, well-shaped data
- [ ] `inference-aiops overview` → the actual Serve deployments and replica
      counts, matching the Ray dashboard UI.
- [ ] `inference-aiops metrics requests` → TTFT / TPOT / e2e values that move
      when you send real load (drive traffic and watch them change; a static
      value means the parser latched onto the wrong series).
- [ ] `inference-aiops metrics queue` / `queue_depth` → `waiting` rises under
      concurrent load and drains afterwards.
- [ ] `kv_cache_stats` → utilisation, prefix-cache hit rate, and preemption
      counters are populated and plausible (force a preemption with a long
      context to confirm the counter is real, not always zero).
- [ ] `gpu_utilization` → per-node values match `nvidia-smi` at the same moment.
- [ ] `ray_cluster_resources`, `ray_dashboard_status`, `ray_job_list`,
      `replica_list`, `serve_deployment_list`, `model_list`, `model_info` → no
      crash, and fields match the Ray dashboard / `/v1/models`.
- [ ] `cost_per_token` → the $/1M figure is arithmetically correct for the
      measured throughput and the GPU $/hr you supplied.

### 3. RCA is right, not just non-crashing
- [ ] Induce **queue backpressure** (concurrency far above the batch cap) →
      `inference-aiops metrics diagnose` ranks queue depth as the top cause and
      names scaling / batch-cap as the knob.
- [ ] Induce **KV-cache preemption** (long contexts, small `gpu_memory_utilization`)
      → the RCA ranks KV-cache pressure first, not queue depth.
- [ ] Leave a deployment idle → `diagnose_low_utilization` flags it with the
      measured utilisation.
- [ ] A misdiagnosis in any of the three is a **blocking** finding: fix it and
      add a fixture test before ticking.

### 4. A reversible write + its undo (governance closes the loop)
- [ ] `inference-aiops serve scale <app> <deployment> --replicas N` → replicas
      actually change; the result carries an `_undo_id`; a row lands in
      `~/.inference-aiops/audit.db`.
- [ ] `inference-aiops undo list` → `inference-aiops undo apply <id>` restores
      the **prior** replica count (proves undo captured pre-state, not a guess).
- [ ] `autoscale_config_update` then `undo apply` → the prior min/max/target
      config comes back exactly.
- [ ] `inference-aiops serve scale-to-zero <app> <deployment> --dry-run` → prints
      the call, changes nothing; then for real → replicas reach 0 and `undo apply`
      brings back the captured count.
- [ ] `lora_load` then `lora_unload` → the adapter appears in and disappears
      from `model_list`.
- [ ] **Sleep Mode (needs `VLLM_SERVER_DEV_MODE=1`)** — `model_is_sleeping`
      reports false; `model_sleep --dry-run` changes nothing; for real → GPU
      memory is released and `model_is_sleeping` reports true; `undo apply`
      wakes it and the engine serves again.
- [ ] `model_sleep` against an **already sleeping** engine records **no** undo
      descriptor (it changed nothing, so there is nothing to reverse).
- [ ] **Non-dev-mode server** (the common production case): all three Sleep-Mode
      tools report that the route does not exist *and* name
      `VLLM_SERVER_DEV_MODE=1` as the reason. Criterion: the message must not
      read as a stale id or a generic failure.

### 5. Multi-node / topology behaviour (needs ≥2 GPU nodes)
- [ ] `drain_replica --dry-run`, then for real → in-flight requests complete
      (no 5xx spike in `request_metrics`) before the replica is removed.
- [ ] A **tensor-parallel** deployment is read correctly by `replica_list` and
      `gpu_utilization` (this is the topology most likely to break assumptions).
- [ ] `replica_restart` cycles a stuck replica and the deployment returns to
      healthy.

### 6. Governance records (it does not gate)
- [ ] A `high`-risk op (`scale_to_zero`, `drain_replica`, `model_undeploy`) runs
      with **no** approver set, and the audit row records it with
      `risk_tier=review`. There is no read-only switch, policy file, or approval
      gate to configure.
- [ ] With `INFERENCE_AUDIT_APPROVED_BY` / `INFERENCE_AUDIT_RATIONALE` set, the
      audit row additionally records the approver and rationale — an annotation
      only, never required.
- [ ] A failed write is audited with `status=error` and records **no** undo token.
- [ ] A tight metrics-poll loop trips the runaway budget guard rather than
      hammering `/metrics`.

### 7. Cleanup
- [ ] `model_undeploy` the throwaway deployment; confirm it is audited and
      tagged `high`, and the Ray cluster returns to idle.

## Criteria to consider this tool live-verified

Record `inference-aiops` as live-verified **only when all of the following hold**:

1. Every box in sections 1–4, 6 and 7 is ticked against a real vLLM + Ray stack,
   and the versions are recorded (e.g. "verified on vLLM 0.8 / Ray 2.40").
   Section 5 is ticked and recorded **separately** as "multi-node verified" —
   do not let a single-node run imply multi-node coverage.
2. Any metric-name, JSON-shape, or unit mismatch found during the run is fixed
   **and covered by a fixture test**, so the mock suite cannot regress it.
3. Section 3 (RCA correctness) passed against **induced** conditions, not merely
   observed ones — an RCA that never fires is not a verified RCA.
4. The run is written up in this repo's release notes with the date, the tool
   version, and the engine/Ray versions, matching how the line records its other
   live-verified tools.

Until then this document is the accurate statement of status — and no positive
claim about real-hardware behaviour should appear in the README or SKILL.md.

## Notes for maintainers

- `inference-aiops doctor` is the single fastest live entry point; start there.
- A CPU-only mock model gets you sections 1, 2 (partially), 4 and 6 with no GPU
  at all — that is the cheapest meaningful gate.
- SGLang and TGI need their own pass of sections 1–3 via the engine-agnostic
  reads; a vLLM-only run does not verify them.
- The verification story for the whole product line is tracked centrally; add
  this tool's result there once green so the verification-debt ledger stays
  accurate.
