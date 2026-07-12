# Security Policy

## Disclaimer

Community-maintained open-source project. **Not affiliated with, endorsed by, or
sponsored by the vLLM or Ray projects.** Product and trademark names belong to
their owners. Source is publicly auditable under the MIT license.

## Reporting Vulnerabilities

Report privately via a GitHub Security Advisory on
[github.com/AIops-tools/Inference-AIops](https://github.com/AIops-tools/Inference-AIops/security/advisories)
or email zhouwei008@gmail.com. Please do not open public issues for security
reports.

## Security Design

### Credential Management
- A bearer token is **optional** (many on-prem inference stacks run open). When
  used, per-target tokens live **encrypted** in `~/.inference-aiops/secrets.enc`
  (Fernet/AES-128 + scrypt-derived key; chmod 600), never in `config.yaml` and
  never in source. The master password is never stored — only a per-store random
  salt and the ciphertext are on disk.
- A legacy plaintext env var `INFERENCE_<TARGET_NAME_UPPER>_TOKEN` is still
  honoured as a fallback with a deprecation warning (migrate with
  `inference-aiops secret migrate`).
- When present, the token is sent as an `Authorization: Bearer` header and held
  only in memory; it is never logged or echoed. The config file holds only host,
  Ray/vLLM ports, scheme, and TLS settings.

### Governed Operations
Every MCP tool runs through the bundled `@governed_tool` harness
(`inference_aiops.governance`):
- **Audit** — every call logged to a local SQLite DB under `~/.inference-aiops/`
  (relocatable via `INFERENCE_AIOPS_HOME`), agent-attributed, secret-redacted.
- **Token/runaway budget** — hard ceilings (`INFERENCE_MAX_TOOL_CALLS` /
  `INFERENCE_MAX_TOOL_SECONDS`) plus an on-by-default guard that trips a tight
  poll/retry loop, preventing unbounded API consumption (e.g. polling a slow
  session).
- **Graduated risk tiers** — `~/.inference-aiops/rules.yaml` `risk_tiers` gate
  writes by environment/tag; the highest tiers require a recorded approver.
- **Undo-token recording** — reversible writes capture the BEFORE state and
  record an inverse descriptor (e.g. a scale op → restore the prior replica
  count, `autoscale_config_update` → restore prior bounds, `routing_policy_update`
  → restore the prior policy) so the change can be rolled back.

### State-Changing Operations
Destructive/traffic-affecting writes — `scale_replicas_down`, `scale_to_zero`,
`drain_replica`, `lora_unload`, `model_hot_swap`, `model_undeploy`,
`deployment_redeploy`, `replica_restart` — are `risk_level=high`, accept a
`dry_run` preview, and (under `risk_tiers`) require a recorded approver
(`INFERENCE_AUDIT_APPROVED_BY` + `INFERENCE_AUDIT_RATIONALE`). The CLI
additionally double-confirms `serve scale-to-zero` and supports `--dry-run`.
Reversible medium/low writes capture before-state and record an undo token.

### SSL/TLS Verification
`verify_ssl` is off by default (inference stacks commonly serve plain HTTP on a
trusted network); set `scheme: https` + `verify_ssl: true` for TLS endpoints.

### Prompt-Injection Protection
All server-returned text (model ids, deployment/replica names, job entrypoints,
Prometheus label values) is passed through a `sanitize()` truncate +
control-character strip before reaching the agent.

### Network Scope
No webhooks, no telemetry, no outbound calls beyond the configured Ray dashboard
and vLLM endpoints. No post-install scripts or background services.

## Static Analysis

```bash
uvx bandit -r inference_aiops/ mcp_server/
uv run ruff check .
```

## Supported Versions

The latest released version receives security fixes. This is a preview (0.x);
pin a version in production.
