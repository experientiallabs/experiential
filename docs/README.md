# Documentation

Customer documentation describes only supported current services. Design notes, raw experiment
results, and plans do not live here.

## Documentation index

| File | Purpose |
|---|---|
| `usage.md` | CLI map for build, bounded optimize model, optimize router, run, config, login, and foreground macOS capture. |
| `reference/providers.md` | Catalog providers, first-build `--provider` flags, environment variables, Azure endpoint and deployment rules, the Bedrock credential chain, and OpenAI-compatible listing metadata plus identity-only operator declaration. |
| `reference/gateway-architecture.md` | Operational local gateway contracts, certified exact-model routing, ownership boundaries, and compatibility locks. |
| `reference/model-chains.md` | Ordered cross-model stages, mandatory hosted authority, bounded recovery and unsupported local-chain serving. |
| `reference/chat-logprobs.md` | Verified Chat probability admission, token records, and guardrail boundaries. |
| `reference/responses-logprobs.md` | Native Responses probability selectors, lifecycle phases, and continuation behavior. |
| `reference/gateway-request-policy.md` | Per-request retry bounds, backoff, opaque route selection, replay and physical-attempt accounting. |
| `reference/gateway-failover-rules.md` | Per-rung conditional failover (`failover_only_on`): the token vocabulary, first-dial and successor rules, the ledger `fallback_reason`, and the fail-closed cases. |
| `reference/gateway-guardrails.md` | Identity-scoped gateway guardrails: data flow, latency, privacy, standard pack, http_json adapters, and inappropriate-content use. |
| `reference/gateway-latency.md` | Routine CI gateway-latency report against a local mock: schedule, artifact schema, and numeric latency badge. |
| `reference/openai-compatible-recipes.md` | Verified Fireworks, Modal, and Experiential Cloud connection recipes through the openai-compatible provider family. |
| `reference/ingest.md` | Current declared local trace source contract for every supported source. |
| `reference/immutable-real-trace-rag.md` | Immutable real-trace retrieval provenance, leakage, persistence, and historical restoration contract. |
| `reference/router_optimization_config.md` | Exact completed-evidence configuration recipe for router optimization. |
| `release-scope.md` | Supported and explicitly excluded release claims. |
| `research/w16-router-sandbox-evidence.md` | Deterministic W16 router and local sandbox evidence, limits, and replay commands. |
| `research/rust-gateway-engine.md` | Native (Rust) gateway data-plane benchmarks: streaming concurrency, the shared SQLite durability ceiling, and horizontal-scaling guidance. |
