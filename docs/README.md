# Documentation

Customer documentation describes only supported current services. Design notes, raw experiment
results, and plans do not live here.

## Documentation index

| File | Purpose |
|---|---|
| `usage.md` | Locked CLI map for build, bounded optimize model, optimize router, run, and config. |
| `reference/providers.md` | Catalog providers, first-build `--provider` flags, environment variables, Azure endpoint and deployment rules, the Bedrock credential chain, and OpenAI-compatible listing metadata plus identity-only operator declaration. |
| `reference/gateway-architecture.md` | Operational local gateway contracts, certified exact-model routing, ownership boundaries, and compatibility locks. |
| `reference/gateway-latency.md` | Routine CI same-host mock comparison of Experiential and pinned LiteLLM: schedule, artifact schema, and status badge. |
| `reference/openai-compatible-recipes.md` | Verified Fireworks, Modal, and Experiential Cloud connection recipes through the openai-compatible provider family. |
| `reference/ingest.md` | Current declared local trace source contract for every supported source. |
| `reference/immutable-real-trace-rag.md` | Immutable real-trace retrieval provenance, leakage, persistence, and historical restoration contract. |
| `reference/router_optimization_config.md` | Exact completed-evidence configuration recipe for router optimization. |
| `release-scope.md` | Supported and explicitly excluded release claims. |
| `research/w16-router-sandbox-evidence.md` | Deterministic W16 router and local sandbox evidence, limits, and replay commands. |
| `research/rust-gateway-engine.md` | Native (Rust) gateway data-plane benchmarks: streaming concurrency, the shared SQLite durability ceiling, and horizontal-scaling guidance. |
