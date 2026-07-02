---
source: https://app.notion.com/38e0f8b3f59181849d1df39c18a7213b
area: Architecture
status: Current
migrated: 2026-07-02
---

# Architecture

World Model Harness turns a frontier LLM into the *environment* your agent steps against,
reconstructed from your own OpenTelemetry traces. This doc is the map: how the packages fit, the
data that flows between them, and where to plug in new pieces.

## The shape

Packages are layered; arrows point "depends on". `core` depends on nothing, the CLI sits on top.

```
                 cli  ──────────────┐
                  │                 │
   ┌──────────────┼─────────────┐   │
serving         engine          │   │
   │           │  │  │  │        │   │
   └───────────┘  │  │  └────────┴───┤
             optimize retrieval      │
                  │     │            │
               ingest   │            │
                  └──────┴──────┬─────┘
                         core ◄──┘  providers  config  tracking
```

- **`core`** — the vocabulary. `types.py` (`Trace`/`Step`/`Action`/`Observation`/`EnvState`/
  `Session`), `render.py` (the one canonical (state, action)→text rendering, shared by retrieval,
  GEPA, and the engine so an embedded step and a shown demo describe a step *identically*), and
  `parsing.py` (robust JSON / observation-contract parsing). Depends on nothing.
- **`providers`** — one `Provider` protocol (`complete`/`embed`/`verify`), five backends
  (Anthropic, Bedrock, Azure OpenAI, OpenAI, OpenAI Responses) behind `get_provider`. `Embedder` is the narrower
  embed-only capability retrieval needs.
- **`config`** — `HarnessConfig` (persisted to `config.toml`), `ArtifactPaths` (the on-disk layout
  of one model), and `WorldModelStore` (named models). The store reads one root's `models/` dir — the default is the project-local `.wmh/` (where `wmh build` writes); callers can pass another root such as `examples/<task>` to read the prebuilt example artifacts committed under `examples/<task>/models/` (e.g. `tau-bench`, `tau-telecom`).
- **`ingest`** — `TraceAdapter` protocol + a registry; the OTel GenAI adapter normalizes spans into
  `Trace`s. New trace sources register here.
- **`retrieval`** — the DreamGym replay buffer. `EmbeddingRetriever` (cosine top-k over phi),
  pluggable `Embedder`s (`embedders.py`, incl. the offline `HashingEmbedder`), and `leakfree.py`
  (`DemoRetriever` — train-only, never-own-trace retrieval shared by GEPA and eval).
- **`optimize`** — `LLMJudge` (the fitness signal) and `GEPAOptimizer` (drives the `gepa` package
  to evolve the env prompt). Provider-only, so it never imports `engine` (avoids a cycle).
- **`engine`** — the heart. `WorldModel` (`step`: retrieve → assemble prompt → predict → parse →
  advance → enrich), `build` (the ingest→split→index→GEPA→persist pipeline), `loader` (the one
  artifact-dir→live-model path), and the operator flows `replay`/`eval`, `demo`, `play`.
- **`serving`** — a thin FastAPI transport over in-process `WorldModel`s, namespaced by model name.
- **`tracking`** — `MeteredProvider` wraps any `Provider` at the boundary to record time/tokens/cost
  per phase (build / GEPA / judge / serve) onto a `RunTracker`. Transparent: nothing it wraps knows.
- **`research`** — the optimization-research surface (`docs/gepa_research.md`). An `Ablation`
  framework (sweep named `Condition`s across seeds → mean+std) over reusable build/eval primitives
  (`optimize_prompt` / `score_prompt`) that wrap the real pipeline (`score_prompt` delegates to
  `engine.replay`, so the `wmh eval` rubric scores experiments too). The first experiment is GEPA
  seed-stability; live runners live in `scripts/`. Parked directions: `docs/research_directions.md`.
- **`cli`** — `build / list / eval / serve / demo / play / providers / examples / config`. `eval` also runs named example-local eval suites (`wmh eval list | run <suite> | results`); `examples` lists/launches the self-contained task examples. Each command is thin: it parses flags and delegates to an `engine` function.

## The two lifecycles

**Build** (`wmh build` → `engine.build.build`):
`ingest` traces → `split_traces` (deterministic train/held-out) → `EmbeddingRetriever.index` →
`GEPAOptimizer.optimize` (replays held-out steps with leak-free RAG, scores with `LLMJudge`,
reflects to mutate the prompt) → persist prompt + frontier + index + metrics under
`.wmh/models/<name>/`.

**Serve / step** (`wmh serve` | `play` | `demo` → `loader.load_world_model` → `WorldModel.step`):
retrieve top-k similar past steps (phi) → assemble the env prompt (`core.render.build_env_prompt`,
the *same* assembly GEPA optimized against) → `provider.complete` → `parse_observation` → advance
the session (history + scratchpad) → enrich the buffer.

## Where to extend

- **A new LLM backend** → implement the `Provider` protocol in `wmh/providers/`, register it in
  `providers/registry.py`, add its env vars to `config.PROVIDER_ENV_VARS`.
- **A new embedding model (phi)** → implement `Embedder` (`embed(texts) -> list[list[float]]`) in
  `wmh/retrieval/embedders.py`; wire it into `get_embedder`. Set `embed_provider` / `embed_dim` in
  config; the persisted dim must match at load (guarded with a clear error).
- **A new trace source / format** → implement the `TraceAdapter` protocol in `wmh/ingest/`, register
  it; select it via `config.trace_adapter`.
- **A different fitness signal** → implement the `Judge` protocol in `wmh/optimize/judge.py`.
- **A new CLI command** → add a thin command in `wmh/cli/app.py` that delegates to an `engine`
  function; keep the logic in `engine` so it stays testable without the CLI.

## Conventions

See [AGENTS.md](http://AGENTS.md): inline `*_test.py` tests, `ruff` + `ty` clean over the whole project
before commit, no `Any`/bare `dict` (use pydantic + `JsonValue`), deep package structure with a
small CLI surface.
