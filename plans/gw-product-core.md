# gw/product-core: models data model, gateway server, providers, waterfalls, Models page, Overview page

## 1. Summary

This workstream ships the heart of the gateway product for the 2026-08-20 launch. One OpenAI-compatible inference endpoint at `api.experientiallabs.ai/v1` (Chat Completions and Responses, streaming, full parameter passthrough, inline PDFs/images) that fronts every model: hosted providers (openai, anthropic, gemini, azure_openai, openrouter, bedrock), bring-your-own-key, our legacy owned models, and local/custom models. Behind it: three new Postgres tables (`models`, `model_providers`, `model_waterfalls`) in the platform Supabase database as the single source of truth for the catalog, a normalizer that turns those tables into the gateway's frozen catalog snapshots, per-model provider waterfalls with org overrides that spill only on capacity errors, cache-aware sticky routing, and exactly one usage event per request as the only interface billing/telemetry/analytics consume. On top: a public management API at `control.experientiallabs.ai` plus WMO CLI parity so an AI agent can self-serve the whole core loop (get key, check credits, list models, call models, read usage), and two first-class frontend surfaces in the platform app: the public `/models` catalog (list, detail with waterfall editing and local-variant add, compare, custom-model add, playground handoff) and the logged-in `Overview` page (personal and workspace usage views with a spend/tokens/requests toggle and a contribution graph). Everything callable gets a dedicated sub-agent integration pass and a permanent unit test.

## 2. Decisions: Silen's answers, word for word

Q: Where does the gateway URL live, and how does traffic reach WMO? (asked twice; second answer is final)

> "Don't we already have a couple of different routes? We have API. and then we have platform. I think we have as well control for platform APIs. I'm pretty sure that's how it's split: platform for the actual platform, control for control API, staging or API for the actual API calls. But I'm open to doing the simplest thing here. I'm not exactly sure how these things work together right now."

> "Yeah the gateway-based URL should be Control. API is just for them using their APIs, not for anything else. That's the separation. Platform is for the UI. Control is for all our platform APIs and then API is where all the requests are being sent through."

Reading (confirmed in the final recap): `platform.experientiallabs.ai` is the UI, `control.experientiallabs.ai` is the management API, `api.experientiallabs.ai` is inference only (`base_url="https://api.experientiallabs.ai/v1"`). All three subdomains already resolve to the shared ingress; `gateway.experientiallabs.ai` does not exist and is not needed.

Q: Do I own the three catalog tables, the gateway server, provider adapters, waterfall execution, the usage-event write, the management API + CLI, and the Models + Overview pages, with the platform-gateway-integration chat owning the persistence callback implementations (key validation, credits, usage projection)?

> "For Q2 yes you own those tables here."

Q: What does "your daily credits" mean on the Overview page?

> "The daily credits are how much you've spent on that day or how much you've done X on that day. Whether you're on the token spend or request tab, it shows you how much and how much versus the previous period. There's a graph that shows the per-day of that. Next to the right of the graph there's the top models: [by] token spend or request depending on which one you clicked. Below that you have: activity, longest streak, average a day, average a week, total. There is no free daily allowance of credits. Don't worry about anything YC offer-related here. Everyone gets $20 in free credits as we had before."

Q: Is Overview the personal view and /telemetry the org view, and what counts as "your" activity?

> "Overview is the personal view for a logged-in account, like your API keys. Realistically it should depend on the account type. For example if it's an admin it should show the org view of the overall workspace, which is how many users you have that spend request tokens. The overview should be a secondary tab, like your usage. Maybe what do you think is best here actually because we're trying to solve enterprise users and that's mainly the gateway and then obviously we have this kind of usage thing as well. How should this look?"

Follow-up answer to my proposed design (Personal | Workspace scope switcher, one metric toggle and period selector re-rendering everything):

> "I don't know maybe we should have our different workspaces or our different aliases as the first thing you see, maybe like cross-org spend if it's an admin account. Yeah so maybe you should land in org first instead of overview and then if you're actually on the account, you land on the overview. Otherwise sounds good."

Q: House provider keys for the platform-funded lane: rows in `provider_connections` under a house org (option A) or env vars (option B)?

> "A. For Q5 is A."

Q: Usage event interface: an append-only store written once per request (model, provider, lane, tokens in/out, cost, latency, org, key, timestamp, attempt trail) plus a management-API read endpoint, as the ONLY thing downstream chats consume?

> "For Q6 sounds good."

Q: Cache-aware routing signal: honor OpenAI's `user` field when present, fall back to sticky (API key, model) with a short time window (option B)?

> "Yeah do B. We're not actually routing anywhere right now. The only place we're doing any sort of thing like that is just the waterfall. If it's not available, waterfall either across models or for API keys."

Q: Where are live-test provider credentials?

> "We have working keys in our [env], probably in our platform, in that main repo, or in our world model optimizer repo."

(Verified: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `AZURE_OPENAI_API_KEY` plus several Azure Foundry keys, AWS credentials for Bedrock, `FIREWORKS_API_KEY`, `MODAL_TOKEN_ID/SECRET`, `TINKER_API_KEY` exist in the local `.env` files of the platform and world-model-harness repos. No Gemini key found locally.)

Q: Build the platform UI on the PR #441 branch (`agent/convergence-pr13-consolidation`), treating its red preview as the coordination chat's blocker?

> "Sounds good for Q9."

Q: The 12 `agent/gateway-*` WMO branches: build against their storage interfaces (from `agent/gateway-replay-message`) without absorbing or editing those branches?

> "1. Yes build against those interfaces."

Q: PDFs/files day one: inline base64 on both endpoints, `/v1/files` deferred?

> "2. You should spawn a sub-agent to do that and iterate on it and make sure it works but not a P0 even if it doesn't land."

Q: Who makes the Playground call the gateway?

> "3. For Q5 it should be the model catalog that actually makes it work there. I don't think maybe this is the correct chat for it. Yeah it should be this chat. You're right."

Q: Final confirmation of the full shared understanding (recap message covering hosts, backend, endpoints, frontend, providers, and my implementer calls).

> "Yep create the draft PR."

### Inherited decisions relied on (transfer prompt and addenda)

Scope and deadline (inherited):

> "Now we can do it all. Stop thinking we can't get it done."

Code quality and backwards compatibility (inherited):

> "We don't necessarily need to make things backwards compatible. We just want to hide the old things. What is the cleanest interface is that if a future employer was reading this code, they would be pleasantly surprised at how readable and composable the code is."

Design (inherited):

> "I think something along the lines of vercel, linear, notion, apple-like tasteful designs that are minimalistic but also include all the relevant information and have some amount of style to them as well. We should use our main brand colors and make it look somewhat like the website, with lines, and keep it simple but keep it on brand."

> "We also want to improve the UI to look better than it is right now... it needs to be cleaner and kind of more sexy than it currently is while still being super easy to digest."

Auth and what is public (inherited, complete answer):

> "Everything should be public. Literally everything should not be a single page. It's not public and there should be a login that's always in the sidebar. The only reason we ever need to log in is if someone needs to get an API key and check usage. If they're not logged in they should still be able to see it but if they want to do anything that requires login they will have to log in. I think we should create a modal version of our login as well so they don't have to get redirected to another page. In cases where they're on a certain page we don't want to redirect them. We want them to open the modal and be able to use it because now we don't need this more complicated onboarding. Now we can just kind of go directly straight to the Here's your API key and your credits. Again only the settings should be hidden if you're not logged in and even then it should be like API keys. If you click on API keys, they have to log in, something like that."

> "We also want to make sure that we can access pretty much everything without authentication. The only thing that needs authentication to access is actually the ability to use credits and use API keys or create API keys."

Identities deferred, org keys stay (inherited):

> "Yeah just assume we have our organization API keys. All the other kind of specific stuff like identities, etc., that's gonna be handled later so don't focus on that right now. The waterfall is per model correct"

Backend and agent self-serve (inherited):

> "In WMO, that's where the backend is going to be for this in terms of aliases and spend and things like that. We want people to be able to use this through an [agentic CLI] so an agent could entirely do what you're able to do through the frontend with the backend. That should be through the WMO repo... Kind of think through how we want to make sure that an agent could use the API gateway the same way that a human would. And now we could have agent users. We should just think through that."

> "an agent should be able to self-serve and it should be relatively easy to do this via the core loop API."

The endpoint (inherited):

> "What we still want to do is make sure that we have a single API endpoint that this works through. Keep the way our API keys work right now but it should still be a fully open API-compatible response endpoint with streaming. We should probably have a very clean interface to connect all models to this one kind of interface with streaming and with the ability to upload PDFs, like temperature, everything else."

> "We need to be both respones and chat completions compatible via openai" [Responses API and Chat Completions, both.]

Resolution and lanes (inherited):

> "I think request resolution is the correct way to go. Including maybe even pass-through, most will be API keys that they bring and others will be specifically things that they use through our platform."

The models page (inherited):

> "We are upgrading the models page in the settings to be first-class... we want a new models page which includes all of the different models from all of the different providers. We want to include things that OpenRouter includes so we want to include everything from uptime to costs to literally everything. Just do your research in terms of other AI gateways such as OpenRouter and all the other ones out there, like Fireworks provides per model, and allow people to click into it."

The model detail page (inherited):

> "We just have a different UI now that's integrated and it's more table-based and when you click into a model through the unified interface it should show all the things like: when a model is released, the context [window], the input of price per million, you're able to go to the playground, you're able to click compare and compare to other models. It should show the different modalities and different things like: providers, pricing, performance, uptime. Actually don't show activity for now but we still want to track it because we will have activity on the platform. Obviously we still want to track it."

Sorting (inherited):

> "In terms of how we can sort models, we should be able to do things like: sort by input modalities, if there are any discounts, the context length, the pricing, any specialized model categories, supported parameters like tools, temperatures, reasoning, [TPS throughput], anything else like model age, provider, so like being able to do [Claude by Anthropic or by Bedrock] or by something else."

The models database (inherited):

> "We need a new models database that is not this old one. We need to think about how we are consolidating the old models with the new models. The old models are things like ones that we owned but again they were still OpenAI compatible. The way we use them, we want to probably hook up these ones very similarly in terms of telemetry so we can probably reuse the same models table but some of the columns we won't use for this."

Local and custom models (inherited):

> "Also remember we still want to be able to use open source, closed source, and local models. For every single one of the models on here there should be the ability to add a local model when we click into it. We should have the ability to add our own custom model that you just serve through the same endpoint that also kind of tracks telemetry. That is another separate page where when you add a local model you are adding another row in the models database."

The per-model waterfall (inherited):

> "Maybe there are multiple ways we can access the model and there's a fallback chain. For example you can have: Claude Opus 5, your primary provider as [Anthropic], [a Bedrock key], AWS. That's how you have the model waterfall per actual model."

Per-model integration testing (inherited):

> "Per model we should probably spawn a separate agent to go out and figure this stuff out, test it, and make sure it works. We should have a unit test per model every time we integrate it to make sure everything works as it should for that model."

Providers and the model roster (inherited):

> "Yeah we need the ones that you mentioned: OpenAI, Anthropic, Gemini, Azure, [OpenRouter], Bedrock, Local. For now if we can, we can add Fireworks, which would be nice to add, and Modal, which would be nice to add. And yes these ones work. Maybe include: [Kimi] K2.6, GLM 5.3. Include the GPT-5.6 line, GPT-5.5, Gemini Flash 3.7, [DeepSeek V4 Flash], and V4 Pro. And then the older Gemini models as well. We just want as much coverage as possible. Whatever chat is working on should spawn a ton of sub-agents just to fill out the directory and have some preferred models, which are the ones that are always at the top. One that's missing is the Qwen series, like Qwen 4, Qwen 9.3.8, 27B. [The frontier ones] should be the ones that are kind of highlighted. We should have a lot of other ones as well."

> "Yes agreed [Fireworks and Modal are] secondary."

The Overview page spec (inherited):

> "The main page should show your name and email and then the usage summary should be the first thing that you see. You should see: your daily credits, your own key, your top models by spend, your activity, a GitHub contribution graph with longest streak, average per day, average per week, and total, API keys. Usage summary should allow you to sort by what usage period: today, last seven days, last 30 days, last year, all time. You should be able to sort by tokens. All of these things change and then it becomes top models by tokens. In the activity section with the GitHub graph it becomes: average a day in tokens, average a week in tokens, total in tokens. Same for spend and requests as well. That should be kind of the first thing that you land on when you open the account. Now this is for an individual user."

WMO work-in-progress architecture (inherited clarification from Silen via the coordinator):

> "state is separated from persistence in the open-source repo, so in hosted mode the platform database is the gateway's actual persistence via callback interfaces - not a mapping of it. For your Stage 1 that means your models / model_providers / model_waterfalls tables must satisfy two masters at once: the catalog UI's read patterns AND the gateway's alias/provider-revision callback contract. Read the contract object shapes in WMO before you freeze columns, and treat any field the contract needs that your schema lacks as a schema requirement, not an integration-chat problem."

Smart routing (inherited):

> "Essentially what we want is for people on our end to be able to manage the different providers for each one and have everything be perfectly OpenAI responses and compatible, which I think we have most of this stuff already built out. We have to make sure that it is built in the kind of correct manner. All of those other things that we thought of, we should also integrate all these different smart things." [The smart things: cache awareness, sequential calls in a session pinned to the same provider for KV-cache hits, and the savings/model-suggestions surfaces.]

## 3. Scope

### IN (this plan ships)

- The `models`, `model_providers`, `model_waterfalls` tables in platform Supabase Postgres, plus seed and consolidation of legacy owned models and house provider connections.
- The catalog normalizer in WMO: tables to `NormalizedGatewayCatalog` snapshots for the gateway hot path.
- The gateway inference server on `api.experientiallabs.ai/v1`: Chat Completions + Responses, streaming, full param passthrough, inline PDFs/images, OpenAI error shapes, agent-self-correcting error texts.
- Provider execution for openai, anthropic, gemini, azure_openai, openrouter, bedrock, local (self-hosted OpenAI-compatible); Fireworks and Modal secondary (same day, never blocking).
- Waterfall execution semantics (capacity-error-only spill, attempt trail, cost) absorbed from llm-waterfall and the gateway branches.
- Cache-aware sticky routing (`user` field, else sticky by key+model).
- One usage event per request (request row plus attempt trail) and the management read API over it.
- The management API on `control.experientiallabs.ai` (catalog reads public; model/waterfall/local/custom writes authed) and WMO CLI parity for the core loop.
- The WMO gateway deployment (Porter app, remote bind, ingress routing).
- Platform frontend: `/models` catalog, model detail (waterfall UI, add-local-variant, keys mount point, playground button), compare flow, add-custom-model page, playground rewire to the gateway, and the `Overview` page (personal + workspace).
- Catalog data fill (sub-agent fan-out), per-model live integration tests with permanent unit tests, and a listing-correctness verification pass.

### OUT (not this chat's)

- Key/BYOK UI anywhere (keys-byok chat; this plan only leaves mount points).
- Credits page, Stripe, $20 free-credit grant mechanics (billing chat; Overview mounts billing's balance component).
- `/telemetry` page and all usage display surfaces there (telemetry chat; consumes my usage events only).
- Sidebar, layout, login modal, public/auth gating shell, hiding old surfaces, the `/models` route stub handoff (shell chat).
- Postgres implementations of the gateway storage interfaces (`GatewayControlStore`, `AttemptLedger`, `SecretResolver`), xpl_ key validation, credits enforcement, and the per-user usage projection/rollups (platform-gateway-integration chat; I build against those interfaces and consume the projection).
- Docs site and llms.txt (docs chat; I hand over the endpoint inventory via Silen).
- `/yc` onboarding and the website repo (website chat).

### DEFERRED (v2+)

- Identities and aliases as user-facing concepts: "All the other kind of specific stuff like identities, etc., that's gonna be handled later so don't focus on that right now."
- Activity shown in the catalog UI: "Actually don't show activity for now but we still want to track it."
- A `/v1/files` upload-and-reference endpoint (inline base64 only day one; the inline path itself is "not a P0 even if it doesn't land").
- Prompt-prefix-hash session detection for cache routing (launching with `user` field + sticky key/model).

## 4. Work packets

Conventions used below: "WMO" = repo `experientiallabs/world-model-optimizer` (local clone `world-model-harness`), branch `gw/product-core`. "Platform" = repo `experientiallabs/platform`, branch `gw/product-core-ui`, based on `agent/convergence-pr13-consolidation` (PR #441, treat as main). WMO work builds against the storage interfaces on `origin/agent/gateway-replay-message` (`wmo/runtime/gateway/interfaces.py`, `contracts.py`; SCHEMA_VERSION 7) without editing those branches. WMO repo rules: no em dashes in any new writing, colocated `*_test.py`, 999-line file cap, whole-repo gate `uv run ruff check . && ruff format --check . && ty check && pytest -q`. Platform repo rules: justify every addition, greenfield no-fallbacks, pgTAP for migrations, prove changes on the PR preview sandbox.

---

### core-P1: the schema (MERGES FIRST)

- **Goal**: land `models`, `model_providers`, `model_waterfalls` in platform Supabase so every other chat can build tonight.
- **Repo + files**: Platform. New migration `supabase/migrations/<ts>_gateway_models_catalog.sql`; pgTAP tests in `supabase/tests/`. Reference existing style: `20260729120000_provider_connections.sql`, `20260819100000_project_provider_widening.sql`.
- **Spec**:
  - `models` (one row per model concept):
    - `id uuid pk default gen_random_uuid()`
    - `slug text not null unique` (URL-safe, e.g. `claude-opus-5`; for org-owned models unique per org: `unique (coalesce(owning_org_id, '00000000-...'::uuid), slug)`)
    - `display_name text not null`
    - `description text`
    - `release_date date`
    - `context_window integer` (tokens)
    - `max_output_tokens integer`
    - `input_modalities text[] not null default '{text}'` (values: text, image, audio, video, pdf)
    - `output_modalities text[] not null default '{text}'`
    - `supported_params jsonb not null default '{}'` (boolean map: tools, temperature, reasoning, top_p, response_format, structured_outputs, stop, seed, logprobs, ...)
    - `category text` (free-form specialized category: coding, reasoning, vision, embedding, ...)
    - `tags text[] not null default '{}'`
    - `owning_org_id uuid references organizations` (null = public catalog; set = custom/local model)
    - `preferred_rank integer` (non-null = pinned at top of catalog, ascending)
    - `status text not null default 'active' check (status in ('active','hidden'))`
    - `created_at/updated_at timestamptz not null default now()`
  - `model_providers` (one row per way to reach a model; this is what the normalizer turns into an `ExactModelDeployment`):
    - `id uuid pk`
    - `model_id uuid not null references models on delete cascade`
    - `provider text not null check (provider in ('openai','anthropic','gemini','azure_openai','openrouter','bedrock','local','fireworks','modal'))`
    - `provider_model_id text not null` (the wire id at that provider)
    - `base_url text` (required when provider = local; the self-hosted OpenAI-compatible endpoint)
    - `region text`, `api_version text` (bedrock/azure)
    - `owning_org_id uuid references organizations` (null = available to all; set = an org's private deployment, e.g. a local variant)
    - `provider_connection_id uuid references provider_connections` (pin a deployment to a specific BYOK connection; null = resolve by org + provider at request time)
    - `billing_source text not null default 'customer_managed' check (billing_source in ('customer_managed','host_managed'))`
    - Prices, integer micro-USD per million tokens to match the gateway contract exactly (`GatewayTokenPrices`): `input_micro_usd_per_million bigint`, `cached_input_micro_usd_per_million bigint`, `output_micro_usd_per_million bigint`, `reasoning_micro_usd_per_million bigint` (null = unknown, never zero-fill)
    - `pricing_source text`, `pricing_effective_at timestamptz`
    - `capabilities jsonb not null default '{}'` (mirrors WMO `ModelCapabilities` / `GatewayDeploymentCapabilities` booleans, including `reports_cached_input_tokens`, `reports_reasoning_tokens`)
    - Stats for the catalog UI: `uptime_30d numeric`, `throughput_tps numeric`, `latency_p50_ms numeric`, `stats_source text check (stats_source in ('openrouter','observed'))`
    - `status text not null default 'active' check (status in ('active','degraded','disabled'))`
    - `created_at/updated_at`
    - `unique (model_id, provider, provider_model_id, coalesce(owning_org_id, zero-uuid), coalesce(base_url,''))`
  - `model_waterfalls` (ordered fallback chain per model; row per rung):
    - `id uuid pk`
    - `model_id uuid not null references models on delete cascade`
    - `org_id uuid references organizations` (null = the default chain; set = that org's override, which fully replaces the default for that org)
    - `position integer not null check (position >= 0)` (0 = primary; ordering IS the waterfall order and the normalizer must preserve it into `ExactModelPool.deployment_ids`)
    - `model_provider_id uuid not null references model_providers`
    - `created_at/updated_at`
    - `unique (model_id, coalesce(org_id, zero-uuid), position)`; `unique (model_id, coalesce(org_id, zero-uuid), model_provider_id)`
  - RLS enabled on all three with grants revoked from `authenticated` (same pattern as `endpoints`): all reads/writes go through the service role behind the management API. Public catalog browsing happens via the API, not direct table access.
  - Indexes: `models(slug)`, `models(preferred_rank) where preferred_rank is not null`, `model_providers(model_id)`, `model_waterfalls(model_id, org_id, position)`.
  - The old `endpoints`/`default_models` tables are untouched (hidden legacy views only; nothing new reads them).
  - pgTAP: table shapes, checks, uniques, RLS state, and a waterfall-ordering round-trip.
- **Depends on**: none. This is the first merge of the whole launch; keep the PR to the migration + tests only.
- **Done when**: `supabase db push` applies cleanly on a fresh local stack, `./run-pgtap.sh` passes, and the three tables match this spec exactly.

### core-P2: seeds and legacy consolidation

- **Goal**: populate the house org, house provider connections, preferred models, and fold legacy owned models in as ordinary rows.
- **Repo + files**: Platform. Migration or seed script alongside `supabase/seed.sql` / `scripts/seed_supabase_local.sh`; a small idempotent backfill script for production run.
- **Spec**:
  - Create a house organization (slug `experiential-labs-house`) owning the platform-funded lane. Insert `provider_connections` rows for it, one per provider we hold keys for (openai, anthropic, azure_openai, openrouter, bedrock; gemini once the key exists), secrets loaded into Supabase Vault via the existing `upsert_provider_connection()` RPC path, values sourced from the deployment env (manual step for Silen where absent).
  - Seed `models` rows for the pinned preferred list with `preferred_rank` 1..N in this order: Kimi K2.6, GLM 5.3, the GPT-5.6 line, GPT-5.5, Gemini Flash 3.7, DeepSeek V4 Flash, DeepSeek V4 Pro, older Gemini models, Qwen 4, Qwen 9.3.8, Qwen 27B.
  - Legacy owned models: for each live legacy serving project (the `endpoints`-backed surface), insert one `models` row (owning_org_id = its org, category `owned`) and one `model_providers` row with provider `local`, `base_url` = the private serving backend URL, `billing_source` `host_managed`. Same telemetry path as everything else. No compatibility shim; old surfaces are hidden, not maintained.
  - Do not seed the long-tail catalog here; that is core-P17's job through the management API.
- **Depends on**: core-P1.
- **Done when**: local seed produces the house org + connections + preferred rows; the backfill script is idempotent (safe to run twice); pgTAP or a pytest asserts the legacy rows resolve through the new tables.

### core-P3: catalog normalizer (tables to gateway snapshots)

- **Goal**: derive the gateway's frozen `NormalizedGatewayCatalog` snapshots from the three tables so the hot path never reads the tables directly.
- **Repo + files**: WMO. New module under `wmo/runtime/gateway/` (e.g. `catalog_source.py`) with colocated test. Read `origin/agent/gateway-replay-message:wmo/common/models/gateway_catalog.py` (`ExactModelDeployment`, `ExactModelPool`, `NormalizedGatewayCatalog`, `normalize_gateway_catalog`) and `wmo/runtime/gateway/catalog_authority.py` for the snapshot write/load protocol first.
- **Spec**:
  - Map each `model_providers` row to an `ExactModelDeployment`: `deployment_id` = row id, `source_alias` = model slug + provider suffix, `provider`/`provider_model` direct, `billing_source` direct, `gateway.prices` from the four micro-USD columns, `capabilities` from the jsonb, `connection_sha256` from the resolved connection config, `exact_model_id` derived per the WIP's singleton derivation unless authored.
  - Map each model's waterfall (org override if present for the requesting org, else default) to an `ExactModelPool` whose `deployment_ids` order is exactly the `position` order. Pools with more than one deployment require an equivalence certification per the contract validator; synthesize an operator-asserted certification (coordinate the exact shape with the integration chat; see Open items).
  - Snapshots are content-addressed by `identity_sha256()`; they must be storable where the hosted gateway can load them at boot (the WIP uses files under `gateway/catalog-snapshots/`; hosted mode likely wants a `catalog_snapshots` blob table owned by the integration chat's store. Define the interface, let their store implement it.)
  - Regeneration triggers: any management-API write to the three tables produces a new snapshot and repoints the affected alias revision (through the integration chat's `GatewayControlStore` implementation).
  - Model slug = the public alias name customers put in the `model` field.
- **Depends on**: core-P1; interface coordination with platform-gateway-integration (their Postgres `GatewayControlStore`).
- **Done when**: a unit test builds the three tables' rows in memory, normalizes, and round-trips `identity_sha256()`; ordering of a 3-rung waterfall is preserved; org override replaces the default pool for that org only. Whole-repo gate passes.

### core-P4: provider execution with streaming (the seven)

- **Goal**: every provider reachable through one clean adapter interface with streaming: openai, anthropic, gemini, azure_openai, openrouter, bedrock, local.
- **Repo + files**: WMO. Build on `wmo/runtime/models/providers/` (ABC `ProviderHttpClient` in `base.py`; existing adapters: anthropic, azure, bedrock, gemini, openai, openai_compatible incl. OpenRouter, tinker_sampling) and the execution layer shapes on `origin/agent/gateway-replay-message` (`wmo/runtime/gateway/execution.py`, `ProviderStream`, `GatewayEvent`). Provider execution code belongs only under `wmo/runtime/models/providers/` per AGENTS.md.
- **Spec**:
  - One interface every provider implements, returning a stream of `GatewayEvent`s (text deltas, tool-call deltas, terminal event with `GatewayUsage`). Non-streaming callers consume the same stream buffered. This is the interface a future reader should admire: small, typed, no per-provider special cases leaking upward.
  - `local` = the openai_compatible adapter pointed at `base_url`; not a separate code path.
  - Map every provider failure into `GatewayFailure` with the 15-class `GatewayFailureClass` enum; classification rules come from core-P6.
  - Full parameter passthrough: temperature, tools, tool_choice, top_p, response_format/structured outputs, stop, seed, reasoning params, max tokens. Providers that do not support a param return the OpenAI `unsupported_parameter` error shape, never silently drop it.
  - Usage extraction per provider including cached input tokens and reasoning tokens where reported (`usage_source` = observed vs unknown, per the ledger contract).
- **Depends on**: none to start (interface + adapters are testable with stubs); core-P3 for end-to-end.
- **Done when**: per-adapter unit tests with stubbed transports cover streaming, tool calls, usage extraction, and failure mapping for all seven; gate passes.

### core-P5: OpenAI surface: /v1/chat/completions + /v1/responses, one translation layer

- **Goal**: perfect OpenAI compatibility on both endpoints for every provider; a client swaps base_url and nothing else.
- **Repo + files**: WMO. Build on `origin/agent/gateway-replay-message` protocol layer (`wmo/runtime/openai_protocol/`, `wmo/runtime/gateway/service.py::create_gateway_app`) and the official OpenAI SDK types (the repo already uses them; keep it that way).
- **Spec**:
  - One internal request/response shape; Responses and Chat Completions both translate to/from it in one shared layer, not per provider. `surface` (`chat_completions` | `responses`) flows into `AuthorizationSnapshot` as the contract requires.
  - Streaming: SSE for both surfaces with correct OpenAI event framing; Responses continuations via `previous_response_id`; idempotency via the standard `Idempotency-Key` header (the ledger's caller-operation idempotency already specifies conflict/replay semantics; follow them).
  - Inline files: base64/data-URL PDFs and images accepted on both surfaces and translated per provider (Anthropic document blocks, Gemini inline data, OpenAI input_file/input_image). Per Silen this gets its own iterating sub-agent and is explicitly not P0: ship text+tools first, keep file support behind the same interface so it lands without churn.
  - Error texts must let an AI agent self-correct from the text alone. Standard: every error says (1) what failed, (2) why, (3) the exact next call to make, e.g. `model 'gpt-5.5' exists but your org has no openai key and no credits; POST control.experientiallabs.ai/api/orgs/{org}/providers with your key, or add credits at platform.experientiallabs.ai/settings`. OpenAI error envelope shapes throughout (`invalid_request_error`, `authentication_error`, etc.), with our informative `message` inside.
  - `GET /v1/models` on api. lists callable models for the presented key (public models plus the org's own), OpenAI list shape.
- **Depends on**: core-P4 (adapters), integration chat's control store for auth (stub it via the Protocol until theirs lands).
- **Done when**: contract tests run the official `openai` Python SDK against a local gateway (stub providers) for: non-stream + stream chat, non-stream + stream responses, tools, unsupported param error, bad key error text, model-not-found error text. Gate passes.

### core-P6: waterfall semantics (absorb llm-waterfall)

- **Goal**: spill only on capacity errors, never on client errors; full attempt trail with cost; per the prior art we own.
- **Repo + files**: WMO. Port the classification and semantics into the gateway failure model; sources: `/Users/silen/Desktop/Projects/llm-waterfall/llm_waterfall/classify.py` (canonical) plus the improved vendored fork at `world-model-harness/packages/llm-waterfall` (v0.1.4: adds `InternalFailure` botocore code, MRO-based botocore/httpx family matching, tinker support). Target: the `GatewayFailure.failover_eligible` / `retryable_same_deployment` assignment inside the provider adapters and `execution.py`'s advance rules on replay-message.
- **Spec**:
  - Capacity (spill): throttling/quota codes (botocore `ThrottlingException`, `TooManyRequestsException`, `ServiceUnavailableException`, `ServiceQuotaExceededException`, `ModelNotReadyException`, `ModelTimeoutException`, `InternalServerException`, `InternalFailure`), SDK classes (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`, `OverloadedError`, httpx transport errors), HTTP 408/429/500/502/503/504/529, conservative substrings last. Structured code always beats message matching. Everything else = client error, fail immediately without trying the next rung (failing over on a bad request masks real bugs).
  - Disable SDK-internal retries everywhere (botocore `total_max_attempts=1`, others `max_retries=0`); retry policy lives in exactly one place (the execution layer's same-deployment retry + advance rules).
  - The attempt trail is never a log: it is the `gateway_attempts` rows plus an optional response header/extension summarizing (provider, model, outcome, latency) per attempt for API callers.
  - The old `.wmo/fallback.toml` and `wmo/providers/waterfall.py` are legacy; do not extend them.
- **Depends on**: core-P4.
- **Done when**: a table-driven unit test (port the fork's 123-case style) proves classification for every enumerated error; an execution test proves a 3-rung chain spills on 429, stops on 401, and records one attempt row per dispatch. Gate passes.

### core-P7: cache-aware sticky routing

- **Goal**: sequential calls in a session hit the same deployment for KV-cache hits.
- **Repo + files**: WMO, execution layer (deployment selection ahead of the waterfall advance).
- **Spec**:
  - Session key: OpenAI `user` field when present, else (api key id, model slug). In-process LRU map session-key -> (catalog_sha256, deployment_id) with a short TTL (default 10 minutes, configurable).
  - On hit, start the waterfall at the sticky deployment if it is still in the resolved pool and healthy; otherwise fall through to position 0. Stickiness never overrides the waterfall's failure semantics; a spill updates the sticky entry.
  - State is process-local by design for launch (matches the WIP's process-local health registry); note the multi-replica gap in Open items.
- **Depends on**: core-P4/core-P6.
- **Done when**: unit test: two calls with the same `user` route to the same deployment; a capacity failure moves both the request and the sticky entry to the next rung; TTL expiry resets. Gate passes.

### core-P8: management API on control. (agent self-serve core loop)

- **Goal**: everything a human does in the frontend, an agent does via this API; the frontend is just another client.
- **Repo + files**: WMO. New management router beside the gateway service (`wmo/runtime/gateway/`), mounted on the control. deploy role.
- **Spec**: endpoint inventory (exact paths; docs chat documents these):
  - Public, no auth: `GET /api/models` (catalog list; filter/sort params: modality, category, provider, min context, max price, supported param, sort by price/age/context/throughput), `GET /api/models/{slug}` (detail: model + providers + default waterfall + stats), `GET /api/models/{slug}/providers`.
  - Authed with an org `xpl_` key (validation via the integration chat's `GatewayControlStore`): 
    - `POST /api/models` (create custom model: a new `models` row + at least one `model_providers` row; org-owned)
    - `POST /api/models/{slug}/providers` (add a deployment, incl. a local variant: provider `local` + base_url)
    - `GET/PUT /api/models/{slug}/waterfall` (read default + org override; PUT replaces the org override with an ordered list of model_provider ids)
    - `GET /api/usage` (the usage-event read: filters org [implied by key], key, model, provider, lane/billing_source, time range; returns request rows with attempt trails, paginated)
    - `GET /api/usage/daily` (per-day rollup for graphs: metric = spend|tokens|requests, group by model|key|member, period param; backed by the integration chat's projection)
    - BYOK provider-connection writes (`POST/DELETE /api/providers/{provider}`) exist here as the API the keys chat's UI calls; this chat implements the endpoint plumbing to the existing Vault RPCs, the keys chat owns all UI.
    - Credits read (`GET /api/credits`) proxies the existing credit ledger read so the core loop is complete; billing chat owns everything else about credits.
  - Every write is idempotent (operation receipts pattern from the WIP). Every error follows the core-P5 self-correction standard.
  - Keep a machine-readable endpoint inventory (one JSON/markdown file in the repo) updated in the same commits; it is the docs chat's source of truth, handed over via Silen.
- **Depends on**: core-P1 (tables), integration chat's store for auth + writes; core-P3 for snapshot regeneration on writes.
- **Done when**: an end-to-end pytest walks the full core loop against a local stack with a stubbed store: list models -> add BYOK -> set waterfall -> add local model -> call it (stub provider) -> read usage, all via HTTP with an org key. Gate passes.

### core-P9: WMO CLI parity

- **Goal**: an agent in a terminal gets as far as an agent speaking HTTP.
- **Repo + files**: WMO. `wmo/cli/app.py` (root Typer app, currently locked to build/config/optimize/run by `wmo/cli/app_test.py` and release tests: update the locks deliberately in the same PR), new `wmo/cli/gateway_cmd.py`.
- **Spec**:
  - New root command group `wmo gateway` speaking to `control.experientiallabs.ai` (env `WMO_GATEWAY_CONTROL_URL` overridable, key via `WMO_API_KEY` or `--key`): `models list`, `models show <slug>`, `models add` (custom/local), `waterfall show|set <slug>`, `providers add <provider>` (BYOK), `usage [--daily]`, `credits`, `call <slug> "prompt"` (one-shot chat via api., streaming to stdout), `key check`.
  - Output: human tables via rich Console by default, `--json` for agents everywhere. Errors print the API's self-correction message verbatim.
  - Sign-up/get-key is a web/manual step at launch; `wmo gateway key check` verifies a pasted key and prints the org. No credential values ever stored in repo files.
- **Depends on**: core-P8.
- **Done when**: CLI tests cover each command against a stubbed HTTP layer; the lock tests are updated and pass; `wmo gateway --help` reads cleanly. Gate passes.

### core-P10: deployment of WMO-as-service

- **Goal**: the gateway runs as a real service behind `api.` and `control.`.
- **Repo + files**: WMO (server entry, config, Dockerfile if needed); Porter/ingress config wherever the platform's deploy scripts live (`platform:scripts/porter/deploy_porter_apps.sh` as reference).
- **Spec**:
  - Remove the loopback-only restriction for the gateway app (it currently refuses non-127.0.0.1 binds); bind configurably, TLS terminates at the ingress.
  - One image, two roles by env (matching the platform's existing pattern): `inference` serves `/v1/*` (api. routes here), `management` serves `/api/*` (control. routes here). Health endpoints `/health/live`, `/health/ready` (already in the WIP lifecycle module).
  - Config via env: Postgres/Supabase connection for the integration chat's store, pepper/secret material per their design, snapshot storage location.
  - Ingress: `api.experientiallabs.ai/v1/*` -> gateway-inference; `control.experientiallabs.ai/api/*` -> gateway-management. platform. untouched. Actual DNS/ingress edits and Porter app creation are manual steps for Silen if agent access is insufficient.
- **Depends on**: core-P5, core-P8; integration chat's store to boot against real data.
- **Done when**: the preview/staging deployment answers `GET /v1/models` with a real key end to end, and `curl` with a swapped base_url streams a completion from at least one hosted provider.

### core-P11: /models catalog page

- **Goal**: the storefront: public, table-first, best-looking thing in the app.
- **Repo + files**: Platform, branch `gw/product-core-ui`. Route `apps/web/app/models/page.tsx` (top-level = public, taking over the stub; shell chat owns the sidebar entry). New `apps/web/components/models-catalog/`. A reusable data-table primitive goes in `apps/web/components/ui/` (none exists today; build it there so other chats reuse it).
- **Spec**:
  - Public, no auth to browse (renders fully signed out). Data from `GET control./api/models` (server-fetched, cached, revalidated on interval).
  - Dense table: model name + provider badges, context window, input $/M, output $/M, modalities (compact icons), supported params (chips: tools/temp/reasoning), throughput tok/s, uptime, release date. Row click -> detail.
  - Preferred models pinned at top, visually highlighted (current frontier per the seed order). Below: everything, virtualized if needed.
  - Filter/sort by: input modalities, discounts (e.g. cached-input price < input price, provider promos from OpenRouter data), context length, pricing, category, supported params, throughput, model age, provider, and same-model-different-provider (each provider row of one model filterable, "Claude by Anthropic or by Bedrock").
  - Design per Contract 6: Vercel/Linear/Notion/Apple minimal, existing tokens in `globals.css` (accent `#168a49`, thin lines, `.mono-label` eyebrows), table-first, dense but instantly digestible. Fill the viewport with flexible units. The user-facing noun rules from AGENTS.md apply ("world model" never renders).
  - Track activity per model from day one (usage events already do); show nothing activity-related in the UI.
- **Depends on**: core-P8 (public reads); shell chat's route stub handoff; core-P17 for real data (build against seeds meanwhile).
- **Done when**: page renders signed-out on the PR preview with seeded data; every listed filter/sort works; `pnpm web:test` and `web:lint` pass.

### core-P12: model detail page

- **Goal**: the join view: everything about one model, per provider, plus actions.
- **Repo + files**: Platform. `apps/web/app/models/[modelSlug]/page.tsx` + `components/models-catalog/detail/`.
- **Spec**:
  - Header: name, release date, context window, modalities, category, supported params. Providers table: each `model_providers` row with input/output (and cached/reasoning where present) price per million, uptime, throughput, latency, status, stats source label.
  - Waterfall section: the default fallback chain rendered as an ordered list (e.g. Opus 5: Anthropic -> Bedrock -> other), and for logged-in org admins an editor that reorders/substitutes rungs and saves as the org override via `PUT /api/waterfall`. Signed-out: chain visible, editing prompts the login modal (shell chat's modal, my trigger).
  - "Add a local variant": button opening a small form (base_url, provider_model_id, optional key ref) that POSTs a `local` deployment for this model; auth-gated via modal.
  - "Use via key": mount point only. Leave a clearly named slot component (`components/models-catalog/detail/UseViaKeyMount.tsx`) that the keys-byok chat fills; do not build key UI.
  - "Open in Playground": links `/playground?model=<slug>`.
  - "Compare": links `/models/compare?models=<slug>`.
  - No activity/usage display.
- **Depends on**: core-P11 (shared components), core-P8.
- **Done when**: detail renders for a seeded model signed-out; org override save round-trips on the preview; the keys mount renders an empty labeled slot; web tests pass.

### core-P13: compare flow

- **Goal**: side-by-side model comparison.
- **Repo + files**: Platform. `apps/web/app/models/compare/page.tsx`.
- **Spec**: URL-driven (`?models=a,b,c`, 2 to 4 models), add/remove via a searchable picker. Columns = models; rows = price in/out, context, modalities, params, throughput, uptime, release date, providers. Highlight best value per row subtly. Public. Entry points: detail-page button and multi-select in the catalog table.
- **Depends on**: core-P11/core-P12.
- **Done when**: comparing 2 and 4 seeded models renders correctly signed-out on the preview; deep link works cold.

### core-P14: add-custom-model page

- **Goal**: bring your own model as a first-class row, served through the same endpoint, telemetry identical.
- **Repo + files**: Platform. `apps/web/app/models/new/page.tsx`.
- **Spec**: auth-gated form (modal on visit if signed out): display name, slug (generated, editable), base_url (OpenAI-compatible), provider model id, context window, modalities, supported params, optional pricing (for spend accounting). Submits `POST /api/models`; success lands on the new detail page with a "call it now" snippet (curl + python with the org key placeholder). Custom models appear only to their org in catalog and `GET /v1/models`.
- **Depends on**: core-P8, core-P12.
- **Done when**: end-to-end on the preview: create via the form, see it on /models while logged in, absent when signed out, callable through the gateway (stub or real local endpoint).

### core-P15: playground rewire

- **Goal**: the playground calls the gateway like any client, model preselected from the catalog.
- **Repo + files**: Platform. `apps/web/components/endpoint-playground/EndpointPlayground.tsx`, `apps/web/app/(workspace)/playground/page.tsx`, server route `apps/web/app/api/playground/chat/route.ts`.
- **Spec**: accept `?model=<slug>`; model selector lists gateway models (`GET /api/models`); the chat call goes to `api.experientiallabs.ai/v1/chat/completions` with the org's key server-side; keep the existing per-response cost/latency evidence display, now fed from gateway response usage. Old project-endpoint mode is removed, not dual-pathed (greenfield rule).
- **Depends on**: core-P5 live enough to answer; core-P11 for the deep link.
- **Done when**: from a model detail page, "Open in Playground" lands with that model selected and a streamed reply renders on the preview.

### core-P16: Overview page

- **Goal**: the logged-in landing: personal usage at a glance; workspace view for admins.
- **Repo + files**: Platform. `apps/web/app/(workspace)/overview/page.tsx` + `components/overview/`. Landing redirect: members -> Overview (personal); admins -> Overview (workspace scope). Shell chat owns the sidebar entry; coordinate the landing rule with them.
- **Spec**:
  - Header: your name and email.
  - Scope: admins get a Personal | Workspace switcher and land on Workspace ("land in org first"; cross-workspace/org spend, how many members spend/request tokens, per-member breakdown, top models org-wide). Members see Personal only and land there. Personal activity = requests made with API keys the user created (`api_keys.created_by`).
  - One page-level metric toggle: Spend | Tokens | Requests. One page-level period selector: today, last 7 days, last 30 days, last year, all time. Both re-render every section (top models by <metric>, all stats in <metric>).
  - Sections, in order: (1) usage summary: the current period's total in the chosen metric plus delta vs the previous equal period, with a per-day graph; (2) top models by the chosen metric, right of the graph; (3) activity: GitHub-style contribution graph (daily cells, all-time capable) with longest streak, average per day, average per week, total, all in the chosen metric; (4) your credits: balance + spent today (billing chat's balance component, my mount); (5) API keys (keys chat's component, my mount: `components/overview/ApiKeysMount.tsx`).
  - Data: `GET /api/usage/daily` (metric, period, scope=self|org, group_by=model|member) from the integration chat's projection. If all-time per-day-per-user is too slow, a rollup table gets requested through Silen (flagged in Open items).
  - Design: same Contract 6 language; stat tiles + one restrained graph style; no clutter.
- **Depends on**: core-P8 usage endpoints; integration chat's projection; keys + billing components existing (mounts render labeled empty slots until then).
- **Done when**: on the preview, a member sees personal data only; an admin lands on workspace scope and can switch; toggling metric/period re-renders every section consistently; mounts render; web tests pass.

### core-P17: catalog data fill (fan-out)

- **Goal**: maximize directory coverage; the catalog looks complete on day one.
- **Repo + files**: No product code. Sub-agents write rows via the management API (or a seed PR if the API is not deployed yet). Reference data: `wmo/providers/openrouter_pricing.py` fetch logic (unauthenticated OpenRouter `GET /api/v1/models`), OpenRouter site, Fireworks per-model pages.
- **Spec**: import the full OpenRouter priced catalog as `models` + `model_providers` rows (slug normalization, prices to micro-USD, context, modalities, params, release dates where published; uptime/throughput stats labeled `stats_source='openrouter'`). Research other gateways for fields we would otherwise miss. Preferred models verified by hand. Listed-but-not-callable is fine; callable is defined by core-P18.
- **Depends on**: core-P1 (or core-P8 for API-path writes).
- **Done when**: catalog lists hundreds of models; every preferred model's row is complete (no null price/context); spot-check of 20 random rows against OpenRouter matches.

### core-P18: per-model live integration tests (fan-out)

- **Goal**: every callable model actually called, verified live, with a permanent unit test.
- **Repo + files**: WMO. One test module per provider under the providers' test layout (colocated `*_test.py`), each with a table of per-model cases; live tests marked (e.g. `pytest -m live`) so the default gate stays offline.
- **Spec**: spawn one sub-agent per callable model: preferred list first, then top models per provider (~30 to 45 total). Each agent: resolve the wire id, call through the gateway (stream + tools if supported), fix mapping quirks, record max-token/temperature/reasoning param quirks into the model's `capabilities`/`supported_params`, and leave a permanent test asserting the wire mapping (offline, stubbed) plus a live smoke case. Credentials from the env keys verified above (Gemini pending Silen). Failures that cannot be fixed demote the row to listed-not-callable.
- **Depends on**: core-P4/core-P5; keys available.
- **Done when**: every callable model has a passing offline unit test in the repo and one recorded live pass; the offline suite runs in the whole-repo gate.

### core-P19: listing-correctness verification

- **Goal**: `GET /v1/models`, the catalog page, and reality never disagree.
- **Repo + files**: WMO test (plus a small checker script a sub-agent runs pre-launch).
- **Spec**: a dedicated sub-agent cross-checks three sets: models returned by `GET api./v1/models` for a reference key, models shown callable on /models, and models with a passing core-P18 test. Any mismatch is fixed or the row demoted. Also verifies detail-page prices match `model_providers` rows exactly.
- **Depends on**: core-P11, core-P18, deployed preview.
- **Done when**: the checker reports zero mismatches on the launch candidate deployment.

### core-P20: Fireworks + Modal (secondary, never blocking)

- **Goal**: two more providers the same day, after the seven are solid.
- **Repo + files**: WMO, same adapter interface as core-P4.
- **Spec**: Fireworks = OpenAI-compatible endpoint (openai_compatible adapter + catalog rows + pricing). Modal = org-deployed OpenAI-compatible endpoints (effectively `local` with Modal auth headers). Keys exist in env (`FIREWORKS_API_KEY`, `MODAL_TOKEN_ID/SECRET`). Explicitly never blocks launch; ship behind the same interface whenever ready.
- **Depends on**: core-P4.
- **Done when**: same bar as any provider: adapter tests + at least one live-verified model each, or consciously dropped at launch with rows absent.

## 5. Interfaces

### Exposed by this workstream

- **Tables** (platform Supabase, owned here, everyone else reads through the API): `models`, `model_providers`, `model_waterfalls`, exact columns per core-P1.
- **The usage event** (Contract 2): one per request. Concretely: one `gateway_requests` row (request_id, org, key_id, alias/model, api_surface, accepted_at, terminal_state, terminal_at) joined to its `gateway_attempts` rows (attempt_ordinal, route_depth, deployment/provider/exact_model_id, billing_source [= the lane: customer_managed = pass-through, host_managed = platform-funded], input/cached/output/reasoning tokens, usage_source, price-rate snapshot, estimated_cost_micro_usd, failure_class, started_at/terminal_at [latency]). Table shapes are the WIP's, ported to Postgres by the integration chat; the read surface below is the only sanctioned consumer path.
- **Inference API** (`api.experientiallabs.ai`): `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models`. OpenAI shapes exactly; auth = existing org `xpl_` bearer keys.
- **Management API** (`control.experientiallabs.ai`): `GET /api/models`, `GET /api/models/{slug}`, `GET /api/models/{slug}/providers` (public); `POST /api/models`, `POST /api/models/{slug}/providers`, `GET/PUT /api/models/{slug}/waterfall`, `GET /api/usage`, `GET /api/usage/daily`, `GET /api/credits` (read proxy), `POST/DELETE /api/providers/{provider}` (BYOK writes; UI owned by keys chat) (authed). Machine-readable endpoint inventory maintained in-repo for the docs chat.
- **CLI**: `wmo gateway` command group mirroring the core loop.
- **Frontend routes** (platform): `/models`, `/models/[modelSlug]`, `/models/compare`, `/models/new`, `/overview`, plus the playground rewire.
- **Mount points**: `UseViaKeyMount` on model detail and `ApiKeysMount` on Overview (keys-byok chat); credits balance slot on Overview (billing chat). Named, empty, labeled slots until filled.
- **Playground deep link**: `/playground?model=<slug>`.
- **Catalog snapshots**: `NormalizedGatewayCatalog` derivation from the three tables (core-P3), consumed by the gateway hot path via the integration chat's store.

### Consumed from others

- **platform-gateway-integration chat**: Postgres implementations of `GatewayControlStore` / `AttemptLedger` / `SecretResolver` (interfaces frozen on `origin/agent/gateway-replay-message:wmo/runtime/gateway/interfaces.py`); xpl_ key validation returning `AuthorizationSnapshot`; credits enforcement for the platform-funded lane; the per-user/per-day usage projection behind `GET /api/usage/daily`.
- **keys-byok chat**: the key management UI components for my two mounts.
- **billing chat**: the credits balance component; the $20 free-credit grant.
- **shell chat**: sidebar entries (/models public, /overview logged-in landing rules), layout, the login modal I trigger from gated actions, hiding of old surfaces.
- **Existing platform assets**: `api_keys` table + hashing (unchanged), `provider_connections` + Vault RPCs (BYOK storage), credit ledger, `components/ui` primitives + design tokens.
- **docs chat**: consumes my endpoint inventory (via Silen).

## 6. Cross-chat dependencies and risks

- **Hard dependency: platform-gateway-integration.** core-P5/P8/P10 cannot go live without their Postgres `GatewayControlStore`/`AttemptLedger`. I build and test against the Protocols with stubs, so code lands regardless; the live deployment blocks on them. If they slip: the gateway can boot with a temporary internal store implementation reading `api_keys` directly, but that duplicates their work; coordinator should sequence their store early.
- **Overlap risk: BYOK write endpoints.** I expose `POST /api/providers/{provider}` as the API the keys chat's UI calls; the keys chat might also claim the endpoint itself. Reconcile: I own the HTTP surface, they own all UI. Same risk on `GET /api/credits` vs billing chat.
- **Overlap risk: usage read endpoints.** Telemetry chat may expect to define `GET /api/usage*`. Contract says they consume my events only; the read endpoint is mine, their pages call it.
- **Overlap risk: the gateway branches.** I build against `agent/gateway-replay-message` interfaces; if the integration chat rebases/renames those Protocols tonight, my stubs and imports break. Freeze the interface file or version it.
- **Sequencing: core-P1 merges first** (keys, billing, telemetry block on the schema). It is deliberately migration-only.
- **Shell dependency**: /models stub handoff and the login modal must exist before core-P11/P12 gating works; empty-slot rendering keeps me unblocked meanwhile.
- **PR #441 base risk**: everything frontend sits on an unmerged 25-commit branch with a red preview (`MIGRATIONS_FAILED`). If #441 stalls, my UI PRs inherit the blockage. Coordinator owns this.
- **Landing-page rule** (admins land on Overview workspace scope) intersects shell chat's post-login routing; needs one shared decision at integration time.
- **Playground**: confirmed mine by Silen ("Yeah it should be this chat"), but the old playground is currently wired to project endpoints some other chat may still reference; removal is greenfield-rule compliant but should be announced.

## 7. Open items

- **`plans/` violates WMO's repo layout test** (`wmo/repo_layout_test.py` allows only `.claude/.github/assets/docs/wmo` at root). This planning PR will fail the repo gate as-is; coordinator should either bless a one-line allowlist addition for `plans/` across chats or relocate plan files to `docs/plans/`.
- **Equivalence certification for multi-rung pools**: the WIP contract requires a `GatewayEquivalenceCertification` on any pool with more than one deployment. The exact accepted shape for an operator-asserted certification needs a decision with the integration chat before core-P3 can emit multi-rung pools (interim workaround: singleton pools + executor-level fallback would violate the contract's spirit; do not ship that without sign-off).
- **Gemini API key**: absent from all local env files. Needed for the gemini adapter's live verification (manual step below).
- **Per-day-per-user rollups at all-time scale**: if the integration chat's projection cannot serve the contribution graph cheaply, a rollup table must be requested through Silen (their schema, my consumer).
- **Multi-replica statefulness**: deployment health and cache-stickiness are process-local at launch (matches the WIP). Fine for one replica per role; horizontal scaling needs shared state later.
- **Sign-up-to-key flow**: "sign up / get key" step of the core loop happens in the web UI (keys chat) at launch; the CLI verifies a key but cannot mint one. Acceptable per "keep the way our API keys work right now"; note for docs.
- **`gateway_requests.key_id`/`identity_id` FKs**: the WIP FKs point at `virtual_keys`/`identities`; hosted mode maps to `api_keys` + a synthetic identity. Integration chat's call (shadow rows vs dropped FKs); my usage read must tolerate either.

## 8. Manual steps for Silen

- Provide a Gemini API key (add to the gateway deployment env and to the house org's provider connection); tell product-core where it landed.
- Confirm the deployed environments (Porter apps) hold the same provider keys found locally (openai, anthropic, openrouter, azure, bedrock/AWS, fireworks, modal); grant the new gateway deploy unit access.
- Approve/create the two gateway deploy roles in Porter and the ingress routes: `api.experientiallabs.ai/v1/*` and `control.experientiallabs.ai/api/*` to the WMO gateway service.
- Provide the WMO gateway deployment its Supabase connection (service credentials) per the integration chat's store design.
- Run the production seed/backfill script from core-P2 (house org, house provider connections with real secrets into Vault, preferred models) once reviewed.
- Hand the endpoint inventory (core-P8 artifact) to the docs chat when it stabilizes.
- Rule on the `plans/` layout-test conflict (Open items, first bullet).
