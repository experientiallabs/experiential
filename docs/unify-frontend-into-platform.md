# Unify the world-model-harness public site into the platform (single frontend)

This is the transfer prompt for the task of collapsing the two frontends (the public
site in this repo's `web/` and the platform's `apps/web`) into one. It is written to be
handed to a fresh agent working primarily in `experientiallabs/platform`. A short
pointer prompt (where to find this and the fundamental problem) is kept with the task
owner; this file is the full brief.

## 0. Who you are and where you work

You are picking up a cross-repo unification. There are two repos:

- **`experientiallabs/world-model-harness`** (aka "wmh"): the engine, CLI, benchmark
  capture, and the public model artifacts. It currently ALSO contains a standalone
  public website at `web/` (Next.js). That website is what we are relocating.
- **`experientiallabs/platform`** (aka "platform"): the hosted product. A monorepo
  with a FastAPI backend and a Next.js app at `apps/web`, built on Supabase
  (Postgres + RLS, Storage, Auth). This is the DESTINATION and where you primarily work.

The mission: **collapse the two frontends into one.** The platform's `apps/web`
becomes the single frontend: an unauthenticated public front page (the wmh gallery
and playground) plus an authenticated product behind a collapsible sidebar (your
catalog, harnesses, usage). wmh reverts to a pure engine/CLI/artifact source and
publishes its public catalog to the platform. Do NOT leave two live frontends.

Honor the platform repo's own conventions. Also honor these from wmh, which the CTO
wants carried over:
- **No em dashes anywhere**: code, comments, docstrings, docs, UI copy, commit
  messages, PR descriptions. Use colons, parentheses, or commas instead.
- **Brand system** (from wmh AGENTS.md rule 14): ink `#0a0a0a`, hairline `#ececec`,
  accents `#0070f3` / `#7928ca` / `#f5a623` / `#ee0000`, teal `#50e3c2`. Restraint,
  hairline borders, soft motion. Reconcile with the platform's existing tokens; do
  not ship two design languages.
- Tests live inline next to code (`foo.py` -> `foo_test.py`); new behavior has a test.
- Module docstrings and Google-style docstrings on significant functions/classes.

## 1. The vision in one paragraph

One product, one domain. Land on the public gallery of world models with no login: a
catalog rendered from BOTH the models committed in the wmh repo AND rows in the
platform's Supabase DB, an in-browser playground, an open-loop trace explorer, and a
build-your-own flow. A collapsible left sidebar (collapsed by default when logged out)
carries login and, once authenticated, the tenant-scoped product: your projects'
private models, the agent-harness catalog, and cross-surface cost/usage. Web login is
the same Supabase identity as `wmh login`, so a model you create on the web syncs to
your machine via the CLI and runs locally, and usage is tracked across CLI, web, and
platform under one account.

## 2. Current state: the wmh side (what moves)

Repo `experientiallabs/world-model-harness`, directory `web/`:
- Next.js **16.2.10**, React **19.2.4**, Tailwind **v4** (CSS-first `@theme` via
  `@tailwindcss/postcss`), TypeScript. App Router.
- Pages: `web/src/app/page.tsx` (gallery homepage, models sorted by fidelity so
  evaluated/starred lead), `web/src/app/models/[name]/page.tsx` (model detail with
  embedded playground), `web/src/app/build/page.tsx` (build-your-own wizard),
  `layout.tsx`, `globals.css`.
- Components (`web/src/components/`): `ModelView` (two-column detail + tabs +
  max-fidelity grid), `Playground` (centered ChatGPT-style composer, wmh-play action
  grammar, optimistic pending action + "getting environment response" spinner),
  `TracesExplorer` (live trace fetch + Hub download panel + open-loop side-by-side
  ComparisonView + task prompt), `LivePanels`, `LiveModel`, `FidelityGrid` (canvas
  wave animation, teal, the "fun" max-fidelity mode), `BuildFlow` (SSE build
  progress), `ModelRecord`, `ServeControls`, `ServeDownPanel`, `Spinner`, `Logo`,
  `Wordmark`, `RandomModelLink`.
- Lib: `web/src/lib/api.ts` (typed client for `wmh serve`), `types.ts`,
  `index-data.ts`, `parse-action.ts` (wmh-play grammar parser).
- Data: `web/src/data/index.json`, generated at build time by
  `web/scripts/build-index.mjs`, which walks model roots (`.wmh/models`, then
  `packages/environment-capture/*/models`) and reads each `card.json` into per-model
  `preview`, `suggestions` (chat starter chips), and `scenarios` (open-loop replays).

Where the wmh site gets data today:
- **Build-time**: `index.json` from committed `card.json` files (the public catalog).
- **Runtime**: a locally running `wmh serve` (FastAPI, default `http://localhost:8000`,
  env `NEXT_PUBLIC_WMH_API`). CORS is currently localhost-only by regex.

`wmh serve` endpoints the frontend uses (you will proxy or reimplement equivalents in
the platform for production; keep the contract identical so the CLI and web share it):
- `GET /healthz`
- `GET /world_models` -> list of models with their cards
- `POST /world_models/{name}/sessions` -> `{session_id, state}` (body `{task}`)
- `GET /world_models/{name}/sessions/{id}`
- `POST /world_models/{name}/sessions/{id}/step` -> `{observation, state}`
- `GET /world_models/{name}/sessions/{id}/usage` -> RunRecord (cost/tokens/time)
- `DELETE /world_models/{name}/sessions/{id}` -> RunRecord (end session)
- `POST /world_models/builds` (202) + `POST /world_models/builds/uploads`
  + `GET /world_models/builds/{id}` + `GET /world_models/builds/{id}/events` (SSE)
- `GET /world_models/{name}/traces` -> local scenarios or a Hub download offer
- `POST /world_models/{name}/traces/download` (202) + `GET .../traces/download`

The 10 committed public models (each has `card.json` + a small RAG index artifact),
under `packages/environment-capture/<bench>/models/<name>/`:
`tau-bench` (fidelity 0.915, starred), `terminal-tasks` (0.866, starred),
`swe-bench` (0.822, starred), `tau-telecom`, `bird-sql`, `continual-learning`,
`crmarena`, `dabstep`, `financebench`, `gaia2` (the last six are RAG-only, unscored).

The ModelCard schema (`wmh/config/card.py`, `ModelCard`), which the catalog renders and
which must map cleanly onto the Supabase row shape:
`schema_version, name, title, description, task, corpus{traces,steps,source},
provider, model_id, fidelity{suite,score,std,run_id}, cost_per_step_usd,
latency_per_step_s, built_at, license, tags[], traces_hf{repo,path,revision,kind}`.

Traces are large and are NOT committed. wmh stores them on the Hugging Face Hub and
downloads on demand (`traces_hf` on the card, public resolve URL, no auth). In the
unified product, traces should be served from platform Storage or streamed the same
way; do not assume a user's localhost has them.

## 3. Current state: the platform side (the destination)

Repo `experientiallabs/platform` (monorepo, FastAPI + `apps/web` Next.js, Supabase).
Key facts from platform PR #259 ("wmh registry"), which pairs with wmh PR #132:

- **Auth + identity**: Supabase Auth (Google/GitHub). `GET /api/whoami` returns
  identity + orgs + projects. `/cli/auth` is a cookie-gated browser approval page that
  mints an org key for `wmh login` and lands it on the CLI's loopback listener.
  `GET /api/cli/config` is a public backend-URL discovery endpoint.
- **Tenancy**: orgs and **projects**. Registry routes are namespaced
  `/api/projects/{project_id}/...` and allowlisted at member strength with RLS.
- **World-model registry**: `GET /api/projects/{id}/world-models`;
  bundle push via `POST .../world-models/{name}/bundle/uploads` (signed Supabase
  Storage upload URL; bytes never flow through the API) then `POST .../bundle`
  (finalize: verify sha256/size, tar safety, move object, repoint row, evict cached
  engine); `GET .../bundle` returns an expiring signed URL + digest for pull.
- **Harness registry**: `harnesses` + `harness_versions` tables (append-only
  `HarnessDoc` lineage, `doc_hash` blake2b-128, org-member read RLS). Routes
  `GET /api/projects/{id}/harnesses`, `.../harnesses/{name}`,
  `.../harnesses/{name}/versions/{v}`, `POST .../harnesses/{name}/versions`.
- **Storage**: Supabase Storage, cap raised to 1GB for large retrieval indexes.
- Testing on the platform: `apps/web` uses tsc + vitest; backend uses pytest + pgTAP.
- Default hosted web URL: `https://experiential-platform-web.vercel.app`.

The wmh CLI side already exists (wmh PR #132): `wmh login/logout/status/push/pull`,
credentials at `~/.wmh/credentials.toml`, env precedence
`WMH_PLATFORM_URL / WMH_PLATFORM_API_URL / WMH_PLATFORM_TOKEN / WMH_PLATFORM_PROJECT`.
A typed Python client lives at `wmh/platform/client.py` (mirror its method set when you
build the web platform client).

## 4. Target architecture

**One frontend: platform `apps/web`.** It serves the public front page and the authed
product. Two data planes:

- **Reads go directly to Supabase** (via `@supabase/ssr` + `@supabase/supabase-js`)
  using the visitor's session. RLS in the DB governs visibility, so no auth logic is
  duplicated in the frontend: anon sees public rows, members see their org/project rows.
- **Writes and actions go through the platform FastAPI**: bundle push (signed URLs),
  finalize, run/step against a hosted engine, engine cache eviction. These need server
  logic and must not be done from the browser directly.

**Catalog from multiple sources, merged (the CTO's "look in multiple places").**
A single `CatalogEntry` type that both sources normalize into:
1. **Repo-committed public models** (vetted showcase): the wmh `card.json` set, brought
   into the platform as a build-time data dependency (see section 6 for the publish
   pipeline). Zero-latency, works with no auth.
2. **Supabase rows**: public rows for everyone, private/org rows once logged in.
Merge and dedupe on a canonical slug. Precedence and provenance: decide with the user
(default proposed: repo-committed wins on display since it is curated; if a DB twin
exists, link to "open in platform" / run). Tag each entry with `source: "repo" | "db"`.

**Auth: Supabase, the same project the platform already uses.** Google/GitHub via
Supabase Auth, SSR cookies (single origin, so this is clean). Web login == platform
login == the identity behind `wmh login`. Needed public env values:
`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`. Confirm RLS lets anon read
public model rows (or add a read-only public view).

**Playground in production talks to the platform engine, not a user's localhost.** The
wmh `web/` playground targets `wmh serve`. In the unified app, sessions/step/usage hit
platform-hosted engine instances (the platform already evicts cached engines on bundle
finalize, so it hosts engines). Keep local `wmh serve` as a dev-only affordance in wmh.

**wmh's role after the merge**: engine + CLI + public-model artifact source. It keeps
`card.json` files as the source of truth for the public catalog and publishes them to
the platform. Its `web/` directory is retired (or reduced to a minimal local preview,
per the open decision in section 8). Do not maintain two production sites.

## 5. The sidebar and login UX (exact spec)

- App shell with a **left sidebar that starts collapsed**, and is always collapsed when
  unauthenticated.
- **A toggle in the top-left** opens and closes the rail, with a smooth animation
  (width plus content fade, roughly 180ms ease; the Notion/Vercel restraint, no bounce).
- **No fixed login button.** In the collapsed rail, login is a **single icon** with a
  **persistent, tasteful tooltip that points at the icon and reads "Log in"**: a small
  rounded chip, hairline border, soft shadow, a small caret aimed at the icon, always
  visible while logged out so login is discoverable without being loud. Clean, in the
  Notion/Apple/Vercel register.
- When authenticated, that same bottom slot becomes the account avatar (tooltip gone),
  and the expanded sidebar reveals nav: Catalog, Your models, Harnesses, Usage, and a
  project switcher.
- The catalog is the front page throughout. Collapsed state must not disturb the landing
  experience; the sidebar overlays or gently pushes content.

## 6. The publish pipeline (repo-committed models -> platform)

The public models must stay committed in wmh (their `card.json` is the source of truth)
yet render in the platform frontend. Two mechanisms, pick with the user (section 8):
- **Build-time data dependency**: wmh emits a `public-catalog.json` (reuse the logic in
  `web/scripts/build-index.mjs`) that the platform vendors at build (git submodule,
  published data package, or CI artifact). The frontend reads it as the "repo" source.
- **Supabase seed job**: a `wmh publish` / CI job upserts committed cards as public rows
  in Supabase. Then "repo" and "db" converge in the DB, but you lose the "read straight
  from committed data" property the CTO asked for; prefer the build-time data path for
  the committed-source requirement and keep Supabase for community/user models.
Whichever is chosen, the small RAG index artifacts committed with each model need a home
(platform Storage) if the public models are to be runnable in the hosted playground.

## 7. Migration plan (phased; land value early, do not big-bang)

- **Phase 0 (buildable immediately, no platform inputs): the shell.** In `apps/web`,
  add the collapsible sidebar, top-left toggle, and login icon + persistent tooltip,
  with the catalog as the front page rendering the repo-only source. Login is a stub
  that opens the Supabase flow (wired in Phase B).
- **Phase A: port the surfaces.** Bring the wmh components (gallery, ModelView,
  Playground, TracesExplorer, FidelityGrid, BuildFlow) into `apps/web`, reconciling the
  design system into one set of tokens. Rewrite `lib/api.ts` to target the platform
  engine endpoints (keep the contract identical to `wmh serve`).
- **Phase B: auth + two-source catalog.** Wire Supabase Auth (Google/GitHub). Add the
  Supabase read client and merge repo + DB into `CatalogEntry` with dedupe/precedence.
- **Phase C: authed product.** Sidebar sections: Your models (`list_world_models` per
  project), Harnesses (versioned), Usage (cross-surface cost). Actions via FastAPI.
- **Phase D: publish pipeline + retire wmh/web.** Stand up the repo->platform publish
  path (section 6), point the platform at it, then remove wmh `web/` (or reduce to a
  local dev preview). Update wmh docs/CI so nothing still expects a wmh site.

## 8. Open decisions to resolve with the user before/while building

1. **Same Supabase project** for the web app as the platform: confirm, and obtain
   `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY`, and confirm anon can
   read public model rows (or add a public read-only view).
2. **Dedup precedence** when a model exists both in-repo and in the DB: repo-committed
   wins on display (proposed default), or DB wins when it has fresher fidelity.
3. **Publish mechanism** (section 6): build-time vendored data vs Supabase seed job.
   Recommendation: build-time data for the committed-source requirement, Supabase for
   community models.
4. **Retire wmh `web/` fully, or keep a minimal local dev preview?**
5. **Single mega-PR vs stacked PRs.** Recommendation: land Phase 0 (shell + repo-only
   catalog) first since it needs nothing from the platform, then stack the rest as the
   Supabase inputs arrive. Same feature branch if a single PR is required.
6. **Where the public models' RAG index artifacts live** in production (platform
   Storage) so the hosted playground can run them.

## 9. Nooks and crannies (do not skip)

- **wmh-play action grammar**: the playground and `parse-action.ts` speak a specific
  grammar (`say ...`, `tool_name {json args}`). Preserve it; it also matches the
  serve-side `_action_label` formatter.
- **Open-loop replay**: the trace explorer replays recorded scenarios showing our
  model's answer side-by-side with the golden observation per step, with the initial
  task prompt shown. Keep the side-by-side ComparisonView.
- **FidelityGrid / max mode**: the "fun" canvas animation (teal wave, resolute-agent
  style grid) is intentional; carry it over.
- **Build-your-own SSE**: builds stream progress over `EventSource`; the platform must
  provide an equivalent SSE (or the platform's own build events).
- **Traces are large and Hub-hosted**: keep the on-demand fetch model; do not commit
  trace corpora; serve from Storage in prod.
- **Cross-surface cost**: usage is metered at the provider boundary in wmh
  (`wmh/tracking`); the platform should aggregate CLI + web + platform usage under the
  account for the Usage view.
- **CORS**: wmh serve is localhost-only today; the platform's own CORS/allowlist rules
  apply once the app is same-origin.
- **Shared interface with the CLI**: the web platform client and the CLI
  (`wmh/platform/client.py`) must speak the same registry API. wmh #132 and platform
  #259 are the shared contract; keep API vs CLI a pure interface difference, no
  duplicate logic.
- **No em dashes** anywhere (repeat, because it is easy to slip). Brand tokens as in
  section 0.

## 10. First actions for you

1. Read `apps/web` in the platform repo end to end: its routing, auth setup, design
   tokens, and how it currently talks to the FastAPI backend and Supabase.
2. Clone/read the wmh `web/` surfaces you are porting (section 2) so you carry the UX
   faithfully.
3. Confirm the section 8 decisions with the user, get the Supabase public env values.
4. Build Phase 0 (the shell) in `apps/web` and get it rendering the repo-only catalog,
   then proceed through the phases.

Report a short plan back before large moves; this is a real product merge, not a spike.
