# web/ — the world-model-harness site

A stackwise-style gallery of world models (the local "db" is `src/data/index.json`, generated
from the model cards on disk), a playground embedded in each model's page, and a guided
build-your-own flow. Next.js (App Router) + Tailwind v4; no other runtime deps.

## Run it

```bash
# 1. serve the bundled models (repo root)
uv run wmh serve --root examples/tau-bench --root examples/terminal-tasks --root examples/swe-bench

# 2. run the site
cd web && npm install && npm run dev
```

The gallery and model records render statically from the index — no backend needed. The
playground and `/build` talk to `wmh serve` at `NEXT_PUBLIC_WMH_API` (default
`http://localhost:8000`) and show the exact command to run when it's unreachable. No provider
keys ever reach the browser.

## The local db

`npm run index` regenerates `src/data/index.json` by walking `examples/*/models/*` and
`.wmh/models/*` for `card.json` (see `wmh/config/card.py`) + `metrics.json`. A new world model =
a card on disk — there is no per-model UI code.

## Gate

`npm run lint && npm run typecheck && npm run build` (CI: `.github/workflows/web.yml`). The
python gate does not touch `web/` (AGENTS.md rule 5).
