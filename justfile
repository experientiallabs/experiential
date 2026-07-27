# Task entry points for the monorepo (uv workspace). `just` = list recipes.

default:
    @just --list

# First-time setup (pipeline step 0): create .env from the template, install everything.
# Idempotent: an existing .env is never touched.
setup:
    @test -f .env && echo ".env exists, leaving it alone" || (cp .env.example .env && echo "created .env from .env.example; fill in the keys you have")
    uv sync --extra dev
    @echo "next: put credentials in .env, then register models with 'uv run wmo providers set'"

# The whole-repo gate (AGENTS.md rule 1): flagship + every Python member.
gate:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check
    uv run pytest -q

test *ARGS:
    uv run pytest -q {{ARGS}}

lint:
    uv run ruff check . && uv run ruff format --check .

# Member-scoped runs from the workspace root, e.g. `just pkg llm-waterfall "pytest -q"`
pkg member task:
    uv run --package {{member}} {{task}}

# web/ carries its own JS gate (AGENTS.md rule 5); no-op until web/ lands
web-gate:
    @test -d web && (cd web && npm run lint && npx tsc --noEmit) || echo 'web/ not landed yet'
