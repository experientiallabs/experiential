# Task entry points. `just` = list recipes.

default:
    @just --list

# First-time setup (pipeline step 0): create .env from the template, install everything.
# Idempotent: an existing .env is never touched.
setup:
    @test -f .env && echo ".env exists, leaving it alone" || (cp .env.example .env && echo "created .env from .env.example; fill in the keys you have")
    uv sync --extra dev
    @echo "next: run 'uv run exp build --help' and start from an explicit local trace export"

# The whole-repo gate (AGENTS.md rule 1).
gate:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check
    uv run pytest -q

test *ARGS:
    uv run pytest -q {{ARGS}}

lint:
    uv run ruff check . && uv run ruff format --check .
