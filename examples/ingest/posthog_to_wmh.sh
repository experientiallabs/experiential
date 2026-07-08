#!/usr/bin/env bash
# Build a world model from PostHog LLM-observability traces, end to end.
#
# PostHog captures LLM traces as `$ai_*` analytics events (not OTLP spans). Pull them live with the
# `posthog` adapter (HogQL query over the events table), or ingest an exported events file.
set -euo pipefail

MODEL="${1:-posthog-demo}"

# Credentials (a PostHog PERSONAL API key + your region host + numeric project id):
#   export POSTHOG_API_KEY=phx_...           # Settings -> Personal API keys
#   export POSTHOG_HOST=https://us.posthog.com   # or https://eu.posthog.com / self-hosted
#   export POSTHOG_PROJECT_ID=12345

# Option A — pull live from PostHog (build ingests via HogQL: event like '$ai_%'):
uv run wmh build --name "$MODEL" --source posthog --pull \
  --project "${POSTHOG_PROJECT_ID:?set POSTHOG_PROJECT_ID}" --no-interactive

# Option B — build from an exported events file instead (single event / JSON array / JSONL /
# a {"results": [...]} HogQL result):
#   uv run wmh build --name "$MODEL" --source posthog --file events.json --no-interactive

echo "Built world model '$MODEL' from PostHog LLM events."
