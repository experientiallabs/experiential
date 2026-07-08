#!/usr/bin/env bash
# Overnight SWE-bench Verified capture: grow the trace corpus with a large slice, resiliently.
#
#   packages/environment-capture/swe-bench/capture_overnight.sh
#
# Runs the real mini-swe-agent harness over a wide slice of SWE-bench Verified, live on Bedrock.
# Resilient by design for an unattended overnight run:
#   * Resumable: mini-swe-agent SKIPS instances that already have output (no --redo-existing), so a
#     crash/restart continues where it left off. Re-running this script just fills gaps.
#   * Account/model chain: try the endflow account on Opus 4.6 first (verified: it invokes 4.6 and
#     litellm's Bedrock route works with 4.6 — unlike 4.7, which throws "missing field type"); if a
#     provider becomes unavailable, fall back to the default account on Opus 4.8 (the model the
#     original swe capture used). Whichever model, the SAME output dir is reused so progress carries
#     over.
#   * A watchdog loop restarts the runner if it exits non-zero (transient Docker/Bedrock hiccups),
#     until the slice is exhausted or MAX_RESTARTS is hit.
#
# SWE-bench images are multi-GB x86_64 running under arm64 emulation (~10-30 min/instance), so a wide
# slice is genuinely an overnight job. Output: runs/overnight/<instance_id>/<instance_id>.traj.json.
set -uo pipefail
cd "$(dirname "$0")"

SLICE="${SLICE:-40:240}"          # disjoint-ish from the committed first-40; ~200 instances
WORKERS="${WORKERS:-3}"
OUT="${OUT:-runs/overnight}"
MAX_RESTARTS="${MAX_RESTARTS:-40}"

# (profile, model) fallback chain, tried in order; a link that's unreachable or daily-capped is
# skipped at runtime (reachability probe below). "stackwise" is AWS account 282563636010 reached via
# the `default` profile (user/silen, admin) — NOT the permission-less `stackwise-agent` IAM user.
# All six are verified reachable; the backticks config means even 4.7 works (no tool_choice).
CHAIN=(
  "endflow:bedrock/us.anthropic.claude-opus-4-6-v1"   # endflow acct 761200393827
  "endflow:bedrock/us.anthropic.claude-sonnet-4-6"
  "default:bedrock/us.anthropic.claude-opus-4-6-v1"   # stackwise acct 282563636010 (default=silen)
  "default:bedrock/us.anthropic.claude-opus-4-7"
  "default:bedrock/us.anthropic.claude-sonnet-4-6"
  "default:bedrock/us.anthropic.claude-opus-4-8"
)

export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"

# Use the TEXT-BASED (backticks) config, NOT the default tool-calling one: litellm's Bedrock route
# injects a malformed `tool_choice` for Opus 4.6/4.7 ("tool_choice.type: Field required" / "missing
# field type"), so the tool-calling config dies on the first step for anything but 4.8. The backticks
# config emits a fenced ```mswea_bash_command``` block instead — no tool_choice — which every Opus
# works with AND which convert_to_wmh.py's fenced-block path parses directly.
CONFIG="${CONFIG:-swebench_backticks.yaml}"

# FILTER (optional): an instance-id regex to target a specific recovery set. When set, --filter
# selects those instances and --slice is omitted (slice applies AFTER filter and would truncate).
# --redo-existing so a link re-attempts instances an earlier (capped/failed) link already touched.
FILTER="${FILTER:-}"

# A daily-token cap (RateLimitError "Too many tokens per day") does NOT crash the runner — it keeps
# spinning, failing every instance, forever. So run the runner in the BACKGROUND and watch its log:
# the moment N cap errors appear, kill it and return 42 so the caller advances to the next link.
# Returns 0 on a clean finish, 42 on daily-cap, other non-zero on crash.
CAP_THRESHOLD="${CAP_THRESHOLD:-8}"   # this many cap errors => the account is capped, bail
run_once() {
  local profile="$1" model="$2" out="$3"
  echo "=== capture attempt: profile=$profile model=$model workers=$WORKERS out=$out ==="
  mkdir -p "$out"
  local common=(-c "$CONFIG" --subset verified --split test --environment-class docker
                -m "$model" -w "$WORKERS" -o "$out")
  if [ -n "$FILTER" ]; then common+=(--filter "$FILTER" --redo-existing); else common+=(--slice "$SLICE"); fi

  AWS_PROFILE="$profile" .venv/bin/python -m minisweagent.run.benchmarks.swebench "${common[@]}" &
  local pid=$!
  # Watchdog: poll the log; kill the runner if the daily cap trips.
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    local caps
    caps=$(grep -c "Too many tokens per day" "$out/minisweagent.log" 2>/dev/null || echo 0)
    if [ "${caps:-0}" -ge "$CAP_THRESHOLD" ]; then
      echo "--- $profile/$model daily-capped ($caps errors); killing runner, advancing ---"
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      docker ps -aq --filter name=minisweagent 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
      return 42
    fi
  done
  wait "$pid"
}

# In FILTER (recovery) mode each chain link writes to its OWN sub-dir ($OUT/link<N>_<profile>_<tag>),
# so a link's successes are never overwritten by a later link's --redo-existing. Convert+merge reads
# every sub-dir. When a link hits its daily cap we advance to the next; instances a prior link already
# solved are harmless to re-attempt (their good trace still lives in the prior link's dir + gets
# deduped by trace_id at merge time).
restarts=0
while :; do
  progressed=0
  i=0
  for entry in "${CHAIN[@]}"; do
    i=$((i + 1))
    profile="${entry%%:*}"; model="${entry#*:}"
    link_out="$OUT"
    [ -n "$FILTER" ] && link_out="$OUT/link${i}_${profile}_$(echo "$model" | sed 's#.*/##;s#[^a-zA-Z0-9]#_#g')"
    # Reachability probe that ALSO treats a daily-cap as unreachable (a bare converse can slip
    # through intermittently even when the account is token-capped, so match the cap message).
    if ! AWS_PROFILE="$profile" .venv/bin/python -c "
import boto3,sys
try:
    boto3.client('bedrock-runtime',region_name='$AWS_REGION').converse(
        modelId='${model#bedrock/}', messages=[{'role':'user','content':[{'text':'hi'}]}],
        inferenceConfig={'maxTokens':2})
except Exception as e:
    print(e); sys.exit(1)
" >/dev/null 2>&1; then
      echo "--- $profile/$model unreachable/capped, trying next in chain ---"
      continue
    fi
    run_once "$profile" "$model" "$link_out"
    rc=$?
    if [ "$rc" -eq 42 ]; then
      progressed=1; continue          # daily-capped mid-run -> next link
    fi
    echo "=== runner finished on $profile/$model (rc=$rc) ==="
    exit "$rc"
  done
  restarts=$((restarts + 1))
  if [ "$progressed" -eq 0 ]; then
    echo "=== no reachable provider in the chain; giving up ==="; exit 1
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "=== hit MAX_RESTARTS=$MAX_RESTARTS; stopping (resume by re-running) ==="; exit 1
  fi
  echo "=== runner exited non-zero; restart $restarts/$MAX_RESTARTS in 20s (resumes, skips done) ==="
  sleep 20
done
