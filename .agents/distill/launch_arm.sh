#!/usr/bin/env bash
# Launch one arm of the Ultra-550B -> Super-120B two-arm distillation comparison.
#
# The two arms differ ONLY in train.loss (ppo vs topk_ce); learning rate (5e-5), steps (45),
# batch (16 tasks x 4 attempts = 64 episodes/step), rollout params and warmup (0) are matched, so
# the comparison actually isolates the objective. Sized on supervision volume rather than step
# count: 45 x 64 = 2,880 episodes ~= 18M loss tokens, matching the ~18M behind the only measured
# win in this project (60k GSM8k sequences x ~300 tokens each). Budget cap is $900/arm; measured
# cost is ~$0.09/episode, so ~$260/arm expected.
#
# Usage: launch_arm.sh <anchor|topk>
set -euo pipefail

arm="${1:?usage: launch_arm.sh <anchor|topk>}"
case "$arm" in
anchor | topk) ;;
*)
    echo "arm must be 'anchor' (ppo) or 'topk' (topk_ce)" >&2
    exit 2
    ;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$here"

# Credentials: TINKER_API_KEY, E2B_API_KEY, WANDB_API_KEY, WMH_E2B_SANDBOX_CAP.
# Never `wandb login`: the key must come from the environment or it lands on the wrong account.
#
# The env file is a convenience, not a requirement. Sourcing it unconditionally under `set -e`
# meant every machine without that exact path exited here, before launching anything. Point
# WMH_ENV_FILE somewhere else, or export the four keys yourself and skip the file entirely.
env_file="${WMH_ENV_FILE:-${HOME}/Documents/experientiallabs/platform/.env.local}"
if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$env_file"
    set +a
else
    echo "note: no env file at ${env_file}; using the ambient environment" >&2
fi

# Fail on the missing key rather than 40 minutes into a paid run.
missing=()
for var in TINKER_API_KEY E2B_API_KEY WANDB_API_KEY; do
    [ -n "${!var:-}" ] || missing+=("$var")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "missing required credentials: ${missing[*]}" >&2
    echo "export them, or set WMH_ENV_FILE to a file that does" >&2
    exit 2
fi

# Three CLI shapes worth knowing, each learned from a rejection:
#  - No --harbor-config under --mode distill: the job template, attempts and schedule come from
#    the distill TOML's [harbor] section instead.
#  - --run-dir is REQUIRED; it holds all durable run state (config snapshot, metrics,
#    checkpoints, rollout artifacts) and is what --resume reattaches to.
#  - The NAME positional is the HARNESS to run, not a label for this run. `pi` is the built-in
#    default agent; anything else must already exist under .wmh/harnesses. The run's identity
#    lives in --run-dir and in the config's [wandb] run_name.
#
# The configs are the `super` pair, matching the comparison this script documents: both are
# Ultra-550B -> Super-120B and differ only in train.loss. The old `nano-` path named a config
# that was never checked in, so the topk arm exited while loading it.
config=".agents/distill/distill-super-${arm}.toml"
if [ ! -f "$config" ]; then
    echo "missing distill config: ${config}" >&2
    exit 2
fi

exec uv run wmh optimize harness pi harbor \
    --mode distill \
    --distill-config "$config" \
    --run-dir ".wmh/distill-runs/super-${arm}-v4" \
    --task-ids .agents/distill/tb2-train-task-ids.json \
    --holdout-task-ids .agents/distill/tb2-holdout-task-ids.json \
    --backend e2b \
    --yes
