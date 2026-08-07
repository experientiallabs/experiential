#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
PYTHON="${PYTHON:-${XROOT}/runtime/venv-harbor-tb2/bin/python}"
E2B_ENV="${E2B_ENV:-${XROOT}/runtime/credentials/e2b.env}"
TMAX_SOURCE="${TMAX_SOURCE:-${XROOT}/assets/tmax-source-7387d2f91423}"
CODE_ROOT="${CODE_ROOT:-${XROOT}/eval-code-b0d0568}"
TASKS_ROOT="${TASKS_ROOT:-${XROOT}/assets/tmax15k-subset-1600/tasks}"
AUDIT_DATASET="${AUDIT_DATASET:-${XROOT}/axolotl-sft/sft-data/calibration-double-pass-17.audit.jsonl}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-${XROOT}/axolotl-sft/scripts/replay_admitted_teacher_trajectories.py}"
OUT="${OUT:-${XROOT}/teacher-replay/verified17-real-verifier-run1}"
LOG="${LOG:-${XROOT}/logs/teacher-replay/verified17-real-verifier-run1.log}"
TEMPLATE_ALIAS="${TEMPLATE_ALIAS:-tmax15k-full-da54e6370473}"
CONCURRENCY="${CONCURRENCY:-4}"

for required in "${PYTHON}" "${E2B_ENV}" "${AUDIT_DATASET}" "${REPLAY_SCRIPT}"; do
  test -r "${required}" || { echo "missing required file: ${required}" >&2; exit 1; }
done
test -d "${TMAX_SOURCE}" || { echo "missing TMax source: ${TMAX_SOURCE}" >&2; exit 1; }
test -d "${TASKS_ROOT}" || { echo "missing task root: ${TASKS_ROOT}" >&2; exit 1; }
test ! -e "${OUT}" || { echo "refusing to overwrite: ${OUT}" >&2; exit 1; }

set -a
# shellcheck source=/dev/null
. "${E2B_ENV}"
set +a
test -n "${E2B_API_KEY:-}" || { echo "E2B_API_KEY is unset" >&2; exit 1; }
export PYTHONPATH="${TMAX_SOURCE}:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$(dirname "${LOG}")"
sha256sum "${AUDIT_DATASET}" "${REPLAY_SCRIPT}"
df -h /mnt/datasets
"${PYTHON}" -c 'import e2b, harbor, nemo_rl; print("teacher replay import gate passed")'

"${PYTHON}" "${REPLAY_SCRIPT}" \
  --audit-dataset "${AUDIT_DATASET}" \
  --tasks "${TASKS_ROOT}" \
  --out "${OUT}" \
  --template-alias "${TEMPLATE_ALIAS}" \
  --run-id qwen35-4b-glm52-verified17-teacher-real-verifier-run1 \
  --concurrency "${CONCURRENCY}" \
  --sandbox-timeout-s 3600 \
  --bash-timeout-s 120 \
  2>&1 | tee "${LOG}"
