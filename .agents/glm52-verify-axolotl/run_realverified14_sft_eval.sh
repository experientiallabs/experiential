#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
CODE_ROOT="${CODE_ROOT:-${XROOT}/eval-code-b0d0568}"
PYTHON="${PYTHON:-${XROOT}/runtime/venv-harbor-tb2/bin/python}"
E2B_ENV="${E2B_ENV:-${XROOT}/runtime/credentials/e2b.env}"
TMAX_SOURCE="${TMAX_SOURCE:-${XROOT}/assets/tmax-source-7387d2f91423}"
TASKS_ROOT="${TASKS_ROOT:-${XROOT}/assets/tmax15k-subset-1600/tasks}"
TASK_IDS="${TASK_IDS:-${XROOT}/axolotl-sft/eval-manifests/realverified14_task_ids.txt}"
LIMIT="${LIMIT:-14}"
SERVER_MANIFEST="${SERVER_MANIFEST:?SERVER_MANIFEST is required}"
API_BASE="${API_BASE:?API_BASE is required}"
MODEL="${MODEL:?MODEL is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
ARM="${ARM:-sft-step100}"
OUT_FAMILY="${OUT_FAMILY:-realverified14-sft}"
TEMPLATE_ALIAS="${TEMPLATE_ALIAS:-tmax15k-full-da54e6370473}"
CONCURRENCY="${CONCURRENCY:-7}"
OUT="${XROOT}/tmax/${OUT_FAMILY}/${RUN_NAME}"
LOG="${XROOT}/logs/tmax-${OUT_FAMILY}/${RUN_NAME}.log"

for required in "${PYTHON}" "${E2B_ENV}" "${TASK_IDS}" "${SERVER_MANIFEST}"; do
  test -r "${required}" || { echo "missing required file: ${required}" >&2; exit 1; }
done
test -d "${TMAX_SOURCE}" || { echo "missing TMax source: ${TMAX_SOURCE}" >&2; exit 1; }
test -d "${TASKS_ROOT}" || { echo "missing task root: ${TASKS_ROOT}" >&2; exit 1; }
test "${LIMIT}" -gt 0 || { echo "LIMIT must be positive" >&2; exit 1; }
test "$(wc -l < "${TASK_IDS}")" -eq "${LIMIT}" || {
  echo "task manifest row count does not match LIMIT=${LIMIT}" >&2
  exit 1
}
test ! -e "${OUT}" || { echo "refusing to overwrite: ${OUT}" >&2; exit 1; }
test ! -e "${LOG}" || { echo "refusing to overwrite: ${LOG}" >&2; exit 1; }

set -a
# shellcheck source=/dev/null
. "${E2B_ENV}"
set +a
test -n "${E2B_API_KEY:-}" || { echo "E2B_API_KEY is unset" >&2; exit 1; }
export PYTHONPATH="${TMAX_SOURCE}:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
cp "${SERVER_MANIFEST}" "${OUT}/server_manifest.json"
sha256sum "${TASK_IDS}" "${SERVER_MANIFEST}"
curl -fsS "${API_BASE}/models" >/dev/null

"${PYTHON}" "${CODE_ROOT}/tools/x_token/run_tmax_e2b_eval.py" \
  --tasks "${TASKS_ROOT}" \
  --task-ids "${TASK_IDS}" \
  --limit "${LIMIT}" \
  --out "${OUT}" \
  --template-alias "${TEMPLATE_ALIAS}" \
  --run-id "${RUN_NAME}" \
  --arm "${ARM}" \
  --model "${MODEL}" \
  --api-base "${API_BASE}" \
  --repeats 1 \
  --concurrency "${CONCURRENCY}" \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-steps 64 \
  --max-tokens 16384 \
  --max-total-response-tokens 65536 \
  2>&1 | tee "${LOG}"
