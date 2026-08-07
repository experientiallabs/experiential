#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
CODE_ROOT="${CODE_ROOT:-${XROOT}/eval-code-b0d0568}"
PYTHON="${PYTHON:-${XROOT}/runtime/venv-harbor-tb2/bin/python}"
E2B_ENV="${E2B_ENV:-${XROOT}/runtime/credentials/e2b.env}"
TMAX_SOURCE="${TMAX_SOURCE:-${XROOT}/assets/tmax-source-7387d2f91423}"
TASKS_ROOT="${TASKS_ROOT:-${XROOT}/assets/tmax15k-subset-1600/tasks}"
TASK_IDS="${TASK_IDS:-${XROOT}/axolotl-sft/eval-manifests/verified17_task_ids.txt}"
SERVER_MANIFEST="${SERVER_MANIFEST:-${XROOT}/axolotl-sft/eval-runtime/tb2-4b-serve-manifest.json}"
API_BASE="${API_BASE:-http://127.0.0.1:8120/v1}"
TEMPLATE_ALIAS="${TEMPLATE_ALIAS:-tmax15k-full-da54e6370473}"
RUN_ROOT="${RUN_ROOT:-${XROOT}/tmax/verified17-sft-step100/full17}"
LOG_ROOT="${LOG_ROOT:-${XROOT}/logs/tmax-verified17-sft-step100/full17}"
BASE_SUMMARY="${RUN_ROOT}/base-run1/summary.json"
OUT="${RUN_ROOT}/sft-step100-run1"
ARM_LOG="${LOG_ROOT}/sft-step100-run1.log"
DRIVER_LOG="${LOG_ROOT}/sft-only-driver.log"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

for required in "${PYTHON}" "${E2B_ENV}" "${TASK_IDS}" "${SERVER_MANIFEST}" "${BASE_SUMMARY}"; do
  test -r "${required}" || { echo "missing required file: ${required}" >&2; exit 1; }
done
test -d "${TMAX_SOURCE}" || { echo "missing TMax source: ${TMAX_SOURCE}" >&2; exit 1; }
test -d "${TASKS_ROOT}" || { echo "missing task root: ${TASKS_ROOT}" >&2; exit 1; }
test ! -e "${OUT}" || { echo "refusing to overwrite: ${OUT}" >&2; exit 1; }

"${PYTHON}" -c 'import json,sys; row=json.load(open(sys.argv[1])); assert row["attempted"] == 17; assert row["scored"] == 16; assert row["status_counts"] == {"episode_error": 1, "scored": 16}' "${BASE_SUMMARY}"

set -a
# shellcheck source=/dev/null
. "${E2B_ENV}"
set +a
test -n "${E2B_API_KEY:-}" || { echo "E2B_API_KEY is unset" >&2; exit 1; }
export PYTHONPATH="${TMAX_SOURCE}:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT}"
cp "${SERVER_MANIFEST}" "${OUT}/server_manifest.json"
sha256sum "${TASK_IDS}" "${SERVER_MANIFEST}" "${BASE_SUMMARY}"
curl -fsS "${API_BASE}/models" >/dev/null

"${PYTHON}" "${CODE_ROOT}/tools/x_token/run_tmax_e2b_eval.py" \
  --tasks "${TASKS_ROOT}" \
  --task-ids "${TASK_IDS}" \
  --limit 17 \
  --out "${OUT}" \
  --template-alias "${TEMPLATE_ALIAS}" \
  --run-id qwen35-4b-verified17-full17-sft-step100-run1 \
  --arm sft-step100 \
  --model hosted_vllm/qwen35-4b-glm52-verified17-sft-step100 \
  --api-base "${API_BASE}" \
  --repeats 1 \
  --concurrency 8 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-steps 64 \
  --max-tokens 16384 \
  --max-total-response-tokens 65536 \
  2>&1 | tee "${ARM_LOG}"
