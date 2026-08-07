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
PHASE="${PHASE:-canary2}"
LIMIT="${LIMIT:-2}"
REPEATS="${REPEATS:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
OUT_ROOT="${OUT_ROOT:-${XROOT}/tmax/verified17-sft-step100/${PHASE}}"
LOG_ROOT="${LOG_ROOT:-${XROOT}/logs/tmax-verified17-sft-step100/${PHASE}}"

for required in "${PYTHON}" "${E2B_ENV}" "${TASK_IDS}" "${SERVER_MANIFEST}"; do
  test -r "${required}" || { echo "missing required file: ${required}" >&2; exit 1; }
done
test -d "${TMAX_SOURCE}" || { echo "missing TMax source: ${TMAX_SOURCE}" >&2; exit 1; }
test -d "${TASKS_ROOT}" || { echo "missing task root: ${TASKS_ROOT}" >&2; exit 1; }

set -a
# shellcheck source=/dev/null
. "${E2B_ENV}"
set +a
test -n "${E2B_API_KEY:-}" || { echo "E2B_API_KEY is unset" >&2; exit 1; }

export PYTHONPATH="${TMAX_SOURCE}:${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
sha256sum "${TASK_IDS}" "${SERVER_MANIFEST}"
"${PYTHON}" -c 'import e2b, harbor, nemo_rl; print("verified17 eval import gate passed")'
curl -fsS "${API_BASE}/models" >/dev/null

arms=(base sft-step100)
models=(
  hosted_vllm/qwen35-4b-base
  hosted_vllm/qwen35-4b-glm52-verified17-sft-step100
)

for index in "${!arms[@]}"; do
  arm="${arms[$index]}"
  model="${models[$index]}"
  out="${OUT_ROOT}/${arm}-run1"
  log="${LOG_ROOT}/${arm}-run1.log"
  test ! -e "${out}" || { echo "refusing to overwrite: ${out}" >&2; exit 1; }
  mkdir -p "${out}"
  cp "${SERVER_MANIFEST}" "${out}/server_manifest.json"
  "${PYTHON}" "${CODE_ROOT}/tools/x_token/run_tmax_e2b_eval.py" \
    --tasks "${TASKS_ROOT}" \
    --task-ids "${TASK_IDS}" \
    --limit "${LIMIT}" \
    --out "${out}" \
    --template-alias "${TEMPLATE_ALIAS}" \
    --run-id "qwen35-4b-verified17-${PHASE}-${arm}-run1" \
    --arm "${arm}" \
    --model "${model}" \
    --api-base "${API_BASE}" \
    --repeats "${REPEATS}" \
    --concurrency "${CONCURRENCY}" \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-steps 64 \
    --max-tokens 16384 \
    --max-total-response-tokens 65536 \
    2>&1 | tee "${log}"
done

echo "matched evaluation complete: ${OUT_ROOT}"
