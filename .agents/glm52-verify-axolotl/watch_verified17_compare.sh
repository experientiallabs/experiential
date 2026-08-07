#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
CODE_ROOT="${CODE_ROOT:-${XROOT}/eval-code-b0d0568}"
PYTHON="${PYTHON:-${XROOT}/runtime/venv-harbor-tb2/bin/python}"
PHASE="${PHASE:-full17}"
RUN_ROOT="${RUN_ROOT:-${XROOT}/tmax/verified17-sft-step100/${PHASE}}"
BASE_ROOT="${BASE_ROOT:-${RUN_ROOT}/base-run1}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${RUN_ROOT}/sft-step100-run1}"
OUT="${OUT:-${RUN_ROOT}/paired-comparison.json}"
LOG="${LOG:-${XROOT}/logs/tmax-verified17-sft-step100/${PHASE}/paired-analysis.log}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_POLLS="${MAX_POLLS:-960}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

test -x "${PYTHON}" || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
test -r "${CODE_ROOT}/tools/x_token/compare_tmax_paired.py" || {
  echo "missing paired comparator under ${CODE_ROOT}" >&2
  exit 1
}
test ! -e "${OUT}" || { echo "refusing to overwrite: ${OUT}" >&2; exit 1; }

for ((poll = 1; poll <= MAX_POLLS; poll++)); do
  if test -s "${BASE_ROOT}/summary.json" && test -s "${ADAPTER_ROOT}/summary.json"; then
    break
  fi
  if ((poll == MAX_POLLS)); then
    echo "timed out waiting for complete matched arms under ${RUN_ROOT}" >&2
    exit 1
  fi
  if ((poll == 1 || poll % 10 == 0)); then
    echo "waiting for complete matched arms: poll=${poll}/${MAX_POLLS}"
  fi
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" "${CODE_ROOT}/tools/x_token/compare_tmax_paired.py" \
  --base "${BASE_ROOT}" \
  --adapter "${ADAPTER_ROOT}" \
  --bootstrap-samples 100000 \
  --bootstrap-seed 20260807 \
  --out "${OUT}"
sha256sum "${OUT}"
