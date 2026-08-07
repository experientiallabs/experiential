#!/usr/bin/env bash
set -euo pipefail

XROOT="${XROOT:-/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200}"
PYTHON="${PYTHON:-${XROOT}/runtime/venv-harbor-tb2/bin/python}"
RUN_ROOT="${RUN_ROOT:-${XROOT}/tmax/verified17-sft-step100/full17}"
BASE_ROOT="${BASE_ROOT:-${RUN_ROOT}/base-run1}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${RUN_ROOT}/sft-step100-run1}"
COMPARATOR="${COMPARATOR:-${XROOT}/axolotl-sft/scripts/compare_tmax_paired_exclusions.py}"
OUT="${OUT:-${RUN_ROOT}/paired-comparison-scored-intersection.json}"
LOG="${LOG:-${XROOT}/logs/tmax-verified17-sft-step100/full17/paired-analysis-scored-intersection.log}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
test -x "${PYTHON}" || { echo "missing Python: ${PYTHON}" >&2; exit 1; }
test -r "${COMPARATOR}" || { echo "missing comparator: ${COMPARATOR}" >&2; exit 1; }
test ! -e "${OUT}" || { echo "refusing to overwrite: ${OUT}" >&2; exit 1; }

for ((poll = 1; poll <= 960; poll++)); do
  if test -s "${BASE_ROOT}/summary.json" && test -s "${ADAPTER_ROOT}/summary.json"; then
    break
  fi
  if ((poll == 960)); then
    echo "timed out waiting for matched summaries" >&2
    exit 1
  fi
  if ((poll == 1 || poll % 10 == 0)); then
    echo "waiting for matched summaries: poll=${poll}/960"
  fi
  sleep 30
done

"${PYTHON}" "${COMPARATOR}" \
  --base "${BASE_ROOT}" \
  --adapter "${ADAPTER_ROOT}" \
  --bootstrap-samples 100000 \
  --bootstrap-seed 20260807 \
  --out "${OUT}"
sha256sum "${OUT}"
