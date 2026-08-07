#!/usr/bin/env bash
set -euo pipefail

XROOT=/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
export TASK_IDS="$AXO_ROOT/eval-manifests/base_headroom4_task_ids.txt"
export LIMIT=4
export SERVER_MANIFEST="$AXO_ROOT/eval-runtime/headroom4-seed1/tb2-4b-serve-manifest.json"
export API_BASE=http://127.0.0.1:8121/v1
export MODEL=hosted_vllm/qwen35-4b-glm52-headroom4-seed1-step100
export RUN_NAME=seed1-step100-run1
export ARM=headroom4-step100
export OUT_FAMILY=base-headroom4-sft
export CONCURRENCY=4

exec bash "$AXO_ROOT/scripts/run_realverified14_sft_eval.sh"
