#!/usr/bin/env bash
set -euo pipefail

XROOT=/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200
AXO_ROOT="$XROOT/axolotl-sft"
export CONFIG="$AXO_ROOT/configs/qwen35_4b_base_headroom4_sft.yaml"
export DATASET="$AXO_ROOT/sft-data/base-headroom4.qwen-materialized.jsonl"
export MANIFEST="$AXO_ROOT/sft-data/base-headroom4.manifest.json"
export GPU_INDEX=0
export SEED=20260807
export RUN_SUFFIX=run1
export OUTPUT="$AXO_ROOT/checkpoints/qwen35-4b-glm52-base-headroom4-sft-lr1e5-r64-run1"
export LOG="$AXO_ROOT/logs/qwen35-4b-glm52-base-headroom4-sft-lr1e5-r64-run1.log"
export WANDB_NAME=qwen35-4b-glm52-base-headroom4-sft-lr1e5-r64-run1
export PREPARED_CACHE="$AXO_ROOT/cache/base-headroom4-run1"

exec bash "$AXO_ROOT/scripts/run_realverified14_sft.sh"
