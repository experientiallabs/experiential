#!/usr/bin/env bash
set -euo pipefail

XROOT=/mnt/datasets/scratch/xtoken/tb2-qwen35-4b-glm52-step200
export HERE="$XROOT/eval-code-b0d0568/xtoken-ops/tblite-eval-9b"
export EVAL_ROOT="$XROOT/headroom4-sft-tblite-seed2-step100-canary10-run2"
export STORAGE_ROOT="$XROOT"
export HARBOR_VENV="$XROOT/runtime/venv-harbor-tb2"
export DOCKER_CONFIG="$XROOT/runtime/docker-config"
export BASE_SNAP="$XROOT/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export PORT=8122
export HOST_ADDR=10.0.0.4
export API_BASE_LOCAL=http://127.0.0.1:8122/v1
export API_BASE_AGENT=http://10.0.0.4:8122/v1
export CUDA_DEVICES=1
export TP=1
export DP_SIZE=1
export MAXLEN=65536
export N_RUNS=1
export N_TASKS=
export TASK_NAMES=hydra-debug-slurm-mode,reverse-engineer-stack-vm,vimscript-vim-quine,anomaly-detection-ranking,api-endpoint-permission-canonicalizer,raft-log-repair-concurrent-access,ekf-localization,competitive-programming-solver,sign-vector-game,malicious-package-forensics
export CONCURRENCY=16
export SEED_BASE=1
export EXPECTED_DOCKER_ROOT_PREFIX=/mnt/cam/
export BASE_MODEL=qwen35-4b-base
export JOB_PREFIX=qwen35-4b-headroom4-seed2-step100-canary10-r2
export ADAPTER_ARM=headroom4-seed2-step100
export ADAPTER_MODEL=qwen35-4b-glm52-headroom4-seed2-step100

WRAPPER="$XROOT/axolotl-sft/scripts/run_tblite_canary_pair.sh"
LOG="$XROOT/logs/tblite-headroom4-seed2-step100-canary10.r2.log"
mkdir -p "$EVAL_ROOT/runtime" "$XROOT/logs"
exec sg docker -c "bash '$WRAPPER' 2>&1 | tee '$LOG'"
