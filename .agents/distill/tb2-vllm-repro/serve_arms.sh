#!/usr/bin/env bash
# v4: max-model-len 81920 so terminus-2's compaction call (prompt ~53.2k + 12,288 output)
# is not rejected. The AGENT envelope is unchanged: context_budget 53,240 / max_tokens 12,288.
set -euo pipefail

VLLM=/mnt/azureuser/venvs/vllm/bin/vllm
BASE=/scratch/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
MERGED=/scratch/repro-tb2/qwen35-9b-distill-v3
LOGS=/scratch/repro-tb2/logs

ENVS="PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 \
HF_HOME=/scratch/hf_cache TMPDIR=/scratch/repro-tb2/tmp"
ARGS="--max-model-len 81920 --gpu-memory-utilization 0.85 --max-num-seqs 32 --enable-prefix-caching"

tmux kill-session -t vllm-base 2>/dev/null || true
tmux kill-session -t vllm-distill 2>/dev/null || true
sleep 5

tmux new-session -d -s vllm-base \
  "CUDA_VISIBLE_DEVICES=0 $ENVS $VLLM serve $BASE \
   --served-model-name base-student --port 8000 $ARGS 2>&1 | tee $LOGS/vllm-base.log"
tmux new-session -d -s vllm-distill \
  "CUDA_VISIBLE_DEVICES=1 $ENVS $VLLM serve $MERGED \
   --served-model-name distill-student --port 8001 $ARGS 2>&1 | tee $LOGS/vllm-distill.log"
echo "relaunched"; tmux ls
