#!/bin/bash
# Box-side: train both BC ablation LoRAs sequentially on GPU 1, identical hyperparams.
# Checkpoints (optimizer state) are deleted after each run — /mnt has <2 GB free.
set -eo pipefail
export PATH=/mnt/azureuser/venvs/vllm/bin:$PATH  # ninja for Qwen3.5 GDN kernels
export HF_HOME=/mnt/azureuser/hf_cache
PY=/mnt/azureuser/venvs/wmh-distill/bin/python
cd /mnt/azureuser/wmh_distill

for arm in mined random; do
  CUDA_VISIBLE_DEVICES=1 "$PY" train_sft_box.py \
    --data "bc_${arm}.jsonl" --out "adapter_bc_${arm}" 2>&1 | tee "train_bc_${arm}.log"
  rm -rf "adapter_bc_${arm}"/checkpoint-*
  echo "ARM ${arm} DONE"
done
echo "ALL TRAINING DONE"
