#!/usr/bin/env bash

# Example:
# nohup bash scripts/curriculum/run_binary_to_multi.sh configs/swin_unetr.yaml configs/curriculum/binary_to_multi.yaml > out/swin_binary_to_multi.out 2>&1 &

export CUDA_VISIBLE_DEVICES=2

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/run_train.sh <model_config> <curriculum_config>"
  echo "Example: bash scripts/run_train.sh configs/swin_unetr.yaml configs/curriculum/binary_to_multi.yaml"
  exit 1
fi

BASE_CONFIG=/data/pchatzi/Comparative-Vascular/configs/base_petct_vascular.yaml
MODEL_CONFIG="$1"
CURRICULUM_CONFIG="$2"

python -m src.trainer_variants.train_curriculum_binary_to_multi \
  --base_config "${BASE_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --curriculum_config "${CURRICULUM_CONFIG}"