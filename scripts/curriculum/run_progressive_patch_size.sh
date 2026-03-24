#!/usr/bin/env bash

# Example:
# nohup bash scripts/curriculum/run_progressive_patch_size.sh configs/swin_unetr.yaml configs/curriculum/progressive_patch_size.yaml > out/swin_progressive_patch_size.out 2>&1 &

export CUDA_VISIBLE_DEVICES=2

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/curriculum/run_progressive_patch.sh <model_config> <curriculum_config>"
  echo "Example: bash scripts/curriculum/run_progressive_patch.sh configs/swin_unetr.yaml configs/curriculum/progressive_patch.yaml"
  exit 1
fi

BASE_CONFIG=/data/pchatzi/Comparative-Vascular/configs/base_petct_vascular.yaml
MODEL_CONFIG="$1"
CURRICULUM_CONFIG="$2"

python -m src.trainer_variants.train_curriculum_progressive_patch_size \
  --base_config "${BASE_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --curriculum_config "${CURRICULUM_CONFIG}"