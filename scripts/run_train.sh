#!/usr/bin/env bash

# Example:
# nohup bash scripts/run_train.sh configs/swin_unetr.yaml > swin.out 2>&1 &

export CUDA_VISIBLE_DEVICES=7

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_train.sh <model_config>"
  echo "Example: bash scripts/run_train.sh configs/swin_unetr.yaml"
  exit 1
fi

BASE_CONFIG=configs/base_petct_vascular.yaml
MODEL_CONFIG="$1"

python -m src.train \
  --base_config "${BASE_CONFIG}" \
  --model_config "${MODEL_CONFIG}"