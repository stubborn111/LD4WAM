#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
NUM_GPUS=${1:-1}
PORT=${2:-27563}
export LDM_DATA_ROOT=${LDM_DATA_ROOT:-../data}
export DINOV3_MODEL_PATH=${DINOV3_MODEL_PATH:-./pretrained/dinov3-vitl16-local}
export OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=bf16 --main_process_port="${PORT}" train.py
