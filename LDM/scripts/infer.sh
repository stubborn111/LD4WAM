#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
NUM_GPUS=${1:-1}
PORT=${2:-27564}
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))
export LDM_DATA_ROOT=${LDM_DATA_ROOT:-../data}
export DINOV3_MODEL_PATH=${DINOV3_MODEL_PATH:-./pretrained/dinov3-vitl16-local}
CHECKPOINT=${CHECKPOINT:-./outputs/ldm/checkpoints/ldm_model_final.pt}
EXTRA_ARGS=("$@")
[[ -n "${DATASETS:-}" ]] && EXTRA_ARGS+=(--datasets ${DATASETS})
[[ -n "${LIMIT_EPISODES:-}" ]] && EXTRA_ARGS+=(--limit-episodes "${LIMIT_EPISODES}")
[[ -n "${VIDEO_KEYS:-}" ]] && EXTRA_ARGS+=(--video-keys ${VIDEO_KEYS})
[[ "${OVERWRITE:-0}" == "1" ]] && EXTRA_ARGS+=(--overwrite)
[[ "${SKIP_WRIST_VIEWS:-1}" == "1" ]] && EXTRA_ARGS+=(--skip-wrist-views)
accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=bf16 --main_process_port="${PORT}" tools/export_latent_dynamics.py --checkpoint "${CHECKPOINT}" --device auto "${EXTRA_ARGS[@]}"
