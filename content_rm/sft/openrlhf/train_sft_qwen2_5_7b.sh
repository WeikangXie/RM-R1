#!/usr/bin/env bash
set -euo pipefail

# Business-domain SFT launcher for OpenRLHF.
# Run after building reviewed SFT data with build_sft_dataset.py.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OPENRLHF_DIR="${PROJECT_ROOT}/rm_r1/OpenRLHF"

DEVICE="${DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/content_rm/data/local/sft/openrlhf}"
SAVE_PATH="${SAVE_PATH:-${PROJECT_ROOT}/content_rm/data/local/checkpoints/qwen2.5-7b-sft}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_LEN="${MAX_LEN:-4096}"
ZERO_STAGE="${ZERO_STAGE:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"

cd "${OPENRLHF_DIR}"

deepspeed --include "localhost:${DEVICE}" --module openrlhf.cli.train_sft \
  --save_path "${SAVE_PATH}" \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --pretrain "${MODEL_PATH}" \
  --bf16 \
  --max_epochs "${MAX_EPOCHS}" \
  --max_len "${MAX_LEN}" \
  --zero_stage "${ZERO_STAGE}" \
  --learning_rate "${LEARNING_RATE}" \
  --dataset "json@${DATASET_DIR}" \
  --train_split train \
  --eval_split test \
  --apply_chat_template \
  --input_key context_messages \
  --output_key response \
  --gradient_checkpointing \
  --packing_samples \
  --adam_offload
