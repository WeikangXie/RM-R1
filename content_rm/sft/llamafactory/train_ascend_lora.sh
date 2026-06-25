#!/usr/bin/env bash
set -euo pipefail

# Ascend 910B / LLaMA-Factory LoRA SFT launcher for the content RM task.
# Run this after generating llamafactory_alpaca data with build_sft_dataset.py.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "${ASCEND_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ENV}"
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/content_rm/data/local/sft/llamafactory_alpaca}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/content_rm/data/local/checkpoints/qwen3-8b-lora-sft-ascend}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${SCRIPT_DIR}/qwen3_8b_lora_sft_ascend.yaml.template}"
RENDERED_CONFIG="${RENDERED_CONFIG:-${OUTPUT_DIR}/qwen3_8b_lora_sft_ascend.rendered.yaml}"

CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen3_nothink}"
CUTOFF_LEN="${CUTOFF_LEN:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"

LORA_RANK="${LORA_RANK:-8}"
LORA_TARGET="${LORA_TARGET:-all}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-50}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-true}"
REPORT_TO="${REPORT_TO:-none}"

export MODEL_PATH DATASET_DIR OUTPUT_DIR CHAT_TEMPLATE CUTOFF_LEN MAX_SAMPLES PREPROCESSING_NUM_WORKERS
export LORA_RANK LORA_TARGET PER_DEVICE_TRAIN_BATCH_SIZE PER_DEVICE_EVAL_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS LEARNING_RATE NUM_TRAIN_EPOCHS LR_SCHEDULER_TYPE WARMUP_RATIO
export LOGGING_STEPS SAVE_STEPS EVAL_STEPS BF16 GRADIENT_CHECKPOINTING FLASH_ATTN
export SAVE_ONLY_MODEL OVERWRITE_OUTPUT_DIR REPORT_TO

mkdir -p "${OUTPUT_DIR}"

python3 - "${CONFIG_TEMPLATE}" "${RENDERED_CONFIG}" <<'PY'
import os
import string
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
rendered_path = Path(sys.argv[2])
template = string.Template(template_path.read_text(encoding="utf-8"))

try:
    rendered = template.substitute(os.environ)
except KeyError as exc:
    raise SystemExit(f"Missing environment variable for config template: {exc}") from exc

rendered_path.parent.mkdir(parents=True, exist_ok=True)
rendered_path.write_text(rendered, encoding="utf-8")
print(f"Rendered LLaMA-Factory config: {rendered_path}")
PY

if [[ "${RENDER_ONLY:-0}" == "1" ]]; then
  echo "RENDER_ONLY=1 set; skipping Ascend and LLaMA-Factory runtime checks."
  exit 0
fi

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "Missing ${DATASET_DIR}/dataset_info.json. Generate LLaMA-Factory data first." >&2
  exit 1
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "llamafactory-cli not found. Install or activate the LLaMA-Factory NPU environment first." >&2
  exit 1
fi

python3 - <<'PY'
import sys

try:
    import torch
    import torch_npu  # noqa: F401
except Exception as exc:
    print(f"Failed to import torch/torch_npu: {exc}", file=sys.stderr)
    sys.exit(1)

if not hasattr(torch, "npu") or not torch.npu.is_available():
    print("torch.npu.is_available() is false. Check Ascend driver, CANN, torch_npu, and ASCEND_RT_VISIBLE_DEVICES.", file=sys.stderr)
    sys.exit(1)

print(f"torch_npu ready; visible NPU count: {torch.npu.device_count()}")
PY

llamafactory-cli train "${RENDERED_CONFIG}"
