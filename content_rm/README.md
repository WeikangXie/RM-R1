# Content RM

This module contains the business-domain data and training entrypoints for the financial content community reward-model project.

The original RM-R1 code is left in place. New business code is grouped here by workflow:

- `data/`: raw-data adapters, SFT dataset builders, rubrics, and ignored local artifacts.
- `sft/`: SFT launchers for OpenRLHF and Ascend/LLaMA-Factory.
- `rl/`: reserved for the later RL route.
- `docs/`: project progress and decision notes.

## Local Data

Local data is intentionally ignored by git:

```text
content_rm/data/local/
```

Current local layout:

```text
content_rm/data/local/raw/comment_data.jsonl
content_rm/data/local/review/human_review.jsonl
content_rm/data/local/review/summary.json
content_rm/data/local/sft/
content_rm/data/local/checkpoints/
```

## Current Label Mapping

- `commentState=PUBLISHED` -> `pass`
- `commentState=HIDE` -> `reject`

The input JSONL is expected to contain only items that already have operator audit results. Generated artifacts do not include `auditState`; the only training/evaluation label is `audit_label`.

## Generate Data Artifacts

From the RM-R1 repository root:

```bash
python3 content_rm/data/prepare_dataset.py \
  --input content_rm/data/local/raw/comment_data.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/review
```

Generated files:

- `human_review.jsonl`: review sheet in JSONL form. Human reviewers fill reasoning and notes, not a second decision label.
- `summary.json`: counts and output paths for quick checks.

Optional debug artifact:

- `normalized_samples.jsonl`: write it with `--write-normalized` when you need to inspect field cleanup.

No `llm_annotation_tasks.jsonl`, `human_review.csv`, or `sft_draft.jsonl` is written by default. LLM tasks are built in memory when `--call-llm` is used.

## Call Company LLM

The script can optionally call the company-internal chat completion endpoint:

```bash
python3 content_rm/data/prepare_dataset.py \
  --input content_rm/data/local/raw/comment_data.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/review \
  --call-llm \
  --llm-base-url http://your-host:your-port \
  --llm-model your-model-name \
  --llm-authorization "Bearer your-token"
```

The endpoint path is built as:

```text
{llm_base_url}/llm/{llm_model}/v1/chat/completions
```

Only the necessary request fields are sent: `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `stream`, `response_format`, and optional `seed`. `response_format` defaults to `json_object`, matching the company interface definition.

When `--call-llm` is enabled, the script also writes `llm_annotations.jsonl` and pre-fills `llm_decision`, `llm_reasoning`, and `violated_rubrics` in the human review file.

## Build Reviewed SFT Data

After LLM annotation and human review, build the OpenRLHF-ready train/test files:

```bash
python3 content_rm/data/build_sft_dataset.py \
  --human-review content_rm/data/local/review/human_review.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --output-dir content_rm/data/local/sft
```

The OpenRLHF SFT files are:

- `content_rm/data/local/sft/openrlhf/train.jsonl`
- `content_rm/data/local/sft/openrlhf/test.jsonl`
- `content_rm/data/local/sft/openrlhf/summary.json`

Each training row uses:

- `context_messages`: model input messages.
- `response`: JSON string containing `violated_rubrics`, `reasoning`, and `decision`.
- `audit_label`: the operator pass/reject label.

The final `decision` always uses `audit_label`. LLM output is used as a reasoning candidate only.

To build LLaMA-Factory Alpaca SFT files instead, pass `--write-llamafactory-alpaca`:

```bash
python3 content_rm/data/build_sft_dataset.py \
  --human-review content_rm/data/local/review/human_review.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --output-dir content_rm/data/local/sft \
  --write-llamafactory-alpaca
```

The LLaMA-Factory files are:

- `content_rm/data/local/sft/llamafactory_alpaca/train.json`
- `content_rm/data/local/sft/llamafactory_alpaca/test.json`
- `content_rm/data/local/sft/llamafactory_alpaca/dataset_info.json`
- `content_rm/data/local/sft/llamafactory_alpaca/summary.json`

The two output modes are exclusive for a single script run: without the flag it writes only `openrlhf/`; with the flag it writes only `llamafactory_alpaca/`.

## Train SFT With OpenRLHF

From the RM-R1 repository root:

```bash
bash content_rm/sft/openrlhf/train_sft_qwen2_5_7b.sh
```

Useful environment overrides:

```bash
DEVICE=0 \
MODEL_PATH=/path/to/base/model \
DATASET_DIR=/path/to/content_rm/data/local/sft/openrlhf \
SAVE_PATH=/path/to/save/checkpoint \
bash content_rm/sft/openrlhf/train_sft_qwen2_5_7b.sh
```

The launcher uses OpenRLHF with:

- `--dataset json@${DATASET_DIR}`
- `--train_split train`
- `--eval_split test`
- `--apply_chat_template`
- `--input_key context_messages`
- `--output_key response`

## Train SFT On Ascend 910B

The Ascend route is independent from OpenRLHF and uses LLaMA-Factory LoRA SFT. Build LLaMA-Factory data first, then run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
MODEL_PATH=/data/models/Qwen3-8B \
DATASET_DIR=/data/content_rm/sft/llamafactory_alpaca \
OUTPUT_DIR=/data/content_rm/checkpoints/qwen3-8b-lora-sft-ascend \
bash content_rm/sft/llamafactory/train_ascend_lora.sh
```

See [`sft/llamafactory/README.md`](sft/llamafactory/README.md) for the full list of environment overrides.
