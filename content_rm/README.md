# Content RM

This module contains the business-domain data and training entrypoints for the financial content community reward-model project.

The original RM-R1 code is left in place. New business code is grouped here by workflow:

- `data/`: raw-data adapters, SFT dataset builders, rubrics, and ignored local artifacts.
- `sft/`: SFT launchers for OpenRLHF and Ascend/LLaMA-Factory.
- `rl/`: reserved for the later RL route.
- `docs/`: project progress and decision notes.

For the current project handoff and next-stage checklist, start with [`docs/handoff.md`](docs/handoff.md).

## Local Data

Local data is intentionally ignored by git:

```text
content_rm/data/local/
```

Current local layout:

```text
content_rm/data/local/raw/comment_data.jsonl
content_rm/data/local/review/llm_annotations.jsonl
content_rm/data/local/review/second_pass_annotations.jsonl
content_rm/data/local/review/human_review.jsonl
content_rm/data/local/review/summary.json
content_rm/data/local/review/async/
content_rm/data/local/sft/
content_rm/data/local/checkpoints/
```

## Data Commit Guidelines

- Do not commit local business data, generated datasets, checkpoints, or rendered configs. Keep them under ignored paths such as `content_rm/data/local/`.
- Generated data should be grouped by stage and target use: `content_rm/data/local/<stage>/<platform-or-purpose>/`.
- SFT datasets should use `content_rm/data/local/sft/<platform-or-purpose>/`, for example `openrlhf/`, `llamafactory_alpaca/`, or `post-train-platform/`.
- When adding a new data format, update this README with the generation command, output directory, and row-field contract in the same change.

## Current Label Mapping

- `commentState=PUBLISHED` -> `pass`
- `commentState=HIDE` -> `reject`

The input JSONL is expected to contain only items that already have operator audit results. Generated artifacts do not include `auditState`; the only training/evaluation label is `audit_label`.

## Environment

The lightweight Content RM tools are managed by the root uv project. This environment is independent from the nested OpenRLHF training project.

```bash
uv sync --dev
uv run pytest
```

## First-Pass Annotation

Every generated business record is keyed by `comment_id`. The raw company field `commentId` must be a UUID and unique within the input file; the same value is sent to the asynchronous platform as `custom_id`.

Validate the input and estimate asynchronous batches without calling the platform:

```bash
uv run python content_rm/data/prepare_dataset.py \
  --input content_rm/data/local/raw/comment_data.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/review \
  --async-action plan
```

For the synchronous endpoint, use `--call-llm`. It writes the enriched `llm_annotations.jsonl`, which contains the normalized review context and the model result; it does not create a full `human_review.jsonl`.

```bash
uv run python content_rm/data/prepare_dataset.py \
  --input content_rm/data/local/raw/comment_data.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/review \
  --call-llm
```

For thousands of rows, use the recoverable asynchronous flow. The default batch size is 500:

```bash
uv run python content_rm/data/prepare_dataset.py \
  --input content_rm/data/local/raw/comment_data.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/review \
  --async-action submit

# Re-run the same command arguments with one of these actions:
# --async-action status
# --async-action collect
# --async-action retry-failed
```

The run manifest, task snapshot, task IDs, attempts, and raw result pages are stored under `content_rm/data/local/review/async/first_pass/`. Credentials are never stored there. Both synchronous and asynchronous modes reuse `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_AUTHORIZATION` from `content_rm/data/config.py` or the environment.

`--write-normalized` optionally writes `normalized_comments.jsonl` for debugging. With no LLM mode selected, the script only validates/counts the input and writes `summary.json`.

## Second-Pass Review For Disagreements

The second pass reads `llm_annotations.jsonl` directly and selects successful rows where the first-pass `decision` differs from `audit_label`.

Dry-run the disagreement selection first:

```bash
uv run python content_rm/data/second_pass_review.py \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output content_rm/data/local/review/second_pass_annotations.jsonl \
  --dry-run
```

Submit the selected rows asynchronously, then use `status`, `collect`, and `retry-failed` with the same arguments:

```bash
uv run python content_rm/data/second_pass_review.py \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output content_rm/data/local/review/second_pass_annotations.jsonl \
  --async-action submit
```

Without `--async-action` or `--dry-run`, the script retains the synchronous call mode.

## Sparse Human Review

Open `content_rm/data/review_calibration.html` and load `llm_annotations.jsonl`. Optionally load `second_pass_annotations.jsonl`, an existing sparse `human_review.jsonl`, and `rubrics.md`. A row is added to the exported human review only after clicking **保存当前**.

Each sparse human row contains only `comment_id`, `violated_rubrics`, `reasoning`, and `review_note`. To remove an override, delete that `comment_id` row from `human_review.jsonl`.

## Build Reviewed SFT Data

`llm_annotations.jsonl` is required; second-pass and sparse human annotations are optional:

```bash
uv run python content_rm/data/build_sft_dataset.py \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --second-pass-annotations content_rm/data/local/review/second_pass_annotations.jsonl \
  --human-review content_rm/data/local/review/human_review.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/sft
```

The OpenRLHF SFT files are:

- `content_rm/data/local/sft/openrlhf/train.jsonl`
- `content_rm/data/local/sft/openrlhf/test.jsonl`
- `content_rm/data/local/sft/openrlhf/summary.json`

Each training row uses:

- `comment_id`: the company comment UUID.
- `context_messages`: model input messages.
- `response`: JSON string containing `violated_rubrics`, `reasoning`, and `decision`.
- `audit_label`: the operator pass/reject label.

Annotation precedence is sparse human review, successful `status=ok` second pass for disagreements, then an agreeing first pass. Unresolved disagreements and failed annotations without a human override are skipped. The final `decision` always uses `audit_label`.

To build LLaMA-Factory Alpaca SFT files instead, pass `--write-llamafactory-alpaca`:

```bash
uv run python content_rm/data/build_sft_dataset.py \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --second-pass-annotations content_rm/data/local/review/second_pass_annotations.jsonl \
  --human-review content_rm/data/local/review/human_review.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/sft \
  --write-llamafactory-alpaca
```

The LLaMA-Factory files are:

- `content_rm/data/local/sft/llamafactory_alpaca/train.json`
- `content_rm/data/local/sft/llamafactory_alpaca/test.json`
- `content_rm/data/local/sft/llamafactory_alpaca/dataset_info.json`
- `content_rm/data/local/sft/llamafactory_alpaca/summary.json`

To build the post-train platform single-turn JSONL file instead, pass `--write-post-train-platform`:

```bash
uv run python content_rm/data/build_sft_dataset.py \
  --llm-annotations content_rm/data/local/review/llm_annotations.jsonl \
  --second-pass-annotations content_rm/data/local/review/second_pass_annotations.jsonl \
  --human-review content_rm/data/local/review/human_review.jsonl \
  --rubrics content_rm/data/rubrics.md \
  --output-dir content_rm/data/local/sft \
  --write-post-train-platform
```

The post-train platform files are:

- `content_rm/data/local/sft/post-train-platform/all.jsonl`
- `content_rm/data/local/sft/post-train-platform/summary.json`

Each row in `all.jsonl` uses:

- `system`: system prompt from the final SFT messages.
- `prompt`: user prompt from the final SFT messages.
- `response`: unchanged JSON string containing `violated_rubrics`, `reasoning`, and `decision`.

The three output modes are exclusive for a single script run: without a format flag it writes only `openrlhf/`; with `--write-llamafactory-alpaca` it writes only `llamafactory_alpaca/`; with `--write-post-train-platform` it writes only `post-train-platform/`.

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

## Next: SFT Evaluation

The next stage is to evaluate the trained SFT model on business-task metrics. Platform ROUGE/BLEU is useful as a training signal, but the model should be selected by audit behavior:

- valid JSON/schema rate
- `decision` accuracy against `audit_label`
- pass/reject precision, recall, and F1
- false pass count and rate
- rubric match quality
- reasoning quality by sampled human review

The recommended v1 route is offline evaluation: freeze a gold eval set under `content_rm/data/local/eval/`, export model predictions from the platform or deployment service, then score the prediction JSONL with a local eval script. See [`docs/handoff.md`](docs/handoff.md) for the proposed file contracts and immediate next tasks.
