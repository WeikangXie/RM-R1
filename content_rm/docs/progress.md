## 总体路线
1. 数据准备
2. SFT
3. SFT 的 eval
4. SFT 版部署
5. RL
6. eval
7. 部署

## 数据（阶段性完成）
1. 已拿到少量原始 AI 回复样本，来源为 `content_rm/data/local/raw/comment_data.jsonl`，该 raw 数据不纳入远程仓库。
2. 当前数据阶段先跑通闭环：原始业务 JSONL -> 可选 LLM 标注 -> 人工复核底稿 -> 后续 SFT 数据生成。
3. 当前确认的数据口径：
   - `commentState=PUBLISHED` 映射为运营审核通过，即 `pass`。
   - `commentState=HIDE` 映射为运营审核不通过，即 `reject`。
   - `audit_label` 是唯一进入训练/评测产物的审核结论标签。
4. 第一版 LLM 预标注只要求输出命中的 rubrics、简短 reasoning、最终 `decision: pass|reject`。
5. `prepare_dataset.py` 已支持可选调用公司内 LLM 接口；默认生成 `human_review.jsonl` 和 `summary.json`，传入 `--call-llm` 后额外生成 `llm_annotations.jsonl` 并预填人工复核文件。
6. `normalized_samples.jsonl` 仅作为 `--write-normalized` 调试产物，旧版已生成的调试/中间文件已清理。
7. 当前样本量用于代码开发和流程 smoke test；后续做稳定评测指标时，计划扩展到约 200 条左右，并尽量平衡通过/不通过样本。
8. 目录已重构为 `content_rm/data/`、`content_rm/sft/`、`content_rm/rl/`、`content_rm/docs/`；本地数据统一放入 ignored 的 `content_rm/data/local/`。

## SFT（进行中）
1. SFT 数据不能只有最终通过/不通过标签；如果希望模型学习按 rubrics 推理，训练 target 中需要包含 reasoning 和 final decision。
2. 已规划 SFT 数据生成口径：最终 `decision` 使用运营 `audit_label`，LLM 只提供 reasoning 候选，人工复核后再进入训练。
3. SFT 训练入口使用 OpenRLHF，本地 JSONL 字段为 `context_messages` 和 `response`。
4. `build_sft_dataset.py` 已改为直接从 `human_review.jsonl`、`rubrics.md` 和可选 `llm_annotations.jsonl` 生成最终 SFT 数据，不再依赖 `sft_draft.jsonl`。
5. `build_sft_dataset.py` 现在按训练后端输出数据：默认生成 `sft/openrlhf/train.jsonl`、`sft/openrlhf/test.jsonl`；传入 `--write-llamafactory-alpaca` 时只生成 `sft/llamafactory_alpaca/train.json`、`test.json`、`dataset_info.json`。
6. OpenRLHF 与 Ascend/LLaMA-Factory 是两条独立 SFT 训练路线；OpenRLHF 使用 `content_rm/sft/openrlhf/train_sft_qwen2_5_7b.sh`。
7. 已新增 Ascend 910B / LLaMA-Factory LoRA SFT 入口：`content_rm/sft/llamafactory/train_ascend_lora.sh`，默认 Qwen3-8B、LoRA、`qwen3_nothink`、`ASCEND_RT_VISIBLE_DEVICES=0`，实际模型/数据/输出路径通过环境变量覆盖。
8. 下一步需要等待 LLM 标注和人工复核结果，再生成正式 SFT 数据；在公司 Ascend 环境上先用 200 条数据跑通 5 step smoke test，再扩大到 1000-2000 条正式训练。

## SFT 的 eval（未开始）

## SFT 版部署（未开始）

## RL（未开始）

## eval（未开始）

## 部署（未开始）
