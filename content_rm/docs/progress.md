## 总体路线
1. 数据准备
2. SFT
3. SFT 的 eval（当前下一阶段）
4. SFT 版部署
5. RL
6. eval
7. 部署

## 数据（阶段性完成）
1. 已拿到少量原始 AI 回复样本，来源为 `content_rm/data/local/raw/comment_data.jsonl`，该 raw 数据不纳入远程仓库。
2. 当前数据闭环已经跑通：原始业务 JSONL -> 可选 LLM 标注 -> 人工复核底稿 -> 后续 SFT 数据生成。
3. 当前确认的数据口径：
   - `commentState=PUBLISHED` 映射为运营审核通过，即 `pass`。
   - `commentState=HIDE` 映射为运营审核不通过，即 `reject`。
   - `audit_label` 是唯一进入训练/评测产物的审核结论标签。
4. 第一版 LLM 预标注只要求输出命中的 rubrics、简短 reasoning、最终 `decision: pass|reject`。
5. `prepare_dataset.py` 已支持可选调用公司内 LLM 接口；默认生成 `human_review.jsonl` 和 `summary.json`，传入 `--call-llm` 后额外生成 `llm_annotations.jsonl` 并预填人工复核文件。
6. `second_pass_review.py` 已支持筛选 `llm_decision != audit_label` 的样本并做二次复核，输出 `second_pass_annotations.jsonl` 和 `second_pass_summary.json`。
7. `review_calibration.html` 用于人工校准 `violated_rubrics`、`human_reasoning`、`review_note`；`review_note` 只是内部备注，不进入 SFT target。
8. `normalized_samples.jsonl` 仅作为 `--write-normalized` 调试产物，旧版已生成的调试/中间文件已清理。
9. 目录已重构为 `content_rm/data/`、`content_rm/sft/`、`content_rm/rl/`、`content_rm/docs/`；本地数据统一放入 ignored 的 `content_rm/data/local/`。
10. 可选数据增强方向：借鉴 Autodata / Agentic Self-Instruct 思路，用 agent 生成金融内容审核候选样本，再通过 weak solver、strong solver/judge、规则校验和人工抽检筛选；第一阶段只作为 SFT 训练数据补充，不替代真实人工复核数据，测试集优先保留真实样本。

## SFT（阶段性完成）
1. SFT 数据不能只有最终通过/不通过标签；如果希望模型学习按 rubrics 推理，训练 target 中需要包含 reasoning 和 final decision。
2. SFT 数据生成口径已确定：最终 `decision` 使用运营 `audit_label`，LLM 只提供 reasoning 候选，人工复核后再进入训练。
3. SFT 训练入口使用 OpenRLHF，本地 JSONL 字段为 `context_messages` 和 `response`。
4. `build_sft_dataset.py` 已改为直接从 `human_review.jsonl`、`rubrics.md` 和可选 `llm_annotations.jsonl` 生成最终 SFT 数据，不再依赖 `sft_draft.jsonl`。
5. `build_sft_dataset.py` 现在按训练后端输出数据：默认生成 `sft/openrlhf/train.jsonl`、`sft/openrlhf/test.jsonl`；传入 `--write-llamafactory-alpaca` 时只生成 `sft/llamafactory_alpaca/train.json`、`test.json`、`dataset_info.json`；传入 `--write-post-train-platform` 时只生成 `sft/post-train-platform/all.jsonl`。
6. OpenRLHF 与 Ascend/LLaMA-Factory 是两条独立 SFT 训练路线；OpenRLHF 使用 `content_rm/sft/openrlhf/train_sft_qwen2_5_7b.sh`。
7. 已新增 Ascend 910B / LLaMA-Factory LoRA SFT 入口：`content_rm/sft/llamafactory/train_ascend_lora.sh`，默认 Qwen3-8B、LoRA、`qwen3_nothink`、`ASCEND_RT_VISIBLE_DEVICES=0`，实际模型/数据/输出路径通过环境变量覆盖。
8. 已在后训练平台使用 1552 条数据完成两版 Qwen3-8B LoRA SFT 训练。
9. 平台指标观察：
   - `qwen3-8B-20260701-SFT`：ROUGE-1 54.88%、ROUGE-2 35.38%、ROUGE-L 44.64%、BLEU-4 38.10%。
   - `QWEN3-8B-20260701_001-SFT`：ROUGE-1 60.19%、ROUGE-2 41.22%、ROUGE-L 49.35%、BLEU-4 43.59%。
10. 第二版平台文本指标和 loss 曲线更好，建议作为当前候选模型进入下一阶段评估；但 ROUGE/BLEU 不能替代审核业务指标。

## SFT 的 eval（当前下一阶段）
1. 当前尚未实现正式评估脚本，也没有冻结的独立评估集。
2. 评估阶段目标：验证模型是否真的学会审核任务，而不是只验证生成文本和参考答案的相似度。
3. 建议优先采用离线预测文件模式：
   - 固定 gold：`content_rm/data/local/eval/sft_eval_set.jsonl`
   - 固定预测：`content_rm/data/local/eval/predictions/<run_id>.jsonl`
   - 固定报告：`content_rm/data/local/eval/reports/<run_id>/`
4. 核心指标：
   - JSON 可解析率、schema 合法率、`decision` 合法率。
   - overall accuracy、pass/reject precision、recall、F1。
   - false pass 数量和比例，这是审核任务中的高优先级风险。
   - rubric exact match、rubric micro/macro F1。
   - reasoning 非空率和人工抽检错误包。
5. 下一步具体实现：
   - 新增 `content_rm/eval/`。
   - 新增评估集构建脚本和预测打分脚本。
   - 先用 5-10 条手工预测做 smoke test。
   - 再比较两版 SFT 模型的业务指标。

## SFT 版部署（未开始）

## RL（未开始）

## eval（未开始）

## 部署（未开始）
