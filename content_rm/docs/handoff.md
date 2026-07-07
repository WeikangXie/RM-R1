# Content RM Handoff

更新时间：2026-07-07

## 当前状态

- 当前分支：`codex/post-train-platform-sft`
- 当前阶段：SFT 数据与训练链路已跑通，下一阶段进入 SFT 模型评估。
- 工作区状态：本地业务数据、训练数据、checkpoint 都应留在 `content_rm/data/local/`，不提交到仓库。
- 主要入口：
  - 数据准备：`content_rm/data/prepare_dataset.py`
  - 二次复核：`content_rm/data/second_pass_review.py`
  - 人工校准：`content_rm/data/review_calibration.html`
  - SFT 数据构建：`content_rm/data/build_sft_dataset.py`
  - OpenRLHF 训练：`content_rm/sft/openrlhf/train_sft_qwen2_5_7b.sh`
  - LLaMA-Factory/Ascend 训练：`content_rm/sft/llamafactory/train_ascend_lora.sh`

## 已完成事项

### 数据阶段

- 已确认业务标签口径：
  - `commentState=PUBLISHED` -> `audit_label=pass`
  - `commentState=HIDE` -> `audit_label=reject`
  - `audit_label` 是训练和评估中的最终业务标签。
- 已实现 raw -> human review 的数据准备流程。
- 已实现可选公司内 LLM 一轮标注，LLM 调用参数集中在 `content_rm/data/config.py`。
- 已实现针对 `llm_decision != audit_label` 样本的二次复核流程。
- 已实现轻量人工校准页面，用于修正 `violated_rubrics`、`human_reasoning`、`review_note`。
- 已约定 `review_note` 只作为内部备注，不进入 SFT target。

### SFT 阶段

- SFT target 格式已确定为：

```json
{
  "violated_rubrics": ["命中的规则名称"],
  "reasoning": "引用具体内容依据的解释",
  "decision": "pass 或 reject"
}
```

- `decision` 始终使用运营 `audit_label`。
- `reasoning` 优先使用人工校准后的 `human_reasoning`，缺失时可回退到 LLM reasoning。
- `build_sft_dataset.py` 已支持三种输出：
  - OpenRLHF：`content_rm/data/local/sft/openrlhf/train.jsonl`、`test.jsonl`
  - LLaMA-Factory Alpaca：`content_rm/data/local/sft/llamafactory_alpaca/train.json`、`test.json`
  - 后训练平台：`content_rm/data/local/sft/post-train-platform/all.jsonl`

### 训练观察

已在后训练平台完成两版 Qwen3-8B LoRA SFT 训练，训练数据规模为 1552 条。平台指标仅作为参考，不能替代业务评估。

| 版本 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| qwen3-8B-20260701-SFT | 54.88% | 35.38% | 44.64% | 38.10% | loss 下降但训练强度偏弱 |
| QWEN3-8B-20260701_001-SFT | 60.19% | 41.22% | 49.35% | 43.59% | 全部文本指标更好，loss 曲线更充分 |

当前候选模型建议优先评估第二版 `QWEN3-8B-20260701_001-SFT`，但最终结论必须以审核任务指标为准。

## 关键决策

- SFT 学习的是“运营审核标签 + rubrics 下可解释 reasoning”，不是单纯文本相似度。
- 一轮 LLM 标注只作为候选解释，不覆盖运营标签。
- 二次复核用于解释运营标签是否能被 rubrics 支撑；`status=need_review` 的样本不能自动入训。
- `violated_rubrics`、`reasoning`、`decision` 必须三元组一致：
  - `decision=reject` 时通常需要非空违规规则和拒绝理由。
  - `decision=pass` 时通常不应保留违规命中。
- 当前仓库没有已提交的自动合并 `second_pass_annotations` 到 `human_review` 的 CLI；若下一阶段需要自动合并，应先补脚本并记录命令。

## 下一阶段：SFT Eval

### 目标

评估 SFT 模型是否真的学会审核任务，而不是只看 ROUGE/BLEU：

- 输出是否稳定为合法 JSON。
- `decision` 是否与运营 `audit_label` 一致。
- `reject` 召回率是否足够，尤其关注 false pass。
- `violated_rubrics` 是否命中正确规则。
- `reasoning` 是否引用具体内容，且能解释最终 decision。

### 建议 v1 评估方式

先采用离线预测文件模式：

1. 准备冻结评估集：`content_rm/data/local/eval/sft_eval_set.jsonl`
2. 用后训练平台或部署服务跑模型，导出预测：`content_rm/data/local/eval/predictions/<run_id>.jsonl`
3. 编写评估脚本读取 gold + prediction，输出：`content_rm/data/local/eval/reports/<run_id>/`

预测文件建议字段：

```json
{
  "sample_id": "样本 ID",
  "model": "模型版本名",
  "response": "模型原始输出文本"
}
```

### 核心指标

- JSON 可解析率
- schema 合法率
- `decision` 合法率
- overall accuracy
- pass/reject precision、recall、F1
- false pass 数量和比例
- rubric exact match
- rubric micro/macro F1
- reasoning 非空率
- 错误样本包：`invalid_json`、`invalid_schema`、`false_pass`、`false_reject`、`rubric_mismatch`

### 立即可做的任务

1. 新增 `content_rm/eval/` 目录和 README。
2. 新增 `build_eval_set.py`，从校准后的 review 数据生成冻结评估集。
3. 新增 `score_predictions.py`，消费预测 JSONL 并输出 metrics/report。
4. 先用 5-10 条手写预测样本做 smoke test。
5. 再用两版 SFT 模型分别导出预测并比较，优先关注 false pass 和 reject recall。

