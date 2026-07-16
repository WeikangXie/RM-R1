# Content RM Handoff

更新时间：2026-07-16

## 当前状态

- 当前分支：`codex/post-train-platform-sft`
- 当前阶段：28,005 条业务数据的一次标注和全量二次标注均已完成；下一步是在不做全量人工复核的前提下构建 SFT 数据，核对实际可用量与分布，再训练新一版模型。
- 当前代码链路已实现并通过测试，但真实业务数据的运行结果来自公司电脑，不能根据本机样本重新推断。
- 本机 `content_rm/data/local/` 只有开发样本和部分旧产物，其中当前 `comment_data.jsonl` 仅 37 条；真实业务数据、标注结果、训练数据和 checkpoint 均只保存在公司电脑，不得上传 GitHub。
- 当前测试基线：`25 passed`。

主要入口：

- 原始数据合并：`content_rm/data/combine_raw_data.py`
- 一次标注：`content_rm/data/prepare_dataset.py`
- 二次复核：`content_rm/data/second_pass_review.py`
- 人工校准：`content_rm/data/review_calibration.html`
- SFT 数据构建：`content_rm/data/build_sft_dataset.py`
- 异步平台基础设施：`content_rm/infrastructure/`

## 公司电脑上的真实数据进度

### 原始数据与 comment_data

- 原始帖子：28,094 条。
- 成功合并为 `comment_data`：28,005 条。
- 无法关联原帖等脏数据：22 条。
- 涉及产品：490 个。
- 运营标签由 `commentState` 映射：`PUBLISHED -> pass`，`HIDE -> reject`。

| 维度 | pass | reject | 合计 |
| --- | ---: | ---: | ---: |
| 全部数据 | 12,808 | 15,197 | 28,005 |
| AI_CHECK_IN | 9,319 | 10,662 | 19,981 |
| AI_OPINION_SHARE | 3,397 | 4,406 | 7,803 |
| AI_PRODUCT_RULE | 92 | 129 | 221 |

`AI_PRODUCT_RULE` 数量少，但覆盖交易规则、产品费用、基金表现等重要场景，当前决定保留，并在构建和评估时单独统计。

### 一次标注

- 模型：`qwen3-32b-mx`。
- run 目录：`data/local/review/first-pass-20260713-203101/`。
- 初次 collect：25,265 条成功、2,740 条失败。失败几乎都发生在模型输出 `<think>` 后 JSON 被 `max_tokens` 截断。
- 将一次/二次标注默认 `max_tokens` 提高到 2048，并支持 retry 使用新的单任务参数后，恢复到 27,991 条成功、14 条失败。
- 最后 14 条是 rubric 名称的近义写法或格式差异；在 `rubric_utils.py` 增加集中别名规范化并重新 collect 后，28,005 条全部成功。
- 最终一次标注 decision：pass 22,878 条，reject 5,127 条；pass 占 81.7%，明显高于运营 pass 的 45.7%。
- 与运营标签一致：14,891 条；不一致：13,114 条。
- 分歧方向（`一次 decision -> audit_label`）：
  - `pass -> reject`：11,592 条。
  - `reject -> pass`：1,522 条。

### 全量二次标注

- 二次 run 目录：`data/local/review/first-pass-20260713-203101/second-pass-full-20260715-190436/`。
- 输入：一次标注的全部 13,114 条分歧记录。
- 平台/解析层技术成功 13,113 条，失败 1 条。
- 技术成功不等于可入训：
  - `annotation.status=ok`：1,736 条，可作为合法二次复核候选。
  - `annotation.status=need_review`：11,377 条，不能自动进入 SFT。
- 输出：
  - `second_pass_annotations.jsonl`
  - `second_pass_summary.json`

二次标注允许模型独立推翻运营候选标签。模型 decision 与运营 `audit_label` 不一致，或 decision 与 rubrics 自相矛盾时，记录为 `need_review`，而不是技术失败。

### 当前理论可用 SFT 规模

在没有人工复核、且 SFT 构建阶段未发现其他结构问题的前提下：

- 一次标注与运营一致：14,891 条。
- 合法二次复核：1,736 条。
- 理论自动可用上限：16,627 条，占原始 28,005 条的约 59.4%。
- 尚未解决：11,377 条 `need_review` 加 1 条二次技术失败。

这只是根据标注 summary 推导的上限。最终数量必须以 `build_sft_dataset.py` 输出的 `summary.json` 为准，因为构建器还会检查未知 rubric、decision/rubrics 一致性、重复或孤立 `comment_id` 等问题。

## 已实现的数据与训练契约

- `comment_id` 是全链路唯一业务主键，并直接作为异步 API 的 `custom_id`；业务产物不再包含 `sample_id`。
- `llm_annotations.jsonl` 是包含上下文的一次标注主数据。
- `human_review.jsonl` 是可选的稀疏人工覆盖，只在校准页面点击“保存当前”后产生记录。
- 同步和可恢复异步流程共用同一数据契约；异步支持 `plan/submit/status/collect/retry-failed`、分页恢复、原子 manifest 和失败项重试。
- 异步平台代码已移入 `content_rm/infrastructure/`，不依赖数据处理实现；`data` 脚本通过稳定的本地模块导入使用它。
- 二次模型不读取一次标注结论，只审核原始上下文；运营标签作为候选结论，最终一致性由程序校验。
- 二次 reasoning 提示词要求直接陈述文本证据和 rubric 判断，避免“运营的 reject 合理”之类流程化解释。

SFT 监督来源优先级：

1. 存在合法稀疏人工复核时使用人工结果。
2. 一次标注与运营标签不一致时，只接受 `status=ok`、decision 等于运营标签且 rubrics 一致的二次复核。
3. 一次标注与运营标签一致且结构合法时使用一次标注。
4. 失败、`need_review`、未解决分歧或结构矛盾的记录跳过。

SFT target 固定为：

```json
{
  "violated_rubrics": ["命中的规则名称"],
  "reasoning": "引用具体内容依据的审核判断",
  "decision": "pass 或 reject"
}
```

最终 `decision` 仍使用运营 `audit_label`。当前流程不会自动把运营噪声重标为另一个 decision；无法被 rubrics 支撑的分歧会留在 `need_review`，等待人工复核或被跳过。

## 下一步操作

从公司电脑的 `content_rm` 目录执行，不传 `--human-review`，先生成后训练平台格式：

```powershell
uv run python data/build_sft_dataset.py `
  --llm-annotations data/local/review/first-pass-20260713-203101/llm_annotations.jsonl `
  --second-pass-annotations data/local/review/first-pass-20260713-203101/second-pass-full-20260715-190436/second_pass_annotations.jsonl `
  --rubrics data/rubrics.md `
  --output-dir data/local/sft/first-pass-20260713-203101 `
  --write-post-train-platform
```

首先保存并检查 `data/local/sft/first-pass-20260713-203101/post-train-platform/summary.json`，重点关注：

- `usable_sft_rows` 是否接近但不超过 16,627。
- `annotation_source_counts` 中 `first_pass`、`second_pass` 与各类跳过原因。
- 最终 `label_counts`，尤其是 reject 是否因模型 pass 倾向而大量流失。
- 分别按 `extend_type` 和 rubric 补充统计，特别检查只有 221 条原始数据的 `AI_PRODUCT_RULE`。

确认数据质量后，再用新数据训练一版 SFT。旧模型不再作为候选。

## 后续需要解决的问题

1. **运营标签噪声或隐含规则**：已有明显无违规内容却被运营 reject 的案例。需要从 `pass -> reject`、`reject -> pass` 和 `need_review` 中分层抽样，区分运营个人倾向、rubrics 缺失与模型漏判。
2. **模型 pass 倾向**：一次标注 pass 率 81.7%，且 88.4% 的分歧是 `一次 pass -> 运营 reject`。应关注新 SFT 数据的 reject 数量、false pass 和 reject recall。
3. **二次复核不是独立裁判**：一次和二次都使用 `qwen3-32b-mx`，错误具有相关性；二次 `status=ok` 只能作为自动筛选信号，不能代替独立人工金标。
4. **Rubrics 边界仍不完整**：重点包括一般性知识与个性化建议、客观费用说明与诱导交易、模板化或低价值回复、风险提示充分性、以及事实准确性如何依赖企业知识库。
5. **AI_PRODUCT_RULE 稀缺**：仅 221 条，暂不删除；训练后需单独评估，并考虑定向补充真实数据。
6. **事实核验**：产品规则、费率和基金表现不能只依赖语言模型判断，后续应接入权威产品资料或知识库证据。
7. **独立评估尚未建立**：当前没有冻结的人工金标集和正式 scorer。新 SFT 训练完成后，需要按 JSON/schema 合法率、decision precision/recall/F1、false pass、rubric match 和 reasoning 质量评估。
8. **数据切分泄漏风险**：当前构建器只按 `audit_label` 分层随机切分，尚未按产品、时间、主题或 `extend_type` 隔离；正式评估集应避免相似样本跨训练和测试集。
9. **本机旧产物易误用**：`data/local/` 中可能存在旧 `sample_id` 或旧 schema 文件，只用于历史开发，不应与本次公司侧 28,005 条产物混用。

## 已弃用的旧实验

此前基于约 1,552 条数据训练的两版 Qwen3-8B LoRA，以及对应 ROUGE/BLEU 结果，现标记为已弃用的历史实验：

- 数据量过小，且来自旧数据与标注契约。
- 旧评估样本和方法不足以支持模型比较。
- 不再推荐其中任何一版作为当前候选，也不据此判断项目效果。

新一轮模型效果必须在本次扩大数据量后的训练和独立业务评估完成后重新判断。
