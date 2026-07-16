# Content RM Progress

更新时间：2026-07-16

## 总体路线

1. 原始业务数据处理：已完成代码与本次公司侧执行。
2. 大规模一次标注：已完成。
3. 分歧二次标注：已完成。
4. 新 SFT 数据构建与质量分析：当前阶段。
5. 使用扩大后的数据重新训练 SFT：待开始。
6. 建立独立人工评估集并进行 SFT eval：待开始。
7. SFT 部署：未开始。
8. RL、RL eval 与部署：未开始。

这里的“已完成”需区分代码与数据：仓库本机只有 sample；28,005 条真实业务数据及其标注、训练产物只存在于公司电脑，禁止上传 GitHub。

## 已完成的工程里程碑

### 数据契约与目录重构

- `comment_id` 成为唯一业务主键，并直接作为平台 `custom_id`；新产物移除 `sample_id`。
- 原始 `commentId` 在读取时映射为 `comment_id`；生成产物使用严格 Pydantic schema。
- `llm_annotations.jsonl` 改为包含上下文的一次标注主文件。
- `human_review.jsonl` 改为只有实际人工保存记录的稀疏覆盖，不再由一次标注流程全量生成。
- SFT 构建按人工复核、合法二次复核、与运营一致的一次标注依次选择监督内容；没有人工记录并不阻止合法模型标注入训。
- 仓库初始化为轻量 uv 项目；同步与异步调用共用现有 LLM 配置，并移除无用 `response_format`。

### 异步基础设施

- 异步平台代码已集中到 `content_rm/infrastructure/`，与 `data` 业务数据处理代码解耦。
- 修复原先依赖外部 `app.infrastructure...` 的错误导入，改用仓库内模型和配置。
- 支持任务切批、提交、状态查询、分页 collect、失败重试、manifest 原子更新和中断恢复。
- retry 只提交任务失败、单条错误、缺失结果或非法 JSON/schema 的记录，并允许失败任务使用新的请求参数。
- 一次和二次标注默认 `max_tokens` 提高到 2048，以解决 Qwen thinking 内容占满输出长度的问题。

### 原始数据合并

- 新增 `combine_raw_data.py`，按白名单保留评论字段并合并帖子上下文。
- 只处理 `OPERATOR_AUDITED` 评论；找不到原帖等记录写入脏数据文件。
- 当前只输出分布统计，不做 pass/reject 或 `extendType` 采样。
- 公司侧本次运行：28,094 条帖子，成功合并 28,005 条评论，22 条脏数据，覆盖 490 个产品。
- 真实数据标签：pass 12,808、reject 15,197。
- 类型分布：`AI_CHECK_IN` 19,981、`AI_OPINION_SHARE` 7,803、`AI_PRODUCT_RULE` 221。
- `AI_PRODUCT_RULE` 决定保留，不因样本少而删除。

### 2026-07-13 至 2026-07-15：全量一次标注

- 使用 `qwen3-32b-mx` 标注公司侧 28,005 条数据。
- 初次结果为 25,265 条成功、2,740 条失败；2,717 条主要失败表现为 `finish_reason=length`，且 raw content 包含 `<think>`。
- 完成可恢复 retry 改造并提高 `max_tokens` 后，结果恢复至 27,991 条成功、14 条失败。
- 对最后 14 条 rubric 近义名称增加集中 alias 规范化后，最终 28,005 条全部成功。
- 最终一次 decision：pass 22,878、reject 5,127。
- 与运营标签一致 14,891 条，分歧 13,114 条：
  - `一次 pass -> 运营 reject`：11,592。
  - `一次 reject -> 运营 pass`：1,522。
- 一次模型 pass 率为 81.7%，运营 pass 率为 45.7%，确认存在需要重点分析的 pass 倾向。

### 2026-07-15：二次复核语义与一致性修正

- 二次 prompt 改为基于原始上下文独立审核，不向模型展示一次标注结果。
- 运营标签只作为候选结论；模型可以输出不同 decision。
- reasoning 要求直接说明原文证据、rubric 和判断，禁止用“运营的 reject 合理”等流程元话语替代审核判断。
- 模型 decision 与运营标签不一致时转为 `need_review`，不再被错误记为技术失败。
- `decision=reject` 却没有 rubrics、或 `decision=pass` 却保留 rubrics 等结构矛盾同样转为 `need_review`。
- 增加显式交易行动指引的判断提示，覆盖建议切换份额、赎回或转换等边界。
- 修复后的 30 条 smoke test 全部技术成功；观察到大量一次 `reject` 被二次改为 `pass`，但该结果同时受到模型 pass 倾向和运营标签噪声影响，不能直接作为质量结论。

### 2026-07-15：全量二次标注

- 对全部 13,114 条一次分歧记录完成异步二次标注。
- 技术成功 13,113 条，技术失败 1 条。
- 二次 annotation 状态：
  - `ok`：1,736。
  - `need_review`：11,377。
- `need_review` 是业务判断未与运营标签收敛或结构不一致，不是平台调用失败，也不能自动进入 SFT。
- 二次结果位于 `data/local/review/first-pass-20260713-203101/second-pass-full-20260715-190436/`。
- 在构建器没有发现其他结构问题的假设下，理论自动可用 SFT 上限为 14,891 + 1,736 = 16,627 条，约占 28,005 条的 59.4%；实际数量待 SFT summary 确认。

### SFT 构建与训练入口

- `build_sft_dataset.py` 已支持 OpenRLHF、LLaMA-Factory Alpaca 和公司后训练平台三种互斥输出。
- 构建器会拒绝重复或孤立 `comment_id`、未知 rubric，并统计一次、二次、人工来源及所有跳过原因。
- `status=need_review`、技术失败、未解决分歧和 decision/rubrics 不一致的记录不会自动入训。
- OpenRLHF 与 Ascend/LLaMA-Factory 训练入口已经实现，但尚未使用本次 28,005 条数据重新训练。

## 当前数据质量判断

1. 原始运营 pass/reject 数量相对均衡，但经过模型一致性筛选后，最终 SFT 很可能偏向 pass，必须以构建 summary 的 `label_counts` 验证。
2. 部分运营 reject 案例无法由现有 rubrics 或文本内容解释，可能来自个人审核倾向、随机噪声或未显式记录的内部规则。
3. 一次和二次标注都使用 `qwen3-32b-mx`，二次结果与一次结果存在相关性，不能视为独立金标。
4. `AI_PRODUCT_RULE` 只有 221 条，应保留、单独统计，并在后续定向补数和评估。
5. Rubrics 仍需澄清一般建议与个性化建议、费用说明与诱导交易、模板/低价值内容、风险提示和事实准确性的边界。
6. 涉及产品规则、费率或表现的事实判断需要企业知识库或权威产品材料支持。
7. 当前尚无冻结的独立人工金标集；随机按标签切分也可能让同产品、相似主题或相邻时间数据跨越 train/test。
8. 本机 `content_rm/data/local/` 只有 37 条当前 sample 和若干旧产物，不能作为真实进度或规模的依据。

## 当前下一步

1. 在公司电脑运行 `build_sft_dataset.py`，先不传 `human_review`，生成后训练平台数据和 summary。
2. 核对 `usable_sft_rows`、`annotation_source_counts`、`label_counts` 与全部 skip reason，确认记录数守恒。
3. 补充按 `extend_type`、rubric 和分歧方向的统计，重点抽查 reject、`need_review` 和 `AI_PRODUCT_RULE`。
4. 根据抽样结果决定是否对高风险分歧做稀疏人工复核；不要求先完成 11,377 条全量人工审核。
5. 使用扩大后的新 SFT 数据重新训练模型。
6. 训练后建立冻结人工评估集和 scorer，再评价 JSON/schema 合法率、decision precision/recall/F1、false pass、rubric match 和 reasoning 质量。

## 已弃用的历史 SFT 实验

此前使用约 1,552 条旧数据完成过两版 Qwen3-8B LoRA：

- `qwen3-8B-20260701-SFT`：ROUGE-1 54.88%、ROUGE-2 35.38%、ROUGE-L 44.64%、BLEU-4 38.10%。
- `QWEN3-8B-20260701_001-SFT`：ROUGE-1 60.19%、ROUGE-2 41.22%、ROUGE-L 49.35%、BLEU-4 43.59%。

这些结果现已弃用，只保留为历史记录：数据量过小、数据契约已变化、旧评估样本与方法不可靠，不能继续推荐其中任何模型，也不能用于衡量本次大规模数据训练的预期效果。

## 尚未开始

- 使用本次新数据的 SFT 训练。
- 冻结人工评估集与正式 scorer。
- SFT 模型部署。
- RL 数据、训练、评估与部署。
