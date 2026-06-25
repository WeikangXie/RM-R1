# 问题日志

## 2026-06-15 数据阶段

### 问题：训练/评测标签第一阶段怎么来？是否可以用 LLM 先标注，再人工复核？
- 回答：可以。运营审核结果适合作为最终业务标签，LLM 适合作为 rubric reasoning 的预标注工具，最终训练金标应以人工复核后的结果为准。
- 可能缺乏的知识：业务 outcome 标签、弱标注、人工复核金标、推理链蒸馏之间的区别。
- Decision：第一阶段使用公司内模型做 LLM 预标注，再人工复核。

### 问题：`decision` 是否只需要 `pass|reject`，为什么需要 `review`？
- 回答：第一版只需要 `pass|reject`。`review` 会让训练目标变成三分类，并且运营结果只有通过/不通过，当前没有必要引入。
- 可能缺乏的知识：训练标签空间需要和业务目标保持一致；不确定性可以用 `confidence` 或人工复核流程表达，不一定要放进模型最终标签。
- Decision：`decision` 只保留 `pass|reject`。

### 问题：为什么要输出 `suggested_fix`？是否只需要 `reasoning`？
- 回答：`suggested_fix` 对审核产品或改写系统有价值，但对当前“判断是否通过”不是必要字段。
- 可能缺乏的知识：审核判别任务和内容改写任务的目标不同。
- Decision：第一版不输出 `suggested_fix`，只保留 rubric 命中、reasoning、decision。

### 问题：SFT 输入是 JSONL，数据里没有推理过程，模型怎么学会按 rubrics 推理？
- 回答：模型不是从 JSONL 这种文件格式里学推理，而是从每条样本的目标输出里学。若目标输出只有 `pass/reject`，模型只学分类；若目标输出包含 reasoning 和 final decision，模型才会学习推理链路。
- 可能缺乏的知识：SFT 的监督信号来自 target/output，不来自文件后缀。
- Decision：后续 SFT 草稿要面向“reasoning + final decision”的输出格式。

### 问题：基线 scorer 是做什么的？SFT 标准训练过程中必要吗？
- 回答：基线 scorer 不是 SFT 的必要环节，它用于数据 QA 和评测对照，帮助发现标签冲突、规则覆盖不足，以及判断 SFT 是否超过简单规则。
- 可能缺乏的知识：训练流程和评测/数据诊断流程可以分离。
- Decision：第一版先不把基线 scorer 作为训练阻塞项；必要时作为后续评测辅助加入。

### 问题：单条评分输出是否仍然包含通过/不通过预测？
- 回答：是。即使不是 A/B 偏好判别，模型最终也需要输出审核预测结果。
- 可能缺乏的知识：单条评分任务通常仍需要明确 final label，解释只是辅助。
- Decision：模型输出保留 `decision: pass|reject`。

### 问题：RM-R1 可用的 SFT 数据格式是不是简单 JSONL？是否还需要推理过程？
- 回答：可以是 JSONL，但不是任意 JSONL。关键是每条样本必须包含输入 prompt 和目标输出；要训练推理奖励模型，目标输出应包含推理过程和最终答案。
- 可能缺乏的知识：数据容器格式和训练语义格式的区别。
- Decision：数据阶段先生成 SFT 草稿 JSONL，后续经 LLM 标注和人工复核后再作为训练数据。

### 问题：当前只有每个 extendType 约 10 条、总计几十条，能否先写代码？
- 回答：可以。几十条足够开发数据处理代码和 smoke test；200 条主要是为了后续评测指标更稳定。
- 可能缺乏的知识：流程验证样本量和统计评估样本量的要求不同。
- Decision：基于当前 37 条样本先实现第一版数据处理闭环。

### 问题：`audit_state` 是否还需要进入生成产物？
- 回答：不需要。输入数据采集阶段会保证样本都已有运营审核结果，真正用于训练和评测的结论字段是 `audit_label: pass|reject`。
- 可能缺乏的知识：原始系统状态字段和训练标签字段应该分离，训练产物越少保留无关字段越不容易产生歧义。
- Decision：`audit_state` 从 `normalized_samples.jsonl`、`human_review.jsonl`、`human_review.csv`、`sft_draft.jsonl` 等所有生成产物中移除。

### 问题：人工复核文件是否需要 `human_decision`？
- 回答：不需要。`audit_label` 已经是运营审核结论，再新增 `human_decision` 会形成两个结论字段。人工复核阶段只补充或修正 reasoning，并记录必要备注。
- 可能缺乏的知识：标注复核可以复核解释链，而不是重新生成业务结论标签。
- Decision：`human_review` 只保留 `audit_label`、`llm_decision`、`llm_reasoning`、`human_reasoning`、`review_note`、`violated_rubrics` 等字段。

### 问题：LLM prompt 是否需要强调不要输出 `review` / `suggested_fix`？
- 回答：不需要额外负向强调。保留正向 JSON schema 和 `decision: pass|reject` 约束即可，避免提示词冗余。
- 可能缺乏的知识：结构化输出提示词中，清晰的正向 schema 通常比堆叠负向约束更稳定。
- Decision：从 LLM system prompt 中删除“不要输出 review / suggested_fix”。

### 问题：`prepare_dataset.py` 是否应该直接调用公司内 LLM 接口生成预标注？
- 回答：可以接入，但应做成可选步骤。默认生成人工复核文件；在传入内网接口地址、模型名和 Authorization 时才调用 LLM，避免本机或无内网环境下数据准备失败。后续已决定不再持久化 `llm_annotation_tasks.jsonl`。
- 可能缺乏的知识：数据准备脚本需要区分“离线产物生成”和“依赖外部服务的标注生成”，这样流程更容易复现和排错。
- Decision：新增 `--call-llm`、`--llm-base-url`、`--llm-model`、`--llm-authorization` 等参数；请求体只发送必要字段：`model`、`messages`、`max_tokens`、`temperature`、`top_p`、`stream`、`response_format`，以及可选 `seed`。

## 2026-06-16 SFT 阶段

### 问题：在 LLM 标注测试期间，是否可以先写 SFT 代码？
- 回答：可以先写，但不应直接启动训练。当前最稳的做法是先补“人工复核产物 -> 最终 SFT train/test JSONL”的构建脚本，以及 OpenRLHF 的启动脚本。
- 可能缺乏的知识：SFT 训练需要的是已定稿的 prompt/target 样本；LLM 预标注文件还不是最终训练集，需要经过人工复核或至少经过字段收敛。
- Decision：新增 SFT 构建脚本和训练启动脚本；最终 `decision` 使用运营 `audit_label`，`reasoning` 优先用 `human_reasoning`，为空时回退到 `llm_reasoning`。

## 2026-06-17 数据文件收敛

### 问题：`llm_annotation_tasks.jsonl`、`human_review.csv`、`sft_draft.jsonl` 是否需要由 prepare 阶段保留？
- 回答：不需要默认保留。LLM task 可以由代码从 sample 和固定 prompt 现场构造；CSV 是 `human_review.jsonl` 的重复视图；`sft_draft` 是训练前草稿，后续应由 SFT 构建阶段直接生成训练样本。
- 可能缺乏的知识：数据流水线中应区分“长期契约产物”和“可重建中间态”，减少冗余文件会降低理解成本和同步风险。
- Decision：`prepare_dataset.py` 默认只生成 `human_review.jsonl` 和 `summary.json`；`--call-llm` 时额外生成 `llm_annotations.jsonl`；`normalized_samples.jsonl` 仅在 `--write-normalized` 时作为调试产物输出。

### 问题：原始 `parentInfo` 和 `text` 应该如何取舍？
- 回答：训练和复核阶段只使用清洗后的 `text`。`text` 更完整，`parentInfo` 可能截断；若 `text` 含 HTML，则先清洗。
- 可能缺乏的知识：模型输入应尽量使用完整、稳定、语义明确的字段，截断字段更适合作为展示摘要而不是训练输入。
- Decision：`human_review.jsonl` 使用 `text` 字段，不再输出 `parent_info`。

## 2026-06-17 进入 SFT 阶段

### 问题：数据准备告一段落后，SFT 阶段第一步做什么？
- 回答：先把 SFT 构建脚本接到新的 prepare 产物上，不再依赖已移除的 `sft_draft.jsonl`。实际训练需要等 LLM 标注和人工复核完成后再启动。
- 可能缺乏的知识：SFT 阶段包括“生成最终训练集”和“启动训练”两件事，前者可以先做，后者依赖标注质量和训练环境。
- Decision：`build_sft_dataset.py` 改为读取 `human_review.jsonl`、`rubrics.md` 和可选 `llm_annotations.jsonl`，直接生成 OpenRLHF 所需的 `train.jsonl` / `test.jsonl`。

### 问题：Hugging Face Transformers 是什么？它和当前 SFT 训练代码有什么关系？
- 回答：Transformers 是大模型领域常用的 Python SDK，用来统一加载模型、加载 tokenizer、套用 chat template，并在训练时执行模型前向计算和 loss 计算。它不是一个模型本身，而是 OpenRLHF 底层用来操作 Qwen、Llama 等模型的核心库。当前项目里，OpenRLHF 会基于 Transformers 加载 `Qwen/Qwen2.5-7B-Instruct`，把 `context_messages` 通过 chat template 拼成模型输入，再让模型学习输出 `reasoning + decision`。
- 可能缺乏的知识：模型仓库、模型架构、训练框架、tokenizer、chat template、OpenRLHF 之间的分工；尤其是 JSONL 中的 `role/content` 消息并不是模型直接看到的最终字符串，需要由 tokenizer/chat template 转换。
- Decision：理解当前 SFT 代码时，先把 Transformers 视为“模型加载与文本/token 转换层”，重点关注 `model_path`、`tokenizer`、`chat_template`、`input_key`、`output_key` 如何共同决定模型实际学习的输入和输出。

## 2026-06-24 Ascend 容器训练环境排查

### 问题：容器启动命令里的 `PROJECT=/path/to/RM-R1` 是什么？它和脚本里的 `PROJECT_ROOT` 是同一个东西吗？
- 回答：`/path/to/RM-R1` 是占位符，实际应替换为容器内 RM-R1 代码目录的绝对路径。可以先 `cd` 到 RM-R1 目录，再用 `pwd` 的输出作为 `PROJECT`。`PROJECT` 是人工在 shell 中设置的便捷变量；`PROJECT_ROOT` 是 `train_ascend_lora.sh` 根据脚本位置自动推导出的项目根目录。当前脚本路径是 `$PROJECT/content_rm/sft/llamafactory/train_ascend_lora.sh`，二者应指向同一个 RM-R1 根目录。
- 可能缺乏的知识：Linux 绝对路径、shell 环境变量、脚本内部路径推导、容器内路径与本机路径不是同一套文件系统。
- Decision：容器排查时先确认 `pwd`，再显式设置 `PROJECT=<容器内 RM-R1 绝对路径>`；不要把 `/path/to` 当成真实目录。

### 问题：`ls "$PROJECT/content_rm/sft/llamafactory/train_ascend_lora.sh"` 是什么意思？`ls` 也能看 `.sh` 文件吗？
- 回答：这条命令用于检查指定路径下的训练启动脚本是否存在。`ls` 既可以列目录，也可以显示单个文件；如果文件存在，会输出该文件路径或文件信息，如果不存在会报 `No such file or directory`。如果要查看 `.sh` 文件内容，应使用 `cat`、`head` 或 `sed -n '1,80p'`。
- 可能缺乏的知识：`ls` 的对象可以是文件或目录；检查文件存在和查看文件内容是两个不同动作。
- Decision：用 `ls <file>` 做存在性检查；用 `sed -n '1,80p' <file>` 查看脚本内容。

### 问题：`test -f "$DATASET_DIR/dataset_info.json" && echo ok` 是在做什么？
- 回答：`test -f` 判断目标路径是否存在且是普通文件；`&&` 表示前一个命令成功时才执行后一个命令。因此这条命令的含义是：如果 `$DATASET_DIR/dataset_info.json` 存在，就打印 `ok`；如果不存在，则不打印 `ok`。它用于快速确认 LLaMA-Factory 数据目录是否配置正确。
- 可能缺乏的知识：shell 条件判断、`&&` 短路执行、LLaMA-Factory 本地数据目录必须包含 `dataset_info.json`。
- Decision：启动 Ascend 训练前必须确认 `DATASET_DIR` 指向包含 `train.json`、`test.json`、`dataset_info.json` 的 `llamafactory_alpaca` 目录。

### 问题：容器里执行 `cd ~` 后为什么进入 `/home/HwHiAiUser`？
- 回答：`~` 表示当前登录用户的 home 目录。当前容器用户是 `HwHiAiUser`，所以 `cd ~` 后 `pwd` 输出 `/home/HwHiAiUser`。这个目录是用户目录，但不一定是持久化存储，是否会在容器重启后保留取决于平台挂载配置。
- 可能缺乏的知识：Linux home 目录、`~` 展开、容器用户目录和持久化挂载目录不是同一概念。
- Decision：训练数据、模型和 checkpoint 优先放在平台明确挂载的 `/data` 下，不默认依赖 `/home/HwHiAiUser` 持久化。

### 问题：容器里执行 `cd /data` 是进入什么目录？
- 回答：`/data` 是从根目录 `/` 开始的绝对路径，表示根目录下的 `data` 目录。当前平台将存储卷挂载到了 `/data`，因此 LLaMA-Factory、RM-R1 代码、训练数据和训练输出放在 `/data` 的子目录下是合理的。
- 可能缺乏的知识：Linux 绝对路径、根目录 `/`、挂载点、容器内路径与宿主机/本机路径不同。
- Decision：容器内路径排查以 `pwd` 和 `ls /data` 为准；训练脚本中的 `MODEL_PATH`、`DATASET_DIR`、`OUTPUT_DIR` 都应使用容器内可访问的绝对路径。

### 问题：`df -h /data` 是什么命令？输出里的 `Filesystem` 表示什么？
- 回答：`df` 是查看文件系统磁盘空间的命令，`-h` 表示 human-readable，会用 `G`、`T` 等易读单位展示容量。`df -h /data` 查看的是 `/data` 所在文件系统的容量和挂载信息。输出中的 `xx.xxx.xxx.xx:/ai-model` 表示 `/data` 背后是一个远程网络存储路径，而不是容器本地临时磁盘；它被挂载到当前容器的 `/data`。
- 可能缺乏的知识：文件系统、挂载点、网络存储/NFS、本地磁盘、容器 overlay、`tmpfs` 等不同存储类型；`Size`/`Used`/`Avail`/`Use%`/`Mounted on` 各列含义。
- Decision：`/data` 当前容量约 1.1T、可用约 612G，确认是适合放模型、数据和 checkpoint 的持久化挂载目录；重要训练产物不要放到未确认持久化的临时目录。

### 问题：`echo $PROJECT`、`${PROJECT}` 和 `export PROJECT=...` 分别是什么意思？
- 回答：`echo $PROJECT` 会输出当前 shell 中 `PROJECT` 变量的值；未设置时通常输出空行。在脚本中打印变量可以写 `echo "$PROJECT"` 或 `echo "${PROJECT}"`，大括号不是必须，但推荐使用，因为 `${PROJECT}_suffix` 能清晰表达变量名边界。只在终端执行 `PROJECT=/data/xwk/RM-R1` 时，变量默认只存在于当前 shell，不会自动传给子进程；若希望启动脚本可见，需要执行 `export PROJECT=/data/xwk/RM-R1`，或用 `PROJECT=/data/xwk/RM-R1 bash script.sh` 只对这一次命令传入。
- 可能缺乏的知识：shell 变量、环境变量、子进程继承、`export`、变量名边界、大括号展开。
- Decision：容器排查时可以用 `echo "$PROJECT"` 检查人工设置的项目路径；但当前 Ascend 启动脚本不依赖外部 `PROJECT`，而是通过脚本位置自动推导 `PROJECT_ROOT`，如需打印脚本推导结果应打印 `echo "PROJECT_ROOT=${PROJECT_ROOT}"`。

### 问题：为什么 shell 脚本里 `echo` 变量时推荐写双引号？`echo ${SCRIPT_DIR}` 不可以吗？
- 回答：`echo ${SCRIPT_DIR}` 可以运行，但更推荐写 `echo "${SCRIPT_DIR}"` 或 `echo "SCRIPT_DIR=${SCRIPT_DIR}"`。双引号的主要作用是保护变量展开后的内容，避免路径里有空格、通配符或特殊字符时被 shell 拆成多个参数或触发额外展开。是否在前面加 `SCRIPT_DIR=` 这样的说明文字不是关键；关键是变量引用放在双引号中更稳。
- 可能缺乏的知识：shell 参数拆分、变量展开、双引号保护、路径中空格导致命令行为变化。
- Decision：脚本中引用路径变量时统一使用 `"${VAR}"`；调试打印时使用 `echo "VAR=${VAR}"`，既清楚又避免变量展开带来的边界问题。

### 问题：`grep -R "qwen3_nothink" /data/LLaMA-Factory 2>/dev/null | head` 这类命令怎么理解？
- 回答：`grep` 用来按模式搜索文本，默认 pattern 可视为基础正则；当搜索内容没有正则特殊符号时，效果接近普通字符串搜索。如果想明确按普通字符串搜索，可以用 `grep -F`。`-R` 表示递归搜索目录。`2>/dev/null` 表示把标准错误 stderr 丢弃，常用于屏蔽无权限、目录不存在、特殊文件无法读取等报错，从而聚焦搜索结果。`| head` 使用管道把 `grep` 的标准输出传给 `head`，`head` 默认只展示前 10 行。
- 可能缺乏的知识：grep/正则/普通字符串搜索、标准输出 stdout、标准错误 stderr、文件描述符 `2`、`/dev/null`、管道 `|`、`head` 命令。
- Decision：在容器里检查 LLaMA-Factory 是否支持某个模板时，用真实路径搜索，例如 `grep -R "qwen3_nothink" /data/LLaMA-Factory/src /data/LLaMA-Factory 2>/dev/null | head`；若希望避免正则语义，可使用 `grep -RF "qwen3_nothink" ...`。
