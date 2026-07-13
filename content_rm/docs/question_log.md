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
- Decision：`audit_state` 不进入任何生成产物，最终审核结论统一使用 `audit_label`。

### 问题：人工复核文件是否需要 `human_decision`？
- 回答：不需要。`audit_label` 已经是运营审核结论，再新增 `human_decision` 会形成两个结论字段。人工复核阶段只补充或修正 reasoning，并记录必要备注。
- 可能缺乏的知识：标注复核可以复核解释链，而不是重新生成业务结论标签。
- Decision（2026-07-13 更新）：`human_review` 是稀疏人工覆盖，只保留 `comment_id`、`violated_rubrics`、`reasoning`、`review_note`。

### 问题：LLM prompt 是否需要强调不要输出 `review` / `suggested_fix`？
- 回答：不需要额外负向强调。保留正向 JSON schema 和 `decision: pass|reject` 约束即可，避免提示词冗余。
- 可能缺乏的知识：结构化输出提示词中，清晰的正向 schema 通常比堆叠负向约束更稳定。
- Decision：从 LLM system prompt 中删除“不要输出 review / suggested_fix”。

### 问题：`prepare_dataset.py` 是否应该直接调用公司内 LLM 接口生成预标注？
- 回答：可以接入，但应做成可选步骤。默认生成人工复核文件；在传入内网接口地址、模型名和 Authorization 时才调用 LLM，避免本机或无内网环境下数据准备失败。后续已决定不再持久化 `llm_annotation_tasks.jsonl`。
- 可能缺乏的知识：数据准备脚本需要区分“离线产物生成”和“依赖外部服务的标注生成”，这样流程更容易复现和排错。
- Decision（2026-07-13 更新）：同步与异步调用统一读取 `config.py`/环境变量中的 LLM 配置；请求体不再发送平台未使用的额外格式字段。

## 2026-06-16 SFT 阶段

### 问题：在 LLM 标注测试期间，是否可以先写 SFT 代码？
- 回答：可以先写，但不应直接启动训练。当前最稳的做法是先补“人工复核产物 -> 最终 SFT train/test JSONL”的构建脚本，以及 OpenRLHF 的启动脚本。
- 可能缺乏的知识：SFT 训练需要的是已定稿的 prompt/target 样本；LLM 预标注文件还不是最终训练集，需要经过人工复核或至少经过字段收敛。
- Decision（2026-07-13 更新）：最终 `decision` 使用运营 `audit_label`；监督解释按人工复核、合法二次复核、同意运营标签的一次标注依次选择。

## 2026-06-17 数据文件收敛

### 问题：`llm_annotation_tasks.jsonl`、`human_review.csv`、`sft_draft.jsonl` 是否需要由 prepare 阶段保留？
- 回答：不需要默认保留。LLM task 可以由代码从 sample 和固定 prompt 现场构造；CSV 是 `human_review.jsonl` 的重复视图；`sft_draft` 是训练前草稿，后续应由 SFT 构建阶段直接生成训练样本。
- 可能缺乏的知识：数据流水线中应区分“长期契约产物”和“可重建中间态”，减少冗余文件会降低理解成本和同步风险。
- Decision（2026-07-13 更新）：`prepare_dataset.py` 不生成全量人工底稿；LLM 模式生成丰富版 `llm_annotations.jsonl`，`--write-normalized` 仅用于输出 `normalized_comments.jsonl` 调试视图。

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

## 2026-06-25 Worktree 与 LLaMA-Factory 环境排查

### 问题：当前 Codex thread 是 worktree 形式工作，`.worktreeinclude` 是否真的生效？
- 回答：当前 worktree 位于 `/Users/xieweikang/.codex/worktrees/e997/RM-R1`，主工作区位于 `/Users/xieweikang/claudeProjects/RM-R1`。`.worktreeinclude` 中包含 `content_rm/data/local/` 和 `*.rendered.yaml`，这些路径虽然被 `.gitignore` 忽略，但在当前 worktree 中确实存在对应本地数据和 rendered yaml，说明 Codex 创建 worktree 时已经把这些 ignored 资源带入。
- 可能缺乏的知识：`.worktreeinclude` 不是 Git 原生机制，而是 Codex/worktree 创建流程使用的本地文件包含规则；它影响 ignored 文件是否被复制到临时 worktree，不影响 `git status`、`git add`、`git commit`。
- Decision：`.worktreeinclude` 只用于让 side worktree 具备本地数据/配置上下文；不要把它理解为“允许提交 ignored 文件”的规则。

### 问题：worktree 的工作形式应该如何把改动提交回主库？
- 回答：当前 worktree 是 detached HEAD。稳妥做法是在 worktree 内先创建分支，例如 `git switch -c codex/post-train-platform-sft`，再 `git add`、`git commit`。之后可以回到主工作区 `main` 执行 `git merge codex/post-train-platform-sft`，或从 worktree 推送该分支后走 PR。
- 可能缺乏的知识：Git worktree 可以共享同一个仓库对象库，但每个 worktree 有自己的工作区和 HEAD；detached HEAD 上的提交不挂在分支名上，后续容易丢失引用。
- Decision：Codex worktree 中的有效改动先挂到 `codex/` 前缀分支，再通过本地 merge 或远端 PR 回到主库；不要把重要提交长期留在 detached HEAD。

### 问题：如何在容器中查看 LLaMA-Factory 用的是哪个目录？
- 回答：要区分“程序安装目录”和“训练使用目录”。程序安装目录可以用 `python -c "import llamafactory, inspect, os; print(os.path.dirname(inspect.getfile(llamafactory)))"` 或 `python -m pip show llamafactory` 查看；训练实际读取的数据、模型和输出目录应优先看环境变量 `MODEL_PATH`、`DATASET_DIR`、`OUTPUT_DIR`，以及启动脚本生成的 rendered yaml 中的 `model_name_or_path`、`dataset_dir`、`output_dir`。
- 可能缺乏的知识：Python 包安装位置、命令行入口、训练配置文件和运行时环境变量是不同层面的路径；`llamafactory` 包在哪，不等于训练数据从哪读。
- Decision：排查“本次训练读哪个数据目录”时优先看 rendered yaml 的 `dataset_dir`；排查“LLaMA-Factory 程序从哪来”时再看 Python package path 或 pip metadata。

### 问题：为什么 `which llamafactory-cli` 没有输出，但 `pip show llamafactory` 能显示版本？`readlink -f "$(which llamafactory-cli)"` 是什么意思？
- 回答：`pip show llamafactory` 有输出说明当前 Python 环境能看到 `llamafactory` 包；`which llamafactory-cli` 没输出说明命令行入口不在当前 shell 的 `PATH` 中，或者当前安装方式没有生成该入口，也可能 shell 使用的 Python 环境和安装包所在环境不一致。`readlink -f "$(which llamafactory-cli)"` 是先用 `which` 找 CLI 路径，再用 `readlink -f` 解析软链接最终指向；如果 `which` 本身为空，这条命令就没有实际意义。
- 可能缺乏的知识：Python 包和 shell 命令入口是两回事；`PATH` 决定 shell 能直接找到哪些命令；`readlink -f` 用来解析软链接真实路径；命令替换 `$()` 会先执行括号内命令并把输出作为外层命令参数。
- Decision：先用 `python -c "import sys; print(sys.executable)"` 和 `python -c "import sysconfig; print(sysconfig.get_path('scripts'))"` 确认当前 Python 与 scripts 目录；只有 `which llamafactory-cli` 有输出时，才继续对该路径执行 `readlink -f`。

## 2026-06-29 SFT 训练数据与训练机制理解

### 问题：模型判断和运营审核结果不一致的样本是否应该进入 SFT 训练集？
- 回答：可以进入，而且这类样本通常是高价值纠错样本。若模型判断 `pass` 但运营结果是 `reject`，训练 target 应写 `decision: reject`，并在 reasoning 中解释命中的 rubric；若模型判断 `reject` 但运营结果是 `pass`，训练 target 应写 `decision: pass`，并解释为什么没有命中违规规则或为什么该表达仍可通过。
- 可能缺乏的知识：SFT 学习的是最终 target，不是模型预判；模型预判主要用于发现 hard cases 和错判样本。业务金标、模型预测、rubric reasoning 是三个不同层次。
- Decision：最终 SFT 样本的 `decision` 始终以运营 `audit_label` 为准；模型判断只作为辅助诊断信号。若运营 label、rubrics 和 reasoning 之间存在冲突，样本先进入复核池，不直接入训练。

### 问题：SFT 的最终对齐目标是否应该是运营审核偏好？
- 回答：是。当前任务不是训练通用金融合规模型，而是训练贴合平台运营审核口径的 AI 回复审核助手。因此监督信号应是“运营 `audit_label` + rubrics 下可解释的 reasoning”。裸 `pass/reject` 能教模型分类，但泛化弱；带 `violated_rubrics` 和具体证据的 reasoning 才能把运营偏好结构化。
- 可能缺乏的知识：运营 outcome label 是业务目标，rubric reasoning 是把业务偏好转成模型可学习模式的中间表示；LLM 预标注不能覆盖运营金标。
- Decision：训练 target 中 `decision` 使用运营 `audit_label`；`violated_rubrics` 和 `reasoning` 必须与该 decision 一致，用来解释运营口径。

### 问题：如果一条 SFT 样本中 `violated_rubrics` / `reasoning` 与 `decision` 自相矛盾，应该如何处理？
- 回答：这类样本不应直接进入训练。例如 reasoning 明确说“命中不得承诺收益、不得暗示收益”，但 `decision` 写成 `pass`，会教模型学习“一边判定违规一边通过”的错误模式。若运营结果确实是 `pass`，应把 `violated_rubrics` 改为空或改成符合通过口径的解释；若 reasoning/rubric 判断正确，则应把 `decision` 改为 `reject`。
- 可能缺乏的知识：SFT 不会理解“字段冲突是脏数据”，它只会按 target token 学习；字段内部一致性比单个字段是否存在更重要。
- Decision：新增或复核 SFT 数据时必须检查三元组一致性：`decision=reject` 时通常需要非空违规规则和拒绝理由；`decision=pass` 时不应保留违规命中和拒绝理由，除非 rubrics 明确允许某种例外且 reasoning 写清楚。

### 问题：SFT 样本从 JSONL 到模型计算经历哪些转换？
- 回答：人看到的 JSONL 行不是模型直接看到的文本。训练框架会先把 `system`、`prompt`、`response` 组装成 chat messages，再按目标模型的 chat template 拼成连续字符串，最后由模型 tokenizer 转成 token ids。对 Qwen 类模型，最终文本会包含 system/user/assistant 等模板标记和 assistant 的标准答案。
- 可能缺乏的知识：JSONL 是数据容器；chat template 决定模型实际看到的字符串；tokenizer 决定字符串如何切成 token id。不同模型的 chat template 和特殊 token 可能不同。
- Decision：排查训练输入时不能只看 JSONL 字段，还要确认平台/训练框架如何映射 `system`、`prompt`、`response`，以及使用了哪个模型的 tokenizer 和 chat template。

### 问题：SFT 训练时模型是否会像推理一样边采样边生成？
- 回答：不会。SFT 训练通常使用 teacher forcing：模型一次性读取完整 token 序列，对每个位置预测下一个 token，并用训练数据中的真实下一个 token 计算交叉熵 loss。训练时不会先采样出 token 再拿采样文本和标准答案比较；采样生成是推理阶段的行为。
- 可能缺乏的知识：next-token prediction、teacher forcing、cross entropy、训练前向计算和推理自回归生成的区别。
- Decision：理解 SFT loss 时按“给定完整上下文，逐 token 提高标准答案 token 的概率”来理解，不按“生成一段文本后再和标准答案比相似度”理解。

### 问题：SFT loss 通常在哪些 token 上计算？一条样本如何结束？
- 回答：在指令微调中，system/user prompt 通常作为上下文，label 会被 mask 成 `-100`，不参与 loss；assistant response token 才是主要监督目标。一条样本在 tokenize 后已经有确定长度，末尾通常包含 EOS 或 chat end token；若超过最大长度如 4096，会按平台策略截断或丢弃。训练不会让模型自行生成到最大长度。
- 可能缺乏的知识：label mask、attention mask、padding、EOS/chat end token、max sequence length 的作用。
- Decision：后续分析训练问题时要区分“上下文 token”和“监督 token”；重点确认 response 是否完整落在最大长度内，以及平台是否只对 assistant 输出计算 loss。

### 问题：tokenizer 词表是怎么来的？中文和英文是否使用不同词表？
- 回答：tokenizer 和词表来自基座模型，SFT 通常不重新训练 tokenizer。现代 LLM 的 tokenizer 一般是 BPE、SentencePiece 或类似子词方法，词表中同时包含中文字符/片段、英文词或子词、数字、标点、换行、JSON 符号和特殊 token。中文和英文不是两套独立词表，而是在同一个模型词表中按子词规则切分。
- 可能缺乏的知识：tokenizer 是模型契约的一部分；微调主要更新模型权重，不改变输入 token id 的定义。
- Decision：SFT 数据应使用与基座模型一致的 tokenizer；不要在微调阶段自行改变词表，除非有明确的新 token 扩展方案和对应 embedding 初始化策略。

### 问题：一次 training step 具体做了什么？
- 回答：一个 step 通常取一个 batch 的样本，完成 tokenize/padding/mask 后前向计算 logits，再对有效 label token 计算交叉熵 loss，反向传播并更新参数。若使用 gradient accumulation，多个 micro-batch 会先累积梯度，达到设定次数后才执行一次 optimizer update；平台显示的 step 通常指 optimizer step。
- 可能缺乏的知识：batch、micro-batch、gradient accumulation、optimizer step 与样本条数不是同一个概念。
- Decision：解释训练曲线时不要把 step 简单理解成“训练了几条样本”；要结合 batch size、gradient accumulation、epoch、有效 token 数一起看。

## 2026-07-07 进入 SFT Eval 阶段

### 问题：SFT 训练结束后，是否可以直接用平台 ROUGE/BLEU 判断模型好坏？
- 回答：不能只用 ROUGE/BLEU。当前任务的核心是金融内容社区 AI 回复审核，平台文本相似度指标只能说明模型输出和参考答案文字更接近，不直接等价于审核能力更强。
- 可能缺乏的知识：生成式文本指标会被 reasoning 文案相似度、JSON 格式、参考答案写法影响；审核任务更关注 `decision` 是否正确、违规 rubrics 是否合理、是否漏放高风险样本。
- Decision：SFT 后评估阶段以业务审核指标为主，包括 JSON/schema 合法率、`decision` accuracy、pass/reject precision/recall/F1、false pass、rubric match、reasoning 人工抽检。ROUGE/BLEU 仅作为辅助参考。

## 2026-07-08 Git Worktree 与分支理解

### 问题：本地工作区、worktree、branch、HEAD 分别是什么？
- 回答：本地工作区本质上也是一个 worktree，即一份真实文件夹；Codex 也会为不同 thread 创建独立 worktree。branch 不是文件夹，而是一个名字，指向某个最新 commit；从这个 commit 可以沿父子关系追溯到一串历史。HEAD 是当前 worktree 的位置：可以指向 branch，也可以直接指向 commit。
- Decision：理解当前状态时先确认“我在哪个物理文件夹/worktree”，再确认 `HEAD` 是否挂在某个 branch 上。

### 问题：什么是 detached HEAD？
- 回答：正常状态是 `HEAD -> branch -> commit`，新提交会让 branch 自动移动到新 commit。detached HEAD 是 `HEAD -> commit`，当前 worktree 直接站在某个 commit 上，没有挂在 branch 名字上；可以改代码，但若继续提交，最好先创建分支保护新 commit。
- Decision：detached HEAD 不是错误，但不适合作为长期开发状态。

### 问题：为什么同一个本地 branch 不能同时被两个 worktree checkout？
- 回答：branch 是共享的本地指针。如果两个 worktree 同时挂在同一个 branch 上，一个 worktree 提交后会移动该 branch 指针，但另一个 worktree 的真实文件不会自动刷新，容易造成“分支指针已更新、文件内容仍旧”的混乱状态。
- Decision：多个 worktree 可以基于同一个 commit，但应使用不同 branch，或让后创建的 worktree detached 在该 commit 上。

### 问题：checkout / 本地检出是什么意思？
- 回答：Git 里的 checkout 可以理解为把某个 branch/commit 对应的文件内容铺到当前 worktree，并让 HEAD 指向它。Codex UI 里的“本地检出/移交到 local”是把当前 Codex 临时 worktree 的对话状态移交到主工作区，让 Cursor 打开的本地项目目录与该对话使用同一份工作区。
- Decision：本地检出不是 push、不是 PR、也不是自动合并到 `main`；它会影响主工作区当前 checkout 的分支。

### 问题：新建 thread 选择某个分支时，能看到哪些内容？
- 回答：新 thread 看到的是该 branch 当前指向的 commit 以及被 Git 跟踪的文件。如果原 worktree 中有未提交改动，新 thread 看不到；如果文件在 ignored 目录中，例如 `content_rm/data/local/`，即使选择同一 branch 也不会自动出现。
- Decision：代码文件要让新 thread 看见，应先 commit 到 branch；ignored 本地数据要让新 thread 看见，应放到主工作区并通过 `.worktreeinclude` 或手动复制带入目标 worktree。

### 问题：`.worktreeinclude` 的作用是什么？
- 回答：`.worktreeinclude` 是 Codex 创建新 worktree 时用于携带 ignored 本地文件的清单，不是 Git 原生同步机制。它不会实时同步不同 worktree，也不会把某个 worktree 中后来新增的 ignored 文件自动复制到其他 worktree。
- Decision：`content_rm/data/local/` 这类本地数据若需要被新 Codex thread 使用，应优先放在主工作区，并确认 `.worktreeinclude` 包含对应路径。

### 问题：worktree 是否可以像另一台机器一样提交和推送？
- 回答：可以。每个 worktree 都是一份独立物理工作区，可以在自己的 branch 上 commit/push。push 被远端拒绝时，处理方式与多人协作类似：先 fetch/pull 或 rebase，解决冲突后再 push。区别是多个 worktree 共享同一个本地 Git 仓库的 branch、commit 对象和远端配置。
- Decision：worktree 的文件改动彼此隔离，但本地 branch 名字是共享资源；操作前应确认当前 worktree、当前 branch 和是否有未提交改动。

### 问题：detached HEAD 上的 commit 能否 push？它是不是孤立 commit？
- 回答：detached HEAD 上也可以产生 commit。若没有 branch、tag、HEAD 或 reflog 等引用指向它，之后切走就可能变成不好找的 dangling commit；但执行 `git push origin HEAD:<remote-branch>` 会把当前 HEAD 指向的 commit 推到远端分支，让远端 branch 指向它，因此不是制造孤立 commit。commit hash 基本可视为全局唯一；普通 commit 通常有一个父 commit，merge commit 可以有多个父 commit，root commit 没有父 commit。
- Decision：detached HEAD 上若要保留工作，优先先 `git switch -c <branch>` 再提交/推送；也可以用 `git push origin HEAD:<remote-branch>` 直接把当前 commit 推成远端分支。

### 问题：`git push --set-upstream origin codex/post-train-platform-sft` 中的 `--set-upstream` 和 `origin` 分别是什么意思？
- 回答：`origin` 是远端仓库的名字，通常是 clone 仓库时 Git 自动创建的默认远端别名，不是分支名。`--set-upstream` 用来建立本地分支和远端分支的 tracking 关系，例如让本地 `codex/post-train-platform-sft` 默认跟踪 `origin/codex/post-train-platform-sft`。
- 可能缺乏的知识：远端名、远端分支、本地分支和 upstream tracking 是不同概念。远端名可以通过 `git remote -v` 查看；upstream 关系建立后，在该分支上可以直接执行 `git push` 或 `git pull`，不必每次完整指定远端和分支。
- Decision：第一次推送新本地分支时可以使用 `git push --set-upstream origin <branch>`；之后若 upstream 已建立，日常推送直接使用 `git push` 即可。

### 问题：除了 `origin`，一个本地 Git 仓库能否关联多个远端仓库？
- 回答：可以。一个本地仓库可以配置多个 remote，例如 `origin`、`upstream`、`company`、`backup`。`origin` 只是默认名称，不具备唯一性或特殊权限。
- 可能缺乏的知识：remote 是本地配置中的远端地址别名；每个 remote 可以有独立的 fetch/push 地址。常见用法是 `origin` 指向自己的 fork 或主推送仓库，`upstream` 指向原始项目仓库，`company` 指向公司内部仓库。
- Decision：需要查看所有远端时使用 `git remote -v`；需要和特定远端交互时显式写远端名，例如 `git fetch upstream`、`git pull upstream main`、`git push company <branch>`。
