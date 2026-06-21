# 日志

## [2026-06-06] ingest | GLM-5、MiMo-V2-Flash、DeepSeek-V4 技术报告

初始化 `wiki/` 知识库结构，并沉淀三份原始技术报告：

- `raw/glm-5-2602.15763.pdf`
- `raw/mimo-v2-flash-2601.02780.pdf`
- `raw/deepseek-v4-hf-technical-report.pdf`

创建了来源页、模型实体页、核心概念页和第一版跨报告比较页。DeepSeek-V4 下载时没有找到 arXiv 条目；当前沉淀基于 Hugging Face 官方技术报告 PDF。

## [2026-06-06] maintenance | 知识库语言切换为中文

将 `wiki/` 下已生成的沉淀页统一改写为中文。文件名和目录名保持英文，便于命令行检索和稳定链接；技术术语在必要时保留英文或英文缩写。

## [2026-06-06] deepen | 扩展长上下文、后训练和评测模块

对第一版沉淀做第二轮优化：扩展 `efficient-long-context-attention.md`、`post-training-for-agentic-models.md` 和 `multi-token-prediction.md`，并新增五个细讲页面：

- `wiki/concepts/deepseek-sparse-attention.md`
- `wiki/concepts/asynchronous-agent-rl.md`
- `wiki/concepts/multi-teacher-on-policy-distillation.md`
- `wiki/concepts/million-token-context-serving.md`
- `wiki/concepts/agentic-evaluation-benchmarks.md`

技术机制主要来自三份已下载技术报告；评测体系页补充了 SWE-bench Verified、BrowseComp、Terminal-Bench 2.0 和 MCP-Atlas 的外部来源链接。

## [2026-06-06] ingest | MiniMax-M2 Series、Kimi K2.5 技术报告

新增两份 arXiv 技术报告到 `raw/`：

- `raw/minimax-m2-series-2605.26494.pdf`
- `raw/kimi-k2.5-2602.02276.pdf`

创建了对应来源页和模型页，并新增三个细讲页面：`forge-agent-native-rl.md`、`agent-swarm.md`、`multimodal-agentic-training.md`。同步更新 agentic engineering、后训练、MTP、MoE 扩展、agentic 评测体系和 2026 横向比较页。

## [2026-06-19] ingest | MSA、IndexCache 技术报告

新增两份稀疏注意力机制论文：

- `raw/Lai 等 - 2026 - MiniMax sparse attention.pdf`（arXiv:2606.13392v2）
- `raw/Bai 等 - 2026 - IndexCache Accelerating sparse attention via cross-layer index reuse.pdf`（arXiv:2603.12201v1）

新增来源页 `wiki/sources/msa.md` 和 `wiki/sources/indexcache.md`；新增概念页 `wiki/concepts/cross-layer-index-reuse.md`，覆盖 IndexCache 与 Kascade / TidalDecode / HySparse 的跨层共享谱系。

更新 `wiki/concepts/deepseek-sparse-attention.md`：增补与 MSA 的对比表，并把 indexer 自身 O(NL²) 瓶颈和 IndexCache 的解法明确写出。更新 `wiki/concepts/efficient-long-context-attention.md`：路线表从三条扩展为四条（拆出 token-level / block-level 两种内容稀疏），新增"indexer 自身瓶颈"小节。

未新建模型页：MSA 是机制论文，配套发布的 MiniMax-M3 信息有限；IndexCache 是推理优化方法，挂在已有 GLM-5 模型页之下。

## [2026-06-19] deepen | MiniMax-M3 模型页与稀疏注意力机制对比

补足上一次 ingest 故意留空的两块：

- 新增 `wiki/models/minimax-m3.md`，主要事实从 MSA 论文实验配置摘出（109B / 6B-active、41 层、64 query head / 4 KV head、B=128, n=16），HF release blog 暂未沉淀，待补。
- 新增 `wiki/comparisons/sparse-attention-mechanisms.md`，把 DSA、MSA、NSA、MoBA、InfLLM-V2、CSA/HCA、IndexCache、推理时稀疏化沿"选择粒度 / 跨头共享 / 跨层共享"三条独立优化轴铺开对比，并讨论"主注意力稀疏化"和"indexer 跨层复用"是两件独立的事。
- 在 MSA 来源页加 M3 模型反链；index.md 模型栏与比较栏分别增补条目。

## [2026-06-19] maintenance | .obsidian 脱离版本控制

拉取远端 commit `6c72010 chore: ignore obsidian settings`，删除原本被跟踪的 `.obsidian/app.json`、`appearance.json`、`core-plugins.json`，并加入 `.gitignore: .obsidian/`。本地工作区中 `.obsidian/workspace.json` 和 `app.json` 不再进入版本控制。

## [2026-06-19] deepen | MSA 公开 kernel 信息补全

WebFetch 对 `huggingface.co` ECONNREFUSED、对 `github.com` "unable to verify"，但能到 `raw.githubusercontent.com`。从 [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) 的 README 拿到几条论文外信息，沉淀到 `msa.md` 新增的"公开 kernel 实现"小节：目标 GPU 是 NVIDIA SM100（Blackwell）而非论文实验的 H800（SM90）；除了 BF16/FP8 还支持 NVFP4/FP4；页大小 128、top-k 16 与论文一致；MIT 许可；BibTeX 是占位符。同时把这条 H800 vs SM100 的不一致加进 `msa.md` 和 `minimax-m3.md` 的待追问。HF model card（M3 模型自身）目前 WebFetch 通道仍无法访问。

## [2026-06-20] maintenance | 模型页统一补齐"模态"字段

用户提出模型页应统一标注模态。回 `raw/` 原始 PDF 逐一求证 6 个模型页后补齐 `模态` 字段：GLM-5 / MiMo-V2-Flash / MiniMax-M2 / DeepSeek-V4 均核实为**纯文本**（报告里的 "multimodal verifier / MM-Claw / Multimodal LM" 都是 RL 管线判官或评测名，非模型自身输入；DeepSeek-V4 把多模态列为 Outlook 未来方向）；Kimi K2.5 = 多模态（文本+图像+视频）；MiniMax-M3 原生多模态（作样式参考）。

- 各模型页「关键事实」表加 `模态` 行（DeepSeek-V4 为变体表，改放表下说明）。
- CLAUDE.md 补模型页 skeleton：「关键事实表必须含模态字段」。

附带订正：本轮先前误记 "raw/ 已无 GLM-5 PDF"（疑 OneDrive 占位符致 `ls` 异常），实际 8 个源 PDF 全在位、各 source 页引用无失链。`raw/` 未改。



## [2026-06-20] deepen | DSA / MSA 训练分阶段对比

针对"GLM-5 是否分阶段训练 DSA""MSA warmup 是否同构"两问，回 `raw/glm-5-2602.15763.pdf` 与 `raw/Lai 等 - 2026 - MiniMax sparse attention.pdf` 求证，按原文修订三页：

- `wiki/concepts/deepseek-sparse-attention.md`：DSA warmup 精确化（1000 步 × 14 序列 × 202,752 token，LR 5e-3→2e-4）+ sparse adaptation（恒定 1e-5 + 20B tokens）+ indexer 配置（32 head/dim 128）+ RL insights 段补全 deterministic torch.topk 对比。
- `wiki/comparisons/sparse-attention-mechanisms.md`：新增「训练分阶段对比（第四条轴）」小节，DSA/MSA 沿 6 维度对照 + 三条推论；MSA RL 判断硬化为引 Outlook 原文。
- `wiki/sources/msa.md`：待追问首条同步硬化措辞。

订正：上一版 DSA 页 warmup 漏步数/batch，且把"warmup 只训 indexer"误写成主文本结论（实际只在附录 GLM-4.7-Flash 消融明写）。`raw/` 未改。


## [2026-06-20] ingest | DeepSeek-V3.2 技术报告

新增 `raw/DeepSeek-V3.2.pdf`（arXiv:2412.19437，2024-12-02）并沉淀 wiki 页面：

- **新增 `wiki/sources/deepseek-v32.md`**：来源页。覆盖 DSA 架构（lightning indexer + fine-grained token selection，MLA MQA 模式）、两阶段训练配方（dense warmup 2.1B tokens / sparse training 943.7B tokens）、后训练策略（sparse attention 继续训练而非冻结 indexer）、评测要点及待追问。
- **更新 `wiki/concepts/deepseek-sparse-attention.md`**：新增「架构概览」小节，以 V3.2 为原始出处描述 DSA 组件；「与 DeepSeek-V4 的关系」改为「与 DeepSeek-V3.2 / DeepSeek-V4 的关系」并标注 V3.2 为 DSA 起源；新增「Post-training 中 indexer 的处理对比」小节，对比 V3.2（继续训练 indexer）与 GLM-5（冻结 indexer + deterministic torch.topk）的路线差异。
- **更新 `wiki/sources/deepseek-v4.md`**：CSA 描述中加入对 V3.2 / DSA 的明确引用。
- **更新 `wiki/index.md`**：来源列表新增 DeepSeek-V3.2 条目。

求证与修正：(1) GLM-5 DSA warmup 精确为 1000 步 × 14 序列 × 202,752 token 非「用 202,752-token 序列」一句话；(2) V3.2 sparse training 943.7B tokens vs GLM-5 仅 20B，量级差 ~47 倍；(3) V3.2 post-training 继续用 sparse attention（GLM-5 则冻结 indexer），这是两条不同的工程路线。`raw/` 内容未改。


## [2026-06-20] deepen | MiniMax-M3 规格表分层（NVIDIA blog 交叉验证）

之前 `wiki/models/minimax-m3.md` 把 MSA 论文里 109B 实验版的细节（41 层、d_model=3072、64 query heads、词表 200K 等）当成发布版 M3 的真实规格记录，与 HF/GitHub README 的 ~428B/~23B 数字混在同一张「关键事实」表里，逻辑上不一致。

这一轮拉取 [NVIDIA Developer Blog](https://developer.nvidia.com/blog/deploy-long-context-reasoning-and-agentic-workflows-with-minimax-m3-on-nvidia-accelerated-infrastructure/) 原文做交叉验证，拿到一份发布版 M3 的官方规格表：428B 总参 / 22B 激活 / 600M visual encoder / 128 experts × 4 active / 1M context / BF16+MXFP8。

修订：

- `wiki/models/minimax-m3.md`：「关键事实」拆成两栏——「生产版（NVIDIA blog 规格表）」和「MSA 论文实验版（109B，仅作机制层参考）」，避免再把实验配置当 M3 真实规格；NVIDIA blog 来源补到段落正文。「待追问」更新为：(1) 109B → 428B 是单纯放大还是 routing/attention 改设计，无端到端 release-grade technical report；(2) HF README ~23B 与 NVIDIA blog 22B + 600M visual 差 1B 可对得上，但 NVIDIA 写「4 experts active」、MSA 论文写「top-4 routed + 1 shared」，shared expert 是否保留没有第三方来源能 cross-check；(3) NeMo AutoModel CP 上限 128K，与 1M context 之间存在工具链 gap。
- `wiki/concepts/moe-frontier-model-scaling.md`：M3 行更新为 NVIDIA blog 精确口径（428B / 22B + 600M / 128 routed × 4 active）。
- 索引页之前已经是 ~428B / ~23B 的描述，与新口径在 round-up 意义上一致，未再动。

`raw/` 未改。这次的来源是 NVIDIA blog，独立于 MiniMax 自己的 README，置信度比之前一档来源高。

## [2026-06-20] deepen | CSA core attention = Shared-KV MQA

用户提问"DeepSeek-V4 的 CSA 是个 MQA 吗"。回 `raw/deepseek-v4-hf-technical-report.pdf` §2.3.1 核实：CSA 的 core attention 确实是 MQA，原文 "Shared Key-Value MQA … performs core attention in a Multi-Query Attention (MQA) (Shazeer, 2019) manner, where each compressed KV entry … serves as both attention key and value"。即稀疏选出 top-k 压缩 entry 后所有 query head 共享同一份 entry，且 K=V（比普通 MQA 共享度更高）；query 侧仍是 MLA 式 latent 上投影（latent 与 indexer query 共享），head 数大故用 grouped output projection。

修订两处此前不够精确的记录：

- `wiki/sources/deepseek-v4.md`：架构段补一段，明确 CSA = 「MLA 式 latent query +（token 压缩 → DSA 稀疏选择）→ Shared-KV MQA core」，不是单纯 MLA。
- `wiki/comparisons/sparse-attention-mechanisms.md`：CSA/HCA 行的「底层」从 `MLA` 改为 `MLA query + Shared-KV MQA core`，「跨头共享」从「共享」细化为「MQA：所有 query head 共用一份 K=V 压缩 entry」。

`raw/` 未改。

## [2026-06-20] deepen | 新增 MLA 概念页，理清 DSA/CSA 的注意力血统

用户追问"DSA 是从 MLA 进化的吗、MLA 又是从 GQA 还是 MQA 进化的"。回 `raw/DeepSeek-V3.2.pdf` §（"Instantiate DSA Under MLA"）+ 附录 A（MHA/MQA modes of MLA）核实后沉淀：

- **新增 `wiki/concepts/multi-head-latent-attention.md`**：讲清 MLA 与 GQA/MQA 是「正交两条轴」——GQA/MQA 减 KV head 数，MLA 压低秩 latent；MLA 在 decode 做矩阵吸收后退化成 MQA（即 V3.2 附录 A 的「MQA mode」），故近亲是 MQA 但带低秩补偿。跨报告信号串起 V3.2（DSA 架在 MLA-MQA mode）、V4（CSA 继承 latent query + Shared-KV MQA）、GLM-5。MLA 内部机制（下/上投影、decoupled RoPE、矩阵吸收、KV cache 等效 GQA-2.25）按通识写，明确标注 DeepSeek-V2 原文不在 `raw/`、待补后逐条核实。
- 反链补齐：`deepseek-sparse-attention.md` line 20（MLA-MQA mode 处）、`sources/deepseek-v32.md`（DSA 实例化处 + 附录 A mode 说明）、`sources/deepseek-v4.md`（CSA latent query 处）各加 MLA 页链接。
- `wiki/index.md`：细讲模块新增 MLA 条目。

待补来源：用户表示可提供 DeepSeek-V2（MLA 首次提出）原论文。补入 `raw/` 后应回原文核实 MLA 内部机制那几条，并把本页「待追问」逐条消化。`raw/` 本轮未改。

## [2026-06-20] ingest | DeepSeek-V2（MLA 原始论文），核实并升级 MLA 概念页

用户把 DeepSeek-V2 原论文放进 `raw/`（`DeepSeek-AI 等 - 2024 - DeepSeek-V2 ...pdf`，arXiv:2405.04434）。回原文 §2.1.1–2.1.3 + Table 1 + 附录 D.1 核实，把上一条 deepen 里按通识写的 MLA 内部机制逐条坐实：

- **新增 `wiki/sources/deepseek-v2.md`**：来源页。236B/21B MoE、128K、8.1T tokens；MLA 低秩 KV 联合压缩（只 cache latent $c^{KV}$）、矩阵吸收（$W^{UK}→W^Q$、$W^{UV}→W^O$，即「MQA mode」数学来源）、query 也低秩压缩、decoupled RoPE（额外 multi-head decoupled query + 全 head 共享 decoupled key 承载 RoPE，因 RoPE 阻断吸收）、KV cache Table 1（MLA = $d_c+d_R\approx\tfrac{9}{2}d_h$，等效 GQA-2.25 组但性能 > MHA）、附录 D.1（MHA>GQA/MQA 消融 = MLA 走压低秩而非减头数的动机）、KV −93.3% / 吞吐 5.76×。
- **升级 `wiki/concepts/multi-head-latent-attention.md`**：删掉「内部机制按通识写、待核实」caveat，新增「机制细节（已据 DeepSeek-V2 原文核实）」小节；跨报告信号补 V2 起源行；待追问从「V2 不在 raw」改为更细的复现追问（$d_c/d_c'/d_R$ 取值、附录 D.2 数字、V2→V3→V3.2→V4 演进表）。
- **`wiki/index.md`**：来源栏新增 DeepSeek-V2 条目。

核实结论：上一轮按通识写的几条（低秩压缩、矩阵吸收→MQA、decoupled RoPE、等效 GQA-2.25、性能 > MHA）全部与原文一致，无需订正。`raw/` 仅新增 V2 PDF（用户提供），未改其他。

## [2026-06-20] deepen | 讲清 MLA query 下投影为何只为训练显存

用户连环追问 MLA 的两种 mode 与维度细节，落点在「Q 不用省 cache，那 Q 下投影到底图什么、是不是减计算量、何以见得」。回 `raw/DeepSeek-V2 ...pdf`（pdftotext 抽取后 grep）核实 §2.1.2 原文 line 412–416：

> "Moreover, in order to reduce the activation memory during training, we also perform low-rank compression for the queries, **even if it cannot reduce the KV cache**."

据此修订两页，把「Q 压缩省 activation 显存」从一句带过升级为可追溯的解释：

- `wiki/concepts/multi-head-latent-attention.md`「机制细节」：原单行 bullet 扩成带四条子项的说明——(1) 纠正「激活显存 ≠ 计算量」（论文盯的是 memory 非 FLOPs，低秩顺带省 FLOPs 但非卖点）；(2) V2 展开 query = 128×128 = 16384 维，比 hidden 5120 大 3.2×，故特别吃显存；(3) 机制 = 1536 维 latent 作「细腰」+ recomputation，只常驻小 latent、backward 重算大 query（~10×）；(4) 旁证 V2-Lite「does not compress the queries」。
- `wiki/sources/deepseek-v2.md`：query 压缩 bullet 补 §2.1.2 原文引用、16384 维细节、V2-Lite 旁证，并把 $d_c'$ 标实为 1536。

求证副产物：确认 Read 工具其实能按页读 PDF（`pages` 参数），此前一直用 pdftotext 是工作流选择而非能力限制；表格/公式保真场景该优先 Read 按页读。`raw/` 未改。

## [2026-06-20] maintenance | 降级 MLA query 压缩的「重算」机制为推测

承接上一条 deepen。用户质疑「重计算是论文写的还是推断的」，复核后确认是**我表述过头**：line 412 只给「query 压缩 → 省 activation memory」结论、未给机制；line 1118 的 recomputation 是**块级通用训练技巧**，论文**没把它专门挂到 query 压缩上**。「存细腰 latent + backward 重算大 query」是我补的因果链接，非原文机制，且实际工程中 gradient checkpointing 多在 transformer block 级别整体包、未必 query 专属。

又确认此环境 WebSearch 返回空（US-only，不可用），且即便可用，第三方博客讲解对「作者意图」是二手推断、易循环，不能升级证据等级；官方 modeling 代码也大概率只见块级重算。故不强凑佐证，改为如实标注：

- `wiki/concepts/multi-head-latent-attention.md`：机制细节里删掉「**机制**：…重算大 query」那条，改为一句「具体怎么省论文没展开，推测见待追问」；待追问首条新增完整说明（推测内容 + 为何只是推测 + 求证难点）。
- `wiki/sources/deepseek-v2.md`：query 压缩 bullet 删去「1536 latent 作细腰 + recomputation…重算大 query」断言，改为只留原文确证部分（结论 + 16384 维 + V2-Lite 旁证），机制以括注形式指向概念页待追问。

教训：把「论文结论」与「我对机制的推断」混写成同一句，是这页「已据原文核实」标题下不该有的。`raw/` 未改。

## [2026-06-20] deepen | 厘清 V3.2「MQA mode」的两条轴，订正「DSA 选 MQA 因长上下文友好」

用户追问 DSA 训练用哪种 mode。回 `raw/DeepSeek-V3.2.pdf` 核实后定论：「MQA mode」横跨两条不该混的轴——compute form（MHA/MQA 等价算法形态，训练/prefill 走 MHA、decode 走 MQA）与 selection 结构（DSA 让 latent 跨所有 query head 共享，是 kernel 效率要求）；masked-MHA-mode simulate DSA 是两轴正交的铁证。

- `wiki/concepts/multi-head-latent-attention.md`：新增「『MQA mode』一词横跨两条轴」小节 + 训练/decode × compute-form/selection 的 2×2 表（钉死：选哪些 token 恒共享一份 latent；latent 怎么算 → 训练展开、decode 吸收）。
- `wiki/sources/deepseek-v32.md`：**订正**原「DSA 选 MQA mode 因长上下文友好」——属臆测、无原文支撑；真实理由是 kernel 级 KV 跨头共享。

详细推导与原文锚点见两页正文。短 prefill 展开形态有原文确证，长上下文训练形态留白。`raw/` 未改。

## [2026-06-20] verify | 交叉检验 MLA「MQA mode / MHA mode」两条轴

用户要求对 MLA 页的两条轴论述做外部交叉检验，再回 V3.2 原文复查。外部源（中文技术博客，provenance 链接已落到 MLA 页正文）支持 5 条论断中的前 4 条：①正交两条轴、②矩阵吸收后退化 MQA、③训练展开/decode 吸收、④展开 vs 吸收算力随 seq_len 有 crossover（外部资料更精确，据此把原静态二分升级为 crossover 描述）。第⑤条（「MQA mode」横跨两条轴 + DSA top-k 跨头共享）博客层面无人这样拆分，但回 `raw/DeepSeek-V3.2.pdf` 逐句复查**全部坐实**，属本页原创综合、无需订正。

- `wiki/concepts/multi-head-latent-attention.md`：crossover 段补 blockquote（算力是 seq_len 函数）+ 第⑤点补原文锚点和「kernel 约束 → MQA mode」因果，并标注「论文一手坐实、博客层面原创综合」。

`raw/` 未改。

## [2026-06-20] verify | V3.2 训练形态追问 + V3.1-Terminus 背景（Tavily 交叉验证）

承接 MLA 两条轴。用户问「V3.2 是不是也用 MHA mode 训练」「要不要搜 V3.1」。回 `raw/DeepSeek-V3.2.pdf` 复查：论文**从未直接陈述** V3.2 训练用 MHA mode（Figure 7 caption 主语是 V3.1-Terminus），故「V3.2 训练走 MHA 展开」是合理强推断而非明文，MLA 页留白成立、不填死。本会话新增 Tavily 搜索 API（key 在 `~/AppData/Local/hermes/.env`，国外源可达），坐实两点：V3.1-Terminus 是 V3.1 的 update、非新基座、架构未变（与论文「唯一架构改动是 DSA」咬合）；MHA/MQA mode 分工是 MLA 自 V2 的通用惯例、非 V3.1 独创。

- `wiki/concepts/multi-head-latent-attention.md`：轴一补「mode 分工是 MLA 通用惯例」+ 外部佐证 + V3.1-Terminus 定位（provenance 链接落正文）。
- `wiki/sources/deepseek-v32.md`：核心结论段补 blockquote 交代训练起点 V3.1-Terminus + 「唯一架构改动是 DSA」。

未建 V3.1-Terminus 页（不在 `raw/`、非主线模型）。`raw/` 未改。

## [2026-06-20] deepen | MLA 展开/吸收 crossover 定量推导 + V3.2 masked-MHA 解释

承接前一条对话。用户问「crossover 是什么」「V3.2 在 V3.1-Terminus 上训了什么」，并要求沉淀定量推导与短-prefill masked-MHA 的因果。

回 `raw/DeepSeek-V2 ...pdf` line 670–676 坐实 MLA 配置：$d=5120,\ n_h=128,\ d_h=128,\ d_c=512,\ d_c'=1536,\ d_R=64$。用 execute_code 数 prefill 主导 FLOPs（KV 上投影 + QK score + AV 三项），把「展开算力−吸收算力」写成 $\text{diff}(L)=A L+B L^2=L(A+BL)$：$A=+1.68\times10^7$（展开独有的 latent 上投影固定开销，线性）、$B=-4.9\times10^4$（展开 score/AV 每对维度 192/128 < 吸收 576/512，二次）。解出 **crossover $L^*=-A/B\approx341$ token**：短于它吸收省、长于它展开省、decode（L≈1）恒吸收省。这定量解释了 V3.2 为何对短序列 prefill 专门用 masked MHA（line 264）——几百 token 内 DSA 无收益、吸收上投影不划算，稠密 MHA 展开最省。

落盘（`raw/` 未改）：

- `wiki/concepts/multi-head-latent-attention.md`：留白段后新增「定量：crossover ≈ 341 token」小节——$A/B$ 系数表 + $L^*=-A/B$ + decode 恒吸收 + 一段把 V3.2 masked-MHA 升级为可解释取舍的 blockquote + 诚实限定（341 是数量级、只数三主导项；V2 配置，V3.2 长序列走 DSA 稀疏后语义改写）。
- `wiki/sources/deepseek-v32.md`：「短上下文优化」补 blockquote 解释「为什么短 prefill 走稠密 MHA 而非 DSA/吸收」+ 反链 MLA 页 crossover 小节。
- V3.2 两阶段训练（dense warmup 2.1B 只训 indexer / sparse training 943.7B 全参，含步数/LR/stop-gradient/masked-MHA）此前已在 `deepseek-v32.md` § 训练方案完整沉淀，本轮未重复，仅补 crossover 因果。

## [2026-06-21] refactor | MLA 页结构重排 + 回原文核对引用 + 去工具痕迹

用户反馈 `multi-head-latent-attention.md` 结构杂乱，要求重构（保留全部事实/引用不动）；随后要求回对应 paper 核对引用并优化表述，去掉 `pdftotext`/行号等内部抽取痕迹。

- **结构重排**：定义节瘦身（只留两条正交轴 + MLA≈带低秩补偿的 MQA）；机制细节（低秩压缩/矩阵吸收/query 压缩/Decoupled RoPE/等效 GQA-2.25/效率数字）上移到深水区之前；「两种 mode 与『MQA mode』歧义」独立成节、收拢原先散在定义节与跨报告信号里的重复论述；crossover 独立成节。读者梯度变为 是什么→怎么实现→易混点→算力取舍→跨报告/意义/待追问。
- **回原文核对**：`pdftotext -layout` 抽取 `raw/DeepSeek-V3.2.pdf` 与 `raw/DeepSeek-V2 ...pdf`（临时 txt 用后即删，`raw/` 未改）。V3.2 五处英文引文逐字核对无误；V2 坐实章节号（MLA=§2.1、KV 压缩 §2.1.2、RoPE §2.1.3、cache 对比 §2.1.4）与维度 $d_c=512,\ d_c'=1536,\ d_R=64,\ n_h=d_h=128$、93.3%/5.76×。
- **去工具痕迹**：删除「逐句复查 / pdftotext -layout 抽取」「(2026-06-20)」等内部说明；所有 `line NNN` 行号换成读者向定位（Figure 7 caption、§ Instantiate DSA Under MLA、效率讨论一节）；裸 `raw/DeepSeek-V3.2.pdf` 路径改为指向来源页的链接。
- **顺手优化**：待追问里「$d_c'$ 未抄全」一条升级为「已据原文坐实 $d_c=512/d_c'=1536/d_R=64$、V2-Lite 不压缩 query」。

落盘：`wiki/concepts/multi-head-latent-attention.md`（事实/引用零改动，仅重排 + 改出处标注方式）。

## [2026-06-21] deepen | wiki 图文化试点：MLA 页内嵌 Figure 7 原图

用户提出纯文字 wiki 能否进化到图文交错并先做效果试点（全库此前 `![` 图片语法为 0，却密集引用看不见的 `Figure/Table`）。新引入 PyMuPDF 抽图，把 MLA 页一次图文化到位，跑齐三种素材类型：

- **矢量图（Fig 7，MHA/MQA mode 互转）**：切到 `wiki/assets/deepseek-v32/fig7-mha-mqa-mode.png`，内嵌「两种 mode」节。
- **概念示意图（V2 Fig 3，四种注意力对比）**：切到 `wiki/assets/deepseek-v2/fig3-mha-gqa-mqa-mla.png`，内嵌定义节「正交两条轴」处，直接图证「只 cache latent / KV −93.3%」。
- **表格（V2 Table 1）**：判定走 Markdown 重排而非截图（可 `rg`、公式 LaTeX、零体积），置于 Table 1 引用处。

每张图均经视觉模型逐框核对、与现有论述一致，属回一手图的 tier-1 确证；alt 文本写成完整图注以备失链降级。`wiki/assets/` 进版本控制（`raw/` PDF 仍忽略）。事实/引用零改动，`raw/` 未改。抽图方法已沉淀为 schema 约定（见下条 maintenance + CLAUDE.md「Figures」节）。

## [2026-06-21] maintenance | 图文化升级为正式约定（AGENTS.md / CLAUDE.md / skill 同步）

试点（同日两条 deepen）效果获认可后，把图文化从一次性做法固化为 schema 约定：

- **AGENTS.md + CLAUDE.md** 各新增「Figures & visual material (图文化)」小节（两文件同步、措辞对齐）：assets 目录规范（`wiki/assets/<source-slug>/<figure-slug>.png`、进版本控制、不留孤儿）、PyMuPDF 抽图方法（矢量图 `get_pixmap(clip)` 300 DPI + `get_textbox(clip)` 校验边界、不轻信 `get_drawings()` bbox）、纯文字表格走 md 重排、alt 文本即完整图注、内嵌 `raw/` 图为 tier-1 原文确证而 `vision_analyze` 仅为辅助且不入读者向正文。两边写回检查单各加一项「引用的图是否已内嵌 + assets 无孤儿」。
- **`.agents/skills/llm-wiki/SKILL.md`** 加精简版「Figures」小节，指回两份 schema。
- **顺手修订两处过期声明**：CLAUDE.md 原写「`.obsidian/` is committed / 浏览须 Obsidian-friendly」与 `[2026-06-19] maintenance`（`.obsidian/` 已 git-ignore）矛盾，改为「可作 Obsidian vault 浏览但 `.obsidian/` 已忽略、链接用相对 Markdown 路径」；「What not to do」里「today there is none（无任何 tooling 依赖）」更新为「唯一依赖 PyMuPDF」。

PyMuPDF（`fitz`）正式登记为本库唯一 tooling 依赖。本轮仅改 schema 文档与 skill，未动任何 `wiki/` 内容页，`raw/` 未改。

## [2026-06-21] maintenance | 修复坏引用语法（`^[...]` → 行内链接）+ 清方括号雷

用户发现引用渲染异常（`prop` 黑 / `erly` 紫断裂）。根因：本库此前用的 `^[url 标题]` / `[^url 标题]` **不是合法 Markdown**——`^[...]` 是 Pandoc 行内脚注扩展（GitHub/Obsidian 不认），`[^...]` 标准脚注需短标识符 + 文末配对定义，直接塞 URL+标题会乱渲染；叠加 `prop[erly]` 里的 `[erly]` 被当成链接文本染色。

全库扫 `^[` / `[^`：读者向页面仅 3 处，全在最近 MLA 追问线。统一改为方案 A（正文行内链接 `（来源：[标题](url)）`，任何渲染器通用）：

- `multi-head-latent-attention.md`：轴一 bullet 两条外链（Lior Sinai / V3.1-Terminus 公告）+ `prop[erly]` 还原为 `properly`；crossover 段吴建明博客外链。
- `deepseek-v32.md`：V3.1-Terminus blockquote 末外链；顺手清掉同处方括号雷 `continued [training]` → `continued training`、去掉内部行号痕迹 `line 115–116`。

`wiki/log.md` 历史条目里的 `line NNN` 是时间线追溯记录、非读者向正文，按惯例保留。事实/引用对象未变，仅修语法与表述，`raw/` 未改。

防复发：在 AGENTS.md（Coding Style 段）+ CLAUDE.md（Conventions 段）各加一条约定——外部来源一律用行内链接 `（来源：[标题](url)）`，**禁用 `^[...]` / `[^...]`**，并禁止 `prop[erly]` / `continued [training]` 这类 prose 内裸方括号（会被当链接语法染色），省略词补全改用中文括注。这套坏语法最近三处出自同一追问线，立约定即为防下次 deepen 加引用时再写出来。

## [2026-06-21] maintenance | 新增 wiki/TODO.md，分离"待办"与"时间线"

把"未来批量回填其他页关键图"这类未完成工程待办从 log 里分离出去——log 是已完成动作的时间线，不该混进未做的计划。

- 新增 `wiki/TODO.md`：图文化批量回填清单（已知引用待处理：V3.2 Figure 3 成本图、V3.2 Table 10 indexer 配置、MiMo Table 7 MOPD 对比、V2 附录 D.1 消融；待勘查：DSA / MSA / Agent-Swarm / MoonViT 等架构图是否值得内嵌），附回填规则与按页推进提醒。
- `index.md` 顶部加维护备忘行，指向 log.md（时间线）与 TODO.md（待办）。
- AGENTS.md（Project Structure）+ CLAUDE.md（three layers）各点明 `TODO.md` 用途、明确"计划写 TODO、不写 log"。

`raw/` 未改，无内容页事实变动。

## [2026-06-21] ingest | Kimi Linear、Gated Attention 技术报告

新增两篇线性注意力 / 注意力门控论文到 `raw/`：

- `raw/Team 等 - 2025 - Kimi linear An expressive, efficient attention architecture.pdf`（KDA，arXiv:2510.26692）
- `raw/Qiu 等 - Gated attention for large language models...pdf`（Qwen 团队，NeurIPS 2025）

这是知识库首次纳入**线性注意力（RNN-state）**与 softmax **注意力门控**两条与现有稀疏注意力正交的路线。新建：来源页 `sources/kimi-linear.md`、`sources/gated-attention.md`；模型页 `models/kimi-linear.md`（48B-A3B，纯文本研究模型）；概念页 `concepts/linear-attention-and-delta-rule.md`（朴素线性→DeltaNet→GDN→KDA 演进链）、`concepts/attention-gating.md`（门的五维设计空间、非线性 + 去 sink）。

更新：`efficient-long-context-attention.md` 把「四条路线」扩成「五条」（新增线性/混合行）；`multi-head-latent-attention.md` 跨报告信号加 Kimi Linear（MLA 被稀释成 1/4 NoPE 全局层这一新用法）；index 加 2 来源 / 1 模型 / 2 概念。交叉引用双向补齐。机制均据两篇 PDF 原文核实；`raw/` 未改。

## [2026-06-21] ingest | Gated DeltaNet 原文 + Qwen3-Next 系采用证据

接上一条 ingest，补齐 Gated Attention / 线性注意力的**前身与下游采用谱系**。新增三篇到 `raw/`：

- `raw/Yang 等 - 2025 - GATED DELTA NETWORKS...pdf`（GDN，ICLR 2025，KDA 与 Qwen3-Next 系线性层的一手前身）
- `raw/Cao 等 - 2026 - Qwen3-coder-next technical report.pdf`（采用证据，编码 agent）
- `raw/Team - 2026 - Qwen3.5-omni technical report.pdf`（采用证据，全模态）

新建来源页 `sources/gated-delta-net.md`（gated delta rule = 门控快速清空 + delta 定向更新互补，tier-1 原文）、`sources/qwen3-coder-next.md`、`sources/qwen3.5-omni.md`（两者均「继承 Qwen3-Next/Qwen3.5 架构、未重新推导」，标为采用证据而非机制 tier-1 来源）。

更新概念页：`linear-attention-and-delta-rule.md` 把 GDN 一环从二手转述升级为原文确证、跨报告信号加 Qwen3-Next 系（全局层用 gated attention 而非 MLA 的对照）；`attention-gating.md` 补 NeurIPS 2025 Best Paper、采用模型表（Qwen3-Next / Qwen3-Coder-Next / Qwen3.5-Omni / 非 Qwen 的 Trinity Large）、Kimi Linear=「换掉 gated-attention 全局层为 MLA」的第三方分析。`sources/gated-attention.md` 补 Best Paper 与采用谱系。index 加 3 来源。外部佐证（Sebastian Raschka 两篇）用行内链接标注，与「继承非重验」的限定一并写明；`raw/` 未改。

## [2026-06-21] deepen | linear-attention 页图文化回填（嵌 4 图 + 1 表重排）

按 `TODO.md` 的 linear-attention 回填清单，给该批纯文本页补图。新增 `wiki/assets/kimi-linear/`、`wiki/assets/gated-delta-net/` 两目录共 3 张 PNG（300 DPI，PyMuPDF 渲染页区裁剪，crop 框据 caption y0 + 图内标签位推导、`get_textbox` 验证、`vision_analyze` 核对像素）：

- `concepts/linear-attention-and-delta-rule.md`：嵌 Kimi Linear Figure 3（模型架构 N×KDA+1×MLA）、Figure 2（KDA vs DPLR kernel 时间）、Gated Delta Figure 1（H1/H2 混合栈 + Gated Delta Rule block）。
- `sources/kimi-linear.md`：嵌 Figure 1（Pareto 性能/加速图）；Table 1（3:1 hybrid ratio 消融）**重排为 Markdown 表**（含 output gate / conv 消融行，纯文字表不截图）。

两处校正：原 TODO 把 Figure 3 记成「Neural Parameterization」——那是 p4 段落标题，真正 Figure 3 是 p5 模型架构图；图中只标 N×/1×，3:1 是正文 N=3（已在图注写明）。GDN Figure 1 经 vision 核实 H2 = Mamba2+GDN+SWA（非「Mamba2 替代 SWA」），图注照此写。`gated-delta-net.md` 的关键架构图即已嵌概念页（双向引用经该页链回满足）；两个 Qwen 采用页无一手机制架构图，跳过。事实/引用零改动，仅新增图片与表格重排；`raw/` 未改。

## [2026-06-21] ingest | 新建 Qwen3-Coder-Next、Qwen3.5 模型页（HF config 坐实架构）

补上两个 Qwen 模型的模型页——此前只作为 source（采用证据）入库、缺关键架构事实。架构字段改由 **HuggingFace 官方 `config.json` 坐实**（域名可达，逐字段拉取）：

- `models/qwen3-coder-next.md`：79.7B 总参（HF safetensors 计 79,674,391,296）/ ~3B 激活，48 层，`full_attention_interval=4` 即 **3 GDN : 1 gated full-attention**，512 expert 选 10，hidden 2048，head_dim 256，上下文 262,144，纯文本（config 无 `vision_config`）。
- `models/qwen3.5.md`：Qwen3.5 多模态 hybrid MoE 家族页（397B-A17B≈403B / 122B-A10B / 35B-A3B / 27B dense 等）。`layer_types` 字段**逐层列出** 3 linear_attention + 1 full_attention，坐实 3:1；config 含 `vision_config`/`video_token_id` → 多模态。Qwen3.5-Omni 即以此为基座。

回填双向引用：两个 source 页（qwen3-coder-next / qwen3.5-omni）把「比例/模态待核实」的待追问降级为「已据 HF config 坐实」并加 `## 相关页面` 指向模型页；`attention-gating.md` 采用表、`linear-attention-and-delta-rule.md` 跨报告信号的 Qwen 提及改指模型页；index 模型区加 2 条。架构事实标注来源为 HF config（外部佐证，非 raw/ 报告原文）。注：当初这条曾写「需补 Qwen3.5 base 报告」，**事后核实该 base 报告并不存在**（Qwen3-Next 只有博客、Qwen3.5 只有 Omni 报告 + 权重）——见后一条 maintenance 修正。`raw/` 未改。

## [2026-06-21] maintenance | 修正「Qwen3.5/Qwen3-Next base 报告」幻觉

用户指出：所谓「Qwen3.5 base 报告」根本还没写、「Qwen3-Next base 报告」也不存在——前一条 ingest 把它们当成「待补的 tier-1 文献」是凭空臆想。Tavily 核实：Qwen3-Next 仅 2025-09 官方博客 + HF 权重（无独立 report）；Qwen3.5 系列目前只有 Qwen3.5-Omni（arXiv:2604.15804）+ HF 权重；arXiv:2603.00729 是 **Qwen3-Coder-Next**（已入库那篇），非 Qwen3-Next base。

改两处死任务：`sources/qwen3.5-omni.md` 待追问从「需补 Qwen3.5 base 报告」改为「该 base 报告不存在，架构 tier-1 = HF config，文字级动机在 Gated Attention/GDN 原论文 + 博客」；`sources/qwen3-coder-next.md` 加一句澄清 Qwen3-Next 无独立报告、本篇是该家族首篇 report。前一条 ingest 日志补一句指向本修正。教训：标「待补某报告」前先确认该报告真的存在，别把模型名直接脑补成「应有同名技术报告」。`raw/` 未改，无内容事实变动（只改证据来源表述）。

## [2026-06-21] deepen | GDN block 读写状态拆解 + Qwen3-Next 官方博客入库

源于一轮关于 Qwen3.5 用的 GDN 的连续追问（KV-cache 收益 → 状态维度 $d_v\times d_k$ → 输入接口 → q/k/v/α/β 五支投影 → conv 作用 → q/k 不做 softmax 乘法 → q 读更新后状态 → 训练并行/推理递推 → Qwen 怎么用 GDN）。重读 `raw/` 的 GDN 原文（Yang 等 2025，§ 2.1/3.1/3.3/3.4 + 附录 S.1 消融）逐条核实后回填。

- **`concepts/linear-attention-and-delta-rule.md` 新增「GDN block 怎么读写那块状态」小节**（tier-1 原文确证）：固定矩阵 cache 与 $d_k/d_v$ 含义、outer-product 关联记忆的容量上限（memory collision）、输入即 hidden state（Llama macro architecture）、五支投影表（q/k/v 走 Linear→Conv→SiLU(→L2Norm)、α/β 仅 Linear）、short conv 是 depthwise causal 不升维（S.1 消融：去掉 ppl 27.35→28.95）、q/k 靠结合律换序不做 $QK^\top$、先写后读时序、训练 chunkwise 并行 vs decode 递推（SFT 同走并行路径）。

- **新增来源页 `sources/qwen3-next-blog.md` + raw 材料**：Qwen3-Next 无技术报告，提取 **Alibaba Cloud 官方博客镜像**（qwenlm.github.io 原页已 404 重定向 qwen.ai JS 站）正文存入 `raw/Qwen Team - 2025 - Qwen3-Next blog ….md`（清理导航噪音、正文逐字保留、文件头标注来源/日期）。博客坐实三件此前只 config 推断的事，升级为 tier-2 官方外部佐证：**3:1 = 官方原话「75% GDN / 25% standard」**、选 GDN 因 in-context learning 强于 SWA/Mamba2、全局层 output gating 去 Attention Sink/Massive Activation；另记 Zero-Centered RMSNorm + norm weight decay、512-expert（10+1）MoE、native MTP。注意区分：博客讲「Qwen 怎么组装架构」，**不是 GDN 机制本身**（机制 tier-1 仍在 GDN/Gated Attention 原论文，GDN 具体实现 tier-1 在 transformers modeling + HF config）。

- **双向引用**：`linear-attention-and-delta-rule.md` 跨报告信号、`attention-gating.md` 采用谱系、`models/qwen3.5.md`、`models/qwen3-coder-next.md`、`sources/qwen3.5-omni.md`、`sources/qwen3-coder-next.md` 均链向新博客页；`index.md` 来源区加 1 条。新博客页 `## 相关页面` 反向链回上游论文与两个模型页。`raw/` 仅新增博客 md（git-ignored，与 PDF 同性质），未改任何已有源。

## [2026-06-21] deepen | delta rule 命名溯源 + 「Qwen 机制沿用 GDN」钉一笔

承接上一条的连续追问，给 `concepts/linear-attention-and-delta-rule.md` 演进链补两段（均不改既有事实，仅澄清/溯源）：

- **演进链表格上方加「delta rule 名字的来历」blockquote**：delta = 误差项 $\delta=y-\hat y$（Widrow-Hoff 1960 / LMS，GDN § 2.2 引），更新量正比于误差（错多少改多少，区别于 perceptron 定步长）；搬到线性注意力即 fast-weight/test-time SGD，$(S_{t-1}k_t-v_t)$ 就是那个 delta，$\beta_t$=学习率、$\alpha_t$=weight decay。tier-1（GDN § 2.2/3.1 + Widrow et al. 1960）。
- **表格下方加「Qwen 停在 GDN 这一环」blockquote**：澄清用户问的「机制沿用」——Qwen3-Next/3.5 线性层用 GDN 原版 gated delta rule（head-wise 标量门，**未升级到 KDA 的 channel-wise 门**），Mamba2 式 α 是 GDN 原文自定（非 Qwen 改动）。同时钉一句「机制沿用 ≠ 模块实现无改动」，把 value-head 2×、投影/输出门融合等实现层改动归到 transformers modeling 那一层，与机制演进链分开。判据=HF config + 代码（tier-1，非 Qwen 架构论文——那不存在）。

`raw/` 未改，无事实变动。

## [2026-06-21] deepen | 厘清「KDA 比 GDN 多存什么」（多的不是状态 cache）

承接关于 KDA 门维度的追问，给 `concepts/linear-attention-and-delta-rule.md` 的「KDA 的硬件效率」小节开头加一段澄清（tier-1，据 Kimi Linear 报告 + 状态方程推证）：channel-wise 门常被误读成「状态变大」，但需长期 cache 的 $S_t$（$d_v\times d_k$ per head）GDN=KDA 不变——门只是乘在状态上的衰减系数，不进入状态。增量在三处且都非状态显存：瞬时门值（×$d_k$，激活）、门投影参数（低秩压住）、DPLR 算子复杂度（专门化算子抵消）。对应把待追问那条「参数/显存增量多少」从全开放标成「已部分厘清 + 仍待补精确数字」。`raw/` 未改。
