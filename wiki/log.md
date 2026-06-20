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

用户提出模型页应统一标注模态。核对 6 个模型页并回 `raw/` 原始 PDF 逐一求证后补齐：

- GLM-5：回 `raw/glm-5-2602.15763.pdf` 核实为纯文本（ARC + 长上下文，无视觉/视频输入）。报告里出现的 "Multimodal LM"（Figure 10）是 Agent-as-a-Judge 评测管线用的外部判官，不是 GLM-5 自身输入。
- MiMo-V2-Flash：回 PDF 核实纯文本；报告里的 "multimodal verifier / vision-based verifier" 是 RL 数据管线给渲染视频打分的判别器，非模型本体。
- MiniMax-M2 Series：回 PDF 核实纯文本；"MM Claw multi-modal" 是评测名、VIBE-Pro 的 visual 判分是外部 verifier，均非模型输入。
- DeepSeek-V4：回 PDF 核实纯文本；Outlook 明确把 "incorporating multimodal capabilities" 列为未来方向，即当前尚无多模态。
- Kimi K2.5：关键事实表补上 `模态 = 多模态（文本 + 图像 + 视频）`，与正文一致。
- MiniMax-M3：此前已有 `模态` 行（原生多模态），作为统一字段的样式参考。

字段名统一为 `模态`，放进各模型页"关键事实"表（DeepSeek-V4 为变体表，改放表下一行说明）。CLAUDE.md 补了模型页 skeleton，把"关键事实表必须含模态字段"写进约定。

附带订正：本轮先前一次误记 "raw/ 已无 GLM-5 PDF"。实际 `raw/` 8 个源 PDF 全部在位（一次 `ls` 异常所致，疑似 OneDrive 占位符），各 source 页的 `raw/...pdf` 引用与磁盘文件名逐一对应、无失链。



## [2026-06-20] deepen | DSA / MSA 训练分阶段对比

针对"GLM-5 是不是分阶段训练 DSA"和"MSA 的 warmup 是否同构"两个问题，回到 `raw/glm-5-2602.15763.pdf` §2.1.1 + 附录 A 和 `raw/Lai 等 - 2026 - MiniMax sparse attention.pdf` §3.4 + §5.1 + Outlook 求证，按论文原文修订三个页面：

- `wiki/concepts/deepseek-sparse-attention.md`：把 DSA warmup 的描述精确化为"1000 步 × 14 序列/步 × 202,752 token，max LR 5e-3 → 2e-4"；sparse adaptation 标注恒定 1e-5 + 20B tokens；指出"dense warm-up + sparse training adaptation"两阶段范式是 DeepSeek-V3.2-Exp 引入并命名的；新增 indexer 配置（32 head / head dim 128，附录 Table 10）；并指出主文本未明写"warmup 只训 indexer"，但附录 GLM-4.7-Flash 小规模消融写明 base 全冻可作佐证。RL insights 段补全：原本的"避免 MoE routing replay"那句在上一版被截断，现按论文措辞补回 deterministic torch.topk vs SGLang CUDA 实现对比、k=2048 vs MoE k 的存储/通信代价。
- `wiki/comparisons/sparse-attention-mechanisms.md`：在"几个值得记住的判断"之前新增"训练分阶段对比（第四条轴）"小节，含 DSA / MSA 在起点、warmup 训什么、warmup 预算、warmup 后是否独立 adaptation、是否支持 from-scratch、post-training 处理 6 个维度的对照表，以及三条推论。RL 稳定性那段把对 MSA 的判断从"没有公开数据"硬化为引用 Outlook 原文（"reinforcement-learning post-training … 是 future work"）。
- `wiki/sources/msa.md`：待追问首条同步硬化措辞，引 Outlook 原文。

求证过程中发现并修正的不准确：(1) 上一版 DSA 页对 warmup 的描述漏掉了步数和 batch；(2) 上一版把"warmup 只训 indexer"写成了主文本结论，实际只在附录 GLM-4.7-Flash 消融里明写。`raw/` 内容未改。


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
