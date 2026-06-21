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

## [2026-06-20] deepen | 厘清 V3.2「MQA mode」的两条轴，订正「DSA 选 MQA 因长上下文友好」

用户顺着 MHA/MQA mode 追问到「MQA mode 适合哪个训练阶段」「DSA 训练到底用哪种 mode」。回 `raw/DeepSeek-V3.2.pdf` 核实（pdftotext 抽取后 grep + sed 读附录 A），定位三条原文：

- **line 147**（Instantiate DSA Under MLA）："we implement DSA based on the **MQA mode of MLA**, where each latent vector … shared across **all query heads** … At the kernel level, each key-value entry **must be shared across multiple queries** for computational efficiency."
- **line 1045**（Figure 7 caption）："For **DeepSeek-V3.1-Terminus**, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding."（限定 dense 基座）
- **line 248**："for short-sequence prefilling, we specially implement a **masked MHA mode to simulate DSA**."

核实结论：「MQA mode」在 V3.2 里横跨两条**不该混的轴**——(轴一·compute form) MHA/MQA 是同一注意力的等价算法形态，训练/prefill 用 MHA、decode 用 MQA；(轴二·selection 结构) 「DSA based on MQA mode」指 latent 跨所有 query head 共享、top-k 所有 head 选同一组，是 kernel 效率要求，**非训练算术**。line 248 的 masked-MHA-mode 是铁证：DSA 下 prefill 的 compute form 仍是 MHA，与 selection（DSA/MQA 共享）正交。

订正与修订：

- `wiki/concepts/multi-head-latent-attention.md`：在「两种 mode」段后新增「『MQA mode』一词横跨两条轴」小节（三 bullet + 长上下文训练形态的留白）；跨报告信号 V3.2 行把旧表述明确归到「轴二·selection 结构」。
- `wiki/sources/deepseek-v32.md`：line 29 **订正**原「DSA 选 MQA mode 因其长上下文效率更友好」——此说把两条轴混了、无原文支撑；改为 kernel 效率（KV 跨头共享）+ 两条轴正交（masked MHA mode 佐证）。

发现的不准确：上一版 `deepseek-v32.md` 与 MLA 概念页都写了「DSA 选 MQA mode 因长上下文友好」，属臆测；真实理由是 kernel 级 KV 跨头共享，且「选 MQA mode」本身是 selection 结构而非 compute form 取舍。`raw/` 未改。

补充（同一追问线收尾）：用户把问题收窄到最具体一问「DSA 训练时那份 latent 是展开还是吸收」。在 MLA 概念页「两条轴」小节后补一张训练/decode × compute-form/selection 的 2×2 表，钉死结论：**选哪些 token 恒共享（一份 latent）；latent 怎么算 → 训练展开（MHA mode）、decode 吸收（MQA mode）**。训练展开两动因：compute-bound 下 128 维点积 < 吸收 512 维；训练需 $W^{UQ}/W^{UK}/W^{UV}/W^O$ 各自 live 让梯度分别回流（吸收是推理期预合并固定权重的优化）。短 prefill 展开形态有原文确证，长上下文训练形态仍留白。

## [2026-06-20] verify | 交叉检验 MLA「MQA mode / MHA mode」两条轴

用户要求对 MLA 概念页最新两段（MQA mode / MHA mode）做外部交叉检验，再回 V3.2 原文复查。检验通道说明：本会话 `web_search` 工具不存在（仅 `delegate_task` 的 `web` toolset 有，但 glm-5.2 子代理反复在首次工具调用前空返回，不可用）；改用 `terminal + curl`，DuckDuckGo/HF 不可达，**搜狗 + cn.bing 可达**。命中并抓取的中文技术源：博客园「罗西的思考《探秘Transformer(28)—DeepSeek MLA》」(cnblogs/rossiXYZ/18827618)、吴建明「MLA计算流全图解&吸收矩阵对比分析」(cnblogs/wujianming-110117/19112429)。

外部交叉检验结论（5 条论断）：

- ① MLA vs GQA/MQA 正交两条轴（减头 vs 压秩）— ✅ 支持。
- ② 矩阵吸收 $W^{UK}→Q$、$W^{UV}→O$，吸收后退化 MQA — ✅ 支持（罗西文目录 2.2 权重吸收 / KQ 合并 / VO 合并 / 3.3.3 MQA形式）。
- ③ 训练/prefill 用 MHA 展开、decode 用 MQA 吸收 — ✅ 支持（罗西文「2.2.4 训练 MHA 不合并」）。
- ④ 展开 vs 吸收的算力权衡 — ✅ 支持，且外部资料更精确：吴建明实测 decode 吸收恒省，prefill 随 $seq\_len$ 增长「展开−吸收」差值由正转负，存在 crossover。**据此补进 MLA 页**（原本只写成 128 维 < 512 维的静态二分）。
- ⑤ 「MQA mode」横跨两条轴 + DSA top-k 跨 head 共享 — 博客层面 **NOT-FOUND**（无人这样区分），但回 `raw/DeepSeek-V3.2.pdf` 逐句复查 **全部坐实**：line 149–152（kernel 约束→MQA mode，latent shared across all query heads）、line 1071–1072（MHA train/prefill、MQA decode，限 V3.1-Terminus）、line 264（masked MHA mode simulate DSA）。

修订（仅 `wiki/concepts/multi-head-latent-attention.md`，`raw/` 未改）：

- 第④点：2×2 表后新增一段 blockquote，讲清「展开 vs 吸收算力是 $seq\_len$ 的函数、有 crossover」，附吴建明文 provenance 链接。
- 第⑤点：三条 bullet 补上 V3.2 原文行号锚点 + 补全「shared across all query heads of the query token」整句 + 点明「先 kernel 约束 → Therefore MQA mode」的因果；小节末加 blockquote 标注「论文一手坐实，但博客层面无人这样拆分，属本页原创综合」。结论：第⑤点**无需订正**，原标注「已据原文核实」名副其实。

## [2026-06-20] verify | V3.2 训练形态追问 + V3.1-Terminus 背景（Tavily 交叉验证）

承接 MLA 两条轴。用户先问「V3.2 是不是也用 MHA mode 训练」，再问「要不要搜 V3.1」。

回 `raw/DeepSeek-V3.2.pdf` 复查训练形态：论文**从未直接陈述 V3.2 训练用 MHA mode**——Figure 7 caption（line 1071–1072）主语是 V3.1-Terminus；V3.2 自身只写了 selection 结构（DSA based on MQA mode = latent 跨头共享）+ 两阶段配方（dense warm-up 2.1B「keep dense attention」只训 indexer / sparse training 943.7B 全参 adapt）+ 推理侧短 prefill 用 masked MHA mode（line 264）。结论：「V3.2 训练走 MHA 展开」是**合理强推断而非明文**，page line 31 的留白成立、不填死。

本会话**新增 Tavily 搜索 API**（key 在 `~/AppData/Local/hermes/.env` 的 `TAVILY_API_KEY`，curl POST api.tavily.com/search 即可，国外源可达），取代此前不可用的 web_search / 子代理 web。两个问题用 Tavily 坐实：

- **V3.1-Terminus 定位**：官方公告（api-docs.deepseek.com/news/news250922）+ OpenRouter/Fireworks/SambaNova/Medium 一致——它是 V3.1 的 update（语言一致性 + agent 能力），**非新基座、架构未变**，沿用 hybrid reasoning。与论文 line 115–116「the only architectural modification of V3.2 is DSA」咬合：V3.1-Terminus 注意力 = 不带 DSA 的标准 MLA。
- **MHA/MQA mode 通用性**：Lior Sinai 博客明确「吸收只能推理期做，训练必须各权重分开让梯度回流」（讲 MLA 本身、V2 起），TransMLA 确认 MLA「introduced with DeepSeek V2」。→ mode 分工**不是 V3.1 独创，是 MLA 通用惯例**，论文只是拿 dense 训练起点 V3.1-Terminus 举例。

落盘（`raw/` 未改）：

- `wiki/concepts/multi-head-latent-attention.md` 轴一 bullet 下新增子项：mode 分工是 MLA 自 V2 的通用惯例、非 V3.1 独创 + Lior Sinai「训练权重分开让梯度回流」外部印证（独立佐证既有第二动因）+ V3.1-Terminus 定位 + 两条 provenance 链接。
- `wiki/sources/deepseek-v32.md` 核心结论段新增 blockquote：交代训练起点 V3.1-Terminus 是什么（官方公告口径）+ 论文 line 115–116「唯一架构改动是 DSA」+ Figure 7 caption 写成「For V3.1-Terminus」实为 MLA 通用 compute-form 惯例。
- 未新建 V3.1-Terminus 来源页/模型页：它不在 `raw/`、非本库主线模型，仅作为 V3.2 训练起点在 V3.2 来源页注明定位即可。

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

用户提出纯文字 wiki 能否进化到图文交错，并先做效果试点。结论：图表信息密度最高却恰是单模态 LLM 的盲区，且全库此前 `![` 图片语法为 0、却密集引用 `Figure/Table` 这类看不见的视觉锚点。

试点管线（新引入 PyMuPDF=1.27.2）：

- **切图**：`raw/DeepSeek-V3.2.pdf` p20 的 Figure 7（MHA/MQA mode 互转图）是**纯矢量绘制**（`get_images()` 空），故按页面区域 `get_pixmap(clip)` 300 DPI 渲染裁剪，存到新目录 `wiki/assets/deepseek-v32/fig7-mha-mqa-mode.png`。用 `get_textbox(clip)` 抽图内文字层验证裁剪边界精确（含 (a)/(b) 子标题，排除上方附录标题与下方 caption）。
- **视觉核对**：`vision_analyze` 读图（首个 provider 拒图片格式不可用，换模型后可用、读图质量高）逐方框/箭头/变量复述，与 MLA 页现有论述交叉核对——矩阵吸收（$W^{UK}$ 移 query 侧、$W^{UV}$ 移 attention 输出后）、MHA per-head 展开 vs MQA latent 跨头共享、四处 `apply RoPE` 只在 $q^R/k^R$ 分量——**全部一致**，反而比原文字更精确，属回一手图的 tier-1 确证。

落盘：`wiki/concepts/multi-head-latent-attention.md`「两种 mode」节内嵌 Figure 7（图 + 原文 caption 引文），alt 文本即高质量图注（视觉模型读图 + 回原文核对）。事实/引用零改动，仅新增图片。`wiki/assets/` 不在 `.gitignore`（图需随 wiki 进版本控制，`raw/` PDF 仍被忽略）。`raw/` 未改。

待定（试点性质）：是否把图文化升级为正式约定（assets 目录规范、`![]` alt 文本与证据 tier 标注规则、写回检查单增「引用的 Figure 是否已内嵌」），以及是否批量回填其他页的关键图——待用户拍板后再动 AGENTS.md / CLAUDE.md。

补充（同日，效果获认可后续推）：在正式立约定前先多跑两种素材类型，把 MLA 页一次图文化到位：

- **概念示意图（矢量）**：`raw/DeepSeek-V2 ...pdf` p7 Figure 3（MHA/GQA/MQA/MLA 四种注意力对比，纯矢量）切到 `wiki/assets/deepseek-v2/fig3-mha-gqa-mqa-mla.png`，内嵌进 MLA 页**定义节**「正交两条轴」处。视觉模型核对：四子图完整、图例「Cached During Inference」= 斜线填充编码、MLA 唯一缓存 Compressed Latent KV、projection 上投影——直接图证 line 26-27 的「只 cache latent / KV −93.3%」。
- **表格（纯文字）**：V2 Table 1（每 token KV cache 对比）drawings 仅 2 条线，判定**走 Markdown 重排而非截图**——可被 `rg` 检索、公式 LaTeX 渲染、零图片体积。从原文 § 2.1.4 重排成 4 行 md 表，置于 line 26 Table 1 引用处下方。

至此三种素材类型跑齐：矢量图（Fig 7）、概念示意图（Fig 3）、表格重排（Table 1）。**经验沉淀**：(1) 矢量图 `get_images()` 为空、必须 `get_pixmap(clip)` 区域裁剪，边界靠 `get_textbox(clip)` 文字层校验；(2) `get_drawings()` 的 bbox 可能含页面外辅助路径、不可直接信，以图内文字块定边界；(3) 纯文字表格优先重排 md，不截图；(4) alt 文本写成完整图注（视觉模型读图 + 回原文核对），即使图失链也有 tier-1 文字降级。落盘仅 `wiki/concepts/multi-head-latent-attention.md` + 新增 2 图，事实/引用零改动，`raw/` 未改。

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
