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


## [2026-06-22] ingest | Qwen3、Qwen3-VL 技术报告

新增两份到 `raw/`：

- `raw/Yang 等 - 2025 - Qwen3 technical report.pdf`（arXiv:2505.09388，2025-05）——Qwen 系基座前作。
- `raw/Bai 等 - 2025 - Qwen3-VL Technical Report.pdf`（arXiv:2511.21631，2025-11）——Qwen3-VL 多模态。

**关键澄清（影响家族谱系）**：

- **Qwen3 是标准 GQA + RoPE + RMSNorm + SwiGLU 一脉**——hybrid linear attention / gated full attention 那一套是 [Qwen3-Next 博客](sources/qwen3-next-blog.md) 引入的下一代。此前 wiki 里 Qwen3-Next/3.5/3-Coder-Next/3.5-Omni 都暗指"在 Qwen3 基础上换 hybrid 栈"，但没有 Qwen3 基座页可指——这条 ingest 把这个空缺补上。
- **Qwen3-VL backbone 用的是 Qwen3 标准 GQA，不是 Qwen3-Next/3.5 hybrid**。这意味着 Qwen 系多模态有两条独立线：[Qwen3-VL](sources/qwen3-vl.md)（Qwen3 backbone，文本+图像+视频，256K）vs [Qwen3.5-Omni](sources/qwen3.5-omni.md)（Qwen3.5 hybrid backbone，加音频）——LLM backbone 不同。Qwen3-VL 摘要原文 "Built on the Qwen3 series" 已坐实。

新增 4 页：

- `wiki/sources/qwen3.md`：3 阶段 36T pretrain（30T+5T STEM+长上下文 4K→32K，RoPE ABF + YARN+DCA 推理 4×）、4 阶段后训练（Long-CoT cold start → Reasoning RL 仅 3995 query/170 步 → **Thinking Mode Fusion**（/think /no_think + budget 自然涌现）→ General RL），小模型走 Strong-to-Weak Distillation 完胜 RL（Table 21：Qwen3-8B on-policy distill 1800 GPU·h vs RL 17920 GPU·h，AIME'24 67.6→74.4，且 pass@64 也涨——RL 不涨）。Figure 1 后训练 pipeline 已嵌图（`wiki/assets/qwen3/fig1-post-training-pipeline.png`，PyMuPDF p9 clip + textbox 校验 + vision_analyze 核对四阶段顺序与轻量模型分支）。
- `wiki/models/qwen3.md`：8 个变体表（layers / Q/KV head / tie embed / context）逐行据 Table 1/2 核实；唯二注意力侧改动 = 去 QKV-bias + 加 QK-Norm；MoE 去 shared expert + global-batch load balancing loss + 128/8 routed。
- `wiki/sources/qwen3-vl.md`：三块架构升级——**Interleaved MRoPE**（t/h/w 跨频段交错，修 Qwen2.5-VL 的 spectral imbalance）、**DeepStack**（ViT 3 个层级 → 经 3 个独立 merger → residual add 到 LLM **前 3 层**，§ 2.2 原文"first three LLM layers"，不是每层都注）、**文本时间戳**（`<3.0 seconds>` token 替换 T-RoPE 的绝对时间 position id）。Vision encoder = SigLIP-2 SO-400M（小模型 SigLIP-2 Large 300M）+ 2D-RoPE + CoMP。4 阶段 pretrain（S0 67B align-merger / S1 ~1T@8K 全参 / S2 ~1T@32K / S3 100B@262144）。3 阶段后训练（long-CoT SFT + 知识蒸馏 + RL），agentic 走两阶段（10k cold-start SFT 在 Qwen2.5-VL-32B → 蒸到 120k → tool-integrated RL），三 reward = Answer Accuracy + Multi-Turn Reasoning + Tool-Calling（防 reward hacking 偷懒一次 tool call）。Figure 1 整体架构图已嵌（`wiki/assets/qwen3-vl/fig1-framework.png`，p3 上半 clip，vision 核对 DeepStack 注入到 Block 1/2/3 与文本时间戳 token 标注位）。
- `wiki/models/qwen3-vl.md`：6 变体（2B/4B/8B/32B dense + 30B-A3B/235B-A22B MoE）映射回 Qwen3 backbone 同名变体（注意 Qwen3-VL-2B 用的是 Qwen3-1.7B，2B 是含 vision encoder 总参）。每尺寸释 Instruct + Thinking 两份独立权重（与 Qwen3 base 的单权重双模式不同）。

反链与索引：

- `wiki/sources/qwen3-next-blog.md`：来源块加"前作 = Qwen3 报告 + 模型页"链接，钉死"本博客差异 = 标准 GQA → 3:1 hybrid + MoE 128/8 → 512/10+1 + QK-Norm → Zero-Centered RMSNorm"。
- `wiki/sources/qwen3-coder-next.md` + `wiki/models/qwen3-coder-next.md`：相关页面加 Qwen3 前作链接。
- `wiki/sources/qwen3.5-omni.md` + `wiki/models/qwen3.5.md`：相关页面加 Qwen3 前作 + Qwen3-VL 横向（两条多模态线 backbone 不同）链接。
- `wiki/index.md`：来源区加 2 条（Qwen3 / Qwen3-VL），模型区加 2 条。

写回清单：双向反链已通；2 张图均经 vision_analyze 核对，alt 文本写成完整图注备失链；assets 无孤儿（fig1 在源页与模型页均被指）；待追问均为真问题（QK-Norm 实现细节、DeepStack 选哪 3 ViT 层、Qwen3-VL/Omni 为何 backbone 不统一、Thinking Budget 自然涌现机制）。`raw/` 未改。

## [2026-06-22] deepen | 图文化批量回填：4 项已知引用 + 3 项勘查内嵌

按 `TODO.md` 清单逐项推进，共新增 7 张图片到 `wiki/assets/`、6 个页面被修改：

**已知引用回填（4 项）**：
- `sources/deepseek-v32.md`：嵌 **Figure 3**（H800 推理成本，prefilling+decoding 双子图，PyMuPDF p6 clip + textbox 校验 + vision 核对 V3.2 成本斜率远缓于 V3.1-Terminus）。
- `concepts/deepseek-sparse-attention.md`：**Table 10**（GLM-5 附录，indexer 32 head / dim 128）从单行文字重排为 Markdown 表，含 GLM-4.5 对照列。
- `concepts/multi-teacher-on-policy-distillation.md`：**Table 7**（MOPD 前后 student vs. best teacher）重排为 12 行 Markdown 表，含 teacher 类型标注（RL/SFT/Self）。修正 BrowseComp student 数字 45.4→44.9（原文 44.9）。
- `concepts/multi-head-latent-attention.md`：**附录 D.1**（DeepSeek-V2 Table 8，7B dense MHA>GQA>MQA 消融）重排为 Markdown 表，补在「压低秩而非减头数」论据处。

**勘查内嵌（3 项）**：
- `concepts/deepseek-sparse-attention.md`：勘查 V3.2 PDF → **Figure 2**（DSA 注意力架构，lightning indexer + top-k selector + MLA MQA mode 完整数据流）值得内嵌，已截图嵌架构概览段。
- `comparisons/sparse-attention-mechanisms.md`：勘查 MSA/IndexCache PDF → 嵌 **MSA Figure 1**（Index Branch + Main Branch + attention mask 可视化）到 MSA 行下；嵌 **IndexCache Figure 2**（标准 DSA vs IndexCache 伪代码对比）到 IndexCache 行下。
- `concepts/agent-swarm.md` + `concepts/multimodal-agentic-training.md`：勘查 Kimi K2.5 报告 → 嵌 **Figure 3**（Agent Swarm orchestrator + frozen subagents 并行架构）到 agent-swarm.md 定义段；嵌 **Figure 10**（agentic RL framework: Rollout Manager / Core Agent Loop / Inference+Training Engine）到 multimodal-agentic-training.md 的 Joint multimodal RL 段。

每张图均经 PyMuPDF 300 DPI 渲染 + `get_textbox(clip)` 校验边界 + `vision_analyze` 核对像素内容与描述一致。`TODO.md` 全部 7 项标记完成。`raw/` 未改。

## [2026-06-23] distill | On-Policy Distillation 5 家跨报告对比

提问驱动：「OPD / 在线策略蒸馏在已收录的哪些 LLM 里用到了」。回答时发现同名 OPD 在 5 家报告里分叉成三类截然不同的用法（融合多 teacher / 强到弱迁移 / RL 阶段间能力召回），且这层综合在已有页面里没有归属——是典型 distill 候选。

新增：

- `wiki/comparisons/on-policy-distillation.md`：5 家（MiMo MOPD / DeepSeek-V4 OPD / Qwen3 Strong-to-Weak / Qwen3-VL Strong-to-Weak / GLM-5 cross-stage）沿三轴对比——OPD 目的（融合 / 强到弱 / 召回）/ KL 形式（token-level vs full-vocab vs logit-level）/ pipeline 位置（替代 mixed RL / 替代 small-model 4 阶段 / 流水线最终阶段）。附 Qwen3-8B Table 21（off-policy → +RL vs +on-policy distill）完整复刻，论证 on-policy distill 1/10 GPU·h 同时 pass@64 涨（RL 不涨）。

源页回填反向链接（5 处）：

- `concepts/multi-teacher-on-policy-distillation.md`：页首加引用框指向对比页（"本页只覆盖 MiMo MOPD 单家"）。
- `sources/mimo-v2-flash.md`：MOPD 段下加 token-level KL + ORM 加权混合 + teacher 类型（RL/SFT/Self）的指针。
- `sources/deepseek-v4.md`：新增"后训练：OPD 替代 mixed RL"完整章节（之前只在 comparisons 顶表里一句话），写明 full-vocab vs token-level 取舍、hidden-state caching / teacher 排序调度 / TileLang kernel / FP4 / ZeRO 分片五项工程支撑。
- `sources/qwen3.md`：第 4 条核心结论后补 teacher 是 Qwen3-32B / Qwen3-235B-A22B（单 teacher）的关键事实。
- `sources/qwen3-vl.md`：后训练章节补 §4.3 Strong-to-Weak 小节，钉住"text-only data fine-tune LLM backbone（视觉模块不参与）"原文细节。
- `sources/glm-5.md`：on-policy cross-stage distillation 段下钉住"目的是 swiftly recover earlier stages 而非融合 teacher"，与另外两家划界。
- `index.md`：比较区第三条。

证据：5 处机制结论全部在 PyMuPDF 重读 raw/ PDF 后据原文 §/Table 校验——MiMo §4.1+§4.4 公式 5-9、DeepSeek-V4 §5.1+§5.1.2+§5.2.2、Qwen3 §4.5+Table 21、Qwen3-VL §4.3+§4.1、GLM-5 §3.5（含 reference [28] = Thinking Machines Lab on-policy distillation 博客，与 MOPD 同源）。`raw/` 未改。

写回清单：5 家源页反向链接全部双向通（对比页 → 各源页 ✓，各源页 → 对比页 ✓）；MOPD 概念页与对比页互引；index.md 比较区已加；无新增图（对比页主体是表与文，不依赖 raw 图）。

## [2026-06-23] deepen | 修正 GLM-5 在 OPD 对比页里的归类：多 teacher（纵向）而非单 teacher

提问驱动："GLM-5 不也是多个教师模型吗？" 重读 raw/glm-5-2602.15763.pdf §3.5 第一段确认：原文 "the final checkpoints from the preceding training stages serve as teacher models, where the training prompts are sampled from the corresponding teachers' RL training sets and mixed in appropriate proportions"——**teachers 复数，且 prompt 按归属阶段路由**。前一日 distill 条目里把 GLM-5 笼统写成"teacher 是上游 checkpoint"会让人误读为单 teacher，事实上它和 MiMo MOPD 一样是多 teacher × prompt 路由 × KL 当 advantage 的完整 OPD 结构。

修正：

- `comparisons/on-policy-distillation.md` 顶表 GLM-5 行：把 teacher 列从模糊的"上游 checkpoint"改成"**多 teacher**：SFT / Reasoning RL / General RL 各阶段 final checkpoint，prompt 按归属阶段路由"，KL 形式列写明"token-level KL log-ratio 当 advantage（§3.5 公式 2），GRPO group size = 1"。
- 同页轴一第三类正文：删掉"teacher 不是其他模型，而是自己流水线上游的 checkpoint"这种把"非外部"歧义读成"单 teacher"的表述；补充"GLM-5 同样是多 teacher × prompt 路由 × KL 当 advantage，差别不在 teacher **数量**，在 teacher **性质**"，引入「横向 teacher（领域专家，平行专长）vs 纵向 teacher（同一血脉不同代快照）」的精确区分——这才是 GLM-5 与 A 类的真正分歧。
- 同页轴三 GLM-5 段：补"多 teacher / 按 prompt 归属路由"、"GRPO group size = 1（因 advantage 直接来自 teacher gap，单 prompt 不再需要多 rollout）"的工程细节。
- `sources/glm-5.md`：同步把对比页引用框的措辞替换为"算法形式与 MiMo MOPD 高度相似，多 teacher × prompt 路由 × KL 当 loss"。

也额外修正了相关的两处认知误差（仅记录、不改页面，因页面里没出现）：(a) GLM-5 cross-stage distillation 更新的不是 SFT model，而是 General RL 末端 checkpoint（§3.5 原文 "as the final stage"）；(b) MOPD 等 OPD 的 KL loss 不是直接写 KL，而是把 reverse-KL log-ratio 当 advantage 塞回 GRPO 框架——数学上等价于最小化 KL(student ∥ teacher)，但 infra 上能完全复用 RL 训练栈，这是 GLM-5/MiMo 选择 token-level KL 路线的关键原因。`raw/` 未改。

## [2026-06-23] distill | OPD 数学依据：4 层论证 + 多 teacher 混采的边界

提问驱动："有没有什么数学依据？"——5 家报告里 OPD 的 loss 形式各异（MiMo 公式 7-8 token-level surrogate / GLM-5 公式 2 advantage 代换 / V4 公式 29 full-vocab weighted KL / Qwen3 §4.5 logit KL），但都落在同一个数学骨架下。这层共用骨架在任何单份报告里都没明写，对比页里也只提了"reverse KL"一笔，是典型 distill 候选。

落点选择：放进 `concepts/multi-teacher-on-policy-distillation.md` 新增「数学依据」一节（不是对比页——对比页讲分歧，不该塞共用基础），并修改概念页头部引导框为"本页以 MiMo 为骨架但数学依据一节适用所有家"。对比页轴二开头 + 相关页面区、index.md MOPD 行加跳转到该锚点。

新增章节结构（4 层 + 1 边界）：

1. **OPD loss = 最小化 reverse-KL** —— likelihood-ratio gradient 推导，给出 advantage = log(π_T/π_θ) 的代换；这是 GLM-5 公式 (2) 和 MiMo 公式 (8) 的来源，infra 与 RL 共用。V4 full-vocab KL 是同目标的精确版本（不简化为 token-level 估计）。
2. **reverse-KL 的 mode-seeking 性质** —— 与 forward-KL mass-covering 对比表；为什么对 GLM-5"召回早期能力"特别关键（student 不被强迫覆盖三阶段 mode 并集）；引 Bishop PRML §10.1.2、Minka 2005 α-divergence TR。
3. **on-policy 消除 exposure bias** —— 引 Ross et al. 2011 DAgger 的 O(T) vs O(T²) regret 证明；同时给出 Qwen3 Table 21 pass@64 涨（distill 拓宽分布）vs RL 不涨（只 sharpen）的数学解释。
4. **teacher 固定 → 良定义收敛** —— D_KL ≥ 0 单调有界，下界 0 当且仅当分布相等；这是 GLM-5 §3.5 敢用 "swiftly recover" 措辞的依据（RL 的 reward landscape 不允许 swift）。
5. **数学没闭合的边界：多 teacher 混采** —— 上述四层只论证「逼近一个固定 teacher」，不直接论证「多 teacher 同时召回不互相覆盖」。在 prompt 集不重叠的假设下退化为条件混合分布 π* = π_{T_k(x)}，前述论证逐 teacher 成立；prompt 集重叠时 routing 错位无数学保证（MiMo Table 7 里 BrowseComp −6.8、Arena-Hard Creative Writing −3.9 落后 best teacher 的可能来源）——这一步外包给数据 curation。

反向链接：

- `comparisons/on-policy-distillation.md` 轴二开头 + 相关页面区指向锚点。
- `index.md` MOPD 行摘要加锚点跳转，让"数学依据"在索引层就露出。

证据：第一、四层（reverse-KL likelihood-ratio gradient + 下界为 0）是 KL 散度的标准结果；第二层引 Bishop PRML §10.1.2 与 Minka 2005 MSR-TR-2005-173；第三层引 Ross et al. 2011 DAgger（AISTATS）。四层结论是公认数学事实，「四层合力解释 OPD 为什么 work」与「多 teacher 混采没闭合的边界」是本页原创综合（推断 / 本页原创综合 tier 3），但每条子论证均有 tier 1 / tier 2 支撑。`raw/` 未改。

## [2026-06-23] ingest | Thinking Machines Lab On-Policy Distillation 博客（Kevin Lu 2025-10-27）

提问驱动: 验证"Thinking Machines Lab on-policy distillation 博客"是否可达,以巩固之前 distill 数学依据节里的"reference [28]"。Tavily extract 抓回 46.6 KB 全文,远超之前推断的密度,且内容直接对接已收录 5 家技术报告(GLM-5 §3.5 ref [28]、MiMo §4 引用)。

落点选择: 独立成 `wiki/sources/thinking-machines-on-policy-distillation.md` (而非塞进 MOPD 概念页),因为它是 5 家技术报告共同引用的奠基性文献,独立成页能让对比页与机制讨论有共同祖宗页可指,长期复用价值大。raw markdown 全文存到 `raw/thinking-machines-on-policy-distillation-2025-10-27.md`(已 git-ignored,与 PDF 一致处理)。

新增 `wiki/sources/thinking-machines-on-policy-distillation.md`:

- 来源元数据: Kevin Lu 与 Thinking Machines Lab, 2025-10-27, DOI 10.64434/tml.20251026, Tinker cookbook 仓库链接。
- 核心结论 6 点: (1) 三方对照表(SFT off-policy+dense / RL on-policy+sparse / OPD on-policy+dense); (2) per-token reverse KL 公式 + reverse-KL 三性质(unhackable / mode-seeking / 降 exposure bias); (3) `O(1)` vs `O(N)` bits/episode 信息论效率, 7-10× steps / 50-100× compute; (4) Qwen3 Table 21 复现(60→70% AIME'24 约 150 步,RL 17920 GPU·h vs distill 1800 GPU·h); (5) Personalization 实验 = GLM-5 cross-stage 召回的直接 setup(Qwen3-8B + 内部文档 SFT → IF-eval 85→45% → 用原版 Qwen3-8B teacher distill → 召回到 83%); (6) phase-alternating 框架(Cobbe 2020 PPG)作 continual learning recipe。
- Pseudocode 4 步原文复刻(teacher sampling client / student rollout / `advantages = -reverse_kl` / 走标准 RL `forward_backward(loss_fn="importance_sampling")`)。
- 三个关键洞察整理: RL 在语义策略空间 search(类比科研发现 vs distill 类比论文宣讲)、RL 子网络脆弱(Mukherjee 2025)、forking tokens 实例(Wang 2025 高熵少数 token 主导有效 RL,teacher 罚分叉而非已注定错误答案)。
- 学术血脉钉死: DAGGER(Ross 2010)、Process Reward Modeling(Lightman 2023)、Agarwal 2023、MiniLLM(Gu 2023)、Qwen3(2025) 5 篇直接祖先;后向影响 GLM-5/MiMo 共两份。
- 待追问 4 条: discount factor 之争、同尺寸 teacher 反而更适合的边界、博客没涉及的多 teacher 路由、forking tokens 与 entropy bonus 的关系。

## [2026-06-23] deepen | OPD 数学依据扩展为 7 层论证

提问驱动: 上一轮 distill 的"数学依据"4 层是基于推断;拿到博客全文后发现可以补 3 层一手出处证据,把推断升级成原文确证。

`wiki/concepts/multi-teacher-on-policy-distillation.md` § 数学依据 OPD 为什么 work 扩展:

- 第二层(reverse-KL)补"unhackable"性质——博客原文「low KL always corresponds to a high probability of desirable behavior」, RL reward model 是学出来的 scalar 估计可能被 hack, reverse-KL 直接以 teacher 输出分布为锚没有学习中介。
- 新增第五层「`O(1)` vs `O(N)` bits/episode」——博客 § Discussion · Dense supervision 给的信息论解释, 7-10× steps / 50-100× compute 自蒸馏实验; 解释 Qwen3 Table 21 的 1/10 GPU·h 不是工程优化是 reward density 数量级差; 引 Lightman 2023 process reward modeling 作 RL 追上 distill 效率的对应路径。
- 新增第六层「RL 子网络脆弱(Mukherjee 2025)」——这一层不是 OPD 本身的数学性质,但回答"为什么 cross-stage 召回是必要而非可选"。引 Mukherjee 2025 (RL 只调小子网络) + 博客 personalization 实验 (IF-eval 85→45→83) 双重证据, 给出 GLM-5 流水线设计的完整数学/系统解释。
- 新增第七层「phase-alternating(Cobbe 2020)」——博客 § Personalization 末段 fine-tune→distill 交替 recipe; GLM-5 单次实例 vs MiMo co-evolution 多次迭代都落在这个框架下。
- "把四层拼起来"→"把七层拼起来"统一更新, 第二条增补 unhackable 性质。

反向链接(8 处):

- `index.md` 来源区第 26 条加新源页一行 + MOPD 概念页摘要更新为 7 层。
- `comparisons/on-policy-distillation.md` 相关页面区加新源页一行 + MOPD 概念页跳转更新为 7 层。
- 5 家源页(glm-5 / mimo-v2-flash / qwen3 / qwen3-vl / deepseek-v4)的 OPD 段落引用框各加一句"算法源头是 [Thinking Machines Lab On-Policy Distillation 博客]"; Qwen3 源页特别注明"Table 21 反过来被博客复现, 互为 inspiration"。

证据: 7 层中第一、四层是 KL 散度标准结果, 第二、三层引 Bishop PRML / Minka 2005 / Ross 2011 DAgger 经典文献(tier 2), 第五、六、七层引博客原文(tier 1 一手出处)。本节"7 层合力解释 OPD 为什么 work"与"多 teacher 混采没闭合的边界"是本页原创综合(tier 3), 但每条子论证均有 tier 1/2 支撑。`raw/` 新增 1 个 markdown 快照(git-ignored)。


## [2026-06-25] ingest | Agentic Reinforced Policy Optimization

新增 `raw/Dong 等 - 2025 - Agentic reinforced policy optimization.pdf`（arXiv:2507.19849v1，人大 + 快手）沉淀为 agentic RL 算法页：

- 新增 `wiki/sources/agentic-reinforced-policy-optimization.md`：覆盖 entropy spike 观察、entropy-based adaptive rollout、advantage attribution estimation、GPG theorem、Table 1/2/3 结果、Figure 7 tool-call efficiency 与 Figure 8 scaling 边界。
- 新增 `wiki/concepts/agentic-reinforced-policy-optimization.md`：把 ARPO 放进 agentic 后训练谱系，明确它不是 reward/harness/system，而是「高熵工具反馈步的 rollout 采样结构」。
- 更新 `wiki/concepts/post-training-for-agentic-models.md`、`agentic-engineering.md`、`agentic-evaluation-benchmarks.md`、`asynchronous-agent-rl.md`、`forge-agent-native-rl.md`：补 ARPO 与 GLM-5 异步 RL / Forge / Agent Swarm / benchmark 设置的层级关系。
- 更新 `wiki/sources/qwen3.md`、`wiki/models/qwen3.md`：说明 Qwen3-8B/14B 在 ARPO 论文中只是外部后训练算法的 backbone，非 Qwen3 官方报告新增内容。
- 更新 `wiki/index.md`：来源区与细讲模块新增 ARPO，后训练概念摘要同步。

图文化：用 PyMuPDF 从原 PDF 裁出并嵌入 4 张图：Figure 3（ARPO overview）、Figure 4（adaptive rollout + advantage attribution）、Figure 7（tool-call efficiency）、Figure 8（hyperparameter scaling），均经 `get_textbox(clip)` 和 vision_analyze 核对。`raw/` 未改。

## [2026-06-25] refactor | 全库切到 OKF v0.1 frontmatter

按 `TODO.md` 的 OKF v0.1 待办做结构性改造，正文事实不变、`raw/` 未改：

- 为 52 个非保留 Markdown 文件补 OKF frontmatter：`Source` 19、`Model` 11、`Concept` 18、`Comparison` 3、`TodoList` 1；`index.md` / `log.md` 保持 OKF 保留文件形态。
- `sources/*.md` frontmatter 补 `resource` 指向对应 `raw/` PDF/博客快照；`TODO.md` 保留在 bundle 内，类型定为 `TodoList`，并移除已执行的 OKF 待办段。
- `AGENTS.md` / `CLAUDE.md` 同步新增 OKF frontmatter 约定。
- 体检通过：frontmatter 类型映射无缺失，Markdown 相对链接 0 断链，`wiki/assets/` 18 个文件 0 孤儿，所有 concept 文件 frontmatter 后仍有 H1。

## [2026-06-25] ingest | DAPO、GSPO、SAPO policy optimization 论文

新增三篇 LLM RL policy optimization 方法论文到 wiki：

- `wiki/sources/dapo.md`：ByteDance Seed + 清华 AIR 的 DAPO，沉淀 Clip-Higher / Dynamic Sampling / token-level loss / Overlong Reward Shaping / DAPO-Math-17K；重排 Table 1 progressive ablation（Qwen2.5-32B AIME24 avg@32：naive GRPO 30 → DAPO 50）。
- `wiki/sources/group-sequence-policy-optimization.md`：Qwen GSPO，把 GRPO token-level importance ratio 改为 sequence likelihood ratio / sequence-level clipping，说明 MoE expert routing 波动与 Routing Replay 负担。
- `wiki/sources/soft-adaptive-policy-optimization.md`：Qwen SAPO，用 temperature-controlled soft gate 替代 hard clipping，说明 $\tau_{neg}>\tau_{pos}$ 与 Qwen3-VL-30B-A3B preliminary cold-start 训练结果。
- 新增 `wiki/comparisons/llm-rl-policy-optimization.md`：按抽象层级对比 DAPO（GRPO recipe）、GSPO（sequence-level ratio）、SAPO（soft trust region）、ARPO（agentic partial rollout），避免把四者都粗糙归成“比 GRPO 好”。
- 更新 `wiki/concepts/post-training-for-agentic-models.md`、`agentic-evaluation-benchmarks.md`、`wiki/sources/qwen3.md`、`wiki/models/qwen3.md`、`wiki/sources/qwen3-vl.md`、`wiki/models/qwen3-vl.md`、`wiki/index.md`：补 RL optimizer 谱系、Qwen/Qwen3-VL 的外部方法论文反链，并明确这些不是原模型报告正文事实。

图文化：新增 `wiki/assets/dapo/`、`wiki/assets/gspo/`、`wiki/assets/sapo/` 共 13 张图，覆盖 DAPO Figure 1/2/4/5/6、GSPO Figure 1/2/3、SAPO Figure 1/2-3/4/5/6；均用 PyMuPDF clip 渲染并经 `get_textbox(clip)` + vision_analyze 核对。`raw/` 未改。

## [2026-06-25] ingest | vLLM-Omni 技术报告

新增 `raw/vllm-omni-2602.02204.pdf`（arXiv:2602.02204v1）与官方博客快照 `raw/vllm-omni-blog-2025-11-30.md`，把 infra 论文作为「模型能力落地 infra」纳入 wiki：

- 新增 `wiki/sources/vllm-omni.md`：覆盖 stage graph 抽象、每 stage 独立 execution engine、AR / DiT stage support、streaming stage output、unified connector、Qwen-Omni / BAGEL / MiMo-Audio / DiT 结果。
- 新增 `wiki/concepts/any-to-any-multimodal-serving.md`：把 Qwen-Omni / Qwen3.5-Omni 这类 Thinker / Talker / Vocoder 多阶段模型的 serving 问题抽成概念页，并与 Qwen3-VL（multimodal input→text output）和百万 token serving（KV/state cache disaggregation）区分。
- 更新 `wiki/sources/qwen3.5-omni.md`、`wiki/models/qwen3.5.md`、`wiki/concepts/million-token-context-serving.md`、`wiki/concepts/multimodal-agentic-training.md`、`wiki/index.md` 加双向引用。
- 图文化：从原 PDF 裁出并嵌入 Figure 3（vLLM-Omni architecture）、Figure 4（Qwen2.5-Omni stage graph）、Figure 5（unified connector）、Figure 6（Qwen-Omni end-to-end results）、Figure 8（DiT-based results），均经 `get_textbox(clip)` 与 vision_analyze 核对。

`raw/` 新增 1 个 PDF + 1 个博客快照；既有 raw 未改。

## [2026-06-29] ingest | DSpark 技术报告

新增 `raw/DSpark_paper.pdf`（PKU + DeepSeek-AI，5 人共同一作，提出 DSpark = semi-AR drafter + confidence-scheduled verification）沉淀为 speculative decoding / 生产 serving 主题页：

- 新增 `wiki/sources/dspark.md`：覆盖 (a) semi-AR 结构（DFlash parallel backbone + Markov/RNN head 注入 intra-block 转移，保留 per-token exact softmax 以满足 lossless rejection sampling），(b) confidence head + Sequential Temperature Scaling（用累积乘积的 ECE 做位置 1→γ 顺序校准），(c) hardware-aware prefix scheduler（按 profile 出的 SPS(B) 表把"验多长"做成全局吞吐最大化，配合早停保 lossless），(d) 训练目标（w_k=exp(-(k-1)/γ) 位置权 + L_tv 直接最小化总变差 ≡ 最大化期望接受率），(e) 生产部署对照 MTP-1 的 +60–85% / +57–78% per-user 速度 + 严格 SLA 下 Pareto 外推（论文自己也对"6.6×/5.0×"这类极端比值做了诚实降级解读），(f) HAI-LLM 工程细节（hidden-state-only target 通信、anchor-bounded sequence packing、异步 scheduler 顺便放开全局贪心 early stop、变长 verification 只需改 index-attention/compress kernel）。
- 关键反向链接：DSpark 是 [DeepSeek-V4](models/deepseek-v4.md) 生产端 MTP-1 的实际替代品，V4 preview 上线两周后切换——`sources/deepseek-v4.md` 待追问、`models/deepseek-v4.md` 架构段与相关页面、`concepts/multi-token-prediction.md` 新增「当 MTP-1 不够：DSpark 接管」段 + V4 行重写、`concepts/million-token-context-serving.md` 关键判断加 speculative decoding 一档全部接上。
- 图文化：用 PyMuPDF 裁出并嵌入 3 张图，全部经 `page.get_textbox(clip)` 核对：Figure 1（架构 + 解码循环：A B C → D anchor → parallel backbone EFGH + sequential block + confidence c1-c4 → scheduler keep EFG/drop H → target 验证）、Figure 2（position-wise conditional acceptance：揭示"parallel drafter 反胜 autoregressive drafter"的反直觉现象 = 位置 1 容量优势 + DFlash suffix decay + DSpark 同时拿到两边）、Figure 7（V4-Flash/V4-Pro live traffic 下 throughput-TPS Pareto 前沿外推）。
- `wiki/index.md` 来源区第 33 条新增 DSpark；概念区 MTP 行摘要更新。
- 证据：所有机制结论均按 `raw/DSpark_paper.pdf` § 标号校验——semi-AR (§3.1 公式 4-6)、confidence + STS (§3.2.1 公式 7-8 + Figure 6)、scheduler (§3.2.2 Algorithm 1 + 附录 A 因果性反例)、训练 (§3.3 公式 9-12)、生产 (§5.1-5.4 + Figure 7-8)。"DSpark 替换 MTP-1"出自 §5.4 原文 "MTP-1 represents the former production setup, having been superseded by DSpark two weeks following the DeepSeek-V4-preview release."。`raw/` 未改。

## [2026-07-11] ingest | DoReMi 论文

新增 `raw/Xie 等 - 2023 - DoReMi Optimizing data mixtures speeds up language model pretraining.pdf`（arXiv:2305.10429v4, NeurIPS 2023, Google DeepMind + Stanford）：

- 新增 `wiki/sources/doremi.md`：覆盖三步流程（reference model → Group DRO proxy → large model）、minimax 目标公式、Algorithm 1 pseudocode、The Pile / GLaM 实验（2.6x 加速、22/22 domain ppl 改善、6.5pp 下游提升）、scale 消融（280M proxy 最优，1B proxy 退化）、domain weight 轨迹、局限。
- 新增 `wiki/concepts/data-mixture-optimization.md`：把 DoReMi 放进方法论谱系（启发式 → Group DRO → DoGE bi-level → RegMix 回归 → TANDEM twin network），含产业实践对比（Qwen3 instance-level mixture）。
- 更新 `wiki/sources/qwen3.md`：instance-level data mixture 行加 DoReMi 反向链接。
- `wiki/index.md` 来源区 +1 条、概念区 +1 条。

图文化：用 PyMuPDF 从原 PDF 裁出并嵌入 7 张图：Figure 1（三步流程）、Figure 2（加速效果）、Figure 3（下游准确率）、Figure 4（per-domain perplexity）、Figure 5（跨尺度）、Figure 6（消融）、Figure 8（domain weight 轨迹），均经 `get_textbox(clip)` + vision_analyze 核对。`raw/` 未改。

## [2026-07-11] ingest | TANDEM 论文

新增 `raw/TANDEM-bi-level-data-mixture-2606.04401.pdf`（arXiv:2606.04401, NeurIPS 2025, JD.com + Oxford + 人大）：

- 新增 `wiki/sources/tandem.md`：覆盖 bi-level formulation -> penalized single-level form -> twin network 更新规则、与 DoReMi/DoGE 的 hyper-gradient 对比（Table 1）、三场景实验（data-abundant / data-restricted / SFT）、Proposition 1（数据充足时 uniform 是最优解）、收敛定理 O(T^{-1/4})、synchronization 与 probing K 的消融。
- 更新 `wiki/concepts/data-mixture-optimization.md`：TANDEM 段从概述升级为详细描述，加 source 链接。
- 更新 `wiki/sources/doremi.md`：相关页面加 TANDEM 反向链接。
- `wiki/index.md` 来源区 +1 条。

图文化：原 PDF 嵌入字体损坏（MuPDF `FT_New_Memory_Face: broken file`），Figure 1/3/5 的文字标签渲染为乱码，无法嵌入可用图。页面以 prose + Markdown table 代替（Table 1 hyper-gradient 对比、Table 3 实验结果、Table 2 复杂度对比均 re-typeset 为 Markdown table）。`raw/` 未改。

## [2026-07-11] deepen | TANDEM 图文化补全

用户重新下载了字体完好的 PDF（`raw/2606.04401v1.pdf`），替换了原损坏文件。用 PyMuPDF 重新裁出并嵌入 5 张图：Figure 1（架构 + 计算流程）、Figure 3（三场景 mixture ratio 演化）、Figure 5（各方法学到的 mixture ratio）、Figure 7（Dist(u,w) 同步 vs 不同步）、Figure 8（variance vs K），均经 `get_textbox(clip)` + vision_analyze 核对。source 页 resource 指向更新为新文件名。`raw/` 未改（用户操作）。

## [2026-07-11] ingest | Gemma 4 技术报告

沉淀 Gemma 4（arXiv:2607.02770v1，Google DeepMind，2026-06-19）。原生多模态 dense + MoE 家族（E2B/E4B/12B/26B-A4B/31B），Apache 2.0。

新增页面：
- `wiki/sources/gemma-4.md`（来源页，含 Table 1/3/5/6/9 重新排版的 Markdown 表 + Figure 1 嵌入）
- `wiki/models/gemma-4.md`（模型实体页，含模态/变体/架构/KV 优化/MTP/QAT 关键事实表）
- `wiki/assets/gemma-4/fig1-mtp-drafter.png`（Figure 1 MTP drafter 架构图，300 DPI vector render）
- `wiki/assets/gemma-4/fig2-image-resizing.png`（Figure 2 图像缩放示意，300 DPI vector render）

更新页面：
- `concepts/efficient-long-context-attention.md`：模式稀疏行加入 Gemma 4（5:1 SWA/GA + key-as-value + p-RoPE + KV sharing，全局 KV -37.5%），新增机制段
- `concepts/multi-token-prediction.md`：用法表加 Gemma 4 行（cross-attention 复用主模型 KV，无需 MTP prefill），新增"Gemma 4 的经验"段
- `concepts/moe-frontier-model-scaling.md`：对比表加 26B-A4B 行
- `comparisons/2026-open-model-technical-reports.md`：范围/对比表/二版综合加 Gemma 4
- `index.md`：来源段和模型段各加一条

核心发现：Gemma 4 走效率优先路线，长上下文用 SWA/GA 混合 + KV 侧压缩（而非内容稀疏/线性注意力），128K 级足够有效（RULER 128k 31B=96.4）。12B encoder-free 证明可去掉独立 ViT/USM 编码器。MTP drafter 用 cross-attention 复用主模型 KV，消除 MTP prefill。31B 以 Elo 1451 居 Arena Text open dense 首位。`raw/` 未改。

## [2026-07-11] ingest | UniClawBench

新增 proactive agent 评测基准论文：

- `raw/2607.08768v1.pdf`（arXiv:2607.08768v1）

UniClawBench（HKU MMLab + Meituan）是 capability-driven 的 proactive agent benchmark：400 双语真实世界任务，按 5 维能力（Multimodal / Long Context / Skill Usage / Exploration / Cross-Platform）组织而非场景分类；三角色闭环评测（executor in Docker + hidden supervisor with rubric + user simulator with coarse signal），Information Firewall 隔离评分标准。

新增/更新文件：

- 新增 `wiki/sources/uniclawbench.md`：来源页，含 Figure 1（overview）和 Figure 2（三角色闭环评测流程）嵌入，Table 1/2 跨模型与跨框架结果转写为 Markdown 表。
- 新增 `wiki/assets/uniclawbench/fig1-overview.png`、`fig2-evaluation-pipeline.png`。
- 更新 `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表新增 UniClawBench 行，新增"UniClawBench 的差异化定位"段（capability-driven / 三角色闭环 / 真实环境三点 + framework > model 和半途失败两个发现）。
- 更新 `wiki/index.md`：来源栏新增条目，评测体系概念栏摘要更新。

核心发现：最强模型 Overall PR < 50%；framework 选择对能力影响常超模型选择（GPT-5.4 在 OpenClaw/EDICT/Nanobot 下 PR = 0.407/0.338/0.290）；Long Context 和 Multimodal 是主要瓶颈；多轮 user feedback 持续提升 pass rate。论文非模型报告，不建模型页。`raw/` 未改。

## [2026-07-11] ingest | HunyuanOCR-1.5 (arXiv:2607.04884v1)

腾讯 + 中科院信工所 + 南开的轻量端到端 OCR VLM 报告，1B 参数，围绕「更快 + 更好」升级 HunyuanOCR-1.0（不重设计架构）。两条主线：(1) DFlash block-diffusion 推测解码——90.7M / 5 层 draft model，joint FlexAttention block-diagonal mask 一次 forward 训练 K=16 个 draft block，Transformers 6.37× / vLLM 2.14× 加速，输出越长加速越明显（表格 > 公式 > 文本）；(2) Agentic Data Flow——agent 驱动数据构造，自动搜物料、开发渲染/QA pipeline、跑 hard-case 挖掘，补 331 种低资源语言 + 七种古文字 + 多图 QA。RL 用 IcePop（GRPO 变体，train-inference ratio mask）+ 三组件 reward（文档解析事实性 / QA 一致性判官 / 退化抑制）。OmniDocBench v1.6 总分 94.74（1B SOTA），CHAOS-Bench recall 14.15（远超其他 OCR 模型 3–6），Chronicles-OCR 古文字 0.54/0.79（1B SOTA）。`raw/` 未改。

新增/修改页面：
- `wiki/sources/hunyuan-ocr-1.5.md`（新建，嵌入 4 张图：架构 / DFlash mask / Agentic Data Flow / RL 框架）
- `wiki/models/hunyuan-ocr-1.5.md`（新建）
- `wiki/assets/hunyuan-ocr-1.5/`（fig1-architecture / fig2-dflash-training / fig3-agentic-data-flow / fig4-rl-framework）
- `wiki/concepts/multi-token-prediction.md`（+DFlash 行）
- `wiki/concepts/agentic-engineering.md`（+Agentic Data Flow 跨报告信号 + 相关页面）
- `wiki/concepts/post-training-for-agentic-models.md`（+HunyuanOCR-1.5 综合框架条目）
- `wiki/index.md`（+来源 +模型两行）
## [2026-07-11] ingest | InternVLA-A1.5 技术报告

新增 `raw/2607.04988v1.pdf`（arXiv:2607.04988v1，2026-07-06，上海 AI Lab / Physical Intelligence Team）--统一 VLA（Vision-Language-Action）机器人操作模型。

新建：

- `wiki/sources/internvla-a1.5.md`：来源页（图文交错）。覆盖三痛点（语义侵蚀 / 异质目标干扰 / 从零学像素生成）→ 三设计（VLM 持续 VQA 训练 / MoT 共享 full attention 解耦 / latent foresight 蒸馏 frozen WAN2.2-5B）。架构段嵌 Figure 2（MoT 框架：VLM Qwen-3.5 2B + 460M unified expert，共享 full attention 层、独立 GDN 层）；foresight 段嵌 Figure 4（foresight reasoning 机制：learnable tokens → frozen WAN DiT → gradient 只回 expert）。训练配方表（3 阶段 / 300K+600K+60K steps）、数据配方（1.2M episodes / 861M frames + 3M 多模态样本）、评测表（6 项仿真全最优 + 真实世界 4 任务领先 π0.5/Motus）、消融（Table 8：去 video loss / 去 foresight tokens 均降）。
- `wiki/models/internvla-a1.5.md`：模型页。关键事实表含模态 = 多模态（文本+图像+机器人状态→连续动作）、VLM backbone = Qwen-3.5 2B（3:1 GDN:full attention）、unified expert 460M、WAN2.2-5B frozen。技术身份段讲三个组合创新 + 与 wiki 已收录架构的关系（GDN 在 VLA 领域的采用证据）。

更新：

- `wiki/concepts/linear-attention-and-delta-rule.md`：跨报告信号新增 InternVLA-A1.5 行--GDN 混合注意力跨出语言/多模态对话、进入机器人实时控制领域的采用证据，论文明写 backbone "interleaves 3 Gated DeltaNet linear attention layers with 1 standard full attention layer"（§ 2 原文确证）。相关页面补 InternVLA-A1.5。
- `wiki/models/qwen3.5.md`：相关页面新增"作为 backbone 被采用：InternVLA-A1.5"反链。
- `wiki/index.md`：来源区 + 模型区各加 1 条。

图文化：2 张图（Figure 2 框架 + Figure 4 foresight 机制），PyMuPDF 300 DPI 渲染 + `get_textbox` 校验 + `vision_analyze` 核对。`raw/` 未改。
## [2026-07-11] ingest | KAT-Coder-V2 / V2.5 技术报告

新增快手 KwaiKAT 团队两篇 agentic coding 模型报告：

- `raw/2603.27703v1.pdf`（KAT-Coder-V2，arXiv:2603.27703，2026-03-29，22 页）
- `raw/2607.05471v1.pdf`（KAT-Coder-V2.5，arXiv:2607.05471，2026-07-06，24 页）

新建/修改文件：

- `wiki/sources/kat-coder-v2.md`（新建）：Specialize-then-Unify 五域分治（SWE/WebCoding/Terminal/WebSearch/General）+ KwaiEnv 模块化沙箱 + AutoBuilder + Agentic Scaling（task complexity/intent alignment/scaffold generalization）+ MCLA（K=8 forward prefill 取 log-prob 平均降 MoE RL 方差）+ turn-level GSPO adaptation + Tree Training（DFS 展平树状轨迹，梯度等价，6.2× 加速）+ OPD 专家融合。嵌入 Figure 4（AutoBuilder）和 Figure 5（Tree Training）。
- `wiki/sources/kat-coder-v2.5.md`（新建）：系统性基础设施重构。Environment Scaling Engine（verifiable task mining + AutoBuilder 100K+ 环境/12 语言，成功率 16.5%->57.2%）+ Data Scaling Flywheel（hint-boosted near-miss 恢复 + hint-free replay + 九维 process filtering + harness rewriting）+ KwaiClawEnv（Service/Task/Eval 三层 + 两级 scaling）+ harness randomization（format/context-structure/control-flow 三轴）+ Gateway Server（retokenization drift 消除）+ reliability-hardened sandbox（错误率 16%-><2%）+ asymmetric PPO with hindsight-augmented critic + harness-oriented reward（三层 10 项 rule-based + GRM model-based）+ 长上下文 MOPD 稳定化（cold start + drift-aware truncation）。嵌入 Figure 2（SE 数据管线）、Figure 3（KwaiClawEnv）、Figure 4（RL 架构）。
- `wiki/models/kat-coder.md`（新建）：V2/V2.5 变体表，模态=纯文本，基座未公开，代际演进段。
- `wiki/assets/kat-coder-v2/`（fig4-autobuilder.png, fig5-tree-training.png）
- `wiki/assets/kat-coder-v2.5/`（fig2-se-data-pipelines.png, fig3-kawaiclawenv.png, fig4-rl-infrastructure.png）
- `wiki/concepts/agentic-engineering.md`：+KAT-Coder 跨报告信号 + 相关页面
- `wiki/concepts/post-training-for-agentic-models.md`：+KAT-Coder 综合框架条目
- `wiki/concepts/multi-teacher-on-policy-distillation.md`：+KAT-Coder-V2.5 长上下文 MOPD 待追问
- `wiki/concepts/agentic-evaluation-benchmarks.md`：+KAT Code Bench / KAT Claw Bench / PinchBench / Claw-Eval 四行
- `wiki/comparisons/on-policy-distillation.md`：速览表 +KAT-Coder V2/V2.5 两行，轴一 A 类 +KAT-Coder V2.5 长上下文稳定化
- `wiki/index.md`：来源段 +2 条，模型段 +1 条

核心发现：V2 -> V2.5 的核心变化不是架构升级而是训练基础设施系统性重构--RL 算法从 GRPO 变体切到 asymmetric PPO + hindsight critic，发现 ~16% 训练失败源于沙箱而非算法（降到 <2%），V2.5 的 MOPD 是已收录报告中唯一把长上下文 OPD 不稳定性（student prefix 偏离 teacher 分布）作为独立工程问题处理的，用 cold start + drift-aware truncation 解决。两篇报告均未公开 backbone 架构和参数量。图文化：5 张图（V2: AutoBuilder + Tree Training；V2.5: SE 数据管线 + KwaiClawEnv + RL 架构），PyMuPDF 300 DPI + get_textbox 校验 + vision_analyze 核对。`raw/` 未改。

## [2026-07-11] ingest | daVinci-Agency (arXiv:2602.02619v2)

SII-GAIR 的长周期 agent 数据合成论文。核心：从 GitHub chain-of-PRs（有显式依赖拓扑的 PR 链，非时间排序）挖掘 task decomposition / long-term consistency / verifiable refinement 三种长周期监督信号。Pipeline 三步：query construction（LLM 合成隐去实现细节的 intent-based sub-query）→ cross-stage rollout（前序 patch 叠加到下一 stage 的 base commit，模拟增量开发）→ rejection sampling + scaling success。仅 239 样本 SFT GLM-4.6，Toolathlon +47%（0.157→0.231）、AVG 0.475 超 66k 样本 SWE-Smith（0.373）。消融验证 chain-of-PRs 优于 SinglePR / TemporalChain。两条 scaling law：训练轨迹 horizon 延长 + 推理步数增加都稳步提升。嵌入 4 张图（Fig 1 chain-of-PRs 概念+性能、Fig 2 三层 scope 对比、Fig 4 pipeline、Fig 7 scaling laws）。

- 新增：`wiki/sources/davinci-agency.md`、`wiki/assets/davinci-agency/`（4 PNG）
- 更新：`wiki/concepts/agentic-engineering.md`（跨报告信号 + 相关页面）、`wiki/concepts/post-training-for-agentic-models.md`（综合框架新增 SFT 数据结构层）、`wiki/index.md`
- `raw/` 未改

## [2026-07-11] ingest | Seed2.0 Model Card (arXiv:2607.00248v1)

字节跳动 Seed 团队的 Seed2.0 Series（Pro / Lite / Mini）Model Card，87 页。**这是 Model Card 而非技术报告**--不含架构设计、参数量、训练数据、训练方法，核心内容是部署洞察 + 评测框架 + benchmark 结果 + 真实世界 case studies。

新建文件：

- `wiki/sources/seed2.md`：来源页（图文交错）。部署洞察段（MaaS 使用分布 + agentic coding 查询分布 + 定价表）嵌入 Figure 1（MaaS 双饼图）和 Figure 2（查询分布柱状图）；评测要点段含语言/视觉/agentic benchmark 表（对标 GPT-5.2/Claude-4.5/Gemini-3-Pro），竞赛编程段嵌入 Figure 7（ICPC 五场 Pass@8 柱状图）；Case studies 段概括 Vibe Coding / GUI 操作 / 科研编程 / 复杂数学四类案例。待追问列出 5 条架构/开源/定价/VideoCut 工具未披露项。
- `wiki/models/seed2.md`：模型页。关键事实表标注模态 = 多模态（文本+图像+视频，已据原文核实），总参数/激活参数/架构/上下文均标"未披露"。技术身份段解释 Model Card 定位与技术报告的区别，列出部署导向/四维评测框架/诚实差距承认/丰富 case studies 四项价值。
- `wiki/assets/seed2/`：fig1-maas-usage-distribution.png, fig2-query-distribution.png, fig7-icpc-results.png

更新：

- `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表新增 NL2Repo-Bench / Ainstain Bench / GDPVal-Verified 三行；"对已沉淀报告的影响"段加 Seed2.0 四维评测框架行。
- `wiki/concepts/agentic-engineering.md`：跨报告信号加 Seed2.0 生产部署视角行（前端 >50%、bug fixing 主导、四维框架、case studies）；相关页面加 Seed2.0 反链。
- `wiki/concepts/moe-frontier-model-scaling.md`：对比表加 Seed2.0 行（参数/激活均未披露）。
- `wiki/comparisons/2026-open-model-technical-reports.md`：范围段加 Seed2.0 + Model Card 注释；对比表加 Seed2.0 行（列标"未披露"）；二版综合加 Seed2.0 工程哲学。
- `wiki/index.md`：来源段 +1 条，模型段 +1 条。

核心定位：Seed2.0 Pro 在数学竞赛（AIME 2025 98.3、IMO/CMO Gold）和搜索 agent（BrowseComp 77.3、BrowseComp-zh 82.4）方面领先，ICPC Pass@8 73.02% 超过 GPT-5.2/Gemini-3-Pro；诚实承认 coding 不如 Claude（SWE-Evo 8.5 vs 27.1）、长尾知识不如 Gemini（SimpleQA 36.0 vs 72.1）。定价比 frontier 模型低约一个数量级。前代 Seed 家族包括 Seed1.6/1.8、Seed1.5-VL、Seed-OSS、Seed-Coder、Seed Diffusion、Seed-Prover、Seedream/Seedance。与已收录的 [DAPO](dapo.md) 同为 ByteDance Seed 团队。图文化：3 张图（Figure 1 MaaS 分布 + Figure 2 查询分布 + Figure 7 ICPC 结果），PyMuPDF 300 DPI 渲染 + `get_textbox` 校验 + `vision_analyze` 核对。`raw/` 未改。

## [2026-07-11] ingest | VibeThinker-3B 技术报告

`raw/2606.16140v1.pdf`（arXiv:2606.16140v1, Sina Weibo, 2026-06-15）。

新增：

- `wiki/sources/vibethinker-3b.md`：来源页（图文交错）。嵌入 Figure 1（六 benchmark 柱状图）、Figure 2（参数效率图）、Figure 3（训练流水线）。Table 1/2 核心评测数据重排为 Markdown。待追问 5 条（MGPO 的 $D_{ME}$ 具体形式、参数 merge 方法、Long2Short $\lambda$ 消融、CLR claim 提取敏感性、context window 策略与 SFT 质量耦合）。
- `wiki/models/vibethinker-3b.md`：模型页。关键事实表含模态=纯文本（已据原文核实）。技术身份段定位为后训练系统工程而非架构创新。
- `wiki/assets/vibethinker-3b/`：fig1-benchmark-bars.png, fig2-param-efficiency.png, fig3-pipeline.png

更新：

- `wiki/comparisons/llm-rl-policy-optimization.md`：MGPO 加入速览表（prompt-level 梯度权重层）+ 新增「MGPO：不争 ratio 也不争采样位置，而争 prompt 的梯度贡献」段（与 DAPO Dynamic Sampling 的 hard filter vs soft weighting 对比）+ 相关页面加 VibeThinker-3B 双向引用。
- `wiki/concepts/post-training-for-agentic-models.md`：综合框架加 VibeThinker-3B 条目（Spectrum-to-Signal + Long2Short + CLR + context window 失效发现 + Parametric Compression-Coverage Hypothesis）。
- `wiki/index.md`：来源段 +1 条，模型段 +1 条。

核心定位：VibeThinker-3B 是 3B dense reasoning 模型，基于 Qwen2.5-Coder-3B，用 Spectrum-to-Signal 后训练范式（MGPO + curriculum SFT + Long2Short RL + offline self-distillation + Instruct RL + CLR）在 verifiable reasoning 上追平旗舰（AIME26 94.3 vs DeepSeek V3.2 94.2 / GLM-5 95.8），但 GPQA-D 差 14 点。MGPO 与 DAPO 的 Dynamic Sampling 动机一致但实现不同（soft weighting vs hard filter）。Long2Short RL 的零和 length-aware reward shift 是训练时效率优化，与 CLR 的 test-time scaling 互补。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + `get_textbox` 校验 + `vision_analyze` 核对 Fig 3。

## [2026-07-11] ingest | KVpop - Key-Value Cache Compression with Predictive Online Pruning

- `wiki/sources/kvpop.md`：新建。NXAI + JKU Linz（Hochreiter 团队）的 learned eviction 方法。核心机制：固定 per-head KV budget $B=s+w+k$（sink+window+top-k），用 future-attention target 在 eviction boundary 监督 keep-or-drop（transposed-attention pass 不 materialize $S \times S$ 矩阵，复用 sparse LSE 近似 dense LSE），可选 mLSTM 延迟打分利用近未来上下文。Qwen3-8B 在 88% 压缩下保留 100% teacher 性能，Qwen3-4B 保留 97%。比 DMS 快（固定 per-head budget vs dynamic gate 的参差 cache）。Table 1/2 转 Markdown。6 张图（Fig1 overview / Fig2 transposed-attn / Fig3 stateful scorer / Fig4a latency / Fig4b VRAM / Fig6 eviction patterns）PyMuPDF 300 DPI 抽取，`get_textbox` 校验，`vision_analyze` 交叉确认 Fig1/Fig3。
- `wiki/concepts/efficient-long-context-attention.md`：路线表加「学习式驱逐」行（KVpop），标题改「五条路线 + 学习式驱逐」，加一段说明 eviction 与 sparse retrieval 正交（前者 bound memory，后者减 compute）。
- `wiki/comparisons/sparse-attention-mechanisms.md`：主表加 KVpop 行；相关页面加 KVpop 链接；新增「eviction vs. sparse retrieval」段。
- `wiki/index.md`：来源段 +1。

非模型方法论文（learned eviction / KV cache compression），不建 models/ 页。`raw/` 未改。待追问：abstract/contributions 数字不一致（v1→v2 修订疑似未同步），未验证 MLA/GDN 架构迁移性。

## [2026-07-11] ingest | LoopCoder-v2 技术报告

`raw/2606.18023v1.pdf`（arXiv:2606.18023v1, Beihang + IQuest Research + Langboat, 2026-06-16）。

新增：

- `wiki/sources/loopcoder-v2.md`：来源页（图文交错）。嵌入 Figure 1（PLT overview：标准 loop vs PLT + gain–cost 权衡 + per-loop 诊断证据）和 Figure 3（gain–cost scissors：Δp(r) 崩溃 vs Ω(r) 恒定）。Table 2 主评测结果 + Table 3 per-loop 行为特征 + Table 4 instruct vs thinking 对比转 Markdown。待追问 4 条（CLP offset 自适应、与 MTP 关系、数据组成影响、跨参数量 scaling）。
- `wiki/models/loopcoder-v2.md`：模型页。关键事实表含模态=纯文本（已据原文核实）。变体表列 R=1/2/3/4 的 SWE-bench Verified + Avg。
- `wiki/concepts/looped-transformers.md`：新概念页。覆盖 looped Transformer 谱系（UT -> MELT/PLT/LT2 效率路线）、PLT 的 CLP offset 结构性代价、跨报告信号（LoopCoder-v2 / Huginn-3.5B / scaling law r^0.46 / 稳定性）、与高效长上下文注意力 + MTP 的正交关系、latent loop + explicit CoT 超加性互补。
- `wiki/assets/loopcoder-v2/`：fig1-overview.png, fig3-gain-cost-scissors.png

更新：

- `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表加 Mind2Web + BFCL (v3) 两行（LoopCoder-v2 用到但 wiki 未列）；解读注意事项末段加「推理时计算量」变量（R=2 SWE-bench 64.4% -> R=3 27.6% -> R=4 22.4%，强非单调）。
- `wiki/index.md`：来源段 +1，模型段 +1，概念段 +1。

核心定位：LoopCoder-v2 不是常规 dense LLM，而是 weight-tied looped Transformer（14 层共享 block，R=1/2/3/4 变体）。PLT 通过 CLP（cross-loop position offset）+ shared-KV G-SWA 使延迟和 KV-cache 不随 R 增长。gain–cost 剪刀是核心机制洞察：第二个 loop 的 refinement gain 最大（effective rank 达峰、attention routing 变化最大、output shift 最大），之后收益急缩且振荡（cos θ < 0）；CLP offset cost Ω(r) 恒定。固定成本 + 递减收益 = R=2 饱和。7B R=2 在 SWE-bench Verified 64.4% 超 Kimi-Dev-72B。latent loop + explicit CoT 呈超加性互补（LCB +26.9）。`raw/` 未改。图文化：2 张图，PyMuPDF 300 DPI + `get_textbox` 校验。

## [2026-07-11] ingest | MiniCPM-o 4.5 技术报告

`raw/2604.27393v1.pdf`（arXiv:2604.27393v1, OpenBMB / 清华 NLP, 2026-04-30）。

新增：

- `wiki/sources/minicpm-o-4-5.md`：来源页（图文交错）。嵌入 Figure 3（回合制 vs 全双工对比）、Figure 4（端到端全模态架构）和 Figure 5（TAIL 三种语音生成策略对比）。Table 1 Omni-Flow 设计权衡消融 + Table 9 length reward 对比转 Markdown。待追问 5 条。
- `wiki/models/minicpm-o-4-5.md`：模型页。关键事实表含模态=多模态（文本+图像+视频+音频输入；文本+音频输出，已据原文核实）。
- `wiki/assets/minicpm-o-4-5/`：fig3-turn-vs-duplex.png, fig4-architecture.png, fig5-tail-strategies.png

更新：

- `wiki/concepts/any-to-any-multimodal-serving.md`：加「MiniCPM-o 4.5：端到端架构是 stage graph 的模型侧融合」跨报告信号——与 vLLM-Omni 的 disaggregated serving 互补（端侧端到端 vs 云端 stage graph）。相关页面加链接。
- `wiki/concepts/multimodal-agentic-training.md`：加「MiniCPM-o 4.5：从回合制到全双工交互」跨报告信号——K2.5 的 early vision fusion vs MiniCPM-o 4.5 的四阶段渐进流水线，两种多模态融合策略对照。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：MiniCPM-o 4.5 是首个全双工全模态 LLM（9B），核心创新在交互范式而非单模态能力。Omni-Flow 框架用时分复用式时间窗口（1.0s chunk，LS 控制 + explicit boundary 最优）把感知与生成在 token-level 持续耦合。关键架构决策：LLM 只生成文本 token（3-4 step/s），语音 token 委托给 ~0.3B speech decoder。TAIL 策略考虑累积播放进度做自适应文本-语音交错。四阶段渐进训练（冻结→联合→SFT→RL）。9B 接近 Gemini 2.5 Flash，全模态超 Qwen3-Omni-30B-A3B，INT4 端侧 < 12GB。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + `get_textbox` 校验。

## [2026-07-11] ingest | GLM-5V-Turbo 技术报告

`raw/2604.26752v3.pdf`（arXiv:2604.26752v3, Z.ai & Tsinghua University, 2026-04）。

新增：

- `wiki/sources/glm-5v-turbo.md`：来源页（图文交错）。嵌入 Figure 1（CogViT 与 SigLIP2-SO / DFN-H / MetaCLIP2-H 零样本对比，403M 超 427M–632M 竞品）、Figure 4（多模态 coding / tool-use / GUI agent benchmark 对比 GLM-5V-Turbo / Kimi K2.5 / Claude Opus 4.6）、Figure 5（文本 coding / Claw agent benchmark 对比，含纯文本基座 GLM-5-Turbo）。Table 1 工具集 + Table 2 官方 skill 转散文。评测数据两表转 Markdown。待追问 6 条（参数量/backbone 架构未披露、CogViT-projector 连接方式、MMTP 与 GLM-5 参数共享 MTP 关系、relative visual policy optimization 算法形式、RL 任务清单、ImageMining 开源状态）。
- `wiki/models/glm-5v-turbo.md`：模型页。关键事实表含模态=多模态（已据报告原文 Abstract + § 1 核实）。总参数/激活参数/训练量标注"报告未披露"。
- `wiki/assets/glm-5v-turbo/`：fig1-cogvit-performance.png, fig4-multimodal-eval.png, fig5-coding-claw-eval.png

更新：

- `wiki/concepts/multimodal-agentic-training.md`：新增「跨报告信号」段，加 GLM-5V-Turbo 行（CogViT + 30+ 类别多模态 RL + RL 跨域干扰弱于 SFT + 三个 design lens + RL 基础设施与 GLM-5 异步 agent RL 的传承关系）。
- `wiki/concepts/multi-token-prediction.md`：MTP 表加 GLM-5V-Turbo 行（MMTP：共享 learnable `<|image|>` token 方案，避免跨 pipeline-parallel 传播视觉 embedding，0.5B 消融更稳）。
- `wiki/concepts/agentic-engineering.md`：跨报告信号加 GLM-5V-Turbo 行（多模态 agentic engineering + 三个 design lens + Vision2Web 规格扩展 + workflow-based verification）。
- `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表加 7 行（ImageMining / BrowseComp-VL / Design2Code / Vision2Web / MMSearch-Plus / AndroidWorld / WebVoyager）。
- `wiki/comparisons/2026-open-model-technical-reports.md`：范围段 + 对比表 + 二版综合各加 GLM-5V-Turbo。
- `wiki/models/glm-5.md`：相关页面加 GLM-5V-Turbo 反向链接。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：GLM-5V-Turbo 是 GLM-5 家族的多模态扩展（基座 GLM-5-Turbo），定位 native multimodal agent。报告刻意不披露参数量/训练量，主轴是五条线（模型设计 / 多模态训练 / RL / 工具链 / agent 框架集成）+ 三个 design lens（感知是天花板 / 分层优化 / 清晰规格+可靠验证+受控评测）。CogViT 两阶段预训练（distillation MIM + contrastive IT）以 403M 超 427M–632M 竞品。MMTP 用共享 `<|image|>` token 解决多模态 MTP 的图像 token 传递问题。30+ 类别多模态联合 RL 观察到 RL 跨域干扰弱于 SFT。加视觉未侵蚀文本 coding（CC-Backend/CC-RepoExploration 反超 GLM-5-Turbo）。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + `get_textbox` 校验 + `vision_analyze` 交叉确认。

## [2026-07-11] ingest | Keye-VL-2.0 技术报告

- **来源**：`raw/2606.10651v1.pdf`（arXiv:2606.10651v1，快手 Keye Team，2026-06-10，31 页）
- **新建页面**：
  - `wiki/sources/keye-vl-2.md`（图文交错 source 页，嵌入 Figure 1/2/4/5）
  - `wiki/models/keye-vl-2.md`（模型页，已据 HF config 核实：31.1B 总参 / 3B 激活 / 48 层 / 128 experts / 8 per-token / GQA 4 KV heads / 256K / SigLIP-384-14 ViT）
- **提取图表**：`wiki/assets/keye-vl-2/`
- **更新已有页面**：
  - `wiki/concepts/deepseek-sparse-attention.md`：新增「Keye-VL-2.0：DSA 从 MLA 走向 GQA」段
  - `wiki/concepts/efficient-long-context-attention.md`：DSA 路线行加 Keye-VL-2.0
  - `wiki/concepts/multi-teacher-on-policy-distillation.md`：新增「Keye-VL-2.0 的 Cross-Modal MOPD」段
  - `wiki/concepts/post-training-for-agentic-models.md`：综合框架新增 Keye-VL-2.0
  - `wiki/comparisons/on-policy-distillation.md`：速览表 Keye-VL-2.0 行（第 6 家 OPD 报告）
- **核心发现**：DSA 不依赖 MLA（indexer MQA + aggregation GQA）；MOPD 首次多模态大规模应用（13 teacher + top-k overlap estimator）；DSA RL 稳定性方案谱系扩张。


## [2026-07-11] ingest | JoyAI-VL-Interaction 技术报告

`raw/2606.14777v1.pdf`（arXiv:2606.14777v1, JD.com, 2026-06, 21 页）。

新增：

- `wiki/sources/joyai-vl-interaction.md`：来源页（图文交错）。嵌入 Figure 1（交互范式：实时告警/异步委托/持续解说）、Figure 2（AdaCodec 预测式视频编码：ViT 256 token vs P-token 16 token）、Figure 3（系统总览：双并发循环 + 三层记忆）。Table 1/2 评测结果转 Markdown，数据六族表转 Markdown。待追问 6 条（AdaCodec reset 阈值、RL 窗口大小、记忆参数、延迟 breakdown、TML 定量对比、delegation protocol 规范）。
- `wiki/models/joyai-vl-interaction.md`：模型页。关键事实表含模态=多模态（文本+图像+视频；语音经外置 ASR/TTS 转导，已据原文核实）。技术身份四点（interaction model 范式 / vision-first / delegate 闭合循环 / AdaCodec 无界流式）。与 Qwen3.5-Omni / Kimi K2.5 / Qwen3-VL 对照表。
- `wiki/assets/joyai-vl-interaction/`：fig1-interaction-paradigm.png, fig2-adacodec-encoding.png, fig3-system-overview.png（PyMuPDF 300 DPI + get_textbox 校验）

更新：

- `wiki/concepts/multimodal-agentic-training.md`：加「JoyAI-VL-Interaction：从看懂到每秒决定是否行动」跨报告信号（AdaCodec / 角色加权 SFT / answer-centered window sampling / 涌现能力，与 K2.5 对照）。
- `wiki/concepts/any-to-any-multimodal-serving.md`：加「JoyAI-VL-Interaction：不是 stage graph，而是决策在模型 + 可插拔外设」跨报告信号（双并发循环 vs stage graph / 语音在模型外 / 记忆围绕 prefix reuse）。
- `wiki/concepts/post-training-for-agentic-models.md`：综合框架加 JoyAI-VL-Interaction 一条（角色加权 SFT + GRPO + answer-centered window sampling，与 ARPO step-level rollout 的平行演进）。
- `wiki/concepts/agentic-evaluation-benchmarks.md`：末段加 JoyAI-VL-Interaction 的评测思路（不跑 offline benchmark，直接与 Doubao/Gemini 真实产品做 head-to-head 人工盲评）。
- `wiki/index.md`：来源段 +1，模型段 +1（本次补录：源页/模型页已先于 index/log 提交，本条目补齐索引）。

核心定位：JoyAI-VL-Interaction 是 JD.com 的 8B 视觉驱动交互模型，提出 interaction model 范式——模型持续观看视频流，每秒内部决定说话（`</response>`）/ 静默（`</silence>`）/ 委托后台（`</delegation>`），而非等用户提问才响应。基座是 Qwen3-8B + Qwen3-VL ViT，视频编码用 AdaCodec（预测式 I-frame/P-frame 结构，~16× 压缩）。4M+ 时间对齐流式数据六族，角色加权 SFT 对抗 silence 主导（w_repeated_silence=0.4, w_response=1.5），GRPO RL 用 answer-centered window sampling 压缩 rollout horizon。完整系统含双并发循环（实时 + 异步委托）、三层记忆（约 2 小时）、vLLM-native serving。vs Doubao 胜率 77.6% / vs Gemini 87.9%（6 场景 58 case 人工盲评），监控告警 100% 获胜。论文声称首个开源视觉驱动交互模型 + 完整可部署系统。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + get_textbox 校验。

## [2026-07-11] ingest | Xiaomi-GUI-0 技术报告

`raw/2606.31410v2.pdf`（arXiv:2606.31410v2, 小米 SeerRay Team, 2026-07-01, 40 页）。

新增：

- `wiki/sources/xiaomi-gui-0.md`：来源页（图文交错）。嵌入 Figure 1（混合基础设施三层架构：Resource/Scheduling/Execution & Collection，Device-Pull 调度）、Figure 5（teacher 打分与接管：student rollout + teacher 逐步打分 + 有界接管产生 deviation–diagnosis–recovery 段）、Figure 6（Agentic RL 在线训练框架：推理/环境执行/数据传输解耦，异步状态机驱动每条轨迹）。Table 1 cascade reward 转 Markdown，Table 2 RealMobile 能力域分布转 Markdown，Table 4 主要结果转 Markdown，Table 3 sub-goal 分解转散文，Table 7 action space + Table 8 异常语义转散文。待追问 6 条（backbone 视觉前端是否修改 / teacher 模型身份 / 5000 异常态样本分布 / Agentic RL 任务规模来源 / cascade reward 三值表达力 / RealMobile 评测方差）。
- `wiki/models/xiaomi-gui-0.md`：模型页。关键事实表含模态=多模态（文本+图像输入；文本输出）（已据报告原文核实：native end-to-end multimodal GUI agent）。总参数/激活参数标注"继承 Qwen3-VL-30B-A3B 基座，未提及新增模块"。
- `wiki/assets/xiaomi-gui-0/`：fig1-hybrid-infrastructure.png, fig5-teacher-takeover.png, fig6-agentic-rl-framework.png（PyMuPDF 300 DPI + get_textbox 校验）

更新：

- `wiki/concepts/post-training-for-agentic-models.md`：综合框架加 Xiaomi-GUI-0 一条（三阶段 dense→sparse 课程 + cascade reward top-down early-exit + turn-level batching + error-driven flywheel，复用 GSPO + DAPO dynamic sampling）。
- `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表加 RealMobile + MobileBench-OL 两行，AndroidWorld 行补 Xiaomi-GUI-0 78.9%；末段加 Xiaomi-GUI-0 真机评测三点结构性差异（全真机真应用 / 细粒度 sub-goal + veto + conditional branching / 双验证 XPath + logical semantic rules）+ Safety & Reflection 共同瓶颈。
- `wiki/concepts/agentic-engineering.md`：跨报告信号加 Xiaomi-GUI-0 行（训练/评测分布对齐而非模型规模 + error-driven flywheel 定向修复 + 真机闭环，与 KAT-Coder / daVinci-Agency 互补）；相关页面加反向链接。
- `wiki/models/qwen3-vl.md`：相关页面加 Xiaomi-GUI-0 反向链接（下游 GUI agent）。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：Xiaomi-GUI-0 是小米 SeerRay Team 的 native end-to-end multimodal GUI agent，核心论点是 benchmark 高分不等于真实可用性——真实设备上的账号状态 / 权限弹窗 / 支付验证 / 风控拦截持续改变执行状态分布。以 Qwen3-VL-30B-A3B-Instruct 为基座，构建真机为主的混合基础设施（Device-Pull 调度）+ error-driven data flywheel（首个关键错误标注 + teacher 打分接管）+ 三阶段训练（SFT → Step RL → Agentic RL，GSPO + cascade reward + turn-level batching）。RealMobile 72.0% / AndroidWorld 78.9%，开源最强对手 MAI-UI-8B 在 RealMobile 仅 33%。Safety & Reflection 是所有模型最弱域（Gemini 3.1 Pro 也仅 62.5%）。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + get_textbox 校验。

## [2026-07-12] ingest | Agent-World 技术报告

`raw/2604.18292v1.pdf`（arXiv:2604.18292v1, 人大高瓴人工智能学院 + ByteDance Seed, 2026-04-20, 48 页）。

新增：

- `wiki/sources/agent-world.md`：来源页（图文交错）。嵌入 Figure 1（两阶段闭环总览 + 环境 scaling 曲线）、Figure 2（Agentic Environment-Task Discovery 流程：MCP/工具文档/PRD 主题 → database mining → 工具生成验证 → graph/programmatic 任务合成）、Figure 5（Continuous Self-Evolving Agent Training 总框架：多环境 RL + 自演化 arena 闭环）、Figure 8（环境数量 scaling 关系，18.4%→38.5%）。Table 1 主结果（18 模型 × 3 benchmark Avg 列）转 Markdown，Table 2 自演化效果（两模型 × 两轮）转 Markdown。发现一处论文内部矛盾：§4.3.4 prose 称 Agent-World-14B τ2-Bench 45.3%→50.5%，但 Table 2 实为 60.2%→65.4%（BFCL/MCP 两列一致，仅 τ2 列不一致，增量 +5.2 一致），以表值为准并标注。待追问 7 条（τ2 矛盾、MCP-Mark 绝对分偏低、GPT-OSS-120B 双重角色天花板、5K RL 样本规模、自演化轮数上限、DB complexification N 与 K=5 依据、MCP-Mark/MCP-Atlas/MCP-Universe 关系）。
- `wiki/models/agent-world.md`：模型页。关键事实表含模态=纯文本（已据报告原文核实，全文围绕 MCP 工具/数据库/代码交互，23 benchmark 均为文本/工具/代码）。总参数/激活参数标注"继承 Qwen3-8B/14B dense 基座，未提及新增模块"。技术身份四点（训练方法论产出 / 环境即基础设施 / agent-environment co-evolution / 可执行 reward 双路）。
- `wiki/assets/agent-world/`：fig1-overview-scaling.png, fig2-discovery-pipeline.png, fig5-self-evolving-framework.png, fig8-env-scaling.png（PyMuPDF 300 DPI + get_textbox 校验；fig2/fig5 含图内标签无 caption 污染，fig1/fig8 纯矢量/图表用像素占比 + 矢量绘制数确认非空白）

更新：

- `wiki/concepts/agentic-engineering.md`：跨报告信号加 Agent-World 行（瓶颈定位在可扩展真实环境合成 + 连续自演化训练，1978 环境/19822 工具生态，与 KAT-Coder/daVinci-Agency/Xiaomi-GUI-0 互补：解决"训练环境本身如何可扩展合成并持续演化"）；相关页面加反向链接。
- `wiki/concepts/post-training-for-agentic-models.md`：综合框架加 Agent-World 一条（多环境 GRPO + 可执行 reward 双路 + 自演化 arena 诊断驱动课程，与 Forge self-evolution/daVinci-Agency 互补）。
- `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表加 8 行（BFCL V4 / MCP-Mark / MCP-Universe / SkillsBench / ARC-AGI-2 / WebWalkerQA / GAIA / HLE）；末段加 Agent-World 的动态 arena 评测方法论（评测做成动态诊断 arena 而非静态 benchmark，与 JoyAI head-to-head / Xiaomi 真机评测形成对照，环境数量 scaling 首次量化"训练环境多样性"为独立性能变量）。
- `wiki/concepts/forge-agent-native-rl.md`：与其他路线关系段加 Agent-World 一句（self-evolution 在数据/课程层而非训练系统层的互补路线），保持双向链接。
- `wiki/models/qwen3.md`：相关页面加 Agent-World 下游 agent 训练反向链接。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：Agent-World 是人大 + ByteDance Seed 的自演化 agent 训练场，核心论点是通用 agent 的瓶颈在可扩展真实环境合成与连续自演化训练机制，而非模型参数。两阶段闭环：(1) Agentic Environment-Task Discovery——从真实 MCP servers（~2.8K）/ 工具文档（~0.5K）/ 工业 PRD（~0.2K）挖主题，deep-research agent 从 web 建主题对齐数据库（含数据库复杂化迭代）+ coding agent 生成并交叉验证可执行工具，合成 1978 环境 / 19822 工具生态；任务合成走 graph-based（DAG 工具图 + 随机游走）与 programmatic（解代码 + verifier 脚本）两条路，均靠 sandbox 执行推导 ground truth 并保留可验证性。(2) Continuous Self-Evolving Agent Training——多环境 GRPO RL（可执行 reward：rubric LLM judge / V_code 脚本）+ 自演化 arena（分层采样 K=5 环境 → 动态合成新任务 → agentic diagnosis 定位弱环境 → 定向任务扩展 → continue RL），形成 agent-environment co-evolution。Qwen3-8B/14B 基座 + 冷启动 SFT（40K 轨迹）+ GRPO（clip 0.2/0.28）+ 2 轮自演化。23 benchmark 上 Agent-World-8B/14B 一致超过环境扩展基线，BFCL V4 55.8 与 DeepSeek-V3.2-685B 54.1 竞争力强；环境数量 scaling（0→2000）四域均分 18.4%→38.5%。`raw/` 未改。图文化：4 张图，PyMuPDF 300 DPI + get_textbox 校验。

## [2026-07-12] ingest | Qwen-AgentWorld: Language World Models for General Agents

`raw/2606.24597v1.pdf`（arXiv:2606.24597v1, Qwen Team, 2026-06-24, 47 页, Technical Blog）。

新增：

- `wiki/sources/qwen-agent-world.md`：来源页（图文交错）。嵌入 Figure 1（总览：7 域 native LWM + Decouple/Unify 两范式）、Figure 5（三阶段训练管线 CPT→SFT→RL）、Figure 8（跨域泛化：仅 Terminal RL 训练，MCP/SWE/Search 同步提升）、Figure 9（Controllable Sim RL vs Real RL：Sim RL 50.3% vs Real RL 45.6% + web_extractor 调用分化）、Figure 12（LWM 推理模式：多步因果推理 / 信息泄漏预防 / 认知边界觉察）。Table 1（7 域 action/observation/capability）、Table 2（SFT/RL 数据统计）、Table 3（7 类 turn loss masking）、Table 5（16 模型主结果）、Table 6/8/9（三应用结果）转 Markdown。待追问 7 条（GUI 域差距根因 / Factuality 持续最低 / 10M+ trajectories 构成拆解 / fictional-world 四策略独立贡献 / LWM warm-up 与 auxiliary loss 组合 / 与 Agent-World 互补边界 / state is bottleneck 定量证据）。
- `wiki/models/qwen-agent-world.md`：模型页。关键事实表含模态=纯文本（已据原文核实：GUI 域用 accessibility tree / UI view hierarchy 文本表示而非像素帧，§1 明确；多模态扩展列 future work）。变体表 35B-A3B / 397B-A17B 均基于 Qwen3.5 checkpoint。
- `wiki/assets/qwen-agent-world/`：fig1-overview.png, fig5-training-pipeline.png, fig8-cross-domain-generalization.png, fig9-sim-rl-vs-real-rl.png, fig12-reasoning-patterns.png（PyMuPDF 300 DPI + get_textbox 校验；fig5 纯矢量无文字层，其余含图内标签无 caption/footnote 污染）

更新：

- `wiki/concepts/agentic-engineering.md`：跨报告信号加 Qwen-AgentWorld 行（瓶颈定位在 world modeling 缺失拼图，Decouple Sim RL 可控模拟超真实环境 + Unify LWM warm-up 跨任务迁移，与 Agent-World code-driven 路线互补——"trades determinism for generality"）；相关页面加反向链接。
- `wiki/concepts/post-training-for-agentic-models.md`：综合框架加 Qwen-AgentWorld 一条（native LWM 三阶段 + turn-level 信息论 loss masking + hybrid rubric-and-rule reward 9:1 + LWM warm-up 跨任务迁移 + prompt-output 极端不对称，与 Agent-World code-driven 路线互补）。
- `wiki/concepts/agentic-evaluation-benchmarks.md`：benchmark 表加 AgentWorldBench 行；末段加 Qwen-AgentWorld 第四种评测思路（评测 world model 而非 agent policy）——reference-grounded judging / differentiated matching criteria / double-blind Turing test 校准 judge + GUI 域差距暗示模态表示是变量。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：Qwen-AgentWorld 是 Qwen Team 的 native language world model，核心论点是 world modeling 是通用 agent 缺失的拼图——当前 LLM agent 研究几乎只关注 policy 侧而忽略 world model。首个覆盖 7 域（MCP/Search/Terminal/SWE/Android/Web/OS）的单一 LWM，三阶段 "CPT injects, SFT activates, RL sharpens"：CPT 注入 state-transition dynamics + world knowledge（turn-level 信息论 loss masking 按 7 类 keep ratio 5%–100%）、SFT 激活 next-state-prediction thinking（rejection sampling 69.2% retention）、RL 用 GSPO + hybrid rubric-and-rule reward（9:1）锐化保真度（三种稳定性解法：单 turn 展开防 Echo Trap / rubric+rule 而非 reference/turing-test / tag extraction 防 self-praise）。AgentWorldBench 2,170 样本 / 5 维 rubric / reference-grounded judging，Qwen-AgentWorld-397B-A17B 总均分 58.71 超 GPT-5.4 58.25。两应用范式：(1) Decouple——LWM 作模拟器做 Sim RL，可控模拟（注入扰动 / 构造虚构世界）超真实环境训练（WideSearch 50.3% vs 45.6%）；(2) Unify——LWM RL warm-up（单轮无工具）跨 7 agentic benchmark（含 3 OOD 域）一致提升（Claw-Eval +11.3），机制是 prediction-driven action refinement（预测准确率 69.9%→78.3%）。基于 Qwen3.5 checkpoint。`raw/` 未改。图文化：5 张图，PyMuPDF 300 DPI + get_textbox 校验。


## [2026-07-12] ingest | MinerU2.5-Pro 技术报告

`raw/2604.04771v2.pdf`（arXiv:2604.04771v2, 上海 AI Lab + PKU + SJTU + 商汤, 2026-04-09, 43 页）。

新增：

- `wiki/sources/mineru-2-5-pro.md`：来源页（图文交错）。嵌入 Figure 1（OmniDocBench v1.6 性能对比四子图）、Figure 2（Data Engine 三组件总览：DDAS + CMCV + Judge-and-Refine）、Figure 3（DDAS 两粒度流水线）、Figure 4（v1.5 元素匹配偏差示例）。Table 1（三阶段训练配置）、Table 2（主结果 18 模型）、Table 3（训练阶段消融）转 Markdown；Table 4/5/6 元素级结果转散文 + 关键数字。待追问 6 条（HunyuanOCR 1.0 分数分歧根因 / CMCV 模型池选择偏差 / Judge-Refine 与 CMCV 池同源盲点 / 192K Hard 子任务分布 / GRPO mid-reward 阈值 / 200× 参数口径）。
- `wiki/models/mineru-2-5-pro.md`：模型页。关键事实表含模态=多模态（文本+图像输入；结构化文本输出）（已据原文核实）。技术身份四点（架构不变作控制变量 / CMCV 多模型交叉验证 / render-then-verify 打破自肯定偏差 / 三阶段对应数据质量层级）。与 HunyuanOCR-1.5 对照（路线正交：推测解码+agentic 数据 vs 数据中心方法论+评测修正）。
- `wiki/assets/mineru-2-5-pro/`：fig1-omnidocbench-v16-performance.png, fig2-data-engine-pipeline.png, fig3-ddas-two-granularity.png, fig4-matching-bias-examples.png（PyMuPDF 300 DPI + get_textbox 校验；fig4 纯图像无文字层，其余含图内标签无 caption 污染）

更新：

- `wiki/concepts/agentic-evaluation-benchmarks.md`：末段加 MinerU2.5-Pro 跨域评测方法论信号（MGAM 修正匹配粒度偏差 + Hard 子集区分力 + 自报分 vs 统一重测分不可混用，与 Qwen-AgentWorld differentiated matching criteria / Xiaomi 饱和域 / Agent-World scaling 呼应）。
- `wiki/comparisons/llm-rl-policy-optimization.md`：「与模型报告的关系」加 MinerU2.5-Pro 一条（Stage 3 GRPO+DAPO 的非 agentic 应用：任务指标直接作 reward / mid-reward hard filter / 增益小但定向，说明 GRPO+DAPO recipe 适用于任何 token-level loss 与任务级指标错位的结构化输出任务）。
- `wiki/concepts/data-mixture-optimization.md`：加「data-centric AI 的另一分支：难度感知采样 + 标注精修」段（MinerU2.5-Pro Data Engine 与 DoReMi/TANDEM domain reweighting 正交：优化 instance 价值+标注可信度而非 domain 比例，信号源是多模型一致性非 proxy loss，把标注质量作一等问题）。
- `wiki/sources/hunyuan-ocr-1.5.md`：待追问加 HunyuanOCR 1.0 自报分（92.03）vs MinerU 统一重测分（89.87）分歧条目；相关页面加 MinerU2.5-Pro 反向链接。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：MinerU2.5-Pro 是上海 AI Lab + PKU + SJTU + 商汤的数据中心文档解析报告，核心论点是架构成熟后系统化数据工程是推动性能的主要杠杆。完全保留 MinerU2.5 的 1.2B 解耦 coarse-to-fine 架构（NaViT-675M + Qwen2-0.5B）不变，Data Engine 三组件协同：(1) DDAS 在页级+元素级两粒度联合优化多样性（ViT embedding + K-Means 聚类）与难度（CMCV 加权）；(2) CMCV 用三个异构模型（MinerU2.5 / PaddleOCR-VL / Qwen3-VL-30B）输出共识分 Easy/Medium/Hard，锚定待改进模型相对外部表现，Medium 训练价值最高；(3) Hard 样本用 Judge-and-Refine（Qwen3-VL-235B，render-then-verify 把 LaTeX/HTML 渲染成图与原图配对，打破 self-reflection 自肯定偏差）+ Targeted Expert Annotation（Gemini 3 Pro 预标注 + 人工）。数据 <10M → 65.5M（自动）+ 192K（专家）。三阶段训练：大规模 SFT（+1.31）→ Hard-SFT（+0.96，表格 TEDS +2.50）→ GRPO 对齐（+0.45，公式 CDM +0.81，沿用 DAPO clip-higher + dynamic sampling，reward 直接用评测指标）。评测协议贡献 OmniDocBench v1.6：MGAM 修正 v1.5 匹配粒度偏差（预测端三阶段搜最优粒度）+ Hard 子集 296 页（Base/Hard/Full 三层）。OmniDocBench v1.6 Full 95.69 居首，超 GLM-OCR 95.15、PaddleOCR-VL-1.5 94.87、同架构 baseline 92.98，超 200× 参数通用 VLM（Qwen3-VL-235B 89.78、Gemini 3 Pro 92.85、GPT-5.2 86.52）。跨源分歧：HunyuanOCR 1.0 自报 92.03 vs 统一重测 89.87（差 2.16，根因未明）。`raw/` 未改。图文化：4 张图，PyMuPDF 300 DPI + get_textbox 校验。

## [2026-07-12] maintenance | 补录 index 孤儿条目（Ling-2.6 / Unlimited OCR）

`wiki/sources/ling-2.6.md`、`wiki/sources/unlimited-ocr.md` 及对应模型页已于 2026-07-11 ingest 并提交，但 index.md 漏收（"有页无索引"孤儿）。本次补录 4 条 index 条目：来源段 +2（ling-2.6 / unlimited-ocr）、模型段 +2（Ling-2.6 / Ring-2.6 纯文本 / Unlimited OCR 多模态）。模态据各模型页 关键事实 表核实。`raw/` 未改。

## [2026-07-12] ingest | GLM-OCR 技术报告

`raw/2603.10910v2.pdf`（arXiv:2603.10910v2, 智谱 AI + 清华, 2026-03-16, 17 页）。

新增：

- `wiki/sources/glm-ocr.md`：来源页（图文交错）。嵌入 Figure 1（OmniDocBench v1.5 性能对比四子图）、Figure 2（架构 + MTP + 两任务工作流）。Table 1（训练配方）、Table 2（Stage 4 reward 设计）、Table 3（公开 benchmark 6 模型）、Table 4（OmniDocBench v1.5 详细）、Table 5（自建真实场景 6 类）、Table 6（吞吐对比）转 Markdown 或散文。待追问 6 条（MTP k 值与共享参数细节 / 高并发吞吐反噬 / Stage 4 GRPO 配置 / CogViT 与 GLM-5V-Turbo 是否同源 / PP-DocLayout-V3 错误传播定量 / v1.5→v1.6 提升是否全归 MGAM）。
- `wiki/models/glm-ocr.md`：模型页。关键事实表含模态=多模态（文本+图像输入；结构化 Markdown/JSON 输出）（已据原文核实）。技术身份四点（MTP 训练+推理共用共享参数 / 解耦两阶段降幻觉 / 文档解析与 KIE 统一 / GRPO task-aware reward）。与 MinerU2.5-Pro（精度 vs 效率）+ HunyuanOCR-1.5（MTP vs DFlash）对照表。
- `wiki/assets/glm-ocr/`：fig1-omnidocbench-v15-performance.png, fig2-architecture-mtp-workflow.png（PyMuPDF 300 DPI + get_textbox 校验）

更新：

- `wiki/concepts/multi-token-prediction.md`：MTP 用法表加 GLM-OCR 行（训练+推理共用共享参数多头，10/5.2 tokens/step，~50% 吞吐）；新增「GLM-OCR 的经验：MTP 在确定性 OCR 任务下的双重收益」段（MTP 不只加速还提升结构化输出质量，OCR/结构化输出是 MTP 甜区，与 HunyuanOCR DFlash 观察呼应）。
- `wiki/sources/mineru-2-5-pro.md`：相关页面加 GLM-OCR 反向链接（头号竞争者，v1.6 上 95.15 < 95.69，MTP 加速 vs 数据中心方法论路线对照）。
- `wiki/index.md`：来源段 +1，模型段 +1（注意并发：index 已含别会话新增的 Ling-2.6 / Unlimited OCR 条目，本次只追加 GLM-OCR 两行）。

核心定位：GLM-OCR 是智谱 AI + 清华的 0.9B 轻量 OCR VLM（CogViT ~400M + GLM ~500M），核心论点是不靠大模型 scaling 而靠架构-解码-任务结构对齐拿效率增益。MTP 是核心加速点——k 个共享参数辅助头预测未来 k token，训练 10 tokens/step、推理平均 5.2 tokens/step、~50% 吞吐提升，且 MTP 同时是训练目标（不只推理加速），还带来结构化输出质量收益（鼓励向前规划，更少破损标签）。两阶段 pipeline（PP-DocLayout-V3 布局 + 并行区域识别）降幻觉，文档解析与 KIE 统一为条件结构化生成。五阶段训练：视觉编码器（MIM+CLIP+大 ViT 蒸馏）→ VL 预训练 → MTP 预训练 → SFT with MTP → RL（GRPO + task-aware reward：edit distance/CDM/TEDS/field-F1 + 结构验证 + 全局重复/畸形惩罚）。OmniDocBench v1.5 Overall 94.62 居首（0.9B 超 PaddleOCR-VL-1.5 94.50、MinerU2.5 90.67、Qwen3-VL-235B 89.15、Gemini-3 Pro 90.33），表格最强（TableTEDS 93.96/TEDS-S 96.39），吞吐 1.86 pages/s 约为 MinerU2.5 的 3.9×。跨源评测版本差异：GLM-OCR 自报 v1.5 = 94.62，MinerU2.5-Pro 统一重测 v1.6 = 95.15（+0.53，符合 MGAM 提分预期，与 HunyuanOCR 1.0 的 92.03 vs 89.87 反向分歧形成对照）。`raw/` 未改。图文化：2 张图，PyMuPDF 300 DPI + get_textbox 校验。

## [2026-07-12] ingest | MinerU2.5 技术报告

`raw/2509.22186v2.pdf`（arXiv:2509.22186v2, 上海 AI Lab + PKU + SJTU, 2025-09-29, 57 页）。

新增：

- `wiki/sources/mineru-2-5.md`：来源页（图文交错）。嵌入 Figure 3（Data Engine 三阶段总览）、Figure 7（IMIC 策略三任务图示 + 阈值）。Table 2（数据增强）、Table 3（推理性能）转 Markdown。待追问 5 条（IMIC 随机性来源 / IMIC 阈值依据 / Stage 0 数据规模 / OTSL 与 MTP 兼容性 / 三阶段独立运作自评）。
- `wiki/models/mineru-2-5.md`：模型页。关键事实表含模态=多模态（文本+图像输入；结构化 Markdown 输出）（已据原文核实）。含 MinerU2.5 → MinerU2.5-Pro 基座改进对照表（架构/数据/hard case 挖掘/标注精修/三阶段关系/RL/OmniDocBench 七维）。
- `wiki/assets/mineru-2-5/`：fig3-data-engine-overview.png, fig7-imic-strategy.png（PyMuPDF 300 DPI + get_textbox 校验）

更新：

- `wiki/concepts/data-mixture-optimization.md`：「data-centric AI 另一分支」段下新增「IMIC → CMCV：从单模型内省到多模型交叉验证」子段——补 IMIC 具体定义（单模型 n 次随机推理 + PageIoU/CDM/TEDS 阈值）+ CMCV 改进动机（单模型内省无法区分模型特定盲点 vs 普遍难题）+ Medium 训练价值最高的定位。
- `wiki/sources/mineru-2-5-pro.md`：相关页面加 MinerU2.5 基座反向链接（本报告 CMCV 改进其 IMIC，Data Engine 三组件协同改进其独立三阶段）。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：MinerU2.5 是上海 AI Lab + PKU + SJTU 的 1.2B 解耦文档解析 VLM（NativeRes-ViT 675M + Qwen2-0.5B），coarse-to-fine 两阶段（下采样图布局分析 + 原生分辨率裁剪内容识别）。是 MinerU2.5-Pro 的基座——架构完全被继承不变。Data Engine 三阶段独立运作：Data Curation（多维过滤）+ Pre-training Preparation（强模型精修标注）+ Fine-tuning Construction（IMIC + 专家）。IMIC（Iterative Mining via Inference Consistency）是核心 hard case 挖掘策略——用 MinerU2.5 Stage-1 checkpoint 单模型多次随机推理，算配对一致性（Layout PageIoU<0.8/Formula CDM<0.3/Table TEDS<0.6 = hard），低一致性送人工。这正是 MinerU2.5-Pro CMCV 的改进对象——IMIC 只能捕获单模型认知不确定性，无法区分模型特定盲点 vs 普遍难题；CMCV 用三异构模型交叉验证，Medium（外部一致、待改进模型不同）训练价值最高。训练：Stage 0 模态对齐 + Stage 1 预训练（6.9M×2，MinerU2.5-Pro 扩到 65.5M）+ Stage 2 微调（630K×3：43K layout+300K text+147K formula+140K table）。OTSL 表格语言（5 token vs HTML 28+，序列缩短 50%）。ADR 复合公式原子分解重组。部署 vLLM A100 2.12 pages/s，按布局类型动态调重复惩罚。`raw/` 未改。图文化：2 张图，PyMuPDF 300 DPI + get_textbox 校验。


## [2026-07-12] verify | Agent-World 内部矛盾核实（τ2-Bench prose vs Table 2）

`raw/2604.18292v1.pdf`（arXiv:2604.18292v1）重读 p16 Table 2 + p17 §4.3.4 prose 核实 wiki/sources/agent-world.md:175 既记的内部矛盾。确认：§4.3.4 prose 称 Agent-World-14B τ2-Bench 45.3%->50.5%，但同节 Table 2 实为 60.2%->65.4%；BFCL-V4（52.4->55.8）与 MCP-Mark（29.5->38.1）两列 prose 与表一致，仅 τ2 列不一致且增量（+5.2）一致。主表 Table 1 Agent-World-14B τ2 Avg=65.4 与 Table 2 +2 rounds 值吻合，故以表值为准，prose 的 45.3/50.5 为论文自身笔误。结论与既记一致，故仅把注释从「待核实」升级为「已据原文核实」并补 p16/p17 页定位，不改任何事实数字。`raw/` 未改。


## [2026-07-12] maintenance | 待追问批量回收（第一批：11 条 + PDF 核实）

TODO.md 清空（全部已完成项移除，留 header + "当前无待办"）。

待追问批量回收第一轮，从 265 条清理 11 条（265->254）：

- **已坐实移出（4 条）**：`sources/qwen3-coder-next.md`（HF config 已核实层数/比例）、`sources/qwen3.5-omni.md`（HF config 已核实 GDN:attention 比例）、`sources/qwen3-vl.md`（正文已核实 backbone = Qwen3 标准 GQA）、`sources/qwen3-next-blog.md`（已坐实 Qwen3.5 是 Qwen3-Next 延续）。
- **HF config 新坐实（2 条）**：`sources/gemma-4.md`（26B-A4B MoE 配置：128 expert / top-8 / 无 shared expert / 30 层 5:1 SWA/GA，已据 HF config 核实并补入正文）；`sources/internvla-a1.5.md`（Qwen3.5-2B backbone 配置：24 层 / 3:1 GDN:full / hidden 2048 / Unified Expert hidden 1024 / foresight tokens 50，已据 HF config 核实并补入正文）。
- **去重移出（2 条）**：`sources/deepseek-v2.md`（MLA 维度已在概念页坐实，source 页不再重复待追问）；`sources/agent-world.md`（τ2-Bench 矛盾已由子 agent 据原文核实为论文笔误，正文 blockquote 已升级为"已据原文核实"，待追问段不再重复）。
- **PDF 核实移出（1 条）**：`sources/keye-vl-2.md`（DSA top-k=2048，已据原文 p17 核实，补入架构段正文）。
- **PDF 核实更新措辞（2 条）**：`sources/kvpop.md`（Abstract 98%/97% vs contributions/conclusion 95%/94% 矛盾已据 p1/p2/p11 核实，标注"三处原文已核实"）；`sources/deepseek-v32.md`（indexer head 数已据 §2.1 p3 核实为"报告未覆盖"，标注"仅给符号 $H_I$，未赋值"）。
- **精简措辞（1 条）**：`concepts/multi-head-latent-attention.md`（主要维度已坐实，精简为只留投影矩阵形状待补）。

`raw/` 未改。

## [2026-07-13] ingest | Mach-Mind-4-Flash 技术报告

`raw/2607.09375v1.pdf`（arXiv:2607.09375v1，理想汽车 Foundation Model Team，2026-07-10）。

新增：
- `wiki/sources/mach-mind-4-flash.md`
- `wiki/models/mach-mind-4-flash.md`
- `wiki/assets/mach-mind-4-flash/`：fig1-benchmark-comparison.png, fig4-post-training-pipeline.png, fig13-hmpo-overview.png（PyMuPDF 300 DPI + vision_analyze 校验）

更新：
- `wiki/concepts/multi-teacher-on-policy-distillation.md`：新增「Mach-Mind-4-Flash 的 MOPD」段--统一 RL/OPD loss（唯一把 RL 和 OPD 混进单一加权 loss 的实现）、Early Stopping Rollout（8K 截断）、teacher-student 参数量匹配策略、Appendix C 消融的 code teacher 跨域迁移效应。
- `wiki/comparisons/on-policy-distillation.md`：速览表 +1 行（Mach-Mind），A 类融合派讨论补 Mach-Mind 段（统一 RL/OPD loss + 跨域迁移量化数据）。
- `wiki/concepts/post-training-for-agentic-models.md`：综合框架 +1 条（后训练 scaling 作为紧凑模型追赶前沿主路径）。
- `wiki/concepts/moe-frontier-model-scaling.md`：对比表 +1 行（35B/3B），解释段补 Mach-Mind 定位。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：Mach-Mind-4-Flash 是理想汽车的 35B MoE / 3B 激活 agentic 模型，基座 Qwen3.5-35B-A3B（不改预训练权重）。三支柱：(1) 统一 RL/OPD 训练框架（`L = α·L_OPD + β·L_RL`，三模式切换）；(2) specialization-then-integration（三轨并行 RL 共 10+ 专家 → MOPD 融合）；(3) HMPO token 效率（中位长度 adaptive budget + 乘法 reward 防 hacking，仅数学训练压缩 19–46% 跨域泛化）。MOPD 用 k1 estimator + PPO clipped surrogate（Appendix B 完整推导）。3B 激活在 AIME'26 92.70 / IFBench 82.82 / Behavioral-SafetyBench 80.74 追平或超越 10–30× 激活规模模型。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + vision_analyze 校验。

## [2026-07-29] ingest | Mi-Memory: A Lifecycle Memory Framework for Personal AI

沉淀小米 Darwin Agent Team 的 Personal AI 记忆全生命周期框架论文（arXiv:2607.18975v1，52 页）。

**新增页面**：
- `wiki/sources/mi-memory.md`——源页，覆盖四个生命周期角色（Structure / Expansion / Evolution / Deployment）的完整机制、审计契约四类工件、全部评测结果和证据分级。
- `wiki/concepts/agent-memory-lifecycle.md`——概念页，提炼 lifecycle 范式定义、四角色分工、与传统 RAG 的区别、跨报告信号。
- `wiki/assets/mi-memory/`——9 张图：fig1 lifecycle / fig2 architecture / fig3 memstack runtime / fig4 controlled-reference / fig5 IKB path / fig6 memfuse / fig7 mem-gallery accuracy / fig8 dual-loop / fig9 litemem。

**关键发现**：
- 论文核心贡献是 lifecycle audit contract 本身——用 typed evidence / diagnostic traces / strategy artifacts / gate-rollback 四类工件统一异构模块，而非单一 leaderboard 数字。
- MemStack（Structure）在 LoCoMo / PersonaMem-V2 / LongMemEval 上达 93.59% / 57.24% / 87.47%（controlled reference vs reproduced MemBrain）。
- E2MEND（Evolution）在 LoCoMo 上从 75.58% 提升至 94.74%（+19.16pp，descriptive），但非 repeated-run statistical claim。
- LiteMem（Deployment）达 90.81%，保留 90.0% 服务端改善，97.0% absolute ratio vs full MemStack。
- 证据分级诚实：显式区分 controlled reference / module-level / preliminary-internal / transfer-feasibility / design-only 五级，MemFuseBench 为内部 benchmark。

`raw/` 未改。临时文件 `raw/_mi_memory_text.txt` 已清理。

## [2026-07-30] ingest | nrehiew 博客：SFT, RL, and On-Policy Distillation Through a Distributional Lens

新增博客源页 `wiki/sources/nrehiew-sft-rl-opd.md`，原始快照存于 `raw/nrehiew-sft-rl-opd.md`（`raw/` 未改）。

博客用分布视角（target distribution / 数据来源 / KL 方向）统一对照 SFT、RL、OPD 三方法，核心论点：**on-policy 数据是抗遗忘的承重墙**，不是 RL 本身或显式 KL 惩罚。关键实验：在 Minimal Code Editing 任务上，OPD student 不论从 SFT 还是 RL teacher 蒸馏，结果几乎一致且都略超 RL teacher--说明 teacher 提供信号但 on-policy 采样决定几何形状。

同步 deepen 两个已有页面：

- `wiki/concepts/multi-teacher-on-policy-distillation.md`：新增「分布视角三轴对照」段（SFT/RL/OPD 的 target/数据/KL 方向/梯度范围/遗忘）+「on-policy 数据是承重墙」对照实验 +「OPSD」变体（self-distillation + reference solution prefix + per-token clipping）+「Student 超越 Teacher」机制（OPD 监督更精准 + KL matching≠reward maximization + entropy collapse）。待追问补充 3 条。
- `wiki/comparisons/on-policy-distillation.md`：新增「domain reward 噪声决定 teacher 类型」段（MiMo Table 7 的 RL domain vs self-distill domain 分歧 + student-teacher 差距方向）。待追问补充 2 条，相关页面补 nrehiew 源页。

无图嵌入（博客原图为 SVG 概念图，核心信息已用表格和 prose 覆盖）。


## [2026-07-30] ingest | Kimi K3 技术报告

`raw/2607.24653v1.pdf`（arXiv:2607.24653v1，Moonshot AI Kimi Team，2026-07-27，47 页）。

新增：

- `wiki/sources/kimi-k3.md`：来源页（图文交错）。嵌入 Figure 7（scaling law 2.5×）、Figure 2（架构总览）、Figure 3（KDA scaled sigmoid decay）、Figure 5（Quantile Balancing）、Figure 12（KDA-aware prefix cache）。Table 1（K2 vs K3 架构对比）、Table 2（主表 6 模型 × 30+ benchmark）转 Markdown。待追问 10 条（g_min=-5 选择 / Block AttnRes N=8 在 93 层是否最优 / MoonViT-V2 追平 SigLIP 量化 / MOPD 9 teacher 混采策略 / MoonEP E/R bound tightness / prefix cache hit rate 实测 / cosine vs WSD 独立超参细节 / Per-Head Muon 改善量化 / SiTU-GLU β 选择 / Cyber Tier 2 可复现性）。
- `wiki/models/kimi-k3.md`：模型页。关键事实表含模态=多模态（文本+图像+视频）（已据原文 § 2.4 + Abstract 核实，原生多模态单一共享 backbone）。技术身份按三信息流轴（序列 KDA-MLA / 深度 AttnRes / 宽度 Stable LatentMoE）+ 原生视觉 + Per-Head Muon 组织。含 K2→K2.5→Kimi Linear→K3 同族演进表。
- `wiki/concepts/attention-residuals.md`：新概念页。Full AttnRes（每层 attend 所有前层，softmax kernel + RMSNorm）+ Block AttnRes（N=8 × S=12，O(Nd) 内存，online softmax 合并）。定位为深度维 attention，类比 Transformer 解序列 RNN 瓶颈。
- `wiki/concepts/stable-latentmoe.md`：新概念页。LatentMoE（routed 在 ℓ=d/2 latent 空间）+ Normalized（RMSNorm）+ SiTU-GLU（bounded |f|≤β1·β2=100，β1=4 β2=25）+ Quantile Balancing（balanced assignment 对偶 LP 的 exact coordinate minimizer，无学习率，histogram 估分位数）。支撑 896-expert/16-active sparsity 56 在 2.8T 稳定。
- `wiki/assets/kimi-k3/`：fig2-architecture.png, fig3-kda-decay.png, fig5-quantile-balancing.png, fig7-scaling-law.png, fig12-prefix-cache.png（PyMuPDF 300 DPI + get_textbox 校验）

更新：

- `wiki/concepts/linear-attention-and-delta-rule.md`：演进表加 K3 行（scaled sigmoid lower-bounded decay + full-rank gate）；新增「Kimi K3 的 KDA 升级」段（g_min=-5 有界 → 所有因果 tile 用 dense Tensor Core MMA + KCP 跨设备 CP）；跨报告信号 + K3；相关页面 + K3。
- `wiki/concepts/multi-head-latent-attention.md`：跨报告信号加 K3 Gated MLA（NoPE + full-rank output gate + FP32 attention output）；相关页面 + K3。
- `wiki/concepts/attention-gating.md`：跨报告信号加 K3（full-rank 门同时用在 KDA + Gated MLA，比 Gated Attention 消融的 head-specific elementwise 门更重）；相关页面 + K3。
- `wiki/concepts/moe-frontier-model-scaling.md`：对比表 + K3 行（2.78T/104.2B，首个开源 3T，896 routed/16 active sparsity 56）；解释段补 K3 定位 + Stable LatentMoE + MoonEP。
- `wiki/concepts/multi-token-prediction.md`：用法表 + K3 行（预训练 MTP 层 → EAGLE-3 draft，融合 1st/4th/final AttnRes block 特征，直接优化 LK loss = acceptance rate 负对数）；新增「Kimi K3 的经验」段（LK loss vs KL surrogate + AttnRes block 多粒度特征利用）。
- `wiki/concepts/asynchronous-agent-rl.md`：新增「Kimi K3 的 partial rollout + AgentENV」段（λ 比例 partial rollout + Firecracker microVM 51M sandbox + external KV cache pool KDA/MLA 同步 + auto-throttling + gradient-buffer reuse）+ 三家对比（K3 vs GLM-5 vs Ring-2.6）。
- `wiki/concepts/post-training-for-agentic-models.md`：新增「Kimi K3」段（3-stage SFT→9-专家 RL→MOPD + Unified White-Box RL Env harness-agnostic + AgentENV 1M partial rollout）；综合框架 + K3。
- `wiki/concepts/multi-teacher-on-policy-distillation.md`：新增「Kimi K3 的 MOPD」段（9 teacher = 3 域 × 3 reasoning effort 矩阵，首次把 effort 作正交 teacher 维度 + R_max clip + top-k 无优势）。
- `wiki/comparisons/2026-open-model-technical-reports.md`：范围 + K3；对比表 + K3 行；二版综合 + K3 哲学（两条 scaling 轴同时 push）。
- `wiki/comparisons/on-policy-distillation.md`：速览表 + K3 行；A 类融合派讨论 + K3（9 teacher 矩阵 + R_max clip）；未用 OPD 列表 + Ling-2.6。
- `wiki/index.md`：来源段 +1，模型段 +1，概念段 +2（Attention Residuals / Stable LatentMoE）。

核心定位：Kimi K3 是 Moonshot AI 的首个开源 3T 级模型（2.78T 总参 / 104.2B 激活），原生多模态（文本+图像+视频），1M token 上下文。架构沿三条信息流轴重新设计：(1) 序列长度——Hybrid KDA-MLA 3:1 混合，KDA 升级 scaled sigmoid lower-bounded decay（g_min=-5 有界 → 所有因果 tile 用 dense Tensor Core MMA）+ full-rank output gate，Gated MLA 用 NoPE + full-rank gate + FP32 attention output；(2) 网络深度——Attention Residuals（Block AttnRes N=8 × S=12，每层 attend 所有前层 block 表示，解除标准残差的深度 RNN 瓶颈）；(3) 模型宽度——Stable LatentMoE（LatentMoE routed 在 ℓ=d/2 latent 空间 + Normalized RMSNorm + SiTU-GLU bounded activation |f|≤100 + Quantile Balancing aux-loss-free 的 exact 对偶 LP 解），支撑 896 routed/16 active/2 shared（sparsity 56）。配 MoonViT-V2（27 层 ViT ~401M，从零训练非 SigLIP init）原生视觉 + Per-Head Muon optimizer。综合架构+数据+训练改进相对 Kimi K2 拿到 2.5× scaling efficiency。后训练三阶段 SFT→9-专家 RL（3 域 × 3 reasoning effort）→ MOPD 融合，QAT 贯穿（MXFP4/MXFP8），Unified White-Box RL Env（harness-agnostic，配置化实例化 Kimi Code/Claude Code/Codex/OpenClaw/Hermes），AgentENV microVM 沙箱（51M+ sandbox，Firecracker，incremental checkpointing 49ms/133ms，pause/resume/fork/snapshot）支撑 1M partial rollout。基础设施含 FlashKDA kernel + KCP（KDA Context Parallelism，prefix scan + 单次 all-gather）+ MoonEP 完美平衡 EP（E/R bound tight）+ KDA-aware prefix cache（unified paged layout，hash block 512 / physical block 6144 解耦）+ cache-aware affinity scheduling + budget-based admission control。评测 frontier 级但承认落后 Claude Fable 5 / GPT-5.6 Sol：ProgramBench/SWE-Marathon/BrowseComp/MCPMark-Verified/AutomationBench/τ3-Banking/Harvey Lab-AA 等 8 项 #1，WebDev Arena 1,678 Elo 首个开源登顶，Intelligence Index v4.1 #4/580（57.1），Vals Index #2/39（74.7）。`raw/` 未改。图文化：5 张图，PyMuPDF 300 DPI + get_textbox 校验。


## [2026-07-30] ingest | Laguna M.1/XS.2 技术报告

`raw/2605.27605v1.pdf`（arXiv:2605.27605v1，Poolside Team，2026-05-28，37 页）。

新增：

- `wiki/sources/laguna-m1-xs2.md`：来源页（图文交错）。嵌入 Figure 1（agentic benchmark 对比）、Figure 2（dispatch/GEMM overlap）、Figure 3（web 数据工作流）、Figure 5（Hive runtime）、Figure 6（AutoMixer pipeline）、Figure 7（outlier 激活累积）。Table 4（XS.2 数据混合）、Table 5（base 评测）、Table 6/7（agentic 评测）、Table 9（SWA 架构消融）转 prose/Markdown。待追问 7 条（WSD 缩放律外部验证 / softplus vs sigmoid 门 / AutoMixer proxy 成本与 lambda sensitivity / CISPO (1,4) vs MiniMax-M1 原设置 / 合成代码环境 vs SWE-Smith 规模对照 / dispatch kernel 开源 / 256K 纯 RoPE scale 翻倍无训练的真实长程表现）。
- `wiki/models/laguna.md`：模型页。关键事实表（M.1 vs XS.2 两列）+ 四轴技术身份（流程/注意力/数据/后训练）+ 同重量级定位表（含 Devstral 2 / Qwen3.6 / Gemma 4 / MiMo-V2-Flash）+ 评测摘要。
- `wiki/assets/laguna/`：6 张图（fig1-benchmarks / fig2-dispatch-overlap / fig3-web-workflow / fig5-hive-runtime / fig6-automixer / fig7-outlier-activations，PyMuPDF 300 DPI + caption 定位裁剪 + get_textblocks 校验）。

更新（6 concept + 2 comparison + index）：

- `wiki/concepts/data-mixture-optimization.md`：方法谱系表 + AutoMixer 行；新增「AutoMixer：产业落地的 per-capability 回归路线」段（~60 个 0.5B MoE proxy + Dirichlet 扰动 + KL 正则 + per-capability 独立回归器，frontier-scale MoE 首验，held-out 泛化 + commonsense 负相关被建模）；待追问改「MoE 是否有效」为已答 + 新增成本/lambda/零和解追问；相关页面 + Laguna。
- `wiki/concepts/attention-gating.md`：跨报告信号 + Laguna（softplus per-head gating [67]=Gated Attention 报告，非 Qwen 团队又一生产采用，配 16B proxy Table 9 消融数据）；相关页面 + Laguna。
- `wiki/concepts/efficient-long-context-attention.md`：模式稀疏行 + Laguna（3:1 收紧 vs Gemma 4 的 5:1，window 512）；机制层新增「Laguna XS.2」段（3:1 + softplus 门 + GA partial RoPE(50%) + theta=5e5；消融链 Table 9 显示直接加 SWA-1024 让长上下文退化，靠 partial RoPE + 缩窗 + Q-head 分配捞回；256K 纯 RoPE scale 翻倍无训练）。
- `wiki/concepts/moe-frontier-model-scaling.md`：对比表 + Laguna M.1 / XS.2 两行；解释段新增「Laguna」段（不追绝对规模，工业流程复用速度为核心，M.1 到 XS.2 五周，四架构改动全靠 proxy 消融翻 flag）。
- `wiki/concepts/post-training-for-agentic-models.md`：新增「Laguna：Model Factory + 三阶段 + CISPO + 合成代码环境」段（合成代码环境双端检查贯穿 SFT/RL；CISPO + length-weighted LOO + 首个公开 vs GRPO/GSPO 消融理由；IF judge + multi-harness 防过拟合；TITO + 权重同步 + FP8 KV cache）；综合框架 + Laguna。
- `wiki/concepts/asynchronous-agent-rl.md`：新增「跨报告信号：Laguna 的在线 agentic RL 基础设施」段（在线路线非异步解耦；TITO + render_assistant_messages_raw chat-template 对齐断言；trainer 到 inference 每 2 step NCCL RDMA 同步 + KV-cache reset + staleness 上限 10 自然消除；FP8 KV cache 翻倍并发；与 GLM-5/K3/Ring-2.6 三家定位对比）。
- `wiki/concepts/agentic-engineering.md`：跨报告信号 + Laguna（Model Factory 工业化流程为核心瓶颈，五周交付实证，合成代码环境贯穿 SFT/RL + CISPO + IF judge + multi-harness）；相关页面 + Laguna。
- `wiki/comparisons/llm-rl-policy-optimization.md`：description 更新含 CISPO；对比表 + CISPO 行（MiniMax-M1 源头，Laguna 采用）；新增「CISPO：不争 ratio 单元也不争采样，而争 clipping 的形状与方向」段（asymmetric (1,4) 有效 [0,5]，下界不裁上界裁 5，与 DAPO Clip-Higher 动机相反的开放问题）；相关页面 + Laguna。
- `wiki/comparisons/2026-open-model-technical-reports.md`：范围 + Laguna；对比表 + Laguna 行；二版综合 + Laguna 哲学（模型开发流程本身是主杠杆，工艺到工业转型样本）。
- `wiki/index.md`：来源段 +1，模型段 +1。

核心定位：Laguna 是 Poolside 的 MoE agentic coding 模型族（M.1 225.8B/23.4B、XS.2 33.4B/3B），纯文本。论点核心不在单点架构创新而在 Model Factory 工业化流程——M.1 预训练结束后五周从零交付 XS.2，四项架构改动（3:1 SWA/GA、WSD、expert 调制、dense 层 3 到 1）全靠 16B MoE proxy 消融后翻配置 flag。XS.2 用 3:1 interleaved SWA/GA（vs Gemma 4 的 5:1）+ softplus per-head gating（[67]=Gated Attention 报告）+ GQA 8 KV heads + GA partial RoPE(50%) theta=5e5 / SWA window 512 theta=1e4。预训练 30T+ tokens，数据侧三件套：高召回 web 管线（Propella 多维标注 + composite score 排序，恢复 34% 被误杀高质量文档）+ Hive 合成数据（~13% 混合，4 类 pipeline）+ AutoMixer 自动混合优化（~60 个 0.5B proxy + Dirichlet + per-capability 回归 + KL 正则，是 DoReMi/DoGE/RegMix/TANDEM 谱系在 frontier-scale MoE 的产业落地）。全程 Muon（Moonlight variant，distributed 实现降开销到 <1% step time）+ WSD 调度 + 自拟合 WSD peak-LR 缩放律 lr*(N,D)=10^4.488·N^-0.4639·D^-0.2661。稳定性三教训（expert collapse 到 Moonlight LR scaling；LM head 输入梯度 all-reduce 强制 FP32；padding token 跳过 routing）。后训练三阶段 mid-train(~60B) 到 SFT(3×40B) 到 agentic RL，RL 用 CISPO（[14]=MiniMax-M1）asymmetric clip (1,4) + length-weighted LOO advantage，首个公开 vs GRPO/GSPO 消融选择理由；合成代码环境（git commit 双端检查 到 30 到 60k 可验证任务）贯穿 SFT/RL；IF judge + multi-harness（OpenHands/OpenCode2/Mini-SWE-Agent）防过拟合；TITO + chat-template 对齐断言 + trainer 到 inference 每 2 step RDMA 同步 + FP8 KV cache rollout。量化 FP8/INT4/NVFP4 + FP8 KV cache，INT4 混合精度（前 30 层 INT4 / 后 10 层 INT8，因 ~30 层起 outlier 累积），NVFP4 用 QAD + DeepSeek-V4 hidden-state caching。评测：M.1 SWE-bench Verified 79.6 追平 Devstral 2、Terminal-Bench 2.0 52.5 次于 Claude Sonnet 4.6；XS.2 SWE-bench Verified 73.4 略胜 Qwen3.6-35B-A3B。诚实披露四 agentic benchmark 均有 reward hacking 漏洞，已 patch + 自建 judge 后置检测。raw/ 未改（临时文件 raw/_laguna_text.txt 留作参考，未删）。图文化：6 张图，PyMuPDF 300 DPI + caption 定位裁剪。

## [2026-07-31] ingest | DynamixSFT 技术报告

`raw/2508.12116v2.pdf`（arXiv:2508.12116v2，Microsoft Research Asia + UMich + KAIST，v2 标注 2026-07-03，14 页）。

新增：

- `wiki/sources/dynamix-sft.md`：来源页（图文交错）。嵌入 Figure 1（overview：数据集 collection → MAB arms → Prior-scaled Boltzmann + 1-Step Look-ahead Reward → mixture distribution → Trained LLM）、Figure 2（三策略 mixture proportion + instance coverage 可视化，头条结果）、Figure 7（prior vs w/o prior dynamics，证明 prior 必要性）。Table 1（preliminary G1/G2/G3 + 课程顺序）、Table 2（主结果 4 setting 摘选 3 子表）、Table 3（prior 消融）、Table 4（efficiency overhead）转 Markdown。待追问 4 条，核心是 reward 计算在 §4.2 / Algorithm 1（virtual gradient step，需反向传播 + 参数更新）与 §8（solely forward passes, no backward computation）之间存在描述矛盾，+12.7% 开销推导 19/50×1/3 依赖后者，若是后者则开销推导站不住。
- `wiki/assets/dynamix-sft/`：3 张图（fig1-overview / fig2-sampling-strategies / fig7-prior-effect，PyMuPDF 300 DPI + caption 定位裁剪 + get_textbox 校验，图内标签齐全、无 caption/footnote/左栏正文 bleed）。

未新建模型页：DynamixSFT 是 SFT 数据混合方法论文，非模型发布，基座用现成 LLaMA3.2 1B / LLaMA3.1 8B，按 IndexCache/MSA 先例不配模型页。未更新比较页：4 个现有比较页（前沿模型技术报告 / 稀疏注意力 / OPD / RL policy optimization）均不匹配 SFT 数据混合主题。

更新（1 concept + index）：

- `wiki/concepts/data-mixture-optimization.md`：frontmatter description 扩展含 SFT 在线无 proxy 分支；定义段点明主谱系是预训练 domain reweighting、DynamixSFT 是 SFT 在线无 proxy 另一分支；方法谱系表 + DynamixSFT 行；新增「DynamixSFT：SFT 阶段的在线无 proxy 分支」段（MAB + Prior-scaled Boltzmann + 1-step look-ahead reward；与 proxy-model 谱系范式分叉——直接在目标模型在线优化 vs 用 proxy 预测；prior 乘性进采样概率 vs AutoMixer KL 加性进优化目标；learning progress 理论支撑 r≈η||∇L||² + telescoping 累积 reward = L(θ0)-L(θT+1)；开销 +12.7% vs DoReMi 8% FLOPs 量级；消融去 prior 27.7→25.9 即便按比例初始化仍 26.3）；待追问 + reward forward/backward 矛盾与 frontier-scale 孰优；相关页面 + DynamixSFT。
- `wiki/index.md`：来源段 +1。

核心定位：DynamixSFT 是 MSRA + UMich + KAIST 的 SFT 指令微调数据集动态混合优化方法。把「从 K 个候选数据集采一个 batch」建模为 Multi-Armed Bandit（每数据集 = arm），两个关键设计（均原文确证）：(1) Prior-scaled Boltzmann Exploration——在 softmax(βQ) 上乘原始数据集比例 p^(0) 作先验软锚定采样分布 + γ/K minimum floor 防 never-sampling；(2) 1-Step Look-ahead Reward——每 T_update 步对每数据集做虚拟单步更新，用 (L_pre-L_post)/(L_pre+ε) 作 reward，min-max 归一化 + EMA 平滑；理论上一阶 Taylor r≈η||∇L||² 故衡量 learning progress 而非 raw loss，telescoping 累积 reward = L(θ0)-L(θT+1) 故 MAB 优化累积 reward 等价长期收敛（非启发式 trick）。TÜLU-2/3 上相对 Full Coverage +5.1%/+5.3%（10 benchmark 平均），一致优于 HBO/MultiDDS/MultiUAT，开销仅 +12.7%（1.13×，vs HBO +139%/MultiUAT +380%/MultiDDS +760%）。与 DoReMi/RegMix/TANDEM/AutoMixer 的 proxy-model 谱系范式分叉（在线无 proxy vs proxy 放大），但实验仅到 8B，frontier-scale 孰优未决；dataset-level，与 Qwen3 instance-level 不同粒度。内部矛盾：§4.2/Algorithm 1 reward 来自 virtual gradient step（需反向传播）与 §8 solely forward passes 描述不一致，+12.7% 推导依赖后者，需作者澄清。`raw/` 未改。图文化：3 张图，PyMuPDF 300 DPI + get_textbox 校验。
