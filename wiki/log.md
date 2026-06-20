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
