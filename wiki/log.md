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
