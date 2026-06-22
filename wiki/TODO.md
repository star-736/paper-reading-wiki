# 待办

本文件是**未完成的工程待办**，与 `log.md`（已完成动作的时间线）分开。完成某项后，把它从这里移除，并按 workflow 规则在 `log.md` 记一条对应条目。

## 图文化批量回填

把各页已引用但尚未内嵌的关键 Figure/Table 按图文化约定（见 `AGENTS.md` / `CLAUDE.md` → "Figures & visual material"）回填。原则：机制/架构图截图内嵌，纯文字表格重排为 Markdown。

### 已知引用、待处理

（全部完成，2026-06-22。）

- [x] `sources/deepseek-v32.md` 引用 **Figure 3**（H800 上 DSA vs V3.1-Terminus token 成本对比）→ 已截图内嵌（prefilling + decoding 双子图）。
- [x] `concepts/deepseek-sparse-attention.md` 引用 **附录 Table 10**（indexer 配置：32 head / head dim 128）→ 已重排为 md（GLM-5 报告附录，含 GLM-4.5 对照）。
- [x] `concepts/multi-teacher-on-policy-distillation.md` 引用 **Table 7**（MOPD 师生 benchmark 对比）→ 已重排为 md（12 项 benchmark，含 teacher 类型标注）。注：修正 BrowseComp student 数字 45.4→44.9。
- [x] `concepts/multi-head-latent-attention.md` 引用 **附录 D.1**（7B dense MHA>GQA/MQA 消融）→ 已据 DeepSeek-V2 Table 8 重排为 md（BBH/MMLU/C-Eval/CMMLU 四项）。

### Linear-attention 页回填（2026-06-21 ingest 产出纯文本，已勘查图清单）

这批页（Kimi Linear / Gated Delta / Gated Attention）是图文化嵌进 ingest 硬步骤**之前**生成的，故无图。已回 PDF 勘查，候选按价值分档（页码为 PDF 页）。**高价值优先**：

- [x] **Kimi Linear Figure 3**（模型架构图，N× KDA + 1× MLA 块栈 + KDA/MoE 展开）→ 已嵌 `concepts/linear-attention-and-delta-rule.md`。注：原 TODO 写「Figure 3 Neural Parameterization」有误——"Neural Parameterization" 是 p4 的段落标题，真正的 Figure 3 是 p5 的模型架构图（更有价值，已嵌）。图中只标 N×/1×，3:1 是正文 N=3。
- [x] **Gated Delta Figure 1**（H1=GDN+SWA / H2=Mamba2+GDN+SWA 混合栈 + Gated Delta Rule block 设计）→ 已嵌 `concepts/linear-attention-and-delta-rule.md`（GDN 跨报告信号处）。
- [x] **Kimi Linear Figure 1**（Performance vs. acceleration，Pareto 图）→ 已嵌 `sources/kimi-linear.md`。
- [x] **Kimi Linear Figure 2**（kernel 执行时间 2K–64K，KDA vs DPLR）→ 已嵌 `concepts/linear-attention-and-delta-rule.md`（硬件效率段）。
- [x] **Kimi Linear Table 1**（hybrid ratio 3:1 消融 PPL）→ 已**重排为 Markdown 表**入 `sources/kimi-linear.md`（含 output gate / conv 消融行）。
- 中价值（可选，未做）：Kimi Linear Figure 7（p13，prefill/decode 时间分解）；Gated Delta Table 1（p4，linear RNN 架构对比，宜重排 md 进对比页）。
- 低价值（跳过）：Gated Attention 全篇图多为 gating-score 分布 / massive-activation 分析子图，信息密度对概念页帮助有限；各 benchmark 表与 scaling-law 曲线转文字已够。
- `sources/gated-delta-net.md`：其关键混合架构图即 GDN Figure 1，已嵌概念页（该页链回本来源页，满足双向引用）；source 页本身暂不重复嵌图。`sources/qwen3-coder-next.md`、`sources/qwen3.5-omni.md`：两篇均「继承 Qwen3-Next 架构、未自带独立机制架构图」，无值得内嵌的一手机制图，跳过。

### 待勘查（页面未显式引用 Figure，但 PDF 里可能有值得内嵌的架构图）

（全部勘查完成，2026-06-22。）

- [x] `concepts/deepseek-sparse-attention.md` — 翻 `raw/DeepSeek-V3.2.pdf` 看 DSA 架构图 → **Figure 2 值得内嵌**（lightning indexer + top-k selector + MLA MQA mode 完整数据流），已截图嵌架构概览段。
- [x] `comparisons/sparse-attention-mechanisms.md` — 翻 `raw/` 的 MSA、IndexCache PDF → **MSA Figure 1**（Index Branch + Main Branch + attention mask 可视化）已嵌对比表 MSA 行下；**IndexCache Figure 2**（标准 DSA vs IndexCache 伪代码对比）已嵌对比表 IndexCache 行下。
- [x] `concepts/agent-swarm.md`、`concepts/multimodal-agentic-training.md` — Kimi K2.5 报告 → **Figure 3**（Agent Swarm orchestrator + frozen subagents 并行架构）已嵌 agent-swarm.md 定义段；**Figure 10**（agentic RL framework overview: Rollout Manager / Core Agent Loop / Inference+Training Engine）已嵌 multimodal-agentic-training.md 的 Joint multimodal RL 段。

### 备注

- 回填属 `deepen`（给已有页补图、不改事实）；纯重排表格不改事实也可归 `refactor`。每次回填后跑写回检查单 + 记 `log.md`。
- `vision_analyze` 当前可用（换模型后），切图后用它核对像素与文字描述是否一致。
- 不要一次全做完；按页推进，一页跑顺再下一页。
