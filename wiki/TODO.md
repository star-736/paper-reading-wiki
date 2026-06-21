# 待办

本文件是**未完成的工程待办**，与 `log.md`（已完成动作的时间线）分开。完成某项后，把它从这里移除，并按 workflow 规则在 `log.md` 记一条对应条目。

## 图文化批量回填

把各页已引用但尚未内嵌的关键 Figure/Table 按图文化约定（见 `AGENTS.md` / `CLAUDE.md` → "Figures & visual material"）回填。原则：机制/架构图截图内嵌，纯文字表格重排为 Markdown。

### 已知引用、待处理

- [ ] `sources/deepseek-v32.md`（§ 推理成本）引用 **Figure 3**（H800 上 DSA vs V3.1-Terminus 长序列 token 成本对比）——成本曲线图，宜截图内嵌。
- [ ] `concepts/deepseek-sparse-attention.md` 引用 **附录 Table 10**（indexer 配置：32 head / head dim 128）——纯文字表，宜重排为 md。
- [ ] `concepts/multi-teacher-on-policy-distillation.md` 引用 **Table 7**（MOPD 师生 benchmark 对比，AIME/HMMT/LiveCodeBench/SWE-Bench）——纯文字表，宜重排为 md。
- [ ] `concepts/multi-head-latent-attention.md` 引用 **附录 D.1**（7B dense MHA>GQA/MQA 消融）——若原文是表，可重排补在「压低秩而非减头数」论据处。

### 待勘查（页面未显式引用 Figure，但 PDF 里可能有值得内嵌的架构图）

- [ ] `concepts/deepseek-sparse-attention.md` — 翻 `raw/DeepSeek-V3.2.pdf` 看 DSA 架构图（lightning indexer + token selection）是否值得内嵌。
- [ ] `comparisons/sparse-attention-mechanisms.md` — 翻 `raw/` 的 MSA、IndexCache PDF，看 MSA / NSA / 跨层索引复用的机制示意图。
- [ ] `concepts/agent-swarm.md`、`concepts/multimodal-agentic-training.md` — Kimi K2.5 报告里的 Agent Swarm / MoonViT 架构图。

### 备注

- 回填属 `deepen`（给已有页补图、不改事实）；纯重排表格不改事实也可归 `refactor`。每次回填后跑写回检查单 + 记 `log.md`。
- `vision_analyze` 当前可用（换模型后），切图后用它核对像素与文字描述是否一致。
- 不要一次全做完；按页推进，一页跑顺再下一页。
