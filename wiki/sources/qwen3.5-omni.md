# Qwen3.5-Omni 技术报告

## 来源

- 原始 PDF：[raw/Team - 2026 - Qwen3.5-omni technical report.pdf](../../raw/Team%20-%202026%20-%20Qwen3.5-omni%20technical%20report.pdf)
- 标题：Qwen3.5-Omni Technical Report
- 团队：Qwen Team（Alibaba Group），2026（arXiv:2604.15804）

## 核心结论

Qwen3.5-Omni 是 Qwen-Omni 全模态家族的最新一代，规模扩到数千亿参数、支持 **256k 上下文**，靠 1 亿+ 小时音视频 + 异构图文对训练，具备强全模态能力（Qwen3.5-Omni-Plus 在 215 个音频/音视频子任务上 SOTA，关键音频任务超过 Gemini-3.1 Pro）。

架构上，Thinker 和 Talker 都构建在 **Qwen3.5 引入的 Hybrid Attention MoE** 框架上，其中包含 **Gated DeltaNet（GDN）模块**——报告明说它对长音视频序列建模特别有效，显著降低长上下文推理的 KV-cache I/O 开销、提升吞吐与并发。

> **这页是「采用证据」**：它确认 [GDN 线性注意力](../concepts/linear-attention-and-delta-rule.md) 已用于一个生产级**多模态**模型，并把「GDN 降 KV-cache I/O」从研究主张落到全模态长序列场景。注意力机制本身的一手出处仍是 [GDN 报告](gated-delta-net.md) 与 [Gated Attention 报告](gated-attention.md)；本报告引用而非重新推导，不构成机制层 tier-1 来源。

## 架构与训练

- **Hybrid Attention MoE**：Thinker / Talker 共用，继承自 Qwen3.5，含 GDN 模块；GDN 用于加速长音视频序列、降 KV-cache I/O。
- **全模态能力**：支持 >10 小时音频理解、400 秒 720P 视频（1 FPS）。
- **Chunked Prefilling**：沿用 Qwen3-Omni / Qwen2.5-Omni 的分块预填充，音视频编码器沿时间维输出 chunk，降低 Thinker 和 Talker 的 TTFT。
- **流式生成 ARIA**：Talker 用轻量 MTP 模块预测 RVQ codec token，再经因果流式 ConvNet codec decoder 转波形；ARIA 把双通道生成重构成文本+语音单流交错，降同步开销。

## 与已有沉淀的关系

- 与 [Qwen3-Coder-Next](qwen3-coder-next.md) 同属 Qwen3.5 / Qwen3-Next hybrid 谱系，共享 GDN（+ Qwen3.5 的 gated attention）；区别在模态（全模态 vs 编码）。
- GDN 这一支的机制见 [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)；它把「线性注意力降 KV-cache」的价值延伸到音视频长序列，是该概念页「为什么重要」的一个新场景。
- 多模态架构与已有的 [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)（Kimi K2.5）形成对照——一个走全模态实时交互，一个走视觉 agentic。

## 待追问

- 报告给的是「Hybrid Attention MoE 含 GDN」，但 GDN:attention 的层比例、gated attention 输出门在 Omni 版里是否照用，需读架构节确认（本页不从 Qwen3.5 外推具体比例）。
- 「Qwen3.5 引入 Hybrid MoE」——Qwen3.5 base 报告本身未在 raw/ 中；GDN + gated attention 的完整架构定义可能在 Qwen3.5 base 报告里，Omni 版只是继承。若要 tier-1 坐实 Qwen3.5 架构细节，需补 Qwen3.5 base 报告。
- 总参/激活的确切数字（搜索显示 Qwen3.5 有 397B-A17B 与 27B dense 两档），需从报告配置表确认 Omni 用的是哪档。
