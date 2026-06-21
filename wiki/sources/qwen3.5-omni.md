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

- ~~GDN:attention 层比例~~ **已据 Qwen3.5 HF config 坐实**：`layer_types` 逐层列出 3 个 `linear_attention` + 1 个 `full_attention`（`full_attention_interval=4`），即 3 GDN : 1 gated-attention，多模态（config 含 `vision_config`/`video_token_id`）。见模型页 [Qwen3.5](../models/qwen3.5.md)。
- **不存在「Qwen3.5 base 报告」可补**：截至录入，Qwen3.5 系列只发了本篇 Omni 报告（arXiv:2604.15804）+ HF 权重，没有独立的 base 技术报告。所以架构事实的 tier-1 来源就是 **HF config（权重本身）**，GDN + gated attention 的文字级设计动机散在 [Gated Attention](gated-attention.md) / [GDN](gated-delta-net.md) 两篇原论文和 [Qwen3-Next 官方博客](qwen3-next-blog.md) 里——不要再去找一篇并不存在的 base 报告。
- Qwen3.5-Omni 用的是哪个尺寸档（397B-A17B / 122B-A10B / 35B-A3B / 27B dense）——报告未明说，HF 上 Omni 也未作为独立仓库公开，需 Omni 自身配置才能定档。

## 相关页面

- 模型：[Qwen3.5](../models/qwen3.5.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)、[注意力门控](../concepts/attention-gating.md)、[多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)
- 同家族来源：[Qwen3-Coder-Next](qwen3-coder-next.md)、[Gated Attention](gated-attention.md)、[Gated DeltaNet](gated-delta-net.md)
