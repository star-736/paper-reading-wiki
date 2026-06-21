# Qwen3.5（家族）

## 身份

Qwen3.5 是 Qwen 团队（Alibaba）的一个多变体模型家族，统一采用 **Hybrid Attention MoE** 架构——大多数层用 [GDN 线性注意力](../concepts/linear-attention-and-delta-rule.md)、每隔 3 层插 1 层带输出门的 [gated full attention](../concepts/attention-gating.md)，并原生支持多模态（含 vision/video token）。[Qwen3.5-Omni 技术报告](../sources/qwen3.5-omni.md) 明说其 Thinker/Talker 都构建在 Qwen3.5 的这套 Hybrid MoE 上，因此该家族是「Gated Attention + GDN」在大尺度多模态生产模型上的落地。

> 本页是 Qwen3.5 **家族页**，架构事实以各变体的 HuggingFace `config.json` 为准。Qwen3.5-Omni 本身未作为独立 HF 仓库公开（report 已发），其架构基座即下表的 Qwen3.5 hybrid MoE。

## 关键事实（变体表）

架构字段以 HuggingFace 官方 config 为准（已据 config 核实）。多变体家族，模态写在表下：

| 变体 | 总参 / 激活 | 层数 | hidden | 备注 |
| --- | --- | --- | --- | --- |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | 403B / ~17B（512 expert 选 10） | 60 | 4096 | 旗舰 MoE，`Qwen3_5MoeForConditionalGeneration` |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 122B / ~10B | — | — | 中档 MoE |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | 35B / ~3B | — | — | 小档 MoE |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 27B（dense） | 64 | 5120 | dense 版，`Qwen3_5ForConditionalGeneration`，16 组 ×(3 GDN + 1 Attn) |
| 另有 0.8B / 2B / 4B / 9B dense 与 Base 变体 | — | — | — | — |

**模态**：多模态（文本 + 图像 + 视频）——config 含 `vision_config`、`image_token_id`、`video_token_id`（已据 397B / 27B config 核实）。Qwen3.5-Omni 在此基座上进一步做全模态（音频 + 音视频）实时交互。

**混合结构（已据 config 核实，tier-1）**：397B 与 27B 的 `config.json` 里 `layer_types` 字段**逐层列出** `['linear_attention','linear_attention','linear_attention','full_attention', …]`，`full_attention_interval=4`——即字面坐实 **3 线性层 : 1 全注意力层** 的 3:1 混合，全注意力层是带输出门的 gated attention，线性层是 GDN。线性层 16 key head / 48–64 value head，head_dim 128；全注意力 head_dim 256，上下文 262,144。

## 技术身份

Qwen3.5 把 Qwen3-Next 引入的「3 GDN : 1 gated-attention」混合栈推到旗舰尺度（397B-A17B）并加上原生多模态。这条「GDN 线性层降 KV-cache I/O + 少量全局 gated attention 维持全局信息流 + 去 attention sink」的设计，正是 [Gated Attention 论文](../sources/gated-attention.md)（NeurIPS 2025 Best Paper）与 [GDN 论文](../sources/gated-delta-net.md) 在生产模型上的合体；其设计动机由 [Qwen3-Next 官方博客](../sources/qwen3-next-blog.md) 讲明（3:1 = 75% GDN / 25% standard 的官方原话、选 GDN 因 in-context learning 强于 SWA/Mamba2、全局层加 output gating 去 sink）。Qwen3.5-Omni 把它延伸到 >10 小时音频、400 秒 720P 视频的长序列场景，靠 GDN 显著降低长上下文 KV-cache I/O。

与 [Kimi Linear](kimi-linear.md) 的对照同样成立：同为 3:1 混合，Qwen3.5 全局层用 gated full attention，Kimi Linear 用 MLA。

## 相关页面

- 来源：[Qwen3.5-Omni 技术报告](../sources/qwen3.5-omni.md)、[Gated Attention 技术报告](../sources/gated-attention.md)、[Gated DeltaNet 报告](../sources/gated-delta-net.md)、[Qwen3-Next 官方博客](../sources/qwen3-next-blog.md)
- 概念：[注意力门控](../concepts/attention-gating.md)、[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)
- 同家族：[Qwen3-Coder-Next](qwen3-coder-next.md)（Qwen3-Next 的编码变体）
- 同思路对照：[Kimi Linear](kimi-linear.md)（全局层用 MLA）
