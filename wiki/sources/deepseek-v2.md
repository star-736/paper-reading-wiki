---
type: Source
title: "DeepSeek-V2 技术报告"
description: "DeepSeek-V2 的 arXiv 论文，Multi-Head Latent Attention（MLA）的首次提出处。"
tags: ["source", "deepseek-v2"]
timestamp: 2026-06-20
resource: "../../raw/DeepSeek-AI%20%E7%AD%89%20-%202024%20-%20DeepSeek-V2%20A%20strong%2C%20economical%2C%20and%20efficient%20mixture-of-experts%20language%20model.pdf"
---

# DeepSeek-V2 技术报告

## 来源

- 原始 PDF：[raw/DeepSeek-AI 等 - 2024 - DeepSeek-V2 A strong, economical, and efficient mixture-of-experts language model.pdf](../../raw/DeepSeek-AI%20%E7%AD%89%20-%202024%20-%20DeepSeek-V2%20A%20strong%2C%20economical%2C%20and%20efficient%20mixture-of-experts%20language%20model.pdf)
- 标题：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- 版本/日期：arXiv:2405.04434，2024-05
- 团队：DeepSeek-AI
- 关键身份：**MLA（Multi-Head Latent Attention）的首次提出处**，也是 DeepSeekMoE 的奠基论文之一。下一份报告 [DeepSeek-V3](deepseek-v3.md) 沿用这里的五个 MLA 宽度。后续 [DeepSeek-V3.2](deepseek-v32.md) 的 DSA、[DeepSeek-V4](deepseek-v4.md) 的 CSA 都架在 MLA 之上。
- 概念页：[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)

## 核心结论

DeepSeek-V2 是 236B 总参 / 21B 激活的 MoE 模型，128K 上下文，预训练 8.1T tokens。它的两个架构主张：

- **MLA**：用低秩 KV 联合压缩减小缓存。附录 D.2 的两组 MoE 对照中，MLA 在八个分数中七个更高，小模型 C-Eval 低 0.7 分；具体数值与边界见下文 Table 9。
- **DeepSeekMoE**：细粒度专家 + 共享专家。报告摘要的训练成本 −42.5%、KV cache −93.3%、最大生成吞吐 5.76×，均是 **DeepSeek-V2 整体相对 DeepSeek 67B** 的数字，不是逐项隔离组件的消融。

附录 D.1 的 7B dense、1.33T-token 对照中，MHA 在四项 hard benchmark 上均高于 GQA-8 / MQA；这限定于该实验，而非任意配置的质量定律。MLA 改用低秩压缩，探索另一种质量与缓存的折中。

## 架构与训练

### MLA：低秩 KV 联合压缩（§2.1.2）

- 把输入 hidden $h_t$ 下投影成一个**压缩 latent 向量** $c^{KV}_t$（维度 $d_c$，远小于 $n_h d_h$），不缓存展开的 per-head K/V；完整配置还缓存下节的 decoupled RoPE key。用时再通过上投影 $W^{UK}$、$W^{UV}$ 恢复出 per-head 的 K、V。
- **矩阵吸收**：推理时 $W^{UK}$ 可吸进 query 投影 $W^Q$、$W^{UV}$ 可吸进输出投影 $W^O$，于是不必显式还原 per-head K/V，直接对 latent 算注意力——这正是 [V3.2 所说的「MQA mode」](deepseek-v32.md)的来源。
- query 侧也做低秩压缩（$d_c'=1536$）。原文 §2.1.2 明确给出的目的，是减少训练激活显存；query 本来不属于自回归 KV cache，所以这一步不进一步减少 KV cache。V2-Lite 不压缩 query 是附录 B.1 的配置事实，不能据此反推 query 压缩对能力没有影响。具体 checkpoint / 重算策略未披露，见 [MLA 概念页待追问](../concepts/multi-head-latent-attention.md#待追问)。

### Decoupled RoPE（§2.1.3）

RoPE 与低秩 KV 压缩**不兼容**：若直接对 K 施加 RoPE，位置敏感的 RoPE 矩阵会卡在 $W^{UK}$ 中间，导致它**无法再被吸收**进 $W^Q$。解法是 **decoupled RoPE**——额外引入一组 multi-head **decoupled query** $q^R$ 和一个**所有 head 共享的 decoupled key** $k^R$ 专门承载 RoPE；最终的 K/Q 是「压缩部分（可吸收，不带 RoPE）+ decoupled 部分（带 RoPE）」拼接。decoupled key 也要进 cache，故每 token 总 KV cache = $(d_c + d_R)$ 个元素。

完整投影形状与数值见 [MLA 的投影矩阵与缓存形状](../concepts/multi-head-latent-attention.md#投影矩阵与缓存形状)。

### KV cache 对比（Table 1）

| 注意力 | 每层每 token KV cache（元素数）| 原表的能力概括 |
| --- | --- | --- |
| MHA | $2 n_h d_h$ | 强 |
| GQA（$n_g$ 组）| $2 n_g d_h$ | 中 |
| MQA | $2 d_h$ | 弱 |
| **MLA** | $d_c + d_R \approx \tfrac{9}{2} d_h$ | **强（> MHA）** |

报告 Table 1 将 MLA 缓存写成等效于 2.25 组 GQA；这是按相同每头维度的元素数换算，不是实际采用分数个组。V2 每层每 token 缓存 512+64=576 个元素，**并不小于同为 128 维的 MQA（256 个元素）**。质量概括须结合下面 Table 9 的实验范围。

### 训练

8.1T tokens 预训练；pipeline parallelism 部署不同层；DeepSeekMoE 用 2 shared + 160 routed expert 配置。128K 上下文经 Needle-in-a-Haystack 压力测试。

### 负载均衡（§2.2.2–2.2.4，原文确证）

V2 仍是 auxiliary-loss 配置，不是 V3 起的 auxiliary-loss-free。路由先做 **device-limited routing**：每个 token 的目标专家落在至多 \(M\) 台设备上；\(M\ge 3\) 时作者称与不受限 top-K 大致对齐。训练超参取 \(M=3\)，routed expert 均分到 \(D=8\) 台设备。

平衡由三项辅助损失相加，都是「频率 × 亲和」那种点积：

| 损失 | 管什么 | 训练系数 |
| --- | --- | --- |
| \(L_{\text{ExpBal}}\) | expert 级，防 routing collapse | \(\alpha_1=0.003\) |
| \(L_{\text{DevBal}}\) | 设备上的专家组负载 | \(\alpha_2=0.05\) |
| \(L_{\text{CommBal}}\) | 每台设备收到的 token 量；发送量已由 device-limited routing 封顶 | \(\alpha_3=0.02\) |

损失不能保证严格平衡，所以训练时还有 **device-level token-dropping**：每台设备的 capacity factor 为 1.0，丢掉该设备上亲和最低的 token，直到不超预算；约 10% 的训练序列从不丢 token。评测不丢 token。

[DeepSeek-V3](deepseek-v3.md)（[`raw/2412.19437v2.pdf`](../../raw/2412.19437v2.pdf) §2.1.2）把相对 V2 的架构例外写成 auxiliary-loss-free：bias 只参与 top-K 选择，不进门控值；另留一个系数极小的 sequence-wise balance loss，防止单条序列内极端失衡。谱系对照见 [MoE 负载均衡谱系](../concepts/moe-load-balancing.md)。

## 后训练

SFT + GRPO 强化学习（DeepSeek 自家 RL 算法），对齐到对话与人类偏好。本页聚焦 MLA 架构，后训练细节从略——如需可单独 deepen。

## 评测要点

- 仅 21B 激活参数即在开源模型中达到 top-tier；KV cache −93.3%、吞吐 5.76× 是其「economical / efficient」定位的核心数字。
- 附录 D.1（MHA vs GQA vs MQA 消融）是理解 MLA 动机的关键：它用实验说明「减头数」路线（GQA/MQA）有质量代价，从而论证 MLA「压低秩」这条正交轴的必要性。

### MLA 与 MHA 的直接对照（附录 D.2 / Table 9）

**已据原文核实（`supported`）**：小 MoE 对各训 1.33T tokens，大 MoE 对各训 420B tokens；原文称各组除注意力外架构相同。两组训练预算不同，不能用它们直接画跨规模收益曲线。表内总参数、激活参数也并非严格相等。

| 指标 | 小 MoE / MHA | 小 MoE / MLA | 大 MoE / MHA | 大 MoE / MLA |
| --- | ---: | ---: | ---: | ---: |
| 激活参数 | 2.5B | 2.4B | 25.0B | 21.5B |
| 总参数 | 15.8B | 15.7B | 250.8B | 247.4B |
| 全模型每 token KV cache（元素） | 110.6K | 15.6K | 860.2K | 34.6K |
| BBH（EM，3-shot） | 37.9 | 39.0 | 46.6 | 50.7 |
| MMLU（Acc.，5-shot） | 48.7 | 50.0 | 57.5 | 59.0 |
| C-Eval（Acc.，5-shot） | 51.6 | 50.9 | 57.9 | 59.2 |
| CMMLU（Acc.，5-shot） | 52.3 | 53.4 | 60.7 | 62.5 |

**本页计算**：按 BBH / MMLU / C-Eval / CMMLU 顺序，小模型差值为 +1.1 / +1.3 / −0.7 / +1.1 个百分点，大模型为 +4.1 / +1.5 / +1.3 / +1.8 个百分点。小模型缓存约为 MHA 的 14.1%，大模型约为 4.0%（原文概括为 14% / 4%）。

因此，“MLA 在该实验总体表现更好且缓存显著更小”有原文支撑；“MLA 在所有规模、所有任务都超过 MHA”不成立（`refuted`，小模型 C-Eval 已是反例）。Table 9 未给多 seed 方差或置信区间，不能把每个正差值都称作统计显著优势。其 247.4B / 21.5B 的大模型实验列，也不能直接当作发布版 236B / 21B 的最终 8.1T 训练结果。摘要的 −93.3% 与 5.76× 是相对 DeepSeek 67B 的另一组比较，不能混用分母。

## 相关追问

主记录：[query 压缩如何省激活显存](../concepts/multi-head-latent-attention.md#待追问)。演进表已写入 [从 V2 到 V4](../concepts/multi-head-latent-attention.md#从-v2-到-v4)。

## 相关页面

- 概念：[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)、[DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)、[高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- 后续来源：[DeepSeek-V3](deepseek-v3.md)、[DeepSeek-V3.2](deepseek-v32.md)、[DeepSeek-V4](deepseek-v4.md)
- 比较：[稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)

对应概念与矩阵形状：[Multi-Head Latent Attention (MLA)](../concepts/multi-head-latent-attention.md#投影矩阵与缓存形状)。
