---
type: Concept
title: "MoE 负载均衡谱系"
description: "从 auxiliary loss 到 Loss-Free bias 到 Quantile Balancing：MoE 路由负载均衡的方法谱系与生产采用地图（本 wiki 已收录模型的实际配置）。"
tags: ["concept", "moe", "load-balancing", "loss-free-balancing", "routing"]
timestamp: 2026-08-23
---

# MoE 负载均衡谱系

## 定义

MoE 的 top-K 路由若不加控制，会出现 **routing collapse**（少数专家被反复选中、其余训练不足）或 expert-parallel 下的**计算瓶颈**。负载均衡方法解决"哪些 token 去哪个 expert"的分布控制问题，谱系沿一条主线演进：**怎么在不伤害语言建模梯度的前提下把负载摊平**。

三个世代（按对主梯度的干预方式划分）：

1. **Auxiliary loss（梯度内干预）**：在训练目标上加平衡项 $L_{\text{Balance}} = \alpha \sum_i f_i P_i$（Switch/GShard）。$\alpha$ 在"失衡"与"干扰梯度"间两难——[Loss-Free Balancing](../sources/loss-free-balancing.md) Figure 2 的核心实证。
2. **Auxiliary-loss-free bias（梯度外干预）**：top-K 前给 routing score 加 expert-wise bias，训练步之间按历史负载更新 bias，bias 不进 mixture weights、不产生任何额外梯度（[Loss-Free Balancing](../sources/loss-free-balancing.md)，DeepSeek-AI + PKU，arXiv:2408.15664）。
3. **Exact 解（K3 QB）**：把"调 bias"从固定步长 sign 更新升级为 balanced assignment 对偶 LP 的 exact coordinate minimizer——从 router score 分位数直接推 bias，无学习率、几步收敛（[Kimi K3](../sources/kimi-k3.md) / [Stable LatentMoE](stable-latentmoe.md) § 2.3.3）。

**被淘汰的第四条路：Expert Choice（EC）**。专家挑 token 达到完美均衡且零梯度，但违反因果约束——未来 token 的分数影响当前 token 的分配，每层泄漏 $K\log_2\frac{1-R}{R}$ bits/token。Loss-Free Balancing 论文的泄漏证明 + 实验（chunk 缩小 → 异常 loss drop；shuffle → 消失）是社区放弃 EC 用于自回归 LM 的标准论据。

## 跨报告信号：生产配置地图

各模型实际用的负载均衡（均据对应 raw/ PDF 核实）：

| 模型 | 阵营 | 具体配置 | 出处 |
|---|---|---|---|
| DeepSeek-V2 | aux loss（旧世） | 三重辅助损失（expert 级 $L_{\text{ExpBal}}$ + device 级 + …）+ device-limited routing + token dropping | [V2 报告](../sources/deepseek-v2.md) §2.2.3 |
| **DeepSeek-V3.2 / V4** | **bias（标准配置）** | auxiliary-loss-free strategy，bias update speed 0.001；V4 叠加 weight 1e-4 sequence-wise balance loss | [V4 报告](../sources/deepseek-v4.md) §2 + §训练设置 |
| Kimi K2 / K2.5 | bias | 常规 aux-loss-free bias（384 routed / 8 active） | [K2.5 报告](../sources/kimi-k2.5.md) |
| Kimi K3 | **QB（exact 解）** | Quantile Balancing，分位数推 bias，无学习率；896 routed / 16 active | [K3 报告](../sources/kimi-k3.md) §2.3.3 |
| MiniMax-M2 | bias（变体） | sigmoid gating + learnable expert-specific bias，"greatly reducing reliance on auxiliary losses"，显式引 Wang et al. 2024a | [M2 报告](../sources/minimax-m2-series.md) §2 |
| MiMo-V2-Flash | 混合 | expert bias update factor 0.001 + MoE sequence aux loss 1e-5 | [MiMo 报告](../sources/mimo-v2-flash.md) |
| Ling-2.6 / Ring-2.6 | bias | aux-loss-free，bias-update rate γ=0.001→0.0001（后期衰减） | [Ling-2.6 报告](../sources/ling-2.6.md) |
| Qwen3 | aux loss（演进） | global-batch load balancing loss（Qiu et al., 2025），非 micro-batch 口径 | [Qwen3 报告](../sources/qwen3.md) |
| Laguna XS.2 | aux loss | Qiu et al. 2025 aux loss（只在非 padding token 上算） | [Laguna 报告](../sources/laguna-m1-xs2.md) |
| GLM-5 | 未披露 | 报告只提 256 experts / 80 层为减 EP 通信开销，未写负载均衡方法 | [GLM-5 报告](../sources/glm-5.md) |

读法：**bias 路线是 DeepSeek 系及 Kimi 系的家族传统**（同源于本论文，作者重叠），aux loss 路线在 Qwen 系仍有强生命力（global-batch 口径是它对"鼓励专家专业化"的回答），两家并未收敛到单一答案。MiMo/V4 的混合配置（bias + 轻序列级 loss）暗示纯 bias 在**单序列粒度**上有盲区——原论文只测了 global/batch 口径。[Engram](../sources/engram.md) 的 27B/40B 研究模型同样写 Loss-Free（Appendix A），不是生产部署，只说明 DeepSeek 方法论文继续沿用这条配方。

## 方法对比三轴

| 维度 | aux loss | loss-free bias | QB（K3） | EC |
|---|---|---|---|---|
| 干扰梯度 | 有（α 控制） | 无 | 无 | 无 |
| 因果安全 | 是 | 是（用历史 batch 负载） | 是（下一 step 生效） | **否（未来 token 泄漏）** |
| 超参 | α（两难调） | 更新率 u（1B 上调一次即可） | 无学习率超参 | 容量参数 |
| 大 expert pool | 可用 | ~10²-10³ experts 下 u 两难（慢 vs 振荡） | 几步收敛到 exact 解 | 可用但泄漏更严重 |
| 生产采用 | Qwen3 / Laguna / V2 | V3/V4 / K2 系 / M2 / MiMo / Ling | K3 | 无（自回归 LM） |

## 为什么重要

- **它是 MoE 训练配方里少有的"已收敛"问题**：2017-2024 间的开放争论（loss vs bias vs EC）在 2024 后基本收口——aux-loss-free bias 成 DeepSeek/Kimi/MiniMax/MiMo/Ling 共识，EC 因泄漏出局，aux loss 退守 Qwen 系。检索本 wiki 的 MoE 相关问题时，负载均衡方法不再是有悬念的设计维度。
- **"梯度外控制"这个思路本身可迁移**：bias 不进 loss、不进梯度、只在推理图里改路由——把"优化目标"和"系统约束"解耦的工程模式。K3 QB 沿同一接口（bias）升级求解器而不动训练梯度，是这个解耦的可扩展性证明。
- **给"生产报告引用链"提供了校准案例**：V4 报告引用的是 "Wang et al., 2024a"（本论文）+ "DeepSeek-AI, 2024"（V3 报告）双引——引用链上每个环节的原文都在 wiki 的 raw/ 里，可以逐环核实而不是靠二手转述。

## 待追问

- **GLM-5 / GLM-5V-Turbo 的负载均衡策略未披露**（见来源页待追问 1）。
- **MiniMax-M2 的 learnable bias 语义**：bias 若进梯度就不再严格 loss-free，需回 M2 §2.2.1 确认（见来源页待追问 2）。
- **QB 会不会成为下一个标准？** K3 之后尚未见其他家族跟进；QB 的对偶 LP 视角是否只在 896-expert 级极端稀疏下才有必要（256-384 expert 下 sign 更新够用），还是也会下沉到中等规模？
- **sequence-wise balance loss 的机理**：V4（1e-4）与 MiMo（1e-5）都在 bias 之外补序列级 loss，但权重差一个量级，且都没有公开消融——单序列失衡到底伤什么？

## 相关页面

- 一手出处：[Loss-Free Balancing](../sources/loss-free-balancing.md)（arXiv:2408.15664）
- QB 升级：[Stable LatentMoE](stable-latentmoe.md)、[Kimi K3](../sources/kimi-k3.md)
- aux loss 阵营：[Qwen3](../sources/qwen3.md)、[Laguna M.1/XS.2](../sources/laguna-m1-xs2.md)、[DeepSeek-V2](../sources/deepseek-v2.md)（旧世三重 loss）
- bias 阵营采用：[DeepSeek-V4](../sources/deepseek-v4.md)、[Kimi K2.5](../sources/kimi-k2.5.md)、[MiniMax-M2 Series](../sources/minimax-m2-series.md)、[MiMo-V2-Flash](../sources/mimo-v2-flash.md)、[Ling and Ring 2.6](../sources/ling-2.6.md)
- 上位概念：[MoE 前沿模型扩展](moe-frontier-model-scaling.md)
- 研究模型沿用：[Engram](../sources/engram.md)
