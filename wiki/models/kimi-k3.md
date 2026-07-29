---
type: Model
title: "Kimi K3"
description: "Moonshot AI 的 2.8T / 104B 激活 MoE 模型，首个开源 3T 级模型，KDA + Attention Residuals + Stable LatentMoE + 原生视觉 + 1M 上下文。"
tags: ["model", "kimi-k3", "moe", "hybrid-attention", "multimodal"]
timestamp: 2026-07-30
---

# Kimi K3

## 身份

Kimi K3 是 Moonshot AI（Kimi Team）2026-07-27 发布的开源 MoE 模型，**首个开源 3T 级模型**。原生多模态（文本+图像+视频），1M token 上下文，权重完全开放（HuggingFace: moonshotai/Kimi-K3）。技术报告见 [Kimi K3 技术报告](../sources/kimi-k3.md)。

定位是 frontier-level 但承认仍落后最强闭源（Claude Fable 5、GPT-5.6 Sol）；在开源模型中全面领先，并在 WebDev Arena 成为首个登顶的开源模型。

## 关键事实

| 维度 | 值 | 备注 |
| --- | --- | --- |
| **总参数** | 2.78T（2.8T） | 相对 Kimi K2 (1.04T) +167% |
| **激活参数** | 104.2B（104B） | 相对 Kimi K2 (32.6B) +220% |
| **层数** | 93 | 69 KDA + 24 MLA（3:1 混合） |
| **Hidden dim** | 7,168 | 与 K2 相同 |
| **Latent MoE dim** | 3,584（0.5× hidden） | LatentMoE 解耦 routed expert 宽度 |
| **MoE expert hidden** | 3,072 | |
| **Routed experts** | 896 | 相对 K2 (384) +133% |
| **Active experts/token** | 16 | 相对 K2 (8) +100% |
| **Shared experts** | 2 | 相对 K2 (1) +100% |
| **Attention heads** | 96 | 相对 K2 (64) +50% |
| **Dense layers** | 1 | |
| **Vocabulary** | 160K | |
| **训练上下文** | 1M（progressive 8K→64K→256K→1M） | 相对 K2 (128K) 8× |
| **注意力机制** | Hybrid KDA–MLA | K2 是纯 MLA |
| **激活函数** | SiTU-GLU | K2 是 SwiGLU |
| **MTP 层数** | 1 | 预训练 MTP 层 fine-tune 成 EAGLE-3 draft |
| **ViT** | MoonViT-V2，27 层，~401M 参数 | 从零训练（非 SigLIP init） |
| **模态** | **多模态（文本 + 图像 + 视频）** | 原生多模态，单一共享 backbone，无 post-hoc alignment |
| **量化** | MXFP4 权重 + MXFP8 激活（QAT 贯穿后训练） | 非 expert 组件保持高精度 |
| **权重** | 开源（HuggingFace moonshotai/Kimi-K3） | |

> 模态已据原文 § 2.4 + Abstract 核实："Kimi K3 is natively multimodal: text, images, and videos are processed by a single shared backbone within one context"。视觉是模型自身输入能力（非 RL pipeline verifier 或 eval benchmark 附属）。

## 技术身份

K3 的架构沿三条信息流轴重新设计，每条轴都引入新机制：

**1. 序列长度轴——Hybrid KDA-MLA（3:1）**

69 层 [KDA](../concepts/linear-attention-and-delta-rule.md) + 24 层 Gated MLA。KDA 负责序列内信息混合（带位置/近因感知），MLA 负责无约束全局内容交互。两点升级相对 Kimi Linear：(a) **scaled sigmoid lower-bounded decay**（`g_min=-5`）把 retention factor 下界锁在 e^-5≈6.7e-3，使 reciprocal cumulative decay 不溢出 BF16，**所有因果 tile 用 dense Tensor Core MMA**，消掉 position-pair diagonal 路径；(b) **full-rank output gate**（Kimi Linear 是低秩）。Gated MLA 用 **NoPE**（位置编码全靠 KDA 隐式编码，扩展上下文零修改）+ full-rank output gate + FP32 attention output（纠正 flash attention rounding）。

**2. 网络深度轴——Attention Residuals**

[Attention Residuals](../concepts/attention-residuals.md)（AttnRes）把标准残差（沿深度压成单一状态，类 RNN 瓶颈）换成**每层选择性从所有前层检索表示**（沿深度做 attention）。Block AttnRes（N=8 blocks × 12 层）把内存/通信从 O(Ld) 降到 O(Nd)。详见 [AttnRes 概念页](../concepts/attention-residuals.md)。

**3. 模型宽度轴——Stable LatentMoE**

[Stable LatentMoE](../concepts/stable-latentmoe.md) 把 routed expert 操作解耦到 latent 空间（ℓ=d/2=3584），支持 896 routed / 16 active / 2 shared（sparsity 56）。三组件稳定极端稀疏：(a) Normalized LatentMoE（expert 聚合后 RMSNorm）；(b) **SiTU-GLU**（bounded activation，`|f(x)| ≤ β1·β2=100`，β1=4 β2=25）；(c) **Quantile Balancing**（aux-loss-free routing 的 bias 从 router score 分位数推，balanced assignment 对偶 LP 的 exact coordinate minimizer，无学习率超参）。详见 [Stable LatentMoE 概念页](../concepts/stable-latentmoe.md)。

**4. 原生视觉——MoonViT-V2 from scratch**

MoonViT-V2（27 层 ViT，~401M）**从零训练**（next-token prediction，非 SigLIP init），动机是训练稳定性——SigLIP-init 的 MoonViT-3D 梯度范数高且 spike，from-scratch 全程稳定并追平视觉评测。图像/视频共享参数（intra-frame spatial + inter-frame temporal factorized attention），pixel-shuffle 2×2 砍 4× token。

**5. Per-Head Muon**

Muon optimizer 的注意力投影改进：Newton-Schulz 正交化沿 head 维切分 per-head 做，等化各 head 更新尺度。

## 后训练身份

三阶段：SFT → RL（9 专家 = 3 域 × 3 reasoning effort）→ MOPD 融合。SFT 起即 QAT（MXFP4/MXFP8）。RL 用 partial rollout（λ 比例完成即暂停）+ per-token regularization 容忍 off-policy + reasoning effort budget control + Agentic GRM（tournament + rubric 协议 + verbosity control）。MOPD 9 teacher，per-token OPD reward with R_max clip。Deployment-aware：MTP 层 fine-tune 成 EAGLE-3 draft（融合 1st/4th/final AttnRes block 特征，直接优化 LK loss = acceptance rate 负对数）。

RL 环境是重点工程：Unified White-Box RL Environment（配置化实例化 Kimi Code/Claude Code/Codex/OpenClaw/Hermes 等 harness）、KG-guided task synthesis、kernel/personal-assistant/AET/webdev 任务套件、AgentENV microVM 沙箱（51M+ sandbox 创建）。

## 与同族前作的关系

| 维度 | Kimi K2 | Kimi K2.5 | Kimi Linear | **Kimi K3** |
| --- | --- | --- | --- | --- |
| 总参 | 1.04T | 1.04T | 48B | **2.78T** |
| 激活 | 32.6B | 32B | 3B | **104.2B** |
| 注意力 | MLA | MLA | 3:1 KDA:MLA | **3:1 KDA:Gated MLA** |
| 视觉 | – | MoonViT-3D（SigLIP init） | – | **MoonViT-V2（from scratch）** |
| 上下文 | 128K | 256K（评测常用） | 1M | **1M** |
| 定位 | agentic MoE 基座 | 视觉 agentic + Agent Swarm | 线性注意力研究模型 | **首个开源 3T frontier** |

K3 继承 Kimi Linear 的 3:1 KDA:MLA 混合并升级（scaled sigmoid + full-rank gate + NoPE MLA），把 Kimi K2.5 的 MoE backbone 放大到 2.8T 并加 LatentMoE + AttnRes + Stable 三组件，视觉从 SigLIP-init 转向 from-scratch。

## 相关页面

- 来源：[Kimi K3 技术报告](../sources/kimi-k3.md)、[Kimi K2.5 技术报告](../sources/kimi-k2.5.md)、[Kimi Linear 技术报告](../sources/kimi-linear.md)（KDA 首次提出）
- 概念：[Attention Residuals](../concepts/attention-residuals.md)、[Stable LatentMoE](../concepts/stable-latentmoe.md)、[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)、[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)、[注意力门控](../concepts/attention-gating.md)、[MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)、[Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)、[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[多 token 预测](../concepts/multi-token-prediction.md)、[异步 Agent RL](../concepts/asynchronous-agent-rl.md)
- 比较：[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)、[On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
