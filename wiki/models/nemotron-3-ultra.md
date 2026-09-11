---
type: Model
title: "Nemotron 3 Ultra"
description: "NVIDIA 的 550B / 55B 激活 hybrid Mamba-Attention MoE，NVFP4 预训练 20T、1M 上下文，后训练是统一 RLVR + 两轮 MOPD。"
tags: ["model", "nemotron-3-ultra", "moe", "mamba", "nvfp4"]
timestamp: 2026-09-12
---

# Nemotron 3 Ultra

## 身份

Nemotron 3 Ultra 是 NVIDIA 2026-06 发布的开放权重 MoE，Nemotron 3 家族里最大、也最偏 agentic 的一档。定位是 **吞吐–准确率前沿**：准确率与 GLM-5.1 / Kimi-K2.6 / Qwen-3.5 大致持平，8K/64K decode-heavy 设置下 NVFP4 相对 GLM-5.1 报 5.9× 吞吐。技术报告见 [Nemotron 3 Ultra 技术报告](../sources/nemotron-3-ultra.md)。

## 关键事实

| 维度 | 值 | 备注 |
| --- | --- | --- |
| **总参数** | 550B | Nemotron 3 Super 同架构放大 |
| **激活参数** | 55B | 高于 GLM-5 的 40B、低于 Kimi K3 的 104B |
| **层数** | 108 | hybrid Mamba-2 + GQA Attention |
| **Hidden dim** | 8192 | |
| **Q / KV heads** | 64 / 2 | GQA，不是 MLA |
| **Routed experts** | 512 | Top-k = 22 |
| **Expert hidden / shared intermediate** | 5120 / 10240 | |
| **MoE latent size** | 2048 | LatentMoE（Elango et al. 2026） |
| **MTP** | 2 head，共享权重 | 预训练 + SFT 保留；后训练做 head-only Boosting |
| **预训练** | 20T 文本 token，NVFP4，WSD | 15T 多样性 + 5T 质量；LC-Phase 再 33B |
| **上下文** | 1M | CPT 92% 迭代走 1,048,576，8% 走 4K |
| **注意力** | Hybrid Mamba-2 / GQA | 多数层固定状态，周期全局 Attention |
| **后训练** | SFT → RLVR → MOPD warmup → 两轮 MOPD → MTP Boosting | 保留 RL，MOPD 做融合而非替换 |
| **模态** | **纯文本** | 已据 Abstract + `§2`「20 trillion text tokens」核实；无视觉编码器 |
| **量化** | NVFP4 预训练 + 一份 PTQ checkpoint（W4A16 / W4A4） | Figure 1 里 NVFP4 柱贴近 BF16 |
| **权重** | 开源 Base / Post-trained BF16 / NVFP4 / GenRM | 配方 [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) |

> 模态已据原文 Abstract 与 `§2` 核实：输入是文本，预训练语料是文本 token。评测里的 search / BrowseComp / 办公文件是 agent 工具环境，不是模型本体的多模态输入。

## 技术身份

四件事把它和同档开源 MoE 分开：

1. **Hybrid Mamba-2 而不是 SWA/线性注意力/稀疏 softmax。** 长上下文成本主要靠 SSM 固定状态压 KV，Attention 层仍是 64/2 GQA。和 [Kimi Linear](kimi-linear.md) 的 KDA、[Ling-2.6](ling-2.6.md) 的 Lightning Attention 同属「大多数层不用 softmax KV」这一支，但是 Mamba-2 而不是 delta-rule 线性层。见 [高效长上下文注意力](../concepts/efficient-long-context-attention.md)。
2. **LatentMoE 生产集成，但不是 K3 那套稳定化。** routed 在 latent 2048，512/22；没有公开 Quantile Balancing 或 SiTU-GLU。预训练后期第一层 MaxVio 升到约 12，第二次 loss 发散后把 20T 当权宜上限。
3. **NVFP4 从预训练贯穿到一份部署 checkpoint。** 作者称为当时最大规模稳定 NVFP4 训练；BF16 对照消融说明第二次发散不是低精度本身。
4. **后训练保留 RLVR，再用两轮 MOPD 融合 >10 个域教师。** 这与 [DeepSeek-V4](deepseek-v4.md)「OPD 替换 mixed RL」相反，与 [MiMo-V2-Flash](mimo-v2-flash.md) 同属融合派，但 Ultra 是目前唯一公开跑完 **teacher–student 两轮 co-evolution** 并给出按域恢复率的报告。详见 [MOPD](../concepts/multi-teacher-on-policy-distillation.md)。

## 相关页面

- [Nemotron 3 Ultra 技术报告](../sources/nemotron-3-ultra.md)
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Stable LatentMoE](../concepts/stable-latentmoe.md)
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- [多 token 预测](../concepts/multi-token-prediction.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- [2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
