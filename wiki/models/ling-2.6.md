---
type: Model
title: "Ling-2.6 / Ring-2.6"
description: "Inclusion AI 万亿参数 agentic 模型族：7:1 Lightning Attention + MLA 混合线性注意力 retrofit，Ling-2.6 instant + Ring-2.6 thinking 双线，KPop agentic RL。"
tags: ["model", "ling-2.6", "ring-2.6", "hybrid-linear-attention", "moe", "agentic"]
timestamp: 2026-07-11
---

# Ling-2.6 / Ring-2.6

## 身份

Inclusion AI（蚂蚁集团 Ling Team）的万亿参数 agentic 模型族，2026-06 发布。从 Ling-2.0 base checkpoint 经架构 retrofit 升级而来，不从头训练。家族分两条线：**Ling-2.6**（instant model，快速响应 + 高 token 效率）和 **Ring-2.6**（thinking model，深度推理 + long-horizon agentic）。开源全部 checkpoint。

## 关键事实

| 属性 | Ling-2.6-flash | Ling-2.6-1T | Ring-2.6-1T |
| --- | --- | --- | --- |
| 总参数 | ~104B | ~1T | ~1T |
| 激活参数 | ~5B | ~8B（8/256 experts） | ~8B |
| **模态** | 纯文本 | 纯文本 | 纯文本 |
| 注意力架构 | 7:1 Lightning Attention + MLA | 同左 | 同左 |
| MoE 设计 | 256 routed + 1 shared，8 active | 同左 | 同左 |
| 上下文 | 256K | 256K | 256K（训练 200 turns / 评估 500 turns） |
| 预训练 | 从 Ling-2.0 retrofit，~9.6T tokens | 同左 | 同左 |
| 后训练重点 | token efficiency（Evo-CoT + LPO + 双向偏好对齐） | 同左 | agentic RL（KPop + 异步 RL） |
| 定位 | 部署效率优先 | 旗舰 instant | 旗舰 thinking |

**模态**：纯文本。报告未提及视觉/音频输入能力。结论部分明确将"原生多模态 agent"列为未来方向，当前 2.6 家族为纯文本模型。

## 技术身份

### 架构 retrofit：从 GQA 到 hybrid linear attention

Ling-2.6 的核心架构叙事是**不改从头训练万亿模型**：Ling-2.0-1T 已投入 20T tokens 训练，从头换架构会浪费这笔投资。解法是 smooth multi-stage migration pipeline——Lightning Attention 转换（扩展 $W_{qkv}$ + 引入 gating）-> Linear Warmup -> MLA 转换（QK Norm absorption + Partial RoPE adaptation + TransMLA）-> MLA Warmup，在 ~400B tokens 内完成无损转换。

7:1 混合比例经 scaling law 实验确定（M=2/4/8/16 对比，M=8 最优），比 Kimi Linear / Qwen3-Next 的 3:1 更激进。这与 Lightning Attention 的具体设计有关——它不同于 KDA/GDN 的 delta rule 路线，而是基于 FLA（Flash Linear Attention）kernel 的线性注意力，gating 机制不同。

### 双线后训练：instant vs thinking

Ling-2.6 和 Ring-2.6 共享 base model，后训练分叉：
- **Ling-2.6**：specialization-then-distillation，reasoning specialist（Evo-CoT + LPO + redundancy penalty）和 agentic specialist（GSPO + compression penalty）并行训练后蒸馏回统一模型，最后 bidirectional preference alignment。核心目标是 **capability per output token**——4× token efficiency。
- **Ring-2.6**：在 Ling-2.6 基础上增强 long-horizon agentic。KPop RL（binary KL 替代 IcePop fixed-ratio）+ 异步 RL（partial-rollout pipeline + ARouter + staleness manager）。adaptive thinking 分 high/xhigh 两档推理预算。

### 系统协同设计

报告强调 architecture + optimization + serving + agent training 四层 co-design。基础设施亮点：AllGather-based CP for Lightning Attention（无 head-divisibility 约束）、linghe fused-kernel library（FP8 BS=1 +119%）、ASystem 异步 RL 框架（ARouter spillover overlap >80% 提升）、ASandbox 多环境 agentic rollout（197 MCP servers / 2,400+ tools）。

## 相关页面

- 来源：[Ling and Ring 2.6 技术报告](../sources/ling-2.6.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)（Lightning Attention 对照路线）、[高效长上下文注意力](../concepts/efficient-long-context-attention.md)、[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)、[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[异步 Agent RL](../concepts/asynchronous-agent-rl.md)、[多 Token 预测](../concepts/multi-token-prediction.md)
- 比较：[2026 前沿模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
