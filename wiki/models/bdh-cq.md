---
title: "BDH-CQ"
type: Model
description: "Pathway 的 150M ARC 推理系统：示例递归写入 memory，查询以连续 latent workspace 迭代求解，不输出中间 CoT"
tags: [BDH, recurrent-latent-reasoning, ARC-AGI, in-context-learning]
timestamp: 2026-08-23
---

## 身份

BDH-CQ 是 Pathway 主导发布的 BDH-based ARC 推理系统。它以少量 ARC demonstration pairs 在 inference 时构造任务相关的 recurrent memory，并在连续 latent workspace 中反复计算答案；论文不公开 intermediate reasoning token，也不进行 inference-time parameter update（原文确证，来源报告 §3）。

## 关键事实

| 属性 | 值 | 来源 |
| --- | --- | --- |
| **架构家族** | BDH（post-Transformer sequence-model family） | 已据原文核实 |
| **模型参数** | 150M（文中评测配置） | 已据原文核实 |
| **上下文适配** | demonstrations 顺序更新 recurrent memory $S_t$ | 已据原文核实 |
| **推理** | 查询在 structured latent workspace $H_r$ 中迭代；仅解码最终网格 | 已据原文核实 |
| **模态** | ARC 彩色二维符号网格输入 / 输出；未报告文本或自然图像能力 | 已据原文核实 |
| **inference 参数更新** | 无 | 已据原文核实 |
| **ARC-AGI-1 public** | 29.5% pass@2（118/400 tasks） | 已据原文核实 |
| **成本口径** | 约 0.85 H200 GPU-seconds/task；按 $3/H200-hour 计 $0.00070/task | 已据原文核实 |
| **实现公开度** | dimensions、完整 update rule、训练配方和实现 proprietary | 已据原文核实 |

**技术身份**：BDH-CQ 的公开接口包含两个状态：$S_t$ 累积 demonstrations 所表达的 task-specific association，$H_r$ 在固定的 $S_K$ 条件下进行查询计算。它是 recurrent latent reasoning 的实例，不是已公开为 weight-tied Transformer、PLT 或 linear-attention model 的实例；这些机制之间只能作概念类比，不能相互替代。

## 相关页面

- [BDH-CQ 来源页](../sources/bdh-cq.md) — 架构接口、评测和证据边界
- [Looped Transformers](../concepts/looped-transformers.md) — latent recurrence 的相邻路线与区分
