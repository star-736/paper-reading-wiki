---
type: Model
title: "Mach-Mind-4-Flash"
description: "理想汽车的 35B MoE agentic 模型（3B 激活），基于 Qwen3.5-35B-A3B，specialization-then-integration 后训练 + MOPD 融合 + HMPO token 效率。"
tags: ["model", "agentic", "moe", "post-training"]
timestamp: 2026-07-13
---

# Mach-Mind-4-Flash

## 身份

理想汽车（Li Auto Inc.）Foundation Model Team 的 agentic MoE 模型，通过纯后训练优化将紧凑基座推到前沿性能。arXiv:2607.09375v1，2026-07-10 发布。

## 关键事实

| 字段 | 值 |
| --- | --- |
| 总参数 | 35B |
| 激活参数 | 3B |
| 基座模型 | [Qwen3.5-35B-A3B](../sources/qwen3.5-omni.md)（后训练不改预训练权重） |
| 架构 | MoE（继承 Qwen3.5 架构，报告未改架构） |
| **模态** | 纯文本（报告未涉及多模态输入/输出； multimodal 仅出现在 future work 展望） |
| 后训练范式 | SFT → 三轨并行 RL → [MOPD](../concepts/multi-teacher-on-policy-distillation.md) → HMPO |
| 团队 | 理想汽车 Foundation Model Team |

## 技术身份

Mach-Mind-4-Flash 的核心定位是**后训练 scaling**：不扩大预训练算力，而是通过 specialization-then-integration 流水线把紧凑基座推到 100B 级性能。三根支柱：

1. **统一 RL/OPD 训练框架**：把 RL 和 OPD 深度集成（加权 loss `L = α·L_OPD + β·L_RL`），共享分布式调度、异步 reward routing 和 online sampling 闭环。动态多 teacher 架构支持积木式组合。SonicMoE indexed Grouped GEMM + shared-expert fusion 带来 17% 端到端加速。

2. **Specialization-then-Integration**：并行训练三轨 RL 专家（Reasoning / General / Agent 共 10+ 个 specialist），用 [MOPD](../concepts/multi-teacher-on-policy-distillation.md) 融合成一个 generalist，替代混合 reward RL（capability see-saw 问题）。每个专家用各自的可验证 reward、数据合成和环境（EnvScaling 190+ domain / 3.5K tool，E2B sandbox，跨 scaffold SWE 环境）。

3. **HMPO**：单阶段 token 效率 RL，从正确 rollout 中位长度推导 adaptive budget，乘法 reward 保证 correctness-first。仅数学训练压缩 19–46%，精度损失 ≤0.7pp，跨域泛化。

MOPD 采用 k1 单样本 reverse-KL 估计 + PPO clipped surrogate 修正 off-policy drift（Appendix B），与 MiMo MOPD 算法形式一致但工程实现不同（统一 RL/OPD loss 框架 + Early Stopping Rollout 8K 截断 + teacher-student 参数量匹配策略）。

## 相关页面

- [Mach-Mind-4-Flash 技术报告](../sources/mach-mind-4-flash.md)：完整 ingest 页面。
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)：MOPD 机制详解。
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)：Mach-Mind 在 MOPD 融合派中的位置。
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)：specialization-then-integration 范式。
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)：35B/3B 在 MoE 模型谱系中的位置（当前收录中激活参数最低的 agentic MoE 之一）。
- [Qwen3.5](../models/qwen3.5.md)：基座模型。
