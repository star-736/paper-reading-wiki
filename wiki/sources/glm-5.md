---
type: Source
title: "GLM-5 技术报告"
description: "GLM-5 的 arXiv 技术报告，重点是 agentic engineering、DSA 和异步 RL。"
tags: ["source", "glm-5"]
timestamp: 2026-06-06
resource: "../../raw/glm-5-2602.15763.pdf"
---

# GLM-5 技术报告

## 来源

- 原始 PDF：[raw/glm-5-2602.15763.pdf](../../raw/glm-5-2602.15763.pdf)
- 标题：GLM-5: from Vibe Coding to Agentic Engineering
- 版本/日期：arXiv:2602.15763v2，2026-02-24
- 团队：GLM-5 Team，Zhipu AI 与 Tsinghua University
- 模型页：[GLM-5](../models/glm-5.md)

## 核心结论

GLM-5 把模型能力的主线放在 ARC：agentic、reasoning、coding，而不是普通聊天或静态代码补全。报告中的核心表述是从 "vibe coding" 走向 [agentic engineering](../concepts/agentic-engineering.md)：模型需要完成长周期软件任务、工具调用、仓库探索，并在多轮操作中保持推理状态。

GLM-5 是 744B 总参数 / 40B 激活参数的 MoE 模型，包含 256 个专家和 80 层。基础模型训练总量为 28.5T tokens；中期训练逐步扩展到 32K、128K 和 200K 上下文；SFT 阶段最大上下文为 202,752 tokens。

## 架构与训练

GLM-5 使用 DeepSeek Sparse Attention（DSA）实现 [高效长上下文注意力](../concepts/efficient-long-context-attention.md)。报告把 DSA 描述为基于内容的重要 token 选择机制，用于降低长序列注意力计算。相对密集 MLA 基线，DSA 在长上下文上接近保持性能，同时把注意力计算降低约 1.5-2 倍。

报告还引入 Muon Split 来改善 MLA 训练，并使用带参数共享的 multi-token prediction（MTP）。在私有 prompt 集上，相同 4 个 speculative steps 下，GLM-5 的接受长度为 2.76，DeepSeek-V3.2 为 2.55。

## 后训练

后训练流水线包括多任务 SFT、reasoning RL、agentic RL、general RL 和 on-policy cross-stage distillation。agentic RL 系统是异步的：rollout 生成与训练引擎解耦，由 multi-task orchestrator 调度。

> on-policy cross-stage distillation 的算法形式与 MiMo MOPD 高度相似（reverse-KL log-ratio 当 advantage 塞进 GRPO 框架，公式 2），同样是**多 teacher × prompt 路由 × KL 当 loss**——SFT / Reasoning RL / General RL 三个阶段的 final checkpoint 都当 teacher，prompt 按其归属阶段路由到对应 teacher 算 KL。差别不在 teacher 数量，在 teacher 性质：MiMo / V4 的 teacher 是**横向**专门训出来的领域专家，GLM-5 的 teacher 是**纵向**自己流水线上各阶段留下的快照——融合的是同一个 student 不同时间点的能力，对应问题是 sequential RL 阶段间的能力遗忘。一个有意思的工程后果：advantage 直接来自 teacher gap 而非 group 内对比，所以 GRPO group size 可降到 1（§3.5 原文）。算法源头是 [Thinking Machines Lab On-Policy Distillation 博客](thinking-machines-on-policy-distillation.md)（§3.5 reference [28]）；详见 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md) 轴一第三类。

两个关键稳定性机制是 Token-in-Token-out（TITO）和 direct double-sided importance sampling。TITO 避免重新分词带来的轨迹错位；double-sided importance sampling 用 token 级裁剪控制异步 off-policy 偏差。

## 评测要点

报告宣称 GLM-5 在多个开放模型 ARC benchmark 上达到 SOTA。代表性结果包括 SWE-bench Verified 77.8、SWE-bench Multilingual 73.3、BrowseComp 62.0、带 context management 的 BrowseComp 75.9，以及 Vending-Bench 2 结算余额 $4,432。

## 待追问

- GLM-5 的 agentic 增益中，架构、数据、RL 环境和推理框架各自贡献多少？
- DSA 降低成本，但 RL 稳定性依赖 deterministic top-k，这可能是复现和部署时的关键约束。

