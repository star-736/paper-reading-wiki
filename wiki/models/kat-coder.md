---
type: Model
title: "KAT-Coder"
description: "快手 KwaiKAT 团队的 agentic coding 模型族，V2 和 V2.5 两代报告覆盖 Specialize-then-Unify 范式、KwaiEnv 沙箱基础设施、MCLA/Tree Training/asymmetric PPO 和 MOPD 专家融合。"
tags: ["model", "kat-coder", "agentic-coding"]
timestamp: 2026-07-11
---

# KAT-Coder

## 身份

KAT-Coder 是快手 KwaiKAT 团队的 agentic coding 模型族，核心定位是让模型在真实可执行仓库中自主行动，而非做单轮代码生成。当前沉淀覆盖 V2 和 V2.5 两代报告。

主要来源：[KAT-Coder-V2 技术报告](../sources/kat-coder-v2.md)、[KAT-Coder-V2.5 技术报告](../sources/kat-coder-v2.5.md)。

## 关键事实

| 变体 | arXiv | 日期 | 核心范式 | RL 算法 | 专家融合 |
| --- | --- | --- | --- | --- | --- |
| KAT-Coder-V2 | [2603.27703](https://arxiv.org/abs/2603.27703) | 2026-03-29 | Specialize-then-Unify（5 域分治 + OPD 统一） | Turn-level GSPO adaptation + MCLA + IcePoP | OPD（单 teacher per task） |
| KAT-Coder-V2.5 | [2607.05471](https://arxiv.org/abs/2607.05471) | 2026-07-06 | 系统性基础设施重构（AutoBuilder + KwaiClawEnv + harness scaling + sandbox hardening） | Asymmetric PPO with hindsight-augmented critic | MOPD（5 teacher + cold start + drift-aware truncation） |

**模态**：纯文本。两篇报告均未提及视觉/音频/视频输入模态，模型定位为 coding-focused agentic model，所有交互基于文本指令 + 代码仓库 + 终端环境。

**基座**：两篇报告均未公开 backbone 架构和参数量，只说基于 KAT-Coder-V1（arXiv:2510.18779）继续后训练。V1 的基座也未在 V2/V2.5 报告中披露。

**公开服务**：[streamlake.com/product/kat-coder](https://streamlake.com/product/kat-coder)

## 技术身份

KAT-Coder 的定义性特征不在模型架构（未公开），而在 **agentic post-training 的系统化方法论**：

- **Specialize-then-Unify**：把 agentic coding 分解为正交专家域，独立训练再统一融合。V2 分 5 域（SWE / WebCoding / Terminal / WebSearch / General），V2.5 调整为（SWE / Agent-Claw / Terminal / WebCoding / General）。
- **KwaiEnv**：模块化解耦 datasets / sandboxes / scaffolds / verifiers，万级并发沙箱，网络层代理实现任意 scaffold 零代码集成。V2.5 新增 Gateway Server 中介推理-训练循环 + 强制 token consistency。
- **环境驱动的数据构造**：AutoBuilder 从真实 GitHub 仓库自动构建可验证 SWE 任务（F2P+P2P），V2 产出 30k 样本，V2.5 扩展到 100K+ 环境（12 语言，成功率 16.5%->57.2%）。
- **MoE RL 稳定化**：V2 用 MCLA（K=8 forward prefill 取 log-prob 平均降方差）+ IcePoP（裁剪 train-inference 偏差 token）。V2.5 切换到 asymmetric PPO，Critic 接收 hindsight 信息（final reward / test outcomes / patch signals）降方差。
- **Tree Training**：V2 把树状 agent 轨迹序列化为 DFS 展平序列，共享前缀只算一次，梯度等价于独立训练所有路径，6.2× 端到端加速。
- **长上下文 MOPD 稳定化**：V2.5 在标准 MOPD 上加 off-policy cold start + drift-aware dynamic truncation（top-k overlap 控制 token 权重和截断），解决长轨迹上 teacher-student 分布偏离导致的 KL 信号不可靠。

## 代际演进

V2 -> V2.5 的核心变化不是模型架构升级，而是训练基础设施的系统性重构：

- **RL 算法**：从 GRPO 变体（turn-level GSPO adaptation）-> PPO + hindsight-augmented critic。理由是生产级 harness 产生结构分裂样本，trajectory-level group baseline 难定义。
- **数据质量**：从 F2P+P2P 验证 -> 增加 process-aware 过滤（九维 process scoring）+ near-miss hint 恢复 + hint-free replay。
- **环境可靠性**：V2.5 发现 ~16% 轨迹含沙箱自身失败（非模型问题），通过镜像管理和框架 bug 修复降到 <2%。
- **Harness 泛化**：从多 scaffold 数据构造 -> 系统化 harness randomization（三轴：tool-invocation protocol / context-management / control-flow）+ harness rewriting 注入扰动。
- **OPD -> MOPD**：从单 teacher per task -> 5 teacher + cold start + drift-aware truncation，专门处理长上下文 OPD 不稳定性。
- **新 benchmark**：V2.5 新建 KAT Code Bench（repository-level SWE）和 KAT Claw Bench（业务导向 tool-use）。

## 相关页面

- [KAT-Coder-V2 技术报告](../sources/kat-coder-v2.md)
- [KAT-Coder-V2.5 技术报告](../sources/kat-coder-v2.5.md)
- [Agentic Engineering](../concepts/agentic-engineering.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
