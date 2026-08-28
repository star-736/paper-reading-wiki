---
type: Model
title: "Qwen-AgentWorld"
description: "Qwen Team 的 native language world model 家族（35B-A3B / 397B-A17B），覆盖 7 域 agentic 环境模拟，三阶段 CPT→SFT→RL 训练。"
tags: ["model", "qwen-agent-world", "language-world-model", "world-model", "qwen"]
timestamp: 2026-07-12
---

# Qwen-AgentWorld

## 身份

Qwen-AgentWorld 是 Qwen Team（Alibaba）提出的 **native language world model（LWM）**家族——首个覆盖 7 个 agentic 交互域（MCP / Search / Terminal / SWE / Android / Web / OS）的单一语言世界模型。它不是通用 LLM，而是以环境模拟为目标从 CPT 阶段就开始训练的专用 world model：给定交互历史与当前 action，预测下一环境观测 `ô_{t+1} = f_θ(c, o_≤t, a_≤t)`。

两个变体均基于 [Qwen3.5](qwen3.5.md) checkpoint（35B-A3B / 397B-A17B），通过三阶段管线 "CPT injects, SFT activates, RL sharpens" 训练。详见 [Qwen-AgentWorld 技术报告](../sources/qwen-agent-world.md)。

## 关键事实（变体表）

| 变体 | 总参 / 激活 | 基座 checkpoint | 训练数据 | 备注 |
| --- | --- | --- | --- | --- |
| Qwen-AgentWorld-35B-A3B | 35B / ~3B | Qwen3.5-35B-A3B | 10M+ 环境轨迹 + world knowledge 语料 | 主用于 Application II（agent 基座 warm-up）实验 |
| Qwen-AgentWorld-397B-A17B | 397B / ~17B | Qwen3.5-397B-A17B | 同上 | 总均分 58.71，AgentWorldBench 最优；主用于 Application I（Sim RL 模拟器） |

**模态**：纯文本——7 域的环境观测统一为文本表示；三个 GUI 域（Android/Web/OS）用 accessibility tree 与 UI view hierarchy 的文本表示，而非像素帧（已据原文核实，§1: "For the three GUI domains, environment observations are represented as accessibility trees and UI view hierarchies rather than pixel frames"）。多模态扩展（融合 GUI 截图与文本表示）列为 future work，当前未实现。

**训练阶段**（已据原文核实，tier-1）：
- **Stage 1 CPT**：标准 next-token prediction + turn-level 信息论 loss masking（7 类别，keep ratio 5%–100%）；注入环境 state-transition dynamics 与专门领域 world knowledge（工业控制/网络安全/法律/医疗/金融/时事百科）。
- **Stage 2 SFT**：thinking 轨迹（显式 next-state-prediction 推理链），rejection sampling（10,250→7,094，69.2% retention），256k 上下文，10 模板变体多样化。
- **Stage 3 RL**：[GSPO](../sources/group-sequence-policy-optimization.md) 算法，reward = 五维 rubric（LLM judge，[5,25]）: rule-based verifier（binary，缩放到 [0,25]）= 9:1；128k prompt 上限；三种稳定性解法（每轨迹单 turn 展开 / rubric+rule 而非 reference/turing-test reward / tag extraction 防 self-praise）。

## 技术身份

Qwen-AgentWorld 的核心定位是 **native world model**——从 CPT 阶段就以环境建模为训练目标，而非事后 fine-tune 通用 LLM。这与论文 Related Work 中区分的两类工作形成对照：

- **Learned neural simulator**（如 RLVR-World、WebWorld、SWE-World、SSRL、ZeroSearch、ECHO）：用 RL/SFT 训练 LLM 作环境模拟器，但多为单域或事后 fine-tune。Qwen-AgentWorld 是首个 7 域统一、三阶段 native 训练的 LWM。
- **Code-driven synthetic environment**（如 [Agent-World](../sources/agent-world.md)、AWM、ScaleEnv、AutoForge）：程序化生成环境，保证确定性执行与可验证 reward，但限于可程序化指定的域。Qwen-AgentWorld 的 LWM 模拟是互补的——以确定性换通用性，覆盖 code 难以指定的域（搜索引擎、真实 MCP servers）。

论文论证 world modeling 增强 agent 的两条路径：(1) **Decouple**——LWM 作独立模拟器，靠可控模拟（注入扰动 / 构造虚构世界）做 Sim RL，甚至超过真实环境训练（WideSearch Sim RL 50.3% vs Real RL 45.6%）；(2) **Unify**——LWM RL warm-up（单轮、无工具调用）把 next-state prediction 内化为 meta-reasoning 模式，跨 7 个 agentic benchmark（含 3 个完全 OOD 域）一致提升。后者的机制证据是 prediction-driven action refinement：RL 后模型在执行前系统性地心智模拟环境响应（mailman case study 中正确预测 Postfix 处理流程），预测准确率 69.9%→78.3%。

## 相关页面

- 来源：[Qwen-AgentWorld 技术报告](../sources/qwen-agent-world.md)
- 基座：[Qwen3.5](qwen3.5.md)（35B-A3B / 397B-A17B checkpoint）
- 同基座、不同任务：[WeMM-Embedding](wemm-embedding.md)（Qwen3.5 2B/4B/9B 做通用多模态 embedding）
- RL 算法：[Group Sequence Policy Optimization](../sources/group-sequence-policy-optimization.md)（GSPO）
- 互补对照：[Agent-World](../sources/agent-world.md)（code-driven 环境合成路线）
- 概念：[Agentic Engineering](../concepts/agentic-engineering.md)、[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)、[Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)
