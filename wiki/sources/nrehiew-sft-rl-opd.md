---
type: Source
title: "nrehiew 博客：SFT, RL, and On-Policy Distillation Through a Distributional Lens"
description: "nrehiew 2026 博客，用分布视角统一 SFT / RL / OPD 三方法。核心论点：on-policy 数据（而非 RL 本身或显式 KL 惩罚）是抗遗忘的承重墙。关键实验：OPD student 不论从 SFT 还是 RL teacher 蒸馏，结果几乎一致且都略超 RL teacher。覆盖 OPSD 变体、RL 抗遗忘三解释的综合审视、student 超越 teacher 的机制、pipeline 趋势（GLM-5 / DeepSeek-V4 最终 checkpoint 不经 RL）。"
tags: ["source", "nrehiew-sft-rl-opd", "on-policy-distillation", "post-training"]
timestamp: 2026-07-30
resource: "../../raw/nrehiew-sft-rl-opd.md"
---

# nrehiew 博客：SFT, RL, and On-Policy Distillation Through a Distributional Lens

## 来源

- 原始文件（本地快照）：[`raw/nrehiew-sft-rl-opd.md`](../../raw/nrehiew-sft-rl-opd.md)（`raw/` 已 git-ignored，本地可读）
- 原网址：<https://nrehiew.github.io/blog/sft_rl_opd/>
- 标题：SFT, RL, and On-Policy Distillation Through a Distributional Lens
- 副标题：On forgetting, generalization, and what connects RL to on-policy distillation
- 作者：nrehiew
- 发表：2026（确切日期未标注，文中引用至 2026 年文献）

## 为什么这篇博客在 wiki 里独占一席

[Thinking Machines Lab 博客](thinking-machines-on-policy-distillation.md)是 GLM-5 / MiMo 共同引用的 OPD 算法源头，讲的是「OPD 为什么 work」的数学依据。这篇 nrehiew 博客回答的是另一个互补的问题：**把 SFT、RL、OPD 并排放，什么解释了它们的定性差异？** 答案是 on-policy 数据，而非 RL 的算法独特性或显式 KL 惩罚。

它独占一席的三个理由：

1. **分布视角三轴框架**（target distribution 是什么 / 数据从哪来 / KL 方向）是已有 wiki 里没有的统一视角--概念页有 OPD 的七层数学依据，但没有把 SFT 和 RL 也拉进同一框架做对照。
2. **对照实验**：同一 task 上分别用 SFT teacher 和 RL teacher 做 OPD 蒸馏，student 结果几乎一致--这是「on-policy 数据 > teacher」的直接实验证据，已有 wiki 里没有。
3. **RL 抗遗忘三解释的综合审视**：forward/reverse KL（不完整）、梯度稀疏性（领域受限）、on-policy 数据（最认同），这种跨解释的批判性对比在技术报告里不会出现。

## 核心结论

### 1. 分布视角下的三方法对照

| | SFT | RL | OPD |
| --- | --- | --- | --- |
| target distribution | 固定外部分布（数据集） | 无明确外部目标 | teacher 分布 |
| 数据来源 | 外部数据集（off-policy） | 当前策略采样（on-policy） | 当前策略采样（on-policy） |
| KL 方向 | forward KL（mode-covering） | ~reverse KL（mode-seeking）（[Chen et al., 2025](https://arxiv.org/abs/2510.18874)） | reverse KL |
| 梯度压力 | 均匀施加在所有 token 上 | 只在当前策略采样的高概率区域 | 只在 student 采样区域，朝 teacher 拉 |
| 遗忘 | 严重--无内置机制区分 task-critical token 和 incidental artifact | 轻微--on-policy 局部性 | 轻微--继承 on-policy 局部性 |

**SFT 的遗忘机制**：cross-entropy loss 对每个 demonstrated token 一视同仁地推高概率，不区分 task-critical token（数学运算符）和 style token（"therefore"）。[Diao et al.](https://arxiv.org/abs/2601.02151) 发现 SFT 中存在大量 low-probability low-entropy token--模型本来很自信，却被迫拟合分歧的 ground-truth label，冲击已有表示。

**RL 的抗遗忘机制**：只在当前策略采样的区域施压。[Lai et al., 2025](https://arxiv.org/abs/2507.05386) 指出 RL 有 data-dependent regularization--advantage 归一化时，高多样性 / 高 reward variance 的 group 获得更小更新（模型不确定时自然减速）。[Mukherjee et al.](https://arxiv.org/abs/2505.11711) 发现 RL 只更新小子网络（sparse but full-rank），SFT 则是 dense 更新；[Yuan et al.](https://arxiv.org/abs/2510.04454) 通过剪枝证实 SFT 更新冗余、RL 更新更关键。

### 2. 关键实验：on-policy 数据 > teacher

作者在自己的 Minimal Code Editing 任务上做对照（模型拿到一个被注入 bug 的函数，要求只修 bug 不改其他部分）。该任务同时能测泛化（训一种 corruption 类型、测另一种）和遗忘（LiveCodeBench v6 检查通用代码能力退化）。

先分别训 SFT teacher 和 RL teacher（RL 泛化更好、不遗忘；SFT 遗忘明显），再分别做 OPD 蒸馏到 student：

| Model | Pass@1 ↑ | Norm. Levenshtein ↓ | Added CC ↓ | LiveCodeBench v6 ↑ |
| --- | --- | --- | --- | --- |
| **Teachers** | | | | |
| SFT | 0.775 | 0.450 | 0.450 | 0.286 |
| RL | 0.792 | 0.063 | 0.206 | 0.320 |
| **Students (OPD)** | | | | |
| OPD SFT teacher | **0.800** | 0.059 | **0.206** | 0.297 |
| OPD RL teacher | 0.787 | **0.055** | 0.228 | **0.314** |

（Norm. Levenshtein 和 Added CC 越低越好，衡量「有没有改不该改的部分」。）

**反直觉结果**：

- 两个 OPD student **几乎一样**--不论 teacher 是 SFT 还是 RL 训出来的。
- 两个 student 都**略超 RL teacher**、远超 SFT teacher。
- 即使 teacher 是退化的 SFT 模型，OPD student 的遗忘也**比 SFT teacher 本身轻**。如果 teacher 分布是决定性因素，从 SFT teacher 蒸馏的 student 应该继承更多遗忘--但并没有。

**结论**：teacher 提供信号，但 on-policy 采样决定了几何形状。这暗示可以「暴力 SFT 过训练一个 expert → OPD 蒸馏 → 保留原模型大部分能力」。

### 3. RL 为什么遗忘少？三种解释的综合审视

**解释一：SFT 是 forward KL，RL 是 reverse KL**（[Chen et al., 2025](https://arxiv.org/abs/2510.18874)）。forward KL 的 mode-covering 行为可能牺牲已有 mode 来学新任务；toy 实验显示 reverse KL 遗忘更少。

作者评价：**有用但不完整**。论证依赖显式 KL 正则化对 reference model 的约束，但 RLVR 去掉或大幅削弱 KL 惩罚后仍抗遗忘。

**解释二：SFT 梯度均匀 vs RL 稀疏参数更新**（Mukherjee / Yuan et al.）。

作者评价：**实证成立但领域可能受限**，且没解释 OPD 如何嵌入这个画面。

**解释三（作者最认同）：on-policy 数据**（[Shenfeld et al.](https://arxiv.org/abs/2509.04259)）。用最简版本的 REINFORCE + binary 0/1 reward 看：reward=1 贡献正信号，reward=0 不贡献--reward 充当 filter，类似 rejection sampling。由此 RL 有了隐式的 target distribution：所有 completion 都 reward=1 的 optimal policy。由于在 on-policy 样本上训练，这个 optimal 分布是**所有 optimal policy 中离当前策略最近的**。每步 policy gradient 都在拟合一个隐式低 KL 的目标。

几何含义：on-policy 数据在每个时间步把训练约束在离起始策略近的分布上；SFT 的 target distribution 可以任意远。**这解释了 OPD student 为什么能继承 RL 的抗遗忘--即使 teacher 是 SFT 模型，student 的 state 分布仍然是自己的。**

### 4. OPSD：On-Policy Self Distillation

[OPSD](https://arxiv.org/abs/2601.18734) 是 OPD 的新变体：teacher 和 student 是**同一模型**，但 teacher 计算 log probability 时被提供 reference solution 作为 prefix（privileged information）。

问题：同一模型做 teacher/student 时，大多数 token 输出几乎相同。per-token KL 分析发现 **style / pivot token**（"wait"、"alright"）的 KL 远高于 **math token**（"power"、"exponent"、"logarithm"）。如果在这些不重要的高 KL token 上更新太猛，模型可能 collapse。

解法：引入 per-token clipping 防止过度更新。

作者评价：OPSD 更接近 RLHF 而非 RLVR。RLHF 的 reward model 有偏，需要 KL penalty + trust-region clipping 防止过度优化错误目标；OPSD 的 teacher 信号也不完全相关于 task importance（高 KL token 可能只是 style）。RLVR 的 reward function 偏差低，所以更敢去掉 KL penalty 或放松 trust-region（用 GRPO 替代 PPO）。

### 5. Student 为什么能超越 Teacher

不是新现象（[Agarwal et al. 2023](https://arxiv.org/pdf/2306.13649) 在 GSM8K 上已报告），作者给出两个假设：

1. **OPD 监督更精准**：teacher 在 student 自己的 prefix 上给建议。student 的错误不一定是 teacher 的错误；如果只训 teacher 生成的轨迹，student 可能在自己很少访问的分布区域收到监督。OPD 让 teacher 针对 student 的实际状态给建议。
2. **KL matching ≠ reward maximization**：teacher 分布含 style、不确定性、替代路径、推理结构等信息。匹配它能在不复制 teacher greedy 行为的前提下重塑 student 分布，改善采样行为。即使 teacher 的采样输出不更好，student 仍能进步。

**熵坍缩**：OPD 的 entropy collapse 比 RL 更剧烈（reverse KL mode-seeking 的预期行为，[Gu et al., 2023](https://arxiv.org/abs/2306.08543)）。RL 的 reward 缓慢上升；OPD 的 reward 上升更突然，伴随熵的急剧坍缩。这部分是推测性的。

### 6. 为什么 RL 和 OPD 泛化更好

SFT 惩罚模型不给特定答案概率；RL 的监督绑定 task success 而非特定 token 序列。关键引用 [Ross et al. (DAgger)](https://arxiv.org/abs/1011.0686)：SFT 只看 teacher 访问的 state，test time autoregression 下 student 一步走错就可能进入 teacher 从未访问的 state，导致 compounding error。on-policy 数据聚合减少这种 state 分布不匹配。

### 7. Pipeline 趋势

[GLM-5](glm-5.md) 和 [DeepSeek-V4](deepseek-v4.md) 都用 OPD 做最终 expert merging，最终 checkpoint 不经 RL。[MiMo-V2-Flash](mimo-v2-flash.md) Table 7 显示：Math/Code 偏好 RL teacher；Creative writing / 知识密集型偏好 self-distillation / 蒸馏（因为这些领域 reward 噪声大，LLM judge 是有偏代理）。

值得注意的是：在 RL 领域，最终 merged student 几乎总是超过 teacher；在 self-distilled 领域，student 有时不如 teacher。

### 8. 终极问题

理想的 post-training 算法需要同时具备：蒸馏的**密度** + RL 的**无偏** + 两者的 **on-policy** 特性。目前没有解。outcome reward 太稀疏（RL 昂贵）；Process Reward Model（[Lightman et al.](https://arxiv.org/abs/2305.20050)）大规模训练效率低；logit distillation 密度高但有偏，被迫进入 messy clipping schemes。

## 与已有 wiki 的关系

- [Thinking Machines Lab On-Policy Distillation 博客](thinking-machines-on-policy-distillation.md)：算法源头（reverse-KL、O(1) vs O(N) bits、personalization 召回）。本博客是互补视角--Thinking Machines 讲「OPD 为什么 work」的数学，nrehiew 讲「把 SFT/RL/OPD 并排放，什么解释了它们的差异」。
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)：概念页的七层数学依据是 Thinking Machines 博客的结构化整理；本博客的 on-policy 承重墙实验和分布视角三轴框架是该概念页的补充视角。
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)：对比页列了 5 家 OPD 用法；本博客的 MiMo Table 7 分析（RL domain vs self-distill domain）和 GLM-5/DSV4 最终 checkpoint 不经 RL 的趋势观察是对比页的补充。

## 待追问

- **on-policy 数据 > teacher 的结论是否只在 minimal editing 这种 niche task 上成立**？作者自己说该 task 适合测遗忘和泛化，但在更 broad 的能力域上，teacher 质量是否会重新主导？
- **OPSD 的 per-token clipping 与 KAT-Coder-V2.5 的 drift-aware truncation / Keye-VL-2.0 的 top-k overlap estimator 是否在解决同一个问题**？三者都在 token 级别控制 OPD 的更新质量，但切入点不同（style token 降权 vs 长上下文 drift vs 双方低概率 token 过滤）。
- **entropy collapse 的剧烈程度是否可调**？OPD 比 RL 更剧烈的熵坍缩是 reverse KL mode-seeking 的预期，但是否意味着 OPD student 的多样性损失比 RL 更严重？这与 Qwen3 Table 21 里 OPD pass@64 也涨（不只是 pass@1）的现象是否矛盾？
- **「暴力 SFT 过训练 expert → OPD 蒸馏」是否已在产业报告中出现**？作者提出这是 hopeful result，但 MiMo/V4 的 teacher 都是 RL 训出来的，有没有 SFT teacher + OPD 的实例？

## 相关页面

- [Thinking Machines Lab On-Policy Distillation 博客](thinking-machines-on-policy-distillation.md)：OPD 算法源头，本博客的互补前作。
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)：MOPD 机制 + 跨家共用 OPD 数学依据（七层论证）。
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)：5 家 OPD 用法分歧（目的 / KL 形式 / pipeline 位置）。
- [GLM-5 技术报告](glm-5.md) / [MiMo-V2-Flash 技术报告](mimo-v2-flash.md) / [DeepSeek-V4 技术报告](deepseek-v4.md) / [Qwen3 技术报告](qwen3.md)：本博客引用的 pipeline 趋势来源。
