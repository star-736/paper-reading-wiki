---
type: Source
title: "Thinking Machines Lab On-Policy Distillation 博客"
description: "Kevin Lu 2025-10-27 发表，GLM-5（ref [28]）/ MiMo MOPD 共同引用的 OPD 算法源头。Per-token reverse KL、三方对照表（SFT / RL / OPD = off-policy+dense / on-policy+sparse / on-policy+dense）、`O(1)` vs `O(N)` bits/episode 信息论分析、personalization 召回实验是 GLM-5 cross-stage distillation 思路的直接来源。"
tags: ["source", "thinking-machines-on-policy-distillation"]
timestamp: 2026-06-23
resource: "../../raw/thinking-machines-on-policy-distillation-2025-10-27.md"
---

# Thinking Machines Lab On-Policy Distillation 博客

## 来源

- 原始文件（本地快照）：[`raw/thinking-machines-on-policy-distillation-2025-10-27.md`](../../raw/thinking-machines-on-policy-distillation-2025-10-27.md)（`raw/` 已 git-ignored，本地可读）
- 原网址：<https://thinkingmachines.ai/blog/on-policy-distillation>
- 标题：On-Policy Distillation
- 作者：Kevin Lu，与 Thinking Machines Lab 合作
- 发表：2025-10-27（Thinking Machines Lab: Connectionism）
- DOI：`10.64434/tml.20251026`
- 引用：`Lu, Kevin and Thinking Machines Lab, "On-Policy Distillation", Thinking Machines Lab: Connectionism, Oct 2025.`
- 团队：Thinking Machines Lab
- 实验代码：[Tinker cookbook · recipes/distillation](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)

## 为什么这篇博客在 wiki 里独占一席

它是 [GLM-5](glm-5.md)（reference [28]）、[MiMo-V2-Flash](mimo-v2-flash.md)（§4 引用）等多份 2025-2026 技术报告反复引用的**奠基性文献**——OPD 的 reverse-KL 损失形式、on-policy 采样的直觉、与 RL / off-policy 的三方对照表，技术报告里普遍只用一两句话带过、把推导和实验留给这篇博客。所以把它独立成源页能让对比页与机制讨论有一个共同祖宗，不至于把同一套基础知识在 5 个源页里各讲一遍。

它本身**不是技术报告**，是博客；但内容密度（实验、消融、信息论分析、连续学习应用）超过多数技术报告里 OPD 那一节，且引用的祖先工作（DAGGER、Process Reward Modeling、Agarwal 2023、Gu 2023、Qwen3）把 OPD 的学术血脉钉得很清楚。

## 核心结论

1. **OPD = on-policy 采样 + dense 监督**：与 SFT（off-policy + dense）、RL（on-policy + sparse）形成三方对照——OPD 取两家之长。

   | Method | Sampling | Reward signal |
   | --- | --- | --- |
   | Supervised finetuning | off-policy | dense |
   | Reinforcement learning | on-policy | sparse |
   | On-policy distillation | on-policy | dense |

2. **Loss = per-token reverse KL**（§ Loss function 公式）：

   $$\text{KL}(\pi_\theta \| \pi_\text{teacher}) = \mathbb{E}_{x \sim \pi_\theta}\!\left[\log \pi_\theta(x_{t+1}|x_{1..t}) - \log \pi_\text{teacher}(x_{t+1}|x_{1..t})\right]$$

   博客明确点出 reverse-KL 的三条性质：**unhackable**（低 KL 总对应高 teacher 满意度，不像 reward model 会被 hack）、**mode-seeking**（学一种特定行为而非把质量摊薄到多个次优 option）、**降低 exposure bias**（在 student 自己会遇到的状态上接受监督）。

3. **信息论效率：`O(1)` vs `O(N)` bits/episode**（§ Discussion · Dense supervision）：RL 每个 episode 只传 `O(1)` bits 信息（最终对错），distillation 传 `O(N)` bits（N = token 数，每个 token 都有 KL 信号）。实验显示同一目标 distillation 比 RL 快 **7-10× gradient steps**，compute 效率 **50-100×**。

4. **复现 Qwen3 Table 21**：用 Qwen3-32B 当 teacher 蒸馏 Qwen3-8B-Base 做 AIME'24，从 60%（400k off-policy distill 起点）→ 70%（约 150 步 on-policy distill，约 77K prompts，相比 RL 的 17,920 GPU·h 估算便宜 **9-30×**，看是否把 SFT 数据集摊销/重新生成算进去）。Qwen3 报告原文表格作为对照：

   | Method | AIME'24 | GPQA-Diamond | GPU Hours |
   | --- | --- | --- | --- |
   | Off-policy distillation | 55.0% | 55.6% | Unreported |
   | + Reinforcement learning | 67.6% | 61.3% | 17,920 |
   | + On-policy distillation | 74.4% | 63.3% | 1,800 |

   > 这条数据被 Qwen3、MiMo、GLM-5、Thinking Machines 互相引用形成回环——Qwen3 给数字，Thinking Machines 给算法解释，GLM-5/MiMo 拿来当 OPD 算法选型的实验依据。

5. **Personalization 实验 = GLM-5 cross-stage 召回思路的直接来源**：把已经过 RL 的 Qwen3-8B 用内部公司文档 SFT 之后，IF-eval 从 85% 暴跌到 45%（catastrophic forgetting）；用**原版 Qwen3-8B** 当 teacher、Tulu3 当 prompt 做 on-policy distill，IF-eval 召回到 83%，且 internal QA 知识保留 36→41%。**GLM-5 cross-stage distillation 把这个 setup 中的「内部文档 SFT」换成「整条 RL 流水线」、把「原版 Qwen3-8B teacher」换成「上游 RL checkpoint」，机制完全照搬**。

6. **博客自己提名了 phase-alternating 框架**：fine-tune 学新知识 → on-policy distill 召回行为 → fine-tune 学新知识 → distill ……（引 [Cobbe et al. 2020 *Phasic Policy Gradient*](https://arxiv.org/abs/2009.04416)）。这是连续学习（continual learning）的可行 recipe。

## 实现细节：算法与伪代码

§ Implementation · Pseudocode 给出 Tinker 的实际实现（4 步）：

```python
# 1. 初始化 teacher（main） —— teacher 走 sampling client
teacher_client = service_client.create_sampling_client(
    base_model=teacher_config.base_model,
    model_path=teacher_config.load_checkpoint_path,
)

# 2. student 自己采样 trajectories（main）
trajectories = do_group_rollout(student_client, env_group_builder)
sampled_logprobs = trajectories.loss_fn_inputs["logprobs"]  # student 的 log π_θ

# 3. 计算 reverse KL 当 reward（compute_teacher_reverse_kl）
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = sampled_logprobs - teacher_logprobs            # log π_θ - log π_T = per-token reverse KL
trajectories["advantages"] = -reverse_kl                    # advantage = -KL，最小化 KL ⇔ 最大化 advantage

# 4. 走标准 RL 训练循环（train_step），loss 用 importance_sampling
training_client.forward_backward(trajectories, loss_fn="importance_sampling")
```

四点工程要点：

- **advantage = -reverse_kl**，正负号取反后塞进 RL 框架。这就是 GLM-5 公式 (2) 和 MiMo 公式 (8) 的形式来源——三家算法上等价。
- **discount factor = 0**：博客原文「more mathematically correct 是 > 0，但实践不提升，所以选 0 求简单」——每个 timestep 只优化下一个 token，不考虑未来 token。
- **teacher 只跑 forward** 算 logprobs，不需要训练；rollout 由 student 这个**更小更便宜的模型**采样——这是 OPD 比 RL 便宜的工程根源（trajectory 由便宜模型生成、reward 由贵模型只跑一次 forward）。
- **不需要单独的 reward model**：teacher 的 logprob 就是 reward。

## 三个关键洞察（值得反复读）

### 1. RL searches in the space of semantic strategies（§ Discussion）

博客的核心隐喻：

- **Pre-training** 在**参数空间**做 SGD 搜索——所以参数空间相对每个网络独立，难以从一个网络蒸馏到另一个（Lottery Ticket Hypothesis 一类工作支持）。
- **RL** 在**语义策略空间**搜索——它只在 base model 已有能力（即"非零成功率"）的基础上做局部修改，"stumble" 到新策略；compute 主要花在 **search**（rollout + credit assignment）而不是 update。
- **Distillation** = 学习 RL 找到的**最终策略**的捷径——不需要重走 RL 那条"先发现中间策略再升级"的曲线，直接拟合终点策略即可。如果你只关心最终策略（生产部署常见），就没必要为所有中间策略付 compute。

**类比**：科研发现一个 result 要花大量时间探索，但发现后用自然语言表达给别人很便宜。运动技能（肌肉记忆）反过来——只能反复练习，没法语言传授。**RL ≈ 科研探索；Distillation ≈ 论文宣讲**。

### 2. RL 训练只动小子网络，能力脆弱（§ Personalization）

引 Mukherjee et al. 2025 [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711)：RL 只调整原模型的小子网络——所以这些能力**脆弱**，被大规模 SFT/mid-training 一冲就掉。

实验证据：Qwen3-8B（已 RL）+ 内部文档 SFT → IF-eval **85% → 45%**。这不是数据集脏，是 RL 子网络被冲掉了。OPD 在这种场景下不是「学新东西」，是「用 teacher 当锚把丢掉的子网络拉回来」——博客原文称作 *"re-invoke" capabilities lost during fine-tuning*。

**这是 GLM-5 cross-stage distillation 为什么必要的根本数学/系统原因**——sequential RL 阶段间互相冲掉对方训出的子网络，cross-stage distill 用上游 checkpoint 当 teacher 召回那些子网络。

### 3. "Forking tokens" —— teacher 实际惩罚什么

§ Illustration 给的实例（SimpleBench 题目，正解 "B. 0" 因为冰块会融化）：

- student（Qwen3-4B-Instruct-2507）当成纯数学题做错了。
- teacher（Qwen3-235B-A22B-Instruct-2507）逐 token 评分，**惩罚集中在「开启错误推理分支的开头 token」**（"forking tokens"，引 [Wang et al. 2025](https://arxiv.org/abs/2506.01939) 高熵少数 token 主导有效 RL）。
- **最终错误答案本身没被罚**——在已经走错的轨迹下，错误答案是确定性后果，KL 信号已经在前面给完了。

这个观察对 mode-seeking 的工程意义重大：reverse-KL 的"零质量惩罚"主要打在分叉点，不浪费在已注定的后果上——这是 dense reward 比 sparse reward 在 credit assignment 上**更精确**的具体表现。

## 学术血脉

博客直接点名继承的工作：

- **[DAGGER](https://arxiv.org/abs/1011.0686)**（Ross et al. 2010, AISTATS）—— iterative SFT + teacher 评 student-visited states；OPD 的 on-policy 监督结构来自这里，cumulative regret 从 off-policy 的 O(T²) 降到 O(T) 也来自 DAGGER 原始证明。
- **[Process Reward Modeling](https://arxiv.org/abs/2305.20050)**（Lightman et al. 2023, *Let's Verify Step by Step*）—— RL 每步打分；OPD 的 per-token KL 是其退化到 token 粒度的版本。
- **[Agarwal et al. 2023](https://arxiv.org/abs/2306.13649)** *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* —— OPD 在 LLM 上的早期工作，本博客直接 extend。
- **[MiniLLM (Gu et al. 2023)](https://arxiv.org/abs/2306.08543)** —— reverse-KL 形式的知识蒸馏；OPD 的 mode-seeking 论证来自这一系。
- **[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)**（2025）—— Table 21 的 1/10 GPU·h 实验"served as inspiration for our work"。

后向影响（已经引这篇博客的报告）：

- **[GLM-5](glm-5.md)** §3.5 reference [28]——cross-stage distillation 直接引博客作为算法依据。
- **[MiMo-V2-Flash](mimo-v2-flash.md)** §4——MOPD 引 Agarwal 2023 + Gu 2023 + Lu 2025（即本博客）三篇作为 OPD 文献。

## 待追问

- **discount factor 之争**：博客选 0（实践不提升），但有些 OPD 工作走 > 0；在什么场景 > 0 才有意义？长 horizon agentic task 是否例外？
- **teacher 选择**：博客发现 Qwen3-8B 当 teacher 蒸 8B 自己比 Qwen3-32B 当 teacher 在 personalization 上更好（"sampling from Qwen3-8B is better than Qwen3-32B for preserving chat capabilities"）。这违反"teacher 越强越好"直觉——什么时候同尺寸或更弱的 teacher 反而更适合？是否只在 self-distillation / continual learning 场景成立？
- **多 teacher 路由**：博客的实验全是**单 teacher**——MiMo 多 teacher、V4 >10 个 teacher 在 reverse-KL 框架里的相互覆盖如何？博客在 §Distillation for personalization 末段把 V3.2 specialist distillation 提了一句但没深入。
- **forking tokens 与 entropy bonus 的关系**：博客观察 teacher 主要罚分叉 token；这与 RL 中 entropy regularization、high-entropy token 主导论的理论联系是什么？是否说明在 forking token 上加权 KL 能进一步提效？

## 相关页面

- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)（含数学依据节，跨家 OPD 共用骨架）
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)（5 家技术报告 OPD 用法分歧）
- [GLM-5 技术报告](glm-5.md) / [MiMo-V2-Flash 技术报告](mimo-v2-flash.md) / [Qwen3 技术报告](qwen3.md) / [Qwen3-VL 技术报告](qwen3-vl.md) / [DeepSeek-V4 技术报告](deepseek-v4.md)：直接引用或应用本博客算法的技术报告。
