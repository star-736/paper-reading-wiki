# Multi-Teacher On-Policy Distillation

> 想看 5 家技术报告里 OPD 的目的 / KL 形式 / pipeline 位置怎么分叉，请直接看 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)。本页以 MiMo MOPD 为骨架讲机制细节，但「数学依据」一节适用于所有走 on-policy distillation 路线的家族（MiMo / DeepSeek-V4 / Qwen3 / Qwen3-VL / GLM-5）。

## 一句话定义

Multi-Teacher On-Policy Distillation（MOPD）是 [MiMo-V2-Flash](../models/mimo-v2-flash.md) 的核心后训练范式：student 从自己的分布采样，再从多个 domain-specialized teacher 获得 token-level supervision，把多个专家能力合并进同一个模型。

## 为什么需要 MOPD

顺序后训练常见问题是 capability see-saw：数学 RL 提升后，写作或代码退化；代码 RL 提升后，安全或通用问答退化。简单混合数据或合并权重也不总能保留每个 teacher 的峰值能力。

MOPD 的目标是让 student 同时学习多个 teacher 的专门能力，并尽量避免单领域优化带来的互相覆盖。

## 三阶段流程

1. SFT：先建立基础 instruction-following 能力。
2. Domain-specialized training：分别训练专门 teacher，覆盖搜索、代码、工具使用、数学推理、通用推理和安全等任务。
3. MOPD：student on-policy 采样，并根据 prompt 所属领域，从对应 teacher 获得 token-level KL reward。

## 技术直觉

普通离线蒸馏让 student 学 teacher 生成的数据，但这些数据可能不在 student 当前分布上。MOPD 让 student 学自己真实会生成的轨迹，因此更接近部署时遇到的分布。Teacher 不直接给最终答案，而是通过 logits 或 token-level KL 提供细粒度训练信号。

报告中的 surrogate loss 可以理解为：如果某个 token 在 domain teacher 下概率更高，而在 student 下概率较低，这个 token 对 student 形成正向 advantage；反之则形成惩罚。MOPD 还可以与 ORM/GRPO 的 advantage 混合。

## 数学依据：OPD 为什么 work

> 这一节的论证对所有 on-policy distillation 都成立——MiMo MOPD、DeepSeek-V4 OPD、Qwen3 Strong-to-Weak、GLM-5 cross-stage 的 loss 形式都落在同一个数学框架下，只是 KL 估计方式和 teacher 来源不同。

### 第一层：OPD loss 在数学上等价于最小化 reverse-KL

MiMo 公式 7-8、GLM-5 公式 2、Qwen3 §4.5、V4 公式 29 表面上各不相同，但都可以归到同一个目标：

$$\min_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \; D_{KL}\!\left(\pi_\theta(\cdot|x, y_{<t}) \,\Big\|\, \pi_T(\cdot|x, y_{<t})\right)$$

推导用 likelihood-ratio gradient（log-derivative trick）。对 reverse-KL 求梯度时，由于采样分布 π_θ 本身含 θ，结果是：

$$\nabla_\theta D_{KL}(\pi_\theta \,\|\, \pi_T) \;=\; \mathbb{E}_{y \sim \pi_\theta}\!\left[\underbrace{\log\frac{\pi_T(y)}{\pi_\theta(y)}}_{\text{当 advantage}}\;\nabla_\theta \log \pi_\theta(y)\right] \;+\; \text{熵正则项}$$

主项的形式正好是 policy gradient `E[Â · ∇log π_θ]`，只要把 advantage 定义成

$$\hat{A} \;=\; \text{sg}\!\left[\log \frac{\pi_T(y\mid x, y_{<t})}{\pi_\theta(y\mid x, y_{<t})}\right]$$

塞回 GRPO/PPO 框架即可。这就是 **GLM-5 公式 (2) 和 MiMo 公式 (8) 的来源**——两家用一模一样的代换。所以 OPD 不是新算法，是「把 RL 的 advantage 换成 KL log-ratio」的 reparameterization，infra 完全复用 RL 训练栈。

V4 的 full-vocabulary KL 是同一目标的**精确版本**——它不简化成 token-level 标量估计、而是直接算整个词表上的 D_KL：

$$\mathcal{L}_{OPD}(\theta) \;=\; \sum_{i=1}^N w_i \cdot D_{KL}\!\big(\pi_\theta \,\|\, \pi_{E_i}\big) \quad \text{（V4 公式 29）}$$

代价是要在每个位置 materialize teacher 完整 logits，工程难度大；但**优化目标和 MiMo / GLM-5 严格一致**。

### 第二层：为什么 reverse-KL，不是 forward-KL

OPD 全家都选 reverse-KL `D_{KL}(\pi_\theta \,\|\, \pi_T)`，不是 forward-KL `D_{KL}(\pi_T \,\|\, \pi_\theta)`。这两个方向**不对称**，且对蒸馏行为影响巨大：

| | forward-KL `D(π_T ∥ π_θ)` | reverse-KL `D(π_θ ∥ π_T)` |
| --- | --- | --- |
| 惩罚什么 | π_θ 在 π_T 有质量的地方为 0 | π_θ 在 π_T 为 0 的地方有质量 |
| 行为 | **mass-covering / mode-covering**：student 试图覆盖 teacher 所有 mode，必要时把概率"摊薄"在 mode 之间 | **mode-seeking / zero-forcing**：student 在每个 mode 周围"做选择"，把质量集中到 teacher 认可的若干 mode 上 |
| 多 teacher 场景 | student 被迫做加权平均，容量不够时两头不讨好 | student 在 prompt 路由下贴向**对应**teacher 的 mode，不被迫加权 |

> Bishop *PRML* §10.1.2 用高斯混合等高线图给了直观证明；Minka 2005 *Divergence Measures and Message Passing* TR（MSR-TR-2005-173）把 forward/reverse KL 列在 α-divergence 谱系的两端。

对 GLM-5 这种「召回早期能力」的用法特别关键：reverse-KL 的 mode-seeking 性质让 student 能**强力把分布拉回**到 Reasoning_RL teacher 的某个 mode，而不是被强迫"同时覆盖 SFT + Reasoning_RL + General_RL 所有 mode 的并集"——后者在 student 容量受限时不可行。

### 第三层：on-policy 采样让梯度只作用在 student 实际访问的状态上

OPD 的 KL 期望是对 **student 自己**的分布取的：`E_{y ∼ π_θ}[...]`。这和离线蒸馏（off-policy distill，用 teacher 生成的数据训 student）有根本差异：

- **off-policy**：student 学「teacher 会怎么写」，但 teacher 写过的状态 student 自己可能根本走不到。训完仍有 exposure bias，部署时跑偏。
- **on-policy**：监督直接打在 student 部署时**真实会经过**的状态分布上，没有 distribution shift。

> 数学根据：Ross et al. 2011 *DAgger*（AISTATS）证明 imitation learning 里 on-policy 数据让 cumulative regret 是 O(T)（线性），off-policy 是 O(T²)（平方）。OPD 沿用同一个论证。这也是 Qwen3 Table 21 里 on-policy distill **pass@64 涨**（93.3 vs 起点 90.0）、而 RL pass@64 不动（90.0）的解释——on-policy distill 在 student 自己的轨迹空间上拓宽了概率质量分布；RL 只 sharpen 已有 mode。

### 第四层：teacher 固定 → 良定义的收敛目标

teacher 参数 freeze，π_T 是个不动的目标分布。reverse-KL 朝它收敛是一个**单调、有界、有明确最优**的优化问题：

$$D_{KL}(\pi_\theta \,\|\, \pi_T) \;\geq\; 0, \quad \text{下界 0 当且仅当 } \pi_\theta \equiv \pi_T$$

理论上 student 可以完全召回 teacher 的能力（容量足够、采样足够、step 足够）。这一点比 RL 干净得多——RL 的 reward landscape 可能有多个不一致的 mode、可能 reward hacking、没有干净的全局最优。所以 GLM-5 §3.5 才敢用 **"swiftly recover"** 这种自信措辞——OPD 的优化目标允许 swift，RL 的不允许。

### 把四层拼起来

OPD 能 work（特别是 GLM-5 那种「召回早期能力」用法）建立在四个数学事实的合力上：

1. **loss = reverse-KL** —— likelihood-ratio gradient 给出 advantage = log(π_T/π_θ) 的干净代换，infra 与 RL 共用。
2. **reverse-KL 的 mode-seeking 性质** —— student 在每个 mode 周围做选择，不被迫加权平均，容量利用率高。
3. **on-policy 消除 exposure bias** —— 梯度只作用在 student 部署时会访问的状态分布上，无 distribution shift（Ross 2011 的 O(T) vs O(T²) regret）。
4. **teacher 固定 → 良定义优化** —— reverse-KL ≥ 0 且 0 当且仅当分布相等，单调收敛，没有 RL 的 reward landscape 问题。

### 数学上没闭合的地方：多 teacher 混采

**这四条直接论证「OPD 能逼近一个固定 teacher」，但不直接论证「多 teacher 混采能同时召回多个能力而不互相覆盖」。**

多 teacher 实际上把目标退化成一个**条件混合分布**：

$$\pi^*(y\mid x) \;=\; \pi_{T_{k(x)}}(y\mid x), \quad \text{其中 } k(x) = \text{teacher 路由}(x)$$

如果不同 teacher 的"专长 prompt 集合"几乎不重叠（数学 prompt / 通用 prompt / 搜索 prompt 各走各的），student 在不相交的输入区域上分别拟合各 π_{T_k}，**互不干涉**，前述四层论证逐 teacher 成立。但如果 prompt 集**重叠**（比如某条 prompt 同时合理由 Reasoning_RL teacher 和 General_RL teacher 解），它只被分配给一个 teacher，另一个的视角就丢了——MiMo Table 7 里 BrowseComp（−6.8）、Arena-Hard Creative Writing（−3.9）落后 best teacher 的差距，很可能就是这种 routing 错位。

这是 OPD 在数学上**没**完全闭合的部分，也是为什么 GLM-5 §3.5（"mixed in appropriate proportions"）和 MiMo §4 都把 teacher 选择 / domain routing / 采样比例挂在嘴边——这一步外包给了数据 curation，不是公式能解决的。

## 报告中的结果

MiMo-V2-Flash 报告的 Table 7（MOPD 前后 student vs. best teacher 对比）：

| Benchmark | Student Before MOPD | Best Teacher | Student After MOPD | Δ(Student−Teacher) |
| --- | --- | --- | --- | --- |
| AIME 2025 | 89.3 | 93.9 (RL) | 94.1 | +0.2 |
| HMMT Feb. 2025 | 76.9 | 82.6 (RL) | 84.4 | +1.8 |
| LiveCodeBench | 77.5 | 82.6 (RL) | 83.2 | +0.6 |
| MMLU-Pro | 84.7 | 84.7 (Self) | 84.9 | +0.2 |
| GPQA-Diamond | 84.9 | 84.9 (Self) | 83.7 | −1.2 |
| HLE (w/o Tool) | 21.2 | 21.2 (Self) | 22.8 | +1.6 |
| Arena-Hard (Hard Prompt) | 50.0 | 50.0 (Self) | 54.1 | +4.1 |
| Arena-Hard (Creative Writing) | 90.1 | 90.1 (Self) | 86.2 | −3.9 |
| SWE-Bench Verified | 67.8 | 74.2 (RL) | 73.4 | −0.8 |
| Tau2-Bench | 75.9 | 79.6 (RL) | 80.3 | +0.7 |
| Tau2-Bench (Telecom) | 92.7 | 95.0 (RL) | 95.3 | +0.3 |
| BrowseComp | 42.5 | 51.7 (SFT) | 44.9 | −6.8 |

> Table 7（原文 §4.1）：Best Teacher 列括号标注其类型——RL（领域专项 RL teacher）、SFT（SFT teacher）、Self（student 自身即最强）。MOPD 后 student 在多数项上接近或超过 best teacher，但在 BrowseComp（−6.8）、Arena-Hard Creative Writing（−3.9）、GPQA-Diamond（−1.2）、SWE-Bench Verified（−0.8）上落后。

这说明 MOPD 不是自动合并所有峰值能力，teacher 选择、领域路由和 reward 设计仍然关键。

## 与 DeepSeek-V4 OPD 的关系

[DeepSeek-V4](../models/deepseek-v4.md) 也使用多 teacher OPD，但强调 full-vocabulary logit distillation。它认为 token-level KL 估计虽然省资源，但方差高、训练不稳定，因此通过 teacher hidden-state caching、按 teacher 排序调度和 TileLang kernel 来支持完整 logits 的 KL 计算。

## 待追问

- MOPD 的 domain routing 如何定义？粗粒度领域错误是否会导致负迁移？
- Teacher 数量增加时，student 容量是否足够保留所有能力？
- MOPD 与异步 agent RL 能否形成“先 RL 出 teacher，再 MOPD 融合，再继续 RL”的循环？

