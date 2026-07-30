---
type: Concept
title: "数据混合优化"
description: "LLM 数据混合优化的方法谱系：预训练 domain reweighting（启发式 → Group DRO/DoReMi → 回归/RegMix → bi-level/DoGE·TANDEM → per-capability/AutoMixer）用小 proxy model 预测大模型权重；SFT 阶段另有在线无 proxy 分支（DynamixSFT，Multi-Armed Bandit）。"
tags: ["concept", "data-mixture", "pretraining", "domain-reweighting"]
timestamp: 2026-07-11
---

# 数据混合优化

## 定义

数据混合优化（data mixture optimization / domain reweighting）指在 LLM 训练前或训练中，自动确定各数据 domain（如 Wikipedia / web text / code / books，或 SFT 阶段的各指令数据集）的采样比例（domain weights），使训练后的模型在广泛下游任务上表现更好。核心挑战是：domain 权重对模型性能影响巨大，但搜索空间随 domain 数量指数增长，直接在大模型上搜不现实。

本页主谱系是**预训练 domain reweighting**——共同范式是用小 proxy model 的训练信号预测大模型的最优 domain 权重，区别在于如何利用 proxy model 的信号。[DynamixSFT](../sources/dynamix-sft.md) 则是**SFT 阶段、在线、无 proxy** 的另一分支，直接在目标模型上用 Multi-Armed Bandit 调整，见下文专节。

## 方法谱系

| 方法 | 信号来源 | 优化方式 | 是否需下游任务 | 发表 |
|---|---|---|---|---|
| 启发式 (The Pile) | 人工经验 | 固定权重 | 否 | 2020 |
| 下游调优 (GLaM/PaLM) | 下游任务准确率 | 零阶搜索 / RL | 是 | 2021-2022 |
| [DoReMi](../sources/doremi.md) | excess loss (proxy vs reference) | Group DRO minimax | 否 | NeurIPS 2023 |
| [DoGE](https://arxiv.org/abs/2310.15393) | bi-level optimization 的梯度 | proxy model + 动态 reference | 否（可泛化到 OOD domain） | ICML 2024 |
| [RegMix](https://arxiv.org/abs/2407.01492) | 小模型 validation loss | 回归模型拟合 loss-mixture 曲面 | 否 | ICLR 2025 |
| MDE (Belenki et al.) | mixture-of-data-experts 近似 loss | 回归 + MDE 特征 | 否 | ACL 2025 |
| [TANDEM](https://arxiv.org/abs/2606.04401) | twin network 差值 (bi-level) | penalized single-level + twin | 否 | NeurIPS 2025 |
| [AutoMixer](../sources/laguna-m1-xs2.md) | ~60 个 0.5B MoE proxy 的 per-capability 下游指标 | Dirichlet 扰动 + 非线性回归器 + KL 正则 | 否（能力组代理 benchmark） | 2026（Poolside Laguna） |
| [DynamixSFT](../sources/dynamix-sft.md) | 目标模型自身 1-step look-ahead loss 下降（learning progress） | Multi-Armed Bandit + Prior-scaled Boltzmann（在线，无 proxy） | 否 | 2026（MSRA，**SFT 阶段**） |

## 跨报告信号

### DoReMi：开创性的 Group DRO 路线

[DoReMi](../sources/doremi.md) 是第一个证明「小 proxy model 的 loss 信号可以改善大模型训练效率」的方法。它把 domain reweighting 建模为 minimax 问题——最小化最坏情况 excess loss（proxy model 相对 reference model 的 loss 差）。Group DRO optimizer 交替更新 domain weights（exponentiated gradient ascent，upweight 高 excess loss 的 domain）和 proxy model weights。280M proxy 产出的 domain weights 让 8B 模型加速 2.6x，且不需下游任务知识。

关键洞察：excess loss 兼顾「domain 难度」和「模型可学习性」。只用 hardest domain（proxy loss 高）或只用 easiest domain（reference loss 低）都不如二者之差。最高/最低 entropy domain 可以 downweight：前者接近均匀分布不需多训，后者很快学完。

### DoGE：bi-level optimization 的梯度路线

DOGE（Domain Reweighting with Generalization Estimation，ICML 2024）同样用 proxy model，但把 domain reweighting 建模为显式的 bi-level optimization：内层优化 proxy model，外层优化 domain weights 使 proxy model 的泛化能力最大化。与 DoReMi 的 minimax 不同，DoGE 直接估计 domain 权重对泛化的影响，能处理 out-of-domain (OOD) 目标——即预训练 corpus 中不存在的 domain。在 SlimPajama 上比 DoReMi 和 baseline 更好。

### RegMix：回归预测路线

RegMix（ICLR 2025，Sea AI Lab）彻底改变了优化策略：训练 512 个 1M 参数的小模型，每个用不同的随机 domain 权重，然后用线性回归拟合「domain weights → validation loss」的曲面，在曲面上搜索最优 mixture，直接用于 1B 模型（1000x 放大）。关键发现：(1) web corpus 而非 Wikipedia 对下游性能正向影响最大；(2) domain 间存在复杂交互，常违反直觉。仅需 2% 额外 FLOPs，在 HellaSwag 上比人工选择提升 6.3%。

### TANDEM：twin network 的 bi-level 路线

[TANDEM](../sources/tandem.md)（NeurIPS 2025，JD.com + Oxford + 人大）把 bi-level optimization 简化为 single-level penalized form，用 twin network（proxy model + 动态 reference model）测量 domain 的边际收益。关键创新是每个 episode 同步初始化 twin model（$w_0 = u_0$），然后各自 probing $K$ 步，用 loss 差更新 domain weights。理论上比 DoReMi 更通用（$O(T^{-1/4})$ 收敛保证），还能处理 data-restricted 和 SFT 场景。

TANDEM 的核心洞察是：数据充足时 uniform weighting 已近最优（Proposition 1：generalization gap 趋零时 uniform 是 bi-level 问题的有效解），reweighting 的价值在 data-restricted（小 domain 过拟合）和 SFT（样本多次访问导致泛化 gap）场景才凸显。在 data-restricted 实验中，DoReMi 反而恶化（36.91 vs Uniform 31.53），TANDEM 显著改善（28.07）。

### AutoMixer：产业落地的 per-capability 回归路线

[AutoMixer](../sources/laguna-m1-xs2.md)（Poolside Laguna，2026）把这套范式用到了 33B 生产 MoE 的 30T 预训练上——是已收录报告中**唯一在 frontier-scale 生产 MoE 上验证的数据混合优化**。它训了一群 ~60 个 ~0.5B MoE proxy，每个在不同混合上训 ~60B tokens，覆盖 50+ 异构数据集组，学 surrogate 映射 $\mathcal{M}: x\to y$（混合向量 $x$ → $k$ 个能力组下游指标 $y$）。

与 DoReMi/DoGE/RegMix/TANDEM 的关键区别在三点：

1. **采样 + 约束**：候选混合按 $x\sim\text{Dirichlet}(\alpha x_0)$ 采样并约束 $\|x-x_0\|_1<\epsilon$，优化目标再加 KL 正则 $\lambda D_{KL}(x\|x_0)$ 把解拉回人工先验——避免漂向少数主导源。这比 DoReMi 的 minimax / TANDEM 的 bi-level 更保守，本质是「在人工先验邻域内搜最优」。
2. **per-capability 独立回归器**：每个能力组（coding / math reasoning / STEM knowledge / commonsense / general knowledge）单独拟合 $f_j(x)\approx y_j$（实践用非线性），再在 $\max_x\sum_j w_j f_j(x)$ 下显式恢复能力间 trade-off。效果上优化目标 HumanEval+ +43%、GSM8K +41%，但 commonsense 任务略退（ARC-C -6.8%）——这种**负相关被直接建模并接受**，是 per-capability 回归器相比单一 loss 标量的优势。
3. **held-out 泛化**：优化未见的 MATH +25%、LiveCodeBench +39%、APTBench-4k +35%，说明 surrogate 学到的不是过拟合。

AutoMixer 也补上了「这些方法在 MoE 上是否有效」的待追问——DoReMi/TANDEM 的实验都基于 dense model，AutoMixer 直接在 MoE proxy + MoE 生产模型上验证有效。代价是 ~60 个 proxy 的总训练成本未披露，KL 系数 $\lambda$ 的 sensitivity 也未给。

### DynamixSFT：SFT 阶段的在线无 proxy 分支

[DynamixSFT](../sources/dynamix-sft.md)（Microsoft Research Asia + UMich + KAIST，2026，arXiv:2508.12116）是与上述 proxy-model 谱系**范式分叉**的另一分支：不用小 proxy model 预测大模型权重，而是**直接在目标模型上、训练过程中在线调整**数据集采样比例。它针对 SFT 阶段（指令微调），把「从 K 个候选数据集采一个 batch」建模为 Multi-Armed Bandit，每个数据集 = 一个 arm。

两个关键设计（均为原文确证）：

1. **Prior-scaled Boltzmann Exploration**：在标准 Boltzmann 采样概率 $\propto\exp(\beta Q_k)$ 上乘原始数据集比例 $p^{(0)}$ 作先验，软锚定采样分布到原始比例（保留 LIMA 小而精、FLAN 大而广这类固有定位），再加 $\gamma/K$ minimum floor 防 never-sampling。这与 AutoMixer 的 KL 正则 $\lambda D_{KL}(x\|x_0)$「把解拉回人工先验」思路同源——都是承认原始混合含人工先验知识、不应被 reward 推得太远——但 DynamixSFT 是乘性先验（直接进采样概率），AutoMixer 是加性惩罚（进优化目标）。
2. **1-Step Look-ahead Reward**：每 $T_{\text{update}}$ 步对每个数据集做一次虚拟单步更新，用 $r=(L_{\text{pre}}-L_{\text{post}})/(L_{\text{pre}}+\epsilon)$ 作 reward，min-max 归一化 + EMA 平滑。理论上一阶 Taylor 展开 $r\approx\eta\|\nabla L\|^2$，所以最大化 reward = 最大化梯度下降幅度 = 衡量 learning progress 而非 raw loss；又由 telescoping，累积 reward = $L(\theta_0)-L(\theta_{T+1})$，故 MAB 优化累积 reward 等价于优化长期收敛——非启发式 trick。

与 proxy-model 谱系的对比要点：

- **信号来源**：DoReMi/RegMix 用 proxy 的 excess loss / validation loss，TANDEM 用 proxy 的 bi-level 梯度，AutoMixer 用 proxy 的 per-capability 下游指标；DynamixSFT 用**目标模型自身**的 1-step loss 下降。前者靠 proxy 放大省成本，后者靠「reward 只需前向」省成本——但论文内部对 reward 是否真做了梯度更新存在矛盾（见来源页待追问），需谨慎。
- **开销**：DynamixSFT 仅 +12.7%（1.13×），低于 HBO（+139%）、MultiUAT（+380%）、MultiDDS（+760%），也低于 DoReMi 的 8% 额外 FLOPs 量级（DoReMi 是预训练，绝对量级不同，仅作相对参考）。轻量是它相对其他**动态**方法的主要卖点。
- **阶段**：本谱系其余方法主攻预训练 domain reweighting，DynamixSFT 专注 SFT；SFT 数据集异构性更强（LIMA 1K vs FLAN 50K+，尺度差 50×），prior-scaling 的锚定收益在此更显著（消融：去 prior 后 AVG 27.7→25.9，即便按比例初始化仍只有 26.3）。
- **粒度**：仍是 dataset-level，与 [Qwen3](../sources/qwen3.md) 的 instance-level 标注混合、[MinerU2.5-Pro](../sources/mineru-2-5-pro.md) 的 instance-level 难度采样是不同粒度维度。

DynamixSFT 缺少在 frontier-scale 模型上的验证（实验只到 8B），无法判断「在线无 proxy」与「proxy 放大」哪种范式在极大尺度下更优——这是它留给本谱系的开放问题。

### 产业实践：instance-level 超越 domain-level

[Qwen3](../sources/qwen3.md) 的预训练数据管线采用了 instance-level data mixture：用轻量 Qwen 标注器给 >30T tokens 在「教育价值 / 领域 / 安全」多维度打标，按 instance 而非 domain 优化数据混合。这是 DoReMi 等方法在 domain 定义粗粒度局限上的自然演进方向——DoReMi 论文自己也指出「更细粒度的 domain 可能带来更大增益」。

### data-centric AI 的另一分支：难度感知采样 + 标注精修

[MinerU2.5-Pro](../sources/mineru-2-5-pro.md) 代表 data-centric AI 在文档解析域的另一条路线，与上述 domain reweighting 谱系正交。它不改 domain 权重，而是协同优化三个维度：**覆盖**（DDAS：ViT embedding + K-Means 聚类 + 难度加权采样纠正长尾偏移）、**信息量**（CMCV：三个异构模型输出共识判定 Easy/Medium/Hard，Medium 训练价值最高）、**标注准确性**（Judge-and-Refine render-then-verify + 专家标注精修 Hard 样本）。

与 DoReMi/TANDEM 的关键区别：(1) 优化目标不是「各 domain 采样比例」而是「哪些 instance 值得训 + 标注可信度」；(2) 信号来源不是 proxy model 的 loss 而是多模型输出一致性（query-by-committee 思想）；(3) 把标注质量作为一等问题（Hard 样本标注噪声会传播进模型），而非假设标注已给定。这条路线适用于「标注本身不可靠」的场景（文档解析、结构化抽取），而 domain reweighting 假设标注固定、只调比例。MinerU2.5-Pro 用此方法固定 1.2B 架构把 OmniDocBench v1.6 从 92.98 推到 95.69，是「架构成熟后数据工程是主要杠杆」论点在文档解析域的干净实证。

#### IMIC → CMCV：从单模型内省到多模型交叉验证

MinerU2.5-Pro 的 CMCV 直接改进自 [MinerU2.5](../sources/mineru-2-5.md) 的 IMIC（Iterative Mining via Inference Consistency）。IMIC 的核心是利用**单模型多次随机推理**的一致性挖 hard case：对一样本用 MinerU2.5 Stage-1 checkpoint 推理 n 次，算输出间配对一致性（Layout: PageIoU, Formula: CDM, Table: TEDS），低一致性 = hard case = 送人工标注。阈值：PageIoU <0.8 hard / >0.9 easy，CDM <0.3 / >0.7，TEDS <0.6 / >0.9。

CMCV 的改进动机是 IMIC 的根本局限：**单模型内省只能捕获该模型的认知不确定性，无法区分「模型特定盲点」与「普遍难题」**。模型特定盲点（某模型独有缺陷）可通过跨模型共识直接修正；普遍难题（所有模型都失败）需额外质量精修或人工干预。IMIC 把两者混为一谈都送人工，而 CMCV 用三异构模型（MinerU2.5 + PaddleOCR-VL + Qwen3-VL-30B）交叉验证，按「待改进模型相对外部的表现」分三档：Easy（与外部一致，任一输出可作标注）、Medium（外部一致但待改进模型不同，外部共识作 pseudo-label，**训练价值最高**）、Hard（三模型全分歧，须 Judge-and-Refine 或专家）。Medium 的精确定位 + 可学证明 + 可靠标注是 IMIC 单模型内省无法提供的。这是 query-by-committee 思想在文档解析 hard case 挖掘上的具体演进。

## 为什么重要

数据混合优化是「data-centric AI」在 LLM 预训练中的核心议题。模型架构和训练算法趋同后，数据组成成为决定模型能力的主要变量。DoReMi 等方法把数据配比从人工经验驱动变成可自动优化的工程问题，且优化成本远低于训练本身（DoReMi 仅 8% 额外 FLOPs）。这对降低训练成本、提升数据效率有直接价值。

## 待追问

- DoReMi 的 domain weights 跨尺度迁移机制仍不清楚——280M 产出为何能用于 8B？TANDEM 的 bi-level 理论是否给出更严格的迁移保证？
- RegMix 发现 web corpus 比高质量 Wikipedia 更重要，这与 DoReMi 大幅 upweight Pile-CC 一致——但为什么？
- instance-level mixture（Qwen3）与 domain-level reweighting（DoReMi/RegMix）的效果对比尚无公开 benchmark。
- AutoMixer 已在 MoE proxy + MoE 生产模型上验证有效（此前 DoReMi/TANDEM 仅 dense），但 ~60 个 proxy 的总成本与 KL 系数 $\lambda$ 的 sensitivity 未披露；per-capability 回归器在能力间强负相关时是否会陷入零和解？
- DynamixSFT 的「在线无 proxy」与 DoReMi/RegMix/AutoMixer 的「proxy 放大」在 frontier-scale 下孰优尚无对比（DynamixSFT 实验仅到 8B）；其 reward 计算在 §4.2 / Algorithm 1（virtual gradient step，需反向传播）与 §8（solely forward passes, no backward）之间存在描述矛盾，+12.7% 开销推导依赖后者，需作者澄清实现究竟用了 forward-only 近似还是真做了梯度更新。

## 相关页面

- [DoReMi](../sources/doremi.md) — Group DRO 路线的开创性工作
- [TANDEM](../sources/tandem.md) — Bi-level optimization + twin network 路线，DoReMi 的直接改进
- [AutoMixer / Laguna 技术报告](../sources/laguna-m1-xs2.md) — per-capability 回归 + Dirichlet + KL 正则，frontier-scale MoE 产业落地
- [DynamixSFT 技术报告](../sources/dynamix-sft.md) — SFT 阶段在线无 proxy 的 Multi-Armed Bandit 分支，与 proxy-model 谱系范式分叉
- [Qwen3 技术报告](../sources/qwen3.md) — instance-level data mixture 的产业实践
