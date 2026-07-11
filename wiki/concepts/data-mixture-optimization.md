---
type: Concept
title: "数据混合优化"
description: "LLM 预训练数据混合优化的方法论谱系：从启发式 → Group DRO (DoReMi) → 回归预测 (RegMix) → bi-level optimization (DoGE/TANDEM)，核心都是用小模型预测大模型的最优 domain 权重。"
tags: ["concept", "data-mixture", "pretraining", "domain-reweighting"]
timestamp: 2026-07-11
---

# 数据混合优化

## 定义

数据混合优化（data mixture optimization / domain reweighting）指在 LLM 预训练前，自动确定各数据 domain（如 Wikipedia / web text / code / books）的采样比例（domain weights），使预训练后的模型在广泛下游任务上表现更好。核心挑战是：domain 权重对模型性能影响巨大，但搜索空间随 domain 数量指数增长，直接在大模型上搜不现实。

所有主流方法的共同范式是：用小 proxy model 的训练信号预测大模型的最优 domain 权重，区别在于如何利用 proxy model 的信号。

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

### 产业实践：instance-level 超越 domain-level

[Qwen3](../sources/qwen3.md) 的预训练数据管线采用了 instance-level data mixture：用轻量 Qwen 标注器给 >30T tokens 在「教育价值 / 领域 / 安全」多维度打标，按 instance 而非 domain 优化数据混合。这是 DoReMi 等方法在 domain 定义粗粒度局限上的自然演进方向——DoReMi 论文自己也指出「更细粒度的 domain 可能带来更大增益」。

## 为什么重要

数据混合优化是「data-centric AI」在 LLM 预训练中的核心议题。模型架构和训练算法趋同后，数据组成成为决定模型能力的主要变量。DoReMi 等方法把数据配比从人工经验驱动变成可自动优化的工程问题，且优化成本远低于训练本身（DoReMi 仅 8% 额外 FLOPs）。这对降低训练成本、提升数据效率有直接价值。

## 待追问

- DoReMi 的 domain weights 跨尺度迁移机制仍不清楚——280M 产出为何能用于 8B？TANDEM 的 bi-level 理论是否给出更严格的迁移保证？
- RegMix 发现 web corpus 比高质量 Wikipedia 更重要，这与 DoReMi 大幅 upweight Pile-CC 一致——但为什么？
- instance-level mixture（Qwen3）与 domain-level reweighting（DoReMi/RegMix）的效果对比尚无公开 benchmark。
- 这些方法在 MoE 模型上是否同样有效？DoReMi/TANDEM 的实验都基于 dense model。

## 相关页面

- [DoReMi](../sources/doremi.md) — Group DRO 路线的开创性工作
- [TANDEM](../sources/tandem.md) — Bi-level optimization + twin network 路线，DoReMi 的直接改进
- [Qwen3 技术报告](../sources/qwen3.md) — instance-level data mixture 的产业实践
