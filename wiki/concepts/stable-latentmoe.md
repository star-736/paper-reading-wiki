---
type: Concept
title: "Stable LatentMoE"
description: "LatentMoE + Normalized + SiTU-GLU + Quantile Balancing：把 routed expert 解耦到 latent 空间支持 896-expert/16-active 极端稀疏，三组件稳定 2.8T 规模训练。"
tags: ["concept", "stable-latentmoe", "latentmoe", "moe", "situ-glu", "quantile-balancing"]
timestamp: 2026-07-30
---

# Stable LatentMoE

## 定义

**Stable LatentMoE** 是 Kimi K3 的 channel-mixing（宽度维）机制，由四个部分组成：**LatentMoE**（结构）+ **Normalized LatentMoE**（RMSNorm）+ **SiTU-GLU**（bounded activation）+ **Quantile Balancing**（负载均衡）。目标是支持 896 routed experts / 16 active / 2 shared（sparsity 56）的极端稀疏 MoE 在 2.8T 规模下稳定训练。

来源：[Kimi K3 技术报告](../sources/kimi-k3.md) § 2.3。LatentMoE 概念引自 [32]；Normalized / SiTU-GLU / QB 是 K3 的贡献。

## 为什么需要：极端稀疏放大的两个失效模式

常规 MoE 每个 selected expert 收完整 `d` 维 token 表示，所以通信和 expert-weight 流量随 routing multiplicity 增长。**LatentMoE** 把 routed expert 操作从全模型宽度 `d` 解耦到紧凑 latent 空间 `ℓ`（K3 取 `ℓ = d/2 = 3584`），shared expert 仍走全宽——这让 expert pool 扩到 896 / 16 active 不会让通信爆炸。

但 sparsity 56 放大了 vanilla LatentMoE 的两个失效模式（K3 § 2.3 开篇）：

1. **激活爆炸**：routed 路径是 `W↓ → gated multi-branch FFN → W↑` 近四个连续矩阵乘的 ill-conditioned 链，配 2.8T 规模，内部激活爆炸。
2. **负载失衡**：平衡 ~10³ 个 expert 的负载超出 aux-loss-free bias 更新的适用域——原版 `b ← b + γ·sign(target - load)` 的固定步长 `γ` 在 896 expert 下要么适应太慢、要么负载振荡，imbalanced routing 拖慢 expert-parallel 训练、部分 expert 训练不足。

Stable LatentMoE 三组件分别应对：Normalized + SiTU-GLU 压激活爆炸，QB 解负载失衡。

## 机制（已据 Kimi K3 原文 § 2.3 + 附录 B/C/D 核实）

### 基础 LatentMoE 结构（Eq. 11）

沿用 DeepSeekMoE 的 shared + routed expert 组织。对 `x ∈ R^d`：

```
z = W↓x ∈ R^ℓ                                    # 下投影到 latent 空间
u = Σ_{i∈T_k(x)} p_i · E_routed_i(W↓x)          # routed experts 在 latent 空间操作
y = Σ_{j=1}^{N_s} E_shared_j(x) + W↑ · RMSNorm(u)  # shared 全宽 + routed 上投影回 R^d
```

- `E_shared_j: R^d → R^d`（全宽 shared expert，K3 固定 `N_s = 2`）
- `E_routed_i: R^ℓ → R^ℓ`（latent 空间 routed expert）
- `p_i` = router weight（由 QB 决定，见下）
- `T_k(x)` = Top-k selected experts（K3: k=16 of 896）

### 组件 1：Normalized LatentMoE（§ 2.3.1）

原版 LatentMoE 直接 `W↑ · u`，`u` 的 scale 随 selected experts 和 routing weights 变。K3 在 expert 聚合后、`W↑` 前插入 **RMSNorm**：`y = ... + W↑ · RMSNorm(u)`。

降低 routed branch 对 scale variation 的敏感度，再与全宽 shared branch 合并。除稳定训练外，**持续改善 validation loss 和下游 benchmark**（K3 报告称"consistently improves"，但未给具体 ablation 数字，见待追问）。

### 组件 2：SiTU-GLU（§ 2.3.2 + 附录 B）

**背景**：SwiGLU 的两个乘性因子（`x · Sigmoid(x)` 的 Swish gate 与 up branch 的 `W_u x`）都无界，大坐标重合产生 activation outlier，低精度算术下溢出风险增。原版 GLU 的 sigmoid 门避免无界门增长，但不保留 Swish 近原点线性正区。

**SiTU-GLU（Sigmoid Tanh Unit GLU）** 对 Swish 的线性因子和 up branch 各加 smooth cap `β·tanh(x/β)`：

```
SiTU-GLU(x) = [β1 · tanh(W_g x / β1) ⊙ Sigmoid(W_g x)] ⊙ [β2 · tanh(W_u x / β2)]
```

- **原点附近一阶匹配 SwiGLU**：`β·tanh(z/β) = z + O(z³/β²)`，所以 `β1, β2 → ∞` 时 pointwise 收敛到 SwiGLU。
- **有界输出**：`|tanh| < 1` 且 `0 < Sigmoid < 1`，每坐标 `‖SiTU-GLU(x)‖_∞ ≤ β1·β2`。K3 取 `β1 = 4, β2 = 25`，bound = 100。
- **vs hard clamping**：smooth cap 在饱和区保留非零梯度，K3 实测训练表现更好。

K3 设 `β1=4`（gate branch）、`β2=25`（up branch）。β1·β2=100 的 bound 与 FP8 dynamic range 的关系报告未明说（见待追问）。

### 组件 3：Quantile Balancing（§ 2.3.3 + 附录 C/D）

**背景**：K3 用 aux-loss-free routing（[30]）——加 expert-specific bias `b_j` 到 router score 做 Top-k 选择，但 `b_j` 不进 mixture weights `p_i`（只调 dispatch 不改 gradient-based router 优化）。原版更新 `b_{j}^{t+1} = b_j^t + γ·sign(target - load_j^t)`，`γ` 在适应慢与负载振荡间 trade-off。896 expert 下失效。

**QB 推导**（附录 C，balanced assignment 对偶 LP 的 exact coordinate minimizer）：

考虑 `m` tokens 路由到 `n` experts、每 token 选 `k` 个，target load `q = mk/n` per expert。最大分数 balanced assignment（每 expert 恰服务 `mk/n` tokens）：

```
max Σ_{i,j} x_{i,j} s_{i,j}  s.t.  Σ_j x_{i,j} = k,  Σ_i x_{i,j} = mk/n
```

线性松弛 + 对偶：引入 token 侧乘子 `α_i`、expert 侧乘子 `β_j`，对偶目标：

```
min L(α, β) := Σ_{i,j} max(0, s_{i,j} - α_i - β_j) + k Σ_i α_i + (mk/n) Σ_j β_j
```

交替坐标最小化，**每个子问题有闭式解**：

- `α_i` 固定 `β`：`α_i* = quantile_{1-k/n}(s_i - β)`（token 侧分位数）
- `β_j` 固定 `α`：`β_j* = quantile_{1-k/n}(s_{:,j} - α)`（expert 侧分位数）

两更新都是沿 token / expert 轴的同一分位数——方法得名。**routing 只需 expert thresholds `β ∈ R^n`**（等价于 bias `b = -β`），token thresholds `α ∈ R^m` 是动态训练 batch 的中间变量丢弃。

**实际用的 update（Eq. 14）**：用 Top-(k+1) 路由拿 cutoff `α_i^{(t)}`（第 k+1 大的 biased score），再设 `b_j` 使 expert `j` 收 target load：

```
b̃_j^{(t+1)} ← -quantile_{1-k/n}(s_{:,j} - α^{(t)})
b^{(t+1)} ← b̃^{(t+1)} - mean(b̃^{(t+1)}) · 1
```

margins `s_{i,j} - α_i^{(t)}` 减掉 biased cutoff，旧 bias 只通过 cutoff 进入更新；第二行去掉公共 offset（不影响 Top-k 选择）。**因果性**：更新下一 step 才生效，batch 永远不用从自己算的 bias 路由。**推理时 bias 冻结**，无需分位数计算。

**与 sign-based loss-free 的关系**：expert 侧子问题的 (sub)gradient `∂L/∂β_j = mk/n - Σ_i χ(s_{i,j} - α_i - β_j > 0)` = target load - observed load。SignSGD 一步 recover 原版 sign update（`b = -β` 符号约定）。**QB 是直接跳到同一对偶目标的 exact coordinate minimizer**，所以无学习率超参、几个 step 内平衡 ~10³ expert。vs BIP（同 assignment 但不等式约束 `≤`，非负性给 `α, β` 加 `max(0, ·)` clipping，只能抑过选不能促欠选，平衡明显更慢）。

**Histogram estimation（附录 D）**：规模下分位数跨全 global batch（millions of tokens sharded across ranks + accumulation steps），gather 全量算 exact quantile 不可行。改用 **histogram of margins** 估分位数：一次 all-reduce 汇 per-rank bin counts，从 pooled counts 恢复分位数。counts 可加，histogram 表示 pooled global batch（不管 token 怎么 shard），误差仅 bin width 级，通信成本几百 bin/expert。

![Figure 5：Quantile Balancing 示意（m=8 tokens, n=4 experts, k=1 selected/token）。(a) Token-wise Top-k 产生不均衡负载 (4,3,1,0)——某些 expert 过热、某些 dying。(b) 每条灰条是当前 biased score 的 margin `s_{i,j} + b_j - α_i`，行最大值复现 (a) 的路由；红虚线放在第 (q+1) 大的 margin 处（q=2），使恰好 q=2 个 margin 超过它。(c) 减去列调整后保留的选择给出均衡负载 (2,2,2,2)，红边是 QB 改变的分配。](../assets/kimi-k3/fig5-quantile-balancing.png)

> Figure 5（原文截图，§ 2.3.3 Quantile Balancing）："Illustration of Quantile Balancing with m = 8 tokens, n = 4 routed experts, and k = 1 selected expert per token. (a) Token-wise Top-k routing … produces loads (4, 3, 1, 0) … (c) The retained choices yield the balanced load (2, 2, 2, 2); red edges denote assignments changed by QB."

## 跨报告信号

- **[Kimi K3](../sources/kimi-k3.md)（Moonshot AI，2026-07）**：Stable LatentMoE 的提出者与首个生产采用者。896 routed / 16 active / 2 shared（sparsity 56），latent dim 3584（0.5× hidden 7168）。配 MoonEP 完美平衡 EP（§ 5.2.1，E/R 冗余 expert bound 证明 tight）做 3T 级预训练。
- **[DeepSeek-V2](../sources/deepseek-v2.md) / V3 / V4**：DeepSeek 系 MoE 走的是 shared + routed 但**不走 latent 压缩**——routed expert 在全宽操作。K3 的 LatentMoE（routed 在 ℓ=d/2 空间）是另一条路：牺牲 routed 表达力换通信可承受，靠 shared expert 全宽路径补通用变换。V4-Pro 1.6T / 49B active（384 routed + 1 shared, 6 active）vs K3 2.8T / 104B（896 routed + 2 shared, 16 active）——K3 expert 更多更稀疏，LatentMoE 是支撑这个稀疏度的结构前提。
- **[Kimi K2.5](../sources/kimi-k2.5.md)（前作）**：K2/K2.5 用 384 routed / 8 active / 1 shared 的常规 MoE（无 latent 压缩），SwiGLU + 常规 aux-loss-free bias。K3 把 expert 数翻 2.3×、active 翻 2×、加 latent 压缩 + SiTU-GLU + QB，是同族内的结构性升级。
- **[Ling-2.6](../sources/ling-2.6.md)**：256 routed + 1 shared, 8 active，fine-grained MoE（expert intermediate=2048），前 4 层 dense FFN。规模与稀疏度都远低于 K3，未走 latent 压缩路线，也未提激活爆炸或负载失衡的专门解法——说明这两类问题在 ~1T / 256-expert 规模尚未尖锐到需要 Stable LatentMoE 级别的工程。
- **SiTU-GLU 与 GLU/SwiGLU 家族**：SwiGLU 是当前 LLM FFN 主流（K2/Qwen3/DeepSeek 系都用）。SiTU-GLU 是首个为 3T 规模 + 极端稀疏 MoE 设计的 bounded 变体。与 hard clamping 的区别（smooth cap 保留饱和区梯度）是 K3 报告强调的训练优势点，但缺跨模型的对比 ablation。
- **QB 与负载均衡方法谱系**：aux-loss-based（[33]，加额外 loss 项）、aux-loss-free sign update（[30]，K2/Qwen3 用）、ECHO/UltraEP（预设冗余数或 per-rank token cap，可能无解）、BIP（同 assignment 不等式约束，慢）。QB 的定位是 **aux-loss-free 的 exact 解**——不引入额外 loss、无学习率、几步平衡，是对 sign update 在 10³ expert 规模的直接升级。

## 为什么重要

- **它把开源 MoE 推过 3T 门槛**。2.8T 是开源模型首次达到 3T 级，而支撑这个规模的关键不是单纯加 expert，而是让 896-expert/16-active 的极端稀疏**可训练**——LatentMoE 压通信、Normalized + SiTU-GLU 压激活爆炸、QB 解负载失衡，三者缺一不可。
- **QB 是 aux-loss-free 路由的算法升级**。原版 sign update 在 256-384 expert（K2/Qwen3 规模）够用，到 896 失效。QB 把"调 bias"从 sign-SGD 升级成对偶 LP 的 exact coordinate minimizer，无学习率、几步收敛——这个思路对任何走 aux-loss-free + 大 expert pool 的后续模型都适用。
- **SiTU-GLU 是 bounded activation 的工程化**。SwiGLU 无界在低精度（FP8/MXFP4）下是 outlier 源，K3 的 QAT 全程 MXFP4 让这个问题更尖锐。SiTU-GLU 给了一个"原点附近等价 SwiGLU + 大值有界 + 饱和区有梯度"的解，与 K3 的 deployment-aware QAT 路线配套。

## 待追问

- **Normalized LatentMoE 的 ablation 数字**。报告说"consistently improves validation loss and downstream benchmarks"但未给具体数字。RMSNorm 带来多少 loss 改善、哪些 benchmark 涨？
- **SiTU-GLU 的 β1=4 / β2=25 选择依据**。报告给值但未给选择方法。β1·β2=100 的 bound 是否对应某 activation outlier 阈值？与 MXFP4 dynamic range 的关系？SwiGLU（β→∞）作为 ablation 基线的 loss 差距？
- **QB vs sign update 的收敛速度对比**。报告说 QB "equilibrates within a few update steps even for nearly 10³ experts"，但未给 sign update 在 896 expert 下的失败模式量化（振荡幅度、dying expert 比例）。这个对比是 QB 必要性的核心证据。
- **QB 的 train-inference consistency 细节**。推理时 bias 冻结，但训练时 bias 每 step 更新——bias 在训练后期的稳定性如何？是否会因最后几个 step 的 batch distribution 摆动影响最终 bias？报告说"final bias is frozen at inference"但未说从哪个 step 冻结。
- **LatentMoE 的表达力损失**。routed expert 在 ℓ=d/2 空间操作，相对全宽 expert 牺牲多少表达力？K3 靠 2 个全宽 shared expert 补，但 shared 占比（2 / (2+16) = 11%）是否够？与 DeepSeek 系全宽 routed expert 的 head-to-head 质量对比缺（不同模型规模、不同训练数据，不可直接比）。
- **MoonEP 的 E/R bound 与 QB 的关系**。MoonEP 保证 EP rank 间负载平衡（每 rank 恰收 S×K token），QB 保证 expert 间负载平衡（每 expert 收 mk/n token）。两者作用层级不同（EP rank vs expert），但都追求"完美平衡"。它们是否耦合——QB 的 balanced routing 是否让 MoonEP 的 redundant expert planning 更容易？

## 相关页面

- 来源：[Kimi K3 技术报告](../sources/kimi-k3.md)（§ 2.3 Stable LatentMoE + 附录 B/C/D）
- 模型：[Kimi K3](../models/kimi-k3.md)
- [Attention Residuals](attention-residuals.md)（K3 深度维机制，与 Stable LatentMoE 正交）
- [MoE 前沿模型扩展](moe-frontier-model-scaling.md)（K3 2.8T 是当前开源最大）
- [线性注意力与 delta rule](linear-attention-and-delta-rule.md)（K3 序列维机制）
- 模型对比：[DeepSeek-V4](../models/deepseek-v4.md)（全宽 MoE 路线）、[Kimi K2.5](../models/kimi-k2.5.md)（前作，常规 MoE）、[Ling-2.6](../models/ling-2.6.md)（256-expert 规模）
