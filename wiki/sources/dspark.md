---
type: Source
title: "DSpark 技术报告"
description: "DSpark 的 PKU + DeepSeek-AI 论文：semi-AR drafter（parallel backbone + 轻量 sequential head）+ confidence-scheduled verification，部署进 DeepSeek-V4 生产环境替代 MTP-1，per-user 生成速度 +60–85%。"
tags: ["source", "dspark", "speculative-decoding", "serving", "deepseek"]
timestamp: 2026-06-29
resource: "../../raw/DSpark_paper.pdf"
---

# DSpark 技术报告

## 来源

- 原始 PDF：[raw/DSpark_paper.pdf](../../raw/DSpark_paper.pdf)
- 标题：DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
- 团队：Peking University + DeepSeek-AI（Xin Cheng / Xingkai Yu / Chenze Shao / Jiashi Li / Yunfan Xiong 等 5 人共同一作）
- 开源：DSpark checkpoints（对应 DeepSeek-V4-Flash preview / DeepSeek-V4-Pro preview）+ 训练库 **DeepSpec**（包含 Eagle3、DFlash、DSpark）。
- 部署 context：DeepSeek-V4 preview 发布两周后，DSpark 在生产 serving 引擎里**替换掉了 MTP-1**（即 V3/V3.2 起一直在用的 single-token MTP）。

## 核心结论

DSpark 是 speculative decoding 框架，同时治两个 bottleneck：

1. **草稿质量**——纯 parallel drafter（如 DFlash）一次前向出整块 token，但因为各位置独立预测、无 intra-block 依赖，会有 multi-modal collision、suffix decay；fully autoregressive drafter（如 Eagle3）能维持后段质量但 $T_\text{draft}\propto\gamma$，只能用浅而短的 draft。DSpark 走 **semi-autoregressive**：保留 DFlash 风格的重型 parallel backbone（吃掉 $T_\text{draft}$），在其上贴一个**轻量 sequential head**（Markov / RNN）只注入 intra-block 转移信息——结果在 Qwen3-4B/8B/14B 上 macro-average accepted length 比 Eagle3 提升 26.7%–30.9%，比 DFlash 提升 16.3%–18.4%。
2. **验证浪费**——长 draft block 不等于真加速。在高并发服务里，验证迟早被拒绝的尾部 token 会挤占 target model 的 batch 容量；DSpark 用一个 **confidence head**（per-position 估计 prefix 存活概率，并用 Sequential Temperature Scaling 校准）+ **hardware-aware prefix scheduler**（根据预先 profile 的 $\text{SPS}(B)$ 曲线动态截断 verification 长度），把"该验多长"做成全局吞吐最大化问题。在 DeepSeek-V4 生产 traffic 上，对比 MTP-1 baseline，每用户生成速度 V4-Flash +60–85%、V4-Pro +57–78%；在原本 MTP-1 几乎服务不了的严格交互档（V4-Flash 120 tok/s/user、V4-Pro 50 tok/s/user）DSpark 把 Pareto 前沿外推一档。

![DSpark Figure 1：架构与解码循环。给定 prompt ABC，target model 先出 anchor token D；以 D 为输入，DSpark 用重型 parallel backbone 一次生出 E F G H 的 base logits，再过 Sequential Block 注入 intra-block 转移，同时 Confidence Head 给出 c1–c4；Hardware-Aware Prefix Scheduler 根据 cumulative survival probability 与 profile 出的 SPS(B) 曲线决定保留 EFG、丢掉 H；target model 并行验证 scheduled prefix，接受 E F、拒绝 G 并补上 G\*，完成本轮。](../assets/dspark/fig1-dspark-architecture.png)

> Figure 1（原文截图，§ 3 Architecture）："The DSpark architecture and decoding cycle."

## 架构与训练

### Semi-AR 生成：parallel backbone + 轻量 sequential head（§ 3.1）

- **Parallel stage**：直接用 DFlash 作 backbone（共享 target 的 embedding / LM head 且冻结，KV injection 把 target 多层 hidden state 拼接成 $H_\text{ctx}$ 注入每个 draft 层）。相对原版 DFlash 只做一个**小改动**：把 anchor token 本身当作第一个预测位（输入是 anchor + γ-1 个 mask token，输出 γ 个 logit），略降 draft 算力。
- **Sequential stage**：在 base logits $U_k$ 之上加一个 prefix-依赖的 **transition bias** $B_k$，形成自回归因式分解
  $p_k(v\mid x_0, x_{<k}) \propto \exp\!\bigl(U_k(v) + B_k(x_0, x_{<k}, v)\bigr)$。
  关键点：bias 是局部的、能保留 per-token exact softmax，**不引入全局归一化**——这正是 DSpark 能装进 speculative decoding 严格 rejection-sampling 规则的原因（论文 § 6 专门点名 CRF-NAT 因全局 partition function 无法给 exact per-token 概率而做不到 lossless verification）。
- 两种 head 实例：
  - **Markov head**（默认）：$B_k$ 只依赖 $x_{k-1}$，用低秩分解 $B = W_1 W_2$（$W_1\in\mathbb R^{V\times r}$, $W_2\in\mathbb R^{r\times V}$，$r=256$）；前一 token id 在 $W_1$ 查表得到 transition 向量，再乘 $W_2$ 出整词表 bias。论文用 "of course" vs "no problem" 的例子直接示意：位置 1 抽到 "of" 后，Markov head 把位置 2 上 "course" 抬高、"problem" 压低，截掉 cross-mode collision。
  - **RNN head**：用门控状态 $s_k$ 累积整块前缀（不只上一 token），能力略强；论文测出**收益主要在长 block ($\gamma$ ≥ 12)，复杂度更高、部署不友好**，所以默认用 Markov head。

### Confidence head + Sequential Temperature Scaling（§ 3.2.1）

每个 draft 位输出 $c_k=\sigma(w^\top[h_k; W_1[x_{k-1}]])\in(0,1)$，目标拟合**解析 acceptance rate** $c_k^*=1-\frac12\|p_k^d-p_k^t\|_1$（Leviathan 2023 给出的 per-step 接受率正好是 draft/target 分布总变差距离的余数）。

但 raw confidence 是 over-confident 的（Guo 2017 标准结论），**而 scheduler 用的是"prefix 全活下来"的累积概率 $\prod_{i\le k} c_i$，对绝对值精度敏感**——不只是排序对。所以做 **Sequential Temperature Scaling (STS)**：在 hold-out 集上从 $k=1$ 到 $\gamma$ 顺序 1-D grid search 每个位置的温度标量，最小化**累积乘积**的 ECE，前 k-1 个已校准位置固定不动。温度缩放是 order-preserving 的，不会破坏 confidence head 已学到的排序。Figure 6 显示 STS 把平均 ECE 从 3%–8% 降到 ~1%，AUC ≈ 0.81–0.91 不变。

### Hardware-aware prefix scheduler（§ 3.2.2）

把"每个请求验多长"做成全局吞吐最大化：让 $\Theta = \tau \cdot \text{SPS}(B)$ 最大，其中 $\tau$ 是期望接受 token 数、$B=\sum_r(1+\ell_r)$ 是这一步发给 target 的总 token 数、$\text{SPS}(B)$ 是 engine 在不同 batch size 下的 steps-per-second（**引擎初始化时 profile 一次存成 cost table**）。

关键观察：$a_{r,j}=\prod_{i\le j} c_{r,i}$ 关于 $j$ 单调非增，所以**按 $a_{r,j}$ 全局降序贪心 admit**就自然尊重 intra-block prefix 依赖，扩展即"按 $O(1)$ cost-table 查表"。

**Lossless 与因果性的张力**：confidence head 输入了 $x_{k-1}$（Markov 特征），所以 $a_{r,k+1}$ 实际依赖 $x_{r,k}$。"先排序再 retrospect 选最高"会把 $x_{r,k}$ 偷渡进 $k$ 的 admit 决策，违反 speculative decoding 的 non-anticipating 性质（附录 A 给了构造反例）。论文的解法是在算法 1 里加**早停**：吞吐一旦下降就 break，于是每一步的截断只用到该步及之前已观测的前缀。这保证 lossless（exact target distribution）。

### 训练（§ 3.3）

target 全程冻结；draft 共享其 embedding & LM head 也冻结，只训 backbone + sequential block + confidence head。三项 loss 都按 $w_k=\exp(-(k-1)/\gamma)$ 加权（早位置更重要，因为前缀验证机制下它们的接受决定整块的命运）：

- $\mathcal L_\text{ce}$：标准 next-token CE。
- $\mathcal L_\text{tv}$：draft/target 分布 $L_1$ 距离——直接最小化总变差 = 直接最大化期望接受率（Leviathan 2023）。
- $\mathcal L_\text{conf}$：binary CE，把 confidence head 拟合到解析 acceptance label $c_k^*$。

总 loss $\mathcal L=0.1\,\mathcal L_\text{ce} + 0.9\,\mathcal L_\text{tv} + 1.0\,\mathcal L_\text{conf}$。

## 评测要点

### Offline accepted length（§ 4.2，Table 1）

四个 target × 三类任务上 DSpark 全面占优。Qwen3-4B / 8B / 14B 上 DSpark 比 Eagle3 macro-average 高 30.9% / 26.7% / 30.0%，比 DFlash 高 16.3% / 18.4% / 18.3%；Gemma4-12B 上同样占优，证明跨模型族泛化。Table 1 还暴露强烈的 **domain effect**：结构化任务（math、code）天然接受长（5+），开放 chat 跌到 ~3.5 ——这正是后面 confidence-scheduled verification 的动机。

### Position-wise conditional acceptance（§ 4.3.1，Figure 2）

这是论文最重要的诊断图，回答了"为什么 parallel drafter 反而胜过 autoregressive drafter"这个反直觉现象。

![DSpark Figure 2：Math / Code / Chat 三类任务上 Qwen3-4B 的 position-wise conditional acceptance（前 k-1 个 draft token 都被接受的条件下，第 k 位被接受的频率）。DFlash（深蓝）在 position 1 起点最高（Math 0.93 / Chat 0.72），但向后下滑形成 suffix decay；Eagle3（橙）起点低（Chat 仅 0.53）但向后基本不降甚至略升；DSpark（绿）几乎与 DFlash 重合在 position 1 的高起点，向后保持平直，完整覆盖 Eagle3 的尾段质量。](../assets/dspark/fig2-position-wise-acceptance.png)

> Figure 2（原文截图，§ 4.3.1）："Position-wise conditional acceptance. … Notice that the autoregressive drafter (Eagle3) remains stable or trends upward, while the parallel drafter (DFlash) suffers suffix decay."

论文给出的解释（§ 4.3.1）有两层：

- **Position 1 的容量优势**：在第一位上两类 drafter 都只能根据 target context 预测，差异完全来自架构容量。Eagle3 受 $O(\gamma)$ 延迟约束只能浅；parallel drafter 的 $O(1)$ 让它能堆深。Speculative decoding 是严格 prefix 匹配，**第一位被拒整块作废**，所以这点起点优势在最终 accepted length 上被严重放大。
- **后段独立性的代价**：DFlash 各位置独立预测，跨位置无法条件化，于是出现 "of problem"/"no course" 这类多模式碰撞（multi-modal collision，Gu 2018）。Eagle3 因为按已采样前缀条件化，越往后越确定；DFlash 反而越往后越发散。
- **DSpark 同时拿到两边**：sequential head 用极轻量代价（Figure 4 测出 latency 开销 0.6%–1.3%）把 conditional 信息注回去，保持 DFlash 的高起点并消掉 suffix decay。

### 架构消融（§ 4.3.2，Figure 3 / 4）

- **Drafter depth**：固定 block size = 7，DSpark 层数 1→5 单调改善，**2 层 DSpark 已全面超过 5 层 DFlash**——sequential head 给的"参数效率"显著超过单纯堆 parallel 层。
- **Proposal length**：固定 5 层 drafter，$\gamma$ 从 4 扩到 16，DSpark 相对 DFlash 的差距持续拉大（math: 16%→30% / code: 15%→26% / chat: 18%→22%）。**parallel-only 的边际效用随 block 长度下降**，因为 suffix decay 越长越严重；DSpark 因为有 sequential head 顶住后段质量，扩 block 的回报不会衰减。
- **Latency overhead**：batch 128、上下文 512/1024/2048/4096 平均下，sequential block 在 γ=16 时也只增 1.3% per-round latency——target 验证 batch 大时占的算力压倒一切。

### 生产部署：替代 V4 的 MTP-1（§ 5）

DSpark-5（$\gamma=5$，Markov head，parallel backbone = 3 层 MoE + mHC + sliding window 128）部署进 [DeepSeek-V4](deepseek-v4.md) preview 的 serving engine，对比的 baseline 是 V4 此前在用的 **MTP-1**（即 V3 提出、V3.2 沿用、V4 报告里 depth=1 的 single-token MTP）。论文明确（§ 5.4）："MTP-1 represents the former production setup, having been superseded by DSpark two weeks following the DeepSeek-V4-preview release."——也就是说，目前所有跟 [deepseek.com](https://deepseek.com) 服务说话的用户，背后的 speculative decoding 已经从 MTP-1 切到 DSpark。

为什么 V4 在生产里此前一直只敢用 MTP-1 而不是 MTP-3/5？论文给出的诚实回答（§ 5.4）：静态多 token drafter 在高并发下会因为验证浪费导致 aggregate throughput 严格下降；这正是 DSpark hardware-aware scheduler 要解决的问题——只在系统空闲时延展验证、压力大时主动剪枝。

![DSpark Figure 7：DeepSeek-V4-Flash（左）与 DeepSeek-V4-Pro（右）在生产 live traffic 下的 throughput–TPS 散点与拟合前沿。横轴 TPS = tok/s/user（per-user 生成速度），纵轴 token/s/gpu。MTP-1（灰）的前沿在中等并发时已被 DSpark（粉/紫）的拟合曲线整体外推；V4-Flash 在 80 tok/s/user SLA 下 +51% throughput / +60% TPS，120 tok/s/user 严格档下 MTP-1 几乎撑不住、DSpark 拓出新档（+85% TPS）；V4-Pro 35 / 50 tok/s/user SLA 下对应 +52% / +57–78% TPS。](../assets/dspark/fig7-throughput-tps-pareto.png)

> Figure 7（原文截图，§ 5.4）："Throughput vs. TPS. Aggregate output token throughput against per-request generation speed (tok/s/user) under live traffic."

具体数字（§ 5.4 实测）：

| Engine | SLA anchor | vs MTP-1（matched 吞吐下的 per-user 加速） | 严格 SLA 下 throughput 倍率 |
| --- | --- | --- | --- |
| V4-Flash | 80 / 120 tok/s/user | 60%–85% | 120 档：6.6× |
| V4-Pro | 35 / 50 tok/s/user | 57%–78% | 50 档：5.0× |

论文对"6.6× / 5.0×"留了**诚实的脚注**：严格 SLA 下 MTP-1 batch 跌到极低、相对倍率被分母抬大，因此应把这点理解为"DSpark 把 Pareto 前沿外推到 baseline 撑不起的交互档"，而不是"代表性的乘性提速"。中等 SLA 处的 51%–52% throughput 才是稳态可比对照。

### 生产工程坑（§ 5）

- **Full-vocab 监督要省 target 通信**：训 draft 要 target 的输出分布，跨 worker 传 $|V|\approx 10^5$ 的 logits 是带宽瓶颈，HAI-LLM 改成只传 **LM head 之前的 hidden state**，draft worker 本地走 LM head 投影；通信复杂度从 $O(|V|)$ 降到 $O(d)$。
- **Anchor-bounded sequence packing**：在长 doc 上随机采固定数量 anchor 后把孤立 prediction block pack 进 dense batch，用 **token-level attention index** 维持 exact causal mask，避免 2D padding mask 的算力 / 显存浪费。
- **异步 scheduler 与 ZOS / CUDA graph 调和**：Zero-Overhead Scheduling 要求下一步 batch size 在当前步结束前就已知，同步 scheduling 会 stall pipeline。论文的解法是**用 2 步之前的 confidence 输出估当前的截断长度 $K$**，而**当前要排序的 prefix 仍用最新累计 confidence**——选择是 rank-preserving 的，被截掉的总是排名靠后的。一个意外副效果是它**顺便让全局贪心可以放开 early stop**（早停本来防止未来 token 信息泄漏破坏 lossless）：因为截断长度只依赖 2 步前的历史预测，跟当前 $x_{r,k}$ 物理上独立，构成天然 causal barrier——可以在 jagged SPS 曲线上跨"硬件悬崖"找全局最优而不破坏 lossless 保证。
- **变长 verification 与稀疏 kernel**：在 DeepSeek-V4 架构下，**只需要改 index-attention 和 compress kernel**——这两个原本就是 DSA / CSA 的稀疏 kernel——就能支撑跨请求变长的 verification batch；其他 decode kernel 都按 flatten token 看。

## 待追问

- **DeepSpec / DSpark checkpoints 的开源状态**：论文称发布 V4-Flash / V4-Pro preview 对应的 DSpark checkpoint 与 DeepSpec 训练库（含 Eagle3、DFlash、DSpark），但报告未给具体仓库链接；需要后续从 [github.com/deepseek-ai](https://github.com/deepseek-ai) 或 HuggingFace 上跟进。
- **与 V4 报告里"MTP-1"措辞的对照**：[DeepSeek-V4 技术报告](deepseek-v4.md) 自身在架构与训练里写的是 "MTP depth 1"（即 V3 风格 single-step MTP），并未提及会被 DSpark 替换；这是**两份报告时间差**导致的——V4 preview 报告先发，DSpark 报告两周后跟上并替换。这是 V4 模型页"production 服务速度"主张的真正出处。
- **难样本 unrecoverable draft cost**（§ 5.4 Limitations）：confidence scheduler 只能省 target 端验证，**draft 端 γ-token block 的 parallel backbone 是固定开销**。对于接受率天然极低的复杂请求，这一档算力无法回收；论文留作 future work（difficulty-aware early exit in draft model）。
- **Sequential head 与 [Gated DeltaNet / KDA](gated-delta-net.md) 这类 token-mixer 选择是否正交**：Markov head 走的是 low-rank transition bias，不是 attention 也不是 SSM；与混合线性注意力（KDA / GDN）一类的 backbone 选择应是**正交**的，但论文未做对比。
- **与 Multi-Teacher OPD 训练栈的关系**：DSpark 训练里 target 模型 100% 冻结、走的是 hidden-state caching + full-vocab supervision 思路，跟 [DeepSeek-V4 用的 full-vocab OPD](deepseek-v4.md#后训练opd-替代-mixed-rl)（也是 hidden-state caching + FP4 inference + TileLang KL kernel）共享同一套 HAI-LLM 工程地基；后续如果要写"DeepSeek-V4 的 distillation-as-infrastructure"主题，DSpark 是同一栈在 inference 端的延伸。
