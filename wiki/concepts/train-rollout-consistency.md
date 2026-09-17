---
type: Concept
title: "训练—rollout 一致性"
description: "为什么采样路径与训练路径会对同一条轨迹给出不同概率，以及各家在 token 对齐（TITO）、专家路由重放（R3）、量化契约、ratio 修正与 bit-exact 对齐这五层上的做法与成本阶梯。"
tags: ["concept", "train-rollout-consistency", "agentic-rl", "asynchronous-rl", "tito"]
timestamp: 2026-09-11
---

# 训练—rollout 一致性

## 定义

训练—rollout 一致性（train–rollout consistency，各家也叫 train-inference mismatch / probability mismatch）指的是：policy update 里使用的 log-probability，是否真的等于采样那条轨迹时的概率。

不一致的后果不是报错，而是**静默的梯度污染**。只要更新依赖 rollout log-prob（GRPO 家族、PPO、OPD 全部依赖），重要性比 $r=\exp(\log\pi_{\text{train}}-\log\pi_{\text{rollout}})$ 就会偏离 1；偏差足够大时，更新作用于一条「从未发生过的轨迹」，而每一批新数据又来自被上一次污染的策略，漂移因此逐轮复利。

## 为什么会不一致：五类来源

| 来源 | 机制 | 典型症状 |
| --- | --- | --- |
| tokenization | 模型输出经过 message parsing、工具执行、chat template 渲染再进入下一轮；每一步都可能重新分词、裁掉历史 reasoning 或重新序列化 tool call | 采样出的 token 与 trainer 重建的 token 不是同一串 |
| MoE 专家路由 | router 用学习到的投影给专家打分并取 top-k；训练与采样用不同 kernel / 不同精度时，微小数值差可以逐层逐 token 翻转中标专家 | 更新落到从未参与该 token 的 expert，而贡献过它的 expert 收不到更新 |
| 数值精度 | rollout 与训练用不同 kernel、不同算子融合、不同 batch 形状、不同量化方案，每个差异都在浮点最后几位扰动结果 | 长序列上扰动累积成可测的概率缺口 |
| 异步 off-policy | 一条轨迹（或一个 turn）来自比 trainer 更旧的权重版本 | ratio 结构性地偏离 1，且随 staleness 增长 |
| 参考分布缺失 | 异步下无法维护无界的 $\pi_{\theta_{\text{old}}}$ 历史 | 无法按标准 PPO 方式计算 ratio |

这与「policy 优化算法怎么改」正交：前四类问题在同步运行、算法完全不变时同样存在。

## 五层手段与成本阶梯

按「修的是哪一环」分层，成本从上到下递增。关键判断是：不是所有场景都值得买到最下面一层。

### 第一层：TITO（token-in, token-out）

让 **serving 层而不是 harness** 决定 tokenization：agent 只交换普通 message 并在每轮送全量历史，session server 负责把 message 渲染成 token、checkpoint 每轮完成后引擎返回的 token IDs / log-probabilities / routed experts，后续轮复用最深的可用 checkpoint 只 tokenize 追加的 suffix，并**覆盖 agent 自己提供的任何 token 字段**。

- 已收录信号：[GLM-5](../sources/glm-5.md) 报告把它写成系统机制（[异步 Agent RL](asynchronous-agent-rl.md)）；[Laguna](../sources/laguna-m1-xs2.md) 用 `render_assistant_messages_raw` flag 断言「解码字符串 == rollout 存的 prefix」，从机制上消除「RL 训的格式 ≠ 部署格式」；[Miles v0.1](../sources/miles-v0-1.md) §2.4 给出最完整的形态（Linear / Branching 两种 session 扩展规则、可配置的重放比较 matcher、注册 family 的双重校验）。
- 代价：几乎没有额外显存 / 带宽成本，但要求 serving 层可被改造，且**黑盒 harness 必须把 tokenization 控制权交出来**。

### 第二层：R3（Rollout Routing Replay）

把每个 token 的专家分配当成 rollout 数据的一部分随轨迹一起传，训练时**重放 mask** 而不是重算 top-k，让 token 精确地回到采样时处理它的专家。softmax 仍加在训练 logits 上，router 梯度不丢。

- 一手出处：[R3](../sources/r3.md)（Ma et al.，北大 + 小米，arXiv:2510.11370）。Qwen3-30B-A3B 上约 10% 的 (token, layer) router 选错专家、94% token 至少一层不同；重放后 train–infer KL 从 $1.535\times 10^{-3}$ 降到 $7.54\times 10^{-4}$，接近 dense。不要和 GSPO 文里的 **Recompute** Routing Replay 混：那套从训练重算阶段取路由，`mini_step=1` 时失效。
- [MiMo-V2-Flash](../sources/mimo-v2-flash.md) §4.6.1 在 RL / MOPD 基建里采用（同团队，request-level prefix cache 存 KV + routed experts）；[Miles](../sources/miles-v0-1.md) §2.5 做成 `--use-rollout-routing-replay`，因为 session server 已经记录 routed experts，重放覆盖整个多轮 episode 而非单次 completion。
- 代价：R3 原文只报 rollout 额外延迟 <3%。**60 MB / 轨迹**是 Miles 按 (tokens−1)×layers×k 个 32 位整数估的（例：32K × 60 层 × k=8），不是 Ma et al. 的数字。因此它是 per-recipe 选择而非全局开关，dense 模型上是空操作。

### 第三层：量化 / 精度契约

核心约束是**两边必须共享同一份量化协议**：只量化 rollout，或两边量化方式不同，会让两边从同一权重算出不同结果，误差逐层累积（Miles §3.1 称其可能直接导致 catastrophic train-rollout mismatch）。做法是把量化实现成端到端契约，检查 checkpoint 转换、trainer 前向、rollout、权重导出四个阶段同契约；允许的部署组合只有「两边同格式」或「trainer 留 BF16 而 rollout 量化」两种（NVFP4 例外，它要求每个接触权重的阶段都量化）。

- 各家变体：[MiniMax-M1](../sources/minimax-m1.md) 在 hybrid Lightning Attention 上看到 train/infer kernel 概率对不齐（LM head 高幅激活），reward 不涨；LM output head 改 FP32 后相关从约 0.987 到 0.997（Figure 3），小 dense softmax 模型上没出现。[Ring-2.6](../sources/ling-2.6.md) 用 module-aware FP8——LM Head 走 FP32、Attention/Shared Experts 保 BF16、Routed Experts 用 blockwise FP8，按模块敏感度分配精度；[Kimi K3](../sources/kimi-k3.md) 从 SFT 起即 QAT（MXFP4 权重 + MXFP8 激活），RL 时 rollout 与 training 共用同一量化方案；[Laguna](../sources/laguna-m1-xs2.md) 保留 BF16 权重、只把 KV cache 存 FP8——作者明确写「为安全牺牲一半并发」，因为 FP8 权重的预发布消融显示 train-inference KL mismatch 变大；[VibeThinker-3B](../sources/vibethinker-3b.md)（MGPO）观察到 rollout engine 优化推理吞吐后被放大的概率失配，直接改用全 on-policy。

### 第四层：ratio 修正（mask / clip 形状）

前三层去掉结构性成因后仍会剩数值残差，做法是在 importance ratio 上做修正。[Miles](../sources/miles-v0-1.md) §3.4 把这一层做成与算法并列的可替换组件：**TIS** 把 $r$ 夹到区间（默认 $[0,2]$，只作用在上尾）后当作 per-token 权重，**阻尼**极端 token；**clip-or-pop** 把区间外的 token 权重置零，**丢弃**极端 token。

- 这一层已有一张独立的方法谱系表，见 [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md) 的 IcePop / KPop / CISPO / DIS 各行——本页不重复：[IcePop](../sources/ring-1t.md) 一手出处用 $k=\pi_{\mathrm{train}}(\theta_{\mathrm{old}})/\pi_{\mathrm{infer}}$，$M(k)=k$ 落在 $[\alpha,\beta]$ 否则 $0$（默认 $[0.5,5]$），越界**丢弃**；作者对照 TIS 的阻尼并认为留下的扰动会放大。KPop 换成 binary KL（Ring-2.6 主张低概率 token 的 ratio noise 更大，固定比率会过度 mask 它们）。GLM-5 与 [SAO](../concepts/single-rollout-asynchronous-optimization.md) 的 DIS 用 $\pi_\theta/\pi_{\text{rollout}}$ 出界整段 mask。IcePop 自己的 Limitations 写明：它压住主要失配，**没有**做到完全一致，算子层数值差仍在。
- 与第三层的分工要读清：量化契约决定「两边算出什么」，ratio 修正决定「算出来不一样时怎么用」。Miles 的 §9 参考运行就是这一分工的实例——BF16 训练 + FP8 服务，剩余的 train-inference KL 均值 0.0369 由 TIS 在更新里吸收。

### 第五层：true-on-policy alignment（bit-exact）

不做修正，而是让两边**真的算出同一个数**：同一个 attention kernel（FlashAttention-3，prefill 与 decode 路径 bitwise 一致）、batch-invariant 的 matmul kernel（输出不随 batch 组成变化）、关掉 trainer 侧的 fused kernel 并钉住 per-model kernel contract、SGLang 跑 deterministic-inference mode、TP 的 row-parallel linear 与 all-reduce 做到与并行度无关，最后 rollout 用一次 prefill pass 重新给完成的序列打分。

- 在受支持配置下绝对差恰为 0（Miles §5.3），代价是吞吐，而且覆盖范围极窄——Miles 只注册了 dense Qwen3 0.6B 与 4B，线外模型直接拒绝启动，不出去部分保证。保证也只覆盖「每个采样 token 的 log-probability」这一个指标，不主张两边在输出分布上处处一致。

## 为什么重要

- **代价与规模成正比，且不会报错**。单 token 的偏差无所谓；跨层 × 上万 token × 上千 step 会累积成 policy drift，而最松的重放比较设置会让「一个 agent 从没发起过的 tool call」留在训练历史里，Miles 也不对账 tool-call identifier，错配因此可以完全静默（Miles §2.4.2）。
- **它区分了「训练跑不动」和「训练跑得动但训的是别的东西」**。异步调度、partial rollout、staleness 治理解决的是前者；本页这一轴解决的是后者。
- **「对齐」有明确的成本阶梯**，因此正确的问题不是「要不要做到 bit-exact」，而是「在哪个模型 / 拓扑上值得买到哪一层」：TITO 近乎免费但要求 serving 层可改造；R3 要付 60 MB/轨迹；量化契约会绑死可选格式与硬件（MXFP8 / NVFP4 只在 Blackwell 上可用）；bit-exact 只在 dense 小模型上有 profile。

## 待追问

- **需实验或作者披露**：R3 的带宽成本：Miles 估 60 MB/轨迹，原文只报 <3% 延迟。1M 上下文下是否还有更低成本的等价机制？异步 staleness 大时只重放 mask 够不够？
- **需实验或作者披露**：「量化契约」与「ratio 修正」的边界应该是硬性的吗？Miles 的 NVFP4 例外说明有些格式无法只靠 ratio 修正兜底，但报告没给「哪些格式必须走契约、哪些可以靠修正」的判据。
- **需实验或作者披露**：各家的对齐声明几乎都是自报，缺少跨系统、同配置的独立测量。

## 相关追问

主记录：[log-prob 差与 KL 的统计口径](../sources/miles-v0-1.md#待追问)；[多模态 TITO 精确性](../sources/miles-v0-1.md#待追问)。

## 相关页面

- 来源：[R3](../sources/r3.md)（Rollout Routing Replay 一手出处）、[Miles v0.1](../sources/miles-v0-1.md)（五层手段的完整实现与限制清单）、[GLM-5](../sources/glm-5.md)（TITO + DIS + stale dropping）、[GLM-5.3 官方发布博客](../sources/glm-5-3-blog.md)（`slime` 数值对齐声明）、[MiniMax-M1](../sources/minimax-m1.md)（hybrid 上 LM head FP32 对齐 train/infer）、[Ring-1T](../sources/ring-1t.md)（IcePop 一手出处）、[Ling-2.6](../sources/ling-2.6.md)（module-aware FP8 + KPop）、[Laguna](../sources/laguna-m1-xs2.md)（chat-template 对齐断言 + FP8 KV cache 取舍）、[Kimi K3](../sources/kimi-k3.md)（QAT 贯穿 SFT 与 RL）
- 概念：[异步 Agent RL](asynchronous-agent-rl.md)（staleness 与异步调度）、[RL 权重同步与部署拓扑](rl-weight-synchronization.md)（传输层，与本页正交）、[Single-Rollout Asynchronous Optimization](single-rollout-asynchronous-optimization.md)（DIS）、[Multi-Teacher On-Policy Distillation](multi-teacher-on-policy-distillation.md)（OPD 的 log-prob 依赖）
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)（ratio / mask 形状谱系）
