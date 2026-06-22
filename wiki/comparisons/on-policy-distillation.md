# On-Policy Distillation 跨报告对比

## 为什么开这一页

「在线策略蒸馏 / On-Policy Distillation（OPD）」在 2025–2026 的开源技术报告里高频出现，但**同一个名字下封装的是三种相当不同的训练设计**——目的、teacher 数量、KL 形式、在流水线中的位置都不一样。这一页把已收录报告里用了 OPD 的 5 家并排比对一遍，作为 [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md) 概念页之上的跨家综合视角。

**核心共识**（这 5 家都同意的事，也就是 OPD 之所以叫 OPD）：student 从**自己当前策略**采样轨迹，再用 teacher 在这些**student 轨迹**上的输出分布做监督——以此对齐部署分布、避开离线蒸馏的 exposure bias。off-policy 蒸馏（student 学 teacher 生成的静态数据）是它的对照组。

**核心分歧**：OPD 究竟是

1. 把多个领域 RL teacher **融合**进一个 student（MiMo、DeepSeek-V4），
2. 把 flagship 模型的能力**压到小模型**里（Qwen3、Qwen3-VL），
3. 还是把 RL 流水线里多个阶段的能力**召回**到最终 checkpoint（GLM-5）？

## 5 家速览

| 模型 | OPD 的定位 | Teacher 配置 | KL 形式 | 在 pipeline 中的位置 |
| --- | --- | --- | --- | --- |
| **[MiMo-V2-Flash](../models/mimo-v2-flash.md)** | 多专家**融合**（MOPD 是后训练**主范式**） | >多个 domain-specialized teacher（数学/代码/搜索/工具/通用/安全 各训一个 RL teacher，类型可为 RL/SFT/Self） | token-level reverse KL + training-inference importance sampling 截断，与 ORM advantage 可加权混合 | SFT → 分领域 RL teacher → MOPD（最后阶段） |
| **[DeepSeek-V4](../models/deepseek-v4.md)** | 多专家**融合**（OPD **完全替代了 mixed RL 阶段**） | >10 个 domain teacher（覆盖各领域），加权 reverse KL | **full-vocabulary** reverse KL（不简化成 token-level estimate） | 分领域 specialist 训练（SFT+GRPO）→ OPD |
| **[Qwen3](../models/qwen3.md)** lightweight | 把 flagship **压到小模型**（替代小模型的 RL） | **单 teacher**（Qwen3-32B 或 Qwen3-235B-A22B）；只用在 8/14B + 30B-A3B + 4/1.7/0.6B | logit-level reverse KL（off-policy distill 打底 → on-policy distill） | flagship 走完 4 阶段后，lightweight 只跑两阶段 distill |
| **[Qwen3-VL](../sources/qwen3-vl.md)** lightweight | 同 Qwen3，把 flagship VL 压到小 VL | 单 teacher（flagship Qwen3-VL）；off-policy + on-policy 两阶段 | logit-level reverse KL | long-CoT SFT 之后、RL 之前 |
| **[GLM-5](../models/glm-5.md)** | RL 阶段间**能力召回**（"swiftly recover the skills acquired in earlier SFT and RL stages"） | 上游 SFT / Reasoning RL / General RL checkpoint 作为 teacher | on-policy distillation algorithm（报告引 Thinking Machines Lab on-policy distillation 博客） | **整个后训练的最终阶段**——agentic RL 之后用 distillation 召回早期能力 |

> 已收录但**未**用 OPD 的：DeepSeek-V2、DeepSeek-V3.2、Qwen3-Coder-Next、MiniMax-M2、MSA、IndexCache、Kimi-K2.5、Kimi-Linear。

## 三轴对比

### 轴一：OPD 是干什么用的

把 5 家的 OPD 目标按"想解决什么问题"归三类：

**A. 多专家融合（capability merging）**——典型代表 **MiMo MOPD** 和 **DeepSeek-V4 OPD**。问题是 sequential RL 的 capability see-saw：数学 RL 提升后写作或代码退化，简单合并权重也不保留每个 teacher 的峰值。OPD 用 student on-policy 轨迹 + 多个 domain teacher 的 KL 监督来融合，让 student 同时学多个专家而不互相覆盖。两家都用 **>10 个 teacher**。

**B. 强到弱迁移（capacity transfer）**——典型代表 **Qwen3 Strong-to-Weak** 和 **Qwen3-VL Strong-to-Weak**。问题是 lightweight 模型走完整 4 阶段后训练（long-CoT SFT → reasoning RL → mode fusion → general RL）成本高，且 RL 在小模型上 pass@64 不涨（只 sharpen 已有能力，不扩探索空间）。Qwen3 的回答是**单 teacher**（flagship 32B 或 235B-A22B）→ off-policy distill 打底 → on-policy distill 微调，**仅 1/10 GPU·h** 拿到比 RL 更好的 pass@1，**而且 pass@64 也涨**（详见下文 Table 21 复刻）。

**C. 阶段间能力召回（capability recovery）**——只见于 **GLM-5**。报告原话："continuously optimizing on different RL stages with distinct objectives can lead to the cumulative degradation of previously acquired capabilities. To mitigate this issue, we perform on-policy cross-stage distillation as the final stage"。teacher 不是其他模型，而是**自己流水线上游的 checkpoint**——让最终 checkpoint 在统一模式下"召回"前几个 RL 阶段学到、但被后续阶段冲淡的能力。这和 A/B 类不是一个目的。

### 轴二：KL 形式的工程权衡

OPD 的核心数学是 reverse KL，但具体怎么算分两派：

**Token-level KL estimate（资源友好派）**——MiMo MOPD 走这条。在每个 token 位置只算一个标量 `sg[log(π_teacher(y_t|x,y<t) / π_student(y_t|x,y<t))]` 作为 per-token advantage，套进 policy loss。DeepSeek-V4 报告原文承认"this approach is resource-efficient"，但批评它"leads to high variance in gradient estimation and often causes training instability"。

**Full-vocabulary logit distillation（稳定性派）**——DeepSeek-V4 OPD 走这条。直接算完整词表上的 reverse KL `D_KL(π_θ ∥ π_E_i)`，不简化。代价是工程极重，要解决">10 个 teacher 每个数千亿参数、词表 |V|>100k 全部 materialize 不现实"的问题。V4 的工程答案：

- **teacher hidden-state caching**：只缓存最后一层 hidden state，训练时再过 prediction head 重建 full logits（避开 logits 物化）。
- **按 teacher 排序调度**：data dispatch 时按 teacher index 排序，保证每个 distinct teacher head 在 mini-batch 内只 load 一次，device 上最多同时驻留一个 teacher head。
- **TileLang 专用 kernel** 算 KL，**FP4 量化**所有 inference-only forward（teacher / reference），**ZeRO 式 teacher 权重分片**从中心化分布式存储按需加载。

这是 V4 把 OPD 从"理论上可行"做到"可在>10 个 teacher × 万亿参数规模执行"的关键。**MiMo 的 token-level 简化 + R3 路由稳定**是另一个工程取舍——不上 full-vocab，但用 Rollout Routing Replay 解决 MoE 在 rollout 和训练间专家路由不一致的稳定性问题。

Qwen3 / Qwen3-VL 的 logit-level KL 介于两者之间：**单 teacher + 单 student**，不存在 >10 teacher 的调度负担，但确实算 teacher 完整 logits 与 student 对齐。

### 轴三：能不能在 RL 流水线里换位置

**MiMo MOPD = 替代 sequential RL 的合并步骤**。MOPD 之前的"分领域 RL teacher"环节还在，只是不再 sequential 训一个统一 student；MOPD 把合并步骤从"按序训"换成"on-policy 蒸馏融合"。

**DeepSeek-V4 OPD = 直接替代 mixed RL 阶段**。原文：「a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD)」。pipeline 上看，V4 在 V3.2 的训练流程上把"mixed RL"这一格换成 OPD，前置的 specialist 训练仍是 SFT+GRPO。

**Qwen3 Strong-to-Weak = 替代 lightweight 模型的 4 阶段后训练**。原文："This approach eliminates the necessity of performing an exhaustive four-stage training process individually for every small-scale model"。flagship 仍走完整 4 阶段，lightweight 只走 off-policy distill + on-policy distill 两阶段。**这是把蒸馏从"补丁"提升到了"小模型的主后训练方式"**。

**GLM-5 cross-stage distillation = RL 流水线的最终阶段，目的是恢复而非合并**。前面是 SFT → Reasoning RL → Agentic RL → General RL，最后用 on-policy distillation 把上游 checkpoint 的能力召回。

## 关键数据：on-policy distill vs RL（Qwen3-8B）

Qwen3 报告 Table 21（page 21，原文 § The Effectiveness and Efficiency of On-Policy Distillation）给出干净对比。两条路线**同一起点**（off-policy distilled 8B checkpoint，只用 math + code query）：

| 方法 | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU Hours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Off-policy Distillation（起点） | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | — |
| + Reinforcement Learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | **17,920** |
| + On-policy Distillation | **74.4 (93.3)** | **65.5 (86.7)** | **97.0** | **60.3** | **88.3** | **63.3** | **1,800** |

> 括号内为 pass@64。Table 21 原文给出，pass@1 全胜的同时，**pass@64 也涨**（90.0→93.3，83.3→86.7），而 RL 的 pass@64 持平起点（90.0、83.3 不动）。

这张表的意思被 Qwen3 报告拿来论证两点：(1) on-policy distill 的训练效率 **≈10× of RL**；(2) pass@64 涨说明 distill 在**扩探索空间**，不是单纯 sharpen 已有分布——而 RL 在 pass@64 不涨，是"RL 只 sharpen，不扩展"的强证据。

## 三家融合派的实际效果

MiMo MOPD 的 Table 7（[Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md) 概念页里完整 12 行复刻）给出 MOPD 前后 student vs best teacher 的对比，**12 项 benchmark 中 8 项接近或超过 best teacher**，4 项落后（BrowseComp −6.8、Arena-Hard Creative Writing −3.9、GPQA-Diamond −1.2、SWE-Bench Verified −0.8）。说明融合派 OPD 不是自动合并峰值，**teacher 选择与 domain routing 依然关键**。

DeepSeek-V4 报告没有给可比的"OPD 前后"消融表（它把 OPD 当 mixed RL 的整体替代品，没有"先做 mixed RL，再做 OPD"这条对照路径），但报告把 V4-Pro-Base 定为"DeepSeek 系列最强 foundation model"——OPD 的有效性是通过端到端 benchmark 而非消融来论证的。

## 待追问

- **token-level KL vs full-vocab KL 的真实差距有多大**？MiMo 在 token-level KL 上做出了可与 V4 比拼的 SWE-Bench 73.4 / BrowseComp 58.3，说明 token-level 在恰当稳定性补丁下不是 OPD 的瓶颈。V4 上 full-vocab 的工程代价到底换来了什么——是稳定性、收敛速度，还是 teacher 数量上限？
- **单 teacher（Qwen3）vs 多 teacher（MiMo/V4）哪种更适合谁**？Qwen3 demo 了"单 flagship teacher 也能让 8B 在 pass@64 上扩探索空间"，那 MiMo/V4 的多 teacher 是否在小模型场景下也成立——还是只有大模型 student 容量才撑得住多 teacher？
- **GLM-5 的 cross-stage distillation 应不应该归到 OPD**？它的 KL 算法、on-policy 形式与另外几家一致（都引同一篇 Thinking Machines Lab 博客），但目的（召回而非融合 / 压缩）独立。本页倾向于把它列出来作为第三类用法，而不是排除在 OPD 之外。
- **off-policy distill + on-policy distill 的两阶段是不是更通用**？Qwen3 和 Qwen3-VL 都走这条；MiMo 直接从 SFT 进 MOPD 不做 off-policy 预热；V4 也从 specialist 训练进 OPD 不做 off-policy 预热。两阶段是 Qwen 家族的偏好，还是普适更优？
- **MOPD 的 teacher-student co-evolution 循环**（MiMo 报告提到 distill 后 student 可再进 specialist RL 阶段生成更强 teacher）有没有人真的跑通多轮？目前的报告都只跑了一轮 OPD。

## 相关页面

- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)：MOPD 单家深入（公式、Table 7、与 V4 OPD 的差异）。
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)：5 家后训练范式总览，OPD 是其中一支。
- [异步 Agent RL](../concepts/asynchronous-agent-rl.md)：GLM-5 在 cross-stage distillation 之前的 RL 阶段。
- [MiMo-V2-Flash 技术报告](../sources/mimo-v2-flash.md) / [DeepSeek-V4 技术报告](../sources/deepseek-v4.md) / [Qwen3 技术报告](../sources/qwen3.md) / [Qwen3-VL 技术报告](../sources/qwen3-vl.md) / [GLM-5 技术报告](../sources/glm-5.md)：源页。
