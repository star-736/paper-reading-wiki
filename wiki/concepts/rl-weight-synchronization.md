---
type: Concept
title: "RL 权重同步与部署拓扑"
description: "训练更新后把权重送到 rollout engine 的三条传输路径（NCCL broadcast / P2P RDMA / disk-delta）及其适用条件，含「收益随 fleet 宽度而非模型规模增长」的实测规律与 pause 成本。"
tags: ["concept", "rl-weight-synchronization", "post-training-infra", "asynchronous-rl"]
timestamp: 2026-09-11
---

# RL 权重同步与部署拓扑

## 定义

RL 循环的第三段：trainer 算完更新后把新权重送到每个 rollout engine，引擎再用新策略生成。它同时决定三件互相牵制的事——权重同步占多少 step 时间、支持哪些部署拓扑、以及 in-flight 请求怎么处置。

契约本身很简单（[Miles v0.1](../sources/miles-v0-1.md) §4）：trainer 准备新权重 → 传给每个 rollout engine → 引擎用更新后的策略开始生成；cadence 可配，默认每个训练 step 后同步一次。

## 为什么它会成为瓶颈

- **量级本身就大**：一次完整 NCCL broadcast 广播 Kimi K2 1T-A32B 接近一分钟（Miles §4）。训练与 rollout 在不同 GPU 上时，这一步可以主导整个 step。
- **权重版本必须有明确的生效边界**：异步下如果更新在轨迹中途生效，同一条轨迹的 token 就来自多个权重版本，loss 里「把 policy 当 piecewise-constant」的假设失效。[Laguna](../sources/laguna-m1-xs2.md) 用两个原语处理——权重广播触发 inference 侧 KV-cache reset（防止不同权重版本的 token 混入），权重更新 block 在途 rollout step。
- **它把 staleness 变成可控量**：异步运行里 trainer 永远领先于生成中的轨迹，staleness 上限与同步频率是同一个旋钮的两面。Laguna 设上限 10 optimizer step、每 2 step 广播一次，靠训练 / 推理 GPU 配比让 staleness 实际从未触及；Miles 则把 staleness 的判定交给数据缓冲（见 [异步 Agent RL](asynchronous-agent-rl.md)）。

## 三条传输路径

| 传输 | 路径 | 适用条件 | 代价 / 限制 |
| --- | --- | --- | --- |
| NCCL broadcast（默认） | 每个训练 rank 广播到所有 rollout rank | rank 之间有 NCCL fabric | 一份数据发给所有人，即使某些目标不需要那些分片；每个 bucket flush 取一把共享锁，慢 stage 会卡住其余 stage |
| Peer-to-peer RDMA | 训练 rank 直接写进 rollout-rank 内存 | rank 间可直达 | 实际发送源数 = min(源, 目标)；需要 CPU 常驻模型副本做 re-shard；单节点比 broadcast 慢最多约 70% |
| Disk-delta | 变化的字节写到共享文件系统，rollout host 自行 patch 本地 base | 不需要共享 fabric，只需共同文件系统 | 只支持 Megatron 后端；拒绝与 colocation / LoRA / PD 分离组合；默认的 XOR 编码非幂等 |

colocated（trainer 与引擎同卡）时是本地 handoff，不涉及传输；但 fully-async 调度要求两个独立 GPU 池，所以本地 handoff 与异步执行基本互斥。LoRA 这条路径上还有一个特例：colocated 走 IPC（trainer 序列化 adapter tensor，把 handle 交给引擎进程，不过网络），disaggregated 才走 NCCL；**P2P 与 disk-delta 都不携带 adapter**（Miles §5.1）。

## 关键规律

- **P2P 的收益随 fleet 宽度增长，不随模型规模增长。** Miles Table 8（H100、1 GB bucket、稳态 step 平均）里，Qwen3-30B-A3B 在每侧 2 节点只省 19.1%，而 GLM-5 744B（16 节点）与 Kimi K2 1T（32 节点）省 85–86%。机制是：M 个源 rank、pipeline 深度 p、目标 EP 度 e 时，P2P 能借到约 M/p 倍的聚合发送带宽，同时每个目标少收约 e 倍数据。作者强调优势在**每侧两个节点时就已出现**。
- **单节点是 P2P 的反例。** 单节点给不出额外的聚合带宽，却仍要付 host 侧 re-sharding 与 pinned-memory staging 的固定成本，实测比 broadcast 慢最多约 70%。所以默认保持 broadcast，P2P 只在训练侧与 rollout 侧都跨多节点时才有价值。
- **disk-delta 的前提是稀疏性。** 连续 RL step 只改动模型很小一部分字节，因此只发变化字节远省于发全部。它的五步流程里**只有最后一步暂停生成**（重新加载 patched checkpoint、推进引擎权重版本、恢复生成），前四步只是读写文件，可与完全异步的 rollout 并行——这才是它「便宜」的真正原因，而不是传输量小。
- **一致性与幂等在编码层面取舍。** XOR（默认）更紧凑、生成更快，但是对合运算，同一 delta 应用两次会恢复旧权重，因此必须严格应用一次且只对自己的 base；Overwrite 写入更多字节但幂等，重试或重复无害。两种编码都在原始字节上工作，不感知 tensor 语义，所以 base 与导出策略必须在 tensor 名、dtype、shape、字节布局上一致；Miles 的兜底是校验**结果状态的每个 tensor** 而不是 delta，并在校验失败时于任何引擎 reload 之前停下。
- **验证要靠注入噪声，而不是靠传输代码自证。** Miles 的 opt-in 检查先把引擎 tensor 填成随机值再跑首次权重更新，任何没被写到的 tensor 到达检查时仍持有噪声，因此静默漏写无法蒙混过关；默认要求 bit-exact，也允许按量化格式推出的舍入容差。这类检查只在 CI 与验证运行里开启。

## 与其它层的正交性

- 与 [训练—rollout 一致性](train-rollout-consistency.md) **正交**：传输保证「引擎拿到的是 trainer 的权重」，一致性保证「拿到之后两边算出的概率相同」。Miles 的 disk-delta 校验和 true-on-policy alignment 是两条不同的保证，前者防漏写、后者防数值分歧。
- 与 [KV cache 层](kv-cache-layer.md) **正交**：那边管缓存如何 offload、跨查询复用、跨引擎传输；这边管权重本身。两者在异步 RL 里会交叉——权重广播触发 KV-cache reset（Laguna）或让引擎 weight version 前进（Miles）。
- **与部署拓扑强耦合**：三种异步治理粒度（Miles 的 staleness 上限 + buffer、Laguna 的 GPU 配比 + 每 2 step 同步、Ring-2.6 的 staleness manager）表面上是调度选择，实际上也是权重同步成本的函数——同步越贵，越倾向于拉长 cadence 并靠 staleness 治理兜底。
- **LoRA 把这一层变小一个数量级**：同步的是 adapter 而不是整个模型，训练 step 与权重更新两处同时变轻；代价是 P2P / disk-delta 两条更激进的传输不可用。

## 待追问

- **需实验或作者披露**：disk-delta 与 LoRA 被明确判为不兼容；adapter 的增量传输有没有更省的方案？
- **需实验或作者披露**：权重同步造成的 pause 长度、staleness 上限与同步 cadence 三者如何联合调参？Miles 给出了一整套 staleness 指标但没有调参方法。
- **需补外部来源**：权重同步的 pause 在异步运行里必然是一个全局屏障。有没有把「pause 期间继续跑旧权重轨迹」的代价显式建模进调度目标的公开工作？

## 相关追问

主记录：[新架构 P2P 权重映射成本](../sources/miles-v0-1.md#待追问)。

## 相关页面

- 来源：[Miles v0.1](../sources/miles-v0-1.md)（三种传输的完整实现、Table 7/8、§4.3 五步流程与 §4.4 pause / 校验）、[Laguna](../sources/laguna-m1-xs2.md)（NCCL GPUDirect RDMA + 每 2 step 广播 + KV-cache reset 原语）、[Kimi K3](../sources/kimi-k3.md)（External KV cache pool 与训练状态 offload 到 NVMe）
- 概念：[训练—rollout 一致性](train-rollout-consistency.md)、[异步 Agent RL](asynchronous-agent-rl.md)、[KV cache 层](kv-cache-layer.md)
- 比较：[LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)（算法层代价与本页的系统层代价互补）
