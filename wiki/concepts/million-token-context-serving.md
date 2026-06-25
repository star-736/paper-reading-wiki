---
type: Concept
title: "百万 Token 上下文服务"
description: "DeepSeek-V4 的异构 KV-cache、on-disk cache 和 shared-prefix reuse。"
tags: ["concept", "million-token-context-serving"]
timestamp: 2026-06-06
---

# 百万 Token 上下文服务

## 一句话定义

百万 token 上下文服务不是简单把 context window 调大，而是要同时解决 KV-cache 体积、attention kernel、prefix reuse、prefill/decode 调度、磁盘缓存和容错问题。[DeepSeek-V4](../models/deepseek-v4.md) 是当前知识库中最完整的案例。

## DeepSeek-V4 的核心做法

DeepSeek-V4 用 hybrid CSA/HCA attention 降低长上下文成本。CSA 和 HCA 都会压缩 KV entries，SWA branch 保留最近局部 token。这样同一个模型里出现多种 KV 类型：

- CSA compressed KV；
- HCA compressed KV；
- CSA indexer KV；
- SWA uncompressed KV；
- 还没达到压缩块大小的 tail states。

这些 KV 的大小、更新规则和 cache policy 都不同，因此不能直接套传统 PagedAttention 假设。

## 异构 KV-cache

DeepSeek-V4 把 cache 分成两类：

- Classical KV cache：存 CSA/HCA 已压缩的 KV blocks。
- State cache：存 SWA 的最近窗口，以及 CSA/HCA 尚未压缩的 tail tokens。

State cache 更像 sequence-specific state：每条 request 分配一个固定大小 cache block，随着位置推进更新。Classical KV cache 则按压缩块组织，并要和 sparse attention kernel 的 block alignment 协同设计。

## On-disk KV cache

DeepSeek-V4 使用 on-disk KV cache 处理 shared-prefix requests，减少重复 prefill。CSA/HCA 的 compressed KV 可以直接存盘并复用；SWA KV 因为没有压缩且每层都有，体积约为 compressed CSA/HCA 的 8 倍，所以报告给了三种策略：

- Full SWA caching：全部存，计算零冗余，但写入和存储压力大。
- Periodic checkpointing：每隔 `p` tokens 存一次最近窗口，在存储和重算之间折中。
- Zero SWA caching：不存 SWA，只依赖 compressed KV 重算最近窗口，存储省但计算多。

## 与 GLM-5 的关系

GLM-5 没有主打百万 token，但它的 DP-aware routing 与 PD disaggregation 也服务于长上下文 agent 推理。DP-aware routing 让同一 rollout 固定到同一 DP rank，避免多轮工具调用时重复 prefill；PD disaggregation 把 prefill 和 decode 分开，避免长前缀 prefill 干扰正在 decode 的 rollout。

## 关键判断

百万 token 能力最终是模型架构和服务系统共同决定的。只看 benchmark 上“支持 1M context”不够，还要看：

- KV cache 是否可复用；
- shared prefix 是否能避免重复 prefill；
- SWA/CSA/HCA 的 cache policy 是否一致；
- 长上下文 RL 和 OPD 是否能承受数据与内存压力；
- 推理中断后是否能正确恢复，而不引入 length bias。

