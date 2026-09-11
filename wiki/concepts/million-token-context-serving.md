---
type: Concept
title: "百万 Token 上下文服务"
description: "DeepSeek-V4 的异构 KV-cache、on-disk cache 和 shared-prefix reuse；engine 侧由 LMCache 把 paged KV 做成可跨查询/跨引擎搬运的一层。"
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

## DeepSeek-V4.1：用有界近似重放替代长期 SWA 存储

### 前作 YOCO：精确 prefill early exit 的结构条件

原文确证（[YOCO](../sources/yoco.md) §2.2–2.3、Figure 3）：self-decoder 末端生成共享全局 K/V，cross-decoder 历史位置的输出不会成为未来所需的新 K/V，因而为建立缓存可以跳过历史位置的后半计算。生成当前 logits 的位置仍须经过后半网络。前半的固定大小状态仍存在，“only cache once”只指全局 KV 的份数。

V4.1 §2.2 明确受 YOCO 启发，但在 decoder 保留逐层局部 SWA，使纯粹提前退出不再足以准备所有状态。**本页综合**：YOCO 的精确优化来自取消该依赖，V4.1 的有界重放则在保留该依赖的架构上接受近似，二者不能混称“无损重放”。

### V4.1 的存储与恢复策略

原文确证（[DeepSeek-V4.1-Flash 报告](../sources/deepseek-v41-flash.md) §3.2.1–3.2.2）：V4 生产部署把全局 KV 与 prompt/output 末端 SWA checkpoint 分别持久化，SWA 约占持久缓存一半；精确 Zero SWA recovery 所需的 $L\times W$ token 重算过贵。V4.1 只重放末尾 $W=128$ tokens，并截断段前 SWA 依赖，接受近似状态。

| 状态 | 存放与恢复 |
| --- | --- |
| 全局 KV | 长期缓存至少 72 小时，CSA2 + FP4 缩到约 V4-Flash 的 1/4 |
| Encoder SWA KV | 每机 10% host DRAM 组成短期池，TTL 数分钟；缺失时重放 prefix 末尾 128 tokens，复用且不覆盖既有全局 KV |
| Decoder SWA KV | 不做 prefix caching；每次 prefill 将 prompt 末尾 128 tokens 的 encoder 输出跑过 decoder，为当前 decode 准备局部状态 |

作者报相同工作负载下持久缓存约为 V4-Flash 的 1/8，来自全局缓存约 1/4 与移除长期 SWA 存储两项叠加。运行时全局 KV 的 890 bytes/token 不包括全部 SWA 与其他显存。重放**不与完整前向等价**，命中位置还会影响新 suffix 的状态；§6 将恢复边界与极端长上下文列为未充分刻画的风险。

**本页综合**：缓存寿命应跟复用时间尺度匹配；短期会话状态与长期 prefix 采用不同策略。这里以近似计算换存储，不能泛化为任意模型可无损删除 SWA cache。

## 与 GLM-5 的关系

GLM-5 没有主打百万 token，但它的 DP-aware routing 与 PD disaggregation 也服务于长上下文 agent 推理。DP-aware routing 让同一 rollout 固定到同一 DP rank，避免多轮工具调用时重复 prefill；PD disaggregation 把 prefill 和 decode 分开，避免长前缀 prefill 干扰正在 decode 的 rollout。

## 与其他 serving 系统的关系

[vLLM-Omni](../sources/vllm-omni.md) 把 disaggregation 从本页的「长上下文 KV / prefix / state cache」推广到 [Any-to-any 多模态 serving](any-to-any-multimodal-serving.md)：stage 间不只传 KV cache，还传 multimodal embeddings、Thinker hidden states、Talker codec tokens、audio/image tensors。两者共用的系统直觉是：模型能力越依赖长上下文或多阶段流水线，serving 就越不能是单个 monolithic generate loop。

端侧还有第三条压力：[FreeToken](../sources/freetoken.md) 指出 hybrid-attention（V4-Flash 的 SWA、Qwen3.6 的 GDN）把过去压成少量 recurrent state，checkpoint 很贵；agent harness 又几乎每轮在 thinking / tool-call 边界改写历史，稀疏 checkpoint 一旦落在编辑点之后就整段作废。它把有限状态预算锚在这些 semantic 边界上，和本页 DeepSeek-V4 的 on-disk compressed KV 解决的是同一类 prefix reuse，对象从「压缩 KV 块」换成「recurrent state + 会被 OpenClaw/OpenCode 切开的特殊 token」。Expert 池搬运则是正交瓶颈，见 [端侧 MoE serving](edge-native-moe-serving.md)。

第四条是 **engine 侧 I/O 层**：[LMCache](../sources/lmcache.md) 不改 CSA/HCA/SWA 的 cache policy，而是把任意 paged KV 从 GPU 抽出，按远大于 16-token page 的 chunk 在 CPU / 盘 / RDMA / NVLink 上搬，同时服务跨查询 prefix reuse 和 PD disaggregation。V4 的压缩块减小体积，LMCache 决定这体积能不能按时离开再回来；截断 / 滑动窗口会把 prefix hit 打穿（Company F 约 85%→45%），与 FreeToken 的 semantic-anchor 是同一条「谁改历史谁负责失效」约束。完整抽象见 [KV cache 层](kv-cache-layer.md)。

## 关键判断

百万 token 能力最终是模型架构和服务系统共同决定的。只看 benchmark 上“支持 1M context”不够，还要看：

- KV cache 是否可复用；
- shared prefix 是否能避免重复 prefill；
- SWA/CSA/HCA 的 cache policy 是否一致；
- 长上下文 RL 和 OPD 是否能承受数据与内存压力；
- 推理中断后是否能正确恢复，而不引入 length bias；
- **speculative decoding 是否能承受高并发**：[DSpark](../sources/dspark.md) 在 V4 生产端替换 MTP-1 显示，静态多 token drafter 在长 context + 高并发下反而拖垮吞吐；需要 confidence-scheduled、随负载在线截断的 verification 才能把"长 draft block 的高 acceptance"翻译成实际加速。这一档是 1M context 主张能落到用户感知速度的最后一公里。
- **paged KV 离开 GPU 时 I/O 粒度是否独立于模型 cache policy**：[LMCache](../sources/lmcache.md) 显示 vLLM native page-by-page CPU offload / PD 传输出于 16-token page 吃不满带宽（400 vs 88 Gbps）；远程加载相对 prefill 的胜负还随带宽 × 长度 crossover 翻转。

## 相关页面

- 来源：[DeepSeek-V4 技术报告](../sources/deepseek-v4.md)、[LMCache 技术报告](../sources/lmcache.md)、[vLLM-Omni 技术报告](../sources/vllm-omni.md)、[FreeToken](../sources/freetoken.md)、[DSpark 技术报告](../sources/dspark.md)
- 相邻概念：[KV cache 层](kv-cache-layer.md)、[Any-to-any 多模态 serving](any-to-any-multimodal-serving.md)、[端侧 MoE serving](edge-native-moe-serving.md)
