---
type: Concept
title: "端侧 MoE serving"
description: "把消费级 GPU+CPU+主机带宽当成统一弹性推理平台：expert 池常驻 host，GPU 做共享 LRU 缓存，用实测带宽比 q* 分流 miss；相对 datacenter stage-graph serving 是另一条轴。"
tags: ["concept", "edge-native-moe-serving", "serving", "moe"]
timestamp: 2026-08-29
---

# 端侧 MoE serving

## 定义

**端侧 MoE serving** 处理的不是「模型能不能放进一张 GPU」，而是：稀疏激活已经让**单 token 计算**落在消费级硬件上，但**完整 expert 池**仍然远超 VRAM。系统必须把 GPU、CPU、主机内存和 PCIe 当成一台统一推理机器，并在 agent 工作负载不断改写上下文、桌面还在抢 VRAM 的条件下调度它们。

[FreeToken](../sources/freetoken.md) 给出的抽象是两级 expert 层次 + 带宽自适应分流：

- CPU-resident expert pool = 全部 routed experts 的 source of truth；
- GPU 上非 expert 权重常驻，剩余 VRAM 是跨层共享的弹性 expert cache（slot 粒度是完整的 `(layer, expert)`）；
- Prefill 用 full-layer double buffering 把「几乎 dense 的 expert 搬运」藏进计算；
- Decode 用 LRU 吃相邻 token 的路由局部性，残余 miss 按 $q^{\star}\approx m\,B_P/B_H$ 分给 PCIe 填缓存和 CPU 原地执行，两路 partial **exact merge**；
- Recurrent / hybrid-attention 的 prefix 复用锚在 thinking / tool-call 等 semantic 边界，而不是均匀 checkpoint。

这和传统 datacenter LLM serving 的差别是：后者默认 GPU 独占、expert 常驻或 EP 分片、瓶颈在 KV 与 batching；端侧 MoE 的瓶颈是 **PCIe 与主机带宽如何分担 miss**，以及 **agent 改写历史之后还能不能复用 recurrent state**。

## 跨报告信号

### FreeToken：把「能不能跑」从集群问题改成单机编排问题

[FreeToken](../sources/freetoken.md) 的原文主张是：一旦稀疏激活让计算可行，本地推理就不再是「模型是否装得进 GPU」，而是系统能不能编排这台机器。它在 8GB RTX 4060 笔记本到单卡 RTX PRO 6000 上服务 35B–753B MoE，decode 相对 llama.cpp / KTransformers / Ollama 1.3–2.3×，且把最差 TTFT 压在真实 agent 客户端超时线以下（OpenClaw 120 s watchdog）。相关工作里，专家 offload 线（EdgeMoE / Mixtral-offloading / MoE-Infinity）最终每个 miss 仍是一次 PCIe 传输；CPU–GPU 混合线（Fiddler / KTransformers / HybriMoE）的分流要么启动时钉死，要么 host 侧启发式进不了 CUDA Graph。FreeToken 把分流收成两个实测带宽的闭式比，并嵌进 serving runtime 而不是单请求 harness。

### DeepSeek-V4-Flash：激活 13B 进 32GB，expert 池仍要 140GB 级搬运

[DeepSeek-V4-Flash](../models/deepseek-v4.md) 是这条轴上最干净的模型侧例子：284B / 13B active，6/256 routed experts，部署精度下激活足迹可进 RTX 5090；但 FP4 完整 expert 池约 140 GB，prefill 几乎流一遍。FreeToken 在同一官方 MXFP4 checkpoint 上跑出 22–25 tok/s，且 agent 工作负载相对单轮只掉 ≤12%。这解释了为什么 [百万 token 上下文服务](million-token-context-serving.md) 里的异构 KV / on-disk prefix 还不够：V4 的 hybrid CSA/HCA/SWA 已经压缩 KV，端侧仍然会被 **expert 搬运和工具调用触发的 re-prefill** 打穿。FreeToken 给 SWA/recurrent 状态的回答是 semantic-anchor checkpoint，给 expert 的回答是 LRU + $q^{\star}$，两套 cache 独立回收。

### GLM-5.2：单卡工作站上的 753B 演示，不要和 GLM-5 744B 混读

FreeToken 把 [GLM-5.2](../models/glm-5-3.md)（文中 753B / 40B active，NVFP4，433 GB checkpoint）跑在单卡 RTX PRO 6000 上 14.9 tok/s，llama.cpp 7.3 tok/s。KTransformers 在该箱上没有可服务路径（host-resident experts 要 753 GB–1.5 TB，箱内只有 512 GiB）。这是 **GLM-5.2 权重的端侧 serving 证据**，不是 [GLM-5](../models/glm-5.md) 技术报告（744B / 40B）的参数更正，也不是 GLM-5.3 后训练的能力声明。

### 与 vLLM-Omni：同一「serving 决定能力能否落地」，两条正交轴

[Any-to-any 多模态 serving](any-to-any-multimodal-serving.md) 解决的是 datacenter 里 Thinker / Talker / Vocoder 等多阶段图如何拆 engine、传 hidden/codec、按 stage 配资源。[端侧 MoE serving](edge-native-moe-serving.md) 解决的是单机上 expert 池放不下时，如何用 CPU 带宽补 PCIe、并在 agent 改写上下文后保住 prefix。两者都反对「一个 monolithic generate loop」，但拆的对象不同：前者拆 **模型阶段**，后者拆 **expert 驻留与 miss 路径**。Kimi-K3 在 FreeToken Figure 1 里被标为超出消费级内存（594 GB），说明这条轴目前的上界仍是 host 内存，不是 GPU 算力。

## 为什么重要

1. **开源权重 ≠ 开源可部署。** 前沿 MoE 的能力差距在缩小，可及性差距（谁买得起集群 / API）没有同速缩小。端侧 serving 是把已收录的 V4-Flash / GLM-5.2 级模型变成个人机器上可交互软件的那一层。
2. **读 agent benchmark 时，本地引擎和 harness 一样是一等变量。** FreeToken 显示同一模型在 llama.cpp vs 带宽自适应 runtime 下 decode 与 tail TTFT 可差到超时线两侧；这和 [UniClawBench](../sources/uniclawbench.md) 的 framework > model 是同一类警报，只是这里的 framework 是 serving engine。
3. **Hybrid-attention 的 serving 成本不在 KV 体积，而在「编辑后从哪恢复」。** GDN / KDA / SWA 把过去压成少量状态，checkpoint 预算极紧；预算必须花在 harness 实际会切开的 semantic 边界上，否则每次 tool call 都要重算数千 token。
4. **正确性与性能解耦。** CPU 池做 source of truth，GPU 缓存可在运行时缩小，使「浏览器突然占走几 GB VRAM」不再等于引擎崩溃——这是 datacenter 独占 GPU 假设里不存在的约束。

## 待追问

- $q^{\star}$ 在 batch>1、投机解码或多请求交织时是否仍是闭式最优，还是只对论文的单流 agent decode 成立？
- Semantic-anchor 集合是否绑定特定 harness 的 special token（OpenClaw / OpenCode / SWE-agent），换一套压缩策略要不要重锚？
- 20+ 模型声称与三模型实验之间的泛化缺口：MLA+DSA、LatentMoE、7:1 Lightning 等未评架构会不会打穿 LRU 局部性假设？
- 端侧 serving 有没有类似 SWE-bench 的任务完成率数字，还是只有 tok/s / TTFT？

## 相关页面

- 来源：[FreeToken](../sources/freetoken.md)、[vLLM-Omni 技术报告](../sources/vllm-omni.md)、[DeepSeek-V4 技术报告](../sources/deepseek-v4.md)
- 模型：[DeepSeek-V4](../models/deepseek-v4.md)、[GLM-5.3](../models/glm-5-3.md)、[GLM-5](../models/glm-5.md)（不要把 753B 与 744B 画等号）
- 相邻概念：[Any-to-any 多模态 serving](any-to-any-multimodal-serving.md)、[百万 token 上下文服务](million-token-context-serving.md)、[MoE 前沿模型扩展](moe-frontier-model-scaling.md)
