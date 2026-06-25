---
type: Concept
title: "Any-to-any 多模态 serving"
description: "vLLM-Omni 代表的 omni-modal serving 范式：把 Thinker / Talker / Vocoder、AR LLM / DiT / encoder 等多阶段模型拆成 stage graph，独立调度、批处理、传输与资源配置。"
tags: ["concept", "any-to-any-multimodal-serving", "serving", "multimodal-infra"]
timestamp: 2026-06-25
---

# Any-to-any 多模态 serving

## 定义

**Any-to-any 多模态 serving** 是服务「多输入 + 多输出」模型的系统问题：模型不只接收 text / image / video / audio，也可能生成 text / audio / image / video。因此推理路径常常不是一个 LLM decode loop，而是由多个异构阶段组成：multimodal encoder、AR Thinker、AR Talker、DiT / vocoder / image generator、specialized decoder 等。

[vLLM-Omni](../sources/vllm-omni.md) 给出的抽象是 **stage graph**：

- 节点 = 一个独立 serving stage（AR LLM、DiT、CNN / vocoder、encoder 等）；
- 边 = transfer function（把 hidden states、embeddings、tokens、audio/image tensors 等中间态转成下游输入）；
- 每个 stage 内仍保留 serving 系统擅长的 batching、scheduler、KV manager、parallelism policy；
- stage 间由 unified connector 负责数据移动，支持 shared memory、Mooncake / TCP / RDMA、D2H / D2D / H2D 等传输形式。

这和传统 text-only LLM serving 的差别是：后者的核心问题是单个 autoregressive loop 的 KV-cache、continuous batching、prefill/decode 调度；any-to-any serving 的核心问题是**多个 loop / 多种生成范式之间如何编排、传输、重叠和分配资源**。

![vLLM-Omni Figure 3：any-to-any serving 的 stage graph / disaggregated backend 抽象。Orchestrator 负责 Stage Graph、Stage Manager、Request Queues、Data Store；多个 Exec Engine 分别 serve Stage-1 / Stage-2，每个 engine 内部仍有 Model Runner、Scheduler、KV Manager 和 Parallelism Policy；开发者用 Stage Template（PreProcess Fn + Forward）和 Transfer Fn 组合模型，用户只配置 Stage Graph / Runtime 并提交请求。](../assets/vllm-omni/fig3-vllm-omni-architecture.png)

> Figure 3（[vLLM-Omni](../sources/vllm-omni.md) 原文截图，§ 3.1 Overview）："vLLM-Omni architecture."

## 跨报告信号

### Qwen-Omni / Qwen3.5-Omni：Thinker / Talker / Vocoder 是天然 stage graph

[Qwen3.5-Omni 技术报告](../sources/qwen3.5-omni.md) 解释的是模型内部：Thinker / Talker 都建在 Qwen3.5 Hybrid Attention MoE 上，GDN 降低长音视频序列的 KV-cache I/O，ARIA 把文本 + 语音生成改造成更可流式的交错输出。但模型要真正 serve，还需要把这些模块拆成运行时 stage。

vLLM-Omni 用 Qwen2.5-Omni / Qwen3-Omni 展示了一个典型 stage graph：Thinker 先处理多模态输入并生成 text / hidden；Talker 读取 Thinker hidden + multimodal embeddings 生成 audio codec tokens；Vocoder / DiT 再把 tokens 变成 waveform。论文进一步指出 Qwen3-Omni 中 Talker stage 是主要 latency 来源，因为 audio tokens 数量远多于 text tokens。

![vLLM-Omni Figure 4：Qwen2.5-Omni / Qwen3-Omni 类 Thinker-Talker-Vocoder 模型的 stage graph。Thinker 做 mm_encode + thinker_forward，Thinker2Talker 保存 hidden 并传给 Talker；Talker 用 process_input 拼接 thinker hidden / 多模态 embeddings / talker input embeddings，再用 talker_forward 生成 codec；Talker2Vocoder 把 tokens 交给 Vocoder (DiT) 做 dit_decode。](../assets/vllm-omni/fig4-qwen-omni-stage-graph.png)

> Figure 4（[vLLM-Omni](../sources/vllm-omni.md) 原文截图，§ 3.2 Stage Abstraction）："An example implementation of Qwen2.5-Omni (i.e., stage graph and workflow for the model). Qwen3-Omni is similar."

### Qwen3-VL：多模态输入 serving，不等于 any-to-any serving

[Qwen3-VL](../sources/qwen3-vl.md) 是多模态输入 → text output 的 VL 模型：Vision Encoder + MLP merger + Qwen3 LLM，DeepStack 把 ViT 中间层视觉 token residual-add 到 LLM 前 3 层。它会遇到 encoder / prefill / multimodal embedding cache 的 serving 问题，但输出仍主要是 autoregressive text。

因此 Qwen3-VL 更接近 multimodal-input LLM serving；Qwen-Omni / Qwen3.5-Omni / vLLM-Omni 这一线则进入 any-to-any serving：输出端也多模态，必须服务 Talker、Vocoder、DiT 等下游 generator。

### 百万 token serving：同是 disaggregation，但对象不同

[百万 token 上下文服务](million-token-context-serving.md) 关注 DeepSeek-V4 的异构 KV-cache、on-disk cache、shared-prefix reuse、prefill/decode disaggregation。它的中间态主要是 **KV / state cache**。

vLLM-Omni 关注的 disaggregation 范围更宽：不仅有 prefill→decode 的 KV cache，还有 encode→prefill 的 multimodal cache、Thinker→Talker 的 hidden states、Talker→Vocoder 的 tokens，以及 image/audio tensors。可以把它看成：把 LLM serving 里的「KV-cache transfer」推广为「任意 stage edge 的中间态 transfer」。

![vLLM-Omni Figure 5：unified connector 同时覆盖 intra-stage 与 inter-stage transfer。Intra-stage 包括 Encode2Prefill MM Cache、Prefill2Decode KV Cache；Inter-stage 包括 Thinker2Talker Hidden、Talker2Vocoder Token。右侧 Engine-1 / Engine-2 通过 Connector、Payload、Mooncake Store / Shared Mem. 和 Transfer Engine 进行 D2H / D2D / H2D 传输，Orchestrator 只传 Request Meta。](../assets/vllm-omni/fig5-unified-connector.png)

> Figure 5（[vLLM-Omni](../sources/vllm-omni.md) 原文截图，§ 3.4 Disaggregated Data Transfer）："Disaggregated data transfer with unified connector."

## 为什么重要

1. **多模态模型的瓶颈从单模型 decode 变成 pipeline 编排。** 过去优化一个 LLM engine 就能吃到大头；any-to-any 模型里，一个 request 可能穿过多个 AR / non-AR 组件。某个 stage 没法 batch、某条边靠 Python 手写复制，都会吞掉整体收益。

2. **资源配置必须按 stage，而不是按整个模型。** vLLM-Omni 的 Qwen3-Omni 例子里，Thinker 最大、Talker 更 compute-intensive、Vocoder/DiT 又是另一套算子。把三者 co-locate 成 monolith 会让 GPU memory / parallelism policy 互相牵制。

3. **输出端多模态让 serving 与训练/架构同等关键。** Qwen3.5-Omni 报告强调 GDN 降长音视频 KV-cache I/O；vLLM-Omni 则说明，即使模型结构已优化，runtime 仍要解决 Talker tokens 多、Vocoder 可流式、hidden / tensor transfer 等问题。

4. **non-AR generation 开始进入 LLM serving 语义。** DiT / diffusion 过去属于 image/video generation stack；vLLM-Omni 把 DiT 作为 stage graph 节点，与 AR LLM stage 共同调度。这是「LLM serving」向「generative model serving」扩展的信号。

## 待追问

- Any-to-any serving 的统一 benchmark 应该只看 JCT / RTF，还是也要纳入流式 TTFT、tail latency、输出质量、跨 stage backpressure？
- 当 stage graph 很深或有分支时，scheduler 是否需要类似数据流系统的全局优化，而不仅是每 stage 独立 batching？
- Qwen3.5-Omni 的 ARIA / streaming speech generation 与 vLLM-Omni 的 streaming stage output 能否一一对应，哪些是模型设计、哪些是 serving 设计？
- DiT stage 的质量是否完全不受 serving engine 优化影响？如果开启缓存 / graph compilation / parallelism，是否会改变 deterministic / stochastic sampling 行为？

## 相关页面

- 来源：[vLLM-Omni 技术报告](../sources/vllm-omni.md)、[Qwen3.5-Omni 技术报告](../sources/qwen3.5-omni.md)、[Qwen3-VL 技术报告](../sources/qwen3-vl.md)
- 模型：[Qwen3.5](../models/qwen3.5.md)、[Qwen3-VL](../models/qwen3-vl.md)
- 相邻概念：[百万 token 上下文服务](million-token-context-serving.md)、[多模态 Agentic 训练](multimodal-agentic-training.md)、[Forge Agent-Native RL](forge-agent-native-rl.md)、[异步 Agent RL](asynchronous-agent-rl.md)
