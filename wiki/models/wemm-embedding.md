---
type: Model
title: "WeMM-Embedding"
description: "腾讯微信视觉的通用多模态 embedding 家族（2B/4B/9B），基于 Qwen3.5，支持文本/图像/视频/视觉文档/交错输入，MMEB-v2 上 9B 达 80.6，已部署微信推荐与搜索。"
tags: ["model", "wemm-embedding", "multimodal-embedding"]
timestamp: 2026-08-29
---

# WeMM-Embedding

## 身份

WeMM-Embedding 是腾讯微信视觉团队（WeChat Vision）发布的通用多模态 embedding 家族，把文本、图像、视频、视觉文档和任意交错输入映到同一向量空间，服务检索、推荐、分类和 agent 记忆/工具召回。三档 2B / 4B / 9B 都建在对应尺寸的 natively multimodal [Qwen3.5](qwen3.5.md) 上，用序列末尾的专用 `<embedding>` token 做 last-token pooling。权重和代码已公开。

这是 wiki 里 Qwen3.5 被采用的第三条线：InternVLA-A1.5 拿 2B 做机器人 VLM，Qwen-AgentWorld 拿 35B-A3B / 397B-A17B 做 world model，这里把 2B/4B/9B dense 多模态骨干改成检索/推荐用的 embedding。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **机构** | WeChat Vision, Tencent Inc. |
| **变体** | 2B / 4B / 9B |
| **模态** | **多模态**（文本 + 图像 + 视频 + 视觉文档 + 交错组合；**不支持音频**） |
| **基座** | 对应尺寸的 [Qwen3.5](qwen3.5.md) natively multimodal backbone |
| **池化** | 追加 `<embedding>` token，取最后一层 hidden，L2 归一化 |
| **输出维度** | Matryoshka 截断；2B 在 256 维保留 MMEB-v2 图像/视频 98.7% |
| **训练** | Stage 1 大规模对比 + 分级相关；Stage 2 精炼数据 + 选择性 reranker + 跨尺度蒸馏（9B 改为多变体 merge） |
| **主榜** | MMEB-v2 AVG：2B 77.9 / 4B 79.2 / 9B 80.6 |
| **部署** | 微信视频号、公众号、朋友圈、电商推荐与搜索；14 组 A/B 后全量 |
| **开源** | 权重 + 代码（Hugging Face collection + GitHub） |
| **来源** | [技术报告](../sources/wemm-embedding.md)（arXiv:2608.24053v1，2026-08-25） |

GitHub README 另列（外部佐证，原文未写全）：许可 Apache 2.0；MRL 维度 2B `{64,128,256,512,1024,2048}`、4B `{64,128,256,512,1024,2560}`、9B `{64,128,256,512,1024,2048,4096}`；推理建议 `transformers==5.2.0`，serving 测过 vLLM 0.27.0 与 SGLang 0.5.9。

## 技术身份

WeMM-Embedding 不改 Qwen3.5 的混合注意力骨架，核心是**怎么把 MLLM 变成可部署的通用 retriever**：

1. **统一 pair 格式吃下异构监督**。弱监督图文、caption、检索、分类、QA、分级相关都写成 $(I,q,c,N,y)$，用任务一致 batch 的 in-batch negative，再用 duplicate-aware mask 处理分类这种重复 label。
2. **先铺覆盖再做细**。Stage 1 几百兆对把空间撑开；Stage 2 用 Semantic-ID 降频、MLLM 质检、hard negative，再加「只在有效任务上」的 reranker 序和 9B teacher 的双向相似分布蒸馏。2B 上这四步合计 +2.2 MMEB-v2。
3. **一次前向多种视图**。因果 mask 允许在视频和 ASR 之间、以及序列末尾各放一个 `<embedding>` token，同一 pass 取出 video-only 和联合表示——推荐/搜索里「有的内容有字幕、有的没有」是常见输入。
4. **256 维可上线**。MRL 让存储和索引按流量档位截断，不必为每个维度训一个模型。

和同基座 [InternVLA-A1.5](internvla-a1.5.md) 的差别：那边加 unified expert 和 latent foresight 做动作；这边冻住「生成」目标，只优化向量空间里的匹配。和 [Qwen3-VL](qwen3-vl.md) 也不是一条线——Qwen3-VL-Embedding 是对照里的开源竞品，backbone 走的是标准 GQA 的 Qwen3-VL，不是 Qwen3.5 hybrid。

## 相关页面

- 来源：[WeMM-Embedding 技术报告](../sources/wemm-embedding.md)
- 架构基座：[Qwen3.5](qwen3.5.md)
- 同基座、不同任务：[InternVLA-A1.5](internvla-a1.5.md)、[Qwen-AgentWorld](qwen-agent-world.md)
- 对照开源竞品所在家族：[Qwen3-VL](qwen3-vl.md)（Qwen3-VL-Embedding 的 LLM 线，不是 WeMM 的基座）
