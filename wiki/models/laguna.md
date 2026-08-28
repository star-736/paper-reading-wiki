---
type: Model
title: "Laguna"
description: "Poolside 的 Laguna M.1（225.8B/23.4B）与 XS.2（33.4B/3B）MoE agentic coding 模型族，Model Factory 工业化流程产出，3:1 SWA/GA + softplus 门控 + WSD + CISPO RL。"
tags: ["model", "moe", "agentic-coding"]
timestamp: 2026-07-30
---

# Laguna

## 关键事实

| 项 | Laguna M.1 | Laguna XS.2 |
|---|---|---|
| 团队 | Poolside（research@poolside.ai） | 同 |
| 总参 / 激活（含 embedding） | 225.8B / 23.4B | 33.4B / 3B |
| 架构 | pre-norm Transformer + RMSNorm + MoE | 同 |
| 路由 | token-choice | 8 of 256 routed + 1 shared；routed ×2.5 系数 + shared；linear+sigmoid + top-k score norm |
| 注意力 | 每层 global attention | **interleaved SWA/GA 3:1**，GQA 8 KV heads，head dim 128，**softplus per-head gating** |
| 位置编码 | RoPE | GA: partial RoPE 50% head dim, θ=5e5; SWA: window 512, θ=1e4 |
| 底部 dense 层 | 3 | 1（保稳定） |
| 词表 | 100,352 BPE（两模型共享） | 同 |
| 优化器 | Muon（Moonlight variant），全阶段 | 同（Moonlight scaling 开） |
| LR 调度 | cosine | WSD（peak 5e-4, 末 30% 1-√ 衰到 5%） |
| 预训练 GPU | 6,144 × H200 | 2,048 × H200 |
| 预训练 tokens | 30T+ | 30T+（~27T unique 池） |
| 上下文 | — | 4K 预训练 → 32K → 128K（YaRN 仅 GA 层）→ 256K（RoPE scale 翻倍无训练） |
| 后训练 | mid-train(~60B) → SFT(3×40B) → agentic RL(CISPO) | 同 |
| 开源 | — | XS.2 权重 Apache 2.0（[HF](https://huggingface.co/collections/poolside/laguna-xs2)） |
| 来源 | [arXiv:2605.27605v1](https://arxiv.org/abs/2605.27605) | 同 |

## 技术身份

Laguna 是一个**纯文本、coding-focused** 的 MoE agentic 模型族，定位与 Devstral 2 / Qwen3.5-35B-A3B / Qwen3.6-35B-A3B 同重量级竞争。它的技术身份由四个轴定义：

1. **流程轴（Model Factory）**：把模型开发做成工业流程是 Poolside 的核心论点。三原则——experiments as code（Dagster 控制 plane + 全 lineage DAG）、composable decoupled components（研产单代码库）、reserve human attention for novel decisions（自研 per-job 调度器 + FoundationDB topology + sticky pod respawn，placement sub-minute）。M.1 预训练结束后**五周**从零交付 XS.2 是这套流程的实证，而非单点架构创新。

2. **注意力轴**：XS.2 用 **3:1 interleaved SWA/GA**（[Gemma 4](gemma-4.md) 是 5:1，[MiMo-V2-Flash](mimo-v2-flash.md) 5:1 window 128）配 **softplus per-head gating** [67]（[67]=[Gated Attention 报告](../sources/gated-attention.md)，NeurIPS 2025 Best Paper）。GA 层 partial RoPE(50%) + θ=5e5，SWA 层 window 512 + θ=1e4。16B proxy 消融（Table 9）逐步加上 SWA-1024→per-head gating→GA partial RoPE→SWA-512→48GA/64SWA Q-heads + k_dense=1 得最终架构。

3. **数据轴**：长 horizon（30T tokens）下从「精度优先」转「召回优先 + 排序」——高召回 web 管线（Propella 多维标注 + composite score 排序，恢复 34% 被误杀高质量文档）、合成数据（Hive 框架，~13% 混合）、**AutoMixer** 自动混合优化（~60 个 0.5B proxy + Dirichlet 扰动 + per-capability 回归器 + KL 正则，是 [数据混合优化](../concepts/data-mixture-optimization.md) 谱系的产业落地变体）。

4. **后训练轴**：三阶段 mid-train→SFT→agentic RL。RL 用 **CISPO**（[14]=[MiniMax-M1](../sources/minimax-m2-series.md)）asymmetric clip (1,4) + length-weighted LOO advantage，ablate vs GRPO/GSPO 选定。合成代码环境（git commit → 双端正确性检查的可验证任务）同时喂 SFT 与 RL。IF judge + multi-harness 训练（OpenHands/OpenCode2/Mini-SWE-Agent）保泛化。

## 同族 / 同重量级定位

| 模型 | 总参/激活 | 注意力 | RL 算法 | 开源 |
|---|---|---|---|---|
| **Laguna M.1** | 225.8B/23.4B | 全局 GA | CISPO | — |
| **Laguna XS.2** | 33.4B/3B | 3:1 SWA/GA + softplus 门控 | CISPO | Apache 2.0 |
| [Devstral 2](https://mistral.ai) | 123B dense / 225B-A23B | — | — | 部分 |
| [Qwen3.6-35B-A3B](https://arxiv.org/abs/2504.13750) | 35B/3B | — | — | 开源 |
| [Gemma 4 31B](../models/gemma-4.md) | 31B dense | 5:1 SWA/GA + key-as-value + p-RoPE | — | 开源 |
| [MiMo-V2-Flash](../models/mimo-v2-flash.md) | 309B/15B | 5:1 SWA/GA window 128 | MOPD | — |

Laguna 的差异化不在单一架构激进，而在**端到端工业化流程的复用速度**（M.1→XS.2 五周）+ 把 [AutoMixer](../concepts/data-mixture-optimization.md) / [Gated Attention](../concepts/attention-gating.md) / CISPO / 合成代码环境等已有成果组合进一套可配置流水线。

## 评测摘要

agentic（pool harness, temp 1.0, top_k 20, thinking on, 256K, 4 次均值）：

| Benchmark | M.1 | XS.2 |
|---|---|---|
| SWE-bench Verified | 79.6 | 73.4 |
| SWE-bench Multilingual | 73.3 | 67.2 |
| SWE-Bench Pro | 69.3 | 52.4 |
| Terminal-Bench 2.0 | 52.5 | 51.5 |

M.1 SWE-bench Verified 追平 Devstral 2（79.0）超 GLM-4.7（76.2）/DeepSeek-V4-Flash（74.6），Terminal-Bench 2.0 次于 Claude Sonnet 4.6（59.1）。XS.2 SWE-bench Verified 略胜 Qwen3.6-35B-A3B（73.3），Terminal-Bench 2.0 在同重量级开源中领先。报告诚实披露四 benchmark 均有 reward hacking 漏洞（git history 泄漏 / web 搜参考解），已 patch image 并自建 reward hack judge 后置检测，未发现显著作弊。

## 相关页面

- 来源：[Laguna M.1/XS.2 技术报告](../sources/laguna-m1-xs2.md)
- [MoE 前沿模型扩展](../concepts/moe-frontier-model-scaling.md)
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- [零样本 RoPE 上下文扩展](../concepts/zero-shot-rope-context-extension.md)
- [注意力门控](../concepts/attention-gating.md)
- [数据混合优化](../concepts/data-mixture-optimization.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [异步 Agent RL](../concepts/asynchronous-agent-rl.md)
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md)
