---
type: Model
title: "DeepSeek-V4.1-Flash"
description: "552B backbone + 196B Engram 的原生图文 MoE；CED 使 prefill/decode 激活量分别为 8B/16B，CSA2 + FP4 全局 KV 为 890 bytes/token。"
tags: ["model", "deepseek-v41-flash", "multimodal", "moe", "csa2"]
timestamp: 2026-09-10
---

# DeepSeek-V4.1-Flash

## 身份

DeepSeek-AI 面向长周期、输入占比高的 agent 工作负载开发的原生多模态模型。本页事实已据[本地技术报告](../sources/deepseek-v41-flash.md) §2、§4、§5 重读核实；评测和部署数字均为该版本报告口径。

## 关键事实

| 项目 | 内容 |
| --- | --- |
| 模态 | 图像 + 文本输入，文本自回归输出（§2.1）；未据此扩展为音频或视频原生支持 |
| 参数 | 552B backbone，另有 196B Engram；两项不能混写为“总共 552B” |
| 激活 | prefill 8B/token，decode 16B/token；长输入 prefill 另需末尾 SWA replay |
| 上下文 | 1M tokens |
| 网络 | 20 层 causal encoder + 20 层 decoder，hidden 5120 |
| MoE | 每层 1 shared + 384 routed，激活 6 routed experts |
| 注意力 | 前两层只用 SWA，其他层 CSA2 + SWA；encoder 压缩率 2，decoder 压缩率 1，top-k 512，SWA 128 |
| KV | 全局 KV 890 bytes/token，main KV FP4，SWA FP8；不是总运行显存 |
| 视觉 | DeepSeek-ViT + 3×3 pixel-unshuffle + MLP |
| 条件记忆 | 196B Engram，零索引层 1 和 14；省略短卷积 |
| 预训练 | 45T tokens，从 64K 稀疏训练开始；34T 时扩展 1M；文本/多模态 token 比 7:1 |
| 后训练 | SFT→RL→最终全词表 OPD，超过 40 个异构 teacher |
| 推测解码 | 独立训练 DSpark，3 blocks、5 个 draft 位置；backbone 预训练不使用 MTP |

## 技术身份

CED 让 decoder 的全局 KV 从 encoder 末端直接投影，因此大部分 prompt 不需跑完整 decoder；CSA2 将“重建 KV”和“重选索引”解耦，并用 Full / Reindex / Reuse 静态层模式共享缓存。分层索引只限制后续 decoder 索引器的候选范围，首个索引器仍扫描完整历史（§2.2–2.3）。

运行时全局缓存压缩与持久缓存策略分属两层：前者由 CSA2 + FP4 实现，后者通过短期 encoder SWA 内存池及缺失后的近似重放实现。作者明确承认重放不与完整前向数学等价，长程检索和缓存恢复边界仍需压力测试（§3.2、§6）。

Table 3 报 Terminal-Bench 2.1 90.6、DeepSWE v1.1 74.2、Automation-Bench 54.8；其中前两项分别使用 DeepSeek Harness Minimal 和 mini-SWE。同一模型在其他 harness 上分数不同，详细比较和证据限制见来源页。

## 相关页面

- [DeepSeek-V4.1-Flash 技术报告](../sources/deepseek-v41-flash.md)
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)、[百万 token 上下文服务](../concepts/million-token-context-serving.md)
- [条件记忆](../concepts/conditional-memory.md)、[多 token 预测](../concepts/multi-token-prediction.md)
- [OPD 跨报告对比](../comparisons/on-policy-distillation.md)
