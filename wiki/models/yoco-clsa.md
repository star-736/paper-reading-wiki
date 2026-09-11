---
type: Model
title: "YOCO-CLSA 4B"
description: "YOIO 报告里的 4B 研究模型：16 层 SWA self-decoder + 16 层共享 KV 与 routing index 的 CLSA cross-decoder，k=2048，纯文本。"
tags: ["model", "yoio", "clsa", "yoco", "long-context"]
timestamp: 2026-09-12
---

# YOCO-CLSA 4B

## 身份

[YOIO](../sources/yoio.md) 在同一套 4B 配置上对照的三个检查点之一，也是该方法的主模型。论文写作是 YOCO (CLSA)；另两个对照是同宽同深的 Transformer 与 YOCO (Dense)。团队为 Microsoft Research / Tsinghua University。它不是 [YOCO-3B](yoco.md) 的继续训练，self-decoder 也从 gated retention 换成了窗口 512 的 SWA。

## 关键事实

下表已据原文 §3.1、Appendix B/C、Table 2/5–8/11–12 核实。

| 项目 | 内容 |
| --- | --- |
| 模态 | 纯文本输入、文本输出；报告无视觉或音频实验 |
| 参数 | 4B 量级 dense；未给精确非 embedding 参数或词表 |
| 深度 | 共 32 层，self-decoder / cross-decoder 各 16 层 |
| 宽度 | hidden 2560，FFN 7680 |
| 注意力 | 前半 SWA（窗口 512，RNoPE 的 RoPE 路径），后半 GQA + CLSA，共享一份全局 KV 与一份 top-k index，$k=2048$ |
| Heads | query 20、KV 4、head dim 128；indexer 单头 |
| 预训练 | 稠密 1.0T @ 8K + 80B @ 32K；sparse adaptation 再 40B @ 32K（步数 × 8M tokens 的本页折合） |
| 后训练 | 无 SFT / RL / OPD |
| 缓存边界 | 一份随长度增长的全局 KV，加各 self-decoder 层的 SWA 窗口；routing index 全栈共享一次 |

## 技术身份

前半写出共享 $H/K/V$ 与 $K_{\mathrm{idx}}$；后半每层用自己的 $Q$ 读同一组被选 token。建立历史缓存仍可跳过 prompt 位置的 cross-decoder，这一点继承 YOCO。Decode 时 top-k 只对当前 token 的 $Q_{\mathrm{idx}}$ 算一次。

短任务上与稠密对照接近，GSM8K / DROP 略高、WinoGrande 略低。长上下文质量只评到 32K RULER；128K 的 7.6× decode / 17.1× 端到端是 B200 + vLLM 相对同配置 Transformer 的吞吐，不是 128K 任务分数。

## 相关页面

- [YOIO 来源页](../sources/yoio.md)、[YOCO-3B](yoco.md)
- [跨层索引复用](../concepts/cross-layer-index-reuse.md)、[高效长上下文注意力](../concepts/efficient-long-context-attention.md)
