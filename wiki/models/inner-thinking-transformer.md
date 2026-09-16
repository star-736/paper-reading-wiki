---
title: "ITT 实验模型族"
type: Model
description: "百度与中科院等的 LLaMA2 风格层内循环实验模型；主体为 162M／230M／466M，ATR 与 RTC 实现 token 级额外计算"
tags: [ITT, looped-transformer, adaptive-computation]
timestamp: 2026-09-17
---

## 身份

**Inner Thinking Transformer（ITT）** 是架构方法及其研究用模型族。百度、中科院信息工程研究所、国科大与北京师范大学合作，发表于 ACL 2025。以下事实据[来源页](../sources/inner-thinking-transformer.md)所引官方 PDF §3–4、Table 5 直接核对；尚无本地 `raw/` 副本。

## 关键事实

| 项目 | 内容 |
| --- | --- |
| 模态 | 纯文本输入与文本输出；据语言建模训练目标及文本评测核对 |
| 主体参数规模 | 162M／230M／466M |
| 主干 | LLaMA2 风格，8 层，从随机初始化预训练 |
| 循环位置 | 每隔一层替换成 ITT 层，在该层内复用参数 |
| 思考配置 | ×2／×3／×4；包含第一次全量前向 |
| 计算分配 | ATR 选择 token，RTC 累积结果，步骤编码做乘性调制 |
| 预训练 | RedPajama 50B tokens，序列长 4096 |
| 身份边界 | 不是官方 LLaMA2-7B 的别名，也不是已经核实公开权重的产品模型 |

## 技术身份

ITT 的主干权重在同一层的额外步骤中复用，token 参与额外步骤的集合由 router 决定。这使它在同参数条件下增加有效计算深度，但不保证同 FLOPs 或同延迟。当前主表中 162M ITT ×4 平均分为 42.1，普通 466M 为 43.6；所谓 96.5% 是平均分之比。

正式 PDF 摘要出现 355M／1B／3B，附录另列 1.88B 及等效层数实验，与主体三档规模不统一；相关数字冲突集中记录于[来源页的证据边界](../sources/inner-thinking-transformer.md#消融与正式版的证据边界)，不据此合成一个统一的大模型家族。

## 相关页面

- [ITT 论文来源](../sources/inner-thinking-transformer.md) — 架构图、训练与主表、附录冲突。
- [Looped Transformers](../concepts/looped-transformers.md) — 层内循环与 token 自适应计算的位置。
