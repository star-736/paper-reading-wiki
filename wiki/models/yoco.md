---
type: Model
title: "YOCO-3B / YOCO-3B-1M"
description: "Microsoft Research + 清华的 decoder-decoder 研究模型：13 层 gated-retention self-decoder + 13 层共享全局 KV 的 cross-decoder，1.6T 预训练后逐步扩至 1M。"
tags: ["model", "yoco", "long-context", "gated-retention"]
timestamp: 2026-09-12
---

# YOCO-3B / YOCO-3B-1M

## 身份

[YOCO 论文](../sources/yoco.md) 发布的命名研究模型与长上下文变体，团队为 Microsoft Research / Tsinghua University。YOCO 同时是架构名称；160M–13B 的短预算 scaling 配置不与 1.6T 训练的 3B 主模型混为一谈。

## 关键事实

下表已据原文 §2、§4.1–4.3、Appendix C/E 核实。

| 项目 | 内容 |
| --- | --- |
| 模态 | 纯文本输入、文本输出；多模态只在论文展望中出现 |
| 参数 | 3B 级 dense 模型，非 embedding 参数 2.83B |
| 深度 | 共 26 层，self-decoder / cross-decoder 各 13 层 |
| 宽度 | hidden 3072，FFN 8192 |
| 注意力 | 前半 gated retention，后半 GQA causal cross-attention，共享一份全局 K/V |
| Heads | query 24、KV 8、head dim 128 |
| Tokenizer | `tiktoken-cl100k_base`，词表 100,288 |
| 预训练 | 实际 1.6T tokens，4K 序列，4M tokens/batch；5T 是学习率计划 |
| 长窗版本 | 64K→256K→1M 继续训练，分别 6B/4B/1.5B tokens |
| 后训练 | 本报告无 agent SFT/RL/OPD；长窗扩展未使用 long-instruction tuning 数据 |
| 缓存边界 | 一份随长度增长的全局 KV，加各 self-decoder 层的固定大小递归状态 |

## 技术身份

前半负责生成共享全局 K/V，后半每层用自己的 Q 读取，仍逐层执行 attention 与 FFN。建立历史缓存不需要计算每个历史位置的后半输出，因此 prompt prefill 可精确提前退出；生成当前 logits 的位置仍需运行后半网络。

1M 单针检索接近满分，但 128K 的 8 针检索为 0.56，不能当作百万上下文任务全面解决。来源页保留短任务、长窗和推理计量边界。

## 相关页面

- [YOCO 来源页](../sources/yoco.md)
- [YOCO-CLSA 4B](yoco-clsa.md)：同团队后续 4B 研究模型，self-decoder 改为 SWA，后半用 CLSA 共享 routing index；不是本 3B 的继续训练。
- [DeepSeek-V4.1-Flash](deepseek-v41-flash.md)：其 CED 明确引用 YOCO，但新增逐层 SWA 与近似重放。
- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)、[百万 token 上下文服务](../concepts/million-token-context-serving.md)
