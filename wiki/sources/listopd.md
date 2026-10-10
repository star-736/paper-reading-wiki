---
type: Source
title: "ListOPD：结构化输出上的外推悬崖"
description: "NTU（arXiv:2605.08737v1，2026-05-09，40 页）：把 ExOPD 的 reward-extrapolation λ 接到近确定的 listwise JSON。闭式 clip-safe 阈值 λ⋆(p,b,c) 之上，训练从保格式变成塌格式。这个 λ 是 ExOPD 的 reward scale，不是 GKD 的数据混合比例。"
tags: ["source", "on-policy-distillation", "exopd"]
timestamp: 2026-10-11
resource: "../../raw/2605.08737v1.pdf"
---

# ListOPD：结构化输出上的外推悬崖

## 来源

- 原始 PDF：[`raw/2605.08737v1.pdf`](../../raw/2605.08737v1.pdf)
- 标题：The Extrapolation Cliff in On-Policy Distillation of Near-Deterministic Structured Outputs
- 版本 / 日期：arXiv:2605.08737v1，2026-05-09。页眉为 Preprint
- 作者：Xin Li、Hao Jiang、Annan Wang、Yichi Zhang、Chau Yuen（南洋理工大学）
- 项目页：<https://lixin.ai/ListOPD>
- 模型链接：**未建模型页**。学生是已有的 Qwen3-1.7B，不发布新模型

## 核心结论

1. **这里的 \(\lambda\) 就是 [ExOPD](exopd.md) 的 reward extrapolation，不是 [GKD](generalized-knowledge-distillation.md) 的数据混合比例。** 正文写明把 ExOPD（arXiv:2602.12125）的外推公式从推理接到 listwise 结构化排序（`§` 引言后的相关工作段，参考文献 [42]）。\(\lambda>1\) 把 on-policy 目标变尖，学生可以在域内超过 teacher；过了 \(\lambda^\star\)，同一步在结构化输出上改成违反输出契约。
2. **单点 Bernoulli 上有闭式 clip-safe 阈值。** 与基座无关的标记是

\[\lambda^\star(p,c)=\frac{\log\big((1-p)/(c-1+p)\big)}{\log\big((1-p)/p\big)}.\]

\(p\) 是 teacher 在结构 token 上的模态概率，\(c\) 是 GRPO 式 importance-sampling clip 强度。完整定理是带 warm-start 质量 \(b\) 的 base-relative 形式（Theorem 4.1）。序列级提升在 off-modal-ratio 不变时精确，否则是近似；超临界动力学是有限预算上的经验，不是几乎必然收敛（Theorem 4.2–4.3）。
3. **Fashion 的 K=8 JSON 把悬崖落在预测带里。** 结构位 \(p_{\mathrm{typ}}=0.9993\)、\(c=5\) 时，中性标记 \(\lambda^\star=1.22\)；\(b=0.81\) 的区间是 \([1.18,1.28]\)（Figure 1、Table 1）。五种子细网格把 onset 定在 \([1.204,1.228]\)。\(c=1.5\) 的交叉预测闭式 \(\lambda^\star_{\mathrm{typ}}=1.070\)，观测中点 1.069。
4. **涨的是格式，不是解析后的排序。** Qwen3-1.7B-SFT 在 Fashion 上严格解析率 33.5%，ListOPD 到 94.8%。解析成功样本上的 NDCG@1 跨 \(\lambda\) 持平。USEFUL \(=\) parse \(\times\) NDCG@1 从 0.23 到 0.86，与预注册 3 种子的 8B-SFT 基线在合并种子噪声内打平。Figure 1 把这对写成 Qwen3 1.7B\(\times\)4B。悬崖诊断不依赖评分细则；对 8B 的 parity 用 Gemini 细则，继承那个评判器的暴露。

## 架构与训练

任务是 Amazon Fashion 的 listwise 重排：每组 \(K=8\) 条评论，输出固定 `review_id` 的 JSON 列表，每条带 helpfulness 分。契约失败是外层脚手架截断或重复 id，不是分数看起来不合理。

训练仍是学生 rollout 上、对着 teacher 的逐 token 对数概率做外推 OPD。ListOPD 不另换散度族，改的是 \(\lambda\) 停在 clip-safe 区的哪一侧。ASPO 被写成同一道 IS-clip 不对称的工程缓解，附录里悬崖早一个网格步，作者用来支持「机制不依赖这一个方法名」。

## 后训练

没有单独的生产后训练流水线。实验是在 SFT 学生上做外推 OPD。

## 评测要点

- 五种子细网格、\(N=200\) 的预算延伸、以及 \(c=1.5\) 的小 clip 交叉预测，都被写成落在预先锁死的窗口里。
- 相对「最好的约束 SFT + 置换修复」，训练侧 USEFUL 残差是 \(+0.051\)。作者把这句当成自己的主张，而不是声称全面优于约束解码。
- 主实验学生是 1.7B。8B 只作为 SFT 基线，不是第二个 OPD 学生。

## 证据边界与阅读提示

- **不要把 Fashion 的 \([1.18,1.28]\) 套到 ExOPD 主表的 \(\lambda=1.25\)。** 两边同属 reward scale，但 ExOPD 的主实验是数学和代码，不是近确定 JSON；clip 强度和模态概率也不是这组 Fashion 测量。数值靠近不是同一道悬崖的复现。
- 超临界之后的动力学，作者自己标成有限预算经验。
- parity 依赖 Gemini 评分。格式悬崖本身用严格解析率，不走那条细则。
- 原文没有在数学或代码上检验 \(\lambda^\star(p,b,c)\) 是否还有同一道格式悬崖。校准锚是 Fashion JSON。

## 相关页面

- [ExOPD](exopd.md)
- [OPD 综述](opd-survey.md)
- [GKD](generalized-knowledge-distillation.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
- [Uni-OPD](uni-opd.md)
