---
type: Source
title: "Local teachability collapse"
description: "浙大 + 美团 LongCat（arXiv:2605.13643v2，2026-05-25，29 页）：强到弱 OPD 里，轨迹后段的 teacher 局部对比度会掉下去，尽管平均 advantage 还可以非零。释放规则在句段 margin 的 BIC 下变点处停掉稠密监督。Qwen3-1.7B 上数学均分 36.8→40.1。不是「teacher 仍然校准」。"
tags: ["source", "on-policy-distillation"]
timestamp: 2026-10-11
resource: "../../raw/2605.13643v2.pdf"
---

# Local teachability collapse

## 来源

- 原始 PDF：[`raw/2605.13643v2.pdf`](../../raw/2605.13643v2.pdf)
- 标题：Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation
- 版本 / 日期：arXiv:2605.13643v2，2026-05-25
- 作者：Kaiyuan Liu、Ziyuan Zhuang、Yang Bai、Bing Wang、Rongxiang Weng、Jieping Ye（浙江大学、美团 LongCat、吉林大学）
- 模型链接：**未建模型页**。主设定学生是 Qwen3-1.7B-Non-Thinking，teacher 是 Qwen3-30B-A3B-Instruct-2507

## 核心结论

1. **失败模式是局部对比度塌掉，不是「teacher 还校准、只是 top-K 变平」这句综述转述。** 论文把 local teachability collapse 定义成：沿学生 rollout，teacher 的局部对比度持续下降。后段的平均 teacher–student advantage 仍可以非零，但采样 token advantage 的区分力变差。文中 “calibration” 出现在对 [Uni-OPD](uni-opd.md) 的转述里，指的是用结果去校准轨迹级回报，不是本篇测到 teacher 仍然校准。
2. **监督应停在 teacher 反馈还分得出高下的区段。** 释放规则量的是 teacher 的 top-1 与 top-2 在学生 top-K 候选上的 margin，按 NLTK 句段聚合，再用 BIC 式下变点截断稠密 OPD。变点之前是标准 OPD advantage，之后掩掉稠密监督。作者把它写成启发式，不是「变点之后每个 token 都有害」的定理。
3. **主表没有把全轨迹 OPD 写成最优。** 学生 Qwen3-1.7B-Non-Thinking，teacher Qwen3-30B-A3B-Instruct-2507，prompt 来自 DeepMath（Table 1）。五项数学无权重均分：学生 24.9，全轨迹 OPD 36.8，他们复现的 ExOPD 39.0，最好的固定前缀（4096）39.4，轨迹特异释放 40.1。相对全轨迹 OPD 是 \(+3.3\)。GPQA（不进均分）从 OPD 的 40.7 到 42.4，ExOPD 是 36.6，teacher 是 70.4。
4. **这不是加速生成。** 实现要先做完整 rollout 的诊断再选监督区，作者写明不能当成和 FastOPD 一样的效率主张。

## 架构与训练

OPD 仍是学生自己的回复上、用 teacher 的稠密 advantage。ExOPD 基线改的是目标（reward extrapolation）；释放规则不改目标，只改监督覆盖到哪。固定前缀扫过 \(k\in\{1024,2048,4096,8192\}\)。随机释放保持 batch 内释放点的分布，但在样本间打乱，均分 38.0，低于轨迹特异规则。

Figure 3：多数样本的 RSS-BIC 增益为正，均值 24.0；79% 的样本接受增益大于 6 的下变点。接受样本上，句段 margin 从释放前掉到释放后。相对释放位置均值 0.25、中位数 0.13，所以不是一个固定前缀。

## 后训练

没有新的多阶段后训练。实验接在既有强到弱 OPD 设定上，引用 ExOPD 作为强到弱框架。

## 评测要点

| 方法 | 数学均分 | GPQA |
| --- | ---: | ---: |
| Teacher | 62.2 | 70.4 |
| Student | 24.9 | 31.1 |
| 全轨迹 OPD | 36.8 | 40.7 |
| ExOPD（本文复现） | 39.0 | 36.6 |
| 固定前缀 4096 | 39.4 | 38.6 |
| 轨迹特异释放 | 40.1 | 42.4 |

表内 ExOPD 行不能和 [ExOPD 原文](exopd.md) 的 4B 主表相减。原文主表是同基座双教师、\(\lambda=1.25\)；这里是 30B-A3B 教 1.7B 的强到弱复现。

## 证据边界与阅读提示

- 五项数学里，固定前缀已经能超过全轨迹 OPD。轨迹特异规则相对最好的固定前缀只高 0.7，优势是不必先扫一个全局长度。
- GPQA 上高于表内蒸馏基线，仍然远低于 teacher 的 70.4。
- 附录 A.8 写了更多域。主结论的数字以 Table 1 为准。
- 主实验没有在代码或 agent 轨迹上检验句段 margin 的下变点是否早于答案 token。主实验是 DeepMath 上的强到弱数学。
- 原文没有和 [Prune-OPD](prune-opd.md) 在同一学生、同一任务上的对照。

## 相关页面

- [OPD 综述](opd-survey.md)
- [ExOPD](exopd.md)
- [Uni-OPD](uni-opd.md)
- [Prune-OPD](prune-opd.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
