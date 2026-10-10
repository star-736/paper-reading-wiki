---
type: Source
title: "Rock Tokens：OPD 里下不来的高损失 token"
description: "UMBC 等（arXiv:2605.09253v4，2026-07-01，23 页）：OPD 饱和后仍有一类高损失 token。词表约占 6%，输出频率至多约 18%。因果干预里多数对推理贡献可忽略，其中约 1.5%–3.5% 被标成真正的 pillar。综述引用的是 v1。"
tags: ["source", "on-policy-distillation"]
timestamp: 2026-10-11
resource: "../../raw/2605.09253v4.pdf"
---

# Rock Tokens：OPD 里下不来的高损失 token

## 来源

- 原始 PDF：[`raw/2605.09253v4.pdf`](../../raw/2605.09253v4.pdf)
- 标题：Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation
- 版本 / 日期：arXiv:2605.09253v4，2026-07-01。[OPD 综述](opd-survey.md) 的参考文献写的是 v1
- 作者：Yuxuan Jiang、Runchao Li、Shubhashis Roy Dipta、Dawei Li、Zhao Yang（UMBC、Case Western、Arizona State、VU Amsterdam）
- 模型链接：**未建模型页**。学生是 Qwen3-4B-Instruct，teacher 是 Qwen3-30B-A3B-Instruct

## 核心结论

1. **Rock Token 是饱和之后仍然高损失、而且损失下不来的 token 类型。** 摘要写：它们可以占到生成输出里至多 18% 的 token。引言把词表占比写成大约 6%，把输出频率写成至多 18%。这不是「18% 的词表」。
2. **频率高，所以梯度份额大，但 token 自己几乎不学。** 损失按 token 类型拆开，\(R(v)=\bar\ell_v\cdot\mathrm{Freq}(v)\)。再乘上下文一致率得到 \(R_{\mathrm{ctx}}(v)\)，用来丢掉只在个别位置难的高频符号。Figure 3：rock 的逐 token KL 在早晚 checkpoint 上贴着对角线（\(\Delta\mathrm{KL}\approx 0\)）；稀有高 KL token 则明显下降。
3. **解码出来主要是结构和话语残差。** Top-100 用 Qwen3 tokenizer 解码后，集合在数值标准下仍聚成结构或话语标记。识别设定写的是 Qwen3-4B-Instruct 在 OpenThoughts 上从 Qwen3-30B-A3B-Instruct 做 on-policy 蒸馏（`§` Setup，约第 5 页）。
4. **「对推理可忽略」不能说成每一个都可忽略。** 摘要的因果干预结论是：这些 token 对实际推理性能的功能贡献可忽略，优化带宽花在学生不能或不需要内化的结构残差上。后文的 take-away 把推断时的因果必要性收窄：只有大约 1.5%–3.5% 是真正的 pillar，其余更像为了保住少数语义锚而留下的缓冲。两句同时成立时，引用必须带上这层分寸，不要只留综述的「全部可忽略」。

## 架构与训练

识别用的是学生轨迹上的逐 token KL，不是额外的奖励模型。Figure 2 在 \(N=500\) 条 MATH-500 轨迹上画频率和平均 KL：稀有 token 的 KL 方差极大，Rock Score 把稳定频段上沿的点挑出来。\(K=100\) 被写成在可重复性和大约 60% 蒸馏负担覆盖之间的折中。

效用实验（约第 10 页）把 teacher 写成 Qwen3-30B-A3B-Instruct-2507，学生写成 Qwen3-4B-Instruct-2507，两边都关 thinking。先有一段 off-policy，再进 OPD。Figure 5 比较原始 OPD、随机掩码和他们的梯度稀疏化，纵轴是 AIME24、AIME25、HMMT25 的平均准确率；每 200 step 对应 8000 条 prompt、每条 4 条 rollout。图上没有在正文里再列终点数字，不从像素读。

## 后训练

本文不是一条新的后训练配方发布。它诊断的是已经在跑的 OPD 里，哪些 token 的损失饱和后仍占着梯度。

## 评测要点

- 梯度几何：rock 的 logit 梯度幅度比稀有高 KL token 小一个数量级，但与频率均衡下降方向的余弦是正的，有一条尾巴超过 0.3。
- 稀疏化梯度的主张是：拿掉大部分 rock 的训练信号，平均准确率仍能跟上原始 OPD。随机对照在同一张图上。终点分以图为准，正文没有另附表。

## 证据边界与阅读提示

- **Figure 2(b) 内部不一致。** 图注写逐序列 rock 密度的中位数是 18.5%；图内标注写 median \(=52.5\%\)。摘要的「至多 18%」不要换成这两个中位数里的任何一个。
- v4 相对综述所引的 v1 可能改过检测定义。本页数字来自 v4。
- 识别设定的模型名没有写 2507，效用实验写了 2507。不要把两处自动当成同一次 checkpoint。
- 原文没有 Qwen3-4B / OpenThoughts 以外的第二套模型族。不能据此说 rock 集合在其他模型族上仍是话语标记。

## 待追问

- **数字相互矛盾**：Figure 2(b) 的中位数，图注 18.5% 与图内 52.5% 不能同时当引用值。
- **要找原文口径**：pillar 的 1.5%–3.5% 是在哪一次 knockout、哪一个 benchmark 上数出来的。take-away 没有把分母写死。

## 相关页面

- [OPD 综述](opd-survey.md)
- [ExOPD](exopd.md)
- [The Many Faces of OPD](many-faces-opd.md)
- [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md)
