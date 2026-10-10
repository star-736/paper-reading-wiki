---
type: Source
title: "Healthcare AI GYM 与 TT-OPD"
description: "Upstage（arXiv:2605.02943v1，2026-05-01，22 页）：多轮医疗 agent 上的自蒸馏。周期把 teacher 权重复制成学生时，KL 在一步内从 2.637 掉到 0.343，因为两套权重变成同一份，蒸馏梯度被拆掉。这不是学生前缀把 teacher 带离分布。"
tags: ["source", "on-policy-distillation", "self-distillation"]
timestamp: 2026-10-11
resource: "../../raw/2605.02943v1.pdf"
---

# Healthcare AI GYM 与 TT-OPD

## 来源

- 原始 PDF：[`raw/2605.02943v1.pdf`](../../raw/2605.02943v1.pdf)
- 标题：Healthcare AI GYM for Medical Agents
- 版本 / 日期：arXiv:2605.02943v1，2026-05-01
- 作者：Minbyul Jeong（Upstage AI）
- 代码：首页只写 “GitHub: HEALTHCARE AI GYM Repository”，没有给出 URL
- 模型链接：**未建模型页**。骨干是 Qwen3.5-9B，方法名是 TT-OPD

## 核心结论

1. **KL 2.637→0.343 是权重复制，不是输出分布自己塌成一个模式。** `§6.1` 消融（1）：teacher 每 \(T\) 步被换成学生权重（\(\theta_T\leftarrow\theta_S\)）。复制发生时 KL 从累积值掉到接近 0。例子是 step 10、\(T=30\) 时 \(2.637\to 0.343\)。附录把机制写死：重置前 teacher 是一份旧快照，KL 随学生变好而累积；重置瞬间两者相同，KL 掉向 0，蒸馏梯度被拆掉。随后准确率从 56.9% 单调降到 49.3%，每回合工具轮数从 7.65 降到 5.52。综述里的「输出崩掉」是转述，不是这一个 KL 事件的原文措辞。
2. **这和前缀把 teacher 带离分布不是同一件事。** [Revisiting OPD](revisiting-opd.md) 和 [Many Faces](many-faces-opd.md) 量的是学生前缀上 teacher 条件分布失准。这里 teacher 被硬换成学生参数，所以 KL 变小是因为两个策略变成同一个。不要把三篇收成同一次测量。
3. **完整 TT-OPD 用 EMA，复制消融才去掉 EMA。** 正式更新是 \(\theta_T\leftarrow 0.995\,\theta_T+(1-0.995)\theta_S\)，每 5 步一次。正文另外写了每 30 步的 hard-copy 兜底，用来防止 teacher 和学生分得太开。Figure 4 里造成 KL 锯齿的是「只有周期复制、没有 EMA」那一支。换成 EMA（\(\alpha=0.995\)、无结果条件）后，KL 不再塌，准确率在 step 40 到 53.8%（\(+1.2\)），轮数仍从 7.82 掉到 6.23。
4. **损失是 reverse KL，系数不是 ExOPD 的 \(\lambda\)。** 公式 4：\(\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{GRPO}}(\theta_S;R_{\cos})+\lambda_{\mathrm{distill}}\,D_{\mathrm{KL}}(\pi_{\theta_S}\|\pi_{\theta_T})\)。Algorithm 1 写 \(\lambda_{\mathrm{distill}}=4.0\)。\(D_{\mathrm{KL}}(\pi_S\|\pi_T)\) 是 student 对 teacher 的 reverse KL。这个系数不是 [GKD](generalized-knowledge-distillation.md) 的数据混合比例，也不是 [ExOPD](exopd.md) 的 reward scale。

Teacher 看带结果提示的状态，学生看不到提示。提示只改 teacher 的分布，再通过 KL 传回来。「turn-level」指整段轨迹上的每一轮都算 KL，不是只在最后一轮；「truncated」指超过 \(L_{\max}\) 的轮直接丢掉。\(L_{\max}=12288\)。

## 架构与训练

环境是 gymnasium 兼容的医疗 agent：10 个临床域、3.6K+ 任务、135 个工具、828K 篇段落。奖励在回合结束时给出。训练栈写的是 verl。动态过滤只留组内结果有分有合的 prompt。

四支消融是累加的：只有周期复制 → EMA → 再加上结果提示（无长度控制，step 40 时 91.7% 的回复顶到长度上限，准确率掉到 49.0%）→ 完整 TT-OPD 再加上余弦长度奖励。完整方法在 60 step 验证准确率 61.1%，轮数维持 7.0–7.4，长度 5.7K–9.3K。同图上 GRPO 峰值 62.0%（step 55），长度在 7.7K–10.8K 之间晃。

## 后训练

TT-OPD 是 GRPO 加上上述自蒸馏项，不是替代 GRPO 的独立后训练阶段。基线包括不训 RL 的多轮 AgentRunner（Base+AR），用来把工具和知识库的贡献从 RL 里分开。

## 评测要点

单轮生成在这些基准上被写成 0 分，因为流程在 `submit_answer` 之前被截断。所以主表用训练时的多轮 AgentRunner。引言写 MedQA 87.1%（相对 base \(+16.4\)）、MedMCQA 66.2%、MIMIC-III 62.7%，并写 18 项里 10 项最好、相对非 RL 基线平均 \(+3.9\)。`§5` 的 key findings 改写成 12/18。结论又写回 10/18。MMLU 医疗子集：文本 logprob 83.8%，Base+AR 60.6%，TT-OPD 65.5%。

## 证据边界与阅读提示

- 摘要 / 结论的 10/18 和 `§5` key findings 的 12/18 不能当成同一句的舍入。
- EMA 主方法里的 30 步 hard-copy 兜底，和消融（1）的「只有复制」不是同一条曲线。引用 2.637 时要标明是消融（1）。
- 骨干是 Qwen3.5-9B。不要和 Qwen3-8B 的 [SDPO](sdpo.md) 表相减。
- 周期复制把 KL 打到接近 0，只在 HEALTHCARE AI GYM 的多轮工具环境里测过。原文没有换环境再测。

## 待追问

- **数字相互矛盾**：最好成绩是 10/18 还是 12/18，摘要、`§5` 和结论不一致。

## 相关页面

- [OPD 综述](opd-survey.md)
- [Revisiting On-Policy Distillation](revisiting-opd.md)
- [The Many Faces of OPD](many-faces-opd.md)
- [SDPO](sdpo.md)
- [ExOPD](exopd.md)
- [GKD](generalized-knowledge-distillation.md)
