---
type: Model
title: "MiniLLM"
description: "清华 CoAI + MSR 在 ICLR 2024 发布的学生模型族，由 reverse-KLD on-policy 蒸馏（MiniLLM 方法）训练得到：GPT-2 120M/340M/760M、OPT 1.3B/2.7B/6.7B、LLaMA 7B 三族九档，teacher 分别为 GPT-2-1.5B、OPT-13B、LLaMA-13B，纯文本输入输出。"
tags: ["model", "minillm"]
timestamp: 2026-09-11
---

# MiniLLM

## 身份

论文 [MiniLLM：On-Policy Distillation of Large Language Models](../sources/minillm.md)（ICLR 2024）把用其蒸馏方法训出的 student 统一命名为 MiniLLM，并在仓库公开 code / data / checkpoints（<https://github.com/microsoft/LMOps/tree/main/minillm>）。它不是自研架构的新基座，而是**在既有开源基座上做 reverse-KLD on-policy 蒸馏得到的一组研究检查点**。

## 关键事实

| 项目 | 值 |
| --- | --- |
| 模态 | 纯文本（输入 prompt + 输出 response；论文未涉及视觉或音频） |
| 规模族 | GPT-2 族 120M / 340M / 760M；OPT 族 1.3B / 2.7B / 6.7B；LLaMA 族 7B |
| 对应 teacher | GPT-2-1.5B、OPT-13B、LLaMA-13B（各自先在指令数据上微调）；附录另用 GPT-J-6B 作 teacher |
| 训练方法 | 两阶段：先在 databricks-dolly-15K 上 SFT 取验证 loss 最低点作起点，再跑 reverse-KLD on-policy 优化（single-step decomposition + teacher-mixed sampling α=0.2 + length normalization + PPO clipping + 预训练语言建模损失）|
| 训练数据 | databricks-dolly-15K（过滤后约 12.5K 训练）；$\mathcal{D}_{PT}$ 为 OpenWebText（GPT-2 族）或 RoBERTa 语料 |
| 任务定位 | instruction following；论文未覆盖 agentic / 工具使用 / 长上下文或推理专用训练 |

## 技术身份说明

这批模型的价值在方法而非架构：它们是「把 reverse KL 当作蒸馏目标、并用 policy optimization 优化」这条路线在 120M–7B 规模上的对照实验载体。论文的主结果（[来源页](minillm.md)《实验设置与主结果》的 Table 1）显示 MiniLLM 相对 SFT w/o KD、word-level KD、SeqKD 在 5 个指令跟随评测集上普遍更优，且多数设置下 R-L 反超对应的 teacher（作者归因于 teacher 自身的 exposure bias）。

两点引用限制：其一，语料只到 LLaMA-7B student / LLaMA-13B teacher（论文摘要写成「120M 到 13B」是把 student 与 teacher 规模混装了，见来源页待追问）；其二，没有任何多 teacher 实验，因此不能把 MiniLLM 检查点当作 MOPD 场景的对照。

## 相关页面

- [MiniLLM：On-Policy Distillation of Large Language Models](../sources/minillm.md)：方法、三技巧、主结果表与机制分析。
- [GKD：On-Policy Distillation of Language Models](../sources/generalized-knowledge-distillation.md)：同期另一支源头，把发散度当超参而非固定为 reverse KLD。
- [Multi-Teacher On-Policy Distillation](../concepts/multi-teacher-on-policy-distillation.md)：跨家共用的 OPD 数学依据。
