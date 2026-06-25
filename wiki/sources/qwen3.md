---
type: Source
title: "Qwen3 技术报告"
description: "Qwen 系列 2025-05 基座报告（arXiv:2505.09388），标准 GQA + RoPE + RMSNorm + MoE，36T tokens / 119 语言；后训练核心 = 统一 thinking/non-thinking 双模式 + thinking budget + Strong-to-Weak Distillation 完胜 RL（1/10 GPU 时长）。Qwen3-Next/3.5/3-Coder-Next/3.5-Omni/Qwen3-VL 的基座前作。"
tags: ["source", "qwen3"]
timestamp: 2026-06-22
resource: "../../raw/Yang 等 - 2025 - Qwen3 technical report.pdf"
---

# Qwen3 技术报告

## 来源

- 文件：`raw/Yang 等 - 2025 - Qwen3 technical report.pdf`
- 标题：Qwen3 Technical Report
- 团队 / 日期：Qwen Team（Alibaba），2025-05-15（arXiv:2505.09388v1，2025-05-14 提交）
- 模型链接：[Qwen3 on HuggingFace](https://huggingface.co/Qwen)、[QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)
- 系列定位：Qwen3 是 **Qwen3-Next / Qwen3.5 / Qwen3-Coder-Next / Qwen3.5-Omni / Qwen3-VL 的基座前作**——架构上还是「标准 GQA + RoPE + RMSNorm + SwiGLU + MoE」一脉，hybrid linear attention（GDN）/ gated full attention 那一套是后续 Qwen3-Next 才引入的。本报告本身**不含** linear attention 或 gated attention。

## 核心结论

1. **统一 thinking / non-thinking 模式**：Qwen3 把 chat 模型与推理模型（如 QwQ）合并到一个权重里，由 `/think`、`/no_think` flag 切换，并提供 **thinking budget** 让用户在推理时限定思考 token 数（达到阈值时由系统插入 `Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>` 截断思考、强制生成最终答案）。
2. **强基座**：旗舰 Qwen3-235B-A22B（MoE，22B 激活）AIME'24 85.7 / AIME'25 81.5 / LiveCodeBench v5 70.7 / CodeForces 2056 / BFCL v3 70.8。
3. **多语言扩到 119**：Qwen2.5 的 29 种 → Qwen3 的 119 种语言及方言，预训练数据从 18T → **36T tokens**。
4. **Strong-to-Weak Distillation 完胜 RL**：小模型走「off-policy distillation → on-policy distillation」，相比直接走完整 4 阶段 RL，**性能更好且 GPU 时长仅 1/10**（Qwen3-8B 实验：RL 17,920 GPU·h vs on-policy distill 1,800 GPU·h，AIME'24 67.6 → 74.4、AIME'25 55.5 → 65.5、pass@64 90.0 → 93.3）。on-policy distill 的 teacher 是 **Qwen3-32B 或 Qwen3-235B-A22B**（单 teacher，§4.5 原文），与 MiMo/DeepSeek-V4 的多 teacher 融合派是不同用法——详见 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md) 轴一第二类。**注意 Table 21 这条数据反过来被 Thinking Machines Lab Kevin Lu 当作 OPD 算法解释的实验依据复现**——博客自承"served as inspiration for our work"，详见 [Thinking Machines Lab On-Policy Distillation 博客源页](thinking-machines-on-policy-distillation.md)。

## 架构

> Qwen3 dense 与 MoE 都沿用 **Qwen2.5 风格的标准 Transformer**——GQA + SwiGLU + RoPE + RMSNorm + pre-norm，**没有** linear attention / sliding window / hybrid stack（那是 Qwen3-Next 才引入的）。本报告唯二的注意力侧改动是：

- **移除 QKV-bias**（Qwen2 用过，Qwen3 去掉）。
- **引入 QK-Norm**（Dehghani et al., 2023）以稳定训练——在 Q 和 K 上各加一个 RMSNorm。

MoE 部分相对 Qwen2.5-MoE 的改动：

- **去掉 shared expert**（Qwen2.5-MoE 有，Qwen3 取消）。
- **fine-grained expert segmentation**（沿用 Dai et al., 2024 / DeepSeek-MoE）。
- **global-batch load balancing loss**（Qiu et al., 2025）替换 micro-batch 平衡，鼓励专家专业化。
- 128 总专家 / 8 激活。

Tokenizer：Qwen 自家 BBPE，词表 151,669。

### 变体表（已据报告 Table 1 / Table 2 核实）

| 变体 | Layers | Q / KV heads | Experts (total / active) | Tie embed | Context |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 28 | 16 / 8 | — | Yes | 32K |
| Qwen3-1.7B | 28 | 16 / 8 | — | Yes | 32K |
| Qwen3-4B | 36 | 32 / 8 | — | Yes | 128K |
| Qwen3-8B | 36 | 32 / 8 | — | No | 128K |
| Qwen3-14B | 40 | 40 / 8 | — | No | 128K |
| Qwen3-32B | 64 | 64 / 8 | — | No | 128K |
| Qwen3-30B-A3B | 48 | 32 / 4 | 128 / 8 | — | 128K |
| Qwen3-235B-A22B | 94 | 64 / 4 | 128 / 8 | — | 128K |

注意 KV head 数全部固定为 8（dense）或 4（MoE）——典型 GQA，KV head 远小于 Q head 但都是**完整 head dim**，与 MLA（低秩 latent）正交。

## 预训练

**三阶段 36T tokens，119 种语言**：

- **S1 General Stage**：>30T tokens，seq_len 4096，通识语言/世界知识。
- **S2 Reasoning Stage**：~5T 高质量 tokens，seq_len 4096，提高 STEM / 代码 / 推理 / 合成数据占比，**LR decay 加速**。
- **S3 Long Context Stage**：数百 B tokens，seq_len **4096 → 32768**。长上下文语料 75% 落在 16k–32k、25% 落在 4k–16k。RoPE 基频 **10,000 → 1,000,000**（ABF 技术，Xiong et al., 2023）；推理时再叠 **YARN + Dual Chunk Attention (DCA)**，把有效序列长度再放大 4×（即 128K）。

数据扩张技巧：

- **PDF 文本回收**：把 **Qwen2.5-VL** finetune 成 OCR/extraction 工具，对大规模 PDF 文档抽文，再用 Qwen2.5 提质，"额外得到数万亿 token"。
- **领域专用合成数据**：Qwen2.5-Math / Qwen2.5-Coder 合成数万亿 token 的教材、QA、指令、代码片段。
- **instance-level data mixture**：用一个轻量 Qwen 标注器给 >30T tokens 在「教育价值 / 领域 / 安全」多维度打标，按 instance 而非 domain 优化数据混合。

## 后训练

> 「Thinking Control + Strong-to-Weak Distillation」是这篇报告的后训练核心，4 阶段 pipeline 仅对**旗舰**（235B-A22B / 32B）走完整流程；其余 6 个轻量模型（30B-A3B / 14B / 8B / 4B / 1.7B / 0.6B）走 **2 阶段蒸馏**——off-policy → on-policy。

![Qwen3 post-training pipeline：旗舰模型（Qwen3-235B-A22B / Qwen3-32B）从 Base 出发顺序经过 Stage 1 Long-CoT Cold Start → Stage 2 Reasoning RL → Stage 3 Thinking Mode Fusion → Stage 4 General RL；其余轻量模型（Qwen3-30B-A3B / 14B / 8B / 4B / 1.7B / 0.6B）则从各自的轻量 Base 出发，由训完的旗舰模型作为 teacher 走 Strong-to-Weak Distillation 单独成线，不重复 4 阶段。](../assets/qwen3/fig1-post-training-pipeline.png)

> 论文 Figure 1 原文标题："Post-training pipeline of the Qwen3 series models."（§ 4 Post-training 引言图）

### Stage 1：Long-CoT Cold Start

- 数据：math / code / 逻辑 / STEM，每问题配可验答案或 test case。用 Qwen2.5-72B-Instruct 过滤掉「无 CoT 也答对」的问题（避免学到肤浅猜测），用 QwQ-32B 生 N 候选解，再按 6 条规则筛除（错答、重复、猜测、思考与摘要不一致、语言混杂、与验证集过近）。
- 目标显式声明："instill foundational reasoning patterns ... without overly emphasizing immediate reasoning performance"——故意训得克制，留出后续 RL 提升空间，**首选最少样本与最少步数**。

### Stage 2：Reasoning RL

- 仅 **3,995 query-verifier 对**（精挑：未在 cold-start 用过、对冷启 model 可学、尽量难、覆盖广）。
- 算法 **GRPO**（Shao et al., 2024）。
- 工程 insight：大 batch + 单 query 多 rollouts + off-policy 训练提样本效率；显式控熵稳定增长或持平。
- 结果：单跑 170 步、零手调超参，Qwen3-235B-A22B 在 AIME'24 上 **70.1 → 85.1**。

### Stage 3：Thinking Mode Fusion

继续 SFT（在 Stage 2 reasoning model 上）把 non-thinking 能力融进去：

- **数据来源不对称**：thinking 数据由 **Stage 2 model 自己** rejection sampling 出来（确保 reasoning 能力不退化）；non-thinking 数据则人工 curate 多任务覆盖（代码、数学、指令、多语、写作、QA、角色扮演），用自动 checklist 评分。
- **Chat template 设计**（Table 9）：

  ```
  # thinking
  <|im_start|>user
  {query} /think<|im_end|>
  <|im_start|>assistant
  <think>
  {thinking content}
  </think>
  {response}<|im_end|>

  # non-thinking
  <|im_start|>user
  {query} /no_think<|im_end|>
  <|im_start|>assistant
  <think>

  </think>
  {response}<|im_end|>
  ```

  关键设计：**non-thinking 也保留空 `<think></think>` 块**，让内部格式一致；默认走 thinking、`/no_think` 显式关；多轮里随机插 flag，模型按**最后一个 flag** 走。

- **Thinking Budget 是自然涌现**：模型学会 thinking + non-thinking 后，**未经显式训练**就具备「响应基于截断思考」的能力。budget 阈值到时系统插一句 "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"，模型据此切回最终响应——能力是 fusion 副产物。

### Stage 4：General RL

20+ 任务的奖励体系，三类 reward：

1. **Rule-based**（规则）：reasoning RL 段也用，外加 instruction-following / format adherence 等可精确判定的任务。
2. **Model-based with reference**：用 Qwen2.5-72B-Instruct 对照参考答案打分，处理灵活任务避免 false negative。
3. **Model-based without reference**：基于人偏数据训的 reward model 打 scalar 分，覆盖开放问答与 engagement 类。

覆盖能力：instruction following / format following (`/think` `/no_think` flag 切换) / preference alignment / agent ability（multi-turn real-environment tool calling）/ RAG 等。

### Strong-to-Weak Distillation（小模型路径）

- **Phase 1 Off-policy**：用 teacher（Qwen3-32B 或 Qwen3-235B-A22B）在 `/think` 和 `/no_think` 两种模式下生 response，学生模型直接蒸 response。建立基本推理能力 + mode 切换能力。
- **Phase 2 On-policy**：学生自己 rollout（采 prompt 在某个 mode 下生 response），与 teacher 同 prompt 同 mode 下的 **logits 对齐**（minimize KL），不是只蒸 final response。

### Distillation vs RL 对照（Table 21，Qwen3-8B）

| 方法 | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | GPU·h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Off-policy Distillation（起点） | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | — |
| + Reinforcement Learning | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | **17,920** |
| + On-policy Distillation | **74.4 (93.3)** | **65.5 (86.7)** | **97.0** | **60.3** | **88.3** | **63.3** | **1,800** |

（括号内为 pass@64。）关键观察：on-policy distill 的 **pass@64 也提升**（说明探索空间扩了），RL 的 pass@64 反而**不动**——这是论文给 RL 不如蒸馏的核心证据。

## 评测要点

- **Long-context (RULER, Appendix Table 23)**：旗舰 235B-A22B 平均 95.0；32B / 30B-A3B 平均 ~94。128K 上仍保持 ~90+。
- **Thinking budget scaling**：增加 thinking token 上限对 AIME / GPQA / LiveCodeBench 都持续单调改善（论文摘要原话）。
- 多语言：Belebele 覆盖 139 语言，Table 24–34 给 9 种语言（es/pt/ar/ko/ru/de 等）逐项分。

## 待追问

- **QK-Norm 具体加在哪**（softmax 之前对 Q 和 K 各做 RMSNorm？head-wise 还是整张矩阵？）——报告只引用 Dehghani et al., 2023，未给实现细节。
- **Global-batch load balancing loss 公式**：Qiu et al., 2025 是同期 Qwen 团队论文，本报告未自带推导，若要复用需回那篇核。
- **Thinking Budget 截断指令为何由系统插入而非显式训练**——论文称是 "emerges naturally"，但模型靠什么信号识别 "</think>\n\n" 后切换到 response？是 chat template + non-thinking 训练数据里的空 think 块带的隐式 grammar，还是另有机制？正文未展开。
- **从 Qwen3 到 Qwen3-Next 的架构换代动机**：本报告全 GQA 标准栈，Qwen3-Next 引入 3:1 GDN+gated-attention 是另一篇博客的事。两条路线为何能并存（Qwen3.5 走 hybrid、Qwen3-VL 走标准 GQA Qwen3 backbone），待[Qwen3-VL 来源页](qwen3-vl.md) / Qwen3-Next 博客横向对比。

## 相关页面

- 模型：[Qwen3](../models/qwen3.md)
- 同家族后续：[Qwen3-Next 官方博客](qwen3-next-blog.md)、[Qwen3.5](../models/qwen3.5.md)、[Qwen3-Coder-Next](qwen3-coder-next.md)、[Qwen3.5-Omni](qwen3.5-omni.md)、[Qwen3-VL](qwen3-vl.md)
- 概念：[Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- 外部后训练算法：[Agentic Reinforced Policy Optimization](agentic-reinforced-policy-optimization.md)（用 Qwen3-8B/14B 做 deep search RL backbone，但不是 Qwen3 官方报告的一部分）
