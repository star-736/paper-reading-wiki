---
type: Source
title: "Qwen3-VL 技术报告"
description: "Qwen3-VL 多模态家族报告（arXiv:2511.21631），256K 原生上下文；三块架构升级 = Interleaved MRoPE（t/h/w 频谱均衡）+ DeepStack（ViT 中间 3 层 → LLM 前 3 层 residual add）+ 文本时间戳替换 T-RoPE。LLM backbone 是**标准 GQA 的 Qwen3**，与 Qwen3.5-Omni 的 hybrid 基座是两条路。"
tags: ["source", "qwen3-vl"]
timestamp: 2026-06-22
resource: "../../raw/Bai 等 - 2025 - Qwen3-VL Technical Report.pdf"
---

# Qwen3-VL 技术报告

## 来源

- 文件：`raw/Bai 等 - 2025 - Qwen3-VL Technical Report.pdf`
- 标题：Qwen3-VL Technical Report
- 团队 / 日期：Qwen Team（Alibaba），2025-11-27（arXiv:2511.21631v1，2025-11-26 提交）
- 模型链接：[Qwen3-VL on HuggingFace](https://huggingface.co/Qwen)、[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- 系列定位：**Qwen3-VL = [Qwen3](qwen3.md) backbone + SigLIP-2 视觉编码器 + DeepStack 注入 + Interleaved MRoPE**。注意它走的 LLM 基座是**标准 GQA 的 Qwen3**，**不是** Qwen3-Next/3.5 那条 GDN hybrid 路线——和 [Qwen3.5-Omni](qwen3.5-omni.md) 是两条独立多模态扩展。

## 核心结论

1. **6 个变体覆盖 2B–235B**：4 个 dense（2B/4B/8B/32B）+ 2 个 MoE（30B-A3B / 235B-A22B），统一 **256K context** 原生支持文本 + 图像 + 视频交错输入。
2. **三块架构升级**（相对 Qwen2.5-VL）：① **Interleaved MRoPE**（t/h/w 频谱均衡）；② **DeepStack**（ViT 中间层视觉 token 直接 residual add 进 LLM 前 3 层）；③ **文本时间戳**替换 T-RoPE（视频帧前缀 `<3.0 seconds>` 文本 token）。
3. **多模态没拖后腿语言能力**：旗舰 Qwen3-VL-235B-A22B 在大多语言 benchmark 上 **超过其纯文本对应版** Qwen3-235B-A22B（论文 § 5.11 + Table 5/6 实测）。归功于 square-root reweighting + 训练数据保留大量 text-only 部分。
4. **从 sample-level loss 改为 √-normalized per-token loss**：缓解多模态/纯文本样本因 token 数差异导致的梯度失衡，"提升 multimodal 性能而不牺牲 text 能力"（摘要原话）。

## 架构

> Qwen3-VL 沿用 Qwen2.5-VL 的三模块栈：**vision encoder + MLP merger + LLM**——但三个模块全部升级，且引入跨层注入。

![Qwen3-VL 框架：Vision Encoder（SigLIP-2）以原生分辨率处理任意宽高比的图像/视频输入并输出可变长度的视觉 token；DeepStack 从 ViT 的三个不同层级各取一组视觉特征，经专用 merger 投影后分别 residual-add 进 LLM 的前 3 个 block（图中 LLM Block 1 / 2 / 3）；视频帧前缀以 `<0.0 seconds>` `<4.0 seconds>` 这类文本时间戳 token 标注；位置编码用 Interleaved MRoPE（t/h/w 在频谱上均匀分布）；LLM 解码器是 Qwen3 Dense/MoE。](../assets/qwen3-vl/fig1-framework.png)

> 论文 Figure 1 原文标题："The Qwen3-VL framework integrates a vision encoder and a language model decoder to process multimodal inputs, including text, images, and video. ... we incorporate the pioneering DeepStack mechanism, which injects visual tokens from multiple layers of the vision encoder into corresponding layers of the LLM."（§ 2 Model Architecture 引言图）

### Vision Encoder：SigLIP-2 + 2D-RoPE + CoMP

- 默认 **SigLIP2-SO-400M**；小模型（2B / 4B）用 **SigLIP2-Large 300M**。
- **2D-RoPE** 处理动态分辨率，沿 CoMP（Chen et al., 2025）做绝对 position embedding 插值。
- 在原始 SigLIP-2 预训练 checkpoint 上**继续以动态分辨率训练**。

### MLP Merger：2×2 → 1 token

- 两层 MLP，把视觉特征以 2×2 patch 为单位压成 1 个视觉 token，对齐 LLM 隐藏维度。
- **DeepStack 用的 3 个 merger 是独立专用模块**（不复用主 merger），把不同层级 ViT features 投到 LLM 维度。

### LLM Decoder：Qwen3 backbone

四 dense + 两 MoE 全部用 [Qwen3](qwen3.md) 同名变体作 backbone：

| Qwen3-VL 变体 | LLM backbone |
| --- | --- |
| 2B / 4B / 8B / 32B | Qwen3-1.7B / -4B / -8B / -32B（dense） |
| 30B-A3B | Qwen3-30B-A3B（MoE） |
| 235B-A22B | Qwen3-235B-A22B（MoE） |

> 注意：Qwen3-VL-2B 用的是 **Qwen3-1.7B** 而非同名的 Qwen3-2B（Qwen3 base 报告没有 2B dense；2B 是 VL 算上 vision encoder 之后的整体参数量）。LLM 这一侧是**标准 GQA + RoPE + QK-Norm + 无 shared expert MoE**，详见 Qwen3 base 报告。

### Interleaved MRoPE（§ 2.1）

- Qwen2-VL 原版 MRoPE 把 embedding 维度切成 **t / h / w 三个不交叠子空间**，每子空间有独立 RoPE 频率——结果**每条轴各自频谱不均**，长视频上明显退化。
- 改法（沿 Huang et al., 2025）：**把 t / h / w 在 embedding 维度上交错分布**，每条轴均匀覆盖高低频段，spectral bias 消失。

### DeepStack（§ 2.2）

- 借鉴 DeepStack（Meng et al., 2024）。原版栈不同 scale 的视觉输入；Qwen3-VL **改为栈 ViT 不同层的特征**（low-level / mid-level / high-level）。
- 选 ViT 的 **3 个不同层级**特征 → 经 3 个专用 merger 投影 → **直接 residual add 到 LLM 前 3 层的 hidden state**（注意是只前 3 层，不是每层都注，避免视觉信号在深层被冲淡同时不增加上下文长度）。

### Video Timestamp：T-RoPE → 文本时间戳（§ 2.3）

Qwen2.5-VL 的 T-RoPE（time-synchronized MRoPE）两个问题：

1. **绝对时间 → 巨大稀疏的 position id**——长视频上 position id 飙高且分布稀疏，模型难学。
2. 训练要求**多 fps 均匀采样**——数据构造成本高。

改法（沿 Chen et al., 2024b）：**视频每个时间 patch 前缀一个文本 token**，格式如 `<3.0 seconds>` 或 HMS 格式 `<00:00:03>`，训练时两种格式都喂。代价：序列略长；收益：时序信息直接以文本 token 形式参与 attention，视频 grounding / dense captioning 任务显著改善。

### Square-root Token Reweighting

从 per-sample loss 换成 **per-token loss，按样本 token 数的 √ 归一化**——这样 token 数差异极大的纯文本与多模态混训不会让一边压另一边。摘要明说"boosts multimodal performance without compromising text capabilities"。

## 预训练

Vision encoder 先做 dynamic-resolution continual training。整体训练 4 stage（Table 1）：

| Stage | 目标 | 训练范围 | Token | Seq len |
| --- | --- | --- | --- | --- |
| **S0** | Vision-Language Alignment | 仅 merger | 67B | 8192 |
| **S1** | Multimodal Pre-Training | 全参 | ~1T | 8192 |
| **S2** | Long-Context Pre-Training | 全参 | ~1T | 32,768 |
| **S3** | Ultra-Long-Context Adaptation | 全参 | 100B | **262,144** |

- **S0** 只动 merger 冻结其余——alignment-first，避免 LLM/ViT 被噪声拉偏。
- **S1** VL + text-only 混训，加入 interleaved image-text、grounding、VQA、STEM、少量视频。
- **S2** 序列 4× 到 32K，加入更多视频与 agent 指令跟随。
- **S3** 序列 8× 到 256K，重心放在长视频 + 长文档（包含整本书、数百页技术文档）。

数据扩张要点：

- **Caption 数据**用一个 fine-tuned 的 Qwen2.5-VL-32B 做 **recaption**——基于原图 + 原文 raw text 生成更细的描述（物体属性、空间布局、上下文语义），既扩视觉覆盖也提语言质量。
- **Book-scale interleaved**：用 fine-tuned Qwen2.5-VL-7B 解析整本电子书的图文交错结构，**合并连续页到 256K token 序列**喂 S3。
- **OCR / Grounding / Counting**：OCR 覆盖 **39 种语言**（Figure 2：32 种语言上 > 70% 准确率）；grounding 数据 box-based + point-based 都做；counting 单独建数据集让模型支持数量推理。
- **3D / 空间数据**：构造 RoboSpatialHome、RefSpatialBench 等数据，输出 `[{"point_2d": [x, y], "label": "..."}]` JSON 格式坐标。

## 后训练

3 阶段（§ 4）：① **SFT on long-CoT data**（cold start）→ ② **Knowledge distillation from stronger teachers** → ③ **Reinforcement Learning**。

### Strong-to-Weak Distillation（§ 4.3）

报告原文：「We adopt the Strong-to-Weak Distillation pipeline as described in Qwen3 to further improve the performance of lightweight models」——直接复用 [Qwen3](qwen3.md) 报告里 §4.5 的两阶段范式（off-policy 把多个 teacher 输出合并做 response distill 打底 → on-policy student 生成轨迹、与 teacher logits 对齐做 KL）。**关键设计：用 text-only 数据蒸馏 LLM backbone 部分**（论文 §4.1 强调："we perform this distillation using text-only data to fine-tune the LLM backbone. This method proves highly effective, yielding significant improvements in reasoning abilities across both text-centric and multimodal tasks"）——视觉模块不参与这一步。

> 与 MiMo MOPD / DeepSeek-V4 OPD（多 teacher 融合）的对比详见 [On-Policy Distillation 跨报告对比](../comparisons/on-policy-distillation.md) 轴一第二类（强到弱迁移）。OPD 算法的奠基性介绍见 [Thinking Machines Lab On-Policy Distillation 博客源页](thinking-machines-on-policy-distillation.md)。

### Cold Start SFT（§ 4.2）

- 约 **1.2M 样本**：1/3 text-only + 2/3 多模态（图文 / 视频）。
- 两阶段 sequence length：先 32K 一个 epoch，再 256K 一个 epoch（带 curriculum 把 32K 与 256K 混采）。
- Query / Response 两阶段过滤；模型侧 reward model 基于 Qwen2.5-VL 系列做多维度评分（正确性、完整性、清晰度、helpfulness；视觉任务额外评判视觉信息的准确利用）。
- 数据扩到 8 大领域 30+ 子类，覆盖 embodied spatial reasoning / image-grounded reasoning / video spatio-temporal grounding / 上百页技术文档理解。

### Long-CoT Cold Start（§ 4.2.2）

为 thinking 变体准备的 cold start，要求每条 CoT **包含视觉信息的明确利用**（不是纯文本推理）。

### RL（§ 4.3）

约 **30K RL queries** 覆盖 text + 多模态任务（数学、OCR、grounding、instruction following）。每 query 采 16 个 response。

### Agentic 后训练（§ 4.5）

两阶段（visual agent 训练）：

1. **Stage 1**：合成 10k cold-start agentic 数据（简单 2 轮 VQA），在 **Qwen2.5-VL-32B** 上 SFT 学 "think → act → analyze feedback → answer" 行为，再做 **multi-turn tool-integrated RL**。
2. **Stage 2**：把 stage 1 训出的 visual agent 蒸馏出 **120k 多轮 agentic 交互**数据，用同样的 cold-start SFT + tool-integrated RL pipeline 训 Qwen3-VL 自身。

RL 用 3 个互补 reward：

- **Answer Accuracy Reward**（Qwen3-32B 评判）
- **Multi-Turn Reasoning Reward**（Qwen2.5-VL-72B 评判推理链 / 工具反馈解读）
- **Tool-Calling Reward**：对照 Qwen2.5-VL-72B 离线估计的"该任务应调多少次工具"目标值，防止 reward hacking（模型偷懒只调一次工具就交答案）。

### 双变体输出

每个尺寸都同时释 **Instruct**（non-thinking）+ **Thinking** 两个权重，与 Qwen3 base 的 "单权重双模式" **不同**——VL 这边走的是 **两份独立模型**。Thinking 变体在 MMMU / MathVista / MathVision 等推理类 benchmark 上显著领先 Instruct。

## 评测要点

- **GUI Agent**：Qwen3-VL-235B-A22B 在 ScreenSpot / ScreenSpot Pro / OSWorldG 上 SoTA；OSWorld 在线评测 32B 41 分、AndroidWorld 32B 63.7 分。
- **Video**：8B 变体追平 Qwen2.5-VL 72B；旗舰在 MLVU 长视频上**超过 Gemini 2.5 Pro**（256K context 红利）。评测最多 2,048 帧 / video，总 token < 224K。
- **Text-Centric (§ 5.11)**：Qwen3-VL-235B-A22B-Instruct 在多数语言/推理/code/alignment/agent benchmark 上 **>** Qwen3-235B-A22B-Instruct-2507 的纯文本版（这点是论文卖点之一——"VL 训练没有侵蚀语言能力"）。
- **OCR**：Figure 2 自建测试集 39 语言，32 种 > 70% 准确率。
- **3D / Counting**：RefSpatialBench / RoboSpatialHome / CountBench / ODinW-13 全面参与，输出 JSON point/bbox。

## 待追问

- **DeepStack 选 ViT 哪 3 层？** 论文只说 "three distinct levels"，未指定层号；HF 实现 / config 可能给具体数。
- **Interleaved MRoPE 的具体 dim 分配方案**：论文称沿 Huang et al., 2025，但未给本作具体每多少维度切 t/h/w 周期。
- **Square-root reweighting 公式**：摘要 + 引言点到，正文未给完整 loss 形式。
- **Qwen3-VL backbone 用的是 Qwen3 还是 Qwen3-Next?** Abstract / Introduction 明确写 "Built on the Qwen3 series"，且变体表对应 Qwen3-1.7B/4B/8B/32B/30B-A3B/235B-A22B——这些都是 [Qwen3 base 报告](qwen3.md) 释出的标准 GQA 变体，**不是** Qwen3-Next 的 hybrid 变体。这条已据正文核实。
- **与 Qwen3.5-Omni 的分工**：Omni 走 Qwen3-Next/3.5 hybrid backbone + 文本+图像+视频+音频；VL 走 Qwen3 标准 backbone + 文本+图像+视频。两条线**LLM backbone 不同**——为何不统一？是否暗示 hybrid stack 在视觉任务上有劣势？正文未触及。

## 相关页面

- 模型：[Qwen3-VL](../models/qwen3-vl.md)
- LLM backbone：[Qwen3 技术报告](qwen3.md)、[Qwen3 模型页](../models/qwen3.md)
- 同家族其他多模态：[Qwen3.5-Omni](qwen3.5-omni.md)
- 概念：[Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md)、[多模态 Agentic 训练](../concepts/multimodal-agentic-training.md)
