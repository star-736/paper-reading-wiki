---
type: Source
title: "InternVLA-A1.5 技术报告"
description: "上海 AI Lab 的统一 VLA 模型，Qwen-3.5 2B backbone + 轻量 unified expert + latent foresight，6 项仿真 benchmark 全部最优。"
tags: ["source", "vla", "robotics", "gdn-adoption"]
timestamp: 2026-07-11
resource: "raw/2607.04988v1.pdf"
---

# InternVLA-A1.5 技术报告

## 来源

- **PDF**：`raw/2607.04988v1.pdf`（arXiv:2607.04988v1，2026-07-06）
- **标题**：InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization
- **团队**：Physical Intelligence Team, Shanghai AI Laboratory
- **模型页**：[InternVLA-A1.5](../models/internvla-a1.5.md)

## 核心结论

InternVLA-A1.5 是一个统一 VLA（Vision-Language-Action）模型，把视觉语言理解、视觉前瞻（foresight）和动作生成放在一个框架里。它针对现有统一模型的三个痛点：(1) 加上生成和动作目标后 VLM 的语义能力被侵蚀；(2) 异质目标（视觉 latent 回归、flow-matching 动作预测、语言建模）互相干扰；(3) 视觉预测模块从零学像素级生成，不利用预训练视频生成模型的先验。

三个设计回应：

1. **VLM backbone 持续训练 VQA + subtask 预测**：不因加了动作目标就停掉语义训练，用 chat template 统一所有目标到一个 next-token loss。
2. **Mixture-of-Transformers（MoT）**：VLM 和轻量 unified expert 只通过共享 full attention 层交互，各自维护独立的 GDN 线性注意力层，避免异质目标干扰。
3. **Latent foresight**：用 learnable foresight tokens 从 frozen 预训练视频生成模型（WAN2.2-5B）蒸馏时空先验，推理时丢弃视频模型，零额外延迟。

在 6 项仿真 benchmark（LIBERO、RoboTwin 2.0、EBench、SimplerEnv、LIBERO-Plus、DOMINO）上全部最优或高度竞争；真实世界 4 项任务（3 项指令跟随 + 1 项长周期化学实验 MOF）上组合泛化和长周期执行领先 π0.5 和 Motus。

## 架构与训练

### Mixture-of-Transformers 架构

![InternVLA-A1.5 Figure 2：框架架构。左侧为预训练 VLM（Qwen-3.5 2B），采用 3 GDN + 1 Full Attention 的混合注意力，6 层堆叠；右侧为 Unified Expert（460M），与 VLM 共享 Full Attention 层但维护独立的 GDN 层。Foresight Tokens 经 Unified Expert 处理后输出到冻结的视频生成模型 WAN2.2 5B（仅训练时使用）。VLM 接收 Images / Instruction / State / Mode 输入，输出 QA/SubTask 和 Action Token。](../assets/internvla-a1.5/fig2-framework.png)

> Figure 2（原文截图，§ 2 Model Design）："The architecture adopts a Mixture-of-Transformers design comprising a pretrained VLM for multimodal perception and a lightweight unified expert that shares full attention layers with the VLM while maintaining separate linear attention layers."

两个核心组件：

- **预训练 VLM backbone**（Qwen-3.5 2B）：采用 Qwen3.5 的 hybrid attention（3 层 Gated DeltaNet + 1 层 full attention 交替），初始化自预训练权重。每个 timestep 接收多视角图像 $o_t$、语言指令 $l$ 和机器人本体感知状态 $q_t$，编码为视觉和文本 token。VLM 可输出 VQA 答案、subtask 描述、以及 FAST tokenizer 编码的离散动作 token。
- **Unified Expert**（460M）：与 VLM backbone 相同的 hybrid 架构但 hidden dimension 缩小，仅通过共享 full attention 层与 VLM 交互，维护独立的 GDN 层做模态特定处理。输入为 learnable foresight tokens 和 action query tokens。

**关键设计**：VLM 和 unified expert 只在 full attention 层共享信息，GDN 线性注意力层各自独立。这使得语义处理（VLM）和动作/前瞻处理（expert）在大部分层里互不干扰，只在全局 full attention 层汇总。

### Latent foresight 机制

![InternVLA-A1.5 Figure 4：foresight reasoning 机制。可学习的 foresight query tokens $Q^f$ 经 Unified Expert 处理后产生条件嵌入 $C_t^f$，通过 cross-attention 注入冻结的 WAN2.2 视频生成模型（N× Video DiT Blocks + 3D-VAE Encoder/Decoder）。红色虚线标示梯度只回传到 foresight tokens 和 Unified Expert，视频模型冻结。训练时用 flow-matching 目标对未来帧做视频预测监督。](../assets/internvla-a1.5/fig4-foresight-reasoning.png)

> Figure 4（原文截图，§ 3.2）："Learnable foresight query tokens attend to the current visual-language context through the unified expert and produce conditioning embeddings for the frozen video generation model. The video supervision loss encourages these tokens to encode future-relevant information."

机制要点（§ 3.2，原文确证）：

- 在 unified expert 序列中、flow-matching 动作预测之前插入 $M$ 个 learnable foresight tokens $Q_f \in \mathbb{R}^{M \times d}$。
- 这些 tokens 通过 unified expert transformer 注意当前视觉-语言上下文 $H_t$，输出 contextualized foresight embeddings $Z_t^f$（公式 3）。
- $Z_t^f$ 投影为视频生成模型的条件 $C_t^f = P_{\text{WAN}}(Z_t^f)$，替换 WAN2.2 原本的 T5 text encoder 条件，通过 WAN 原生 cross-attention 层注入。
- 每个动作 chunk 对应均匀采样 $N=4$ 帧未来画面作为预测目标。
- 视频监督 loss 为 flow-matching 目标（公式 4）：$\mathcal{L}_{\text{video}} = \mathbb{E}\|u(x_s, C_t^f, s) - v_s\|^2$，其中 $u$ 是冻结的 WAN denoising transformer。
- WAN 参数冻结，梯度只回传到 foresight tokens 和上游 unified expert 层。
- **推理时视频模型完全丢弃**，不引入额外延迟。

### 动作预测

Stage 2 用 flow-matching 做连续动作预测（替代 Stage 1 的离散 FAST token）。对 ground-truth 连续动作 chunk $a_{t:t+H}$，采样高斯噪声 $\epsilon$ 和 interpolation timestep $\tau \sim \text{Beta}(1.5, 1.0)$，构造插值 $a^\tau = (1-\tau)\epsilon + \tau a$，目标速度为 $a - \epsilon$（公式 5-6）。推理时从高斯噪声出发用 Euler 积分求解（公式 7）。

### 训练配方

三阶段训练（Table 1，原文确证）：

| 配置 | Stage 1 Pretrain | Stage 2 Pretrain | Posttrain |
| --- | --- | --- | --- |
| 优化器 | AdamW | AdamW | AdamW |
| Batch size | 1024 | 1024 | 128 |
| 学习率 | 5×10⁻⁵ | 5×10⁻⁵ | 5×10⁻⁵ → 5×10⁻⁶ |
| 训练步数 | 300,000 | 600,000 | 60,000 |
| Foresight tokens | – | 50 | 50 |
| Action chunk | 50 | 50 | 50 |
| 精度 | bfloat16 | bfloat16 | bfloat16 |

- **Stage 1（VLM Transferring）**：VLM backbone 在 VQA + 机器人数据上联合训练，统一 tokenized 格式，单一 next-token cross-entropy loss 监督 subtask 描述 + FAST 动作 token。动作 token 加到 VLM 词表、共享 embedding 表和输出投影。分解 $\pi_\theta(a_{t:t+H}, \hat l | o_t, l) = \pi_\theta(a_{t:t+H}|o_t, \hat l)\pi_\theta(\hat l|o_t, l)$（公式 1-2）。
- **Stage 2（Foresight and Action Generation）**：引入 unified expert + foresight reasoning。总 loss $= \mathcal{L}_{\text{stage1}} + \alpha\mathcal{L}_{\text{video}} + \beta\mathcal{L}_{\text{action}}$，$\alpha=1, \beta=10$（公式 8）。
- **Post-training**：同 Stage 2 配方，视频生成分支可选保留以在下游演示上微调 foresight tokens。

### Attention masking

VLM token（image / instruction / state / subtask）走标准 Qwen3.5 causal attention。Unified expert 的 foresight tokens 和 noisy action embeddings 作为独立 token group，跨 group causal（foresight 注意 VLM 上下文，action 注意 VLM + foresight），group 内 bidirectional（支持 flow-matching 非自回归并行去噪）。训练时 unified expert 被 mask 掉对 FAST token 的注意力，防信息泄漏和梯度干扰。

### 数据配方

- **机器人操作数据**：6 个来源（1 合成 + 5 真实），共 1.2M episodes / 861M frames。InternData-A1（合成，587K ep / 396M frames）、AgiBotWorld、UMI、DROID、Galaxea、RoboMind 1.0。统一到 InternVLA-A1 的 action space。
- **多模态联合训练数据**：InternVLA-M1 语料，约 3M 样本，4 类 QA（General QA 637K + Box QA 879K + Point QA 832K + Trajectory QA 684K），保持 VLM 的指令跟随和空间 grounding 能力。

## 评测要点

### 真实世界

4 项任务（Figure 7-9）：

| 任务 | π0.5 | Motus | InternVLA-A1.5 |
| --- | --- | --- | --- |
| Sort Tubes | 77.8 | 64.8 | 75.9 |
| Insert Tubes | 51.7 | 44.2 | 72.5 |
| Move Tubes | 72.7 | 56.2 | 80.5 |
| MOF（长周期化学） | 29.3 | 0.0 | 76.4 |

3 项指令跟随任务采用 held-out 设计：训练时只见部分 (tube, target) binding，测试时评未见组合。InternVLA-A1.5 在所有 3 项的 held-out binding 上成功率最高，证明优势不靠记忆 demonstrated binding。MOF 长周期任务（13 步化学操作）领先 π0.5 47 个百分点，归因于显式 subtask 预测 + 学到的 dynamics prior。

### 仿真

6 项 benchmark 全部最优或高度竞争：

| Benchmark | InternVLA-A1.5 | 次优 |
| --- | --- | --- |
| LIBERO | 98.9 | 98.7 (Xiaomi-Robotics-0) |
| LIBERO-Plus（zero-shot） | 84.8 | 84.4 (π0.5) |
| RoboTwin 2.0 | 93.2 | 92.2 (LingBot-VA) |
| SimplerEnv | 80.8 | 79.2 (Xiaomi-Robotics-0) |
| EBench (Test SR) | 35.2 | 30.9 (LingBot-VA) |
| DOMINO（zero-shot SR） | 27.7 | 26.6 (Qwen-VLA-Instruct) |

### 消融

Table 8：去掉 video supervision loss 在所有 4 项 benchmark 上均降，zero-shot 场景降幅更大（LIBERO-Plus 84.8→78.0，DOMINO 27.7→25.3）；去掉 foresight tokens 降幅更大（LIBERO-Plus 84.8→77.9，DOMINO 27.7→23.8），证明 foresight tokens 是从视频模型蒸馏 dynamics 知识的关键接口。

Figure 10 训练效率：InternVLA-A1.5 的 SFT loss 收敛快于 π0.5 和 InternVLA-A1，预训练表示提供了更有利的优化地形。

Figure 11：frozen WAN 模型在 foresight embedding 条件下生成的未来帧预测，在标准和 zero-shot 场景下都能准确跟踪机器人运动和物理一致的场景演化（如液位变化）。

## 待追问

- Foresight tokens 数量 $M=50$ 的选择依据未给消融；是否存在质量/成本的 sweet spot。
- WAN2.2-5B 的预训练覆盖 embodied 场景的程度如何（论文 limitation 承认 priors 受限于视频模型预训练覆盖面）。
- Unified Expert 460M 的 hidden dimension 具体值、与 VLM 2B 的 dimension 比例。
- Qwen-3.5 2B 作为 backbone 的具体配置（层数、head 数等）——论文只说"Qwen-3.5 2B"，未列配置表；需查 HF config 核实。
- Stage 1 到 Stage 2 过渡时 VLM 是否继续训练（论文说 Stage 2 保留 $\mathcal{L}_{\text{stage1}}$，但没明确 VLM 权重是否更新）。

## 相关页面

- 模型：[InternVLA-A1.5](../models/internvla-a1.5.md)
- 架构基座：[Qwen3.5](../models/qwen3.5.md)（VLM backbone = Qwen-3.5 2B，3:1 GDN:full attention 混合）
- [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)（GDN 在 VLA 领域的采用证据）
- [注意力门控](../concepts/attention-gating.md)（Qwen3.5 hybrid 架构的 gated attention 在 VLA 中被继承）
