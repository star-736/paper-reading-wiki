---
type: Concept
title: "多模态 Agentic 训练"
description: "Kimi K2.5 的 early vision fusion、MoonViT-3D、zero-vision SFT 和 joint multimodal RL。"
tags: ["concept", "multimodal-agentic-training"]
timestamp: 2026-06-06
---

# 多模态 Agentic 训练

## 定义

多模态 agentic training 不只是把图像、视频接到语言模型上，而是让模型在视觉、文本、代码、GUI、搜索和工具环境中统一行动。[Kimi K2.5](../models/kimi-k2.5.md) 是当前知识库中最典型的例子。

## 跨报告信号

- **Kimi K2.5**：早期、低比例、持续混合视觉 token 的 early vision fusion；MoonViT-3D 视觉编码器；zero-vision SFT 激活视觉推理；joint multimodal RL 同时覆盖 text 和 vision，视觉 RL 反过来改善文本任务。
- **GLM-5V-Turbo**：CogViT 视觉编码器（403M，两阶段：distillation MIM + contrastive IT）；多模态 RL 覆盖 30+ 任务类别，观察到 RL 跨域干扰弱于 SFT——多域可同时稳定提升，而 SFT 常见跨域 trade-off；推理行为模式可跨域迁移。三个 design lens 具有跨报告参考价值：感知仍是高层能力天花板（许多看似高层的失败始于模型看不准）；agent 能力适合分层优化（低层任务更易构造和验证，高层任务在低层不牢时直接推会失稳）；端到端任务的关键是清晰规格 + 可靠验证 + 受控评测。详见 [GLM-5V-Turbo 来源页](../sources/glm-5v-turbo.md)。
- GLM-5V-Turbo 的 RL 基础设施设计（统一 VLM RL Gym、全流程解耦异步、多模态细粒度内存管理、拓扑感知分区）与 GLM-5 的 [异步 Agent RL](asynchronous-agent-rl.md) 一脉相承，但专门解决多模态引入的额外瓶颈：标准 recompute 方案针对文本设计，不处理多模态激活内存随图片数线性增长的问题；长视频等变长视觉输入需要 CP/TP 分区前移到 data-loading 阶段。

## Early vision fusion

Kimi K2.5 的关键发现是：在固定总视觉-文本 token 预算下，早期、低比例、持续混合视觉 token，往往优于后期一次性大量注入视觉数据。这说明视觉能力不是最后接一个 adapter 就能稳定获得的，视觉-文本对齐需要在训练分布中长期存在。

这个策略也降低了“多模态损害文本能力”的风险。K2.5 报告进一步声称，joint multimodal RL 之后，视觉 RL 还能改善 MMLU-Pro、GPQA-Diamond 和 LongBench v2 等文本任务，说明视觉信号可能反过来强化空间、细节和证据整合能力。

## MoonViT-3D

MoonViT-3D 是 K2.5 的视觉编码器。它继承 native-resolution 和 NaViT packing 思路，把不同分辨率图像切成 patch 并打包成 1D 序列。视频方面，它把连续 4 帧视为一个时空体，共享同一视觉编码器，并在进入 MLP projector 前做轻量 temporal pooling。

这种设计的重点是参数共享：图像和视频不走两套独立模块，而是在同一 embedding space 里学习。好处是图像预训练得到的能力可以迁移到视频，代价是长视频能力仍受 context 和视觉 token 压缩质量约束。

## Zero-vision SFT

Zero-vision SFT 指后训练 SFT 阶段只用文本数据，却能激活视觉推理和视觉工具使用。Kimi 的解释是：联合预训练已经把视觉和文本空间对齐，文本 SFT 学到的推理格式、工具协议和 agent 行为可以跨模态泛化。

这和常见直觉相反：人为构造视觉轨迹未必更好，因为它可能引入过窄模板，让模型过拟合特定视觉任务格式。zero-vision SFT 的成立前提很可能是基础 joint pretraining 已经足够强。

## Joint multimodal RL

K2.5 的 RL 同时覆盖 text 和 vision。可验证任务使用规则奖励；开放式任务使用 GRM；视觉 grounding、OCR、计数、分割等任务使用更细的 task-specific reward。报告还使用 token efficiency 相关训练技巧，避免模型只靠无限拉长思维链换分数。

![Kimi K2.5 Figure 10：Agentic RL 框架总览。Rollout Manager 管理多个 Single Agent Task（堆叠表示并行实例）。每个 task 内：Pluggable Components（Toolset / Judge / Prompt Enhancement）→ Core Agent Loop（Act→Obs 循环，可递归调用子任务）→ Environment（Black-Box Env 走 LLM Gateway 对接外部 LLM API；White-Box Env 走 Env Pool 管理内置环境）。右侧 Inference Engine Service 提供 Token-in/Token-out 推理服务，Training Engine Service 收集 rollout 数据 + Mismatch Correction 信号更新模型。](../assets/kimi-k2.5/fig10-agentic-rl-framework.png)

> Figure 10（原文截图，§ Agentic RL Framework）：K2.5 的 RL 框架以异步协程运行每个 agent task，严格遵循 Token-in-Token-out 范式，Inference Engine 和 Training Engine 解耦。White-box 环境用 Env Pool 管理、black-box 环境通过 LLM Gateway 对接外部 API。

对知识库的启发是：多模态 agent 的训练目标需要覆盖"看懂输入""遵守工具格式""完成环境任务""控制 token 预算"四个层面，而不是单独优化视觉 benchmark。

## 跨报告信号

### JoyAI-VL-Interaction：从"看懂"到"每秒决定是否行动"

[JoyAI-VL-Interaction](../sources/joyai-vl-interaction.md)（JD.com，8B）代表了多模态训练的一个不同方向：不是训练更强的 turn-based VLM，而是训练一个 **interaction model**——持续观看视频流，每秒内部决定说话、静默还是委托后台。它的训练 recipe 有几个与 K2.5 形成对照的设计：

- **AdaCodec 预测式视频编码**：参考帧用完整 ViT token（256），可预测帧用 P-token（~16），按预测代价自适应切换。这不是"编码器后接 LLM"的传统路线，而是把视频编码本身设计成类似视频编解码器的 I-frame/P-frame 结构。
- **角色加权 SFT**：时间对齐数据中 silence 步远多于 response 步，标准 SFT loss 会被 silence 主导。论文对 control token 按角色加权（$w^\text{repeated}_\text{silence}=0.4$，$w^\text{response}=1.5$），把梯度从"继续沉默"拉回"适时响应"。
- **Answer-centered window sampling**：RL 阶段不 rollout 整条长流（数百个 mostly-silent turn），而是围绕每个 gold response 构建保持流式因果性但只保留 timing 相关 turn 的短轨迹，把 horizon 压缩到个位数。
- **涌现能力**：训练数据不含 app 界面视频，但模型能引导用户跨屏滑动完成购物；timing action 和 live commentary 在训练中从未共现，但模型能组合它们执行"每 4 秒解说一次"。

与 K2.5 的对照：K2.5 训练 agent 在工具环境中行动（看 -> 调工具 -> 观察 -> 再行动），JoyAI-VL-Interaction 训练模型在视频流中行动（看 -> 决定说话/沉默/委托 -> 继续看）。前者是"多模态 agentic"，后者是"多模态交互"。

### MiniCPM-o 4.5：从"每秒决定是否行动"到"同时感知与生成"

[MiniCPM-o 4.5](../models/minicpm-o-4-5.md)（OpenBMB，9B）把交互范式再推一步：JoyAI-VL-Interaction 的 AdaCodec 每秒做一次离散决策（说话/沉默/委托），感知与生成仍是交替的；MiniCPM-o 4.5 的 Omni-Flow 框架通过时分复用式时间窗口（1.0s chunk）让感知与生成在 token-level 持续耦合，实现真正的全双工——模型在说话的同时能接收新输入并调整后续输出。

训练策略上三者形成谱系：K2.5 在训练分布中长期混合低比例视觉 token（early fusion）；JoyAI-VL-Interaction 用角色加权 SFT + answer-centered window 解决沉默主导的优化问题；MiniCPM-o 4.5 采用四阶段渐进流水线（冻结基座先训语音模块 → 全解冻联合 → SFT → RL），先隔离新模态影响面再逐步打开全参数。三者共享的核心洞察是：多模态能力的稳定获取需要训练分布的长期结构设计，而非后期简单接 adapter。

## 相关页面

- [Kimi K2.5](../models/kimi-k2.5.md)
- [JoyAI-VL-Interaction](../models/joyai-vl-interaction.md)
- [MiniCPM-o 4.5](../models/minicpm-o-4-5.md)
- [Agent Swarm](agent-swarm.md)
- [Agentic 模型的后训练](post-training-for-agentic-models.md)
- [Any-to-any 多模态 serving](any-to-any-multimodal-serving.md) - 训练出多模态 agent 只是上半场；vLLM-Omni 这类 serving 系统解决多阶段多模态模型如何在线运行。
