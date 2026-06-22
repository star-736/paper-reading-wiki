# 多模态 Agentic 训练

## 定义

多模态 agentic training 不只是把图像、视频接到语言模型上，而是让模型在视觉、文本、代码、GUI、搜索和工具环境中统一行动。[Kimi K2.5](../models/kimi-k2.5.md) 是当前知识库中最典型的例子。

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

## 相关页面

- [Kimi K2.5](../models/kimi-k2.5.md)
- [Agent Swarm](agent-swarm.md)
- [Agentic 模型的后训练](post-training-for-agentic-models.md)
