---
type: Model
title: "GLM-OCR"
description: "智谱 AI + 清华的 0.9B 轻量 OCR VLM，CogViT + GLM 解码器 + MTP（共享参数多头）加速，两阶段 pipeline + 文档解析/KIE 双任务统一。"
tags: ["model", "ocr", "vlm", "mtp", "lightweight"]
timestamp: 2026-07-12
---

# GLM-OCR

## 身份

GLM-OCR 是智谱 AI（zai-org）+ 清华大学发布的 0.9B 轻量多模态 OCR 模型，定位是真实生产系统下的文档理解——不靠大模型 scaling，而是靠架构-解码-任务结构对齐拿效率增益。核心是 Multi-Token Prediction（MTP）机制（训练+推理共用的共享参数多头）+ 两阶段解耦 pipeline（PP-DocLayout-V3 + 并行区域识别）+ 文档解析/KIE 双任务统一。OmniDocBench v1.5 Overall 94.62 居首（0.9B 超 235B 通用 VLM），PDF 吞吐 1.86 pages/s 约为 MinerU2.5 的 3.9×。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **机构** | 智谱 AI（Zhipu AI / zai-org）+ 清华大学 |
| **参数量** | 0.9B（CogViT 视觉编码器 ~400M + GLM 解码器 ~500M） |
| **模态** | 多模态（文本 + 图像输入；结构化文本 Markdown/JSON 输出） |
| **视觉编码器** | CogViT ~400M（与 [GLM-5V-Turbo](../sources/glm-5v-turbo.md) 同族视觉前端），MIM + CLIP 双目标 + 大 ViT 蒸馏 |
| **LLM** | GLM ~500M |
| **加速机制** | MTP：k 个共享参数辅助头预测未来 k token，训练 10 tokens/step，推理平均 5.2 tokens/step，~50% 吞吐提升 |
| **pipeline** | 两阶段解耦：PP-DocLayout-V3 布局分析 → 并行区域级识别 |
| **任务** | Task 1 文档解析（layout + region recognition → Markdown/JSON）+ Task 2 KIE（全图 + prompt → JSON），统一为条件结构化生成 |
| **训练阶段** | Stage 1 视觉编码器 → Stage 2.1 VL 预训练 → Stage 2.2 MTP 预训练 → Stage 3 SFT with MTP → Stage 4 RL（GRPO + task-aware reward） |
| **部署** | vLLM / SGLang / Ollama + MaaS API（0.2 RMB/百万 token）+ LLaMA-Factory 微调 |
| **开源** | 模型权重 + 代码 + SDK（GitHub + HuggingFace） |
| **报告日期** | 2026-03-16（arXiv:2603.10910v2） |

> 模态已据报告原文核实：输入文档图像，输出结构化 Markdown/JSON，不涉及音频/视频。

## 技术身份

GLM-OCR 的核心定位是**生产系统导向的轻量 OCR VLM**，四个技术身份点：

1. **MTP 作为训练+推理共用的加速机制**：与 [DSpark](../sources/dspark.md)（推理时独立 drafter）、[HunyuanOCR-1.5](../sources/hunyuan-ocr-1.5.md) DFlash（block-diffusion draft）等同属 MTP 加速家族，但 GLM-OCR 的 MTP 同时是训练目标（不只推理加速），且用**共享参数**而非独立 draft model 控内存。MTP 还带来结构化输出质量收益——鼓励模型向前规划，产出更少「破损」标签。详见 [多 Token 预测](../concepts/multi-token-prediction.md)。
2. **解耦两阶段降幻觉**：小模型处理复杂布局易幻觉和重复生成，显式布局分析把复杂布局拆成简单子问题 + 支持并行识别。与 [MinerU2.5-Pro](../sources/mineru-2-5-pro.md)（解耦 coarse-to-fine）、Dolphin/MonkeyOCR 同族，区别于 dots.ocr / Nougat 的 end-to-end 路线。
3. **文档解析与 KIE 统一**：两者都建模为「视觉条件下的结构化生成」，共享视觉-文本对齐 / 布局推理 / 结构化输出能力，用 prompt 控制输出格式而非独立 pipeline，提升参数效率与跨任务迁移。
4. **GRPO + task-aware reward 的结构化输出 RL**：Stage 4 每任务独立 reward（edit distance / CDM / TEDS / field-F1）+ 结构验证（标签闭合 / JSON 解析 / 缺失字段惩罚）+ 全局重复/畸形惩罚。与 MinerU2.5-Pro Stage 3（GRPO + 评测指标作 reward）、HunyuanOCR-1.5（IcePop + 三组件 reward）同族，区别在 task-aware 粒度。

与 [MinerU2.5-Pro](../sources/mineru-2-5-pro.md) 的对比：两者都是 0.9–1.2B 级轻量文档解析 VLM + 解耦两阶段 + GRPO RL，但路线正交。MinerU2.5-Pro 走「数据中心方法论」（固定架构，Data Engine DDAS+CMCV+Judge-and-Refine）+ 评测协议修正（MGAM + Hard 子集）；GLM-OCR 走「解码效率 + 任务统一」（MTP 加速 + 双任务共享框架）。OmniDocBench 上：GLM-OCR 自报 v1.5 = 94.62（#1），MinerU2.5-Pro 统一重测 v1.6 上 GLM-OCR = 95.15 < MinerU2.5-Pro 95.69；但 GLM-OCR 吞吐 1.86 pages/s 约为 MinerU2.5（0.48）的 3.9×。两者互补：MinerU2.5-Pro 精度略胜，GLM-OCR 效率显著胜。

与 [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) 的对比：两者都是 1B 级轻量 OCR VLM + GRPO RL，加速路线不同——GLM-OCR 用 MTP 共享参数多头（训练+推理共用），HunyuanOCR-1.5 用 DFlash block-diffusion（推理时 drafter）。GLM-OCR 的 MTP 是「训练时就嵌入」，HunyuanOCR-1.5 的 DFlash 是「训练独立 draft model + 推理验证」。

## 相关页面

- [GLM-OCR 技术报告](../sources/glm-ocr.md)
- [MinerU2.5-Pro](../sources/mineru-2-5-pro.md) - 头号竞争者，精度 vs 效率路线对照
- [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) - 同属轻量 OCR VLM，DFlash vs MTP 加速路线对照
- [多 Token 预测](../concepts/multi-token-prediction.md) - MTP 在 OCR 域的应用
- [GLM-5V-Turbo](../sources/glm-5v-turbo.md) - 同用 CogViT 视觉编码器
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md) - Stage 4 GRPO + task-aware reward
