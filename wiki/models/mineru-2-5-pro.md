---
type: Model
title: "MinerU2.5-Pro"
description: "上海 AI Lab + PKU + SJTU + 商汤的轻量解耦文档解析 VLM，固定 1.2B 架构，纯靠 Data Engine（DDAS + CMCV + Judge-and-Refine）+ 三阶段训练推到 OmniDocBench v1.6 SOTA。"
tags: ["model", "document-parsing", "ocr", "vlm", "data-centric", "lightweight"]
timestamp: 2026-07-12
---

# MinerU2.5-Pro

## 身份

MinerU2.5-Pro 是上海人工智能实验室 + 北京大学 + 上海交通大学 + 商汤科技发布的轻量文档解析 VLM，定位是「数据中心」论点的实证：**完全保留 MinerU2.5 的 1.2B 解耦 coarse-to-fine 架构不变，把所有优化集中在 Data Engine 和训练策略**，证明架构成熟后系统化数据工程是推动性能的主要杠杆。OmniDocBench v1.6 总分 95.69（baseline 92.98，+2.71），超过所有现有方法含 200× 参数模型。

## 关键事实

| 属性 | 值 |
| --- | --- |
| **机构** | 上海人工智能实验室 + 北京大学 + 上海交通大学 + 商汤科技 |
| **参数量** | 1.2B（NaViT-675M 视觉编码器 + Qwen2-0.5B 语言模型） |
| **模态** | 多模态（文本 + 图像输入；结构化文本 Markdown/LaTeX/HTML 输出） |
| **架构** | 解耦 coarse-to-fine（布局分析与内容识别分离），继承自 MinerU2.5，**无结构性修改** |
| **视觉编码器** | NaViT-675M，原生分辨率，Max Resolution 2048×28×28 |
| **LLM** | Qwen2-0.5B |
| **Data Engine** | DDAS + CMCV + Judge-and-Refine + Targeted Expert Annotation |
| **训练阶段** | Stage 1 大规模预训练（65.5M）+ Stage 2 Hard-SFT（192K 专家标注）+ Stage 3 GRPO 对齐 |
| **训练数据** | <10M → 65.5M（CMCV 自动标注）+ 192K（专家标注 Hard） |
| **开源** | 模型权重 + 代码（HuggingFace + GitHub） |
| **报告日期** | 2026-04-09（arXiv:2604.04771v2） |

> 模态已据报告原文核实：输入文档图像，输出结构化文本（Markdown / LaTeX / HTML），不涉及音频/视频。

## 技术身份

MinerU2.5-Pro 的核心定位是**数据中心文档解析的方法论示范**，而非新架构。四个技术身份点：

1. **「架构不变」作为控制变量**：与同代报告（[GLM-OCR](../sources/mineru-2-5-pro.md)、PaddleOCR-VL-1.5 等在架构/分辨率/解码上各自演进）不同，MinerU2.5-Pro 刻意冻结架构，把 +2.71 分全部归因于数据与训练策略。这是 data-centric AI 范式在文档解析领域最干净的实证。
2. **CMCV 把难度评估从单模型内省扩展到多模型交叉验证**：区别于 MinerU2.5 IMIC、PaddleOCR-VL UACS 用单模型推理一致性做难度代理（只能捕获单模型认知不确定性，无法区分模型特定盲点 vs 普遍难题），CMCV 用三个异构模型的输出共识判定难度，并锚定待改进模型相对外部的表现分 Easy/Medium/Hard。Medium（外部一致、待改进模型不同）训练价值最高，因其精确定位能力缺口且外部成功证明可学。
3. **render-then-verify 打破 self-reflection 自肯定偏差**：Hard 样本标注用 Judge-and-Refine 迭代纠正，但朴素 self-reflection 倾向肯定自身输出。根因是跨模态映射不对称（图像→结构强、结构→视觉弱）。把 LaTeX/HTML 渲染成图与原图配对输入，闭合缺失映射 + 渲染误差放大把文本域细微结构缺陷放大为视觉异常。
4. **三阶段渐进训练对应数据质量层级**：大规模 SFT（覆盖）→ Hard-SFT（精修）→ GRPO（指标对齐），各阶段单阶段增益 +1.31 / +0.96 / +0.45 递减但分别主驱动覆盖、表格、公式。Stage 3 GRPO 直接用评测指标（edit distance / CDM / TEDS / IoU）作 reward，沿用 [DAPO](../sources/dapo.md) 的 clip-higher + dynamic sampling。

与 [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) 的对比：两者都是 1B 级轻量文档解析 VLM，但路线正交。HunyuanOCR-1.5 走「更快」（DFlash 推测解码）+「更好」（Agentic Data Flow agent 驱动数据构造补长尾）；MinerU2.5-Pro 走「数据中心方法论示范」（固定架构，Data Engine 三组件协同优化覆盖/信息量/准确性）+「评测协议修正」（MGAM + Hard 子集）。两者都改 RL 阶段（HunyuanOCR-1.5 用 IcePop + 三组件 reward，MinerU2.5-Pro 用 GRPO + 任务指标 reward）。OmniDocBench v1.6 上 HunyuanOCR-1.5 自报 94.74，MinerU2.5-Pro 统一重测 95.69，但评测条件不完全可比（见 [来源页跨源评测分歧](../sources/mineru-2-5-pro.md#跨源评测分歧)）。

## 相关页面

- [MinerU2.5-Pro 技术报告](../sources/mineru-2-5-pro.md)
- [HunyuanOCR-1.5](hunyuan-ocr-1.5.md) - 同属轻量文档解析 VLM，路线正交（推测解码 + agentic 数据 vs 数据中心方法论 + 评测修正）
- [LLM RL policy optimization 对比](../comparisons/llm-rl-policy-optimization.md) - Stage 3 GRPO + DAPO recipe 的非 agentic 应用
- [数据混合优化](../concepts/data-mixture-optimization.md) - Data Engine 作为 data-centric AI 的难度感知采样 + 标注精修分支
