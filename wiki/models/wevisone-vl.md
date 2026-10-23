---
type: Model
title: "WeVisOne-VL"
description: "微信视觉的 2B/4B 统一感知 VLM，基于 Qwen3-VL，用坐标 token 和约 9.6M 像素解码器同时出框与 mask/alpha。"
tags: ["model", "wevisone-vl", "qwen3-vl", "visual-grounding"]
timestamp: 2026-10-24
---

# WeVisOne-VL

## 身份

WeVisOne-VL（WeChat Vision All-in-One）是腾讯微信视觉的感知模型，把开放词汇检测、指代定位、计数、指点、OCR、指代分割和可提示抠图收进一次结构化生成。两个规模都建在 [Qwen3-VL](qwen3-vl.md) 上。离散坐标用 `<0>` 到 `<1000>`；每个需要像素输出的框后面生成 `<OBJ>`，约 9.6M 的解码器读它的隐状态和预测框，在共享 ViT 特征上出全图 mask 或 alpha matte。

它和同实验室的 [WeMM-Embedding](wemm-embedding.md) 不是一条产品线：那边把 Qwen3.5 收成检索向量，这边把 Qwen3-VL 收成可定位、可拒识的感知接口。

## 关键事实

| 字段 | 值 |
| --- | --- |
| **机构** | WeChat Vision, Tencent Inc. |
| **变体** | 论文称 2B / 4B。Hugging Face 把 `WeVisOne-VL-4B` 标成 5B（外部佐证，2026-10-24）；原文没有给加上解码器后的总参数 |
| **模态** | **多模态**（单张图像 + 文本指令输入；结构化文本输出，像素任务另出二值 mask 或连续 alpha）。评测与推理描述都是单图 |
| **基座** | [Qwen3-VL](qwen3-vl.md) 2B 与 4B。论文没有复述 Qwen3-VL 报告里的 LLM 宽度和 SigLIP-2 规格 |
| **坐标** | 专用 token `<0>`–`<1000>`；缺失目标填 `None` |
| **像素接口** | 框后的 `<OBJ>` 隐状态 + 预测框 + ViT pre-merge 特征。解码器约 9.6M，含抠图细化模块。模型卡把权重文件写成 9,609,764 参数（外部佐证） |
| **训练** | 三阶段课程 SFT，再 178K 样本、组大小 8 的 GRPO。分割热身后丢掉当时更新的 VLM 权重，只留解码器和 `<OBJ>` embedding。抠图阶段冻结 VLM、只训解码器。RL 只训 LLM；分割和抠图的奖励仍是框，不算 mask |
| **推理** | 贪心，temperature 0，每图最多 1,960 视觉 token |
| **主结果** | 4B 检测 mean F1：COCO 57.1、LVIS 55.3、Dense200 71.3、VisDrone 44.7。RefCOCOg-Rej 拒识率 67.4% |
| **开源** | 权重与代码已公开。模型卡写腾讯撰写的代码为 Apache 2.0；论文未写权重许可 |
| **来源** | [技术报告](../sources/wevisone-vl.md)（arXiv:2610.31329v1，2026-10-22） |

模型卡另写 video grid 会被像素解码器拒绝，以及 `<OBJ>` 的 token id 为 152678。PDF 没有这两句。

## 技术身份

WeVisOne-VL 不改 Qwen3-VL 的视觉编码器和语言模型骨架。增量是输出词表、一个实例 token，以及挂在这个 token 上的小解码器。

1. **先把任务说成同一种序列。** 检测、指代、计数、指点和 OCR 都是带坐标的 next-token。计数在框之后还可以写总数。OCR 的类别字符串就是转写。
2. **像素不序列化成文本 mask。** `<OBJ>` 把自回归定位交接到稠密头：六个可变形块用框做 reference，动态核在融合特征上出 logit，再经三阶段上采样得到 sigmoid alpha。分割和抠图共用这套头，抠图多一次冻结 VLM 的专项微调。
3. **数据侧先对齐实例，再分任务。** 同一个 query 下既有框也有 mask，抠图伪标签继续绑在这个实例上。拒识样本的目标集为空。
4. **RL 优化的是定位和格式，不是 alpha。** 不确定性分数 $4\mu(1-\mu)\sigma$ 用来挑有相对优势的题。奖励把 IoU 立方，再按真值数与预测数惩罚漏检和重复。指代分割只检查 `<OBJ>` 是否出现。

2B 与 4B 不是单调缩放。框提示的 P3M、PPM，以及 Ref-ZOM 的 cIoU、COCO 指点，表内是 2B 更高。语言提示的 WeRefMat 和密集检测 mean F1 则是 4B 明显更高。

## 相关页面

- 来源：[WeVisOne-VL](../sources/wevisone-vl.md)
- 架构基座：[Qwen3-VL](qwen3-vl.md)、[Qwen3-VL 技术报告](../sources/qwen3-vl.md)
- 同实验室：[WeMM-Embedding](wemm-embedding.md)
- GRPO 出处：[DeepSeekMath](../sources/deepseekmath.md)
