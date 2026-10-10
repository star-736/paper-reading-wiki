---
type: Source
title: "Open X-Embodiment"
description: "21 家机构的开放机器人数据集与 RT-X：约 1M+ 真实轨迹、22 个本体；RT-2-X 是在该混合物上 co-fine-tune 的 55B VLA，不是厨房原版 RT-2。"
tags: ["source", "open-x-embodiment", "rt-2-x", "vla", "robotics"]
timestamp: 2026-10-11
resource: "raw/2310.08864v9.pdf"
---

# Open X-Embodiment

## 来源

- **PDF**：`raw/2310.08864v9.pdf`（arXiv:2310.08864v9，2025-05-14；合作项目页 [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)）
- **标题**：Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **团队**：Open X-Embodiment Collaboration（通讯作者在 PDF 脚注；一作名单按字母序）
- **定位**：数据集 + 跨本体策略论文。RT-1-X / RT-2-X 是在这份混合物上训的策略，不是 [RT-2](rt-2.md) 厨房原版的换名。

**不建模型页。** RT-2-X 沿用 RT-2-PaLI-X 的离散 token 配方，评测对象留在本页和 [OpenVLA](openvla.md)。[RT-2](../models/rt-2.md) 仍只代表厨房 co-fine-tune 的原版。

## 核心结论

把许多实验室的机器人演示收成一个仓库之后，在多种机器人上联合训练可以正迁移：小数据域上 RT-1-X 经常超过只在该域训练的原方法；大数据域上 35M 的 RT-1-X 欠拟合，55B 的 RT-2-X 才能吃下混合物（Table I、§V-A）。

跨机器人带来的新技能和「保住网页先验」不是同一件事。Table II 在 Google Robot 上：

| 行 | 模型 | 历史 | 机器人数据 | 与网页 co-fine-tune | 初始化 | Emergent skills | RT-2 泛化评测 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| (1) | RT-2 55B | 无 | 仅 Google Robot | 是 | Web-pretrained | 27.3% | 62% |
| (2) | RT-2-X 55B | 无 | 机器人混合物 | 是 | Web-pretrained | **75.8%** | 61% |
| (3) | RT-2-X 55B | 无 | 混合物去掉 Bridge | 是 | Web-pretrained | 42.8% | 54% |
| (4) | RT-2-X 5B | 2 | 机器人混合物 | 是 | Web-pretrained | 44.4% | 52% |
| (5) | RT-2-X 5B | 无 | 机器人混合物 | 是 | Web-pretrained | 14.5% | 30% |
| (6) | RT-2-X 5B | 2 | 机器人混合物 | 否 | From scratch | 0% | 1% |
| (7) | RT-2-X 5B | 2 | 机器人混合物 | 否 | Web-pretrained | 48.7% | 47% |

原文确证（Table II、§V-B）。Emergent skills 是 Google Robot 上去做 **RT-2 厨房数据里没有、但 Bridge（另一台 WidowX）里有** 的技能。(2) 相对 (1) 约 3×，是跨机器人数据，不是网页 co-fine-tune 单独造成的。(4) 对 (5)：同样 co-fine-tune，加上长度 2 的历史，emergent / 泛化从 14.5% / 30% 升到 44.4% / 52%。(4) 对 (6)：从零训几乎为零，网页预训练初始化是关键。(4) 对 (7) 历史长度同为 2：5B 上「继续和网页 co-fine-tune」与「网页初始化之后只在机器人数据上 fine-tune」接近；作者写这和 RT-2 原文「必须 co-fine-tune」的结论不同，并归因于 RT-2-X 的机器人数据更多样（§V-B）。**55B 行全部是 co-fine-tune，没有 55B 的「只 fine-tune」对照。**

## 架构与训练

数据集（§III）：**1M+** 条真实轨迹，**60** 个数据集，**22** 个本体（单臂到双臂和四足）。实验用的机器人混合物是当时已入库的 **9** 个本体，少于后来公开的 22（§IV-C）。不要把 1M+ 和 [OpenVLA](openvla.md) 写的「RT-2-X 350k」当成同一个数：350k 是 OpenVLA §5.1 对 RT-2-X 训练集的转述，本 PDF 没有印出 350k。

动作格式（§IV-B）：图像历史 + 语言 → **256-bin** 的 8 维离散动作（7 维末端 + terminate）。各数据集先按自己的尺度归一化再离散化；坐标系和绝对/相对控制**没有**对齐。RT-1-X 是 35M 的 EfficientNet + FiLM + Transformer，**只**在机器人混合物上训练。RT-2-X 用 RT-2-PaLI-X，在原版 VLM 数据与机器人混合物上 **大约 1:1 co-fine-tune**（§IV-C）。推理按机器人要求 3–10 Hz；RT-2 系列走云端。

## 后训练

无单独后训练阶段。RT-2-X 的网页数据是 co-fine-tune 混合物的一半，不是另一段 RL。

## 评测要点

分布内（Table I）：Bridge 上 LCBC 13%、RT-1 40%、RT-1-X 27%、RT-2-X 55B **50%**；Google Robot 上 RT-1 92%、RT-1-X 73%、RT-2-X 55B 91%。小数据五域的 RT-1-X 对原方法 4/5 领先，数字在 Figure 4，本页不从柱图目测。

分布外见上表。去掉 Bridge 后，emergent skills 从 75.8% 掉到 42.8%（行 3），说明那一截增益依赖另一台机器人的数据，不是模型变大本身。

## 证据边界与阅读提示

- 本页 RT-2-X **不是** [RT-2](rt-2.md) Table 4 的厨房 55B。OpenVLA 赢的是本页这条 RT-2-X。
- 「必须持续和网页 co-fine-tune 才能保住语义」不能从 55B 行推出：那些行都开着 co-fine-tune。5B 的 (4) vs (7) 说明网页**初始化**关键，持续混网页不是同量级的杠杆。这和 RT-2 原文 Table 6（同 backbone 上 co-fine-tune > 只 fine-tune）协议不同，不要互填。
- 评测共 3600 次、6 种机器人（§V）。单域试次本页未逐格重列。

## 相关页面

- 厨房原版，不是本页 RT-2-X：[RT-2](rt-2.md) · [模型](../models/rt-2.md)
- 把本页 55B 当对照的开源 7B：[OpenVLA](openvla.md) · [模型](../models/openvla.md)
- 概念：[Vision-Language-Action](../concepts/vision-language-action.md)
- 分层 planner，不是本页动作 token：[SayCan](saycan.md)
