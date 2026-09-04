---
type: Concept
title: "Vision-Language-Action"
description: "把预训练视觉语言模型接到机器人控制：看图、读指令、输出动作。OpenVLA 是离散 256-bin token；π0 是连续 flow + action expert；π0.5 是开世界 co-training。"
tags: ["concept", "vla", "robotics"]
timestamp: 2026-09-05
---

# Vision-Language-Action

## 定义

Vision-Language-Action（VLA）是一类机器人策略：把已经在大规模图文上预训练好的视觉语言模型（VLM）接到控制回路里。模型看见相机画面、读懂自然语言指令，直接给出机器人下一步（或接下来一段）要执行的动作，而不是先输出一段文字计划再交给另一个控制器。

可以把它想成「会看、会读的聊天模型，输出端接上机器人」。训练时监督来自真实轨迹里的控制量；推理时把网络输出还原成电机/末端可以执行的数。这一想法在封闭模型 RT-2 里先被做成产品级配方，由 [OpenVLA](../sources/openvla.md) 写成可复现的开源 7B 模型（论文 §2 把这类系统称为 vision-language-action models）。

标准骨架是 **预训练 VLM + 动作头**。读 AtomicVLA / EmbodiedSkills 之前，动作头只需记住三句话：

1. **[OpenVLA](../models/openvla.md) = 离散动作 token。** 7 维控制、每维 256-bin，覆盖进 Llama 词表，自回归 next-token，再 detokenize。早期开源 generalist 的默认写法。
2. **[π0](../models/pi0.md) = 连续 flow + action expert。** PaliGemma 上看图读指令，300M 专家用 flow matching 一次吐出连续动作块（\(H=50\)，最高 50 Hz）。2026 年论文里的默认低层执行器走这条，而不是 OpenVLA 的 bin。
3. **[π0.5](../models/pi0.5.md) = 开世界 co-training。** 架构仍是 π0 的 flow expert，但同一模型先预测 semantic subtask，再出底层动作；数据混多机器人、web 语义和口头逐步指令。EmbodiedSkills 用的是这一代，不是 256-bin。

[InternVLA-A1.5](../models/internvla-a1.5.md) 是再往后的 MoT + latent foresight 实例：Stage 1 仍用 FAST 离散 token，Stage 2 起是 flow-matching 连续 chunk，并继续做 VQA / subtask；真机表拿 π0.5 当对照。

VLA 不是「多模态聊天 + 另外写个脚本控机器人」：动作在同一个网络里、用同一套（或显式共享的）视觉语言表示预测。它也不是本库里的 GUI agent（点屏幕）或全模态语音模型（听/说）；默认输入是机器人相机与语言指令。不要发明音频。

## 跨报告信号

- **[OpenVLA](../sources/openvla.md)（2024，开源离散 token 基线）**：Prismatic-7B = Llama 2 7B + DINOv2/SigLIP 融合视觉；Open X-Embodiment 约 970k 真实轨迹上全量 fine-tune；只在动作 token 上算交叉熵。跨 WidowX 与 Google robot 共 29 任务，以 7B 超过 55B 的封闭 RT-2-X 约 16.5 个百分点（摘要；Table 4+6）。局限写在原文 §6：单图、无本体感觉、推理频率不够双臂高频、成功率通常 <90%。
- **[π0](../sources/pi0.md)（RSS 2025，连续 flow + action expert）**：PaliGemma 3B + 300M flow matching 专家；观察是多路 RGB + 语言 + 本体感觉，输出 \(H=50\) 连续动作块，最高 50 Hz。预训练约 10,000 小时、7 种配置 × 68 任务，外加 OXE。原文把 OpenVLA 重训到同一混合物：自回归离散化撑不住 chunk 与高频，out-of-box 全面落后（§VI-A、Figure 7）。这是 **原文对照**，不是后文推断。新技能仍要 fine-tune；高层语言是外挂 VLM，不是自己的 subtask 头。
- **[π0.5](../sources/pi0.5.md)（2025，开世界 co-training）**：骨干仍是 π0 的 PaliGemma + flow expert。改动是 FAST 离散预训练再长 expert、**同一模型**先预测 semantic subtask，以及异构数据（MM 约 400 小时 / ~100 家 + ME/CE/HL/WD/VI）。评测在训练未见的厨房/卧室；97.6% 预训练样本不是目标平台家务（§I）。InternVLA-A1.5 真机表与 LIBERO-Plus 拿它当对照。
- **[InternVLA-A1.5](../sources/internvla-a1.5.md)（2026，MoT + latent foresight）**：不是把 OpenVLA 换个 backbone，也不是 π0.5 换 Qwen。VLM 是 Qwen-3.5 2B（3:1 GDN:full attention），旁边 460M unified expert 只在 full attention 层交互。Stage 2 起 flow-matching chunk（size 50）+ 从冻结 WAN2.2 蒸馏的 foresight token。语义侧继续 VQA / subtask。评测协议、数据（1.2M episodes + 3M 多模态）不可与 OpenVLA 附录 E 或 π0.5 家庭进度条直接横比。
- **[ASPIRE](../sources/aspire.md) 已 ingest，但不是第四种动作头**：它写/改 code-as-policy 程序，把验证过的修复写入 skill library；仿真 coder 是冻结的 Claude Opus 4.6，不更新 VLA 权重。LIBERO-Pro 表里 OpenVLA / π0 / π0.5 是对照，不是前作。入口见 [具身 skill 自进化](embodied-skill-self-evolution.md)。
- **未 ingest、只作外部线索**：RT-2 / RT-2-X 是 OpenVLA 原文里的封闭前作。AtomicVLA、EmbodiedSkills、EmbodiSkill、π0.7 尚未 ingest。EmbodiedSkills 若写低层执行器，默认是 π0.5 这一代。不要把 ASPIRE（程序库）、EmbodiSkill（training-free reflection）和 EmbodiedSkills（VLA 上层 AgentLoop）混名。

已入库报告共同支持的判断（综合，非单篇原文）：VLM 先验对语言接地有用；动作接到 VLM 的方式（词表 bin vs 独立 flow 专家 vs 再加 subtask/web co-training）会改变频率、灵巧度和开世界泛化。OpenVLA 证明离散 token 够用到 generalist 基线；π0 证明连续 chunk 才能撑 50 Hz 双臂；π0.5 证明开世界家务靠异构数据而不是换专家；InternVLA-A1.5 证明异质目标还可以用 MoT 再隔一层。

## 为什么重要

后续「具身 + skill」论文默认读者已经知道上面那三句话，但 **skill 不一定是动作头**。[ASPIRE](../sources/aspire.md) 已 ingest 的那条路是程序库，不是 bin / flow / subtask 的第四变体。AtomicVLA / EmbodiedSkills 若把技能写成 token 或 VLA 上层循环，仍未 ingest，不要用 ASPIRE 的数字去填。早期开源 VLA 几乎都在 **离散动作 token + 全量 next-token** 上，数据是 OpenX，观察常常是单张第三人称图——对照 OpenVLA。2025 年起低层默认换成 **π0 的 flow expert**；要做未见过的家、长周期家务，对照的是 **π0.5 的 co-training + subtask**。InternVLA-A1.5 的 MoT / foresight 是再下一层细节，不要当成 VLA 的定义。

对检索：问「VLA 是什么 / 离散 vs flow」走本页；问程序技能库 / 第 N 个任务为什么不从零写走 [具身 skill 自进化](embodied-skill-self-evolution.md)；问 OpenVLA 怎么出动作走 [来源](../sources/openvla.md)；问 50 Hz 动作块走 [π0](../models/pi0.md)；问新房子家务走 [π0.5](../models/pi0.5.md)；问 GDN / latent foresight 走 [InternVLA-A1.5](../models/internvla-a1.5.md)。

## 待追问

- 离散 bin 的量化误差在高频、双臂、接触丰富任务上有多大？OpenVLA §6 只点了推理频率和 action chunking；π0 的对照是把 OpenVLA 重训到 π 混合物，不是 OpenVLA 自己的 Bridge 协议。
- 后续 skill 论文若把「技能」也写成 token：它们是叠在 OpenVLA 的 7D bin 之上，还是替换成 π0/π0.5 的连续专家，或另做原子动作库？AtomicVLA / EmbodiedSkills 尚未 ingest。[ASPIRE](../sources/aspire.md) 走的是程序库，回答不了这个问题。
- InternVLA-A1.5 Stage 1 的 FAST token 与 OpenVLA 的 256-bin、π0.5 预训练 FAST 是否同一离散家族的不同分词器，还是已经换了时间尺度（chunk vs 逐步）？
- π0.7 是否仍用 PaliGemma + FAST→flow 两阶段，本库未 ingest。

## 相关页面

- 开源离散 token 基线：[OpenVLA](../models/openvla.md) · [来源](../sources/openvla.md)
- 连续 flow + action expert：[π0](../models/pi0.md) · [来源](../sources/pi0.md)
- 开世界 co-training：[π0.5](../models/pi0.5.md) · [来源](../sources/pi0.5.md)
- 后续 MoT + latent foresight：[InternVLA-A1.5](../models/internvla-a1.5.md) · [来源](../sources/internvla-a1.5.md)
- 程序库 skill、不是动作头：[ASPIRE](../sources/aspire.md) · [具身 skill 自进化](embodied-skill-self-evolution.md)
- InternVLA 的注意力栈（与 OpenVLA 的 Llama-2 full attention、π0 的 Gemma/PaliGemma 都不同）：[线性注意力与 delta rule](linear-attention-and-delta-rule.md)、[注意力门控](attention-gating.md)
