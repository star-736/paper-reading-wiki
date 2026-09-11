---
type: Concept
title: "Vision-Language-Action"
description: "把预训练视觉语言模型接到机器人控制：看图、读指令、输出动作。SayCan 是分层 LLM planner，不是 VLA；RT-2 造词并封闭落地离散 token；OpenVLA 是开源 256-bin；π0 是连续 flow；π0.5 是开世界 co-training。"
tags: ["concept", "vla", "robotics"]
timestamp: 2026-09-12
---

# Vision-Language-Action

## 定义

Vision-Language-Action（VLA）是一类机器人策略：把已经在大规模图文上预训练好的视觉语言模型（VLM）接到控制回路里。模型看见相机画面、读懂自然语言指令，直接给出机器人下一步（或接下来一段）要执行的动作，而不是先输出一段文字计划再交给另一个控制器。后一种分层——LLM 只选技能名、低层另有控制器——是 [SayCan](../sources/saycan.md)，不是 VLA。

可以把它想成「会看、会读的聊天模型，输出端接上机器人」。训练时监督来自真实轨迹里的控制量；推理时把网络输出还原成电机/末端可以执行的数。**VLA 这个词是 [RT-2](../sources/rt-2.md) 造的**（摘要、§1）：把现成 VLM 在网页 VQA 与机器人轨迹上 co-fine-tune，动作写成 text token，不加新参数。 [OpenVLA](../sources/openvla.md) 把同一套离散 token 写成可复现的开源 7B。

标准骨架是 **预训练 VLM + 动作头**。读 AtomicVLA 之前，动作头只需记住这几句话：

1. **离散动作 token 家族。** [RT-2](../models/rt-2.md) 是封闭定义：8 维（6-DoF + 夹爪 + terminate）、均匀 256-bin，与互联网数据 co-fine-tune，55B 闭环 1–3 Hz。 [OpenVLA](../models/openvla.md) 是开源复现：7 维、分位数 256-bin，覆盖进 Llama 词表，只在机器人数据上 fine-tune。两者动作头同类；OpenVLA 表里赢的是后续 **RT-2-X**，不是原版 RT-2。
2. **[π0](../models/pi0.md) = 连续 flow + action expert。** PaliGemma 上看图读指令，300M 专家用 flow matching 一次吐出连续动作块（\(H=50\)，最高 50 Hz）。2026 年论文里的默认低层执行器走这条，而不是 bin。
3. **[π0.5](../models/pi0.5.md) = 开世界 co-training。** 架构仍是 π0 的 flow expert，但同一模型先预测 semantic subtask，再出底层动作；数据混多机器人、web 语义和口头逐步指令。[EmbodiedSkills](../sources/embodied-skills.md) 的低层执行器就是这一代，不是 256-bin。

[InternVLA-A1.5](../models/internvla-a1.5.md) 是再往后的 MoT + latent foresight 实例：Stage 1 仍用 FAST 离散 token，Stage 2 起是 flow-matching 连续 chunk，并继续做 VQA / subtask；真机表拿 π0.5 当对照。

VLA 不是「多模态聊天 + 另外写个脚本控机器人」：动作在同一个网络里、用同一套（或显式共享的）视觉语言表示预测。它也不是本库里的 GUI agent（点屏幕）或全模态语音模型（听/说）；默认输入是机器人相机与语言指令。不要发明音频。

## 跨报告信号

- **[SayCan](../sources/saycan.md)（2022，分层 LLM planner，不是 VLA）**：冻结 PaLM 给技能标签打 \(p(\ell_\pi \mid i)\)，技能 value function 给 \(p(c_\pi \mid s, \ell_\pi)\)，相乘后执行 BC-Z / MT-Opt 低层策略。101 条厨房指令上规划 84%、执行 74%；去 affordance 规划 67%，整句喂给低层执行 0%（Table 2）。LLM 不输出末端动作。RT-2 §1 针对的就是这一层。
- **[RT-2](../sources/rt-2.md)（2023，VLA 定义、封闭离散 token）**：PaLI-X 5B/55B 与 PaLM-E 12B；动作 8 个数均匀 256-bin，写成 text token；网页 VQA + RT-1 厨房数据 co-fine-tune（混合物里机器人约 50–66%）。约 6,000 次评测：已见任务与 RT-1 持平，未见平均 62% vs RT-1 32% / MOO 35%；涌现评测 PaLI-X-55B 平均 60% vs RT-1 17%（Table 4–5）。消融：从零训失败，co-fine-tune > 只 fine-tune，55B > 5B（Table 6）。局限写在原文 §5：不教新动作，55B 仅 1–3 Hz。**不是** OpenVLA 对照的 RT-2-X。
- **[OpenVLA](../sources/openvla.md)（2024，开源离散 token 基线）**：Prismatic-7B = Llama 2 7B + DINOv2/SigLIP 融合视觉；Open X-Embodiment 约 970k 真实轨迹上全量 fine-tune；只在动作 token 上算交叉熵。跨 WidowX 与 Google robot 共 29 任务，以 7B 超过 55B 的封闭 **RT-2-X** 约 16.5 个百分点（摘要；Table 4+6）。语义泛化一项 RT-2-X 仍强，论文归因于对方做了互联网 co-fine-tune，而 OpenVLA 只在机器人数据上 fine-tune——这正是 RT-2 原论文 Table 6 的配方差异。局限写在原文 §6：单图、无本体感觉、推理频率不够双臂高频、成功率通常 <90%。
- **[π0](../sources/pi0.md)（RSS 2025，连续 flow + action expert）**：PaliGemma 3B + 300M flow matching 专家；观察是多路 RGB + 语言 + 本体感觉，输出 \(H=50\) 连续动作块，最高 50 Hz。预训练约 10,000 小时、7 种配置 × 68 任务，外加 OXE。原文把 OpenVLA 重训到同一混合物：自回归离散化撑不住 chunk 与高频，out-of-box 全面落后（§VI-A、Figure 7）。这是 **原文对照**，不是后文推断。新技能仍要 fine-tune；高层语言是外挂 VLM，不是自己的 subtask 头。
- **[π0.5](../sources/pi0.5.md)（2025，开世界 co-training）**：骨干仍是 π0 的 PaliGemma + flow expert。改动是 FAST 离散预训练再长 expert、**同一模型**先预测 semantic subtask，以及异构数据（MM 约 400 小时 / ~100 家 + ME/CE/HL/WD/VI）。评测在训练未见的厨房/卧室；97.6% 预训练样本不是目标平台家务（§I）。InternVLA-A1.5 真机表与 LIBERO-Plus 拿它当对照。
- **[InternVLA-A1.5](../sources/internvla-a1.5.md)（2026，MoT + latent foresight）**：不是把 OpenVLA 换个 backbone，也不是 π0.5 换 Qwen。VLM 是 Qwen-3.5 2B（3:1 GDN:full attention），旁边 460M unified expert 只在 full attention 层交互。Stage 2 起 flow-matching chunk（size 50）+ 从冻结 WAN2.2 蒸馏的 foresight token。语义侧继续 VQA / subtask。评测协议、数据（1.2M episodes + 3M 多模态）不可与 OpenVLA 附录 E 或 π0.5 家庭进度条直接横比。
- **[ASPIRE](../sources/aspire.md) 已 ingest，但不是第四种动作头**：它写/改 code-as-policy 程序，把验证过的修复写入 skill library；仿真 coder 是冻结的 Claude Opus 4.6，不更新 VLA 权重。LIBERO-Pro 表里 OpenVLA / π0 / π0.5 是对照，不是前作。入口见 [具身 skill 自进化](embodied-skill-self-evolution.md)。
- **[EmbodiedSkills](../sources/embodied-skills.md) 已 ingest，也不是第四种动作头**：它把 π0.5 这一代低层包进 guarded AgentLoop。skill 是 typed 执行合同，policy 只提出 proposal，runtime 先验后验。RoboTwin 2.0 86.20% / LIBERO 97.40% 是**任务特化低层策略**的执行成绩；环本身的证据是消融（去验证 −38.0 pp、去 subtask −51.8 pp）。不要和 ASPIRE 的程序库、EmbodiSkill 的 training-free reflection 混名。
- **未 ingest、只作外部线索**：RT-2-X / Open X-Embodiment（OXE 上的 55B，OpenVLA 的对照）尚未单独 ingest。Inner Monologue（SayCan 的闭环后作）、AtomicVLA、EmbodiSkill、π0.7 尚未 ingest。

已入库报告共同支持的判断（综合，非单篇原文）：VLM 先验对语言接地有用；**LLM 可以只当 planner**（SayCan：技能表上打分 × value function），也可以把 VLM 直接接到低层动作（RT-2 起的 VLA）。**网页数据是否留在微调阶段**会改变语义泛化（RT-2 的 co-fine-tune vs OpenVLA 的纯机器人 fine-tune）；动作接到 VLM 的方式（词表 bin vs 独立 flow 专家 vs 再加 subtask/web co-training）会改变频率、灵巧度和开世界泛化。SayCan 证明冻结 LLM 的厨房常识能用，只要有 affordance 挡住不可行步骤；RT-2 证明离散 token + 网页共微调能把符号/人物/多语迁到控制；OpenVLA 证明同一动作头可以开源、用更多本体数据超过封闭 RT-2-X；π0 证明连续 chunk 才能撑 50 Hz 双臂；π0.5 证明开世界家务靠异构数据而不是换专家；InternVLA-A1.5 证明异质目标还可以用 MoT 再隔一层。

## 为什么重要

后续「具身 + skill」论文默认读者已经知道上面那几句话，但 **skill 不一定是动作头**。[ASPIRE](../sources/aspire.md) 的那条路是程序库；[EmbodiedSkills](../sources/embodied-skills.md) 的那条路是 VLA 上层的 typed 执行合同。都不是 bin / flow / subtask 的第四变体。AtomicVLA 若把技能写成 token 或 skill-MoE，仍未 ingest，不要用这两篇的数字去填。VLA 的定义对照 [RT-2](../sources/rt-2.md)。早期开源 VLA 几乎都在 **离散动作 token + 全量 next-token** 上，数据是 OpenX，观察常常是单张第三人称图——对照 OpenVLA。2025 年起低层默认换成 **π0 的 flow expert**；要做未见过的家、长周期家务，对照的是 **π0.5 的 co-training + subtask**。InternVLA-A1.5 的 MoT / foresight 是再下一层细节，不要当成 VLA 的定义。

对检索：问「VLA 是什么 / 谁造的这个词 / 离散 vs flow」走本页；问 LLM 只选技能、低层另训走 [SayCan](../sources/saycan.md)；问封闭 55B 原版走 [RT-2](../models/rt-2.md)；问开源 256-bin 走 [OpenVLA](../sources/openvla.md)；问 50 Hz 动作块走 [π0](../models/pi0.md)；问新房子家务走 [π0.5](../models/pi0.5.md)；问 GDN / latent foresight 走 [InternVLA-A1.5](../models/internvla-a1.5.md)；问程序技能库走 [具身 skill 自进化](embodied-skill-self-evolution.md)；问 VLA 上层如何先验后验走 [EmbodiedSkills](../sources/embodied-skills.md) 和 [Agent harness](agent-harness.md)。

## 待追问

- 离散 bin 的量化误差在高频、双臂、接触丰富任务上有多大？RT-2 报 55B 1–3 Hz、均匀 bin；OpenVLA §6 只点了推理频率和 action chunking；π0 的对照是把 OpenVLA 重训到 π 混合物，不是 RT-2 / OpenVLA 自己的协议。
- 后续 skill 论文若把「技能」也写成 token：它们是叠在 OpenVLA 的 7D bin 之上，还是替换成 π0/π0.5 的连续专家，或另做原子动作库？[EmbodiedSkills](../sources/embodied-skills.md) 的低层是 π0.5 连续专家，技能本身是 runtime 合同不是 token。AtomicVLA 尚未 ingest。[ASPIRE](../sources/aspire.md) 走的是程序库。
- InternVLA-A1.5 Stage 1 的 FAST token 与 OpenVLA 的 256-bin、π0.5 预训练 FAST 是否同一离散家族的不同分词器，还是已经换了时间尺度（chunk vs 逐步）？
- π0.7 是否仍用 PaliGemma + FAST→flow 两阶段，本库未 ingest。

## 相关页面

- 分层 LLM planner，不是动作头：[SayCan](../sources/saycan.md)
- VLA 定义、封闭离散 token：[RT-2](../models/rt-2.md) · [来源](../sources/rt-2.md)
- 开源离散 token 基线：[OpenVLA](../models/openvla.md) · [来源](../sources/openvla.md)
- 连续 flow + action expert：[π0](../models/pi0.md) · [来源](../sources/pi0.md)
- 开世界 co-training：[π0.5](../models/pi0.5.md) · [来源](../sources/pi0.5.md)
- 后续 MoT + latent foresight：[InternVLA-A1.5](../models/internvla-a1.5.md) · [来源](../sources/internvla-a1.5.md)
- 程序库 skill、不是动作头：[ASPIRE](../sources/aspire.md) · [具身 skill 自进化](embodied-skill-self-evolution.md)
- VLA 上层 AgentLoop，也不是动作头：[EmbodiedSkills](../sources/embodied-skills.md) · [Agent harness](agent-harness.md)
- InternVLA 的注意力栈（与 OpenVLA 的 Llama-2 full attention、π0 的 Gemma/PaliGemma 都不同）：[线性注意力与 delta rule](linear-attention-and-delta-rule.md)、[注意力门控](attention-gating.md)
