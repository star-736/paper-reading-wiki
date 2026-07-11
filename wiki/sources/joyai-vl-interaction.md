---
type: Source
title: "JoyAI-VL-Interaction 技术报告"
description: "JD.com 的 8B 视觉驱动交互模型：每秒自主决定说话/静默/委托，AdaCodec 预测式视频编码，4M 时间对齐流式数据 + 角色加权 SFT + GRPO RL，完整可部署系统。"
tags: ["source", "joyai-vl-interaction"]
timestamp: 2026-07-11
resource: "../../raw/2606.14777v1.pdf"
---

# JoyAI-VL-Interaction 技术报告

## 来源

- 原始 PDF：[`raw/2606.14777v1.pdf`](../../raw/2606.14777v1.pdf)（arXiv:2606.14777v1，21 页）
- 标题：JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence
- 团队：JD.com（姚鼎昱、周俊豪、杨晨旭、秦传宇、侯浩文等；指导：司庆宜、段楠、王嘉琦）
- 发表：2026-06（arXiv）
- 关联：AdaCodec（arXiv:2606.02569，[9]）为其视频编码方法的一手出处；streaming-native training 方法来自团队前作（arXiv:2606.08615，[31]）

## 核心结论

论文提出 **interaction model（交互模型）** 范式：模型不像 turn-based 系统那样等用户提问才"睁眼"，而是持续观看视频流，**每秒自主决定**三件事之一——说话（`</response>`）、保持静默继续观看（`</silence>`）、或把难题委托给异步后台模型（`</delegation>`）。这构成了一个"watch-and-do"的人机协作模式，把 AI 辅助扩展到现实世界中大量"错过即不可挽回"的时刻。

两个开源贡献：（1）[JoyAI-VL-Interaction](../models/joyai-vl-interaction.md) 模型及其可迁移训练 recipe；（2）围绕该模型的完整可部署系统（ASR/TTS、记忆、可视化 UI、后台 brain 全部可插拔）。

在 6 个真实流式场景 58 个 case 的人工评测中，JoyAI-VL-Interaction 对 Doubao 视频通话助手胜率 77.6%、对 Gemini 胜率 87.9%。在时间最敏感的场景（监控告警）中对两者均 100% 获胜。

![JoyAI-VL-Interaction Figure 1：交互范式总览。模型持续处理实时视频流，在三个场景中展示三种行为：左上"Timely Warning"——用户要求"着火时提醒我"，火焰出现时模型立即发出"Fire!"告警；左下"Delegate"——用户要求"把这个 app 界面用 HTML 复现"，模型先说"请稍等"再委托后台模型异步生成 HTML 代码，同时继续响应实时计数请求；右侧"Sustained Commentary"——用户要求"实时解说这个艺术视频"，模型随画面变化持续旁白"Title card: Surrealism..."、"Then a dreamlike painting..."。Real-world Live Stream 经 JoyAI-VL-Interaction 产生 Real-time Response/Silence 或 Delegate -> Async Response -> Background Model 两条路径。](../assets/joyai-vl-interaction/fig1-interaction-paradigm.png)

> Figure 1（原文截图，§ 1 Introduction）：三种交互模式——实时告警（event-driven speak）、异步委托（delegate to background brain）、持续解说（sustained commentary），由模型内部每秒决策驱动。

## 架构与训练

### 基座模型

JoyAI-VL-Interaction 构建在 **JoyAI-VL 1.0** 之上，后者的 LLM 初始化自 [Qwen3-8B](../models/qwen3.md)，视觉编码器用 [Qwen3-VL](../models/qwen3-vl.md) 的 ViT，投影层从头训练。JoyAI-VL 1.0 经三阶段训练（representation alignment -> vision-language pre-training -> post-train with OPD + RL），到这一步仍是传统的 turn-based VLM。每秒交互行为通过后续 interaction-training recipe 获得。

### Native Streaming Video Codec（AdaCodec）

视频流不是逐帧全量编码，而是用 **AdaCodec**（团队前作 [9]）做预测式编码：

- 参考帧（reference frame）用完整 ViT 编码（256 visual tokens）；
- 可预测帧之间编码为紧凑 **P-token**（约 16 tokens），由运动向量 + 残差构建；
- **predictive-cost reset**：当预测代价升高（场景变化）时自动开新参考帧；
- 效果：token 预算增长与场景变化量而非帧数成正比，可消费无界视频流。

![JoyAI-VL-Interaction Figure 2：模型总览与 AdaCodec 视频编码。Round 1（t=1）输入 full RGB frame，经 ViT 编码为 256 visual tokens；Round 2（t=2）输入 residual + motion vector，经 P-Tokenizer 编码为仅 16 visual tokens（约 16 倍压缩）。两类 token 一起送入 VL-interaction model，模型输出 </silence>（保持静默）或 </response>（如"The car is moving towards the left."）。](../assets/joyai-vl-interaction/fig2-adacodec-encoding.png)

> Figure 2（原文截图，§ 3 JoyAI-VL-Interaction Model）：AdaCodec 预测式视频编码——参考帧用完整 ViT token、可预测帧用 P-token，按预测代价自适应切换。

### 数据构造：4M+ 时间对齐流式片段

数据包含 **超过 400 万**时间对齐流式片段，分为六个族：

| 族 | 内容 | 构造要点 |
| --- | --- | --- |
| (1) 主动告警与异常检测 | 安防监控中的火焰、跌倒等 | 从 temporal-grounding 标注转换 + web 视频 trigger window 提议-收紧-验证管线；dense precheck 确保标注的是首次出现 |
| (2) 时间对齐 QA | backward / present / forward 三种时序 | backward=证据出现后提问，present=同时，forward=提前提问但要求模型静默等到证据出现再答 |
| (3) 计数与时间感知 | 物体计数、按间隔递增 | 两遍处理：先筛有物体复现的视频，再铺每秒标签；时间条件化对话（每 n 秒加一、限定回答窗口等） |
| (4) 实时评论与解说 | 体育/艺术视频旁白 | 收集真实解说音频用 ASR 恢复，保留自然说话-停顿节奏，产出真实 silence 标签 |
| (5) 多轮闲聊 | egocentric 短视频/长视频/陪伴式 | 两 VLM agent 对话，随机采样时间点只给 3 帧，锚定时间戳防漂移 |
| (6) 委托 episodes | 跨所有族：视频锚定的知识/STEM/推理难题 | 变量延迟回填结果，迫使模型在委托 pending 期间继续观看、响应新轮次、保持静默 |

所有族统一为**每秒三动作标签**（silence / response / delegation），通过**两级验证**（global 全帧+标注一致性、local 标注时刻+回复绑定），只有同时通过才入语料。

### 训练 recipe

**Continue training（SFT 阶段）：** 将时间对齐交互数据混入大量传统 turn-based 数据做 fine-tune。

**训练目标——角色加权 cross-entropy：** 时间对齐数据中 silence 步远多于 response 步，标准 SFT loss 会被 silence 主导。论文对 assistant token 按角色加权：

- $w^\text{first}_\text{silence} = 1$（一段连续 silence 的第一个）
- $w^\text{repeated}_\text{silence} = 0.4$（连续 silence 的后续）
- $w^\text{response} = 1.5$（response 起始）
- 委托不需要单独权重，因为它总是嵌在 response 内部
- 其余位置权重为 1
- 加权 loss 仅用于时间对齐数据；传统 turn-based 数据用标准 SFT loss

$$L(\theta) = -\frac{1}{|A|} \sum_{j \in A} w_j \log p_\theta(y_j \mid y_{<j})$$

**RL 阶段（GRPO）：** SFT 之后加 RL 优化每秒策略。关键设计：

- **Answer-centered window sampling**：长流会展开成数百个 mostly-silent turn，直接做 rollout 不可行。对每个 gold response 构建一条保持流式因果性但只保留与 timing 相关 turn 的轨迹，把 horizon 从数百压缩到个位数。
- **奖励**：正确且在正确窗口内的 response 得分；恰当 silence 得分；恰当委托得分。惩罚 false alarm、mistimed response、degenerate always-respond。
- **委托两部分打分**：是否把真正难的子任务委托（而非简单问题）、是否在后台返回后用好结果（包括 pending 期间保持响应）。
- **LLM judge** 对 response 内容按 task-specific rubric 打分。
- 基础设施：**EasyVideoR1**（[21]），task-aware reward 系统，统一路由 text/image/video/streaming interaction 任务。

### 涌现能力

训练 recipe 按统一的 per-second 格式定义而非绑定特定域，因此可迁移。论文报告两类**从未训练过的涌现能力**：

- **Shopping App Guidance**：引导用户在手机滑动切换 app 界面时完成购物目标——训练数据不含任何 app 界面视频。
- **Travel Scene Commentary**：按要求每 4 秒解说一次——timing action 和 live commentary 在训练数据中从未共现，模型在推理时组合了这两种能力。

## 系统

![JoyAI-VL-Interaction Figure 3：系统总览。Browser Client（Camera / RTSP Feed / Microphone / Query & Control / Visualization UI）经 SDP Signaling / aiortc / RTSP Pull / PyAV 进入 Live Web Backend；Sampling Module 采样后送 Streaming ASR；Session Control Plane + Realtime Safeguards 管理 Inference Adapter；Adaptation Layer 路由到 Interaction Model 或 Background Brain；Action 可达 Digital World；三层记忆（Working / Mid-term / Long-term）由 Session Context Manager 管理。](../assets/joyai-vl-interaction/fig3-system-overview.png)

> Figure 3（原文截图，§ 4 JoyAI-VL-Interaction System）：完整系统架构——"decision in the model, the rest replaceable"。

### 设计原则

**"decision in the model, the rest replaceable"**：模型是唯一决定何时说话/何时委托的组件，其余一切（ASR/TTS、记忆、可视化 UI、后台 brain）都是可替换的转导和编排。每个组件有开箱即用的开源默认实现，也可换成部署方自选模块。

### 双并发循环

- **实时循环**（与用户）：ingest -> adapt -> model 决策 -> speak（TTS -> UI）
- **异步循环**（与后台 brain）：模型 delegate -> 后台模型/agent 异步执行 -> 结果回填入交互流
- 两条循环由 delegate action 缝合：模型在委托 pending 期间继续观看、响应新轮次、保持静默

### 后台 bridge

后台 brain 默认对接用户自己的大模型 API，也可是任意 agent（论文提到 Hermes Agent [16] 和 OpenClaw [18]）。bridge 暴露 **background-agnostic text contract**：前台发出 tagged query，bridge 归一化为 task-id + 委托问题 + 前台备注 + 有界帧快照，在固定超时下异步执行；结果以 started/ready/error 事件返回，完整产物留在 context 外，只有有界摘要织回交互模型。由此闭合"从看到做"的循环。

### 长程记忆

三层层级记忆（基于团队前作 [31]）：

| 层 | 内容 | 覆盖时长 |
| --- | --- | --- |
| 短期 | 最近 $T_s$ 秒的 raw vision tokens | 当前感知 |
| 中期 | 最多 $M$ 条过去短期 chunk 的文本摘要 | $T_m = MT_s$ 秒 |
| 长期 | 最多 $L$ 条从 $M$ 条中期摘要合并的高度压缩块 | $T_l = LMT_s$ 秒 |

三层合计覆盖约 **2 小时**上下文。因为短期以上各层存为文本，记忆形成稳定的 per-chunk prefix，vLLM 可直接 cache 复用。

### vLLM-native serving

直接在 vLLM 上 serve。关键设计：把记忆系统围绕 prefix reuse 设计——文本记忆每 chunk 预填充一次到 KV cache，后续 step 共享该 prefix。这避开了滑动窗口（破坏 prefix reuse）和全历史重算（爆 context window）的问题。运行时含超时/重连/取消机制，session-level lock 防请求堆积，按需 drop stale frame 或按 inference sequence number 回填。

### 可插拔组件

- **ASR/TTS**：基于 FunASR [6] 和 CosyVoice 3 [5]；TTS 播音期间下一句只显示文本不合成，防音频堆积；高频场景（实时解说/翻译）建议关 TTS
- **Visualization UI**：基于 NVIDIA live-vlm-webui，左侧配置视频源和采样率，右侧对话+实时延迟
- **Bring your own modules**：如人脸识别模块让助手记住/认出特定人

## 评测要点

### 评测设计

不跑 offline video-understanding benchmark，而是与**真实部署产品**做 head-to-head 人工评测：Doubao 视频通话（背后是 Seed 2.0）和 Gemini 视频通话（背后是 Gemini-3.1-flash-live）。

6 个场景、58 个 case：监控告警(10) / 实时计数(10) / 实时翻译(10) / 时间感知(10) / 实时解说引导(9) / 长程记忆(9)。

5 名 LLM 研究背景的 rater 盲评（系统身份隐藏、顺序随机），每 case 按 quality 和 timing 两轴各三级评分（good/fair/poor），等权平均后比较两系统得分，给出 win/tie/loss。

### 结果

| 场景 | vs Doubao 胜率 | vs Gemini 胜率 |
| --- | --- | --- |
| 监控告警 | **100.0%** | **100.0%** |
| 实时计数 | 70.0% | 100.0% |
| 实时翻译 | 80.0% | 100.0% |
| 时间感知 | 80.0% | 50.0% |
| 实时解说引导 | 55.6% | 100.0% |
| 长程记忆 | 77.8% | 77.8% |
| **Overall** | **77.6%** | **87.9%** |

### 关键发现

- 监控告警 100% 获胜：turn-based 系统的 polling interval 直接表现为事件延迟
- Doubao 唯一有竞争力的场景是实时解说引导（22.2% 胜 + 22.2% 平），靠的是更大模型的更丰富知识/风格，但其 timing 反而是弱点——周期性外部 trigger 无法判断何时该说话
- Gemini 唯一接近的场景是时间感知（40% 平 + 10% 胜），因部分 case 是事后提问、时间压力低
- 长程记忆中约半数 case 超出 Doubao（约 5 分钟自动挂断）和 Gemini（约 2 分 15 秒）的会话限制，baseline 根本不在场
- JoyAI-VL-Interaction 自身失利仅限于 quality（如解说时偶尔幻觉），归因于 8B 参数规模

## 待追问

- **AdaCodec 的 predictive-cost reset 阈值如何设定？** 论文引用了 AdaCodec 原文 [9]（arXiv:2606.02569），但未在本报告展开。
- **RL 的 answer-centered window sampling 的具体窗口大小？** 论文只说"压缩到个位数 turn"，未给确切窗口长度。
- **三段式记忆的 $T_s$ / $M$ / $L$ 具体取值？** 论文给出结构但未公开参数。
- **2 小时上下文的延迟实测数据？** 论文声称 sub-second latency 但未给系统级延迟 breakdown。
- **与 TML interaction model 的直接定量对比？** 论文只做定性对比（vision-first vs audio-video），未在同一 benchmark 上对打。
- **委托的 background-agnostic protocol 的完整格式？** Appendix 给了 delegation 训练数据示例（Listing 3），但系统侧 protocol 规范需查 repo。

## 相关页面

- [JoyAI-VL-Interaction](../models/joyai-vl-interaction.md) - 模型实体页
- [多模态 Agentic 训练](../concepts/multimodal-agentic-training.md) - Kimi K2.5 的视觉 agentic 路线与本文的 watch-and-interact 形成对照
- [Any-to-any 多模态 serving](../concepts/any-to-any-multimodal-serving.md) - vLLM-Omni 的 stage graph 与本文的 vLLM-native serving + 双循环架构形成对照
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md) - 本文的 GRPO + answer-centered window sampling 是 agent RL 的一种流式变体
- [Agentic 评测体系](../concepts/agentic-evaluation-benchmarks.md) - 本文不跑 offline benchmark 而做真实产品 head-to-head 的评测方式值得对比
