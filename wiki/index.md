# 索引

这是论文阅读知识库的入口页。后续查询先读这里，再进入相关的来源页、模型页、概念页或比较页。

> 维护备忘：已完成动作记在 [log.md](log.md)（时间线）；未完成的工程待办记在 [TODO.md](TODO.md)。

## 来源

- [GLM-5 技术报告](sources/glm-5.md) - GLM-5 的 arXiv 技术报告，重点是 agentic engineering、DSA 和异步 RL。
- [Macaron-V1 技术报告](sources/macaron-v1.md) - Mind Lab 的开放 agent-model 家族，以 frozen base + Mixture-of-LoRA、HCP 版本化 harness 和 MindForge RSI 为核心；当前未证明跨代持续学习增益。
- [Intern-S2-Mobius 技术报告](sources/intern-s2-mobius.md) - 上海 AI Lab 的架构报告：用全局共享 FFN Memory 将知识存储与 Self-Attn Reasoner 解耦；7B 配对预训练报 1.6× 数据效率，35B 转换路线报更短 CoT / 更快端到端推理，但机制与公平基线仍待核。
- [GLM-5.3 官方发布博客](sources/glm-5-3-blog.md) - 同一 GLM-5.2 base model 上只扩展后训练；公开长周期可验证环境、`slime` 对齐/调度与 coding/cyber 评测，但不是技术报告。
- [GLM-5V-Turbo 技术报告](sources/glm-5v-turbo.md) - GLM-5V-Turbo 的 arXiv 报告，重点是 CogViT 视觉编码器、MMTP、30+ 类别多模态 RL 和 agent 框架集成（Claude Code / AutoClaw）。
- [MiMo-V2-Flash 技术报告](sources/mimo-v2-flash.md) - MiMo-V2-Flash 的 arXiv 技术报告，重点是混合 SWA/GA 注意力、MOPD 和 MTP 加速。
- [DeepSeek-V2 技术报告](sources/deepseek-v2.md) - DeepSeek-V2 的 arXiv 论文，Multi-Head Latent Attention（MLA）的首次提出处。
- [DeepSeek-V3.2 技术报告](sources/deepseek-v32.md) - DeepSeek-V3.2 的 arXiv 论文，引入 DeepSeek Sparse Attention（DSA）。
- [DeepSeek-V4 技术报告](sources/deepseek-v4.md) - DeepSeek-V4 的 Hugging Face 官方 PDF，重点是百万 token 上下文效率。
- [MiniMax-M2 Series 技术报告](sources/minimax-m2-series.md) - MiniMax-M2/M2.7 的 arXiv 技术报告，重点是低激活 MoE、Forge RL 和 self-evolution。
- [Kimi K2.5 技术报告](sources/kimi-k2.5.md) - Kimi K2.5 的 arXiv 技术报告，重点是视觉 agentic intelligence、zero-vision SFT 和 Agent Swarm。
- [MSA 技术报告](sources/msa.md) - MiniMax Sparse Attention 的 arXiv 报告，GQA-block 级稀疏 + 每个 group 独立 top-k，1M context 下 14× prefill / 7× decode。
- [IndexCache 技术报告](sources/indexcache.md) - 清华 + Z.ai 在 DSA 上做跨层索引复用，1/4 retention 即可在 30B 和 GLM-5 上保留质量并拿到端到端加速。
- [Kimi Linear 技术报告](sources/kimi-linear.md) - Moonshot AI 的混合线性注意力，KDA（细粒度门 delta rule）3:1 配 Full MLA，首次在公平对比下全面追平 full attention，1M context KV 降 75%、decode 6.3×。
- [Linear Attention Architectures 技术报告](sources/linear-attention-architectures.md) - ETH Zurich 的机制比较研究：统一 DeltaNet/GDN/KDA/GDN-2 递归记忆表述，CLVR 将内部 write value 路由到共享 residual stream；增益方向一致但仍属 single-run 初步证据。
- [Gated Attention 技术报告](sources/gated-attention.md) - Qwen 团队系统消融 30 个门控变体，SDPA 输出 head-specific sigmoid 门最优（注入非线性 + 消除 attention sink），NeurIPS 2025 Best Paper，已用于 Qwen3-Next 系与 Trinity Large。
- [Gated DeltaNet 技术报告](sources/gated-delta-net.md) - NVIDIA + MIT 的 ICLR 2025 论文，提出 gated delta rule（门控快速清空 + delta 定向更新互补），KDA 与 Qwen3-Next 系线性层的直接前身。
- [Qwen3-Coder-Next 技术报告](sources/qwen3-coder-next.md) - 基于 Qwen3-Next 的 80B-A3B 编码 agent 模型，继承 GDN + gated attention 混合栈，主打 agentic coding 训练。
- [Qwen3.5-Omni 技术报告](sources/qwen3.5-omni.md) - Qwen 全模态家族最新代，Thinker/Talker 用含 GDN 的 Hybrid Attention MoE，把线性注意力降 KV-cache 延伸到长音视频。
- [Qwen3-Next 官方博客](sources/qwen3-next-blog.md) - Qwen3-Next 无技术报告，本官方博客是其架构设计动机的一手出处：3:1 混合（75% GDN / 25% standard）、选 GDN 因 in-context learning 强于 SWA/Mamba2、全局层加 output gating 去 sink、Zero-Centered RMSNorm + 512-expert MoE + MTP。
- [Qwen3 技术报告](sources/qwen3.md) - Qwen 系列 2025-05 基座报告（arXiv:2505.09388），标准 GQA + RoPE + RMSNorm + MoE，36T tokens / 119 语言；后训练核心 = 统一 thinking/non-thinking 双模式 + thinking budget + Strong-to-Weak Distillation 完胜 RL（1/10 GPU 时长）。Qwen3-Next/3.5/3-Coder-Next/3.5-Omni/Qwen3-VL 的基座前作。
- [Qwen3-VL 技术报告](sources/qwen3-vl.md) - Qwen3-VL 多模态家族报告（arXiv:2511.21631），256K 原生上下文；三块架构升级 = Interleaved MRoPE（t/h/w 频谱均衡）+ DeepStack（ViT 中间 3 层 → LLM 前 3 层 residual add）+ 文本时间戳替换 T-RoPE。LLM backbone 是**标准 GQA 的 Qwen3**，与 Qwen3.5-Omni 的 hybrid 基座是两条路。
- [vLLM-Omni 技术报告](sources/vllm-omni.md) - 面向 any-to-any 多模态模型的 fully disaggregated serving 系统：stage graph 拆分 AR / DiT / encoder 等阶段，独立批处理、资源配置与 unified connector 传输，Qwen3-Omni JCT 最高降 91.4%。
- [FreeToken](sources/freetoken.md) - 端侧 MoE serving：CPU 常驻 expert 池 + GPU 共享 LRU + 带宽自适应 $q^{\star}$ 分流 miss，8GB 笔记本到单卡工作站交互式服务 35B–753B；相对 llama.cpp / KTransformers decode 1.3–2.3×。
- [LMCache 技术报告](sources/lmcache.md) - 企业级 KV cache 层：从 vLLM/SGLang 抽出 paged KV，chunked I/O + layer-wise pipelining 做跨查询复用和 PD 传输；同 TTFT 吞吐 2.3–14×，CPU 加载带宽 400 vs 88 Gbps。
- [Thinking Machines Lab On-Policy Distillation 博客](sources/thinking-machines-on-policy-distillation.md) - Kevin Lu 2025-10-27 发表，GLM-5（ref [28]）/ MiMo MOPD 共同引用的 OPD 算法源头。Per-token reverse KL、三方对照表（SFT / RL / OPD = off-policy+dense / on-policy+sparse / on-policy+dense）、`O(1)` vs `O(N)` bits/episode 信息论分析、personalization 召回实验是 GLM-5 cross-stage distillation 思路的直接来源。
- [Agentic Reinforced Policy Optimization](sources/agentic-reinforced-policy-optimization.md) - 人大 + 快手的 agentic RL 算法论文：发现工具反馈后 token entropy spike，用 entropy-based adaptive rollout 在高熵工具调用步分叉 partial rollouts，并用 advantage attribution 学 step-level tool-use 行为。
- [VAPO 技术报告](sources/vapo.md) - ByteDance Seed 的 long-CoT value-model-based PPO：校准 critic、解耦 actor/critic GAE 并按 response 长度自适应 $\lambda$，Qwen2.5-32B 的 AIME 2024 avg@32 报 60.4；证据仍限单 backbone / 单 benchmark。
- [DAPO 技术报告](sources/dapo.md) - ByteDance Seed + 清华 AIR 等开源大规模 LLM RL 系统，四件套（Clip-Higher / Dynamic Sampling / token-level loss / overlong shaping）把 Qwen2.5-32B AIME24 avg@32 从 naive GRPO 30 提到 50。
- [Group Sequence Policy Optimization](sources/group-sequence-policy-optimization.md) - Qwen 团队提出 GSPO：用 sequence likelihood ratio 与 sequence-level clipping 替代 GRPO token-level ratio，稳定 Qwen3-30B-A3B 等 MoE RL 训练。
- [Soft Adaptive Policy Optimization](sources/soft-adaptive-policy-optimization.md) - Qwen 团队提出 SAPO：用 temperature-controlled soft gate 替代 hard clipping，兼顾 sequence coherence 与 token adaptivity，并用于 Qwen3-VL RL 训练。
- [DSpark 技术报告](sources/dspark.md) - PKU + DeepSeek-AI 的 speculative decoding 框架：semi-AR drafter（parallel backbone + 轻量 sequential head）+ confidence-scheduled verification，V4 preview 上线两周后整体替换生产端 MTP-1，per-user 速度 V4-Flash +60–85% / V4-Pro +57–78%。
- [DoReMi](sources/doremi.md) - Google DeepMind + Stanford 的 NeurIPS 2023 论文：用 280M proxy model 跑 Group DRO 优化 domain weights（仅 8% 额外 FLOPs），让 8B 模型预训练加速 2.6x，所有 domain perplexity 改善，不需下游任务知识。
- [TANDEM](sources/tandem.md) - JD.com + Oxford + 人大的 NeurIPS 2025 论文：把数据混合优化建模为 bi-level optimization，用 twin network（proxy + 动态 reference）的 loss 差度量 domain 边际收益，收敛率 O(T^{-1/4})，在 data-restricted 和 SFT 场景显著优于 DoReMi/DoGE。
- [Gemma 4 技术报告](sources/gemma-4.md) - Google DeepMind 的 Gemma 4 arXiv 报告，原生多模态 dense + MoE 家族（E2B~31B），重点是 encoder-free 12B、5:1 SWA/GA + key-as-value + p-RoPE 长上下文、MTP drafter 和 QAT 量化。
- [InternVLA-A1.5 技术报告](sources/internvla-a1.5.md) - 上海 AI Lab 的统一 VLA 机器人模型，Qwen-3.5 2B backbone + 460M unified expert + latent foresight（frozen WAN2.2 蒸馏），6 项仿真 benchmark 全部最优。
- [HunyuanOCR-1.5 技术报告](sources/hunyuan-ocr-1.5.md) - 腾讯 + 中科院信工所 + 南开的轻量端到端 OCR VLM 报告，DFlash block-diffusion 推测解码（Transformers 6.37× / vLLM 2.14×）+ Agentic Data Flow 数据构造 + 三组件 reward RL。
- [UniClawBench](sources/uniclawbench.md) - HKU MMLab + Meituan 的 proactive agent 评测基准，400 双语真实世界任务，5 维能力分解，三角色闭环评测（executor + hidden supervisor + user simulator），跨模型×跨框架实验揭示 framework > model。
- [KAT-Coder-V2 技术报告](sources/kat-coder-v2.md) - 快手 KwaiKAT 的 agentic coding 模型，Specialize-then-Unify 五域分治 + KwaiEnv 模块化沙箱 + MCLA 稳定 MoE RL + Tree Training 6.2× 加速 + OPD 专家融合。
- [KAT-Coder-V2.5 技术报告](sources/kat-coder-v2.5.md) - 快手 KwaiKAT 的 coding-focused agentic 模型，AutoBuilder + KwaiClawEnv + harness randomization + asymmetric PPO with hindsight critic + 长上下文 MOPD 稳定化（cold start + drift-aware truncation）。
- [daVinci-Agency](sources/davinci-agency.md) - SII-GAIR 的长周期 agent 数据合成范式：从 GitHub chain-of-PRs 挖掘 task decomposition / long-term consistency / refinement 监督，239 样本 SFT GLM-4.6 即在 Toolathlon +47%、AVG 超过 66k 样本的 SWE-Smith。
- [Seed2.0 Model Card](sources/seed2.md) - 字节跳动 Seed 团队的 Seed2.0 Series（Pro / Lite / Mini）Model Card，含 MaaS 部署洞察、四维评测框架（Science Discovery / Vibe Coding / Context Learning / Real-World Tasks）和真实世界 case studies，不含架构/训练细节。
- [KVpop 技术报告](sources/kvpop.md) - NXAI + JKU Linz（Hochreiter 团队）的 learned eviction 方法：future-attention target 在 eviction boundary 监督 keep-or-drop，mLSTM 延迟打分利用近未来上下文，Qwen3-8B 88% 压缩下保留 100% teacher 性能。
- [VibeThinker-3B 技术报告](sources/vibethinker-3b.md) - 新浪微博 3B dense reasoning 模型，Spectrum-to-Signal 后训练范式（MGPO + curriculum SFT + Long2Short RL + offline self-distillation + Instruct RL + CLR test-time scaling），AIME26 94.3 追平旗舰，提出 Parametric Compression-Coverage Hypothesis。
- [LoopCoder-v2](sources/loopcoder-v2.md) - 北航 + IQuest Research 的 PLT loop-count 选择研究：7B coder 从头训练 18T tokens，gain–cost 视角发现 R=2 最优（SWE-bench Verified 64.4%）、R≥3 退化，per-loop 可解释性诊断解释饱和机制。
- [Looped Language Models Improve Compositional Tool Calling](sources/looped-tool-calling.md) - Cambridge 的循环模型工具调用研究：循环计算主要提升多调用组合和 output-to-input 依赖绑定；Ouro adaptive exit 以较低平均循环数取得更好的算力—性能折中，但证据仍限静态单轮 benchmark。
- [BDH-CQ 技术报告](sources/bdh-cq.md) - Pathway 的 150M ARC 推理系统：演示写入 recurrent memory、查询在连续 latent workspace 中迭代，不输出中间 CoT；报告 ARC-AGI-1 29.5% pass@2、$0.00070/task 的特定成本口径效率点，核心实现仍 proprietary。
- [MiniCPM-o 4.5 技术报告](sources/minicpm-o-4-5.md) - OpenBMB 9B 全双工全模态交互模型，Omni-Flow 框架沿共享时间轴对齐多模态 I/O 流，LLM 只生成文本 token 委托轻量 speech decoder 生成语音，TAIL 时间对齐交错，端侧 INT4 < 12GB。
- [Keye-VL-2.0 技术报告](sources/keye-vl-2.md) - 快手 Keye Team 的开源 30B-A3B 多模态 MoE 模型，首个把 DSA 适配到 GQA 多模态架构（indexer MQA + aggregation GQA），256K 长视频上下文 + Cross-Modal MOPD（13 个 RL teacher，top-k overlap estimator）。
- [JoyAI-VL-Interaction 技术报告](sources/joyai-vl-interaction.md) - JD.com 的 8B 视觉驱动交互模型：每秒自主决定说话/静默/委托后台，AdaCodec 预测式视频编码 + 4M 时间对齐数据 + 角色加权 SFT + GRPO RL，完整可部署系统，vs Doubao/Gemini 人工盲评胜率 77.6%/87.9%。
- [MOSS-VL 技术报告](sources/moss-vl.md) - OpenMOSS 的 11.3B 实时 VLM：视觉 token 留在 gated cross-attention 通道外，XRoPE 对齐时空位置，Realtime-SFT 学说话/静默/修订；流式公开基准量化 L2--L4，L5“生成时仍感知”仍待专门 benchmark。
- [Xiaomi-GUI-0 技术报告](sources/xiaomi-gui-0.md) - 小米 SeerRay 的 native end-to-end multimodal GUI agent，真机为主的混合基础设施 + error-driven data flywheel + 三阶段训练（SFT→Step RL→Agentic RL），RealMobile 72.0% / AndroidWorld 78.9%。
- [UI-Mate 技术报告](sources/ui-mate.md) - 腾讯 HY Frontier 的开源权重 foundation GUI agent：环境接地数据飞轮 + 在线 GRPO RL + DemoCUA 上下文示范；OSWorld-Verified 77.0%、WindowsAgentArena 66.2%，OSWorkerBench 用成对协议隔离一条示范的增益。
- [Agent-World 技术报告](sources/agent-world.md) - 人大 + ByteDance Seed 的自演化 agent 训练场：Agentic Environment-Task Discovery（1978 环境 / 19822 工具，MCP/工具文档/PRD 挖主题 + graph-based + programmatic 任务合成）+ Continuous Self-Evolving Agent Training（多环境 GRPO RL + 诊断弱环境→定向扩展→continue RL 的 co-evolution），跨 23 benchmark。
- [Qwen-AgentWorld 技术报告](sources/qwen-agent-world.md) - Qwen Team 的 native language world model（LWM），首个覆盖 7 域（MCP/Search/Terminal/SWE/Android/Web/OS）的 agentic 环境模拟器，三阶段 CPT→SFT→RL（"injects/activates/sharpens"）+ AgentWorldBench（5 维 rubric reference-grounded judging）+ 解耦（Sim RL 可控模拟超真实环境）/ 统一（LWM warm-up 跨任务迁移）两种 agent 增强范式。
- [MinerU2.5-Pro 技术报告](sources/mineru-2-5-pro.md) - 上海 AI Lab + PKU + SJTU + 商汤的数据中心文档解析报告，固定 1.2B 架构，Data Engine（DDAS + CMCV + Judge-and-Refine）+ 三阶段训练把 OmniDocBench v1.6 从 92.98 推到 95.69，并修正评测匹配偏差（MGAM）+ 引入 Hard 子集；含 HunyuanOCR 1.0 自报分与统一重测分的跨源分歧。
- [GLM-OCR 技术报告](sources/glm-ocr.md) - 智谱 AI + 清华的 0.9B 轻量 OCR VLM，CogViT + GLM 解码器 + MTP（训练+推理共用共享参数多头，~50% 吞吐提升）+ 两阶段 pipeline + 文档解析/KIE 双任务统一，OmniDocBench v1.5 SOTA 94.62；含 v1.5 自报 vs v1.6 统一重测的评测版本差异。
- [MinerU2.5 技术报告](sources/mineru-2-5.md) - 上海 AI Lab + PKU + SJTU 的 1.2B 解耦文档解析 VLM，coarse-to-fine 两阶段 + Data Engine（IMIC 单模型推理一致性挖 hard case）。MinerU2.5-Pro 的基座，IMIC 是 CMCV 的改进对象。
- [Ling and Ring 2.6 技术报告](sources/ling-2.6.md) - Inclusion AI 的 Ling-2.6 / Ring-2.6 万亿参数 agentic 模型族，7:1 Lightning Attention + MLA 混合线性注意力 retrofit、token efficiency 后训练、KPop agentic RL。
- [Unlimited OCR Works](sources/unlimited-ocr.md) - Baidu 的 OCR 报告，提出 Reference Sliding Window Attention（R-SWA），解码时保持 KV cache 恒定，单次前向传播转录数十页文档。
- [Mach-Mind-4-Flash 技术报告](sources/mach-mind-4-flash.md) - 理想汽车的 35B MoE agentic 模型（3B 激活），基于 Qwen3.5-35B-A3B，specialization-then-integration 后训练（三轨并行 RL + MOPD 融合 + HMPO token 效率），统一 RL/OPD 训练框架。
- [Mi-Memory 技术报告](sources/mi-memory.md) - 小米 Darwin Agent Team 的 Personal AI 记忆全生命周期框架：Structure（MemStack 分层记忆）/ Expansion（MemSense IKB + MemFuse 跨设备因果融合）/ Evolution（D2ACCI 诊断环 + E2MEND 有界策略搜索）/ Deployment（LiteMem Markdown/Git 仓库原生基底），共享审计契约四类工件贯穿全链路。
- [nrehiew 博客：SFT, RL, and OPD Through a Distributional Lens](sources/nrehiew-sft-rl-opd.md) - 分布视角统一 SFT / RL / OPD 三方法。核心论点：on-policy 数据（非 RL 本身或显式 KL 惩罚）是抗遗忘承重墙。关键实验：OPD student 不论从 SFT 还是 RL teacher 蒸馏结果几乎一致。覆盖 OPSD 变体、RL 抗遗忘三解释审视、student 超越 teacher 机制、pipeline 趋势（GLM-5 / DeepSeek-V4 最终 checkpoint 不经 RL）。
- [Kimi K3 技术报告](sources/kimi-k3.md) - Moonshot AI 首个开源 3T 级模型（2.8T/104B 激活），KDA scaled sigmoid + Attention Residuals + Stable LatentMoE + 原生视觉 + 1M 上下文，2.5× scaling efficiency，9-专家 RL + MOPD + AgentENV microVM 沙箱。
- [Laguna M.1/XS.2 技术报告](sources/laguna-m1-xs2.md) - Poolside 的 MoE agentic coding 模型族（M.1 225.8B/23.4B、XS.2 33.4B/3B），Model Factory 工业化流程（M.1 后五周交付 XS.2）、AutoMixer 数据混合、3:1 SWA/GA + softplus 门控、WSD 缩放律、CISPO agentic RL、合成代码环境贯穿 SFT/RL。
- [DynamixSFT 技术报告](sources/dynamix-sft.md) - MSRA + UMich + KAIST 的 SFT 指令微调数据集动态混合优化：把数据集采样建模为 Multi-Armed Bandit，Prior-scaled Boltzmann Exploration 软锚定原始比例 + 1-Step Look-ahead Reward 反映当前训练动力学，TÜLU-2/3 上 +5.1%/+5.3% 且仅 +12.7% 开销；与 DoReMi/RegMix/TANDEM 的 proxy-model 谱系范式分叉。
- [Aioli 技术报告](sources/aioli.md) - Stanford + NYU 的数据混合统一框架（LMO），把 DoReMi/DoGE/Skill-It/DML 表达为同一优化问题的特例，发现现有方法失败原因是参数 A_t 估计不准（对角 vs 完整矩阵、静态 vs 时变）；AIOLI 在线方法用交错训练从当前训练历史拟合 A_t，无需额外 run，6/6 设置优于 stratified。
- [Loss-Free Balancing 技术报告](sources/loss-free-balancing.md) - DeepSeek-AI + PKU 的 MoE 负载均衡方法论文（arXiv:2408.15664）：top-K 前加 expert-wise bias 按历史负载 sign 更新，不产生干扰梯度；1B/3B 上 perplexity 与 MaxVio 双赢，并证明 Expert Choice 的未来 token 泄漏。V3/V4、K2 系、MiniMax-M2、MiMo、Ling-2.6 生产采用的 bias 路由一手出处。
- [Jet-Long](sources/jet-long.md) - NVIDIA 的 tuning-free 零样本长上下文扩展：局部窗保留原版 RoPE，远程窗用解析式 $G=\lceil L/w_{\text{pretrained}}\rceil$ 把位置别名回训练网格；Qwen3-1.7B/4B/8B-Base 上 RULER 相对最强基线 +4.79/+2.18/+2.03 pp，fused kernel 相对 FA2 长上下文 prefill 最高 1.39×。
- [WeMM-Embedding 技术报告](sources/wemm-embedding.md) - 微信视觉的通用多模态 embedding：2B/4B/9B 基于 Qwen3.5，两阶段对齐+精炼，MMEB-v2 上 2B 已超此前 8B 开源、9B 达 80.6，已部署视频号/公众号/朋友圈/电商。

## 模型

- [GLM-5](models/glm-5.md) - 744B 总参数 / 40B 激活参数的 MoE 模型，定位在 agentic、reasoning、coding 能力。
- [Macaron-V1](models/macaron-v1.md) - Mind Lab 的 agent-model 家族：Venti 用 GLM-5.2 base、Tall 用 Qwen3.6-35B-A3B base，均以四个按 turn 路由的 LoRA specialist 和 HCP harness 组成。
- [Intern-S2-Mobius](models/intern-s2-mobius.md) - 上海 AI Lab 的 35B 级纯文本架构转换模型：全局共享 FFN knowledge Memory + 多层 Self-Attn Reasoner，主张以 latent iteration 压缩外显 CoT。
- [GLM-5.3](models/glm-5-3.md) - Z.ai agentic coding 发布版；参数和模态未披露，官方称沿用 GLM-5.2 base model，能力增益来自后训练规模化。
- [GLM-5V-Turbo](models/glm-5v-turbo.md) - GLM-5 家族的多模态 agent 基座模型，CogViT + MMTP + 30+ 类别多模态联合 RL，多模态（文本 + 图像 + 视频 + GUI + 文档 + 网页）。
- [MiMo-V2-Flash](models/mimo-v2-flash.md) - 309B 总参数 / 15B 激活参数的 MoE 模型，优化快速推理和 agentic 工作负载。
- [DeepSeek-V4](models/deepseek-v4.md) - 包含 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 的模型族，目标是原生 1M token 上下文。
- [MiniMax-M2 Series](models/minimax-m2-series.md) - 229.9B 总参数 / 9.8B 激活参数的低激活 MoE agentic 模型系列。
- [MiniMax-M3](models/minimax-m3.md) - 428B 总参数 / 22B 激活参数（+ 600M visual encoder）的原生 MSA 多模态 MoE 模型，配套 MSA 报告释出。
- [Kimi K2.5](models/kimi-k2.5.md) - 1.04T 总参数 / 32B 激活参数的 multimodal agentic MoE 模型，强调 Agent Swarm。
- [Kimi Linear](models/kimi-linear.md) - 48B 总参数 / 3B 激活参数的混合线性注意力 MoE 研究模型，KDA:MLA = 3:1，验证线性注意力可 drop-in 替换 full attention。
- [Qwen3-Coder-Next](models/qwen3-coder-next.md) - 79.7B 总参 / ~3B 激活的编码 agent 模型，基于 Qwen3-Next，3 GDN : 1 gated-attention 混合栈（已据 HF config 核实），纯文本。
- [Qwen3.5](models/qwen3.5.md) - Qwen3.5 多模态 Hybrid MoE 家族（397B-A17B 旗舰到 0.8B dense），3 GDN : 1 gated-attention，Qwen3.5-Omni 的架构基座。
- [Qwen3](models/qwen3.md) - Qwen 系基座家族（0.6B–235B-A22B，6 dense + 2 MoE），标准 GQA + 去 QKV-bias + 加 QK-Norm + 无 shared expert MoE，纯文本。后续 Qwen3-Next/3.5/3-Coder-Next/3.5-Omni/Qwen3-VL 的 LLM 前作。
- [Qwen3-VL](models/qwen3-vl.md) - Qwen3-VL 多模态家族（2B/4B/8B/32B dense + 30B-A3B / 235B-A22B MoE），256K context，LLM backbone 用标准 GQA 的 Qwen3，叠 SigLIP-2 + DeepStack + Interleaved MRoPE + 文本时间戳。
- [Gemma 4](models/gemma-4.md) - Google DeepMind 多模态 dense + MoE 家族（E2B/E4B/12B/26B-A4B/31B），原生文本+图像+音频，5:1 SWA/GA + key-as-value + p-RoPE，12B 为 encoder-free 架构。
- [InternVLA-A1.5](models/internvla-a1.5.md) - 上海 AI Lab 统一 VLA 机器人模型，Qwen-3.5 2B（3:1 GDN:full attention）做 backbone + 460M unified expert + latent foresight，GDN 混合注意力在机器人控制领域的采用。
- [HunyuanOCR-1.5](models/hunyuan-ocr-1.5.md) - 腾讯轻量端到端 OCR VLM（1B），DFlash block-diffusion 推测解码 + Agentic Data Flow 数据构造，OmniDocBench v1.6 总分 94.74。
- [KAT-Coder](models/kat-coder.md) - 快手 KwaiKAT 的 agentic coding 模型族（V2 / V2.5），纯文本，Specialize-then-Unify + KwaiEnv + MCLA/Tree Training/asymmetric PPO + MOPD 专家融合，V2.5 PinchBench 94.9 第一。
- [Seed2.0](models/seed2.md) - 字节跳动 Seed 团队多模态模型族（Pro / Lite / Mini），Model Card 不含架构/训练细节，核心是评测框架和部署洞察。
- [VibeThinker-3B](models/vibethinker-3b.md) - 新浪微博 3B dense reasoning 模型，基于 Qwen2.5-Coder-3B，MGPO + Long2Short RL + CLR，纯文本，verifiable reasoning 追平旗舰。
- [LoopCoder-v2](models/loopcoder-v2.md) - 北航 + IQuest 的 7B PLT coder 模型族，weight-tied looped Transformer（14 层共享 block），R=2 最优（SWE-bench Verified 64.4%），纯文本。
- [BDH-CQ](models/bdh-cq.md) - Pathway 的 150M ARC 网格推理系统：示例递归写入 memory、查询以连续 latent workspace 迭代求解；关键模型实现未公开。
- [MiniCPM-o 4.5](models/minicpm-o-4-5.md) - OpenBMB 9B 全双工全模态交互模型，Qwen3-8B backbone + Whisper + speech decoder 端到端可微，Omni-Flow + TAIL，多模态（文本+图像+视频+音频输入；文本+音频输出），端侧 INT4 < 12GB。
- [Keye-VL-2.0](models/keye-vl-2.md) - 快手开源 30B-A3B 多模态 MoE 模型，GQA+DSA 256K 长视频理解 + Cross-Modal MOPD（13 teacher），多模态（文本+图像+视频），基于 Qwen3-30B-A3B-Thinking-2507。
- [JoyAI-VL-Interaction](models/joyai-vl-interaction.md) - JD.com 8B 视觉驱动交互模型，Qwen3-8B + Qwen3-VL ViT + AdaCodec，每秒自主决定说话/静默/委托，多模态（文本+图像+视频），完整可部署系统。
- [MOSS-VL](models/moss-vl.md) - OpenMOSS 11.3B 实时视觉语言模型，Qwen3-8B + Qwen3-VL 视觉编码器，gated cross-attention 让视觉 KV 独立于解码序列，多模态（文本+图像+视频）。
- [Xiaomi-GUI-0](models/xiaomi-gui-0.md) - 小米 SeerRay 的 native end-to-end multimodal GUI agent，基于 Qwen3-VL-30B-A3B-Instruct，真机闭环训练 + error-driven flywheel，多模态（文本+图像输入；文本输出）。
- [UI-Mate](models/ui-mate.md) - 腾讯 HY Frontier 的桌面 foundation GUI agent（9B / 27B），基于 Qwen3.5-9B 与 Qwen3.6-27B，SFT+在线 RL + DemoCUA，多模态（文本+图像输入；文本输出）。
- [Agent-World](models/agent-world.md) - 人大 + ByteDance Seed 的自演化 agent 训练场产出（8B/14B，Qwen3 dense 基座 + 冷启动 SFT + 多环境 GRPO RL + 2 轮自演化 arena），纯文本，环境合成 + agent-environment co-evolution。
- [Qwen-AgentWorld](models/qwen-agent-world.md) - Qwen Team 的 native language world model 家族（35B-A3B / 397B-A17B，基于 Qwen3.5），覆盖 7 域 agentic 环境模拟，三阶段 CPT→SFT→RL，纯文本（GUI 域用 accessibility tree 文本表示）。
- [MinerU2.5-Pro](models/mineru-2-5-pro.md) - 上海 AI Lab + PKU + SJTU + 商汤的 1.2B 轻量文档解析 VLM（NaViT-675M + Qwen2-0.5B），固定架构，Data Engine（DDAS + CMCV + Judge-and-Refine）+ 三阶段训练，多模态（文本+图像输入；结构化文本输出）。
- [GLM-OCR](models/glm-ocr.md) - 智谱 AI + 清华的 0.9B 轻量 OCR VLM（CogViT ~400M + GLM ~500M），MTP 共享参数多头加速（~50% 吞吐提升）+ 两阶段 pipeline + 文档解析/KIE 双任务统一，多模态（文本+图像输入；结构化 Markdown/JSON 输出）。
- [MinerU2.5](models/mineru-2-5.md) - 上海 AI Lab + PKU + SJTU 的 1.2B 解耦文档解析 VLM（NativeRes-ViT 675M + Qwen2-0.5B），coarse-to-fine 两阶段 + Data Engine（IMIC），MinerU2.5-Pro 的基座，多模态（文本+图像输入；结构化 Markdown 输出）。
- [Ling-2.6 / Ring-2.6](models/ling-2.6.md) - Inclusion AI 万亿参数 agentic 模型族（Ling-2.6 instant + Ring-2.6 thinking），7:1 Lightning Attention + MLA 混合线性注意力 retrofit，KPop agentic RL，纯文本。
- [Unlimited OCR](models/unlimited-ocr.md) - Baidu 的 OCR-specialized VLM（基于 DeepSeek OCR），用 R-SWA 保持恒定 KV cache 实现长文档一次性转录，多模态（文本+图像输入；文本输出）。
- [Mach-Mind-4-Flash](models/mach-mind-4-flash.md) - 理想汽车 35B / 3B 激活的 agentic MoE 模型，基于 Qwen3.5-35B-A3B，specialization-then-integration 后训练 + 统一 RL/OPD loss + MOPD 融合 + HMPO token 效率，纯文本。
- [Kimi K3](models/kimi-k3.md) - Moonshot AI 首个开源 3T 级模型（2.78T/104B 激活），Hybrid KDA-MLA（3:1）+ Attention Residuals + Stable LatentMoE（896 routed/16 active）+ MoonViT-V2 原生视觉 + 1M 上下文，多模态（文本+图像+视频）。
- [Laguna](models/laguna.md) - Poolside 的 MoE agentic coding 模型族（M.1 225.8B/23.4B、XS.2 33.4B/3B），Model Factory 工业化流程，3:1 SWA/GA + softplus 门控 + WSD + AutoMixer + CISPO RL，XS.2 Apache 2.0 开源，纯文本。
- [WeMM-Embedding](models/wemm-embedding.md) - 腾讯微信视觉的通用多模态 embedding 家族（2B/4B/9B），基于 Qwen3.5，文本/图像/视频/视觉文档/交错输入，不支持音频，已部署微信推荐与搜索。

## 概念

- [Agentic engineering](concepts/agentic-engineering.md) - 这些报告如何定义长周期软件工程和工具使用任务。
- [高效长上下文注意力](concepts/efficient-long-context-attention.md) - DSA、混合 SWA/GA、CSA 和 HCA 的对比；位置角 OOD 是正交轴，见零样本 RoPE 扩展。
- [Agentic 模型的后训练](concepts/post-training-for-agentic-models.md) - 面向 agent 的 RL、MOPD、蒸馏、VAPO 这类 value-based credit assignment 与 ARPO 这类 step-level rollout 采样模式。
- [多 token 预测](concepts/multi-token-prediction.md) - MTP 作为训练目标和 speculative decoding 机制；含「当 MTP-1 不够：DSpark 接管 V4 生产端」段，解释为什么 V3/V3.2/V4 一直只敢部署 MTP-1。
- [MoE 前沿模型扩展](concepts/moe-frontier-model-scaling.md) - 多篇报告中的总参数、激活参数和系统成本对比。
- [MoE 负载均衡谱系](concepts/moe-load-balancing.md) - 从 auxiliary loss 到 Loss-Free bias 到 Quantile Balancing 的三代方法谱系 + 生产配置地图（V3/V4、K2/K3、MiniMax-M2、MiMo、Ling-2.6、Qwen3、Laguna），及 Expert Choice 因未来 token 泄漏出局的标准论据。
- [Any-to-any 多模态 serving](concepts/any-to-any-multimodal-serving.md) - vLLM-Omni 代表的 omni-modal serving 范式：Thinker / Talker / Vocoder、AR LLM / DiT / encoder 等多阶段模型拆成 stage graph 独立调度与传输。
- [端侧 MoE serving](concepts/edge-native-moe-serving.md) - 消费级机器上的 expert-offload：host 池做 source of truth，GPU LRU 跟踪路由局部性，$q^{\star}$ 按实测 PCIe/主机带宽分流 miss；与 any-to-any stage graph 正交。
- [KV cache 层](concepts/kv-cache-layer.md) - 把 KV cache 从 engine 内部状态提升为可 offload、跨查询复用、跨引擎传输的一等数据；LMCache 是当前开源实现，与模型侧异构 KV 压缩、多模态 stage 传输、端侧 expert offload 正交。

## 细讲模块

- [DeepSeek Sparse Attention](concepts/deepseek-sparse-attention.md) - DSA 的长上下文稀疏选择、GLM-5 中的训练方式和 RL 稳定性问题。
- [Multi-Head Latent Attention](concepts/multi-head-latent-attention.md) - MLA 的「减头 vs 压秩」定位、MHA/MQA 两种 mode，以及 DSA / CSA 为何架在它的 MQA mode 上。
- [异步 Agent RL](concepts/asynchronous-agent-rl.md) - GLM-5 如何用异步 rollout、TITO 和 token-level clipping 训练 agent。
- [Agentic Reinforced Policy Optimization](concepts/agentic-reinforced-policy-optimization.md) - ARPO 如何用工具反馈后的 entropy spike 指导 partial rollout 分叉，并做共享/分叉段 advantage attribution。
- [Multi-Teacher On-Policy Distillation](concepts/multi-teacher-on-policy-distillation.md) - MiMo-V2-Flash 的 MOPD 范式及其与 DeepSeek-V4 OPD 的关系，并含跨家共用的 [OPD 数学依据](concepts/multi-teacher-on-policy-distillation.md#数学依据opd-为什么-work)（reverse-KL mode-seeking+unhackable / on-policy 消除 exposure bias / teacher 固定的良定义优化 / O(1)-vs-O(N) bits/episode / RL 子网络脆弱性 / phase-alternating + 多 teacher 混采的边界）。
- [百万 token 上下文服务](concepts/million-token-context-serving.md) - DeepSeek-V4 的异构 KV-cache、on-disk cache 和 shared-prefix reuse；engine 侧 I/O 见 KV cache 层。
- [Agentic 评测体系](concepts/agentic-evaluation-benchmarks.md) - SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas、UniClawBench 等 benchmark 的作用和可比性风险；含 UniClawBench 的 capability-driven / 三角色闭环差异化定位。
- [Forge Agent-Native RL](concepts/forge-agent-native-rl.md) - MiniMax-M2 如何把 agent harness、RL 训练、长上下文 rollout 和 serving 加速解耦。
- [Agent Swarm](concepts/agent-swarm.md) - Kimi K2.5 的 PARL 并行 agent 编排，以及 context sharding 解释。
- [多模态 Agentic 训练](concepts/multimodal-agentic-training.md) - Kimi K2.5 的 early vision fusion、MoonViT-3D、zero-vision SFT 和 joint multimodal RL。
- [跨层索引复用](concepts/cross-layer-index-reuse.md) - IndexCache、Kascade、HySparse 等如何让多数层共用 anchor 层选好的 top-k 索引。
- [零样本 RoPE 上下文扩展](concepts/zero-shot-rope-context-extension.md) - 不微调、只改位置映射让 RoPE 模型用过训练窗：YaRN / Self-Extend / DCA / Jet-Long 动态分组，以及 Kimi 系 NoPE 旁路。
- [线性注意力与 delta rule](concepts/linear-attention-and-delta-rule.md) - 朴素线性注意力 → DeltaNet → GDN → KDA 的演进，遗忘门 + delta rule 如何把线性注意力质量追回 softmax。
- [注意力门控](concepts/attention-gating.md) - softmax 注意力里加门（Gated Attention 的 SDPA 输出门、KDA 的输出门）：非线性补偿 + 消除 attention sink。
- [数据混合优化](concepts/data-mixture-optimization.md) - LLM 数据混合优化方法谱系：预训练 domain reweighting（DoReMi/DoGE/RegMix/TANDEM/AutoMixer，用小 proxy model 预测大模型权重）+ SFT 阶段在线无 proxy 分支（DynamixSFT，Multi-Armed Bandit）。
- [Looped Transformers](concepts/looped-transformers.md) - 权重共享的循环 Transformer：用同一 block 反复执行增加有效深度。PLT 通过 CLP + shared-KV G-SWA 使延迟和 KV-cache 不随 loop count 增长；LoopCoder-v2 发现 R=2 饱和（gain–cost 剪刀：refinement gain 递减 + CLP offset cost 恒定）。
- [Agent 记忆生命周期](concepts/agent-memory-lifecycle.md) - Personal AI 记忆从静态存储到全生命周期可审计基础设施：Structure / Expansion / Evolution / Deployment 四角色 + 共享审计契约（typed evidence / diagnostic traces / strategy artifacts / gate-rollback）。
- [Attention Residuals](concepts/attention-residuals.md) - Kimi K3 的深度维信息流机制：每层选择性从所有前层检索表示（沿深度做 attention），解除标准残差的 RNN 瓶颈；Block AttnRes（N=8）降开销到 O(Nd)。
- [Stable LatentMoE](concepts/stable-latentmoe.md) - Kimi K3 的宽度维机制：LatentMoE（routed 在 latent 空间）+ Normalized（RMSNorm）+ SiTU-GLU（bounded activation）+ Quantile Balancing（aux-loss-free 的 exact 对偶 LP 解），支撑 896-expert/16-active 极端稀疏在 2.8T 规模稳定训练。

## 比较

- [2026 前沿模型技术报告对比](comparisons/2026-open-model-technical-reports.md) - GLM-5、MiMo-V2-Flash、DeepSeek-V4、MiniMax-M2 和 Kimi K2.5 的横向比较。
- [稀疏注意力机制对比](comparisons/sparse-attention-mechanisms.md) - DSA、MSA、NSA、MoBA、CSA/HCA、IndexCache 等沿"粒度 / 跨头共享 / 跨层共享"三轴的对比。
- [On-Policy Distillation 跨报告对比](comparisons/on-policy-distillation.md) - MiMo MOPD / DeepSeek-V4 OPD / Qwen3 Strong-to-Weak / Qwen3-VL Strong-to-Weak / GLM-5 cross-stage 的"目的 / KL 形式 / pipeline 位置"三轴对比，附 Qwen3-8B Table 21 OPD vs RL 对照。
- [LLM RL policy optimization 对比](comparisons/llm-rl-policy-optimization.md) - VAPO / DAPO / GSPO / SAPO / ARPO 等方法的抽象层级对比：value-based credit assignment、GRPO recipe、sequence-level ratio、soft trust region、agentic partial rollout。
