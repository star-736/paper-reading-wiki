---
type: Concept
title: "Multi-Head Latent Attention (MLA)"
description: "MLA 的「减头 vs 压秩」定位、MHA/MQA 两种 mode，以及 DSA / CSA 为何架在它的 MQA mode 上。"
tags: ["concept", "multi-head-latent-attention"]
timestamp: 2026-06-20
---

# Multi-Head Latent Attention (MLA)

## 定义

MLA 是 DeepSeek 在 DeepSeek-V2（2024）提出的注意力变体，目标和 GQA / MQA 一样是压缩 KV cache，但走的是**正交的另一条轴**：

- **MHA → GQA → MQA 这条轴：减 KV head 数**。MHA 每个 query head 配一份独立 KV；GQA 让若干 head 共享一组 KV；MQA 所有 head 共享 1 组 KV。head 数越减，KV cache 越小，但 KV 表达力也越弱。
- **MLA 这条轴：低秩压缩**。不动 head 数，而是把每个 token 的 KV 联合**下投影成一个低维 latent 向量** $c^{KV}$ 存进 cache；用时再通过各 head 的**上投影**矩阵恢复出 per-head 的 K、V。完整缓存是 latent 加共享 RoPE key；是否小于 GQA/MQA 要看具体维度，不能笼统排序。原文两组 MoE 对照支持其总体质量接近或超过 MHA，非所有单项都领先。

![DeepSeek-V2 Figure 3：MHA / GQA / MQA / MLA 四种注意力的对比。前三种沿「减 KV head 数」一条轴（MHA 每 head 一份 KV → GQA 分组共享 → MQA 全共享一份），推理时缓存的都是显式 K/V（图中斜线填充 = Cached During Inference）。MLA 走正交的另一条轴：把 K/V 联合压缩成一个 Compressed Latent KV，简图用压缩 latent 表示缓存（完整配置另有 decoupled RoPE key），用时再经 projection 上投影回 per-head K/V。](../assets/deepseek-v2/fig3-mha-gqa-mqa-mla.png)

> Figure 3（原文截图，§ 2.1）：MHA、GQA、MQA、MLA 的简化对比。斜线填充表示推理时需缓存的部分——图中省略了 decoupled RoPE；完整缓存包含压缩 latent 与共享 RoPE key。

**与 MQA 的关系——近亲但不同源**。MLA 在推理时可以做「矩阵吸收」：把上投影 $W_{UK}$ 吸进 query 投影、$W_{UV}$ 吸进输出投影，于是不必显式还原 per-head K/V，直接拿所有 query head 对那一个共享 latent 做注意力——计算形态就**退化成 MQA**。所以若在「GQA vs MQA」里硬找 MLA 的近亲，是 **MQA**：两者都让所有 query head 共享一份 KV；区别是纯 MQA 共享的是**原始 KV**（丢表达力），MLA 共享的是**压缩 latent** 再 per-head 上投影（带低秩补偿）。可以理解为「带低秩补偿的 MQA」。

矩阵吸收带来的「两种 mode」（MHA mode / MQA mode）以及「MQA mode」一词在 V3.2 里横跨两条语义轴的歧义，是本页最易混淆处，单列在[下文「两种 mode 与『MQA mode』歧义」](#两种-mode-与mqa-mode歧义)详述。

## 机制细节（已据 DeepSeek-V2 原文核实）

来源：[DeepSeek-V2 技术报告](../sources/deepseek-v2.md)（arXiv:2405.04434，MLA 首次提出处）。

- **低秩 KV 联合压缩**：把 hidden $h_t$ 下投影成压缩 latent $c^{KV}_t$（维度 $d_c \ll n_h d_h$），完整缓存还需 decoupled RoPE key；用时上投影 $W^{UK}/W^{UV}$ 恢复 per-head K/V。
- **矩阵吸收**：推理时 $W^{UK}$ 吸进 $W^Q$、$W^{UV}$ 吸进 $W^O$，无需显式还原 per-head K/V，直接对 latent 算——这就是「MQA mode」的数学来源。
- **query 也低秩压缩，但目的与 KV 完全不同——省的是「训练激活显存」，不是 cache，也不主要是 compute**：hidden 先下投影成 query latent $c^Q_t$（V2 中 $d_c'=1536$），再上投影回 per-head query。原文 §2.1.2 明说这是「为减少训练时的 activation memory，即便它压不了 KV cache」（"in order to reduce the activation memory during training, we also perform low-rank compression for the queries, **even if it cannot reduce the KV cache**"）。
  - **激活显存 ≠ 计算量**：activation memory 是训练时为反向传播必须**常驻显存的前向中间张量**，是和权重显存、KV cache 并列的第三块开销；低秩分解顺带也让这条路 FLOPs 小一点，但那不是论文卖点。
  - **为什么 query 在 V2 里特别吃显存**：128 head × 128 = **16384 维**的展开 query，比 hidden（5120）还大 3.2×；这是内容 query 的维度；还需加每头 64 维的位置 query。哪些张量保留到 backward、哪些重算，取决于实现，原文未展开。
  - **配置事实**：V2-Lite 不压缩 query（附录 B.1）；原文未给压缩与否的配对消融，不能推出它对质量没有影响。
  - **「具体怎么省的」论文没展开**——它只给「压缩 → 省显存」这个结论，中间机制留白。一个合理推测见[待追问](#待追问)，但未坐实。
- **Decoupled RoPE**：RoPE 与低秩压缩不兼容（位置敏感的 RoPE 矩阵会卡在 $W^{UK}$ 中间、破坏吸收）。解法是额外引入 multi-head decoupled query $q^R$ + 一个**全 head 共享的 decoupled key** $k^R$ 专门承载 RoPE，K/Q = 压缩部分（可吸收）+ decoupled 部分（带 RoPE）拼接；decoupled key 也进 cache，故每 token KV cache = $(d_c + d_R)$ 元素。
- **KV cache 等效 GQA-2.25 组**：Table 1 给出 $(d_c + d_R) \approx \tfrac{9}{2} d_h$，按缓存元素数等效于 2.25 组 GQA；这不意味着比 MQA 更小。质量证据见 [Table 9 的完整对照与边界](../sources/deepseek-v2.md#mla-与-mha-的直接对照附录-d2--table-9)：小模型有一项落后。附录 D.1 用 7B dense 消融证明 MHA 显著优于 GQA/MQA，正是 MLA「压低秩而非减头数」这条轴的动机。

  附录 D.1 消融（Table 8，7B dense 模型，1.33T tokens，仅注意力机制不同、参数对齐 ~7B）：

  | Benchmark (Metric) | # Shots | Dense 7B w/ MQA (7.1B) | Dense 7B w/ GQA-8 (6.9B) | Dense 7B w/ MHA (6.9B) |
  | --- | --- | --- | --- | --- |
  | BBH (EM) | 3-shot | 33.2 | 35.6 | **37.0** |
  | MMLU (Acc.) | 5-shot | 37.9 | 41.2 | **45.2** |
  | C-Eval (Acc.) | 5-shot | 30.0 | 37.7 | **42.9** |
  | CMMLU (Acc.) | 5-shot | 34.6 | 38.4 | **43.5** |

  > Table 8（原文附录 D.1）：MHA 在四项 hard benchmark 上全面优于 GQA 和 MQA——MQA 虽参数最多（7.1B vs 6.9B）却分数最低。这说明「减 KV head 数」有显著质量代价，MLA 因此选择正交的「压低秩」轴来压缩 cache。

  Table 1（原文 § 2.1.4 数据重排为 Markdown；$n_h$=head 数，$d_h$=每 head 维度，$l$=层数，$n_g$=GQA 组数，$d_c/d_R$=KV 压缩与 decoupled 维度）：

  | 注意力机制 | 每 token KV cache（元素数） | 能力 |
  | --- | --- | --- |
  | MHA | $2 n_h d_h l$ | Strong |
  | GQA | $2 n_g d_h l$ | Moderate |
  | MQA | $2 d_h l$ | Weak |
  | **MLA（本文）** | $(d_c + d_R) l \approx \tfrac{9}{2} d_h l$ | **Stronger** |

- **效率数字**：KV cache −93.3%，最大生成吞吐 5.76×（vs DeepSeek 67B MHA）。

## 投影矩阵与缓存形状

**已据原文核实（`supported`）**：DeepSeek-V2 §2.1.2–2.1.3、§3.1.2 与附录 C。下表采用论文的列向量约定，矩阵形状为“输出维度 × 输入维度”；数字是把原文配置代入符号形状所得，不是从 checkpoint 实现猜测。配置为 $d=5120$、$n_h=128$、$d_h=128$、$d_c=512$、$d_c'=1536$、$d_R=64$（本页 $d_R$ 对应论文 $d_h^R$）。

| 矩阵 | 符号形状 | V2 数值形状 | 作用 |
| --- | --- | --- | --- |
| $W^{DQ}$ | $d_c'\times d$ | $1536\times5120$ | hidden → query latent |
| $W^{UQ}$ | $(n_h d_h)\times d_c'$ | $16384\times1536$ | query latent → 各头内容 query |
| $W^{QR}$ | $(n_h d_R)\times d_c'$ | $8192\times1536$ | query latent → 各头位置 query，再施 RoPE |
| $W^{DKV}$ | $d_c\times d$ | $512\times5120$ | hidden → 共享 KV latent |
| $W^{UK}$ | $(n_h d_h)\times d_c$ | $16384\times512$ | KV latent → 各头内容 key |
| $W^{UV}$ | $(n_h d_h)\times d_c$ | $16384\times512$ | KV latent → 各头 value |
| $W^{KR}$ | $d_R\times d$ | $64\times5120$ | hidden → 一份共享位置 key，再施 RoPE |
| $W^O$ | $d\times(n_h d_h)$ | $5120\times16384$ | 拼接各头输出 → hidden |

展开时每头 query / key 都是 $128+64=192$ 维，value 与每头 attention 输出为 128 维。注意两个位置分支的输入不同：$W^{QR}$ 读 $c_t^Q$，$W^{KR}$ 直接读 $h_t$。§3.1.2 还说明压缩 latent 后有额外 RMSNorm；这里列线性映射的维度，不等于完整实现代码。

每层每 token 的缓存仅保留 $c_t^{KV}$ 的 512 个元素与 $k_t^R$ 的 64 个元素，共 **576** 个；不是给 K、V 各存一份 512，也不是给 128 个头各存一份 RoPE key。60 层共 34,560 个元素；仅按 FP16/BF16 两字节存储换算为 69,120 bytes/token，不含页表、对齐等引擎开销。Table 1 按元素数比较，不绑定存储精度。

**本页代数整理**：令 $W_i^{UK},W_i^{UQ},W_i^{UV}$ 为第 $i$ 个头对应的行块，$W_i^O$ 为输出矩阵对应的列块，则内容分支有

$$q_{t,i}^{A}=(W_i^{UK})^\top W_i^{UQ}c_t^Q,\qquad (W_i^{UK})^\top W_i^{UQ}\in\mathbb R^{512\times1536}.$$

因此可直接与历史 $c_j^{KV}$ 点积，不必展开历史内容 key；位置分支仍单独保留。value 侧有

$$W_i^O\sum_j a_{t,j,i}W_i^{UV}c_j^{KV}=(W_i^O W_i^{UV})\sum_j a_{t,j,i}c_j^{KV},\qquad W_i^O W_i^{UV}\in\mathbb R^{5120\times512}.$$

两式是附录 C 所述结合律吸收的逐头写法，$a_{t,j,i}$ 仍由完整内容与位置分数的 softmax 得到。它们说明吸收改变了 query / 输出侧的投影形状，不能把这两侧成本视为完全相同。代数重排本身也不禁止反向传播；把固定权重预合并，是另一层推理实现选择。

V2-Lite 的区别（附录 B.1）：$d=2048$、27 层、16 个头，每头内容维 128、位置维 64，KV latent 仍为 512；不做 query 低秩压缩。因此不能把 V2 的 $W^{DQ}/W^{UQ}$ 数值表照搬到 Lite。

## 两种 mode 与「MQA mode」歧义

矩阵吸收让同一套 MLA 权重有了两种等价算法形态，DeepSeek-V3.2 附录 A 的 Figure 7 把它们命名为 **MHA mode** 和 **MQA mode**：

![DeepSeek-V3.2 Figure 7：MLA 的两种等价算法形态。左 (a) MHA mode——latent c_KV 经 W^UK / W^UV 按 head 展开成每个 head 的 K/V，进 Multi-Head Attention core；右 (b) MQA mode——W^UK 吸收到 query 侧（q^A = W^UK q^C）、W^UV 移到 attention 输出之后（o = W^UV o^C），latent c_KV 作共享 KV 进 Multi-Query Attention core。两处 apply RoPE 只作用在带 R 上标的位置分量 q^R / k^R 上。](../assets/deepseek-v32/fig7-mha-mqa-mode.png)

> Figure 7（原文截图，§ A. MHA and MQA Modes of MLA）：MLA 有 **MHA mode** 和 **MQA mode**，两者之间可相互变换。DeepSeek-V3.1-Terminus **训练 / prefill 用 MHA mode，decode 用 MQA mode**。

**「MQA mode」一词在 V3.2 里横跨两条轴，别混**。它被用在两处不同的事上，是「DSA 到底是不是在 MQA mode 上训练」这类困惑的根源：

- **轴一 · compute form**：MHA mode 与 MQA mode 是**同一注意力的两种等价算法形态**（矩阵吸收，可互相变换）。分工是「训练 / prefill 用 MHA mode、decode 用 MQA mode」——但 Figure 7 的 caption 把这句**限定在 DeepSeek-V3.1-Terminus（dense 基座）**："For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding." 正文进一步说 Figure 7 展示的是 MLA 的「two aspects … as well as the transformation between them」。
  - **适用范围**：V2 附录 C 给出推理期矩阵吸收，V3.2 Figure 7 caption 明确给出 V3.1-Terminus 的训练 / prefill / decode 分工。两者支持这些具体配置的实现说明，不能据此断言所有 MLA 系统都必须如此，也不能用“吸收后无法求梯度”解释训练选择。
- **轴二 · selection 结构**：§「Instantiate DSA Under MLA」里的「we implement DSA based on the MQA mode of MLA」指的是 **KV entry（latent）跨所有 query head 共享**——原文 "each latent vector (the key-value entry of MLA) will be **shared across all query heads of the query token**"，故 DSA 的 top-k **所有 head 选同一组 token**。原文的因果是**先给 kernel 约束再推出 MQA mode**："at the kernel level, each key-value entry **must be shared across multiple queries** for computational efficiency" → "**Therefore**, we implement DSA based on the MQA … mode of MLA"。这是**共享 / 选择结构**的陈述，**不是**「训练前向用 MQA 算术」。
- **两条轴正交的铁证**：DSA 下，短上下文 prefill 仍**「specially implement a masked MHA mode to simulate DSA」**（效率讨论一节）——compute form 是 MHA、selection 是 DSA，同时成立。所以「训练/prefill 走 MHA 形态」与「DSA 基于 MQA mode」并不矛盾，前者是算术形态、后者是 latent 跨头共享。

> **交叉检验结论**：上述三处引述均已回 [DeepSeek-V3.2 原文](../sources/deepseek-v32.md) 逐句核对，准确无误；「两条轴」这个切分本身建立在 selection（§ Instantiate DSA Under MLA）、compute form（Figure 7 caption）、masked MHA（效率讨论）三处不同语境之上，是**论文支撑的合理拆分而非臆测**。需注意：中文技术博客层面**未见有人这样明确区分两条轴**（多数文章只讲「MLA 矩阵吸收→MQA」这条 compute-form 轴），故这一拆分的**外部佐证仅到论文一手原文为止**，是本页的原创综合。

**V3.2 长上下文训练到底展开还是吸收？** 当前所引原文没有明确给出训练 kernel 的算术形态。Figure 7 的分工针对 V3.1-Terminus；§2.3 的 masked MHA 针对 V3.2 短序列 prefill。不能把这两句拼成“V3.2 所有训练都展开”的确证。

## 展开与吸收的计算成本边界

**本页综合**：完整投影形状表明，两条路径除 KV 展开与注意力点积外，query / 输出投影的成本也会变化。只数 KV 上投影、QK、AV 三项得到的“约 341 token”不能作为可信的实现切换阈值，更不能据它证明 V3.2 短 prefill 选择 masked MHA 的原因。

prefill 的 query 长度与历史 cache 长度、decode 的单个新 query 与长 cache，是不同工况，不能把 decode 简单替成整段序列长度 $L=1$。判断哪种实现更快，还需计入全部投影、访存、批大小和 kernel 效率。V3.2 §2.3 确认其短序列 prefill 用 masked MHA 更高效，但没有给出上述 341 阈值或逐项成本归因；本页不作“吸收恒省算力”的断言。

## 跨报告信号

- **[DeepSeek-V2](../sources/deepseek-v2.md)（起源，2024）**：MLA 首次提出。236B/21B MoE，整体相对 DeepSeek 67B 的 KV cache 降 93.3%、最大生成吞吐提至 5.76×（非 MLA 单组件消融），并用附录 D.1 消融论证「减头数」（GQA/MQA）有质量代价、故走「压低秩」这条正交轴。
- **[DeepSeek-V3.2](../sources/deepseek-v32.md)**：把 [DSA](deepseek-sparse-attention.md) **实例化在 MLA 的 MQA mode 上**——latent vector（即 MLA 的 KV entry）被同一 query token 的所有 query head 共享，DSA 的 top-k 就在这些 latent entry 上选。原文："we implement DSA based on the MQA (Shazeer, 2019) mode of MLA"。这里的「MQA mode」是**轴二 · selection 结构**（见上文「两种 mode 与『MQA mode』歧义」），**不是说训练前向用 MQA 算术**——事实上短上下文 prefill 仍用「masked MHA mode to simulate DSA」。
- **[DeepSeek-V4](../sources/deepseek-v4.md)**：CSA 进一步在 MLA 血统上叠压缩——core attention 是 **Shared-KV MQA**，query 仍由**压缩 latent 向量上投影**得到（且该 latent 与 indexer query 共享）。即 MLA 的 latent-query 结构被 CSA 继承，再在压缩 KV entry 上做 MQA。
- **[GLM-5](../models/glm-5.md)**：同样在 MLA-DSA 基座上做长上下文稀疏，RL 阶段冻结 indexer + deterministic top-k（与 V3.2「post-training 继续训练 indexer」是两条路线）。
- **[Kimi Linear](../sources/kimi-linear.md)（KDA，2025，非 DeepSeek 系）**：换了个用法——不是「在 MLA 上加稀疏选择」，而是把 MLA **稀释成 1/4 的全局层**（3 个 [KDA 线性注意力](linear-attention-and-delta-rule.md) 层配 1 个 Full MLA 层），其余层换成线性注意力。且这 1/4 的 MLA 层用 **NoPE**，推理时退化成纯 MQA。这说明 MLA 不仅是稀疏注意力的底座，也能当混合线性注意力里那一小撮「负责全局信息流」的全局层。
- **[Kimi K3](../sources/kimi-k3.md)（Gated MLA，2026-07，首个开源 3T 级）**：继承 Kimi Linear 的 3:1 KDA:MLA 混合，但把 MLA 层升级为 **Gated MLA**——三处改动：(1) **NoPE** 沿用（位置编码全靠 KDA 隐式编码，扩展上下文长度零修改，1M 外推无需 retune RoPE base 或 YaRN）；(2) **Full-rank output gate**（`y = W_o[Sigmoid(W_g x) ⊙ ~o]`，与 KDA 同款全秩门，让每 token 调制从全局注意力读出的通道，见 [注意力门控](attention-gating.md)）；(3) **FP32 attention output** 纠正 flash attention 的 biased rounding error（代价是 on-chip footprint 翻倍，故重设 kernel 把它与 KV staging buffer 重叠）。K3 的 Gated MLA 配 KDA 的混合栈，是 MLA 在 3T 级 frontier 模型里作「全局层」角色的最新证据——且 NoPE 路线（vs DeepSeek 系的 Decoupled RoPE）让长上下文扩展摆脱位置编码 retuning 负担。
- **横向**：见 [稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)——DSA、CSA 的「底层」栏都落在 MLA / MLA-MQA 上，而 MSA 走的是 GQA 底座、NSA 走 GQA、[MoBA](../sources/moba.md) 的门是每 head 独立（1M 续训碰巧从 Llama 3.1 8B / GQA 出发，不是 MoBA 自己发明的头结构）。构成「减头 vs 压秩」两种起点的分野。

## 为什么重要

- **它定义了 DeepSeek 系稀疏注意力的起点**。DSA、CSA 都不是凭空设计的注意力，而是「在 MLA 的 latent 共享结构上再加稀疏选择」。理解 MLA 的 MQA mode（所有 head 共享一个 latent KV）就理解了为什么 DSA 的 top-k「所有 query head 共享一个集合」是自然结果——不是设计取舍，而是 MLA-MQA mode 本来就只有一份 KV 可选。
- **它把「KV cache 压缩」从减头数解放成压低秩**。这条轴使缓存大小由 latent 与位置分支决定，而不直接随 KV head 数增长；其质量与缓存折中需看具体配置，是百万 token 上下文路线（DeepSeek-V4）成立的前提之一。
- **两种 mode 的存在揭示了训练/推理的非对称**。V3.1-Terminus 的训练/prefill 用 MHA mode 展开，decode 用 MQA mode 吸收；其他配置需分别核对实现。同一套权重两种算法形态，是 MLA 工程上最容易被忽略但最关键的点。

## 待追问

- **需实验或作者披露**：**query 压缩「具体怎么省激活显存」论文未展开，待坐实**。一个合理推测：1536 维 latent 当「细腰」checkpoint，只常驻它、backward 时从它重算出 16384 维大 query（~10× 节省）。依据是 V2 训练框架确实用了 recomputation（§ 训练："a portion of the operators are recomputed to save activation memory"），但这句是**块级通用技巧**，论文**没把它专门挂到 query 压缩上**——所以「存细腰+重算 query」是推断，非原文机制。且实际工程中 gradient checkpointing 多在 transformer block 级别整体包，未必是 query 专属。求证难点：博客讲解多为同款二手推断（会循环），官方 modeling 代码大概率只见块级重算；要硬坐实需 DeepSeek 自己的训练代码或作者澄清。
- **现有材料待核**：DeepSeek-V2 → V3 → V3.2（DSA）→ V4（CSA）这条演进链上「每一步在 MLA 上加了什么」，值得做一张演进表（V3 的 MLA 改动、V3.2 加 DSA、V4 加 token 压缩 + Shared-KV MQA）。
- **现有材料待核**：MSA 选 GQA 而非 MLA 作底座，是 MiniMax 与 DeepSeek 的路线分歧还是有明确的效率/质量权衡论证？需要对比两篇论文的动机段。

## 已核实的配套对照

[MLA 与 MHA 的附录 D.2 / Table 9](../sources/deepseek-v2.md#mla-与-mha-的直接对照附录-d2--table-9)：两组 MoE、八个分数中七个提升；小模型 C-Eval 低 0.7 分。

## 相关页面

- 来源：[DeepSeek-V2](../sources/deepseek-v2.md)（MLA 首次提出）、[DeepSeek-V3.2](../sources/deepseek-v32.md)、[DeepSeek-V4](../sources/deepseek-v4.md)、[Kimi K3 技术报告](../sources/kimi-k3.md)（Gated MLA：NoPE + full-rank output gate + FP32 attention output）
- [DeepSeek Sparse Attention](deepseek-sparse-attention.md)
- [线性注意力与 delta rule](linear-attention-and-delta-rule.md)（Kimi Linear 用 MLA 当 1/4 全局层）
- [高效长上下文注意力](efficient-long-context-attention.md)
- [稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)
- 模型：[GLM-5](../models/glm-5.md)、[DeepSeek-V4](../models/deepseek-v4.md)

关联提问页：[DeepSeek-V2 技术报告](../sources/deepseek-v2.md#相关追问)。
