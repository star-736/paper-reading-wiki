# Multi-Head Latent Attention (MLA)

## 定义

MLA 是 DeepSeek 在 DeepSeek-V2（2024）提出的注意力变体，目标和 GQA / MQA 一样是压缩 KV cache，但走的是**正交的另一条轴**：

- **MHA → GQA → MQA 这条轴：减 KV head 数**。MHA 每个 query head 配一份独立 KV；GQA 让若干 head 共享一组 KV；MQA 所有 head 共享 1 组 KV。head 数越减，KV cache 越小，但 KV 表达力也越弱。
- **MLA 这条轴：低秩压缩**。不动 head 数，而是把每个 token 的 KV 联合**下投影成一个低维 latent 向量** $c^{KV}$ 存进 cache；用时再通过各 head 的**上投影**矩阵恢复出 per-head 的 K、V。cache 里只存 latent，所以比 GQA/MQA 还小，但因为上投影是 per-head 的，表达力能追回 MHA 级别。

**与 MQA 的关系——近亲但不同源**。MLA 在推理时可以做「矩阵吸收」：把上投影 $W_{UK}$ 吸进 query 投影、$W_{UV}$ 吸进输出投影，于是不必显式还原 per-head K/V，直接拿所有 query head 对那一个共享 latent 做注意力——计算形态就**退化成 MQA**。这就是 DeepSeek-V3.2 附录 A 所说的 MLA 的「两种 mode」：

> Figure 7：MLA 有 **MHA mode** 和 **MQA mode**，两者之间可相互变换。DeepSeek-V3.1-Terminus **训练 / prefill 用 MHA mode，decode 用 MQA mode**。

所以若在「GQA vs MQA」里硬找 MLA 的近亲，是 **MQA**：两者都让所有 query head 共享一份 KV；区别是纯 MQA 共享的是**原始 KV**（丢表达力），MLA 共享的是**压缩 latent** 再 per-head 上投影（带低秩补偿）。可以理解为「带低秩补偿的 MQA」。

**「MQA mode」一词横跨两条轴，别混（已据 V3.2 原文核实）**。V3.2 里这个词被用在两处不同的事上，是「DSA 到底是不是在 MQA mode 上训练」这类困惑的根源：

- **轴一 · compute form**：MHA mode 与 MQA mode 是**同一注意力的两种等价算法形态**（矩阵吸收，可互相变换）。分工是「训练 / prefill 用 MHA mode、decode 用 MQA mode」——但 Figure 7 的 caption 把这句**限定在 DeepSeek-V3.1-Terminus（dense 基座）**。
- **轴二 · selection 结构**：「we implement DSA based on the MQA mode of MLA」（§ Instantiate DSA Under MLA）指的是 **KV entry（latent）跨所有 query head 共享**，故 DSA 的 top-k **所有 head 选同一组 token**；理由是 kernel 效率——「each key-value entry **must be shared across multiple queries**」。这是**共享 / 选择结构**的陈述，**不是**「训练前向用 MQA 算术」。
- **两条轴正交的铁证**：DSA 下，短上下文 prefill 仍**「specially implement a masked MHA mode to simulate DSA」**（§ 效率讨论）——compute form 是 MHA、selection 是 DSA，同时成立。所以「训练/prefill 走 MHA 形态」与「DSA 基于 MQA mode」并不矛盾，前者是算术形态、后者是 latent 跨头共享。

**落到最具体一问——「DSA 训练时那份 latent 是展开还是吸收？」**：

| 轴 | 训练 | decode |
| --- | --- | --- |
| compute form（展开 / 吸收） | **展开**（MHA mode：那份共享 latent 被 $W^{UK}/W^{UV}$ 上投影成 128 份 per-head K/V 再算） | **吸收**（MQA mode：不展开，吸收矩阵后直接打 latent） |
| selection 结构（top-k 跨头） | **共享**（一份 latent / 一组 token） | **共享**（同左，架构写死） |

即：**「选哪些 token」恒共享（一份 latent）；「latent 怎么参与计算」训练展开、decode 吸收**——选择共享和计算展开是两件事，可同时成立。训练展开有两个动因：(1) compute-bound 下展开后每对点积 128 维 < 吸收的 512 维；(2) 训练要让 $W^{UQ}/W^{UK}/W^{UV}/W^O$ 各自 live、梯度分别回流，而吸收是推理期预合并固定权重的优化。

（仍留白：论文没有一句直说 V3.2 **长上下文**训练前向用哪种算术形态——长上下文走 DSA 稀疏路径，「展开 / 吸收」此时的取舍论文未展开，要坐实需翻开源 inference/训练代码。短上下文 / prefill 用 MHA 展开形态有原文（Figure 7 caption + masked MHA mode）确证。）

## 机制细节（已据 DeepSeek-V2 原文核实）

来源：[DeepSeek-V2 技术报告](../sources/deepseek-v2.md)（arXiv:2405.04434，MLA 首次提出处）。

- **低秩 KV 联合压缩**：把 hidden $h_t$ 下投影成压缩 latent $c^{KV}_t$（维度 $d_c \ll n_h d_h$），**cache 只存 latent**；用时上投影 $W^{UK}/W^{UV}$ 恢复 per-head K/V。
- **矩阵吸收**：推理时 $W^{UK}$ 吸进 $W^Q$、$W^{UV}$ 吸进 $W^O$，无需显式还原 per-head K/V，直接对 latent 算——这就是「MQA mode」的数学来源。
- **query 也低秩压缩，但目的与 KV 完全不同——省的是「训练激活显存」，不是 cache，也不主要是 compute**：hidden 先下投影成 query latent $c^Q_t$（V2 中 $d_c'=1536$），再上投影回 per-head query。原文 §2.1.2 明说这是「为减少训练时的 activation memory，即便它压不了 KV cache」（"in order to reduce the activation memory during training, we also perform low-rank compression for the queries, **even if it cannot reduce the KV cache**"）。
  - **激活显存 ≠ 计算量**：activation memory 是训练时为反向传播必须**常驻显存的前向中间张量**，是和权重显存、KV cache 并列的第三块开销；低秩分解顺带也让这条路 FLOPs 小一点，但那不是论文卖点。
  - **为什么 query 在 V2 里特别吃显存**：128 head × 128 = **16384 维**的展开 query，比 hidden（5120）还大 3.2×；这一大坨中间张量训练时要一直留到 backward。
  - **旁证**：V2-Lite 干脆**不压缩 query**（原文 "it does not compress the queries"，小模型激活显存压力小），反证 query 压缩纯为显存、与能力/cache 都无关。
  - **「具体怎么省的」论文没展开**——它只给「压缩 → 省显存」这个结论，中间机制留白。一个合理推测见[待追问](#待追问)，但未坐实。
- **Decoupled RoPE**：RoPE 与低秩压缩不兼容（位置敏感的 RoPE 矩阵会卡在 $W^{UK}$ 中间、破坏吸收）。解法是额外引入 multi-head decoupled query $q^R$ + 一个**全 head 共享的 decoupled key** $k^R$ 专门承载 RoPE，K/Q = 压缩部分（可吸收）+ decoupled 部分（带 RoPE）拼接；decoupled key 也进 cache，故每 token KV cache = $(d_c + d_R)$ 元素。
- **KV cache 等效 GQA-2.25 组**：Table 1 给出 $(d_c + d_R) \approx \tfrac{9}{2} d_h$，等于只有 2.25 组的 GQA，但性能 **> MHA**。附录 D.1 用 7B dense 消融证明 MHA 显著优于 GQA/MQA，正是 MLA「压低秩而非减头数」这条轴的动机。
- **效率数字**：KV cache −93.3%，最大生成吞吐 5.76×（vs DeepSeek 67B MHA）。

## 跨报告信号

- **[DeepSeek-V2](../sources/deepseek-v2.md)（起源，2024）**：MLA 首次提出。236B/21B MoE，靠 MLA 把 KV cache 降 93.3%、吞吐提 5.76×，并用附录 D.1 消融论证「减头数」（GQA/MQA）有质量代价、故走「压低秩」这条正交轴。
- **[DeepSeek-V3.2](../sources/deepseek-v32.md)**：把 [DSA](deepseek-sparse-attention.md) **实例化在 MLA 的 MQA mode 上**——latent vector（即 MLA 的 KV entry）被同一 query token 的所有 query head 共享，DSA 的 top-k 就在这些 latent entry 上选。原文："we implement DSA based on the MQA (Shazeer, 2019) mode of MLA"。这里的「MQA mode」是**轴二 · selection 结构**（KV 跨头共享、kernel 效率要求"each key-value entry must be shared across multiple queries"），**不是说训练前向用 MQA 算术**——事实上短上下文 prefill 仍用「masked MHA mode to simulate DSA」（见上文「两条轴」）。
- **[DeepSeek-V4](../sources/deepseek-v4.md)**：CSA 进一步在 MLA 血统上叠压缩——core attention 是 **Shared-KV MQA**，query 仍由**压缩 latent 向量上投影**得到（且该 latent 与 indexer query 共享）。即 MLA 的 latent-query 结构被 CSA 继承，再在压缩 KV entry 上做 MQA。
- **[GLM-5](../models/glm-5.md)**：同样在 MLA-DSA 基座上做长上下文稀疏，RL 阶段冻结 indexer + deterministic top-k（与 V3.2「post-training 继续训练 indexer」是两条路线）。
- **横向**：见 [稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)——DSA、CSA 的「底层」栏都落在 MLA / MLA-MQA 上，而 MSA 走的是 GQA 底座、NSA/MoBA 走 MQA/GQA 底座，构成「减头 vs 压秩」两种起点的分野。

## 为什么重要

- **它定义了 DeepSeek 系稀疏注意力的起点**。DSA、CSA 都不是凭空设计的注意力，而是「在 MLA 的 latent 共享结构上再加稀疏选择」。理解 MLA 的 MQA mode（所有 head 共享一个 latent KV）就理解了为什么 DSA 的 top-k「所有 query head 共享一个集合」是自然结果——不是设计取舍，而是 MLA-MQA mode 本来就只有一份 KV 可选。
- **它把「KV cache 压缩」从减头数解放成压低秩**。这条轴让长上下文 serving 的 KV cache 可以压到比 GQA/MQA 更小，同时不像 MQA 那样牺牲表达力，是百万 token 上下文路线（DeepSeek-V4）成立的前提之一。
- **两种 mode 的存在揭示了训练/推理的非对称**。训练/prefill 要并行、用 MHA mode 展开；decode 要省 KV、用 MQA mode 吸收。同一套权重两种算法形态，是 MLA 工程上最容易被忽略但最关键的点。

## 待追问

- **query 压缩「具体怎么省激活显存」论文未展开，待坐实**。一个合理推测：1536 维 latent 当「细腰」checkpoint，只常驻它、backward 时从它重算出 16384 维大 query（~10× 节省）。依据是 V2 训练框架确实用了 recomputation（§ 训练："a portion of the operators are recomputed to save activation memory"），但这句是**块级通用技巧**，论文**没把它专门挂到 query 压缩上**——所以「存细腰+重算 query」是推断，非原文机制。且实际工程中 gradient checkpointing 多在 transformer block 级别整体包，未必是 query 专属。求证难点：博客讲解多为同款二手推断（会循环），官方 modeling 代码大概率只见块级重算；要硬坐实需 DeepSeek 自己的训练代码或作者澄清。
- MLA 的 $d_c$、$d_c'$、$d_R$ 具体取值与各上/下投影维度，[V2 来源页](../sources/deepseek-v2.md) 也未抄全；如需精确复现可回原文 §2.1.2–2.1.3 与配置表补。
- 附录 D.2「MLA vs MHA」的具体对比数字未沉淀，可作为「MLA 真的 > MHA」这一主张的直接证据补充。
- DeepSeek-V2 → V3 → V3.2（DSA）→ V4（CSA）这条演进链上「每一步在 MLA 上加了什么」，值得做一张演进表（V3 的 MLA 改动、V3.2 加 DSA、V4 加 token 压缩 + Shared-KV MQA）。
- MSA 选 GQA 而非 MLA 作底座，是 MiniMax 与 DeepSeek 的路线分歧还是有明确的效率/质量权衡论证？需要对比两篇论文的动机段。

## 相关页面

- 来源：[DeepSeek-V2](../sources/deepseek-v2.md)（MLA 首次提出）、[DeepSeek-V3.2](../sources/deepseek-v32.md)、[DeepSeek-V4](../sources/deepseek-v4.md)
- [DeepSeek Sparse Attention](deepseek-sparse-attention.md)
- [高效长上下文注意力](efficient-long-context-attention.md)
- [稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)
- 模型：[GLM-5](../models/glm-5.md)、[DeepSeek-V4](../models/deepseek-v4.md)
