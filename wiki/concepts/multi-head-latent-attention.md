# Multi-Head Latent Attention (MLA)

## 定义

MLA 是 DeepSeek 在 DeepSeek-V2（2024）提出的注意力变体，目标和 GQA / MQA 一样是压缩 KV cache，但走的是**正交的另一条轴**：

- **MHA → GQA → MQA 这条轴：减 KV head 数**。MHA 每个 query head 配一份独立 KV；GQA 让若干 head 共享一组 KV；MQA 所有 head 共享 1 组 KV。head 数越减，KV cache 越小，但 KV 表达力也越弱。
- **MLA 这条轴：低秩压缩**。不动 head 数，而是把每个 token 的 KV 联合**下投影成一个低维 latent 向量** $c^{KV}$ 存进 cache；用时再通过各 head 的**上投影**矩阵恢复出 per-head 的 K、V。cache 里只存 latent，所以比 GQA/MQA 还小，但因为上投影是 per-head 的，表达力能追回 MHA 级别。

![DeepSeek-V2 Figure 3：MHA / GQA / MQA / MLA 四种注意力的对比。前三种沿「减 KV head 数」一条轴（MHA 每 head 一份 KV → GQA 分组共享 → MQA 全共享一份），推理时缓存的都是显式 K/V（图中斜线填充 = Cached During Inference）。MLA 走正交的另一条轴：把 K/V 联合压缩成一个 Compressed Latent KV，推理时**只缓存这个 latent**（图中唯一斜线填充处），用时再经 projection 上投影回 per-head K/V。](../assets/deepseek-v2/fig3-mha-gqa-mqa-mla.png)

> Figure 3（原文截图，§ 2.1）：MHA、GQA、MQA、MLA 的简化对比。斜线填充表示推理时需缓存的部分——直观说明 MLA「只 cache 压缩 latent」相对前三种「cache 显式 K/V」的区别。

**与 MQA 的关系——近亲但不同源**。MLA 在推理时可以做「矩阵吸收」：把上投影 $W_{UK}$ 吸进 query 投影、$W_{UV}$ 吸进输出投影，于是不必显式还原 per-head K/V，直接拿所有 query head 对那一个共享 latent 做注意力——计算形态就**退化成 MQA**。所以若在「GQA vs MQA」里硬找 MLA 的近亲，是 **MQA**：两者都让所有 query head 共享一份 KV；区别是纯 MQA 共享的是**原始 KV**（丢表达力），MLA 共享的是**压缩 latent** 再 per-head 上投影（带低秩补偿）。可以理解为「带低秩补偿的 MQA」。

矩阵吸收带来的「两种 mode」（MHA mode / MQA mode）以及「MQA mode」一词在 V3.2 里横跨两条语义轴的歧义，是本页最易混淆处，单列在[下文「两种 mode 与『MQA mode』歧义」](#两种-mode-与mqa-mode歧义)详述。

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

  Table 1（原文 § 2.1.4 数据重排为 Markdown；$n_h$=head 数，$d_h$=每 head 维度，$l$=层数，$n_g$=GQA 组数，$d_c/d_R$=KV 压缩与 decoupled 维度）：

  | 注意力机制 | 每 token KV cache（元素数） | 能力 |
  | --- | --- | --- |
  | MHA | $2 n_h d_h l$ | Strong |
  | GQA | $2 n_g d_h l$ | Moderate |
  | MQA | $2 d_h l$ | Weak |
  | **MLA（本文）** | $(d_c + d_R) l \approx \tfrac{9}{2} d_h l$ | **Stronger** |

- **效率数字**：KV cache −93.3%，最大生成吞吐 5.76×（vs DeepSeek 67B MHA）。

## 两种 mode 与「MQA mode」歧义

矩阵吸收让同一套 MLA 权重有了两种等价算法形态，DeepSeek-V3.2 附录 A 的 Figure 7 把它们命名为 **MHA mode** 和 **MQA mode**：

![DeepSeek-V3.2 Figure 7：MLA 的两种等价算法形态。左 (a) MHA mode——latent c_KV 经 W^UK / W^UV 按 head 展开成每个 head 的 K/V，进 Multi-Head Attention core；右 (b) MQA mode——W^UK 吸收到 query 侧（q^A = W^UK q^C）、W^UV 移到 attention 输出之后（o = W^UV o^C），latent c_KV 作共享 KV 进 Multi-Query Attention core。两处 apply RoPE 只作用在带 R 上标的位置分量 q^R / k^R 上。](../assets/deepseek-v32/fig7-mha-mqa-mode.png)

> Figure 7（原文截图，§ A. MHA and MQA Modes of MLA）：MLA 有 **MHA mode** 和 **MQA mode**，两者之间可相互变换。DeepSeek-V3.1-Terminus **训练 / prefill 用 MHA mode，decode 用 MQA mode**。

**「MQA mode」一词在 V3.2 里横跨两条轴，别混**。它被用在两处不同的事上，是「DSA 到底是不是在 MQA mode 上训练」这类困惑的根源：

- **轴一 · compute form**：MHA mode 与 MQA mode 是**同一注意力的两种等价算法形态**（矩阵吸收，可互相变换）。分工是「训练 / prefill 用 MHA mode、decode 用 MQA mode」——但 Figure 7 的 caption 把这句**限定在 DeepSeek-V3.1-Terminus（dense 基座）**："For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding." 正文进一步说 Figure 7 展示的是 MLA 的「two aspects … as well as the transformation between them」。
  - **这条分工不是 V3.1-Terminus 独创，而是 MLA 自 DeepSeek-V2 起的通用工程惯例**。论文把 caption 写成「For DeepSeek-V3.1-Terminus」只是因为 V3.1-Terminus 是 V3.2 的 dense 训练起点、论文拿手头这个基座举例；V3.1-Terminus 本身是 V3.1 的一次 update（增强语言一致性 / agent 能力，**非新基座、架构未变**），其注意力就是不带 DSA 的标准 MLA。第三方讲解明确指出吸收是**推理专属**："the weight matrices can be multiplied to produce a single weight matrix. This can only be done **during inference** because **during training they need to be kept separate so that the gradients flow** properly"——这正是下文「训练展开第二动因」（梯度分别回流）的独立旁证，且讲的是 MLA 本身而非某个版本。（来源：[Lior Sinai, DeepSeek's Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)；[DeepSeek-V3.1-Terminus 官方公告](https://api-docs.deepseek.com/news/news250922)）
- **轴二 · selection 结构**：§「Instantiate DSA Under MLA」里的「we implement DSA based on the MQA mode of MLA」指的是 **KV entry（latent）跨所有 query head 共享**——原文 "each latent vector (the key-value entry of MLA) will be **shared across all query heads of the query token**"，故 DSA 的 top-k **所有 head 选同一组 token**。原文的因果是**先给 kernel 约束再推出 MQA mode**："at the kernel level, each key-value entry **must be shared across multiple queries** for computational efficiency" → "**Therefore**, we implement DSA based on the MQA … mode of MLA"。这是**共享 / 选择结构**的陈述，**不是**「训练前向用 MQA 算术」。
- **两条轴正交的铁证**：DSA 下，短上下文 prefill 仍**「specially implement a masked MHA mode to simulate DSA」**（效率讨论一节）——compute form 是 MHA、selection 是 DSA，同时成立。所以「训练/prefill 走 MHA 形态」与「DSA 基于 MQA mode」并不矛盾，前者是算术形态、后者是 latent 跨头共享。

> **交叉检验结论**：上述三处引述均已回 [DeepSeek-V3.2 原文](../sources/deepseek-v32.md) 逐句核对，准确无误；「两条轴」这个切分本身建立在 selection（§ Instantiate DSA Under MLA）、compute form（Figure 7 caption）、masked MHA（效率讨论）三处不同语境之上，是**论文支撑的合理拆分而非臆测**。需注意：中文技术博客层面**未见有人这样明确区分两条轴**（多数文章只讲「MLA 矩阵吸收→MQA」这条 compute-form 轴），故这一拆分的**外部佐证仅到论文一手原文为止**，是本页的原创综合。

**落到最具体一问——「DSA 训练时那份 latent 是展开还是吸收？」**：

| 轴 | 训练 | decode |
| --- | --- | --- |
| compute form（展开 / 吸收） | **展开**（MHA mode：那份共享 latent 被 $W^{UK}/W^{UV}$ 上投影成 128 份 per-head K/V 再算） | **吸收**（MQA mode：不展开，吸收矩阵后直接打 latent） |
| selection 结构（top-k 跨头） | **共享**（一份 latent / 一组 token） | **共享**（同左，架构写死） |

即：**「选哪些 token」恒共享（一份 latent）；「latent 怎么参与计算」训练展开、decode 吸收**——选择共享和计算展开是两件事，可同时成立。训练展开有两个动因：(1) compute-bound 下展开后每对点积 128 维 < 吸收的 512 维；(2) 训练要让 $W^{UQ}/W^{UK}/W^{UV}/W^O$ 各自 live、梯度分别回流，而吸收是推理期预合并固定权重的优化。

## 定量：展开 vs 吸收的 crossover ≈ 341 token

> **展开 vs 吸收的算力对比不是非黑即白，而是 $seq\_len$ 的函数（外部实测交叉验证）**。第三方按 DeepSeek-V3 配置画计算流图实测：**decode（$seq\_len=1$）下吸收版（MQA mode）算力恒更低**；但 prefill 场景随 $seq\_len$ 增长，「非吸收（展开）算力 − 吸收算力」的差值**由正转负**——即存在一个临界序列长度，超过它后**展开版反而比吸收版更省算力**。直觉：吸收把上投影预合并进 Q/O，省掉的是「逐 token 重复展开 K/V」的开销，这在 cache 长、新 query 少（decode）时划算；而 prefill 一次要算 $seq\_len \times seq\_len$ 个注意力对，展开后每对点积只在 128 维 head_dim 上做、吸收后却在 512 维 latent 上做，序列一长，点积维度的劣势盖过省下的展开开销。所以「训练/prefill 用 MHA 展开、decode 用 MQA 吸收」不只是工程惯例，而是有算力 crossover 支撑的取舍。（来源：[「MLA计算流全图解&吸收矩阵对比分析」实测 Tflops 对比](https://www.cnblogs.com/wujianming-110117/p/19112429)）

上面是定性结论。用 [DeepSeek-V2](../sources/deepseek-v2.md) 原文配置（$d=5120,\ n_h=128,\ d_h=128,\ d_c=512,\ d_R=64$）数一遍 prefill 主导 FLOPs，可把临界点定量到 **≈341 token**：

把「展开算力 − 吸收算力」写成 $\text{diff}(L) = A\cdot L + B\cdot L^2 = L\,(A + BL)$，两项性质相反，crossover 在 $L^* = -A/B$：

| 项 | 系数（V2 配置） | 来源 | 谁吃亏 |
| --- | --- | --- | --- |
| 线性项 $A$ | $+1.68\times10^7$ | **展开独有**：把 latent 上投影成 per-head K/V，每 token 固定一次（$d_c \cdot n_h \cdot d_h \cdot 2$） | 展开 |
| 二次项 $B$ | $-4.9\times10^4$ | **展开省**：QK/AV 每对点积维度更小——展开 $d_h{+}d_R=192$、$d_h=128$ vs 吸收 $d_c{+}d_R=576$、$d_c=512$；attention 对数 $\propto L^2$ | 吸收 |

$$L^* = -A/B \approx 341\ \text{token}$$

- **$L < 341$**：线性项（上投影固定成本）主导 → **吸收赢**（它根本不上投影）。
- **$L > 341$**：二次项（注意力对数 $\propto L^2$、展开每对维度仅吸收的 $\approx 1/3$）主导 → **展开赢**。
- **decode**（query=1 对长 cache）：退化到 $L\approx1$ 一侧，**吸收恒省** ✓——这就是「decode 用 MQA mode」的算力依据。

> **这把 V3.2「短序列 prefill 用 masked MHA mode」（[来源页](../sources/deepseek-v32.md)，效率讨论一节）从工程惯例升级成可解释的取舍**：几百 token 以内，DSA 稀疏选择没收益、吸收又因上投影开销不划算，**稠密 MHA 展开最省**；序列再长才轮到 DSA 稀疏路径接管。理论临界点（~341）与工程决策（短 prefill 走稠密 MHA）对得上。
>
> **诚实限定**：(1) 341 是**数量级而非精确值**——只数了 KV 上投影 + QK score + AV 三个主导项，省略两条路径共有的下投影 / softmax / RoPE apply 与访存、kernel 实际开销；真实 crossover 受实现影响会偏移，外部实测（吴建明文）曲线形状吻合但临界点不必相同。(2) 这是 **V2 配置**；V3/V3.2 的 $n_h,d_c$ 沿用 MLA 不变故量级一致，但 V3.2 长序列实际走 **DSA 稀疏**（只 attend top-k=2048），score 项不再是满 $L^2$，crossover 的语义被 DSA 改写。

（仍留白：论文没有一句直说 V3.2 **长上下文**训练前向用哪种算术形态——长上下文走 DSA 稀疏路径，「展开 / 吸收」此时的取舍论文未展开，要坐实需翻开源 inference/训练代码。短上下文 / prefill 用 MHA 展开形态有原文（Figure 7 caption + masked MHA mode）确证。）

## 跨报告信号

- **[DeepSeek-V2](../sources/deepseek-v2.md)（起源，2024）**：MLA 首次提出。236B/21B MoE，靠 MLA 把 KV cache 降 93.3%、吞吐提 5.76×，并用附录 D.1 消融论证「减头数」（GQA/MQA）有质量代价、故走「压低秩」这条正交轴。
- **[DeepSeek-V3.2](../sources/deepseek-v32.md)**：把 [DSA](deepseek-sparse-attention.md) **实例化在 MLA 的 MQA mode 上**——latent vector（即 MLA 的 KV entry）被同一 query token 的所有 query head 共享，DSA 的 top-k 就在这些 latent entry 上选。原文："we implement DSA based on the MQA (Shazeer, 2019) mode of MLA"。这里的「MQA mode」是**轴二 · selection 结构**（见上文「两种 mode 与『MQA mode』歧义」），**不是说训练前向用 MQA 算术**——事实上短上下文 prefill 仍用「masked MHA mode to simulate DSA」。
- **[DeepSeek-V4](../sources/deepseek-v4.md)**：CSA 进一步在 MLA 血统上叠压缩——core attention 是 **Shared-KV MQA**，query 仍由**压缩 latent 向量上投影**得到（且该 latent 与 indexer query 共享）。即 MLA 的 latent-query 结构被 CSA 继承，再在压缩 KV entry 上做 MQA。
- **[GLM-5](../models/glm-5.md)**：同样在 MLA-DSA 基座上做长上下文稀疏，RL 阶段冻结 indexer + deterministic top-k（与 V3.2「post-training 继续训练 indexer」是两条路线）。
- **横向**：见 [稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)——DSA、CSA 的「底层」栏都落在 MLA / MLA-MQA 上，而 MSA 走的是 GQA 底座、NSA/MoBA 走 MQA/GQA 底座，构成「减头 vs 压秩」两种起点的分野。

## 为什么重要

- **它定义了 DeepSeek 系稀疏注意力的起点**。DSA、CSA 都不是凭空设计的注意力，而是「在 MLA 的 latent 共享结构上再加稀疏选择」。理解 MLA 的 MQA mode（所有 head 共享一个 latent KV）就理解了为什么 DSA 的 top-k「所有 query head 共享一个集合」是自然结果——不是设计取舍，而是 MLA-MQA mode 本来就只有一份 KV 可选。
- **它把「KV cache 压缩」从减头数解放成压低秩**。这条轴让长上下文 serving 的 KV cache 可以压到比 GQA/MQA 更小，同时不像 MQA 那样牺牲表达力，是百万 token 上下文路线（DeepSeek-V4）成立的前提之一。
- **两种 mode 的存在揭示了训练/推理的非对称**。训练/prefill 要并行、用 MHA mode 展开；decode 要省 KV、用 MQA mode 吸收。同一套权重两种算法形态，是 MLA 工程上最容易被忽略但最关键的点。

## 待追问

- **query 压缩「具体怎么省激活显存」论文未展开，待坐实**。一个合理推测：1536 维 latent 当「细腰」checkpoint，只常驻它、backward 时从它重算出 16384 维大 query（~10× 节省）。依据是 V2 训练框架确实用了 recomputation（§ 训练："a portion of the operators are recomputed to save activation memory"），但这句是**块级通用技巧**，论文**没把它专门挂到 query 压缩上**——所以「存细腰+重算 query」是推断，非原文机制。且实际工程中 gradient checkpointing 多在 transformer block 级别整体包，未必是 query 专属。求证难点：博客讲解多为同款二手推断（会循环），官方 modeling 代码大概率只见块级重算；要硬坐实需 DeepSeek 自己的训练代码或作者澄清。
- MLA 各维度已据 V2 原文 §2.1.2–2.1.3 与配置表坐实主要值：KV latent $d_c=512$、query latent $d_c'=1536$、decoupled per-head $d_R=64$（$n_h=128,\ d_h=128$）；V2-Lite 则 $d_c=512$ 但**不压缩 query**。各上/下投影矩阵的完整形状若需精确复现，仍可回原文配置表补全。
- 附录 D.2「MLA vs MHA」的具体对比数字未沉淀，可作为「MLA 真的 > MHA」这一主张的直接证据补充。
- DeepSeek-V2 → V3 → V3.2（DSA）→ V4（CSA）这条演进链上「每一步在 MLA 上加了什么」，值得做一张演进表（V3 的 MLA 改动、V3.2 加 DSA、V4 加 token 压缩 + Shared-KV MQA）。
- MSA 选 GQA 而非 MLA 作底座，是 MiniMax 与 DeepSeek 的路线分歧还是有明确的效率/质量权衡论证？需要对比两篇论文的动机段。

## 相关页面

- 来源：[DeepSeek-V2](../sources/deepseek-v2.md)（MLA 首次提出）、[DeepSeek-V3.2](../sources/deepseek-v32.md)、[DeepSeek-V4](../sources/deepseek-v4.md)
- [DeepSeek Sparse Attention](deepseek-sparse-attention.md)
- [高效长上下文注意力](efficient-long-context-attention.md)
- [稀疏注意力机制对比](../comparisons/sparse-attention-mechanisms.md)
- 模型：[GLM-5](../models/glm-5.md)、[DeepSeek-V4](../models/deepseek-v4.md)
