# Kimi Linear

## 身份

Kimi Linear 是 Moonshot AI（Kimi Team）随 [Kimi Linear 技术报告](../sources/kimi-linear.md) 释出的研究模型，用来在公平对比下验证**混合线性注意力**架构。它不是生产旗舰（那是 [Kimi K2.5](kimi-k2.5.md)），而是一个证明「线性注意力可以 drop-in 替换 full attention 且更优」的 48B 规模载体。

## 关键事实

| 项 | 值 |
| --- | --- |
| 总参数 | 48B |
| 激活参数 | 3B（每 forward 选 8/256 expert，含 1 shared expert） |
| 架构 | 混合线性注意力 MoE（backbone 沿用 Moonlight） |
| token mixer | KDA : MLA = **3:1**（layerwise 交替，3 个 KDA 层配 1 个 Full MLA 层） |
| 全局层 | Full [MLA](../concepts/multi-head-latent-attention.md)，**NoPE**（不加位置编码） |
| 线性层 | [Kimi Delta Attention（KDA）](../concepts/linear-attention-and-delta-rule.md)，channel-wise 细粒度门 |
| MoE sparsity | 32（256 expert 选 8 + 1 shared），首层 dense |
| 优化器 | Muon |
| 训练规模 | 公平对比主实验 1.4T token |
| 模态 | **纯文本**（报告仅在文本 LM / 长上下文 / RL 任务上评测；无视觉输入） |
| 公开 checkpoint | [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)（pre-trained + instruction-tuned） |

## 技术身份

Kimi Linear 的卖点不是参数规模，而是**注意力架构**：把绝大多数层的 token mixer 从 softmax 注意力换成 KDA 这种线性 RNN，只保留 1/4 的全局 MLA 层维持远程信息流。结果是 1M context 下 KV cache 降最多 75%、decode 吞吐最多 6.3×，同时在短/长上下文和 RL 上追平或超过等配方的 full MLA 基线。

模态上需谨慎：报告通篇是文本语言建模与长上下文/RL 评测，没有视觉/多模态输入设计，因此标**纯文本**（已据报告核实——区别于同厂的 Kimi K2.5 多模态）。

## 相关页面

- 来源：[Kimi Linear 技术报告](../sources/kimi-linear.md)
- 概念：[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)、[注意力门控](../concepts/attention-gating.md)、[Multi-Head Latent Attention](../concepts/multi-head-latent-attention.md)、[高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- 同厂模型：[Kimi K2.5](kimi-k2.5.md)（生产旗舰，多模态 agentic）
