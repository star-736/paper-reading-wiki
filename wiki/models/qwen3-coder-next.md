# Qwen3-Coder-Next

## 身份

Qwen3-Coder-Next 是 Qwen 团队（Alibaba）面向 coding agent 的开源模型，基于 **Qwen3-Next** 的 hybrid attention + MoE 架构。它是 [Gated Attention](../concepts/attention-gating.md) 与 [GDN 线性注意力](../concepts/linear-attention-and-delta-rule.md) 落到生产编码模型上的一个实例。来源报告见 [Qwen3-Coder-Next 技术报告](../sources/qwen3-coder-next.md)。

## 关键事实

下表架构字段以 HuggingFace 官方权重 [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) 的 `config.json` 为准（已据 config 核实）：

| 项 | 值 |
| --- | --- |
| 总参数 | 79.7B（≈80B；HF safetensors 计 79,674,391,296） |
| 激活参数 | ~3B（512 expert 选 10 + 1 shared expert，标称 A3B） |
| 层数 | 48 层 |
| token mixer 混合 | **3 GDN 线性层 : 1 full-attention**（`full_attention_interval=4`，即每 4 层一个全注意力层；`use_sliding_window=false`） |
| 全局层 | 带输出门的 gated full attention（16 head / 2 KV head，head_dim 256，`partial_rotary_factor=0.25`） |
| 线性层 | Gated DeltaNet（16 key head / 32 value head，head_dim 128，conv kernel 4） |
| MoE | 512 expert 选 10，shared/routed intermediate 512，`decoder_sparse_step=1`（每层都 MoE） |
| hidden_size | 2048 |
| 上下文 | 262,144 token（`max_position_embeddings`） |
| 模态 | **纯文本**（config 无 `vision_config`，编码 agent 定位；已据 config 核实） |
| 公开权重 | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)、`-Base`、`-FP8`、`-GGUF` |

## 技术身份

它把 Qwen3-Next 的「3:1 混合栈」直接用于编码场景：4 层里 3 层是 [GDN 线性注意力](../concepts/linear-attention-and-delta-rule.md)（省 KV-cache）、1 层是带输出门的全注意力（[gated attention](../concepts/attention-gating.md)，维持全局信息流 + 去 attention sink）。报告主旨不是架构创新，而是用大规模合成可验证编码任务 + RL 把这个 80B-A3B 在 SWE-Bench / Terminal-Bench 上推到有竞争力的水平。

值得记的一点：它和 [Kimi Linear](kimi-linear.md) 是同尺度同思路（线性层 + 少量全局层）但**全局层不同**的对照——Qwen3-Next 系全局层是 gated full attention，Kimi Linear 换成了 MLA（见 [线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md) 跨报告信号）。

## 相关页面

- 来源：[Qwen3-Coder-Next 技术报告](../sources/qwen3-coder-next.md)
- 概念：[注意力门控](../concepts/attention-gating.md)、[线性注意力与 delta rule](../concepts/linear-attention-and-delta-rule.md)
- 同思路对照：[Kimi Linear](kimi-linear.md)（全局层用 MLA）
- 同家族：[Qwen3.5](qwen3.5.md)（共享 hybrid MoE 架构）
