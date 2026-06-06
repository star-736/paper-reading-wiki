# GLM-5

## 身份

GLM-5 是 Zhipu AI 与 Tsinghua University 发布的开放权重旗舰 MoE 模型。技术报告把它定位为从代码辅助走向 [agentic engineering](../concepts/agentic-engineering.md) 的模型。

主要来源：[GLM-5 技术报告](../sources/glm-5.md)。

## 关键事实

| 项目 | 数值 |
| --- | --- |
| 总参数 | 744B |
| 激活参数 | 40B |
| 专家数 | 256 total experts |
| 层数 | 80 |
| 基础训练 tokens | 28.5T |
| 最大 SFT 上下文 | 202,752 tokens |
| 主要注意力变化 | DSA 长上下文稀疏注意力 |
| 主要后训练主题 | 异步 agent RL 与 cross-stage distillation |

## 解释

GLM-5 的差异点不只是模型规模，而是完整 agent stack：长上下文训练、preserved/interleaved thinking、执行环境、异步 rollout，以及真实软件工程任务评测。

最值得关注的研究信号是 DSA 与 RL 的结合。DSA 降低长上下文成本，但报告指出 RL 稳定性依赖 DSA indexer 的 deterministic top-k selection。这说明实现细节本身就是模型能力的一部分。

## 相关页面

- [高效长上下文注意力](../concepts/efficient-long-context-attention.md)
- [DeepSeek Sparse Attention](../concepts/deepseek-sparse-attention.md)
- [异步 Agent RL](../concepts/asynchronous-agent-rl.md)
- [Agentic 模型的后训练](../concepts/post-training-for-agentic-models.md)
- [2026 开放模型技术报告对比](../comparisons/2026-open-model-technical-reports.md)
