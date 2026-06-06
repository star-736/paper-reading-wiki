# Multi-Teacher On-Policy Distillation

## 一句话定义

Multi-Teacher On-Policy Distillation（MOPD）是 [MiMo-V2-Flash](../models/mimo-v2-flash.md) 的核心后训练范式：student 从自己的分布采样，再从多个 domain-specialized teacher 获得 token-level supervision，把多个专家能力合并进同一个模型。

## 为什么需要 MOPD

顺序后训练常见问题是 capability see-saw：数学 RL 提升后，写作或代码退化；代码 RL 提升后，安全或通用问答退化。简单混合数据或合并权重也不总能保留每个 teacher 的峰值能力。

MOPD 的目标是让 student 同时学习多个 teacher 的专门能力，并尽量避免单领域优化带来的互相覆盖。

## 三阶段流程

1. SFT：先建立基础 instruction-following 能力。
2. Domain-specialized training：分别训练专门 teacher，覆盖搜索、代码、工具使用、数学推理、通用推理和安全等任务。
3. MOPD：student on-policy 采样，并根据 prompt 所属领域，从对应 teacher 获得 token-level KL reward。

## 技术直觉

普通离线蒸馏让 student 学 teacher 生成的数据，但这些数据可能不在 student 当前分布上。MOPD 让 student 学自己真实会生成的轨迹，因此更接近部署时遇到的分布。Teacher 不直接给最终答案，而是通过 logits 或 token-level KL 提供细粒度训练信号。

报告中的 surrogate loss 可以理解为：如果某个 token 在 domain teacher 下概率更高，而在 student 下概率较低，这个 token 对 student 形成正向 advantage；反之则形成惩罚。MOPD 还可以与 ORM/GRPO 的 advantage 混合。

## 报告中的结果

MiMo-V2-Flash 报告的 Table 7 显示，MOPD 后 student 在多个 benchmark 上接近或超过 best teacher。例如 AIME 2025 从 89.3 提升到 94.1，HMMT Feb. 2025 从 76.9 到 84.4，LiveCodeBench 从 77.5 到 83.2，SWE-Bench Verified 从 67.8 到 73.4。

但不是所有项都超过 teacher，例如 BrowseComp 的 best teacher 是 51.7，MOPD 后 student 为 45.4。这说明 MOPD 不是自动合并所有峰值能力，teacher 选择、领域路由和 reward 设计仍然关键。

## 与 DeepSeek-V4 OPD 的关系

[DeepSeek-V4](../models/deepseek-v4.md) 也使用多 teacher OPD，但强调 full-vocabulary logit distillation。它认为 token-level KL 估计虽然省资源，但方差高、训练不稳定，因此通过 teacher hidden-state caching、按 teacher 排序调度和 TileLang kernel 来支持完整 logits 的 KL 计算。

## 待追问

- MOPD 的 domain routing 如何定义？粗粒度领域错误是否会导致负迁移？
- Teacher 数量增加时，student 容量是否足够保留所有能力？
- MOPD 与异步 agent RL 能否形成“先 RL 出 teacher，再 MOPD 融合，再继续 RL”的循环？

