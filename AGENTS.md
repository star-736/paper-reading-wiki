# paper-reading-wiki agent guide

这是一个由 LLM 维护的中文论文知识库；`raw/` 保存原始材料，`wiki/` 保存可检索的生成知识。根文件只保留始终适用的规则；**开始编辑前必须按下表阅读对应手册**。不要用 `@` 导入这些手册，以保持按需加载。

## 快速路由

| 任务 | 开始前必须读 |
| --- | --- |
| 任何知识库查询或改动 | `wiki/index.md` |
| 页面、frontmatter、链接、目录或语言 | `.agents/guides/content-schema.md` |
| `ingest`、`deepen`、`distill`、`verify`、`refactor`、`maintenance`、质量剔除 | `.agents/guides/wiki-workflows.md` |
| 机制主张、证据等级、PDF 图表或 VLM | `.agents/guides/evidence-and-visuals.md` |
| 写回检查、Git、并发、提交或推送 | `.agents/guides/delivery-and-concurrency.md` |

## 不可违反的规则

- 人工可读内容使用中文；命令、路径、文件名、slug、YAML 字段、代码、URL、Markdown 链接目标和技术固定标识保持原样。
- 不引入构建系统、包清单、测试运行器或新的工具依赖；图表提取仅允许已安装的 PyMuPDF（`fitz`）。
- 正常工作流不修改 `raw/`；它是证据源。例外只有经 `verify` 确认的**完整质量剔除**，规则见工作流手册。
- 不孤立新增 wiki 页；不得留下断裂链接、孤儿页或孤儿资产。现有主题优先 `deepen`，不要制造近亲页面。
- 所有非平凡主张必须遵守证据等级；不能把推断写成原文确证。
- 用户指令优先于本文件和手册；手册之间若冲突，以更具体、与当前任务直接相关的一份为准。

## 指令体系

`AGENTS.md` 与 `.agents/guides/*.md` 共同构成仓库约定的唯一事实来源；前者是入口索引，后者是按需细则。修改约定时编辑对应手册；只有路由或跨任务硬约束改变时才编辑本文件。`CLAUDE.md` 是 `@AGENTS.md` 的入口 stub，不单独维护。
