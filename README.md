# paper-reading-wiki

论文阅读知识库。一个纯 Markdown 工作区，无需构建工具。

## 目录说明

- **`raw/`** —— 原始内容（论文 / 技术报告 PDF）。只读，不修改。
- **`wiki/`** —— 知识库本体。生成的来源页、模型页、概念页、比较页，入口是 `wiki/index.md`，操作记录在 `wiki/log.md`。
- **`learning/`** —— 个人学习路线与讨论记录，从 [学习入口](learning/index.md) 继续；目录、命名与学习规则见 [学习工作流](.agents/guides/learning-workflows.md)。
- **`plugins/wiki-knowledge-cards/`** —— Obsidian [知识卡片插件](plugins/wiki-knowledge-cards/README.md)，随机展示 wiki 内容；无需构建，本机安装与抽取进度保存在 `.obsidian/`。
- **`.agents/skills/`** —— skill 所在目录（llm-wiki 工作流）。
- **`AGENTS.md` / `CLAUDE.md`** —— `AGENTS.md` 是短入口与硬约束；按任务加载 `.agents/guides/` 中的工作流、证据、schema 与交付细则。`CLAUDE.md` 仅作入口 stub。
