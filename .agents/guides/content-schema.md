# 内容模型与写作规范

## 仓库层次

- `raw/`：原始 PDF、文章、图片或数据，正常工作流只读；文件名保持原样。
- `wiki/`：LLM 生成的知识层。来源、模型、概念、比较分别置于 `wiki/sources/`、`wiki/models/`、`wiki/concepts/`、`wiki/comparisons/`。
- `wiki/assets/<source-slug>/`：被页面引用的图像资产，必须进入版本控制。
- `wiki/index.md`：内容目录；`wiki/log.md`：当前保留知识的已完成动作时间线；`wiki/TODO.md`：未完成工程待办。

`.obsidian/` 不在版本控制中，不可作为跨机器共享设置的依据。常用只读命令是 `rg --files` 与 `rg "term" wiki/ raw/`；仓库没有构建系统、包管理器或测试运行器。

## 语言与 Markdown

- 正文、规则、说明、示例均用中文；技术专名可首次附英文。MoE、DSA、MTP、RL、CoT、FFN 等缩写保持英文。
- 不翻译或改写命令、路径、文件名、slug、YAML 字段、正则、Git trailer、代码、URL 与 Markdown 链接目标；`ingest`、`deepen`、`distill`、`verify`、`refactor`、`maintenance`、`raw/`、`wiki/`、PyMuPDF、`fitz` 同样保持英文。
- 标题简洁、描述性强；除专名外采用 sentence case。文件名为英文 kebab-case，绝不重命名既有 slug。
- 交叉引用使用相对 Markdown 链接，如 `[GLM-5](../models/glm-5.md)`；不要使用 Obsidian `[[wiki-links]]`。
- 外部引用使用普通行内链接，如 `（来源：[标题](url)）`。禁止 `^[url 标题]`、没有定义的 `[^url 标题]`，以及会被误解析为链接的 prose 裸方括号。

## OKF v0.1

`wiki/` 下除 `index.md`、`log.md` 外的所有 Markdown 文件必须以 YAML frontmatter 开头。`type` 必填，目录映射固定：

| 路径 | `type` |
| --- | --- |
| `sources/*.md` | `Source` |
| `models/*.md` | `Model` |
| `concepts/*.md` | `Concept` |
| `comparisons/*.md` | `Comparison` |
| `TODO.md` | `TodoList` |

已知时填写 `title`、`description`、`tags`、`timestamp`；来源页另填 `resource`。页面正文使用中文，slug 与目录名保持英文。

## 页面骨架

- **来源页**：`## 来源`（PDF 路径、标题、版本 / 日期、团队、模型链接）→ `## 核心结论` → `## 架构与训练` → `## 后训练` → `## 评测要点` → 具体的 `## 待追问`。
- **概念页**：`## 定义` → `## 跨报告信号` → `## 为什么重要` → 具体的 `## 待追问` → `## 相关页面`。
- **模型页**：`## 身份` → 含**模态**行的 `## 关键事实` 表 → 技术身份说明 → `## 相关页面`。仅重读来源后才能把模态标为已核实；不要把评测、verifier 或展望中的 vision / multimodal 误判为模型输入模态。
- **TODO**：只放未完成工程；完成后删除条目并按适用工作流处理日志。不要留下空壳 `## 待追问`。
