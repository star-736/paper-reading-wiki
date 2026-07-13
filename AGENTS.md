# Repository Guidelines

## Project Structure & Module Organization

This repository is a lightweight Markdown knowledge-base workspace.

- `raw/` stores source materials before they are processed. Treat these files as source-of-truth inputs and avoid modifying them during wiki maintenance.
- `wiki/` contains generated knowledge-base pages. Start at `wiki/index.md`; append workflow history to `wiki/log.md`. Unfinished engineering follow-ups go in `wiki/TODO.md` (a checklist), **not** in `log.md` — keep `log.md` a record of what already happened, not a plan of what's next.
- `.agents/skills/llm-wiki/` contains the repo-scoped Codex skill; `references/llm-wiki.md` holds the original gist.
- Generated wiki pages are grouped under `wiki/sources/`, `wiki/models/`, `wiki/concepts/`, and `wiki/comparisons/`.

Keep source documents, generated summaries, and logs separated.

## Build, Test, and Development Commands

No build system, package manager, or test runner is currently configured. Useful local commands are:

- `rg --files` lists tracked and visible workspace files quickly.
- `rg "term" wiki/ raw/` searches generated notes and source filenames.

If tooling is added later, document the exact command and expected output.

## Coding Style & Naming Conventions

Write Markdown in concise sections with descriptive headings. Prefer sentence-case headings for article pages unless a proper noun requires capitalization. Use kebab-case filenames, for example `retrieval-augmented-generation.md`. Cross-links use **relative Markdown paths** (e.g. `[GLM-5](../models/glm-5.md)`), not Obsidian `[[wiki-links]]` — the existing wiki uses Markdown links throughout (see `CLAUDE.md` → "Conventions").

Every non-reserved Markdown file under `wiki/` is an **OKF v0.1 concept file** and must start with YAML frontmatter. `type` is required; use the existing directory mapping: `sources/*.md` → `Source`, `models/*.md` → `Model`, `concepts/*.md` → `Concept`, `comparisons/*.md` → `Comparison`, and `TODO.md` → `TodoList`. Recommended fields (`title`, `description`, `tags`, `timestamp`, and `resource` for source pages) should be present when known. `wiki/index.md` and `wiki/log.md` are OKF reserved files and do not need concept frontmatter.

**External source citations use plain inline Markdown links**, written as `（来源：[标题](url)）` (or an inline `[标题](url)` mid-sentence). **Do not use `^[url 标题]` or `[^url 标题]`** — `^[...]` is a Pandoc-only inline-footnote extension GitHub/Obsidian don't render, and `[^...]` is a footnote *reference* requiring a short identifier plus a matching `[^id]: …` definition; either with a URL inside renders as garbled brackets (the 2026-06-21 maintenance fix). Likewise avoid bare brackets like `prop[erly]` / `continued [training]` in prose — `[…]` followed by text or parens is parsed as link syntax and bleeds link styling. Quote source text verbatim; gloss an elided word with a Chinese parenthetical （训练）, not `[brackets]`.

For logs, use parseable dated headings such as `## [2026-06-06] ingest | Source Title`, where the kind comes from the fixed `<kind>` vocabulary (see "Agent-Specific Instructions" below).

## Testing Guidelines

There are no automated tests. Validate contributions manually against the **write-back checklist** in "Wiki workflows & writing discipline" below:

- Markdown renders without broken headings or malformed relative links.
- New source-derived claims trace to the right evidence tier — re-read the `raw/` PDF for mechanism claims rather than trusting an earlier summary. Keep 原文确证 / 外部佐证 / 推断 distinct, and never assert your own mechanism guess under a "已据原文核实" heading.
- Cross-references go both ways; no orphan pages.
- `wiki/index.md` and `wiki/log.md` are updated when pages are added or revised.

## Commit & Pull Request Guidelines

No meaningful commit-message convention is available in the root workspace history. Use short, imperative messages with a scope when helpful, such as `docs: add wiki index` or `raw: add transformer survey`.

After a turn has produced substantive on-disk changes (an ingest, a deepen, a correction — anything that edited `wiki/` and appended to `wiki/log.md`), proactively offer in one short line to commit and push before ending the turn; don't wait to be asked. Keep it a lightweight offer: skip it for pure read/query turns, and when the user is still mid-discussion, batch the changes and offer once at a natural stopping point rather than after every individual change.

Pull requests should describe what changed, list added or processed sources, note updated wiki pages, and call out unresolved contradictions or follow-up research. Include screenshots only when changes affect rendered diagrams, slides, or visual assets.

## Agent-Specific Instructions

Agents should read `wiki/index.md` first when it exists, then inspect relevant pages and raw sources. Do not overwrite raw files. When ingesting a source, update summaries, cross-references, the index, and the chronological log in the same pass.

`AGENTS.md` and `CLAUDE.md` carry the **same conventions** in tool-specific form; either is self-sufficient. When you change a shared convention, update both so they stay in sync.

## Wiki workflows & writing discipline

### Workflows

Each workflow ends by appending a `## [YYYY-MM-DD] <kind> | <title>` entry to `wiki/log.md`.

- **`ingest` — a new PDF landed in `raw/`:** the output is an **图文交错 (text + inline figures)** set of pages, not a text-only summary.

  > **Vision-assisted content extraction（必要时触发）：** 默认用 PyMuPDF 提取文本 + LLM 分析。只在以下痛点页面临时调 `vision_analyze` 辅助识别：
  >
  > | 触发条件 | 为什么需要 VLM | 怎么用 |
  > |---------|--------------|--------|
  > | **表格密集页**（≥3 个数据表，或跨页/合并单元格复杂表） | `pdftotext` / `page.get_text()` 常列错位、结构崩坏 | VLM 读原图 → 输出结构化表格草稿 → 人工校验 → 转 Markdown |
  > | **公式推导页**（连续 ≥5 行 LaTeX 或复杂矩阵/张量运算） | 文本提取可能丢符号、下标、上标、括号层级 | VLM 视觉确认关键公式 → 人工校验 → 写入正文 |
  > | **扫描版或图文混排页**（OCR 后文本顺序错乱、caption 与图分离） | 纯文本丢失空间布局信号 | VLM 按视觉布局理解 → 输出图文关系 → 人工组织 prose |
  > | **架构图 + 旁边解释文字的跨模态页** | 纯文本提取割裂"图-文"联合信号（如"见图 2，我们采用…"但图 2 的内容在文本里为空） | VLM 联合读图和相邻段落 → 提取设计动机 → 人工校验写入 |
  >
  > **约束：** 正常 prose-heavy 页（无密集表格/公式/图）继续用 LLM + PyMuPDF 文本提取，不调 VLM。VLM 输出只能当"阅读辅助"，不写入正文作为已核实事实；provenance 记 `log.md`。

  (1) read the PDF, identify model entities, key claims, mechanisms, contradictions, **and the figures/tables that carry mechanism or headline-result content**; (2) create `wiki/sources/<slug>.md` and, if a new model, `wiki/models/<slug>.md`; (3) update or create relevant `wiki/concepts/*.md` — append cross-source signals to existing concept pages rather than forking new ones; (4) **embed the key figures**: per "Figures & visual material", extract each mechanism/architecture/headline diagram with PyMuPDF into `wiki/assets/<source-slug>/` and embed it on the page that argues it (re-typeset plain-text tables as Markdown). Every ingest should embed ≥1 figure unless the paper genuinely has no diagram worth showing — say so explicitly in the log if you embed none. (5) update `wiki/comparisons/*.md` if it belongs in an existing comparison; (6) add the new pages to `wiki/index.md` with a one-line Chinese summary; (7) run the write-back checklist, then append the log entry. One source at a time unless asked for batch. A single ingest typically touches 5–15 pages — that fan-out is the point.

- **`deepen` — extend/sharpen an existing page, usually from a user question:** re-read the relevant `raw/` PDF and verify against the actual text first — never deepen from memory of an earlier summary. Touch only the pages the question reaches; add cross-references both ways. **`deepen` vs `distill`:** `deepen` sharpens content on a page that **already exists**; when the conversation surfaces a *new* cross-page insight or framing that needs its own home, that's `distill`. A turn can be both — log under the headline action.
- **`distill` — file a Q&A insight back into the wiki:** the query loop often surfaces durable knowledge written down nowhere yet (a connection across two papers, an original framing, a judgment the user pushed you to sharpen). `distill` captures it so it compounds instead of evaporating into chat — the **second engine** of an LLM-wiki alongside `ingest`. **Sniff test:** file it when the insight is **durable** (not a one-off lookup), **reusable** (a future question will want it), and **synthesizing** (connects ≥2 sources/pages or states a framing no single source spells out). A pure fact lookup ("how many tokens did V3.2 train on") is *not* distilled — just answer. The MLA concept page is the archetype: born from "is DSA evolved from MLA", which no page then answered. **Where it lands:** new cross-source framing/comparison → new `wiki/concepts/` or `wiki/comparisons/` page; genuinely-new synthesis tied to one page → a new section there; an unanswered question → that page's `## 待追问`, not a new page. **How:** same rigor as `ingest`/`deepen` — re-verify each claim against `raw/` (don't trust the chat's phrasing), tag the evidence tier, embed any figure it leans on, wire cross-references **both ways** so it isn't an orphan, run the write-back checklist, log as `distill`. Don't fork a near-twin of an existing page — if the topic has a page, extend it (`deepen`) instead.
- **`verify` — cross-check existing claims against external sources and re-read the primary PDF:** external search is reachable via the Tavily API (`TAVILY_API_KEY` in `~/AppData/Local/hermes/.env`; foreign sources reachable); `web_search` as a native tool is **not** available here, and the `delegate_task` web subagent has been unreliable. Record the verdict per claim (supported / refuted / not-found-externally-but-confirmed-in-primary-source) and downgrade anything that turns out to be inference.
- **`refactor` — restructure a page without changing its facts:** reorder, deduplicate, strip internal tooling traces into reader-facing locators. **Facts and citations must not change** — say so in the log entry.
- **`maintenance` — repo-level housekeeping:** language sweeps, schema-field rollouts, `.gitignore` / version-control changes, link fixes, lint passes (stale claims, contradictions, orphan pages, missing cross-references or citations).
- **Query (no log entry unless it produces a page):** read `wiki/index.md` first, drill into linked pages, cite wiki pages or `raw/` files. **The query loop is where most `distill`-worthy knowledge appears** — while answering, watch for synthesis that passes the distill sniff test and offer to file it back (new `wiki/concepts/`/`wiki/comparisons/` page, or a new section on an existing page). A plain answered question earns no log entry; only filing it back does (logged as `distill`).

`<kind>` is a **fixed vocabulary**: `ingest`, `deepen`, `distill`, `verify`, `refactor`, `maintenance`. Heading form `## [YYYY-MM-DD] <kind> | <title>`; keep it parseable so `rg "^## \[" wiki/log.md` works.

### Evidence discipline

This wiki's core value is that claims are traceable, not just plausible. Every non-trivial mechanism claim carries an evidence tier — make it explicit when tiers could be confused:

1. **原文确证 (primary-source confirmed)** — verified against the `raw/` PDF itself. Cite a reader-facing locator (`§ Instantiate DSA Under MLA`, `Figure 7 caption`, `Table 1`), not a raw line number. Only this tier may sit unqualified under a "已据原文核实" heading.
2. **外部佐证 (externally corroborated)** — supported by an independent source (Tavily hit, official blog, third-party analysis) but not in `raw/`. Link the provenance; mark it as corroboration. Blog explanations of an author's *intent* are second-hand and must not be promoted to tier 1.
3. **推断 / 本页原创综合 (inference / original synthesis)** — your own reasoning or a framing no source states outright. **Label it**, and when it's a mechanism guess, park it in `## 待追问` rather than asserting it in the body.

Rules: never blend a paper's conclusion with your inference of its mechanism in one sentence under a "confirmed" heading; when a past entry over-claimed, downgrade it honestly rather than propping it up with weak corroboration; re-reading the primary PDF beats trusting an earlier summary; `raw/` is never modified — say "`raw/` 未改" in the log when no source was touched.

### 多会话并发提交约定

多个 agent 会话同时 ingest 时，**共享文件（index.md / log.md / 概念页 / 比较页）的并发修改会互相覆盖**。踩坑记录与规则如下：

- **根因**：各会话基于各自读到的 HEAD 编辑共享文件。若会话 A 先提交了含自身条目的 HEAD，会话 B 在其旧 HEAD 上编辑的同名文件被 A 的新 HEAD 覆盖，B 的 index/log 编辑即丢失——表现为"源页/模型页已 push 上线，但 index/log 无对应条目"（孤儿状态）。
- **源页 / 模型页各会话可独立提交**：`wiki/sources/<slug>.md`、`wiki/models/<slug>.md`、`wiki/assets/` 互不冲突，push 前只需 `git pull --rebase`（有未提交改动会挡 pull，先 commit/stash 自己的）。
- **共享文件的写回必须等并发收口**：index.md / log.md / 概念页 / 比较页不要各自会话抢提交。两类安全做法：
  1. 约定一个会话在所有人收口后统一补 index/log/概念信号；或
  2. 每个会话只在**已 `git pull --rebase` 到最新 HEAD** 之上 `git add` 自己那几行再 commit（绝不在别人的未提交 hunk 上 patch；`log.md` 只用 `cat >>` 追加、绝不用 patch 改旧条目，避免 Windows+OneDrive 的 CRLF/LF 不匹配把整个文件标成 changed）。
- **写回检查单补一条**：提交前确认 `wiki/index.md` 列出了本次新增的每一个 source/model 页、`wiki/log.md` 有对应条目——否则线上会出现"有页无索引"的孤儿窗口。
- **`git status` 必查**：提交前确认哪些是自己的未提交改动、哪些是别会话落下的；不要把别人的未提交改动（`??` 文件或 `M` 文件）一起 `git add` 带走。

### 克制原则（检索负债优先）

这套 wiki 的实际消费者是**检索它来答问的 agent**，不是回看 md 的人类。所以页面的敌人是**碎片化与冗余**，不是篇幅：

- **开新页前先问它该不该存在（YAGNI 反射）。** 主题已有页 → `deepen` 它，别 fork 近亲页。同一主题散在多个近亲页 = 检索捞不全、答得有遗漏。这是动手前的第一反射，不只是 `distill` 流程里的一条注意事项。
- **删冗余优于堆新段。** 维护时合并重复段落、删去与他页矛盾的旧表述（人类不回看，删旧表述无阅读损失，反而降噪）。注意与「证据分级」配合：过时**结论**按既有纪律**诚实降级**（park 到 待追问 / 标 推测），不是物理删除；这里删的是**重复与冗余表述**，不是可追溯的事实记录。
- **但绝不为求简而砍机制解释与证据链。** 页面的解释密度与证据 tier 正是 agent 答题的弹药——把页面写薄等于自掏弹药库。克制针对的是「重复、占位、为一个判断反复辩护的散文」，不是「机制讲清楚」。`## 待追问` 没真内容就别留空壳。

### Write-back checklist

Before any `ingest`/`deepen`/`verify`/`refactor` turn is done, confirm:
- [ ] New/changed claims trace to the right evidence tier (primary PDF re-read where it matters).
- [ ] Cross-references added **both ways** (new page links out; related pages link back).
- [ ] `wiki/index.md` updated if a page was added or its one-line summary changed.
- [ ] `wiki/log.md` appended with the correct `<kind>` and a title; entry is a short timeline note, not a transcript — detailed derivations live on the page they concern.
- [ ] No orphan page (every new page has ≥1 inbound link), no broken relative links.
- [ ] Internal tooling traces (raw line numbers, `pdftotext`, scratch dates) kept out of reader-facing prose.
- [ ] Figures a page relies on are **embedded** (not just cited as `Figure N`) per "Figures & visual material"; every `wiki/assets/` file is referenced by ≥1 page.

### Page skeletons

- **Sources:** `## 来源` (PDF path, title, version/date, team, model link) → `## 核心结论` → `## 架构与训练` → `## 后训练` → `## 评测要点` → `## 待追问`.
- **Concepts:** `## 定义` → `## 跨报告信号` → `## 为什么重要` → `## 待追问` → `## 相关页面`.
- **Models:** `## 身份` → `## 关键事实`（Markdown 表，**必须含 `模态` 行**，e.g. 纯文本 / 多模态（文本 + 图像 + 视频））→ explanation/技术身份 → `## 相关页面`. For a multi-variant family with a variant table (e.g. `deepseek-v4.md`), put modality as a `**模态**：…` line below the table. Mark modality 已核实 only after checking the source report — incidental "multimodal/vision" mentions are usually RL-pipeline verifiers, eval benchmarks, or future-work outlooks, not the model's own input.
- **Language split:** page content in Chinese; filenames/dirs/slugs in English kebab-case; technical acronyms (MoE, DSA, MTP, RL) stay English inside Chinese prose. Don't re-translate or rename existing slugs.
- **`## 待追问`** is real schema, not boilerplate — populate it with concrete follow-ups.

### Figures & visual material (图文化)

The wiki is **图文交错 (text + inline figures)**, not text-only. When a page leans on a paper's figure or table — especially a mechanism/architecture diagram a reader can't reconstruct from prose — embed the actual image instead of only citing `Figure N`. Tooling: **PyMuPDF (`fitz`)**, the one allowed dependency for this (installed via pip; document it like any tooling).

- **Where images live:** `wiki/assets/<source-slug>/<figure-slug>.png` (e.g. `wiki/assets/deepseek-v32/fig7-mha-mqa-mode.png`). Slugs in English kebab-case like everything else. `wiki/assets/` **is version-controlled** (images ship with the wiki); only `raw/` PDFs stay git-ignored.
- **How to extract:** most paper diagrams are **vector-drawn**, so `page.get_images()` returns empty — render a clipped region with `page.get_pixmap(matrix=fitz.Matrix(300/72,300/72), clip=rect)` at ~300 DPI. Find the crop box from text-block positions (caption bottom, preceding paragraph), and **verify the box with `page.get_textbox(clip)`** — the returned in-figure labels confirm you captured the whole figure and didn't bleed the caption or neighbouring body text. Do **not** trust `page.get_drawings()` bbox blindly; it can include off-page helper paths.
- **Footnote bleed:** a figure at the top of a page may inherit footnotes from the previous page (signals: `Project Page:`, superscript markers like `^a`, separator lines `───`). After rendering, check `get_textbox(clip)` for these signals; if present, tighten `clip.y0` downward and re-render. Do NOT rely on `get_drawings()` or `cluster_drawings()` to auto-exclude footnotes — their background boxes are vector paths too.
- **Tables:** if a table is plain text (few/no vector lines), **re-typeset it as a Markdown table**, don't screenshot it — keeps it `rg`-searchable and lets formulas render as LaTeX. Screenshot a table only when its layout itself carries meaning.
- **Alt text = the caption.** Write the `![…]()` alt text as a full reader-facing caption (what the figure shows, traced through the diagram), so the page degrades gracefully if the image fails to load. Follow it with a blockquote giving the paper's own caption + a reader-facing locator (`§ A. MHA and MQA Modes of MLA`).
- **Evidence tier:** an embedded `raw/` figure is **tier-1 原文确证** (it *is* the primary source). If a vision model read the figure to help you describe it, that reading is an aid — the figure itself is the evidence, but don't assert a mechanism the pixels don't actually show. Keep `vision_analyze` provenance out of reader-facing prose (it's a tooling trace; note it in the log if relevant).
- **Don't orphan assets:** every file under `wiki/assets/` must be referenced by ≥1 page; a page that cites a figure it could embed should embed it.

#### Vision-assisted figure triage（视觉辅助初筛——必要时触发）

默认做法是自己读 PDF、判断哪些图值得 embed。只在以下**明确痛点**时，才调用 `vision_analyze` 做初筛或辅助理解：

| 触发条件 | 为什么需要 VLM | 输出怎么用 |
|---------|--------------|----------|
| 1. **扫描版/无文本层 PDF** | 无法 `pdftotext` 提取正文和 caption | OCR + VLM 读图 → 人工校验 → 按正常流程 embed |
| 2. **论文图数量 >10 张或密集多子图拼接**（如 Figure 3a-f 跨 2 页） | 人工翻页判断"哪张是机制图、哪张只是消融曲线"成本高 | VLM 输出每张图的内容标签（architecture / ablation / data / qualitative）→ 人工确认后只 embed 机制/ headline-result 图 |
| 3. **图的 caption 与正文描述矛盾** | 需要独立验证图内标签是否与正文一致 | VLM 读图 → 标出矛盾点 → 写入 `## 待追问` 或正文 blockquote，不直接采信 VLM 结论 |
| 4. **复杂流程图信息过载**（单图含 ≥5 个交互模块 + 多色数据流） | 需要理解图内各模块的边界和交互关系才能写准 prose | VLM 辅助标注模块名和数据流 → 人工对照 PDF 正文校验 → 写入 prose |

**约束：** `vision_analyze` 的输出只能当**阅读辅助**，不能当证据（不写入正文作为已核实事实）；provenance 记在 `log.md` 里，不暴露给读者。不满足上述条件时，禁止调用 VLM 初筛。

#### Synthetic diagrams（合成图——必要时触发）

除了嵌入论文原图，有时需要**画一张简化概念图**来补全知识结构。默认不画，只在以下情况触发：

| 触发条件 | 画什么 | 工具 | 输出位置 |
|---------|------|------|---------|
| 1. **概念页解释多阶段流水线，且已有 ≥3 个来源报告描述同一流程但视角不同**（如 MOPD = MiMo 说 specialization-then-integration / GLM-5 说 cross-stage distillation / Mach-Mind 说统一 RL/OPD loss） | 一张抽象层流水线图，剥离各报告的具体命名差异，展示通用阶段和决策点 | Excalidraw `.excalidraw` 或 `creative/architecture-diagram` 的 dark SVG | `wiki/assets/<concept-slug>/` |
| 2. **来源页嵌入的原图信息过载**（单图 ≥5 个子模块 + 多色数据流），且 prose 已写但读者仍难建立空间直觉 | 一张删减版概念图，保留核心模块和主干数据流，去掉细枝末节 | 同上 | `wiki/assets/<source-slug>/` |
| 3. **比较页需要并列展示两个架构异同**，文字表格已写但缺乏空间对照 | 一张左右对照架构图，或统一抽象层下的差异标注图 | 同上 | `wiki/assets/<comparison-slug>/` |

**约束：** 合成图是**推断层（tier-3）**的辅助材料，必须在图注或正文中明确标注"本图为根据多报告综合绘制的概念示意，非论文原图"；禁止把合成图当 tier-1 证据引用。不满足上述条件时，禁止生成合成图。
