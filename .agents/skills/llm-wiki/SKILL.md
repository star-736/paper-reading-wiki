---
name: llm-wiki
description: Use when creating, ingesting into, querying, or linting a persistent Markdown wiki maintained by an LLM from raw source documents. Do not use for ordinary documentation edits that are unrelated to knowledge-base maintenance.
---

# LLM Wiki

This skill guides Codex in building and maintaining a persistent Markdown wiki from raw sources. The full pattern is in `references/llm-wiki.md`; read it when the user asks to initialize a wiki, design the schema, or make workflow decisions.

## Core Model

Maintain three layers:

- `raw/`: immutable source documents, articles, papers, images, or data files.
- Wiki pages: generated Markdown summaries, entity pages, topic pages, comparisons, and synthesis pages.
- Schema/guidance: `AGENTS.md` plus any wiki-specific conventions that describe structure and workflows.

Do not overwrite raw sources. Treat them as the source of truth.

## Ingest Workflow

When processing a new source:

1. Read the source and identify key claims, entities, concepts, contradictions, and useful citations.

   > **Vision-assisted content extraction（必要时触发）：** 默认用 PyMuPDF 提取文本 + LLM 分析。只在以下痛点页面临时调 `vision_analyze` 辅助识别：
   >
   > | 触发条件 | 为什么需要 VLM | 怎么用 |
   > |---------|--------------|--------|
   > | **表格密集页**（≥3 个数据表，或跨页/合并单元格复杂表） | `pdftotext` / `page.get_text()` 常列错位、结构崩坏 | VLM 读原图 → 输出结构化表格草稿 → 人工校验 → 转 Markdown |
   > | **公式推导页**（连续 ≥5 行 LaTeX 或复杂矩阵/张量运算） | 文本提取可能丢符号、下标、上标、括号层级 | VLM 视觉确认关键公式 → 人工校验 → 写入正文 |
   > | **扫描版或图文混排页**（OCR 后文本顺序错乱、caption 与图分离） | 纯文本丢失空间布局信号 | VLM 按视觉布局理解 → 输出图文关系 → 人工组织 prose |
   > | **架构图 + 旁边解释文字的跨模态页** | 纯文本提取割裂"图-文"联合信号（如"见图 2，我们采用…"但图 2 的内容在文本里为空） | VLM 联合读图和相邻段落 → 提取设计动机 → 人工校验写入 |
   >
   > **约束：** 正常 prose-heavy 页（无密集表格/公式/图）继续用 LLM + PyMuPDF 文本提取，不调 VLM。VLM 输出只能当"阅读辅助"，不写入正文作为已核实事实；provenance 记 `log.md`。详见 `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" → "Vision-assisted figure triage" / "Synthetic diagrams"。

2. Create or update the relevant wiki pages.
3. Add cross-references between related pages.
4. Update `index.md` with page links and one-line summaries.
5. Append a dated entry to `log.md`, using a heading like `## [2026-06-06] ingest | Source Title`.

Prefer one source at a time unless the user explicitly asks for batch ingest.

## Query Workflow

For questions about the knowledge base, read `index.md` first when it exists, then inspect relevant pages and raw sources. Answer with citations to wiki pages or raw files. When the answer produces durable synthesis, offer to file it back into the wiki as a new or updated page.

## Lint Workflow

When asked to audit or health-check the wiki, look for stale claims, contradictions, orphan pages, missing cross-references, missing source citations, important concepts without pages, and gaps that need new sources.

## Figures (图文化)

The wiki embeds source figures inline, not just text. When a page relies on a paper's diagram/table a reader can't reconstruct from prose, embed the image rather than only citing `Figure N`. Use **PyMuPDF (`fitz`)**:

- Store at `wiki/assets/<source-slug>/<figure-slug>.png`; `wiki/assets/` is committed (only `raw/` is git-ignored). Don't orphan assets.
- Paper diagrams are usually vector-drawn (`page.get_images()` empty) — render a clipped region via `page.get_pixmap(matrix=fitz.Matrix(300/72,300/72), clip=rect)` and verify the crop with `page.get_textbox(clip)`.
- Plain-text tables: re-typeset as Markdown, don't screenshot.
- Alt text = a full reader-facing caption; an embedded `raw/` figure is tier-1 原文确证. Keep vision-tool provenance out of reader-facing prose.

**Vision 与合成图按需触发：** `vision_analyze` 和 Excalidraw/架构图生成不是默认步骤。只在扫描版 PDF、图数量 >10、caption-正文矛盾、信息过载流程图等明确痛点时才调 VLM；只在 ≥3 个来源描述同一多阶段流水线、原图信息过载、比较页需空间对照时才画合成图。详见 `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" → "Vision-assisted figure triage" / "Synthetic diagrams"。

See `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" for the full convention.
