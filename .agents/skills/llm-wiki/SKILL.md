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

   > **Vision-assisted content extraction (trigger only when needed):** Default to PyMuPDF text extraction + LLM analysis. Only call `vision_analyze` on pages that hit one of the pain points below:
   >
   > | Trigger | Why VLM is needed | How to use it |
   > |---------|-----------------|---------------|
   > | **Dense-table pages** (≥3 data tables, or cross-page / merged-cell complex tables) | `pdftotext` / `page.get_text()` often misaligns columns and breaks structure | VLM reads the rendered image → outputs a structured table draft → human validation → convert to Markdown |
   > | **Formula-heavy pages** (≥5 lines of LaTeX or complex matrix / tensor ops) | Text extraction may drop symbols, subscripts, superscripts, or bracket nesting | VLM visually confirms the key formula → human validation → write into prose |
   > | **Scanned or mixed-layout pages** (OCR text order scrambled, caption detached from figure) | Pure text loses spatial-layout signals | VLM understands the visual layout → outputs figure-text relationships → human organizes into prose |
   > | **Cross-modal pages** (architecture diagram + adjacent explanatory text) | Pure text extraction severs the figure-text joint signal (e.g. "see Figure 2, we adopt…" but Figure 2 content is empty in text) | VLM reads figure + neighboring paragraphs jointly → extracts design motivation → human validates and writes |
   >
   > **Constraint:** Normal prose-heavy pages (no dense tables / formulas / figures) stay on LLM + PyMuPDF text extraction, no VLM. VLM output is **reading aid only**, never evidence-tier; provenance goes to `log.md`, never reader-facing prose. See `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" → "Vision-assisted figure triage" / "Synthetic diagrams".

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

**Vision and synthetic diagrams are triggered on demand:** `vision_analyze` and Excalidraw / architecture-diagram generation are not default steps. Only call VLM for scanned PDFs, >10 figures, caption-body contradictions, or information-overload workflow diagrams; only draw synthetic diagrams when ≥3 sources describe the same multi-stage pipeline, the original figure is overloaded, or a comparison page needs spatial contrast. See `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" → "Vision-assisted figure triage" / "Synthetic diagrams".

See `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" for the full convention.
