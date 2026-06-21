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

See `AGENTS.md` / `CLAUDE.md` → "Figures & visual material" for the full convention.
