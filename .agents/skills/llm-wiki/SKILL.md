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
