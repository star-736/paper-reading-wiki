# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo is a **Markdown-only knowledge base**, not software. There is no build, lint, or test toolchain — `rg --files` lists tracked files and `rg "term" wiki/ raw/` searches notes. The repo is browsed as an Obsidian vault (`.obsidian/` is committed), so links must stay Obsidian-friendly.

## Architecture: three layers

The wiki implements the LLM-Wiki pattern documented in `.agents/skills/llm-wiki/references/llm-wiki.md`. Internalize the pattern before doing non-trivial maintenance — the workflows below depend on it.

1. **`raw/`** — immutable source PDFs (papers, technical reports). Source-of-truth inputs. **Never modify.** Filenames here are heterogeneous (e.g. `glm-5-2602.15763.pdf`, `Bai 等 - 2026 - IndexCache ...pdf`); leave them as-is.
2. **`wiki/`** — LLM-generated, interlinked Markdown. Owned entirely by the agent. Subdirectories are typed:
   - `wiki/sources/<slug>.md` — one page per raw document (origin, claims, links to model and concepts).
   - `wiki/models/<slug>.md` — entity pages for individual models.
   - `wiki/concepts/<slug>.md` — topic / mechanism pages cited from multiple sources.
   - `wiki/comparisons/<slug>.md` — cross-source synthesis pages.
   - `wiki/index.md` — content-oriented catalog (read this first when answering queries).
   - `wiki/log.md` — append-only chronological record.
3. **Schema** — `AGENTS.md` (Codex-facing) and this `CLAUDE.md` describe the conventions and workflows. The skill at `.agents/skills/llm-wiki/SKILL.md` is the same pattern condensed.

A single ingest typically touches 5–15 wiki pages across these subdirectories in one pass — that fan-out is the point.

## Workflows

**Ingest (new PDF in `raw/`):**
1. Read the PDF, identify model entities, key claims, mechanisms, contradictions.
2. Create `wiki/sources/<slug>.md` and (if a new model) `wiki/models/<slug>.md`.
3. Update or create relevant `wiki/concepts/*.md` pages — append cross-source signals to existing concept pages rather than forking new ones when the topic already exists.
4. Update `wiki/comparisons/*.md` if the new source belongs in an existing comparison.
5. Add the new pages to `wiki/index.md` under the right section, with a one-line Chinese summary.
6. Append to `wiki/log.md` with a heading like `## [YYYY-MM-DD] ingest | Source Title`.

Prefer one source at a time. Two PDFs currently sit in `raw/` without wiki pages — `Bai 等 - 2026 - IndexCache ...pdf` and `Lai 等 - 2026 - MiniMax sparse attention.pdf` — and are candidates for the next ingest.

**Query:** read `wiki/index.md` first, drill into linked pages, cite wiki pages or `raw/` files in answers. When the synthesis is durable, offer to file it back as a new page (often under `wiki/comparisons/` or `wiki/concepts/`).

**Lint:** check for stale claims, contradictions, orphan pages, missing cross-references, missing source citations, and concepts mentioned without their own page.

## Conventions specific to this wiki

- **Language split:** page **content is written in Chinese** (see existing pages); **filenames, directory names, and slugs stay in English kebab-case** (e.g. `deepseek-sparse-attention.md`). Technical terms and acronyms (MoE, DSA, MTP, RL, SWE-bench) stay in English inside Chinese prose. The 2026-06-06 `maintenance` log entry locks this in — do not re-translate filenames.
- **Page skeleton (sources):** `## 来源` (with PDF path, title, version/date, team, model link) → `## 核心结论` → `## 架构与训练` → `## 后训练` → `## 评测要点` → `## 待追问`. Match this when adding new source pages.
- **Page skeleton (concepts):** `## 定义` → `## 跨报告信号` → `## 为什么重要` → `## 待追问` → `## 相关页面`.
- **Cross-links use relative Markdown paths** (e.g. `[GLM-5](../models/glm-5.md)`), not Obsidian `[[wiki-links]]`. Keep this style consistent within a page; the existing wiki uses Markdown links throughout.
- **Log entries** start with `## [YYYY-MM-DD] <kind> | <title>` where `<kind>` is one of `ingest`, `maintenance`, `deepen`, etc. Keep the prefix parseable so `rg "^## \[" wiki/log.md` still works.
- **Section "待追问"** (open questions) is a real part of the schema, not boilerplate — populate it with concrete follow-ups when ingesting.

## What not to do

- Don't overwrite anything in `raw/`.
- Don't introduce a build system, package manifest, or test runner — `AGENTS.md` says to document tooling exactly when it's added; today there is none.
- Don't translate filenames to Chinese or rename existing English slugs (breaks links and `rg` workflows).
- Don't add a wiki page in isolation: if you create a source page without updating `index.md`, `log.md`, and at least one concept/comparison cross-reference, the ingest is incomplete.
