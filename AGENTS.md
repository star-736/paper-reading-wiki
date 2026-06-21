# Repository Guidelines

## Project Structure & Module Organization

This repository is a lightweight Markdown knowledge-base workspace.

- `raw/` stores source materials before they are processed. Treat these files as source-of-truth inputs and avoid modifying them during wiki maintenance.
- `wiki/` contains generated knowledge-base pages. Start at `wiki/index.md`; append workflow history to `wiki/log.md`.
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

For logs, use parseable dated headings such as `## [2026-06-06] ingest | Source Title`, where the kind comes from the fixed `<kind>` vocabulary (see "Agent-Specific Instructions" below).

## Testing Guidelines

There are no automated tests. Validate contributions manually against the **write-back checklist** in "Wiki workflows & writing discipline" below:

- Markdown renders without broken headings or malformed relative links.
- New source-derived claims trace to the right evidence tier — re-read the `raw/` PDF for mechanism claims rather than trusting an earlier summary. Keep 原文确证 / 外部佐证 / 推断 distinct, and never assert your own mechanism guess under a "已据原文核实" heading.
- Cross-references go both ways; no orphan pages.
- `wiki/index.md` and `wiki/log.md` are updated when pages are added or revised.

## Commit & Pull Request Guidelines

No meaningful commit-message convention is available in the root workspace history. Use short, imperative messages with a scope when helpful, such as `docs: add wiki index` or `raw: add transformer survey`.

After a turn has produced substantive on-disk changes (an ingest, a deepen, a correction — anything that edited `wiki/` and appended to `wiki/log.md`), proactively offer in one short line to commit and push before ending the turn; don't wait to be asked. Keep it a lightweight offer: skip it for pure read/query turns, and when the user is still mid-discussion, batch the edits and offer once at a natural stopping point rather than after every individual change.

Pull requests should describe what changed, list added or processed sources, note updated wiki pages, and call out unresolved contradictions or follow-up research. Include screenshots only when changes affect rendered diagrams, slides, or visual assets.

## Agent-Specific Instructions

Agents should read `wiki/index.md` first when it exists, then inspect relevant pages and raw sources. Do not overwrite raw files. When ingesting a source, update summaries, cross-references, the index, and the chronological log in the same pass.

`AGENTS.md` and `CLAUDE.md` carry the **same conventions** in tool-specific form; either is self-sufficient. When you change a shared convention, update both so they stay in sync.

## Wiki workflows & writing discipline

### Workflows

Each workflow ends by appending a `## [YYYY-MM-DD] <kind> | <title>` entry to `wiki/log.md`.

- **`ingest` — a new PDF landed in `raw/`:** (1) read the PDF, identify model entities, key claims, mechanisms, contradictions; (2) create `wiki/sources/<slug>.md` and, if a new model, `wiki/models/<slug>.md`; (3) update or create relevant `wiki/concepts/*.md` — append cross-source signals to existing concept pages rather than forking new ones; (4) update `wiki/comparisons/*.md` if it belongs in an existing comparison; (5) add the new pages to `wiki/index.md` with a one-line Chinese summary; (6) run the write-back checklist, then append the log entry. One source at a time unless asked for batch. A single ingest typically touches 5–15 pages — that fan-out is the point.
- **`deepen` — extend/sharpen existing pages, usually from a user question:** re-read the relevant `raw/` PDF and verify against the actual text first — never deepen from memory of an earlier summary. Touch only the pages the question reaches; add cross-references both ways.
- **`verify` — cross-check existing claims against external sources and re-read the primary PDF:** external search is reachable via the Tavily API (`TAVILY_API_KEY` in `~/AppData/Local/hermes/.env`; foreign sources reachable); `web_search` as a native tool is **not** available here, and the `delegate_task` web subagent has been unreliable. Record the verdict per claim (supported / refuted / not-found-externally-but-confirmed-in-primary-source) and downgrade anything that turns out to be inference.
- **`refactor` — restructure a page without changing its facts:** reorder, deduplicate, strip internal tooling traces into reader-facing locators. **Facts and citations must not change** — say so in the log entry.
- **`maintenance` — repo-level housekeeping:** language sweeps, schema-field rollouts, `.gitignore` / version-control changes, link fixes, lint passes (stale claims, contradictions, orphan pages, missing cross-references or citations).
- **Query (no log entry unless it produces a page):** read `wiki/index.md` first, drill into linked pages, cite wiki pages or `raw/` files. Offer to file durable synthesis back as a page — only then does it earn a log entry.

`<kind>` is a **fixed vocabulary**: `ingest`, `deepen`, `verify`, `refactor`, `maintenance`. Heading form `## [YYYY-MM-DD] <kind> | <title>`; keep it parseable so `rg "^## \[" wiki/log.md` works.

### Evidence discipline

This wiki's core value is that claims are traceable, not just plausible. Every non-trivial mechanism claim carries an evidence tier — make it explicit when tiers could be confused:

1. **原文确证 (primary-source confirmed)** — verified against the `raw/` PDF itself. Cite a reader-facing locator (`§ Instantiate DSA Under MLA`, `Figure 7 caption`, `Table 1`), not a raw line number. Only this tier may sit unqualified under a "已据原文核实" heading.
2. **外部佐证 (externally corroborated)** — supported by an independent source (Tavily hit, official blog, third-party analysis) but not in `raw/`. Link the provenance; mark it as corroboration. Blog explanations of an author's *intent* are second-hand and must not be promoted to tier 1.
3. **推断 / 本页原创综合 (inference / original synthesis)** — your own reasoning or a framing no source states outright. **Label it**, and when it's a mechanism guess, park it in `## 待追问` rather than asserting it in the body.

Rules: never blend a paper's conclusion with your inference of its mechanism in one sentence under a "confirmed" heading; when a past entry over-claimed, downgrade it honestly rather than propping it up with weak corroboration; re-reading the primary PDF beats trusting an earlier summary; `raw/` is never modified — say "`raw/` 未改" in the log when no source was touched.

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
- **Tables:** if a table is plain text (few/no vector lines), **re-typeset it as a Markdown table**, don't screenshot it — keeps it `rg`-searchable and lets formulas render as LaTeX. Screenshot a table only when its layout itself carries meaning.
- **Alt text = the caption.** Write the `![…]()` alt text as a full reader-facing caption (what the figure shows, traced through the diagram), so the page degrades gracefully if the image fails to load. Follow it with a blockquote giving the paper's own caption + a reader-facing locator (`§ A. MHA and MQA Modes of MLA`).
- **Evidence tier:** an embedded `raw/` figure is **tier-1 原文确证** (it *is* the primary source). If a vision model read the figure to help you describe it, that reading is an aid — the figure itself is the evidence, but don't assert a mechanism the pixels don't actually show. Keep `vision_analyze` provenance out of reader-facing prose (it's a tooling trace; note it in the log if relevant).
- **Don't orphan assets:** every file under `wiki/assets/` must be referenced by ≥1 page; a page that cites a figure it could embed should embed it.
