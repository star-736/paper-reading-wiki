# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo is a **Markdown-only knowledge base**, not software. There is no build, lint, or test toolchain — `rg --files` lists tracked files and `rg "term" wiki/ raw/` searches notes. It can be browsed as an Obsidian vault, but `.obsidian/` is **git-ignored** (per the 2026-06-19 maintenance entry), so don't rely on vault settings being shared; links use relative Markdown paths, not Obsidian `[[wiki-links]]`. The one tooling dependency is **PyMuPDF (`fitz`)**, used to extract figures from `raw/` PDFs for embedding (see "Figures & visual material").

## Architecture: three layers

The wiki implements the LLM-Wiki pattern documented in `.agents/skills/llm-wiki/references/llm-wiki.md`. Internalize the pattern before doing non-trivial maintenance — the workflows below depend on it.

1. **`raw/`** — immutable source PDFs (papers, technical reports). Source-of-truth inputs. **Never modify.** Filenames here are heterogeneous (e.g. `glm-5-2602.15763.pdf`, `Bai 等 - 2026 - IndexCache ...pdf`); leave them as-is.
2. **`wiki/`** — LLM-generated, interlinked Markdown. Owned entirely by the agent. Subdirectories are typed:
   - `wiki/sources/<slug>.md` — one page per raw document (origin, claims, links to model and concepts).
   - `wiki/models/<slug>.md` — entity pages for individual models.
   - `wiki/concepts/<slug>.md` — topic / mechanism pages cited from multiple sources.
   - `wiki/comparisons/<slug>.md` — cross-source synthesis pages.
   - `wiki/index.md` — content-oriented catalog (read this first when answering queries).
   - `wiki/log.md` — append-only chronological record (what already happened).
   - `wiki/TODO.md` — checklist of unfinished engineering follow-ups (what's next). Keep planned work here, never in `log.md`; remove an item when done and log the completed action.
3. **Schema** — `CLAUDE.md` (this file) and `AGENTS.md` (Codex-facing) carry the **same conventions** in tool-specific form; either is self-sufficient, so a tool reading only one still gets the full picture. When you change a shared convention, update **both** so they don't drift. The skill at `.agents/skills/llm-wiki/SKILL.md` condenses the same pattern for skill-triggered loading.

A single ingest typically touches 5–15 wiki pages across these subdirectories in one pass — that fan-out is the point.

## Workflows

Each workflow ends by appending a `## [YYYY-MM-DD] <kind> | <title>` entry to `wiki/log.md`. The `<kind>` vocabulary is fixed — see "Log discipline" below.

**Ingest (`ingest`) — a new PDF landed in `raw/`:** the output is **图文交错 (text + inline figures)** pages, not a text-only summary.
1. Read the PDF, identify model entities, key claims, mechanisms, contradictions, **and the figures/tables that carry mechanism or headline-result content**.
2. Create `wiki/sources/<slug>.md` and (if a new model) `wiki/models/<slug>.md`.
3. Update or create relevant `wiki/concepts/*.md` pages — append cross-source signals to existing concept pages rather than forking new ones when the topic already exists.
4. **Embed the key figures** (per "Figures & visual material"): extract each mechanism/architecture/headline diagram with PyMuPDF into `wiki/assets/<source-slug>/` and embed it on the page that argues it; re-typeset plain-text tables as Markdown. Every ingest should embed ≥1 figure unless the paper has no diagram worth showing — if so, say so explicitly in the log.
5. Update `wiki/comparisons/*.md` if the new source belongs in an existing comparison.
6. Add the new pages to `wiki/index.md` under the right section, with a one-line Chinese summary.
7. Run the **write-back checklist** (below), then append the `ingest` log entry.

Prefer one source at a time unless the user asks for batch ingest.

**Deepen (`deepen`) — extend or sharpen an existing page, usually driven by a user question:**
- Go back to the relevant `raw/` PDF and verify against the actual text before editing — do not deepen from memory of an earlier summary. Every new mechanism claim must trace to the source (see "Evidence discipline").
- Touch only the pages the question reaches; add cross-references both ways. Run the write-back checklist, then log as `deepen`.
- **`deepen` vs `distill`:** `deepen` sharpens content **on a page that already exists**. When the conversation instead surfaces a *new* cross-page insight, comparison, or framing that deserves its own home, that's `distill` (below). A turn can be both — sharpen an old page *and* spin off a new synthesis page; log it under whichever is the headline action.

**Distill (`distill`) — file an insight that emerged from a Q&A conversation back into the wiki:**
The query workflow (below) often surfaces durable knowledge that isn't yet written down anywhere — a connection across two papers, an original framing, a judgment call the user pushed you to sharpen. `distill` is the act of capturing that so it compounds instead of evaporating into chat history. This is the second engine of an LLM-wiki (the first is `ingest`); treat it as first-class, not an afterthought.
- **Sniff test — is it worth a page/section?** File it when the insight is **durable** (not a one-off lookup), **reusable** (a future question will want it), and **synthesizing** (it connects ≥2 sources/pages, or states a framing no single source spells out). A pure one-shot fact lookup ("how many tokens did V3.2 train on") is *not* distilled — just answer it. The MLA concept page is the archetype: it was born from the question "is DSA evolved from MLA", which no existing page answered.
- **Where it lands:**
  - A new cross-source framing or comparison → a new `wiki/concepts/<slug>.md` or `wiki/comparisons/<slug>.md`.
  - An insight that belongs to one existing page but is genuinely new synthesis (not just sharpening) → a new section on that page.
  - An open question with no answer yet → the relevant page's `## 待追问`, not a new page.
- **How to file it:** the same rigor as `ingest`/`deepen` applies — re-verify every claim against `raw/` (don't trust the chat's own phrasing), tag the evidence tier, embed any figure the insight leans on, and **wire cross-references both ways so the new page/section isn't an orphan**. Run the write-back checklist, then log as `distill`.
- Don't let the distilled page silently duplicate an existing one — if the topic already has a page, extend it (that may make the turn a `deepen`) rather than forking a near-twin.

**Verify (`verify`) — cross-check claims already on a page against external sources and re-read the primary PDF:**
- Use this when a claim needs independent corroboration or the user asks "is this actually right". External search is reachable via the Tavily API (`TAVILY_API_KEY` in `~/AppData/Local/hermes/.env`; foreign sources reachable); `web_search` as a native tool is **not** available in this environment, and the `delegate_task` web subagent has been unreliable — don't rely on it.
- Record the verdict per claim (supported / refuted / not-found-externally-but-confirmed-in-primary-source) and downgrade anything that turns out to be inference. Log as `verify`.

**Refactor (`refactor`) — restructure a page without changing its facts:**
- Reorder sections, deduplicate, strip internal tooling traces (`pdftotext`, raw `line NNN` anchors, dated scratch notes) into reader-facing locators. **Facts and citations must not change** — say so explicitly in the log entry. Log as `refactor`.

**Maintenance (`maintenance`) — repo-level housekeeping:** language sweeps, schema-field rollouts across pages, `.gitignore` / version-control changes, link-integrity fixes. Log as `maintenance`.

**Query (no log entry unless it produces a page):** read `wiki/index.md` first, drill into linked pages, cite wiki pages or `raw/` files in answers. **The query loop is where most `distill`-worthy knowledge appears** — while answering, watch for synthesis that passes the distill sniff test, and when you spot it, offer to file it back (a new `wiki/concepts/`/`wiki/comparisons/` page, or a new section on an existing page) per the `distill` workflow. A plain answered question earns no log entry; only the act of filing it back does (logged as `distill`).

**Lint (`maintenance`):** check for stale claims, contradictions, orphan pages, missing cross-references, missing source citations, and concepts mentioned without their own page.

## Evidence discipline

This wiki's core value is that its claims are traceable, not just plausible. Every non-trivial mechanism claim carries an implicit evidence tier — make it explicit when the tiers could be confused:

1. **原文确证 (primary-source confirmed)** — verified against the `raw/` PDF itself. Cite a reader-facing locator (`§ Instantiate DSA Under MLA`, `Figure 7 caption`, `Table 1`), not a raw line number. This is the only tier that may sit unqualified under a "已据原文核实" heading.
2. **外部佐证 (externally corroborated)** — supported by an independent source (Tavily hit, official blog, third-party analysis) but not in `raw/`. Link the provenance; mark it as corroboration, not primary truth. Blog explanations of an author's *intent* are second-hand and must not be promoted to tier 1.
3. **推断 / 本页原创综合 (inference / original synthesis)** — your own reasoning connecting confirmed facts, or a framing no source states outright. **Label it as such**, and when it's a guess about mechanism, park the open question in `## 待追问` rather than asserting it in the body.

Rules that follow from this:
- **Never blend a paper's conclusion with your inference of its mechanism in one sentence** under a "confirmed" heading — that's the exact failure the MLA-query-recompute correction fixed. Split them.
- When you discover a past entry over-claimed, **downgrade it honestly** (move to 待追问, add a 推测 marker) rather than hunting for weak corroboration to prop it up.
- Re-reading the primary PDF beats trusting an earlier wiki summary. Earlier passes can be wrong; the `raw/` file can't.
- `raw/` is never modified by any workflow — say "`raw/` 未改" in the log entry as a standing reassurance.

## Write-back checklist

Before considering any `ingest`/`deepen`/`verify`/`refactor` turn done, confirm:
- [ ] New/changed claims trace to the right evidence tier (primary PDF re-read where it matters).
- [ ] Cross-references added **both ways** (new page links out; pages it relates to link back).
- [ ] `wiki/index.md` updated if a page was added or its one-line summary changed.
- [ ] `wiki/log.md` appended with the correct `<kind>` and a title.
- [ ] No orphan page (every new page has ≥1 inbound link), no broken relative links.
- [ ] Internal tooling traces (raw line numbers, `pdftotext`, scratch dates) kept out of reader-facing prose.
- [ ] Figures a page relies on are **embedded** (not just cited as `Figure N`) per "Figures & visual material"; every `wiki/assets/` file is referenced by ≥1 page.

**Wrap-up — offer to commit & push:** this wiki is a git repo (`origin` on GitHub). After a turn has produced **substantive on-disk changes** (an ingest, a deepen, a correction, or any edit that touched `wiki/` and updated `log.md`), proactively offer — in one short line — to `git commit` and `git push` before ending the turn. Don't wait to be asked. Keep it a lightweight offer, not a forced step: skip it for pure read/query turns, and if the user is mid-discussion (still asking follow-ups, hasn't asked to wrap up), batch the changes and offer once at a natural stopping point rather than after every individual edit. When the user accepts, stage only the files changed this session, write an imperative `docs:`-scoped message, and push to `main`.

## Conventions specific to this wiki

- **Language split:** page **content is written in Chinese** (see existing pages); **filenames, directory names, and slugs stay in English kebab-case** (e.g. `deepseek-sparse-attention.md`). Technical terms and acronyms (MoE, DSA, MTP, RL, SWE-bench) stay in English inside Chinese prose. The 2026-06-06 `maintenance` log entry locks this in — do not re-translate filenames.
- **Page skeleton (sources):** `## 来源` (with PDF path, title, version/date, team, model link) → `## 核心结论` → `## 架构与训练` → `## 后训练` → `## 评测要点` → `## 待追问`. Match this when adding new source pages.
- **Page skeleton (concepts):** `## 定义` → `## 跨报告信号` → `## 为什么重要` → `## 待追问` → `## 相关页面`.
- **Page skeleton (models):** `## 身份` → `## 关键事实`（Markdown 表）→ explanation/技术身份 → `## 相关页面`. The `## 关键事实` table **must include a `模态` row** (e.g. 纯文本 / 多模态（文本 + 图像 + 视频）). When a model is a multi-variant family with a variant table instead (e.g. `deepseek-v4.md`), put modality as a `**模态**：…` line right below the table. Mark modality as 已核实 only after checking the source report — incidental "multimodal/vision" mentions in reports are usually about RL-pipeline verifiers, eval benchmarks, or future-work outlooks, not the model's own input.
- **Cross-links use relative Markdown paths** (e.g. `[GLM-5](../models/glm-5.md)`), not Obsidian `[[wiki-links]]`. Keep this style consistent within a page; the existing wiki uses Markdown links throughout.
- **External source citations use plain inline Markdown links**, written as `（来源：[标题](url)）` (or an inline `[标题](url)` mid-sentence). **Do not use `^[url 标题]` or `[^url 标题]`** — `^[...]` is a Pandoc-only inline-footnote extension that GitHub/Obsidian don't render, and `[^...]` is a footnote *reference* that needs a short identifier plus a matching `[^id]: …` definition elsewhere; stuffing a URL into either renders as garbled brackets (the 2026-06-21 maintenance fix). Also avoid bare square brackets like `prop[erly]` or `continued [training]` in prose: `[…]` immediately followed by text/parens gets parsed as link syntax and bleeds link styling. Quote source text verbatim (`properly`, not `prop[erly]`); if you must gloss an elided word, use a Chinese parenthetical （训练）, not `[brackets]`.
- **Log entries** start with `## [YYYY-MM-DD] <kind> | <title>`. `<kind>` is a **fixed vocabulary**: `ingest` (new `raw/` source filed), `deepen` (existing page extended/sharpened), `distill` (a Q&A insight filed back as a new page/section), `verify` (claims cross-checked against external + primary source), `refactor` (page restructured, facts unchanged), `maintenance` (repo housekeeping / lint / schema rollouts). Keep the prefix parseable so `rg "^## \[" wiki/log.md` still works. See "Workflows" for what each kind does.
- **Log discipline — keep the log a timeline, not a second copy of the page.** Each entry says *what changed, which files, and the headline reason* — a few lines, not the full argument. The detailed reasoning (a derivation, a quote-by-quote verification, a corrected misconception) belongs **on the page it concerns** (often in `## 待追问` or an inline blockquote), with the log pointing to it. If a log entry is growing past ~8 lines, that's a signal the substance should live on the page instead. Always note `raw/` 未改 when no source was touched.
- **Section "待追问"** (open questions) is a real part of the schema, not boilerplate — populate it with concrete follow-ups when ingesting.

## Figures & visual material (图文化)

The wiki is **图文交错 (text + inline figures)**, not text-only. When a page leans on a paper's figure or table — especially a mechanism/architecture diagram a reader can't reconstruct from prose — embed the actual image, don't just cite `Figure N`. The tool for this is **PyMuPDF (`fitz`)** (pip-installed; it's the repo's one tooling dependency).

- **Asset location:** `wiki/assets/<source-slug>/<figure-slug>.png` (e.g. `wiki/assets/deepseek-v32/fig7-mha-mqa-mode.png`), English kebab-case slugs. `wiki/assets/` **is committed** — figures ship with the wiki; only `raw/` PDFs are git-ignored. Never orphan an asset: every file under `wiki/assets/` must be referenced by ≥1 page.
- **Extraction:** paper diagrams are usually **vector-drawn**, so `page.get_images()` is empty — render a clipped region: `page.get_pixmap(matrix=fitz.Matrix(300/72,300/72), clip=rect)` at ~300 DPI. Derive the crop box from text-block coordinates (caption bottom, preceding paragraph) and **confirm it with `page.get_textbox(clip)`**: the in-figure labels it returns prove you grabbed the whole figure without bleeding the caption/body text. `page.get_drawings()` bbox can include off-page helper paths — don't trust it blindly.
- **Tables:** plain-text tables (few/no vector lines) → **re-typeset as a Markdown table**, don't screenshot (keeps them `rg`-searchable, formulas render as LaTeX). Screenshot a table only when its visual layout itself carries meaning.
- **Alt text = caption:** write the `![…]()` alt text as a full reader-facing caption (trace the diagram), so the page degrades gracefully if the image 404s. Follow with a blockquote carrying the paper's own caption + a reader-facing locator.
- **Evidence tier:** an embedded `raw/` figure is **tier-1 原文确证** — it *is* the primary source. A `vision_analyze` reading is only an aid to describe it; don't assert a mechanism the pixels don't show, and keep vision-tool provenance out of reader-facing prose (it's a tooling trace; mention in the log if useful).

## What not to do

- Don't overwrite anything in `raw/`.
- Don't introduce a build system, package manifest, or test runner. The sole tooling dependency is PyMuPDF (`fitz`) for figure extraction; document any further tooling exactly when added.
- Don't translate filenames to Chinese or rename existing English slugs (breaks links and `rg` workflows).
- Don't add a wiki page in isolation: if you create a source page without updating `index.md`, `log.md`, and at least one concept/comparison cross-reference, the ingest is incomplete.
