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

Write Markdown in concise sections with descriptive headings. Prefer sentence-case headings for article pages unless a proper noun requires capitalization. Use kebab-case filenames, for example `retrieval-augmented-generation.md`. Keep links relative and Obsidian-friendly, using either Markdown links or wiki links consistently within a page.

For logs, use parseable dated headings such as `## [2026-06-06] ingest | Source Title`.

## Testing Guidelines

There are no automated tests. Validate contributions manually by checking:

- Markdown renders without broken headings or malformed links.
- New source-derived claims cite or reference the relevant raw source.
- `wiki/index.md` and `wiki/log.md` are updated when pages are added or revised.

## Commit & Pull Request Guidelines

No meaningful commit-message convention is available in the root workspace history. Use short, imperative messages with a scope when helpful, such as `docs: add wiki index` or `raw: add transformer survey`.

Pull requests should describe what changed, list added or processed sources, note updated wiki pages, and call out unresolved contradictions or follow-up research. Include screenshots only when changes affect rendered diagrams, slides, or visual assets.

## Agent-Specific Instructions

Agents should read `wiki/index.md` first when it exists, then inspect relevant pages and raw sources. Do not overwrite raw files. When ingesting a source, update summaries, cross-references, the index, and the chronological log in the same pass.
