# Agent Instructions

This repository uses GSD planning artifacts under `.planning/`.

## Project Context

Before planning or editing, read:

- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- Relevant files under `.planning/codebase/`

## Current Priority

Phase 4 is the active phase: turn the repo into a credible public landing page and contributor-ready project, including a full AES mode matrix.

Do not prioritize SEO copy, benchmark claims, or release polish before build reproducibility and correctness are trustworthy.

## Engineering Rules

- Keep the project positioned as a reproducible benchmark suite, not a production cryptography library.
- Preserve the roadmap direction toward full practical AES mode coverage, not only ECB/CTR/GCM.
- Avoid unsupported performance claims.
- Preserve benchmark credibility by separating correctness, methodology, raw data, and summary claims.
- Prefer portable CMake and documented configuration over local absolute paths.
- Keep generated build artifacts out of source changes.
