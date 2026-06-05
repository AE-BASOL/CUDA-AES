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

Phase 5 execution is complete at source level: CBC, CFB-128, and OFB have AES-128/AES-256 source, KAT, benchmark, and documentation coverage. Runtime CMake/CTest remains blocked in this shell until `nvcc` can find `cl.exe`.

Next priority is Phase 6: authenticated and specialized mode expansion for CCM, XTS-AES, AES-KW, AES-KWP, and GMAC/CMAC boundary documentation.

Do not prioritize SEO copy, benchmark claims, or release polish before build reproducibility and correctness are trustworthy.

## Engineering Rules

- Keep the project positioned as a reproducible benchmark suite, not a production cryptography library.
- Preserve the roadmap direction toward full practical AES mode coverage, not only ECB/CTR/GCM.
- Avoid unsupported performance claims.
- Preserve benchmark credibility by separating correctness, methodology, raw data, and summary claims.
- Prefer portable CMake and documented configuration over local absolute paths.
- Keep generated build artifacts out of source changes.
