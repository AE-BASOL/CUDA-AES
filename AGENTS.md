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

Phase 6 execution is complete at source level: CCM, XTS-AES, AES-KW, and AES-KWP have AES-128/AES-256 source, KAT, benchmark dispatch, and documentation coverage. GMAC and CMAC are documented as authentication/MAC-only boundaries, not encryption throughput modes.

Runtime CMake/CTest remains blocked in this shell until `nvcc` can find `cl.exe`. Close this verification debt from a Visual Studio Developer Command Prompt or by passing `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`.

Next priority is Phase 7: discoverability and SEO for CUDA AES / GPU AES benchmark searches, after preserving the Phase 6 runtime verification caveat.

Do not prioritize SEO copy, benchmark claims, or release polish before build reproducibility and correctness are trustworthy.

## Engineering Rules

- Keep the project positioned as a reproducible benchmark suite, not a production cryptography library.
- Preserve the roadmap direction toward full practical AES mode coverage, not only ECB/CTR/GCM.
- Avoid unsupported performance claims.
- Preserve benchmark credibility by separating correctness, methodology, raw data, and summary claims.
- Prefer portable CMake and documented configuration over local absolute paths.
- Keep generated build artifacts out of source changes.
