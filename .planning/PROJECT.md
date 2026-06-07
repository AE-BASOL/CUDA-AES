# CUDA-AES Benchmark

## What This Is

CUDA-AES Benchmark is a reproducible open-source benchmark suite for GPU AES implementations. It should become a living, credible repository for CUDA developers who search for GPU AES, CUDA AES benchmark, AES GPU performance, and reproducible cryptography benchmark work.

The v1.0 Public Release matured the project from a personal experiment into a public, trusted, searchable, maintainable repository. It now ships CUDA AES kernels for the full practical confidentiality and specialized mode family — ECB, CBC, CFB, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, and AES-KWP (AES-128 and AES-256) — with a CTest known-answer harness, a reproducible benchmark harness, a public docs hub, open-source governance, and a published `v1.0.0` GitHub Release. GMAC/CMAC are documented as authentication/MAC boundaries rather than bulk-encryption throughput.

## Core Value

Anyone landing on the repository can build it, verify AES correctness, reproduce benchmark results, and understand why the results are credible.

## Current State

**Shipped:** v1.0 Public Release — 2026-06-07 (`v1.0.0` GitHub Release published).

- 8 phases, 23 plans, 42 tasks; 40/40 v1 requirements validated.
- Source + KAT + benchmark dispatch for ECB, CBC, CFB-128, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, AES-KWP (AES-128/256).
- Runtime verified from a Visual Studio 2022 Developer Command Prompt: Release build, CTest (1/1 across implemented modes), and a smoke benchmark all pass.
- Public docs hub, governance files, citation metadata, issue/PR templates, changelog, and a v1.0.0 release with raw benchmark artifacts.
- `master` is protected by the `protect-master` ruleset (linear history, PR-only squash/rebase, required review-thread resolution).

**Next milestone goals (v2.0 candidates):** broader benchmarking (multi-GPU results tables, CUDA-version/arch matrix automation, charts), standalone GMAC/CMAC authentication benchmarking, a documented library/API surface, and publication (GitHub Pages, DOI-backed archive, technical report). See `## Requirements → Active` and the v2 list carried in the requirements archive.

## Requirements

### Validated

- CUDA AES kernels exist for ECB, CTR, and GCM modes with 128-bit and 256-bit keys.
- The benchmark executable runs multiple message sizes and repeated runs.
- GPU benchmark timing uses CUDA events.
- CPU throughput comparison exists through OpenSSL EVP.
- Benchmark artifacts are written under `bench/`.
- Optional profiling support exists through NVTX and Nsight-related CMake targets.
- A codebase map exists under `.planning/codebase/`.
- Phase 1 validated portable CMake configuration and build-focused README guidance.
- Phase 2 added deterministic known-answer tests for ECB, CTR, and GCM, plus GCM tag/authentication fixes at source level.
- Phase 3 added reproducible benchmark metadata capture, stable raw CSV output, summary generation, methodology documentation, and CUDA event cleanup at source level.
- Phase 4 added a public README/docs package, open-source governance files, citation metadata, issue/PR templates, an AES mode matrix, and legacy Tezcan provenance notes.
- Phase 5 added CBC, CFB-128, and OFB source, KAT, benchmark dispatch, documentation, and verification evidence at source level.
- Phase 6 added CCM, XTS-AES, AES-KW, AES-KWP source, KAT, benchmark dispatch, documentation, GMAC/CMAC boundary notes, and verification evidence at source level.
- ✓ Improved discoverability for GitHub search, Google indexing, and technical readers searching for GPU AES benchmarks — v1.0 (Phase 7, DOCS-04).
- ✓ Packaged benchmark results as a versioned release with reproducibility notes — v1.0 (Phase 8, REPO-04; `v1.0.0` GitHub Release with raw artifacts).
- ✓ Kept the repo alive with maintenance rules, roadmap, changelog, contribution workflow, templates, and a security reporting path — v1.0 (Phase 8, MAINT-01..04).

### Active

(v2.0 candidates — to be scoped via `/gsd-new-milestone`)

- [ ] Compare multiple GPU models in a standardized results table (BENCH-07).
- [ ] Automate a benchmark matrix across CUDA versions and architectures (BENCH-08).
- [ ] Generate charts from benchmark data (BENCH-09).
- [ ] Benchmark GMAC and CMAC as standalone authentication/MAC workloads (MODE-09).
- [ ] Offer a documented library/API surface for the AES kernels (LIB-01, LIB-02).
- [ ] Publish a project website (GitHub Pages), a DOI-backed archived release, and a paper-style technical report (PUB-01, PUB-02, PUB-03).

### Out of Scope

- Production cryptography library API - the v1 goal is a benchmark suite, not a drop-in security library.
- Claims of being the fastest GPU AES implementation without reproducible comparative evidence.
- Multi-platform package manager distribution in v1 - source build and reproducibility come first.
- Full academic paper writing in v1 - citation and methodology are in scope, a paper can follow later.
- Web app or SaaS interface - GitHub README/docs and optionally GitHub Pages are enough for v1.

## Context

The repository is brownfield CUDA/C++ code. Current code is useful and now has public-facing documentation, governance files, a mode matrix, correctness docs, benchmark methodology docs, and source-level coverage for ECB, CBC, CFB-128, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, and AES-KWP. Phase 7 should improve discoverability without adding unsupported benchmark claims.

A main-branch code review on 2026-06-04 found three high-severity AES-GCM blockers: decrypt accepts unauthenticated ciphertext, IV broadcast uses warp-local `__shfl_sync` incorrectly for a 256-thread block, and tag generation is not standard AES-GCM because it omits the length block and final `E(K, J0)` XOR. Phase 2 fixed these at source level for the supported 96-bit-IV, empty-AAD, full-block GCM scope. Runtime CMake/CTest verification still needs a shell where `nvcc` can find `cl.exe`.

The desired positioning is:

- Primary identity: reproducible GPU AES benchmark suite.
- Primary audience: CUDA/GPU developers.
- Benchmark standard: reproducible, with environment, commands, raw outputs, and result summaries.
- SEO target terms: GPU AES, CUDA AES, CUDA AES benchmark, AES GPU benchmark, AES encryption CUDA, GPU cryptography benchmark, AES-128 CUDA, AES-256 CUDA, AES GCM CUDA, AES CBC CUDA, AES XTS CUDA, AES CCM CUDA.

Current external best-practice signals:

- GitHub recommends repository basics such as README, license, citation, contributing, and code of conduct for healthy open-source repositories.
- Google Search documentation emphasizes useful, crawlable, clear content over keyword stuffing.
- Research/open-source guidance commonly recommends license, versioning, identifiers/citation metadata, and contributor documentation for discoverability and reproducibility.
- NIST SP 800-38A defines ECB, CBC, CFB, OFB, and CTR; the broader 800-38 series also covers CMAC, CCM, GCM/GMAC, XTS-AES, and AES key wrap variants. For this project, CMAC can be tracked separately as authentication/MAC benchmarking rather than encryption throughput.

## Constraints

- **Technical stack**: CUDA C++, CMake, OpenSSL, NVIDIA tooling - keep the stack close to the existing implementation.
- **Credibility**: Benchmark claims must be reproducible and should distinguish kernel-only timing from end-to-end timing.
- **Security**: Present the project as benchmark/research software unless and until cryptographic API hardening is complete.
- **Portability**: Remove local absolute paths before public release.
- **Correctness gate**: Do not present GCM benchmark output as standard AES-GCM until the review findings in `.planning/reviews/2026-06-04-main-branch-code-review.md` are fixed and tested.
- **Hardware specificity**: Results must name GPU model, compute capability, driver, CUDA Toolkit, OS, compiler, clocks/persistence mode, and command line.
- **SEO**: Use clear natural-language headings and project metadata; do not degrade README quality with keyword stuffing.
- **Maintenance**: A living repo needs issues, releases, changelog, security contact, and contribution expectations.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Position as a reproducible benchmark suite | The existing code is benchmark-oriented and this is the strongest credible public identity. | ✓ Good — shipped v1.0 |
| Target CUDA/GPU developers first | This audience judges by buildability, methodology, correctness, and results. | Active |
| Use reproducible benchmark standard | Prestige depends on trust, not only peak throughput numbers. | Active |
| Treat library/API use as out of scope for v1 | Avoid implying production cryptographic safety before tests and API hardening exist. | Held — revisit in v2 (LIB-01/02) |
| Build SEO through documentation quality | Search engines and developers both reward clear, useful, crawlable content. | ✓ Validated v1.0 (Phase 7) |
| Plan full AES mode coverage | The prestige target is stronger if the roadmap covers all important AES modes, not only current ECB/CTR/GCM code. | Active |
| Treat GCM review findings as blockers | Public benchmark credibility depends on standard AES-GCM correctness and authentication semantics. | Active |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? Move to Out of Scope with reason.
2. Requirements validated? Move to Validated with phase reference.
3. New requirements emerged? Add to Active.
4. Decisions to log? Add to Key Decisions.
5. "What This Is" still accurate? Update if drifted.

**After each milestone**:
1. Full review of all sections.
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state.

---
*Last updated: 2026-06-08 after v1.0 Public Release milestone*
