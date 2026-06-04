# CUDA-AES Benchmark

## What This Is

CUDA-AES Benchmark is a reproducible open-source benchmark suite for GPU AES implementations. It should become a living, credible repository for CUDA developers who search for GPU AES, CUDA AES benchmark, AES GPU performance, and reproducible cryptography benchmark work.

The existing code already runs CUDA AES kernels for ECB, CTR, and GCM with 128-bit and 256-bit keys, compares GPU throughput against OpenSSL CPU baselines, and writes benchmark output. The project now needs to mature from a personal experiment into a public, trusted, searchable, maintainable repository that can eventually benchmark the full practical AES mode family: ECB, CBC, CFB, OFB, CTR, GCM/GMAC, CCM, XTS-AES, AES-KW, and AES-KWP.

## Core Value

Anyone landing on the repository can build it, verify AES correctness, reproduce benchmark results, and understand why the results are credible.

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

### Active

- [ ] Plan and implement benchmark coverage beyond ECB/CTR/GCM for standard AES modes.
- [ ] Improve discoverability for GitHub search, Google indexing, and technical readers searching for GPU AES benchmarks.
- [ ] Package benchmark results as versioned releases with reproducibility notes.
- [ ] Keep the repo alive with clear maintenance rules, roadmap, changelog, and contribution workflow.

### Out of Scope

- Production cryptography library API - the v1 goal is a benchmark suite, not a drop-in security library.
- Claims of being the fastest GPU AES implementation without reproducible comparative evidence.
- Multi-platform package manager distribution in v1 - source build and reproducibility come first.
- Full academic paper writing in v1 - citation and methodology are in scope, a paper can follow later.
- Web app or SaaS interface - GitHub README/docs and optionally GitHub Pages are enough for v1.

## Context

The repository is brownfield CUDA/C++ code. Current code is useful and now has public-facing documentation, governance files, a mode matrix, correctness docs, and benchmark methodology docs. Phase 5 should move from documentation into implementation by adding CBC, CFB, and OFB coverage.

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
| Position as a reproducible benchmark suite | The existing code is benchmark-oriented and this is the strongest credible public identity. | Active |
| Target CUDA/GPU developers first | This audience judges by buildability, methodology, correctness, and results. | Active |
| Use reproducible benchmark standard | Prestige depends on trust, not only peak throughput numbers. | Active |
| Treat library/API use as out of scope for v1 | Avoid implying production cryptographic safety before tests and API hardening exist. | Pending |
| Build SEO through documentation quality | Search engines and developers both reward clear, useful, crawlable content. | Pending |
| Plan full AES mode coverage | The prestige target is stronger if the roadmap covers all important AES modes, not only current ECB/CTR/GCM code. | Pending |
| Treat GCM review findings as blockers | Public benchmark credibility depends on standard AES-GCM correctness and authentication semantics. | Pending |

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
*Last updated: 2026-06-04 after Phase 4 completion*
