# Roadmap: CUDA-AES Benchmark

**Created:** 2026-06-04
**Project:** CUDA-AES Benchmark
**Core Value:** Anyone landing on the repository can build it, verify AES correctness, reproduce benchmark results, and understand why the results are credible.

## Overview

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 1 | Repository And Build Foundation | Complete 2026-06-04. Make the repo clean, portable, and buildable by an outside developer. | REPO-01, BUILD-01, BUILD-02, BUILD-03, BUILD-04, BUILD-05 |
| 2 | Correctness Baseline | Prove AES outputs before publishing benchmark claims and close the GCM review blockers. | TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07, TEST-08 |
| 3 | Reproducible Benchmark Harness | Complete 2026-06-04. Make benchmark runs repeatable, inspectable, resource-clean, and summarizable. | BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, BENCH-06 |
| 4 | Open-Source Documentation Package | Complete 2026-06-04. Turn the repo into a credible public landing page and contributor-ready project, including a full AES mode matrix. | REPO-02, REPO-03, DOCS-01, DOCS-02, DOCS-03, DOCS-05, MODE-01 |
| 5 | Confidentiality Mode Expansion | Complete 2026-06-05. Add the missing SP 800-38A confidentiality modes: CBC, CFB, and OFB. | MODE-02, MODE-03, MODE-04 |
| 6 | Authenticated And Specialized Mode Expansion | Complete 2026-06-05 at source level. Add CCM, XTS-AES, AES-KW, AES-KWP, and document GMAC/CMAC boundaries. | MODE-05, MODE-06, MODE-07, MODE-08 |
| 7 | Discoverability And SEO | Optimize GitHub and web discoverability without compromising technical quality. | DOCS-04 |
| 8 | Release And Maintenance Loop | Publish a v1 release and define how the repo stays alive. | REPO-04, MAINT-01, MAINT-02, MAINT-03, MAINT-04 |

## Phase 1: Repository And Build Foundation

**Status:** Complete 2026-06-04

**Goal:** Make the repo clean, portable, and buildable by an outside developer.

**Requirements:** REPO-01, BUILD-01, BUILD-02, BUILD-03, BUILD-04, BUILD-05

**Success criteria:**
1. Fresh clone does not rely on checked build output or IDE state.
2. CMake configuration has no private absolute paths.
3. CUDA architecture and OpenSSL locations are configurable or discovered.
4. README build commands match the actual supported platform.
5. A clean configure/build command is documented and works on the maintainer machine.
6. CUDA host compiler setup is documented or surfaced through a clear configure-time diagnostic.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `01-01` | 1 | Portable CMake, OpenSSL/CUDA package discovery, configurable CUDA architecture, host compiler diagnostics |
| `01-02` | 1 | Repository hygiene audit, ignore-rule cleanup, canonical top-level source boundary note |
| `01-03` | 2 | README build prerequisites, configure/build commands, Windows `cl.exe` troubleshooting, GCM limitation note |

**Wave 2 blocked on Wave 1 completion.**

## Phase 2: Correctness Baseline

**Goal:** Prove AES outputs before publishing benchmark claims and close the GCM review blockers.

**Requirements:** TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07, TEST-08

**Success criteria:**
1. Known-answer tests cover ECB, CTR, and GCM for AES-128 and AES-256.
2. GCM decrypt rejects tag mismatches before plaintext is accepted as valid.
3. A small smoke test runs quickly without 1 GiB allocation.
4. Correctness checks are available through documented build/test commands.
5. Benchmark docs state that results are trusted only after correctness checks pass.
6. GCM IV/counter state is broadcast correctly across all warps in a 256-thread block.
7. GCM tag generation matches standard AES-GCM, including length block and final `E(K, J0)` XOR.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `02-01` | 1 | Deterministic KAT/smoke harness for ECB, CTR, and GCM across AES-128/AES-256 |
| `02-02` | 2 | Standard AES-GCM tag generation, block-wide IV/J0 state, and tag verification semantics |
| `02-03` | 3 | README/CMake/testing documentation exposing correctness commands and verified scope |

**Wave 2 blocked on Wave 1 completion. Wave 3 blocked on Wave 2 completion.**

## Phase 3: Reproducible Benchmark Harness

**Status:** Complete 2026-06-04 at source level; runtime CMake/CTest/benchmark execution still needs a CUDA host compiler environment.

**Goal:** Make benchmark runs repeatable, inspectable, resource-clean, and summarizable.

**Requirements:** BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, BENCH-06

**Success criteria:**
1. A script captures hardware/software environment and benchmark parameters.
2. Raw results are written in stable CSV or JSON format.
3. Summary generation computes clear tables from raw output.
4. Kernel-only and end-to-end timing are separated or explicitly labeled.
5. Methodology documentation explains limitations, warmup, repetitions, clocks, and CPU baseline.
6. CUDA timing events are destroyed or wrapped in RAII so long runs do not leak event resources.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `03-01` | 1 | Benchmark run metadata, stable raw output fields, timing scope labels, configurable smoke parameters, and CUDA event cleanup |
| `03-02` | 2 | Summary generation from raw benchmark output |
| `03-03` | 3 | Benchmark methodology documentation and Phase 3 verification evidence |

**Wave 2 blocked on Wave 1 completion. Wave 3 blocked on Wave 2 completion.**

## Phase 4: Open-Source Documentation Package

**Status:** Complete 2026-06-04

**Goal:** Turn the repo into a credible public landing page and contributor-ready project, including a full AES mode matrix.

**Requirements:** REPO-02, REPO-03, DOCS-01, DOCS-02, DOCS-03, DOCS-05, MODE-01

**Success criteria:**
1. README first screen states what the project is, who it is for, and why it matters.
2. README includes quick start, benchmark snapshot, correctness status, methodology links, and clear caveats.
3. `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, and `CHANGELOG.md` exist.
4. Documentation pages cover architecture, correctness, benchmark methodology, results, and profiling.
5. Documentation includes a mode matrix for ECB, CBC, CFB, OFB, CTR, GCM/GMAC, CCM, XTS-AES, AES-KW, and AES-KWP.
6. The legacy Tezcan code has clear provenance and license/status notes.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `04-01` | 1 | README landing page and public docs hub for architecture, correctness, benchmark methodology, results, and profiling |
| `04-02` | 2 | License, contributing, security, citation, changelog, issue templates, and PR template |
| `04-03` | 3 | AES mode matrix, legacy Tezcan provenance documentation, and Phase 4 verification evidence |

**Wave 2 blocked on Wave 1 completion. Wave 3 blocked on Wave 2 completion.**

## Phase 5: Confidentiality Mode Expansion

**Status:** Complete 2026-06-05 at source level; runtime CMake/CTest/benchmark execution still needs a CUDA host compiler environment.

**Goal:** Add the missing SP 800-38A confidentiality modes: CBC, CFB, and OFB.

**Requirements:** MODE-02, MODE-03, MODE-04

**Success criteria:**
1. CBC mode has AES-128 and AES-256 kernels or host/device path, known-answer tests, and benchmark rows.
2. CFB mode has AES-128 and AES-256 coverage, including segment-size decision documented.
3. OFB mode has AES-128 and AES-256 coverage and tests.
4. Benchmark output can compare ECB, CBC, CFB, OFB, CTR, and GCM using the same harness.
5. Docs explain that ECB/CBC/CFB/OFB/CTR are confidentiality-only modes.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `05-01` | 1 | CBC AES-128/AES-256 implementation, KAT coverage, and benchmark dispatch |
| `05-02` | 2 | CFB-128 and OFB AES-128/AES-256 implementation, KAT coverage, and benchmark dispatch |
| `05-03` | 3 | Public docs, mode matrix updates, verification evidence, and handoff |

**Wave 2 blocked on Wave 1 completion. Wave 3 blocked on Wave 2 completion.**

## Phase 6: Authenticated And Specialized Mode Expansion

**Status:** Complete 2026-06-05 at source level; runtime CMake/CTest/benchmark execution still needs a CUDA host compiler environment.

**Goal:** Add CCM, XTS-AES, AES-KW, AES-KWP, and document GMAC/CMAC boundaries.

**Requirements:** MODE-05, MODE-06, MODE-07, MODE-08

**Success criteria:**
1. CCM mode has correctness tests and benchmark output with nonce/tag parameters documented.
2. XTS-AES mode has correctness tests and storage-sector-oriented benchmark parameters.
3. AES-KW and AES-KWP have correctness tests and benchmark coverage appropriate for key-wrap workloads.
4. GCM/GMAC and CMAC are documented distinctly so encryption throughput and MAC/authentication throughput are not confused.
5. The mode matrix shows implementation, test, benchmark, and documentation status for every tracked mode.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `06-01` | 1 | CCM AES-128/AES-256 implementation, ciphertext/tag KAT coverage, benchmark dispatch, and AEAD parameter docs |
| `06-02` | 2 | XTS-AES full-block data-unit implementation, KAT coverage, benchmark dispatch, and storage-sector docs |
| `06-03` | 3 | AES-KW/AES-KWP wrap/unwrap implementation, tamper KAT coverage, benchmark dispatch, and key-wrap workload docs |
| `06-04` | 4 | Public docs, mode matrix updates, GMAC/CMAC boundaries, verification evidence, and handoff |

**Wave 2 blocked on Wave 1 completion. Wave 3 blocked on Wave 2 completion. Wave 4 blocked on Wave 3 completion.**

## Phase 7: Discoverability And SEO

**Goal:** Optimize GitHub and web discoverability without compromising technical quality.

**Requirements:** DOCS-04

**Success criteria:**
1. GitHub description and topics target CUDA AES and GPU benchmark searches.
2. README headings naturally include important terms such as GPU AES, CUDA AES benchmark, AES-GCM CUDA, and reproducible benchmark.
3. Docs use clear page titles, stable links, and descriptive filenames.
4. Optional GitHub Pages plan is ready if public docs need stronger indexing.
5. No keyword stuffing or unsupported performance claims are introduced.

**Plans:**

| Plan | Wave | What it builds |
|------|------|----------------|
| `07-01` | 1 | README discoverability wording, stable docs landing page, GitHub metadata recommendations, and DOCS-04 verification evidence |

## Phase 8: Release And Maintenance Loop

**Goal:** Publish a v1 release and define how the repo stays alive.

**Requirements:** REPO-04, MAINT-01, MAINT-02, MAINT-03, MAINT-04

**Success criteria:**
1. v1 release notes include build/test status, benchmark environment, raw artifacts, and known limitations.
2. Issue and PR templates guide useful contributions.
3. Benchmark result contribution checklist exists.
4. Security policy explains benchmark-vs-library scope and reporting path.
5. `CHANGELOG.md` and roadmap are ready for ongoing updates.

## Coverage

- v1 requirements: 40
- mapped requirements: 40
- unmapped requirements: 0

---
*Roadmap created: 2026-06-04*
