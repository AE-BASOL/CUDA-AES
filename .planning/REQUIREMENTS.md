# Requirements: CUDA-AES Benchmark

**Defined:** 2026-06-04
**Core Value:** Anyone landing on the repository can build it, verify AES correctness, reproduce benchmark results, and understand why the results are credible.

## v1 Requirements

### Repository Foundation

- [x] **REPO-01**: Developer can clone the repository without generated build artifacts or IDE metadata being treated as source.
- [x] **REPO-02**: Developer can understand the project positioning from the first screen of `README.md`.
- [x] **REPO-03**: Developer can see license, contribution, security, citation, changelog, issue, and pull request policies.
- [x] **REPO-04**: Maintainer can publish versioned releases with clear release notes.

### Build Portability

- [x] **BUILD-01**: Developer can configure the project without editing local absolute paths.
- [x] **BUILD-02**: Developer can select CUDA architecture through CMake configuration.
- [x] **BUILD-03**: Developer can build the benchmark target from documented commands.
- [x] **BUILD-04**: Developer can identify required CUDA, compiler, OpenSSL, and platform prerequisites.
- [x] **BUILD-05**: Developer can configure or diagnose the CUDA host compiler requirement without editing source files.

### Correctness

- [x] **TEST-01**: Developer can run deterministic AES-128 and AES-256 ECB known-answer tests.
- [x] **TEST-02**: Developer can run deterministic AES-128 and AES-256 CTR known-answer tests.
- [x] **TEST-03**: Developer can run deterministic AES-128 and AES-256 GCM known-answer tests, including tag behavior.
- [x] **TEST-04**: Developer can run a small smoke test without allocating 1 GiB buffers.
- [x] **TEST-05**: Maintainer can see correctness status before trusting benchmark output.
- [x] **TEST-06**: GCM decrypt rejects unauthenticated ciphertext before plaintext is accepted as valid.
- [x] **TEST-07**: GCM IV/counter initialization is correct for all threads in a multi-warp block.
- [x] **TEST-08**: GCM tag generation matches standard AES-GCM including length block and final `E(K, J0)` XOR.

### Benchmark Reproducibility

- [x] **BENCH-01**: Developer can run one documented benchmark command that records GPU, driver, CUDA Toolkit, OS, compiler, build type, clocks/persistence note, mode, size, and run count.
- [x] **BENCH-02**: Developer can distinguish kernel-only throughput from end-to-end throughput.
- [x] **BENCH-03**: Developer can save raw benchmark output in machine-readable CSV or JSON.
- [x] **BENCH-04**: Developer can generate summary tables from raw output.
- [x] **BENCH-05**: Reader can inspect benchmark methodology and limitations before reading claims.
- [x] **BENCH-06**: Benchmark runs release CUDA event resources and do not leak timing resources across long runs.

### AES Mode Coverage

- [x] **MODE-01**: Reader can see a mode matrix covering ECB, CBC, CFB, OFB, CTR, GCM/GMAC, CCM, XTS-AES, AES-KW, and AES-KWP.
- [x] **MODE-02**: Developer can run correctness tests and benchmarks for CBC mode.
- [x] **MODE-03**: Developer can run correctness tests and benchmarks for CFB mode.
- [x] **MODE-04**: Developer can run correctness tests and benchmarks for OFB mode.
- [x] **MODE-05**: Developer can run correctness tests and benchmarks for CCM mode.
- [x] **MODE-06**: Developer can run correctness tests and benchmarks for XTS-AES mode.
- [x] **MODE-07**: Developer can run correctness tests and benchmarks for AES-KW and AES-KWP key wrap modes.
- [x] **MODE-08**: Reader can distinguish encryption modes from authentication/MAC-only coverage such as GMAC and CMAC.

### Documentation And SEO

- [x] **DOCS-01**: Reader can understand what GPU AES modes are implemented and what is not implemented.
- [x] **DOCS-02**: Reader can follow build, test, and benchmark instructions without private local knowledge.
- [x] **DOCS-03**: Reader can find pages for benchmark methodology, correctness, results, architecture, and profiling.
- [x] **DOCS-04**: Search engines and GitHub search can infer that the repository is about CUDA AES, GPU AES benchmark, AES GPU performance, and reproducible cryptography benchmarking.
- [x] **DOCS-05**: Reader can cite the project using `CITATION.cff`.

### Maintenance

- [x] **MAINT-01**: Contributor can open useful issues and pull requests using templates.
- [x] **MAINT-02**: Maintainer can evaluate benchmark-result contributions with a documented checklist.
- [x] **MAINT-03**: Maintainer can track future work through roadmap and changelog.
- [x] **MAINT-04**: Security-sensitive reports have an explicit reporting path and scope.

## v2 Requirements

### Broader Benchmarking

- **BENCH-07**: Developer can compare multiple GPU models in a standardized results table.
- **BENCH-08**: Developer can run benchmark matrix automation across CUDA versions and architectures.
- **BENCH-09**: Reader can view charts generated from benchmark data.
- **MODE-09**: Developer can benchmark GMAC and CMAC as standalone authentication/MAC workloads.

### Library/API

- **LIB-01**: Developer can use AES kernels through a documented library API.
- **LIB-02**: Developer can integrate the library into another CUDA project.

### Publication

- **PUB-01**: Reader can access a project website through GitHub Pages.
- **PUB-02**: Reader can cite a DOI-backed archived release.
- **PUB-03**: Reader can follow a paper-style technical report.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Production cryptographic library guarantee | v1 is a benchmark suite; security hardening requires separate API and audit work. |
| Fastest-in-world claim | Needs broader controlled comparisons before it is credible. |
| Package manager distribution | Build reproducibility and correctness come first. |
| Full academic paper | Useful later, but v1 needs repo quality and reproducible evidence first. |
| Web application | GitHub docs and optional static pages are sufficient for v1. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REPO-01 | Phase 1 | Complete |
| BUILD-01 | Phase 1 | Complete |
| BUILD-02 | Phase 1 | Complete |
| BUILD-03 | Phase 1 | Complete |
| BUILD-04 | Phase 1 | Complete |
| BUILD-05 | Phase 1 | Complete |
| TEST-01 | Phase 2 | Complete |
| TEST-02 | Phase 2 | Complete |
| TEST-03 | Phase 2 | Complete |
| TEST-04 | Phase 2 | Complete |
| TEST-05 | Phase 2 | Complete |
| TEST-06 | Phase 2 | Complete |
| TEST-07 | Phase 2 | Complete |
| TEST-08 | Phase 2 | Complete |
| BENCH-01 | Phase 3 | Complete |
| BENCH-02 | Phase 3 | Complete |
| BENCH-03 | Phase 3 | Complete |
| BENCH-04 | Phase 3 | Complete |
| BENCH-05 | Phase 3 | Complete |
| BENCH-06 | Phase 3 | Complete |
| MODE-01 | Phase 4 | Complete |
| REPO-02 | Phase 4 | Complete |
| REPO-03 | Phase 4 | Complete |
| DOCS-01 | Phase 4 | Complete |
| DOCS-02 | Phase 4 | Complete |
| DOCS-03 | Phase 4 | Complete |
| DOCS-05 | Phase 4 | Complete |
| MODE-02 | Phase 5 | Complete |
| MODE-03 | Phase 5 | Complete |
| MODE-04 | Phase 5 | Complete |
| MODE-05 | Phase 6 | Complete |
| MODE-06 | Phase 6 | Complete |
| MODE-07 | Phase 6 | Complete |
| MODE-08 | Phase 6 | Complete |
| DOCS-04 | Phase 7 | Complete |
| REPO-04 | Phase 8 | Complete |
| MAINT-01 | Phase 8 | Complete |
| MAINT-02 | Phase 8 | Complete |
| MAINT-03 | Phase 8 | Complete |
| MAINT-04 | Phase 8 | Complete |

**Coverage:**
- v1 requirements: 40 total
- Mapped to phases: 40
- Unmapped: 0

---
*Requirements defined: 2026-06-04*
*Last updated: 2026-06-05 after Phase 6 completion*
