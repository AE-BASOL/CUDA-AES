# Milestones

## v1.0 Public Release (Shipped: 2026-06-07)

**Phases completed:** 8 phases, 23 plans, 42 tasks

**Delivered:** A buildable, correctness-verified, reproducibly-benchmarkable open-source CUDA AES repository covering the full practical mode family (ECB, CBC, CFB-128, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, AES-KWP for AES-128/256), published as the `v1.0.0` GitHub Release.

**Key accomplishments:**

- Portable CMake dependency discovery for CUDA, OpenSSL, CUDA architecture selection, and optional profiling tools
- Repository hygiene cleanup with generated artifacts untracked and canonical source ownership documented
- Build-focused README with CUDA/OpenSSL prerequisites, Windows host compiler guidance, and explicit GCM caveats
- CTest-registered CUDA AES known-answer harness for ECB, CTR, and GCM
- AES-GCM tag generation, IV broadcast, and authentication semantics corrected for Phase 2 scope
- README and testing map now expose correctness commands and verified AES mode coverage
- Benchmark CLI metadata, Phase 3 raw CSV schema, kernel-only timing labels, and CUDA event cleanup
- Python summary generator for Phase 3 raw benchmark CSV files with timing-scope-preserving tables
- Benchmark methodology docs, maintainer benchmark map, and Phase 3 verification evidence
- Public CUDA-AES landing page with architecture, correctness, benchmark methodology, results, and profiling docs
- Open-source governance files, citation metadata, changelog, issue templates, and PR template
- CBC AES-128/AES-256 CUDA kernels with NIST-style KAT coverage and benchmark rows
- CFB-128 and OFB AES-128/AES-256 CUDA modes with deterministic KATs and benchmark rows
- CCM, XTS-AES, AES-KW, and AES-KWP CUDA modes with KAT coverage, benchmark dispatch, and GMAC/CMAC boundary documentation
- README discoverability wording, stable docs landing page, and verified GitHub metadata for CUDA AES / GPU AES benchmark search (DOCS-04)
- v1.0.0 release notes, changelog, benchmark contribution checklist, issue/PR templates, security reporting path, and a verification-passed release gate (Release build + CTest + smoke benchmark from a VS 2022 Developer Command Prompt)
- Published `v1.0.0` GitHub Release with raw benchmark artifacts; tag realigned onto protected master history

**Known follow-ups:** GitHub private vulnerability reporting is optional and still toggled manually in repo Settings (SECURITY.md provides a public fallback).

---
