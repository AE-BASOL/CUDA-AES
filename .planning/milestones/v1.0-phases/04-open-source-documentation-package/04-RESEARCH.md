---
phase: 04-open-source-documentation-package
researched_at: 2026-06-04T15:37:00+03:00
status: complete
---

# Phase 4 Research: Open-Source Documentation Package

## Research Complete

Phase 4 should turn the repository from an internally credible benchmark into a public-facing open-source project. Phase 1-3 established build portability, correctness checks, benchmark metadata, raw output, summary generation, and methodology documentation at source level. Phase 4 should package those facts into README-first public documentation and add governance files.

## Current Repository Facts

- `README.md` currently explains build, correctness, benchmark run commands, raw output, summary generation, and benchmark methodology.
- No `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, or `CHANGELOG.md` files exist.
- No issue or pull request templates exist.
- `.planning/codebase/` contains maintainer maps for architecture, benchmarking, concerns, conventions, integrations, stack, structure, and testing.
- `cihangirTezcanAESimplementation/TEZCAN_README.md` exists, but public-facing provenance and legacy status need to be clearer in top-level docs.
- The current implemented benchmark modes are ECB, CTR, and GCM-shaped paths for AES-128 and AES-256.
- Future roadmap modes include CBC, CFB, OFB, CCM, XTS-AES, AES-KW, AES-KWP, and GMAC/CMAC boundary documentation.

## Planning Implications

Phase 4 should not add new AES kernels. It should create documentation surfaces and governance scaffolding that future mode-expansion phases can fill in.

Recommended documentation surfaces:

- README first screen: project identity, audience, current status, correctness/benchmark caveats, quick start.
- `docs/architecture.md`: canonical source layout and runtime flow.
- `docs/correctness.md`: KAT coverage, GCM scope, runtime environment limitation.
- `docs/benchmark-methodology.md`: command, metadata, raw output, summary, timing scope.
- `docs/results.md`: placeholder/snapshot guidance that does not invent performance numbers.
- `docs/profiling.md`: NVTX, Nsight target, PTX dump.
- `docs/modes.md`: AES mode matrix for implemented, tested, benchmarked, and documented status.
- `docs/legacy-tezcan.md`: provenance/status of the legacy implementation folder.

Recommended governance files:

- `LICENSE`: choose one project-wide license if the user has not already specified one. The executor should avoid inventing legal commitments beyond a standard permissive default unless user approval is needed.
- `CONTRIBUTING.md`: build/test/benchmark workflow and contribution expectations.
- `SECURITY.md`: benchmark-vs-library scope and reporting path.
- `CITATION.cff`: citation metadata for the project.
- `CHANGELOG.md`: initial unreleased/v1 development log.
- `.github/ISSUE_TEMPLATE/*.md` and `.github/pull_request_template.md`: contributor intake.

## Requirement Mapping

- REPO-02: README first screen needs clear project positioning.
- REPO-03: governance, contribution, security, citation, changelog, issue, and PR policy files.
- DOCS-01: implemented/not-implemented GPU AES mode status.
- DOCS-02: build, test, benchmark instructions without private local knowledge.
- DOCS-03: pages for benchmark methodology, correctness, results, architecture, and profiling.
- DOCS-05: `CITATION.cff`.
- MODE-01: mode matrix covering ECB, CBC, CFB, OFB, CTR, GCM/GMAC, CCM, XTS-AES, AES-KW, and AES-KWP.

## Validation Architecture

Validation should be mostly source/document checks:

- Required files exist.
- README first screen includes identity, audience, quick start, correctness status, benchmark caveats.
- Docs pages exist and link to one another from README.
- Mode matrix includes all required modes and distinguishes implemented/tested/benchmarked/documented/future.
- Governance files mention benchmark/research scope and do not claim production cryptography safety.
- `CITATION.cff` is present and parseable enough for GitHub.

## Recommended Plan Split

| Plan | Wave | Focus |
|------|------|-------|
| 04-01 | 1 | README landing page and docs hub for architecture/correctness/benchmark/results/profiling |
| 04-02 | 2 | Governance, contribution, security, citation, changelog, issue and PR templates |
| 04-03 | 3 | AES mode matrix, legacy provenance documentation, and Phase 4 verification |

Wave 2 depends on Wave 1 links and docs structure. Wave 3 depends on the docs directory and governance context so final verification can check the complete package.

