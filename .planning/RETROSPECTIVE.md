# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Public Release

**Shipped:** 2026-06-07
**Phases:** 8 | **Plans:** 23 | **Tasks:** 42
**GSD span:** 2026-06-04 → 2026-06-07 (underlying CUDA code predates GSD adoption; ~99 commits total on the repo, 11 `feat` commits during the milestone).
**Sessions:** not separately instrumented.

### What Was Built
- Full practical AES mode family on CUDA — ECB, CBC, CFB-128, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, AES-KWP (AES-128/256) — with source, KAT coverage, and benchmark dispatch.
- Standard-conformant AES-GCM corrections (length block + final `E(K, J0)` XOR, block-wide IV/J0 broadcast, tag-verifying decrypt) for the supported 96-bit-IV / empty-AAD / full-block scope.
- Reproducible benchmark harness: environment/parameter capture, stable raw CSV, kernel-only vs end-to-end timing labels, RAII CUDA event cleanup, and a Python summary generator.
- Public repository package: README landing page, docs hub (architecture/correctness/methodology/results/profiling), governance files, citation metadata, issue/PR templates, changelog, mode matrix, and SEO wording.
- v1.0.0 GitHub Release with raw benchmark artifacts, plus a verification-passed release gate (Release build + CTest 1/1 + smoke benchmark).

### What Worked
- Phase/wave structure kept dependencies explicit (e.g. correctness before benchmark claims; source before docs/verification waves).
- Treating the GCM review findings as hard blockers prevented publishing benchmark numbers under an incorrect "AES-GCM" label.
- A single release-gate verification pass (from a VS 2022 Developer Command Prompt) retroactively exercised the runtime for every phase that had only been verified at source level — one environment fix cleared four phases of debt at once.

### What Was Inefficient
- **Carried verification debt.** Phases 2, 3, 5, and 6 were marked complete "at source level" because the build environment couldn't find `cl.exe` for `nvcc`. Runtime verification was deferred to the Phase 8 release gate, so correctness/benchmark claims sat unexecuted for most of the milestone.
- **Late host-compiler setup.** The `nvcc` → `cl.exe` toolchain requirement surfaced repeatedly across phases before being solved once at the end.
- **Protected-branch friction at release.** A merge-commit/direct-push attempt to `master` was rejected by the `protect-master` ruleset, costing a failed attempt and a tag realignment (moving `v1.0.0` onto the rebased-equivalent commit).

### Patterns Established
- Configure/build CUDA from a **Visual Studio Developer Command Prompt**, or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>` — verify this *before* marking CUDA phases complete, not at release.
- `master` is governed by the `protect-master` ruleset: **PR-only, squash/rebase (no merge commits), required review-thread resolution.** Rebase/squash rewrite SHAs, so release tags must be moved onto the rebased-equivalent commit on `master` and force-updated.
- Planning bookkeeping (milestone archival, etc.) goes through a **feature branch + PR**, never a direct commit to `master`.

### Key Lessons
1. Stand up the full build/test toolchain at Phase 1 and gate phase completion on a real build+test run — deferring runtime verification converts into a single high-risk catch-up at release.
2. Under a linear-history ruleset, plan the release flow (branch → PR → rebase-merge → realign tag) from the start; don't discover it at tag time.
3. Source-level "done" and runtime-verified "done" are different states — track them distinctly so the milestone's true readiness is never overstated.

### Cost Observations
- Model mix: not instrumented this milestone.
- Notable: the most expensive avoidable cost was re-encountering the same `cl.exe`/`nvcc` environment gap across multiple phases instead of solving it once up front.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Key Change |
|-----------|--------|------------|
| v1.0 Public Release | 8 | Established phase/wave planning, correctness-before-claims gating, and a protected-branch PR/rebase release flow. |

### Cumulative Quality

| Milestone | Modes Covered | Verification | Notes |
|-----------|---------------|--------------|-------|
| v1.0 Public Release | 10 (ECB, CBC, CFB-128, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, AES-KWP) | KAT harness + release-gate runtime pass (CTest 1/1, smoke benchmark) | Runtime verification was deferred to release; close earlier in v2. |

### Top Lessons (Verified Across Milestones)

1. Verify the build/test toolchain early; do not accept "source-level complete" as the phase exit criterion for compiled code.
