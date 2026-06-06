---
phase: 08-release-and-maintenance-loop
plan: 03
subsystem: release
tags: [v1-release, changelog, verification, release-candidate]
requires:
  - phase: 08-release-and-maintenance-loop
    provides: "Benchmark contribution checklist, maintenance loop, security policy, and issue template updates from plans 08-01 and 08-02"
provides:
  - "Source-controlled v1.0.0 release notes draft"
  - "Release gate evidence with verification-blocked CUDA host compiler result"
  - "Release-candidate changelog state that avoids premature final v1 publication"
affects: [release-management, maintenance-loop, benchmark-results]
tech-stack:
  added: []
  patterns: [release-candidate handoff, environment-limited release gate evidence]
key-files:
  created:
    - docs/release-v1.md
    - .planning/phases/08-release-and-maintenance-loop/08-VERIFICATION.md
  modified:
    - CHANGELOG.md
    - docs/release-v1.md
    - .planning/phases/08-release-and-maintenance-loop/08-USER-SETUP.md
key-decisions:
  - "Kept v1.0.0 in release-candidate state because Release CMake configure is blocked by missing `cl.exe`."
  - "Did not promote CHANGELOG.md to a final dated 1.0.0 section until runtime verification passes."
  - "Did not invent benchmark numbers or attach stale benchmark artifacts."
patterns-established:
  - "Final release publication requires `verification-passed`; blocked runtime gates remain explicit release-candidate handoffs."
requirements-completed: [REPO-04, MAINT-03]
requirements-pending: []
duration: 8 min
completed: 2026-06-06
---

# Phase 8 Plan 03: v1 Release Package Summary

**v1.0.0 release-candidate notes, changelog debt, and blocked runtime release gate evidence**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-06T21:19:06+03:00
- **Completed:** 2026-06-06T21:26:56+03:00
- **Tasks:** 4 completed
- **Files modified:** 4 tracked deliverable files

## Accomplishments

- Added `docs/release-v1.md` as the source-controlled `v1.0.0` GitHub Release draft.
- Attempted the Release CMake configure command required by the release gate.
- Recorded the exact `nvcc fatal : Cannot find compiler 'cl.exe' in PATH` blocker in `08-VERIFICATION.md`.
- Kept `CHANGELOG.md` in `Unreleased` with explicit `v1.0.0` release-candidate verification debt rather than promoting a final dated release.
- Updated `08-USER-SETUP.md` with the CUDA/MSVC-ready release gate and GitHub Release draft actions.

## Task Commits

1. **Task 1: Draft v1.0.0 release notes and artifact manifest** - `3d407e8` (`docs(08-03): draft v1 release notes`)
2. **Task 2: Attempt final v1 runtime release gate** - `00949bf` (`docs(08-03): record release gate blocker`)
3. **Task 3: Normalize release gate evidence** - `00949bf` (`docs(08-03): record release gate blocker`)
4. **Task 4: Update changelog according to release gate result** - `693994b` (`docs(08-03): keep changelog release candidate`)

**Plan metadata:** committed after summary creation.

## Files Created/Modified

- `docs/release-v1.md` - `v1.0.0` release notes draft and publication checklist.
- `.planning/phases/08-release-and-maintenance-loop/08-VERIFICATION.md` - Phase 8 requirement evidence and release gate blocker.
- `CHANGELOG.md` - Release-candidate entry and verification debt.
- `.planning/phases/08-release-and-maintenance-loop/08-USER-SETUP.md` - Maintainer actions for release gate, private vulnerability reporting, and GitHub Release draft.

## Decisions Made

- Left the release as a release candidate because CUDA compiler detection failed before build, CTest, benchmark, or summary generation could run.
- Preserved `CHANGELOG.md` under `Unreleased` because a final `1.0.0` section would imply the runtime release gate passed.
- Listed only the raw artifact manifest; no benchmark numbers or verified asset paths were invented.

## Deviations from Plan

None - the plan explicitly allowed a `verification-blocked` release-candidate handoff when the CUDA host compiler environment is unavailable.

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope changes.

## Issues Encountered

Release CMake configure failed in the current shell:

```text
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

This is the known CUDA host compiler environment blocker, not a source-level release package failure.

## User Setup Required

External release actions require manual setup. See [08-USER-SETUP.md](./08-USER-SETUP.md) for:

- Running the final release gate from a Visual Studio Developer Command Prompt or with `CMAKE_CUDA_HOST_COMPILER`.
- Publishing or saving the GitHub Release draft for tag `v1.0.0`.
- Enabling GitHub private vulnerability reporting if the maintainer chooses that path.

## Verification

- `Test-Path docs\release-v1.md` - passed.
- `rg "v1.0.0|release candidate|CTest|run_metadata.csv|thr_gpu.csv|thr_cpu.csv|summary.md|known limitations|benchmark-result-contributions.md|maintenance.md" docs\release-v1.md` - passed.
- `Test-Path .planning\phases\08-release-and-maintenance-loop\08-VERIFICATION.md` - passed.
- `rg "REPO-04|MAINT-03|release gate|ctest|CudaProject|summary.md|verification-passed|verification-blocked|release candidate" .planning\phases\08-release-and-maintenance-loop\08-VERIFICATION.md docs\release-v1.md` - passed.
- `rg "Unreleased|1.0.0|Known Verification Debt|release candidate|cl.exe|CTest" CHANGELOG.md` - passed.
- `rg "fastest|fastest-in-world|universal ranking|production cryptography library" docs\release-v1.md` - passed as a guardrail check: matches are disclaimers or checklist items, not unsupported claims.

## Self-Check: PASSED

All source-controlled task acceptance criteria and plan-level verification commands passed. Runtime release publication remains blocked until the maintainer reruns the release gate in a CUDA/MSVC-ready shell and records `verification-passed`.

## Next Phase Readiness

Plan 08-03 is complete as a release-candidate handoff. Phase 8 can proceed to final phase-level verification with the runtime release gate caveat preserved.

---
*Phase: 08-release-and-maintenance-loop*
*Completed: 2026-06-06*
