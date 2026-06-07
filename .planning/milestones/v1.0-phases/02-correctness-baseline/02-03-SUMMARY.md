---
phase: 02-correctness-baseline
plan: 03
subsystem: documentation
tags: [readme, testing, correctness]
requires:
  - phase: 02-correctness-baseline
    provides: KAT harness and GCM fixes
provides:
  - README correctness command and coverage statement
  - Updated testing map
  - Explicit GCM scope and production caveat
affects: [documentation, onboarding, benchmark]
tech-stack:
  added: []
  patterns:
    - Correctness status appears before benchmark interpretation
key-files:
  created: []
  modified: [README.md, .planning/codebase/TESTING.md]
key-decisions:
  - "Keep README benchmark claims gated by known-answer tests."
patterns-established:
  - "Documentation states verified scope and remaining crypto limitations."
requirements-completed: [TEST-04, TEST-05]
duration: 15min
completed: 2026-06-04
---

# Phase 2 Plan 03 Summary

**README and testing map now expose correctness commands and verified AES mode coverage**

## Accomplishments

- Added README `Correctness` section with `ctest --test-dir build --output-on-failure`.
- Listed ECB-128/256, CTR-128/256, and GCM-128/256 known-answer coverage.
- Documented Phase 2 GCM scope: 96-bit IV, empty AAD, full 16-byte blocks.
- Updated `.planning/codebase/TESTING.md` from "no known-answer tests" to the new CTest/KAT structure.
- Preserved the "not production cryptography" caveat.

## Task Commits

Committed in the Phase 2 execution commit.

## Files Created/Modified

- `README.md` - Correctness command and coverage.
- `.planning/codebase/TESTING.md` - Updated source-level test map.

## Decisions Made

Correctness documentation is placed before run/benchmark guidance so readers see the verification gate before performance interpretation.

## Deviations from Plan

None.

## Issues Encountered

Runtime CTest could not run because CMake configure did not complete in this shell.

## Verification

- `rg "Correctness|known-answer|ctest|CudaAesKat|smoke" README.md CMakeLists.txt tests .planning\\codebase\\TESTING.md` found the expected docs and wiring.
- `rg "ECB-128|ECB-256|CTR-128|CTR-256|GCM-128|GCM-256|96-bit|AAD|production" README.md .planning\\codebase\\TESTING.md` found coverage and caveats.
- `rg "not production cryptography|not production" README.md` remains satisfied through the README production caveat.

## Next Phase Readiness

Phase 3 can build benchmark reproducibility around an explicit correctness command and visible coverage statement.

---
*Phase: 02-correctness-baseline*
*Completed: 2026-06-04*
