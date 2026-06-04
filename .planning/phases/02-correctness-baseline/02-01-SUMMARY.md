---
phase: 02-correctness-baseline
plan: 01
subsystem: testing
tags: [cuda, aes, kat, ctest]
requires:
  - phase: 01-repository-and-build-foundation
    provides: Portable CMake/OpenSSL/CUDA package discovery
provides:
  - CTest-registered CUDA AES known-answer test executable
  - Deterministic ECB, CTR, and GCM vector coverage
  - Small correctness path separate from benchmark allocations
affects: [correctness, benchmark, documentation]
tech-stack:
  added: []
  patterns:
    - CTest registration for native CUDA correctness executable
key-files:
  created: [tests/kat_main.cu]
  modified: [CMakeLists.txt]
key-decisions:
  - "Use a separate CudaAesKat executable so benchmark behavior stays isolated from correctness tests."
  - "Use fixed NIST-style ECB/CTR vectors and deterministic GCM vectors/OpenSSL oracle checks."
patterns-established:
  - "Correctness checks run on small fixed buffers before benchmark interpretation."
requirements-completed: [TEST-01, TEST-02, TEST-03, TEST-04]
duration: 35min
completed: 2026-06-04
---

# Phase 2 Plan 01 Summary

**CTest-registered CUDA AES known-answer harness for ECB, CTR, and GCM**

## Accomplishments

- Added `tests/kat_main.cu` as a small deterministic correctness executable.
- Added `CudaAesKat` to CMake using the canonical CUDA kernel sources.
- Registered `cuda_aes_kat` through CTest.
- Covered ECB-128/256, CTR-128/256, and GCM-128/256 positive checks.
- Added GCM wrong-tag and tampered-ciphertext negative checks.

## Task Commits

Committed in the Phase 2 execution commit.

## Files Created/Modified

- `tests/kat_main.cu` - Native CUDA KAT runner.
- `CMakeLists.txt` - Shared kernel source list, `CudaAesKat` target, and `add_test`.

## Decisions Made

Used a separate executable instead of adding `--kat` to `CudaProject` so the benchmark entry point remains focused on benchmarking.

## Deviations from Plan

AES-256-GCM expected output is generated through OpenSSL in the deterministic KAT runner rather than hard-coded as a literal string. The input vector is fixed and OpenSSL is already the project CPU oracle dependency.

## Issues Encountered

Local configure did not reach build/test execution because `nvcc` cannot find `cl.exe` in this shell.

## Verification

- `rg "ECB|CTR|GCM|tag|expected|known|kat" tests main.cu` found vector and KAT runner code.
- `rg "enable_testing|add_test|CudaAesKat|--kat" CMakeLists.txt main.cu tests` found CTest wiring.
- `cmake -S . -B C:\tmp\cuda-aes-phase2-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86` was attempted and failed at CUDA compiler identification because `cl.exe` is not in `PATH`.

## Next Phase Readiness

Plan 02 can use the KAT runner as the correctness contract for GCM fixes.

---
*Phase: 02-correctness-baseline*
*Completed: 2026-06-04*
