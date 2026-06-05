---
phase: 05-confidentiality-mode-expansion
plan: 01
subsystem: cuda-modes
tags: [cuda, aes, cbc, kat, benchmark]
requires:
  - phase: 02-correctness-baseline
    provides: deterministic KAT harness and GCM correctness baseline
  - phase: 03-reproducible-benchmark-harness
    provides: raw benchmark CSV schema and timing lifecycle
provides:
  - AES-128 and AES-256 CBC CUDA mode entry points
  - CBC deterministic known-answer tests
  - CBC benchmark rows and CPU baseline selection
affects: [phase-05, phase-06, correctness, benchmarking, docs]
tech-stack:
  added: []
  patterns: [feedback-mode kernels using shared AES block helper]
key-files:
  created: [aes_block_device.cuh, aes128_cbc.cu, aes256_cbc.cu]
  modified: [aes_common.h, CMakeLists.txt, tests/kat_main.cu, main.cu, docs/benchmark-methodology.md]
key-decisions:
  - "CBC encryption is implemented as a single chained GPU path because CBC encryption has a true feedback dependency."
  - "CBC decryption is parallelized across blocks by decrypting each ciphertext block and XORing the IV or previous ciphertext block."
  - "Feedback-mode benchmark docs explicitly warn against reading CBC as CTR-like parallel throughput."
patterns-established:
  - "aes_block_device.cuh centralizes reusable device AES block encrypt/decrypt helpers for later feedback modes."
requirements-completed: [MODE-02]
duration: 11 min
completed: 2026-06-05
---

# Phase 05 Plan 01: CBC Mode Summary

**CBC AES-128/AES-256 CUDA kernels with NIST-style KAT coverage and benchmark rows**

## Performance

- **Duration:** 11 min
- **Started:** 2026-06-05T11:17:06+03:00
- **Completed:** 2026-06-05T11:28:01+03:00
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- Added reusable device AES block encrypt/decrypt helpers for feedback-mode kernels.
- Added AES-128 and AES-256 CBC encrypt/decrypt kernels and canonical CMake/prototype wiring.
- Added CBC-128 and CBC-256 deterministic encrypt/decrypt KATs.
- Added `cbc-128` and `cbc-256` benchmark rows with OpenSSL CBC CPU baselines.
- Documented feedback-mode benchmark dependency limitations.

## Task Commits

1. **Task 1: Add CBC kernel declarations and source wiring** - `3da8f60` (feat)
2. **Task 2: Add CBC known-answer tests** - `cb3fac7` (test)
3. **Task 3: Add CBC benchmark dispatch and baseline selection** - `401de7a` (feat)

## Files Created/Modified

- `aes_block_device.cuh` - Shared device AES block helpers for feedback modes.
- `aes128_cbc.cu` - AES-128 CBC encrypt/decrypt kernels.
- `aes256_cbc.cu` - AES-256 CBC encrypt/decrypt kernels.
- `aes_common.h` - Public CBC kernel declarations.
- `CMakeLists.txt` - CBC source files wired into canonical kernel source list.
- `tests/kat_main.cu` - CBC-128 and CBC-256 deterministic KATs.
- `main.cu` - CBC benchmark mode dispatch and OpenSSL baseline selection.
- `docs/benchmark-methodology.md` - Feedback-mode benchmark caveat.

## Decisions Made

- CBC encryption intentionally uses a single chained GPU execution path because each ciphertext block depends on the prior ciphertext block.
- CBC decryption uses block-level parallelism because each output block can use the current and previous ciphertext blocks independently.
- The shared device block helper was added to avoid duplicating AES round logic across CBC, CFB, and OFB.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Runtime CMake verification remains environment-limited in this shell. `cmake -S . -B C:\tmp\cuda-aes-phase5-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86` failed because `nvcc` could not find `cl.exe` in `PATH`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for Plan 05-02 to add CFB-128 and OFB using the shared AES block helper. Runtime CUDA verification still needs a Visual Studio Developer Command Prompt or `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`.

---
*Phase: 05-confidentiality-mode-expansion*
*Completed: 2026-06-05*
