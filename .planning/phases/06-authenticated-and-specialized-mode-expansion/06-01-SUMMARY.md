---
phase: 06-authenticated-and-specialized-mode-expansion
plan: 01
subsystem: ccm
tags: [cuda, aes, ccm, correctness, benchmark, docs]
requires: []
provides:
  - CCM AES-128/AES-256 source declarations and build wiring
  - CCM deterministic ciphertext/tag KAT coverage
  - CCM benchmark mode dispatch and AEAD scope documentation
affects: [MODE-05, aes_common, cmake, kat, benchmark, docs]
key-files:
  created: [aes128_ccm.cu, aes256_ccm.cu]
  modified: [aes_common.h, CMakeLists.txt, tests/kat_main.cu, main.cu, README.md, docs/modes.md, docs/correctness.md, docs/benchmark-methodology.md, docs/results.md]
requirements-completed: [MODE-05]
completed: 2026-06-05
---

# Phase 06 Plan 01: CCM Coverage

## PLAN COMPLETE

Implemented source-level CCM coverage for AES-128 and AES-256 in the canonical top-level build.

## Accomplishments

- Added `aes128_ccm.cu` and `aes256_ccm.cu`.
- Declared CCM encrypt/decrypt kernels in `aes_common.h`.
- Wired CCM sources into `CUDA_KERNEL_SOURCES` for both `CudaProject` and `CudaAesKat`.
- Added deterministic CCM AES-128 and AES-256 ciphertext/tag KATs using 12-byte nonce, empty AAD, 16-byte tag, and full-block payload scope.
- Added CCM wrong-tag rejection coverage.
- Added `ccm-128` and `ccm-256` to benchmark mode dispatch with tag-aware round-trip checks.
- Added a CCM-specific OpenSSL EVP CPU timing helper.
- Updated README and docs to state CCM AEAD scope and limitations.

## Task Commits

This inline execution is committed as one Wave 1 change after plan completion.

## Verification

Source-level checks passed:

- `rg "aes128_ccm|aes256_ccm|aes128_ccm.cu|aes256_ccm.cu" aes_common.h CMakeLists.txt aes128_ccm.cu aes256_ccm.cu`
- `rg "CCM-128|CCM-256|wrong tag|run_ccm" tests/kat_main.cu`
- `rg "ccm-128|ccm-256|isCcm|CCM|nonce|tag|AAD|authenticated" main.cu README.md docs`

Runtime verification was attempted:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase6-build -DCMAKE_BUILD_TYPE=Release
```

Result:

```text
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

This matches carried verification debt. Run from a Visual Studio Developer Command Prompt or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>` to close runtime verification.

## Deviations From Plan

- Runtime CMake/CTest could not run in this shell because `nvcc` cannot find `cl.exe`.
- The CCM implementation is intentionally scoped to 12-byte nonce, empty AAD, 16-byte tag, and full 16-byte blocks, matching the Phase 6 research boundary.

## Self-Check

PASSED at source level with environment-limited runtime verification.
