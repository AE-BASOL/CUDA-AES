---
phase: 06-authenticated-and-specialized-mode-expansion
plan: 02
subsystem: xts
tags: [cuda, aes, xts, storage, correctness, benchmark, docs]
requires:
  - phase: 06-01
    provides: CCM source, KAT, benchmark, and docs coverage
provides:
  - XTS-AES AES-128/AES-256 source declarations and build wiring
  - XTS deterministic full-block KAT coverage
  - XTS benchmark dispatch and storage-sector documentation
affects: [MODE-06, aes_common, aes_tables, aes_block_device, cmake, kat, benchmark, docs]
key-files:
  created: [aes128_xts.cu, aes256_xts.cu]
  modified: [aes_common.h, aes_tables.cu, aes_block_device.cuh, CMakeLists.txt, tests/kat_main.cu, main.cu, README.md, docs/modes.md, docs/correctness.md, docs/benchmark-methodology.md, docs/results.md]
requirements-completed: [MODE-06]
completed: 2026-06-05
---

# Phase 06 Plan 02: XTS-AES Coverage

## PLAN COMPLETE

Implemented source-level XTS-AES coverage for AES-128-XTS and AES-256-XTS in the canonical top-level build.

## Accomplishments

- Added `aes128_xts.cu` and `aes256_xts.cu`.
- Added `d_xtsTweakRoundKeys` and `init_xts_tweak_roundKeys()` so XTS uses separate data and tweak key schedules.
- Declared XTS encrypt/decrypt kernels in `aes_common.h`.
- Wired XTS sources into `CUDA_KERNEL_SOURCES`.
- Fixed `AesBlock` to use a byte/word union so block helper byte loads and AES word operations share the same storage.
- Added deterministic AES-128-XTS and AES-256-XTS full-block KATs with fixed sector tweak vectors.
- Added `xts-128` and `xts-256` benchmark dispatch and OpenSSL XTS CPU baseline selection.
- Updated docs to describe XTS as storage-oriented, confidentiality-only, two-key, full-block scoped, and without ciphertext stealing.

## Task Commits

This inline execution is committed as one Wave 2 change after plan completion.

## Verification

Source-level checks passed:

- `rg "aes128_xts|aes256_xts|aes128_xts.cu|aes256_xts.cu|tweak|sector" aes_common.h CMakeLists.txt aes128_xts.cu aes256_xts.cu`
- `rg "XTS-128|XTS-256|run_xts|sector|tweak" tests/kat_main.cu`
- `rg "xts-128|xts-256|isXts|XTS|sector|tweak|storage|ciphertext stealing" main.cu README.md docs`

Runtime verification was attempted:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase6-build -DCMAKE_BUILD_TYPE=Release
```

Result:

```text
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

This remains environment-limited. Run from a Visual Studio Developer Command Prompt or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>` to close runtime verification.

## Deviations From Plan

- Runtime CMake/CTest could not run in this shell because `nvcc` cannot find `cl.exe`.
- XTS coverage is intentionally full-block only; ciphertext stealing for non-block-multiple sectors is documented as out of scope.

## Self-Check

PASSED at source level with environment-limited runtime verification.
