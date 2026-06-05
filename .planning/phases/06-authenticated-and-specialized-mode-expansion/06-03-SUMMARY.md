---
phase: 06-authenticated-and-specialized-mode-expansion
plan: 03
subsystem: key-wrap
tags: [cuda, aes, aes-kw, aes-kwp, correctness, benchmark, docs]
requires:
  - phase: 06-01
    provides: CCM source, KAT, benchmark, and docs coverage
  - phase: 06-02
    provides: XTS-AES source, KAT, benchmark, and docs coverage
provides:
  - AES-KW/AES-KWP AES-128/AES-256 source declarations and build wiring
  - AES-KW/AES-KWP deterministic wrap/unwrap KAT coverage
  - AES-KW tamper unwrap rejection coverage
  - GPU key-wrap benchmark rows and workload documentation
affects: [MODE-07, aes_common, cmake, kat, benchmark, docs]
key-files:
  created: [aes128_kw.cu, aes256_kw.cu]
  modified: [aes_common.h, CMakeLists.txt, tests/kat_main.cu, main.cu, README.md, docs/modes.md, docs/correctness.md, docs/benchmark-methodology.md, docs/results.md]
requirements-completed: [MODE-07]
completed: 2026-06-05
---

# Phase 06 Plan 03: AES-KW And AES-KWP Coverage

## PLAN COMPLETE

Implemented source-level AES-KW and AES-KWP coverage for AES-128 and AES-256 key-encryption keys in the canonical top-level build.

## Accomplishments

- Added `aes128_kw.cu` and `aes256_kw.cu`.
- Declared AES-KW and AES-KWP wrap/unwrap kernels in `aes_common.h`.
- Wired key-wrap sources into `CUDA_KERNEL_SOURCES`.
- Added deterministic AES-KW AES-128/AES-256 wrap and unwrap KATs.
- Added deterministic AES-KWP AES-128/AES-256 wrap and unwrap KATs.
- Added AES-KW tamper unwrap rejection coverage through unwrap status bytes.
- Added `kw-128`, `kw-256`, `kwp-128`, and `kwp-256` benchmark labels.
- Added a key-wrap-specific benchmark branch that batches fixed-size records and emits GPU `WRAP`/`UNWRAP` rows.
- Updated docs so AES-KW/AES-KWP rows are described as key-management workloads, not bulk encryption throughput.

## Task Commits

This inline execution is committed as one Wave 3 change after plan completion.

## Verification

Source-level checks passed:

- `rg "aes128_kw|aes256_kw|kwp|aes128_kw.cu|aes256_kw.cu" aes_common.h CMakeLists.txt aes128_kw.cu aes256_kw.cu`
- `rg "AES-KW|AES-KWP|KW-128|KW-256|KWP-128|KWP-256|tamper|unwrap" tests/kat_main.cu`
- `rg "kw-128|kw-256|kwp-128|kwp-256|AES-KW|AES-KWP|key-wrap" main.cu README.md docs`

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
- AES-KW/AES-KWP benchmark rows currently emit GPU key-wrap workload rows only. CPU baseline rows are documented as not emitted for key-wrap modes yet.

## Self-Check

PASSED at source level with environment-limited runtime verification.
