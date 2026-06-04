---
phase: 2
slug: correctness-baseline
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-04
---

# Phase 2 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | CTest plus native C++/CUDA correctness executable or `CudaProject --kat` path |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir C:\tmp\cuda-aes-phase2-build --output-on-failure` |
| Full suite command | `ctest --test-dir C:\tmp\cuda-aes-phase2-build --output-on-failure` |
| Estimated runtime | Less than 30 seconds for small vectors once build is available |

## Sampling Rate

- After every task commit: run static checks for touched symbols and the quick CTest command if build environment is available.
- After every plan wave: run the full CTest command if configure/build succeeds.
- Before verify-work: all KATs and GCM negative tests must pass in an environment with CUDA host compiler configured.
- Max feedback latency: one small-vector test run per change.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | TEST-01/TEST-02/TEST-03 | T2-01 | Fixed vectors fail if bytes are wrong | KAT | `ctest --test-dir C:\tmp\cuda-aes-phase2-build --output-on-failure` | no | pending |
| 02-01-02 | 01 | 1 | TEST-04 | T2-02 | Smoke path avoids benchmark-size allocations | smoke | `CudaProject --kat` or CTest wrapper | no | pending |
| 02-02-01 | 02 | 2 | TEST-06/TEST-07/TEST-08 | T2-03/T2-04/T2-05 | GCM rejects bad tags and matches standard tags | KAT + negative | `ctest --test-dir C:\tmp\cuda-aes-phase2-build --output-on-failure` | no | pending |
| 02-03-01 | 03 | 3 | TEST-05 | T2-06 | Docs expose correctness status before benchmark claims | static docs | `rg "correctness|known-answer|GCM|ctest|--kat" README.md CMakeLists.txt` | yes | pending |

## Wave 0 Requirements

- Create a small correctness harness before GCM kernel edits.
- Add deterministic vector fixtures inline or in a small `tests/` source file.
- Add CMake/CTest registration so correctness checks are discoverable.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Local configure/build on Windows | TEST-01..TEST-08 | Current shell may not have `cl.exe` in PATH | Run from Visual Studio Developer Command Prompt or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`, then run CTest. |

## Validation Sign-Off

- [x] All plans include automated verify commands or environment-limited fallback.
- [x] Sampling continuity: every plan adds or uses correctness checks.
- [x] Wave 0 covers missing test infrastructure.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 30 seconds for small vectors.

