---
phase: 06
slug: authenticated-and-specialized-mode-expansion
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-05
---

# Phase 06 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | CTest with `CudaAesKat` plus source-level grep checks |
| **Config file** | `CMakeLists.txt` |
| **Quick run command** | `rg "CCM|XTS|AES-KW|AES-KWP|GMAC|CMAC" tests/kat_main.cu docs main.cu aes_common.h CMakeLists.txt` |
| **Full suite command** | `cmake --build <build-dir> --config Release; ctest --test-dir <build-dir> --output-on-failure; <build-dir>/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/smoke-phase6` |
| **Estimated runtime** | Source checks: under 10 seconds; full CUDA build/test/benchmark depends on local CUDA/MSVC environment |

## Sampling Rate

- **After every task commit:** Run the plan-specific `rg` verification command.
- **After every plan wave:** Run the CMake/CTest path if the CUDA host compiler environment is available.
- **Before `$gsd-verify-work`:** Full source evidence must be green; runtime debt must be explicit if `cl.exe` blocks CUDA build.
- **Max feedback latency:** One task without a source-level verification command.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | MODE-05 | T-CCM-01 | CCM tag is computed and verified, not only round-tripped | source + KAT | `rg "aes128_ccm|aes256_ccm|CCM-128|CCM-256" aes_common.h CMakeLists.txt tests/kat_main.cu main.cu` | yes | pending |
| 06-01-02 | 01 | 1 | MODE-05 | T-CCM-02 | CCM docs disclose nonce/tag/AAD/message assumptions | docs | `rg "CCM|nonce|tag|AAD|authenticated" README.md docs` | yes | pending |
| 06-02-01 | 02 | 2 | MODE-06 | T-XTS-01 | XTS uses two-key/tweak semantics and storage-scope docs | source + KAT | `rg "aes128_xts|aes256_xts|XTS|sector|tweak" aes_common.h CMakeLists.txt tests/kat_main.cu main.cu docs` | yes | pending |
| 06-03-01 | 03 | 3 | MODE-07 | T-KW-01 | KW/KWP unwrap failure is observable for invalid wrapped data | source + KAT | `rg "AES-KW|AES-KWP|kw-128|kwp-128|unwrap|wrap" tests/kat_main.cu main.cu docs aes_common.h CMakeLists.txt` | yes | pending |
| 06-04-01 | 04 | 4 | MODE-08 | T-DOC-01 | MAC-only workloads are not described as encryption throughput | docs | `rg "GMAC|CMAC|MAC-only|authentication-only|not bulk encryption" README.md docs .planning/phases/06-authenticated-and-specialized-mode-expansion` | yes | pending |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:

- `CudaAesKat` already exists and is CTest-registered.
- `tests/kat_main.cu` already provides small deterministic vector style.
- `main.cu` already provides smoke benchmark flags `--runs`, `--sizes`, and `--bench-dir`.
- Runtime execution still requires a shell where `nvcc` can find `cl.exe` or a configured `CMAKE_CUDA_HOST_COMPILER`.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| CUDA/MSVC runtime verification | MODE-05, MODE-06, MODE-07 | Current shell may lack `cl.exe` in PATH | Run CMake configure/build, `ctest --test-dir <build> --output-on-failure`, and a one-size smoke benchmark from a Visual Studio Developer Command Prompt or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`. |
| Documentation claim review | MODE-08 | Requires human judgment about wording and benchmark interpretation | Review README and docs to confirm GMAC/CMAC are described as MAC/authentication workloads, not encryption throughput. |

## Validation Sign-Off

- [x] All tasks have source-level verification commands.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 10 seconds for source checks.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending execution
