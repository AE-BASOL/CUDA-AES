---
phase: 08-release-and-maintenance-loop
status: passed
verified_at: 2026-06-06T22:06:40+03:00
requirements: [REPO-04, MAINT-01, MAINT-02, MAINT-03, MAINT-04]
release_gate: verification-passed
---

# Phase 8 Verification: Release And Maintenance Loop

## Result

Phase 8 source-controlled release and maintenance artifacts pass. The runtime release gate also passes locally after running CMake from a Visual Studio 2022 Developer Command Prompt and correcting AES correctness issues exposed by the KAT executable.

The earlier ordinary-shell blocker was:

```text
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

That environment blocker is resolved by entering the VS developer environment before configure/build. The follow-up KAT failures were source-level AES byte-order/state-layout bugs and are now fixed.

## Requirement Evidence

| Requirement | Status | Evidence |
|-------------|--------|----------|
| REPO-04 | Passed | `docs/release-v1.md` defines the `v1.0.0` release tag/title, release notes, verified build/test commands, smoke benchmark gate, benchmark environment fields, raw artifact manifest, known limitations, maintenance/security links, and publication checklist. |
| MAINT-01 | Passed | Markdown issue and pull request templates guide useful bug reports, mode requests, benchmark-result issues, and PRs. Templates include GitHub chooser frontmatter where applicable. |
| MAINT-02 | Passed | `docs/benchmark-result-contributions.md` defines the canonical benchmark result checklist with commit hash, CTest status, configure/build command, benchmark command, raw CSV artifacts, generated `summary.md`, environment metadata, clocks/persistence note, and claims acknowledgement. |
| MAINT-03 | Passed | `docs/maintenance.md` defines changelog, roadmap, triage, benchmark review, security, and v2-boundary loops. `CHANGELOG.md` now has a dated `1.0.0` section plus a fresh `Unreleased` section. |
| MAINT-04 | Passed | `SECURITY.md` prefers GitHub private vulnerability reporting when enabled and gives a safe public fallback that does not include sensitive details. |

## Release Gate Commands

Configure:

```text
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\cmake.exe" -S . -B build-vs2022-release3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CXX_COMPILER=cl -DOPENSSL_ROOT_DIR=C:/Strawberry/c -DOPENSSL_USE_STATIC_LIBS=FALSE'
```

Build:

```text
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\cmake.exe" --build build-vs2022-release3 --config Release'
```

CTest:

```text
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-vs2022-release3 --output-on-failure'
```

Result:

```text
100% tests passed, 0 tests failed out of 1
```

Smoke benchmark:

```text
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && build-vs2022-release3\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke-local'
```

Summary generation:

```text
python scripts\summarize_benchmarks.py bench\v1-smoke-local\thr_gpu.csv bench\v1-smoke-local\thr_cpu.csv -o bench\v1-smoke-local\summary.md
```

Result:

- all benchmark round-trip checks passed;
- all AES-KW/AES-KWP wrap checks passed;
- `summary.md` generated 36 summary rows.

## Environment

From `bench/v1-smoke-local/run_metadata.csv`:

| Field | Value |
|-------|-------|
| OS | windows |
| Compiler | msvc 1944 |
| CUDA runtime version | 12090 |
| CUDA driver version | 13010 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| Compute capability | 8.6 |
| Global memory bytes | 4294508544 |
| Benchmark directory | `bench\v1-smoke-local` |
| GPU clocks/persistence | not captured; record fixed clocks and persistence mode manually when publishing comparative claims |

## Verified Artifact Set

- `bench/v1-smoke-local/run_metadata.csv`
- `bench/v1-smoke-local/thr_gpu.csv`
- `bench/v1-smoke-local/thr_cpu.csv`
- `bench/v1-smoke-local/summary.md`

## Correctness Fixes Verified

The passing KAT suite covers ECB, CTR, CBC, CFB, OFB, GCM, CCM, XTS-AES, AES-KW, and AES-KWP for AES-128/AES-256 where implemented. The local fixes corrected:

- host key expansion byte handling for AES round schedule words;
- device round-key upload order for little-endian word kernels;
- bytewise `MixColumns` and `InvMixColumns` indexing in the shared AES block core and ECB kernels;
- T-table final-round `SubBytes+ShiftRows` byte selection in CTR/GCM/CCM kernels.

## Claim Guardrails

No unsupported fastest-in-world, universal ranking, or production cryptography library claim is added. Benchmark artifacts are raw local smoke artifacts and the GPU timing rows remain `kernel_only`.

## Verification Commands

- `cmd.exe /s /c 'call "...VsDevCmd.bat" -arch=x64 && "...cmake.exe" --build build-vs2022-release3 --config Release'` - passed.
- `cmd.exe /s /c 'call "...VsDevCmd.bat" -arch=x64 && "...ctest.exe" --test-dir build-vs2022-release3 --output-on-failure'` - passed.
- `cmd.exe /s /c 'call "...VsDevCmd.bat" -arch=x64 && build-vs2022-release3\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke-local'` - passed.
- `python scripts\summarize_benchmarks.py bench\v1-smoke-local\thr_gpu.csv bench\v1-smoke-local\thr_cpu.csv -o bench\v1-smoke-local\summary.md` - passed.
- `rg "1.0.0|verification-passed|v1-smoke-local|CTest|summary.md|known limitations" docs\release-v1.md .planning\phases\08-release-and-maintenance-loop\08-VERIFICATION.md CHANGELOG.md` - passed.
- Guardrail check: no unsupported ranking claim was added to `docs/release-v1.md`.

## Verification Complete
