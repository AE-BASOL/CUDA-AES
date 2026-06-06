# v1.0.0 Release Notes

**Status:** Verified release package; GitHub publication remains a maintainer action.

These notes are the source-controlled basis for the canonical `v1.0.0` GitHub Release. The local release gate passed on 2026-06-06 from a Visual Studio 2022 Developer Command Prompt using the local NVIDIA/CUDA stack.

## Release Tag

- Tag: `v1.0.0`
- Release title: `CUDA-AES Benchmark v1.0.0`
- Release surface: GitHub Release for the `v1.0.0` tag

## Summary

CUDA-AES Benchmark v1.0.0 defines a reproducible benchmark-suite contract for CUDA AES workloads. It includes:

- portable CMake configuration with CUDA architecture and host compiler guidance;
- deterministic CTest known-answer coverage for implemented modes;
- reproducible benchmark raw artifact schema and summary generation;
- documentation for methodology, correctness, results, mode coverage, profiling, contribution, security, and maintenance;
- AES-128 and AES-256 benchmark coverage for ECB, CBC, CFB-128, OFB, CTR, GCM, CCM, XTS-AES, AES-KW, and AES-KWP.

This release does not claim production cryptography library API stability or production security hardening.

## Build And Test Status

The local release gate passed with this Windows CUDA/MSVC environment:

- OS: Windows
- Compiler: MSVC 1944
- CUDA runtime version: 12090
- NVIDIA driver version: 13010
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Compute capability: 8.6
- Build directory: `build-vs2022-release3`

Verified commands:

```text
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\cmake.exe" -S . -B build-vs2022-release3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CXX_COMPILER=cl -DOPENSSL_ROOT_DIR=C:/Strawberry/c -DOPENSSL_USE_STATIC_LIBS=FALSE'
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\cmake.exe" --build build-vs2022-release3 --config Release'
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-vs2022-release3 --output-on-failure'
```

CTest result: `100% tests passed, 0 tests failed out of 1`.

## Smoke Benchmark Gate

The smoke benchmark ran locally with one 1 MiB run per implemented workload:

```text
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && build-vs2022-release3\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke-local'
python scripts\summarize_benchmarks.py bench\v1-smoke-local\thr_gpu.csv bench\v1-smoke-local\thr_cpu.csv -o bench\v1-smoke-local\summary.md
```

All benchmark round-trip checks and AES-KW/AES-KWP wrap checks passed. The generated summary contains 36 rows.

## Raw Artifact Manifest

Use only this verified local artifact set for any `v1.0.0` release asset upload:

- `bench/v1-smoke-local/run_metadata.csv`
- `bench/v1-smoke-local/thr_gpu.csv`
- `bench/v1-smoke-local/thr_cpu.csv`
- `bench/v1-smoke-local/summary.md`

Do not attach stale throughput files. Do not include benchmark numbers in release notes unless they come from the verified release-gate artifact set.

## Known Limitations

- Runtime verification in an ordinary Windows shell can be blocked when `nvcc` cannot find `cl.exe`; use a Visual Studio Developer Command Prompt or pass an explicit CUDA host compiler.
- GPU timing rows are `kernel_only` and exclude allocation, transfers, validation, and summary generation.
- CPU baseline rows are OpenSSL comparison rows, not a controlled CPU performance study.
- GCM and CCM scope is limited to 96-bit IV/nonce, empty AAD, 16-byte tag where applicable, and full 16-byte blocks.
- XTS-AES scope is full 16-byte blocks with a 16-byte sector tweak; ciphertext stealing is out of scope.
- AES-KW and AES-KWP rows are key-wrap workloads, not bulk encryption throughput rows, and currently emit GPU rows only.
- Partial-block behavior, non-empty AAD, GMAC/CMAC benchmarking, charts, matrix automation, GitHub Pages, DOI releases, paper-style reports, and production library/API packaging are future work.

## Security, Maintenance, And Contributions

- Security reporting: [security policy](../SECURITY.md)
- Post-v1 maintenance loop: [maintenance.md](maintenance.md)
- Benchmark result contribution checklist: [benchmark-result-contributions.md](benchmark-result-contributions.md)
- Result package and claims policy: [results.md](results.md)

## Publication Checklist

- [x] Runtime release gate records `verification-passed`.
- [x] `CHANGELOG.md` has a dated `1.0.0` section and a fresh `Unreleased` section.
- [x] Release notes include the exact verified commands.
- [x] Raw artifact manifest lists only current verified files.
- [ ] GitHub Release draft uses the `v1.0.0` tag.
- [ ] Any release assets are produced by the verified release-gate run.
- [x] No ranking, fastest-in-world, or production-library claims are added.
