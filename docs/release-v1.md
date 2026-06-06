# v1.0.0 Release Notes Draft

**Status:** Release candidate; runtime release gate is currently `verification-blocked`.

These notes are the source-controlled draft for the canonical `v1.0.0` GitHub Release. They are not final publication notes until the release gate records a passing Release configure/build, CTest run, smoke benchmark, and summary generation from current raw artifacts.

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

Release publication requires these commands to pass in a CUDA host compiler environment:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

On Windows, run from a Visual Studio Developer Command Prompt or pass:

```text
-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
```

Current source-controlled status: `verification-blocked` in this shell because `nvcc` cannot find `cl.exe` during CUDA compiler detection. Keep this document as a release candidate until `.planning/phases/08-release-and-maintenance-loop/08-VERIFICATION.md` records `verification-passed`.

## Smoke Benchmark Gate

After CTest passes, run an isolated smoke benchmark and generate a summary from the current raw CSV files:

```text
.\build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke
python scripts\summarize_benchmarks.py bench\v1-smoke\thr_gpu.csv bench\v1-smoke\thr_cpu.csv -o bench\v1-smoke\summary.md
```

Use `./build/CudaProject` on single-config generators where appropriate.

## Benchmark Environment Fields

The release artifact record must name:

- commit hash;
- GPU model;
- GPU compute capability;
- CUDA Toolkit version;
- NVIDIA driver version;
- operating system;
- compiler and version;
- CMake configure command and cache-relevant options;
- benchmark command;
- GPU clocks and persistence-mode note.

## Raw Artifact Manifest

Attach raw benchmark artifacts only from the current verified release-gate run:

- `run_metadata.csv`
- `thr_gpu.csv`
- `thr_cpu.csv` where applicable
- generated `summary.md`

Do not attach stale throughput files. Do not include benchmark numbers in release notes unless they come from the verified release-gate artifact set.

## Known Limitations

- Runtime verification in an ordinary Windows shell can be blocked when `nvcc` cannot find `cl.exe`.
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

- [ ] Runtime release gate records `verification-passed`; current attempt is `verification-blocked` on missing `cl.exe`.
- [ ] `CHANGELOG.md` has a dated `1.0.0` section and a fresh `Unreleased` section.
- [ ] Release notes include the exact verified commands.
- [ ] Raw artifact manifest lists only current verified files.
- [ ] GitHub Release draft uses the `v1.0.0` tag.
- [ ] Any release assets are produced by the verified release-gate run.
- [ ] No ranking, fastest-in-world, or production-library claims are added.

Until these items pass, keep this document and any GitHub release draft marked as a release candidate.
