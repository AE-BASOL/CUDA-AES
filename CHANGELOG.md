# Changelog

All notable changes to CUDA-AES Benchmark will be documented here.

## Unreleased

### Added

- Portable CMake and build-focused README guidance.
- CTest-registered known-answer tests for ECB, CTR, and GCM.
- Source-level GCM tag/authentication fixes for 96-bit IV, empty AAD, and full blocks.
- Reproducible benchmark metadata capture and Phase 3 raw CSV schema.
- Benchmark summary generator.
- Public documentation hub for architecture, correctness, benchmark methodology, results, and profiling.
- Release-candidate `v1.0.0` notes, benchmark artifact manifest, maintenance loop, and GitHub template updates.

### Known Verification Debt

- Runtime CMake/CTest/benchmark verification needs a shell where `nvcc` can find `cl.exe` on Windows.
- The `v1.0.0` release candidate remains blocked from final publication until Release configure/build, CTest, smoke benchmark, and summary generation pass in a shell where `nvcc` can find `cl.exe` or an explicit CUDA host compiler is provided.
