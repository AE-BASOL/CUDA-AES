# Changelog

All notable changes to CUDA-AES Benchmark will be documented here.

## Unreleased

- No changes yet.

## [1.0.0] - 2026-06-06

### Added

- Portable CMake and build-focused README guidance.
- CTest-registered known-answer tests for ECB, CTR, CBC, CFB, OFB, GCM, CCM, XTS-AES, AES-KW, and AES-KWP.
- Source-level GCM tag/authentication fixes for 96-bit IV, empty AAD, and full blocks.
- Reproducible benchmark metadata capture and Phase 3 raw CSV schema.
- Benchmark summary generator.
- Public documentation hub for architecture, correctness, benchmark methodology, results, and profiling.
- `v1.0.0` release notes, benchmark artifact manifest, maintenance loop, and GitHub template updates.

### Fixed

- Corrected AES host key expansion byte handling, device round-key word order, bytewise `MixColumns` indexing, and T-table final-round `ShiftRows` byte selection so the full CUDA AES KAT suite passes locally.

### Verification

- Release build completed from a Visual Studio 2022 Developer Command Prompt with CUDA architecture `86`, CUDA runtime `12090`, driver `13010`, and an NVIDIA GeForce RTX 3050 Ti Laptop GPU.
- `ctest --test-dir build-vs2022-release3 --output-on-failure` passed: 1/1 tests, including the full AES KAT executable.
- Local smoke benchmark passed all round-trip and key-wrap checks for 1 MiB inputs and generated `bench/v1-smoke-local/run_metadata.csv`, `thr_gpu.csv`, `thr_cpu.csv`, and `summary.md`.
