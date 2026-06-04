---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: concerns
---

# Concerns

## Summary

The codebase is functional benchmark/experimentation code, but it has portability, maintainability, correctness-verification, and security-hardening concerns. The highest-risk items are build reproducibility, duplicated implementations, weak automated testing, and GCM semantics.

## Build Reproducibility

- `CMakeLists.txt` and `v3/CMakeLists.txt` contain absolute local paths for CUDA, OpenSSL, and Nsight.
- `CMakeLists.txt` ends with a placeholder `CMAKE_CUDA_HOST_COMPILER` path, which is not a usable compiler location.
- README build instructions use Linux-style `make`, while the current build configuration links Windows `.lib` files and uses MSVC flags.
- `cmake-build-debug/` is present in the workspace with generated binaries and build files, even though build outputs are ignored.
- `cihangirTezcanAESimplementation/AES_final.cu` includes `192-es.cuh` and `192-ctr.cuh`, but those files were not present in the mapped file list.

## Correctness Risks

- There are no formal known-answer tests for AES modes.
- GCM implementation does not expose a complete AEAD interface with AAD handling.
- GCM decrypt kernels accept a `tag` parameter, but comments indicate host-side verification expectations and the implementation writes `tagOut`; explicit reject/accept semantics are not visible as a public API.
- OpenSSL EVP return values are not checked in `cpu_aes_throughput()`.
- The benchmark uses random inputs, making regression comparison harder unless seeds and outputs are recorded.
- Partial-block behavior is not covered because benchmark sizes are multiples of 16 bytes.

## Memory And Resource Risks

- CUDA resources are manually managed throughout `main.cu`.
- There are paths where allocation succeeds and later operations can exit without centralized cleanup.
- Phase 3 destroys CUDA events in the main benchmark loop and GF multiply benchmark; lifecycle consistency should still be reviewed if new timed paths are introduced.
- Reinterpret casts to wider word types require alignment assumptions that are true for CUDA allocations but less explicit for arbitrary user buffers.
- Pinned host memory allocations at 1 GiB can pressure host memory and may fail on smaller systems.

## Security And Cryptography Risks

- The project is benchmark code, not a hardened cryptographic library.
- GCM lacks complete authenticated-encryption API semantics and negative authentication tests.
- Debug output prints key and IV prefixes in `main.cu`, which is acceptable for benchmarking but unsafe in production contexts.
- The legacy code prints candidate/found keys in exhaustive-search paths under `cihangirTezcanAESimplementation/`.
- AES implementations should be validated against standard test vectors before any security-sensitive use.

## Performance Measurement Risks

- Benchmark results depend heavily on hard-coded architecture `86`.
- CPU baseline uses OpenSSL but does not pin CPU frequency, affinity, or account for warmup beyond repeated runs.
- GPU benchmark includes memory allocation/copy setup outside the timed event region for kernel timing, so results are kernel-throughput oriented rather than end-to-end throughput. Phase 3 raw rows label this with `timing_scope=kernel_only`.
- GCM tag generation has custom partial GHASH composition logic that warrants focused validation and profiling.
- NVTX support is always enabled in CMake through `ENABLE_NVTX`, which may require headers/libraries depending on the CUDA Toolkit installation.

## Maintainability Risks

- Top-level and `v3/` code are duplicated, so fixes can diverge.
- AES-128 and AES-256 modes have similar duplicated logic instead of shared templates or common helpers.
- GCM kernels duplicate CTR counter encryption logic.
- Comments include encoding artifacts, making parts of `profiling_helpers.h` and CMake comments harder to read.
- There is no clear module boundary between production-like benchmark code and exploratory debug code in `main.cu`.

## Repository Hygiene

- `.idea/` and `cmake-build-debug/` exist in the workspace despite being ignored patterns.
- The checked generated build directory can mislead codebase scans and should not be used as source of truth.
- There is no CI configuration to keep the build reproducible.
- There is no dependency discovery for OpenSSL; paths are manually embedded.

## Recommended Next Fixes

- Replace absolute dependency paths in `CMakeLists.txt` with cache variables or `find_package(OpenSSL REQUIRED)`.
- Remove or repair the placeholder `CMAKE_CUDA_HOST_COMPILER`.
- Decide whether `v3/` is needed; if so, document its purpose, otherwise consolidate it.
- Add known-answer tests for ECB, CTR, and GCM.
- Add GCM negative tag tests and AAD support if GCM is intended beyond benchmarking.
- Add a small deterministic benchmark/test mode that avoids 1 GiB allocation by default.
- Keep generated build outputs out of repository operations and scans.

## Imported Review Findings

The 2026-06-04 main branch review in `.planning/reviews/2026-06-04-main-branch-code-review.md` escalates several concerns into blockers:

- GCM decrypt currently accepts unauthenticated ciphertext because computed tags are not checked before plaintext is accepted.
- GCM IV/counter broadcast uses warp-local `__shfl_sync`, which is incorrect for threads outside the first warp in a 256-thread block.
- GCM tag output is not standard AES-GCM because it omits the length block and final `E(K, J0)` XOR.
- Public CMake configuration is blocked by local absolute paths and CUDA host compiler detection problems.
- Benchmark timing paths used to create CUDA events without destroying them; Phase 3 closes this in the main benchmark loop and GF multiply benchmark.
