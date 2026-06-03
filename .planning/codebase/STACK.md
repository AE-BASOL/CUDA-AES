---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: tech
---

# Stack

## Summary

This repository is a CUDA/C++ AES benchmark and experimentation codebase. The primary build target is a CUDA executable named `CudaProject` from the top-level sources in `CUDA-AES/`, with a second near-copy under `CUDA-AES/v3/` and an older/alternate implementation under `CUDA-AES/cihangirTezcanAESimplementation/`.

## Languages

- CUDA C++ in `*.cu` files.
- C++ headers in `*.h` and CUDA headers in `*.cuh`.
- CMake build definitions in `CMakeLists.txt`, `v3/CMakeLists.txt`, and `cihangirTezcanAESimplementation/CMakeLists.txt`.

## Runtime And Toolchain

- CMake minimum version is `3.28` for the top-level and `v3` targets in `CMakeLists.txt` and `v3/CMakeLists.txt`.
- CUDA Toolkit is required through `find_package(CUDAToolkit REQUIRED)`.
- C++ standard is set to C++17 in `CMakeLists.txt`.
- CUDA standard is set to CUDA C++17 in `CMakeLists.txt` and `v3/CMakeLists.txt`.
- The legacy implementation uses CMake `3.25` and CUDA C++14 in `cihangirTezcanAESimplementation/CMakeLists.txt`.
- GPU architecture is hard-coded to compute capability `86` in all CMake files, matching an RTX 30-series Ampere target.

## Core Dependencies

- CUDA runtime via `CUDA::cudart` in `CMakeLists.txt`.
- OpenSSL EVP APIs are included by `main.cu` for CPU baseline comparisons.
- Windows system libraries are linked in `CMakeLists.txt`: `crypt32.lib`, `ws2_32.lib`, `user32.lib`, `gdi32.lib`, `advapi32.lib`, and `kernel32.lib`.
- Optional NVTX support is enabled through `ENABLE_NVTX` and implemented in `profiling_helpers.h`.
- The code includes `<immintrin.h>` in `main.cu` for CPU carry-less multiply benchmarking.

## Build Configuration

- Top-level build source list is explicitly enumerated in `CMakeLists.txt`:
  - `aes_tables.cu`
  - `aes128_ctr.cu`
  - `aes128_gcm.cu`
  - `aes256_ctr.cu`
  - `aes256_gcm.cu`
  - `aes128_ecb.cu`
  - `aes256_ecb.cu`
  - `main.cu`
- `CUDA_SEPARABLE_COMPILATION` is enabled for `CudaProject`.
- CUDA compile options include `--use_fast_math`, `-lineinfo`, `--ptxas-options=-v`, and `-O3`.
- Host C++ compile options include `-O3 /EHsc`, which is MSVC-oriented.
- `CUDA_MAXIMUM_REGISTER_COUNT` is set to `64` for CUDA source files.

## Environment-Specific Configuration

The build files contain absolute Windows paths that must exist locally:

- CUDA include path: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include`
- OpenSSL include path: `C:/Users/efebasol/CLionProjects/aes_openSSL/openssl-3.3.3/include`
- OpenSSL libraries:
  - `C:/Users/efebasol/CLionProjects/aes_openSSL/openssl-3.3.3/libssl.lib`
  - `C:/Users/efebasol/CLionProjects/aes_openSSL/openssl-3.3.3/libcrypto.lib`
- Nsight Systems path in the `nsight-profile` target:
  - `C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.1.3/target-windows-x64/nsys.exe`
- Placeholder CUDA host compiler path:
  - `C:/Path/To/VisualStudio/VC/Tools/MSVC/<version>/bin/Hostx64/x64/cl.exe`

## Build Outputs

- README says the default executable is `CudaProject`.
- The checked local build directory `cmake-build-debug/` contains generated artifacts such as `CudaProject.exe`, `aes_ecb.exe`, Ninja files, CMake caches, `.pdb`, `.ilk`, `.lib`, and `.exp` files.
- `.gitignore` excludes CMake output folders, CUDA binary intermediates, IDE metadata, Nsight reports, benchmark binaries, and result folders.

## Profiling And Benchmarking Targets

- `ptx-dump` in `CMakeLists.txt` creates `bench/ptx_lookup.txt` by running `nvcc -ptx` against `aes128_ecb.cu`.
- `nsight-profile` in `CMakeLists.txt` profiles `CudaProject.exe` with Nsight Systems and writes output under the build directory.
- Runtime benchmark output is written under `bench/` by `main.cu`, including `thr_gpu.csv`, `thr_cpu.csv`, `gf_mult.csv`, and `ghash_partials.txt`.

## Versioned Variant

`v3/` largely mirrors the top-level CUDA benchmark, with notable benchmark parameter differences in `v3/main.cu`:

- `NUM_RUNS` is `3` instead of top-level `5`.
- `SIZES` omits the 1 GiB case and covers 1 MiB, 10 MiB, and 100 MiB.
- Round-trip check output is commented out in the `v3` benchmark loop.

