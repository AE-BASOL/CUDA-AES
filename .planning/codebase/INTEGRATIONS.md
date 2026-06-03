---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: tech
---

# Integrations

## Summary

This project has no web APIs, databases, queues, auth providers, or deployed service integrations. Its integrations are local native dependencies: CUDA, OpenSSL, Windows system libraries, and NVIDIA profiling tools.

## CUDA Runtime

- `main.cu` and all AES kernel modules depend on `cuda_runtime.h`.
- `CMakeLists.txt` requires `CUDAToolkit` and links `CUDA::cudart`.
- Device memory and synchronization are used throughout `main.cu` with APIs such as `cudaMalloc`, `cudaMallocHost`, `cudaMemcpy`, `cudaMemset`, `cudaDeviceSynchronize`, `cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`, and `cudaFree`.
- Constant memory uploads happen in `aes_tables.cu` through `cudaMemcpyToSymbol`.
- The core shared GPU constants are declared in `aes_common.h` and defined in `aes_tables.cu`.

## OpenSSL EVP

- `main.cu` includes `<openssl/evp.h>`.
- `cpu_aes_throughput` in `main.cu` uses `EVP_CIPHER_CTX_new`, `EVP_EncryptInit_ex`, `EVP_DecryptInit_ex`, `EVP_CIPHER_CTX_set_padding`, `EVP_EncryptUpdate`, `EVP_DecryptUpdate`, `EVP_EncryptFinal_ex`, `EVP_DecryptFinal_ex`, and `EVP_CIPHER_CTX_free`.
- Cipher selectors used in `main.cu` include `EVP_aes_128_ecb`, `EVP_aes_256_ecb`, `EVP_aes_128_ctr`, `EVP_aes_256_ctr`, `EVP_aes_128_gcm`, and `EVP_aes_256_gcm`.
- OpenSSL is linked via absolute local library paths in `CMakeLists.txt`.

## NVIDIA NVTX

- `profiling_helpers.h` conditionally includes `<nvtx3/nvToolsExt.h>` when `ENABLE_NVTX` is defined.
- `CMakeLists.txt` defines `ENABLE_NVTX` for `CudaProject`.
- `NVTX_PUSH` and `NVTX_POP` wrap benchmark sections in `main.cu`, including CTR preview and benchmark ranges.

## NVIDIA Nsight Systems

- `CMakeLists.txt` defines a custom `nsight-profile` target.
- The target launches a hard-coded Windows `nsys.exe` path and profiles the built `CudaProject.exe`.
- Output is configured under `${CMAKE_BINARY_DIR}/bench/my_run`.

## Windows Native Libraries

`CMakeLists.txt` links Windows system libraries to support OpenSSL and native runtime needs:

- `crypt32.lib`
- `ws2_32.lib`
- `user32.lib`
- `gdi32.lib`
- `advapi32.lib`
- `kernel32.lib`

## Filesystem Outputs

- `main.cu` uses `<filesystem>` to create the `bench/` directory.
- CSV and debug artifacts are written with `std::ofstream`:
  - `bench/thr_gpu.csv`
  - `bench/thr_cpu.csv`
  - `bench/gf_mult.csv`
  - `bench/ghash_partials.txt`
- The legacy file encryption implementation in `cihangirTezcanAESimplementation/file-encryption.cuh` reads and writes local files for AES CTR file encryption experiments.

## External Research Code

- `cihangirTezcanAESimplementation/TEZCAN_README.md` states that the legacy code is published CUDA AES optimization code associated with an IEEE publication.
- That folder is integrated as source code, not as a package manager dependency.

## Integration Risks

- Absolute Windows paths in `CMakeLists.txt` make the top-level build non-portable.
- The placeholder `CMAKE_CUDA_HOST_COMPILER` path in `CMakeLists.txt` is not a valid compiler path.
- `v3/CMakeLists.txt` duplicates the same absolute dependency paths and therefore shares the portability issue.
- The top-level README describes Linux-style `make` usage, while the current CMake file is MSVC/Windows-oriented.

