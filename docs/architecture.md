# Architecture

CUDA-AES Benchmark is a single-process CUDA C++ benchmark executable. The canonical build target is `CudaProject` from the root `CMakeLists.txt`.

## Runtime Flow

1. `main.cu` parses benchmark and debug flags.
2. `init_T_tables()` prepares AES S-boxes, inverse tables, T-tables, and U-tables in device constant memory.
3. Host code generates input buffers, keys, and IV/counter data.
4. Host key expansion runs through `expandKey128()` or `expandKey256()`.
5. Expanded round keys are copied to device constant memory with `init_roundKeys()`.
6. Mode-specific CUDA kernels process ECB, CTR, or GCM data.
7. CUDA events record kernel-only GPU timing.
8. OpenSSL EVP records CPU baseline throughput.
9. Raw benchmark rows and metadata are written under `bench/` or `--bench-dir`.

## Canonical Source Boundary

The canonical implementation is the top-level source set:

- `main.cu`
- `aes_common.h`
- `aes_tables.cu`
- `aes128_ecb.cu`, `aes256_ecb.cu`
- `aes128_ctr.cu`, `aes256_ctr.cu`
- `aes128_gcm.cu`, `aes256_gcm.cu`
- `profiling_helpers.h`

`v3/` is a local experimental variant and is not the canonical build target. `cihangirTezcanAESimplementation/` is legacy/provenance code.

## Module Boundaries

- `main.cu` owns orchestration, validation, benchmark timing, OpenSSL comparison, and artifact writing.
- `aes_common.h` exposes shared kernel declarations and device constants.
- `aes_tables.cu` owns AES lookup tables and host key expansion.
- Mode files own their corresponding AES kernels.
- `profiling_helpers.h` wraps optional NVTX ranges.

## Constraints

- Device round keys are global process state.
- Benchmark sizes are block-aligned.
- GCM is benchmark/research code, not a general AEAD library API.
- Current GPU timing is kernel-only, not end-to-end throughput.

