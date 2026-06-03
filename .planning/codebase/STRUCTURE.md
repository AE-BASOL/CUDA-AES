---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: arch
---

# Structure

## Repository Layout

- `README.md` documents build and benchmark usage.
- `CMakeLists.txt` builds the primary `CudaProject` executable.
- `.gitignore` excludes build outputs, IDE files, CUDA intermediates, profiler reports, and benchmark result folders.
- `main.cu` contains the primary benchmark executable.
- `aes_common.h` declares shared device constants, kernel entry points, and host utility functions.
- `aes_tables.cu` defines device constant memory and host table/key setup utilities.
- `aes128_ecb.cu`, `aes256_ecb.cu`, `aes128_ctr.cu`, `aes256_ctr.cu`, `aes128_gcm.cu`, and `aes256_gcm.cu` implement mode-specific kernels.
- `profiling_helpers.h` defines NVTX helper macros.
- `v3/` contains a second version of the same benchmark family.
- `cihangirTezcanAESimplementation/` contains legacy/alternate CUDA AES code and its own build file.
- `cmake-build-debug/` is a generated local build directory present in the workspace.
- `.idea/` contains JetBrains CLion project metadata.

## Primary Source Files

- `aes128_ecb.cu`: AES-128 ECB encrypt/decrypt kernels using register-local block state and per-thread striding.
- `aes256_ecb.cu`: AES-256 ECB encrypt/decrypt kernels.
- `aes128_ctr.cu`: AES-128 CTR keystream generation and encrypt/decrypt kernels.
- `aes256_ctr.cu`: AES-256 CTR keystream generation and encrypt/decrypt kernels.
- `aes128_gcm.cu`: AES-128 GCM encryption/decryption kernels with GF(2^128) multiplication.
- `aes256_gcm.cu`: AES-256 GCM encryption/decryption kernels.
- `aes_tables.cu`: S-box, inverse S-box, T-table, U-table, and key schedule setup.
- `main.cu`: benchmark runner, CLI parsing, GPU launch orchestration, OpenSSL CPU comparison, CSV output, and debug routines.

## Header Files

- `aes_common.h` is the only top-level shared API header for the primary implementation.
- `profiling_helpers.h` provides `NVTX_PUSH` and `NVTX_POP`, which compile to no-ops when `ENABLE_NVTX` is not defined.
- `cihangirTezcanAESimplementation/AES_final.h` contains typedefs, constants, helper functions, AES tables, and declarations for the legacy implementation.

## Version Variant

`v3/` mirrors the top-level files:

- `v3/CMakeLists.txt`
- `v3/main.cu`
- `v3/aes_common.h`
- `v3/aes_tables.cu`
- `v3/aes128_ecb.cu`
- `v3/aes256_ecb.cu`
- `v3/aes128_ctr.cu`
- `v3/aes256_ctr.cu`
- `v3/aes128_gcm.cu`
- `v3/aes256_gcm.cu`
- `v3/profiling_helpers.h`

This appears to be a forked local variant rather than a formal package or module boundary.

## Legacy Implementation Folder

`cihangirTezcanAESimplementation/` contains:

- `CMakeLists.txt` building a `cuda_aes` executable from `AES_final.cu`.
- `TEZCAN_README.md` with provenance and performance claims for the published AES optimization code.
- `AES_final.cu` as the legacy executable entry point.
- `AES_final.h` with common constants, tables, types, and helpers.
- `128-es.cuh` and `256-es.cuh` for exhaustive search kernels.
- `128-ctr.cuh` and `256-ctr.cuh` for CTR kernels.
- `file-encryption.cuh` for AES CTR file encryption experiments.

`AES_final.cu` includes `192-es.cuh` and `192-ctr.cuh`, but those files were not present in the tracked file listing from `rg --files`; this is a likely build issue for the legacy folder.

## Generated And Local-Only Directories

- `cmake-build-debug/` includes generated CMake/Ninja files and binaries. It should not be treated as source.
- `.idea/` includes CLion metadata. It is ignored by `.gitignore` but present in the workspace.
- Runtime benchmark outputs are expected under `bench/`, which is created by `main.cu` at runtime.

## Naming Conventions

- Primary kernels follow `{aes}{keybits}_{mode}_{operation}` naming, for example `aes128_ecb_encrypt`.
- Mode files follow `aes{keybits}_{mode}.cu`.
- Shared constants use `d_` prefixes for device constants, such as `d_sbox` and `d_roundKeys`.
- Host table copies use `h_` prefixes in `aes_tables.cu`, such as `h_sbox`.
- Legacy code uses short typedefs `u8`, `u16`, `u32`, and `u64` in `cihangirTezcanAESimplementation/AES_final.h`.

