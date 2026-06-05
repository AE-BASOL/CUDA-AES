---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: arch
---

# Architecture

## Summary

The primary system is a single-process native benchmark executable. Host code in `main.cu` prepares inputs, expands AES keys, initializes GPU constant memory, launches CUDA kernels by AES mode, validates round trips, compares against OpenSSL CPU throughput, and writes benchmark artifacts. Device code is split by AES key size and mode into separate `.cu` modules.

## Primary Runtime Flow

1. `main.cu` parses command-line flags such as `--decrypt`, `--ctr-preview`, `--gcm-debug`, `--gf-mult`, and `--help`.
2. `main.cu` calls `init_T_tables()` from `aes_tables.cu` to generate and upload S-box, inverse S-box, encryption T-tables, and decryption U-tables to device constant memory.
3. For each mode in `MODES`, `main.cu` generates random input, key material, and IV material.
4. Host key expansion is performed through `expandKey128()` or `expandKey256()` from `aes_tables.cu`.
5. Expanded round keys are copied to device constant memory through `init_roundKeys()`.
6. The relevant CUDA kernel declared in `aes_common.h` is launched from `main.cu`.
7. CUDA events measure GPU elapsed time.
8. OpenSSL EVP computes CPU baseline throughput.
9. Results are printed and appended to CSV files under `bench/`.

## Module Boundaries

- `main.cu` is the orchestration, validation, benchmarking, and reporting layer.
- `aes_common.h` is the shared contract between host orchestration and kernel implementations.
- `aes_tables.cu` owns AES lookup tables, inverse tables, primary round-key storage, XTS tweak round-key storage, and host key expansion.
- `aes128_ecb.cu` and `aes256_ecb.cu` implement AES ECB encrypt/decrypt kernels.
- `aes128_cbc.cu` and `aes256_cbc.cu` implement AES CBC encrypt/decrypt kernels.
- `aes128_cfb.cu` and `aes256_cfb.cu` implement AES CFB-128 encrypt/decrypt kernels.
- `aes128_ofb.cu` and `aes256_ofb.cu` implement AES OFB encrypt/decrypt kernels.
- `aes128_ctr.cu` and `aes256_ctr.cu` implement AES CTR encrypt/decrypt kernels.
- `aes128_gcm.cu` and `aes256_gcm.cu` implement AES GCM encrypt/decrypt kernels.
- `aes128_ccm.cu` and `aes256_ccm.cu` implement benchmark-scoped AES CCM encrypt/decrypt kernels.
- `aes128_xts.cu` and `aes256_xts.cu` implement full-block XTS-AES encrypt/decrypt kernels.
- `aes128_kw.cu` and `aes256_kw.cu` implement AES-KW and AES-KWP wrap/unwrap kernels for fixed-size key-wrap records.
- `aes_block_device.cuh` provides shared device AES block encrypt/decrypt helpers for feedback modes.
- `profiling_helpers.h` abstracts optional NVTX range markers.
- `v3/` is a duplicated benchmark variant with small benchmark-loop differences.
- `cihangirTezcanAESimplementation/` is a separate legacy/alternate implementation, including exhaustive search and file encryption kernels.

## Shared Data Model

- AES block data is represented as byte buffers on the host and device.
- Kernel modules commonly reinterpret block data as `uint4`, `uint32_t*`, or `uint64_t*` for vectorized or word-level operations.
- AES round keys are expanded on the host and stored in device constant memory `d_roundKeys`.
- XTS-AES uses a second host-expanded tweak key schedule stored in device constant memory `d_xtsTweakRoundKeys`.
- S-boxes and T/U tables live in device constant memory and are declared in `aes_common.h`.

## GPU Execution Model

- ECB kernels process multiple 16-byte blocks per thread using strided loops.
- CBC encryption, CFB encryption, and OFB keystream generation are feedback-dependent and execute as chained paths for correctness.
- CBC and CFB decryption can expose block-level parallelism because each block can read the current and previous ciphertext blocks.
- CTR kernels generate per-block keystream from IV/counter state and XOR with input.
- GCM kernels combine CTR encryption/decryption with GHASH-like tag generation in a single kernel.
- CCM kernels use benchmark-scoped CBC-MAC plus CTR processing with 96-bit nonce, empty AAD, 16-byte tag, and full-block payload assumptions.
- XTS-AES kernels use two key schedules, a 16-byte sector tweak, and full 16-byte block data units. Ciphertext stealing is not implemented.
- AES-KW and AES-KWP kernels process fixed-size key-wrap records. AES-KW wraps 16-byte key data to 24-byte records; AES-KWP wraps 20-byte key data to 32-byte records.
- The top-level benchmark uses `THREADS_PER_BLOCK = 256` in `main.cu`.
- The legacy implementation uses `BLOCKS = 1024` and `THREADS = 1024` in `cihangirTezcanAESimplementation/AES_final.h`.

## Entry Points

- Main benchmark entry point: `main.cu:int main(int argc, char** argv)`.
- Legacy implementation entry point: `cihangirTezcanAESimplementation/AES_final.cu:int main()`.
- Debug/utility entry points inside `main.cu`:
  - `ctr_preview()`
  - `gf_mult_bench()`
  - `gcm_debug_run()`

## Command-Line Surface

`main.cu` recognizes:

- `--decrypt` to benchmark decrypt mode.
- `--ctr-preview` to print a 32-byte AES-128 CTR preview.
- `--gcm-debug` to run a small GCM debug path and write GHASH partials.
- `--gf-mult` to run CPU/GPU GF multiply benchmark.
- `--help` or `-h` to print usage.

The code contains commented-out `getopt_long` support for `--block N`; manual parsing is currently used for Windows compatibility, and `--block N` is disabled.

## Data Flow

- Random host plaintext is allocated with pinned memory in `main.cu`.
- Host input is copied to device buffers with `cudaMemcpy`.
- AES kernels write ciphertext or plaintext into device output buffers.
- Results are copied back to host memory for round-trip validation and optional CPU baseline work.
- CPU OpenSSL throughput reads host data and runs EVP operations into a temporary vector.
- CSV outputs persist mode, size, run, operation, timing, and throughput.

## Architectural Constraints

- Only one executable target is modeled at the top level.
- There is no library target for kernel modules, so reuse currently happens by source duplication rather than linkable components.
- Device constants are global process state. `init_roundKeys()` overwrites the active key schedule before each mode run.
- Most streaming/block modes assume 16-byte block-aligned sizes in the benchmark size list. AES-KW and AES-KWP use fixed record-size batching rather than streaming buffers.
- GCM code is optimized for benchmark experimentation, not a complete general-purpose AEAD API with AAD and robust tag verification semantics.
- CCM has the same benchmark-suite constraint: current coverage is not a complete AEAD API with arbitrary AAD, nonce lengths, tag lengths, or partial blocks.
- GMAC and CMAC are documented boundaries for future authentication/MAC benchmarking, not implemented encryption modes in the canonical executable.
