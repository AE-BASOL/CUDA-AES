---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: quality
---

# Testing

## Summary

The repository now has a small CTest-registered known-answer-test executable for correctness, plus the older benchmark-embedded round-trip checks in `main.cu`. Performance measurements are emitted to console and CSV files. Generated CMake `Testing/` directories are local build state, not source-level tests.

## Test Frameworks

- CTest is enabled in the root `CMakeLists.txt`.
- `CudaAesKat` is registered as the `cuda_aes_kat` test.
- No CI workflow files were found in the mapped source listing.
- Known-answer vectors live in `tests/kat_main.cu`.

## Known-Answer Test Coverage

`tests/kat_main.cu` covers:

- ECB-128 encrypt/decrypt
- ECB-256 encrypt/decrypt
- CTR-128 encrypt/decrypt
- CTR-256 encrypt/decrypt
- GCM-128 ciphertext/tag/decrypt tag
- GCM-256 ciphertext/tag/decrypt tag
- GCM wrong-tag rejection
- GCM tampered-ciphertext rejection

Run it with:

```bash
ctest --test-dir build --output-on-failure
```

The GCM correctness scope is 96-bit IV, empty AAD, and full 16-byte blocks. Broader AEAD API coverage, partial-block behavior, and non-empty AAD are future work.

## Embedded Correctness Checks

`main.cu` performs round-trip checks during the benchmark loop:

- It creates random original plaintext.
- It encrypts on the GPU.
- It decrypts on the GPU.
- It copies decrypted output back to host.
- It compares every byte against the original plaintext.
- It prints `PASS` or `FAIL` for each mode and message size.

The top-level `main.cu` prints `ROUND_TRIP_CHECK` rows. In `v3/main.cu`, the round-trip check label output is commented out, although the surrounding validation code remains present.

## Debug Modes

The executable supports small focused debug paths:

- `--ctr-preview` runs AES-128 CTR over a 32-byte zero buffer and prints the output bytes.
- `--gcm-debug` encrypts a 64-byte sample, computes partial GHASH data, and writes `bench/ghash_partials.txt`.
- `--gf-mult` compares CPU and GPU GF multiplication throughput and writes `bench/gf_mult.csv`.

## Benchmark Coverage

Top-level `main.cu` benchmarks:

- Modes: `ecb-128`, `ecb-256`, `ctr-128`, `ctr-256`, `gcm-128`, and `gcm-256`.
- Sizes: 1 MiB, 10 MiB, 100 MiB, and 1 GiB.
- Runs: 5 per mode/size combination.
- Operations: encryption by default, decryption when `--decrypt` is supplied.
- Configurable smoke parameters: `--runs N`, `--sizes bytes[,bytes]`, and `--bench-dir PATH`.
- Raw output schema version: `phase3.v1`.
- GPU rows use `timing_scope=kernel_only`; CPU rows use `timing_scope=cpu_baseline`.
- `run_metadata.csv` captures command line, run count, selected sizes, OS/compiler hints, CUDA runtime/driver versions, GPU model, compute capability, and clocks/persistence note.

`v3/main.cu` benchmarks the same modes but uses 3 runs and omits the 1 GiB size.

## CPU Baseline

- `cpu_aes_throughput()` in `main.cu` uses OpenSSL EVP as a CPU throughput baseline.
- The benchmark writes GPU rows to `bench/thr_gpu.csv` and CPU rows to `bench/thr_cpu.csv`.
- CPU and GPU outputs are not compared byte-for-byte in the CPU baseline function; CPU baseline is used for throughput comparison.

## Reproducibility Smoke Command

After building `CudaProject`, run a small benchmark without the 1 GiB allocation:

```bash
./build/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/smoke
```

Expected raw artifacts:

- `bench/smoke/run_metadata.csv`
- `bench/smoke/thr_gpu.csv`
- `bench/smoke/thr_cpu.csv`

## Build Verification

The intended manual build flow in `README.md` is:

- Create a build directory.
- Run CMake in Release mode.
- Run the build tool.
- Execute `CudaProject`.

The current CMake files are Windows/MSVC-oriented, while the README uses Linux-style `make` commands.

## Test Gaps

- No tests for partial block handling; benchmark sizes and Phase 2 KATs are block-aligned.
- No non-empty AAD coverage for GCM.
- No memory checker or sanitizer configuration.
- No CI build matrix across CUDA versions, host compilers, or GPU architectures.
- No tests covering the legacy `cihangirTezcanAESimplementation/` executable.

## Practical Manual Checks

Useful manual checks before changing kernels:

- Build the top-level `CudaProject` target from `CMakeLists.txt`.
- Run `CudaProject --ctr-preview` and preserve output as a regression reference.
- Run `CudaProject --gcm-debug` and inspect `bench/ghash_partials.txt`.
- Run the default benchmark on a smaller size set before expensive 1 GiB runs.
- Run with `--decrypt` to exercise decrypt paths.
- Compare GPU ciphertext for small fixed inputs against OpenSSL outputs in a future automated test.
