---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: quality
---

# Testing

## Summary

There is no formal test framework in the repository. Correctness checks are embedded in the benchmark executable in `main.cu`, and performance measurements are emitted to console and CSV files. The local `cmake-build-debug/Testing/` directory is generated CMake state, not source-level tests.

## Test Frameworks

- No unit test framework was found.
- No CTest test definitions were found in the source CMake files.
- No CI workflow files were found in the mapped source listing.
- No known-answer-test fixture files were found.

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

`v3/main.cu` benchmarks the same modes but uses 3 runs and omits the 1 GiB size.

## CPU Baseline

- `cpu_aes_throughput()` in `main.cu` uses OpenSSL EVP as a CPU throughput baseline.
- The benchmark writes GPU rows to `bench/thr_gpu.csv` and CPU rows to `bench/thr_cpu.csv`.
- CPU and GPU outputs are not compared byte-for-byte in the CPU baseline function; CPU baseline is used for throughput comparison.

## Build Verification

The intended manual build flow in `README.md` is:

- Create a build directory.
- Run CMake in Release mode.
- Run the build tool.
- Execute `CudaProject`.

The current CMake files are Windows/MSVC-oriented, while the README uses Linux-style `make` commands.

## Test Gaps

- No automated known-answer tests for AES ECB, CTR, or GCM.
- No tests for partial block handling; benchmark sizes are block-aligned.
- No tag verification test harness for GCM.
- No AAD coverage for GCM.
- No negative tests for wrong GCM tags.
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

