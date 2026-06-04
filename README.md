# CUDA-AES Benchmark

CUDA-AES is a CUDA benchmark and research repository for AES kernels on NVIDIA GPUs. The current canonical implementation benchmarks AES-128 and AES-256 in ECB, CTR, and GCM-shaped code paths, with OpenSSL used as the CPU comparison path.

This repository is being prepared as an open-source GPU AES benchmark project. It is not production cryptography software.

## Current Status

- Canonical source lives in the top-level `.cu` and `.h` files built by the root `CMakeLists.txt`.
- `v3/` is a local experimental variant and is not the canonical Phase 1 build target.
- `cihangirTezcanAESimplementation/` is legacy/provenance code.
- Phase 2 adds known-answer checks for ECB, CTR, and GCM. GCM coverage is limited to 96-bit IV, empty AAD, and full 16-byte blocks.

## Prerequisites

- NVIDIA GPU with a CUDA-capable driver
- CUDA Toolkit with `nvcc` available
- CMake 3.28 or newer
- A CUDA-compatible host C++ compiler
- OpenSSL development package discoverable by CMake

On Windows, use a Visual Studio Developer Command Prompt or pass the host compiler explicitly with `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`.

If OpenSSL is installed in a custom location, pass `OPENSSL_ROOT_DIR`:

```bash
cmake -S . -B build -DOPENSSL_ROOT_DIR=/path/to/openssl
```

## Configure And Build

Choose the CUDA architecture for your GPU. For example, `86` targets Ampere GPUs such as RTX 30-series cards.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

For multi-config generators such as Visual Studio, `--config Release` selects the Release configuration at build time.

## Correctness

Run the small known-answer tests before interpreting benchmark output:

```bash
ctest --test-dir build --output-on-failure
```

The CTest suite runs the `CudaAesKat` executable. It uses deterministic vectors for:

- ECB-128 and ECB-256
- CTR-128 and CTR-256
- GCM-128 and GCM-256

The GCM checks cover the Phase 2 supported shape: 96-bit IV, empty AAD, full 16-byte blocks, ciphertext/tag match, wrong-tag rejection, and tampered-ciphertext rejection. This project is still a benchmark/research repository, not production cryptography software.

## Run

Linux:

```bash
./build/CudaProject
```

Windows:

```powershell
.\build\Release\CudaProject.exe
```

The benchmark currently exercises ECB, CTR, and GCM-shaped paths for AES-128 and AES-256 across several message sizes. Run correctness checks before interpreting benchmark output.

For a small reproducibility smoke run:

```bash
./build/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/smoke
```

Windows:

```powershell
.\build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\smoke
```

Benchmark artifacts are written under `bench/` by default, or under the directory supplied with `--bench-dir`:

- `run_metadata.csv` records schema version, command line, run count, selected sizes, OS/compiler hints, CUDA runtime/driver versions, GPU name, compute capability, and a clocks/persistence note.
- `thr_gpu.csv` records GPU rows with `timing_scope=kernel_only`; this is CUDA event timing around the kernel launch, not end-to-end application throughput.
- `thr_cpu.csv` records OpenSSL CPU baseline rows with `timing_scope=cpu_baseline`.

Raw result columns are stable for Phase 3: `schema_version`, `benchmark_run_id`, `timing_scope`, `device`, `cipher`, `block_size`, `run_index`, `run_count`, `time_ms`, `GiB/s`, `operation`, and `command_line`.

Generate a summary table from the raw CSV files:

```bash
python scripts/summarize_benchmarks.py bench/thr_gpu.csv bench/thr_cpu.csv -o bench/summary.md
```

The summary groups by device, cipher, operation, block size, and timing scope, then reports count, min, mean, median, and max for time and throughput. It preserves `timing_scope`, so kernel-only GPU rows are not mixed with CPU baseline or future end-to-end rows.

Example output:

```text
[RUN 3/5] [GPU] ctr-128 processed 100 MiB in 12.3 ms -> 7.9 GiB/s
```

Use repeated runs, fixed GPU clocks, persistence mode notes, and a quiet system when comparing throughput numbers. Treat kernel-only and end-to-end timings as different metrics; this phase labels the current GPU metric as kernel-only.

## Benchmark Methodology

Benchmark results are only meaningful after the CTest known-answer checks pass. The benchmark runner still performs embedded round-trip checks, but the deterministic KAT suite is the correctness gate for interpreting throughput.

The current GPU timing scope is `kernel_only`: CUDA events are recorded around the kernel launch and synchronization. Allocation, host-to-device copy, device-to-host copy, output validation, and summary generation are outside this timed region. End-to-end throughput is not currently emitted as a benchmark row; do not compare `kernel_only` rows against future `end_to_end` rows without keeping the timing scope separate.

The CPU baseline uses OpenSSL EVP and is recorded separately with `timing_scope=cpu_baseline`. It is a comparison point, not a tuned CPU benchmark: CPU affinity, turbo behavior, OpenSSL build flags, and system load can affect it.

Recommended methodology for publishable runs:

1. Build in Release mode and record the exact CMake command.
2. Run `ctest --test-dir build --output-on-failure`.
3. Use an explicit benchmark command with `--runs`, `--sizes`, and `--bench-dir`.
4. Record GPU clocks, persistence mode, driver, CUDA Toolkit, OS, compiler, GPU model, and command line. `run_metadata.csv` captures what the executable can query; clocks and persistence mode remain a manual note.
5. Run on a quiet system and repeat enough runs for stable median throughput.
6. Publish raw CSV files and generated `summary.md` together. Treat summary tables as derived output, not the source of truth.

Known limitations:

- GCM correctness coverage is limited to 96-bit IV, empty AAD, and full 16-byte blocks.
- Partial-block behavior and non-empty AAD are not benchmarked in v1.
- The GPU result is kernel-only timing, not full application throughput.
- CPU baseline rows are not a controlled CPU performance study.

## Optional Tooling

Generate a PTX dump for `aes128_ecb.cu`:

```bash
cmake --build build --target ptx-dump
```

Run the Nsight Systems helper target by configuring the executable path first:

```bash
cmake -S . -B build -DNSYS_EXECUTABLE="/path/to/nsys"
cmake --build build --target nsight-profile
```

## Roadmap Direction

The near-term roadmap focuses on making this a credible open-source GPU AES benchmark:

- Portable build and repository hygiene
- Standard AES-GCM correctness with tag verification and known-answer tests
- Complete AES mode roadmap beyond the currently implemented modes
- Reproducible benchmark harness and published result format
- Documentation, contribution workflow, and SEO-ready project presentation
