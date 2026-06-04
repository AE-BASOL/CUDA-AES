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

Example output:

```text
[RUN 3/5] [GPU] ctr-128 processed 100 MiB in 12.3 ms -> 7.9 GiB/s
```

Use repeated runs, fixed GPU clocks, persistence mode notes, and a quiet system when comparing throughput numbers. Treat kernel-only and end-to-end timings as different metrics; this phase labels the current GPU metric as kernel-only.

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
