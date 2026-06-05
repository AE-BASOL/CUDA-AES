# CUDA-AES Benchmark

CUDA-AES Benchmark is a reproducible GPU AES benchmark suite for CUDA developers. It measures CUDA kernels for AES-128 and AES-256, compares them with an OpenSSL CPU baseline, records raw benchmark artifacts, and documents the correctness and methodology behind the numbers.

This repository is benchmark and research software, not a production cryptography library.

## Current Coverage

Implemented in the canonical top-level build:

| Mode | AES-128 | AES-256 | Correctness tests | Benchmark rows | Notes |
|------|---------|---------|-------------------|----------------|-------|
| ECB | Yes | Yes | Yes | Yes | NIST-style known-answer coverage |
| CBC | Yes | Yes | Yes | Yes | Confidentiality-only feedback mode |
| CFB | Yes | Yes | Yes | Yes | CFB-128 full-block segment scope |
| OFB | Yes | Yes | Yes | Yes | Confidentiality-only chained keystream mode |
| CTR | Yes | Yes | Yes | Yes | 96-bit IV/counter helper in benchmark |
| GCM | Yes | Yes | Yes | Yes | 96-bit IV, empty AAD, full blocks |
| CCM | Yes | Yes | Yes | Yes | 96-bit nonce, empty AAD, 16-byte tag, full blocks |
| XTS-AES | Yes | Yes | Yes | Yes | Storage-sector mode, 16-byte sector tweak, full blocks |

Planned coverage includes AES-KW, AES-KWP, and distinct GMAC/CMAC authentication benchmarking.

## Quick Start

Prerequisites:

- NVIDIA GPU with a CUDA-capable driver
- CUDA Toolkit with `nvcc`
- CMake 3.28 or newer
- CUDA-compatible host C++ compiler
- OpenSSL development package discoverable by CMake

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

On Windows, use a Visual Studio Developer Command Prompt or pass the host compiler explicitly:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
cmake --build build --config Release
```

Run correctness checks before interpreting benchmark output:

```bash
ctest --test-dir build --output-on-failure
```

Run a small reproducibility smoke benchmark:

```bash
./build/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/smoke
python scripts/summarize_benchmarks.py bench/smoke/thr_gpu.csv bench/smoke/thr_cpu.csv -o bench/smoke/summary.md
```

Windows executable paths may use `.\build\Release\CudaProject.exe` depending on the generator.

## Benchmark Artifacts

The benchmark writes raw artifacts under `bench/` by default, or under `--bench-dir`:

- `run_metadata.csv` records schema version, command line, run count, selected sizes, OS/compiler hints, CUDA runtime/driver versions, GPU name, compute capability, and clocks/persistence note.
- `thr_gpu.csv` records GPU rows with `timing_scope=kernel_only`; this is CUDA event timing around the kernel launch, not end-to-end application throughput.
- `thr_cpu.csv` records OpenSSL CPU baseline rows with `timing_scope=cpu_baseline`.
- `summary.md` is generated from raw CSV files by `scripts/summarize_benchmarks.py`.

Raw result columns use Phase 3 schema `phase3.v1`:

```text
schema_version,benchmark_run_id,timing_scope,device,cipher,block_size,run_index,run_count,time_ms,GiB/s,operation,command_line
```

## Documentation

- [Architecture](docs/architecture.md) - canonical source layout and runtime flow
- [Correctness](docs/correctness.md) - KAT coverage, GCM scope, and verification limits
- [Benchmark Methodology](docs/benchmark-methodology.md) - reproducible run procedure, raw files, timing scope, and summary generation
- [Results](docs/results.md) - how to package and interpret benchmark results
- [Profiling](docs/profiling.md) - NVTX, Nsight, and PTX dump helpers
- [Mode Matrix](docs/modes.md) - implemented, tested, benchmarked, documented, and planned AES mode coverage
- [Legacy Tezcan Implementation](docs/legacy-tezcan.md) - provenance and non-canonical legacy implementation notes

## Contributing And Governance

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)

## Methodology Summary

Benchmark results are only meaningful after deterministic correctness tests pass. The GPU timing scope is currently `kernel_only`, which excludes allocation, host-to-device copy, device-to-host copy, output validation, and summary generation. Do not compare kernel-only rows against future end-to-end rows without preserving `timing_scope`.

ECB, CBC, CFB, OFB, and CTR are confidentiality-only modes; they do not authenticate ciphertext. CBC, CFB, and OFB also have feedback dependencies, so their rows should not be interpreted as CTR-like parallel throughput.

Use repeated runs, fixed GPU clocks, persistence-mode notes, and a quiet system when comparing throughput numbers. Publish raw CSV files and generated summaries together.

## Repository Layout

- `main.cu` - benchmark runner, CLI parsing, GPU launch orchestration, OpenSSL CPU comparison, CSV output, and debug routines
- `aes_common.h`, `aes_tables.cu` - shared AES declarations, constants, lookup tables, and key expansion helpers
- `aes128_*.cu`, `aes256_*.cu` - canonical AES kernel implementations
- `tests/kat_main.cu` - deterministic known-answer tests
- `scripts/summarize_benchmarks.py` - raw CSV to Markdown summary generator
- `docs/` - public documentation
- `v3/` - local experimental variant, not the canonical build target
- `cihangirTezcanAESimplementation/` - legacy/provenance implementation

## Current Limitations

- Runtime CMake/CTest verification in the current development shell is blocked until `nvcc` can find `cl.exe`.
- GCM coverage is limited to 96-bit IV, empty AAD, and full 16-byte blocks.
- CCM coverage is limited to 96-bit nonce, empty AAD, 16-byte tag, and full 16-byte blocks.
- XTS-AES coverage is limited to full 16-byte blocks with a 16-byte sector tweak; ciphertext stealing is not implemented.
- Partial-block behavior and non-empty AAD are not benchmarked in v1.
- CPU baseline rows are not a controlled CPU performance study.
- This project does not claim to be the fastest GPU AES implementation.

## Roadmap Direction

The v1 roadmap focuses on:

- Open-source documentation and governance
- Full practical AES mode coverage
- Discoverability for CUDA AES and GPU AES benchmark searches
- Versioned releases with reproducible raw benchmark artifacts
