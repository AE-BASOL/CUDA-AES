# Benchmark Result Contributions

Benchmark result contributions are useful when another maintainer can connect the numbers to a specific commit, environment, command line, and raw artifact set. Treat this checklist as required for benchmark-result issues and pull requests.

This project accepts results as reproducible, environment-specific measurements. Do not present them as universal rankings or fastest-in-world claims unless the contribution also includes controlled comparative evidence.

## Scope

State what the result covers:

- commit hash
- mode or modes
- operation, such as encryption, decryption, wrap, or unwrap
- input sizes
- run count
- benchmark branch or tag, if different from the commit hash

## Correctness Gate

Run correctness checks before interpreting benchmark output:

```text
ctest --test-dir build --output-on-failure
```

Include the CTest status in the issue or pull request. If CTest does not pass, the benchmark result is diagnostic evidence, not a publishable throughput result.

## Environment

Record the full environment:

- GPU model
- GPU compute capability
- CUDA Toolkit version
- NVIDIA driver version
- operating system
- compiler and version
- CMake command and cache-relevant options, including `CMAKE_BUILD_TYPE`, `CMAKE_CUDA_ARCHITECTURES`, and any explicit host compiler setting
- GPU clocks and persistence-mode note

## Commands

Include the exact configure/build command and benchmark command.

Example configure/build commands:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

Example benchmark command:

```text
./build/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/result-001
```

On Windows, the executable may be under `.\build\Release\CudaProject.exe` depending on the generator.

## Required Raw Artifacts

Attach or link the raw result package:

- `run_metadata.csv`
- `thr_gpu.csv`
- `thr_cpu.csv` where applicable
- generated `summary.md`

Generate the summary from the raw CSV files:

```text
python scripts/summarize_benchmarks.py bench/result-001/thr_gpu.csv bench/result-001/thr_cpu.csv -o bench/result-001/summary.md
```

Keep the raw CSV files and generated summary together. Do not copy only selected console output into an issue or pull request.

## Timing Scope

Preserve the timing-scope labels from the raw files:

- `kernel_only` GPU rows measure the CUDA kernel region with CUDA events.
- `cpu_baseline` rows are OpenSSL CPU comparison rows, not a tuned CPU benchmark study.
- Future `end_to_end` rows must stay separate from kernel-only rows.

Do not compare rows with different timing scopes as if they measure the same thing.

## Claims Policy

By submitting benchmark results, acknowledge that:

- results are environment-specific measurements for a named commit and hardware/software environment
- raw artifacts are required to support any throughput statement
- feedback modes, AEAD modes, storage modes, and key-wrap workloads have different benchmark semantics
- fastest-in-world, universal ranking, or broad production-performance claims are unsupported unless controlled comparative evidence is included

Benchmark result contributions that do not meet this checklist can still be useful as troubleshooting reports, but they should not be treated as release-quality benchmark evidence.
