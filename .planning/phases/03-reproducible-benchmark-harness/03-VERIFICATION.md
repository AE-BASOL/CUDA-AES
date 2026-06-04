---
phase: 03-reproducible-benchmark-harness
status: passed_with_environment_limit
verified_at: 2026-06-04T15:45:00+03:00
---

# Phase 3 Verification

## Result

Phase 3 satisfies the planned reproducible benchmark harness requirements at source level. Runtime CMake, CTest, and benchmark execution remain environment-limited in the current shell because CUDA `nvcc` cannot find the Visual Studio host compiler `cl.exe`.

## Checks

| Check | Result | Evidence |
|-------|--------|----------|
| BENCH-01 environment and parameter capture | Passed | `main.cu` records command line, run count, selected sizes, CUDA runtime/driver versions, GPU properties, compute capability, and writes `run_metadata.csv`. |
| BENCH-02 timing scope distinction | Passed | GPU raw rows use `timing_scope=kernel_only`; CPU raw rows use `timing_scope=cpu_baseline`; README and codebase docs explain that kernel-only excludes allocation and transfers. |
| BENCH-03 raw machine-readable output | Passed | `thr_gpu.csv` and `thr_cpu.csv` use schema `phase3.v1` with stable columns including `schema_version`, `benchmark_run_id`, `timing_scope`, `device`, `cipher`, `block_size`, `run_index`, `run_count`, `time_ms`, `GiB/s`, `operation`, and `command_line`. |
| BENCH-04 summary generation | Passed | `scripts/summarize_benchmarks.py` reads raw CSV files, validates required columns, groups by device/cipher/operation/block size/timing scope, and writes Markdown summaries. |
| BENCH-05 methodology documentation | Passed | README and `.planning/codebase/BENCHMARKING.md` document correctness prerequisites, warmup/repetitions guidance, timing scope, CPU baseline limitations, raw files, summaries, and known limitations. |
| BENCH-06 CUDA event cleanup | Passed | `main.cu` destroys CUDA events after elapsed time capture in the main benchmark loop and `gf_mult_bench()`. |
| Summary script interface | Passed | `python scripts\summarize_benchmarks.py --help` succeeded. |
| Summary script syntax | Passed | `python -m py_compile scripts\summarize_benchmarks.py` succeeded. |
| Synthetic raw summary | Passed | Temporary project-local Phase 3 CSV samples produced two summary rows and were deleted after verification. |
| Runtime build/test/benchmark | Environment-limited | `cmake -S . -B C:\tmp\cuda-aes-phase3-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86` failed because `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`. |

## Source Verification Commands

```powershell
rg "runs|sizes|bench-dir|cudaRuntimeGetVersion|cudaDriverGetVersion|cudaGetDeviceProperties" main.cu README.md
rg "timing_scope|kernel_only|end_to_end|schema|run_id" main.cu README.md .planning/codebase
rg "cudaEventCreate|cudaEventDestroy|CudaEvent|GpuTimer" main.cu profiling_helpers.h
rg "summarize|median|mean|min|max|timing_scope" scripts
python scripts\summarize_benchmarks.py --help
python -m py_compile scripts\summarize_benchmarks.py
rg "methodology|warmup|repetitions|kernel-only|end-to-end|CPU baseline|raw|summary" README.md .planning/codebase
```

## Runtime Verification Command

Run from a Visual Studio Developer Command Prompt or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase3-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build C:\tmp\cuda-aes-phase3-build --config Release
ctest --test-dir C:\tmp\cuda-aes-phase3-build --output-on-failure
C:\tmp\cuda-aes-phase3-build\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\smoke
python scripts\summarize_benchmarks.py bench\smoke\thr_gpu.csv bench\smoke\thr_cpu.csv -o bench\smoke\summary.md
```

If a Visual Studio generator places the executable under `Release\`, adjust the executable path accordingly.

## Requirement Mapping

- BENCH-01: Covered by benchmark flags and `run_metadata.csv`.
- BENCH-02: Covered by `timing_scope=kernel_only` and methodology docs.
- BENCH-03: Covered by Phase 3 raw CSV schema.
- BENCH-04: Covered by `scripts/summarize_benchmarks.py`.
- BENCH-05: Covered by README methodology and `.planning/codebase/BENCHMARKING.md`.
- BENCH-06: Covered by `cudaEventDestroy` calls in benchmark timing paths.

## Verification Complete

