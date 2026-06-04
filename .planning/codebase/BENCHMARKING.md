---
mapped_at: 2026-06-04
last_mapped_commit: phase-03
focus: benchmarking
---

# Benchmarking

## Summary

The canonical benchmark executable is `CudaProject` from the root `CMakeLists.txt`. Phase 3 makes the benchmark workflow reproducible at the artifact level: raw CSV rows have a stable schema, run metadata is captured in a sidecar file, summary tables are generated from raw CSV files, and CUDA timing events are destroyed after use.

## Correctness Prerequisite

Run CTest before interpreting benchmark output:

```bash
ctest --test-dir build --output-on-failure
```

The CTest `CudaAesKat` target covers ECB, CTR, and GCM known-answer checks for AES-128 and AES-256. GCM scope is 96-bit IV, empty AAD, and full 16-byte blocks.

## Reproducibility Command

Small smoke run:

```bash
./build/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/smoke
```

Default output directory is `bench/`. Use `--bench-dir` for isolated runs.

## Raw Output

`run_metadata.csv` captures:

- schema version
- benchmark run id
- command line
- run count
- selected sizes
- OS/compiler hints
- CUDA runtime and driver versions
- GPU name, compute capability, and memory
- clocks/persistence note

`thr_gpu.csv` and `thr_cpu.csv` use Phase 3 schema `phase3.v1`:

```text
schema_version,benchmark_run_id,timing_scope,device,cipher,block_size,run_index,run_count,time_ms,GiB/s,operation,command_line
```

GPU rows use `timing_scope=kernel_only`. CPU baseline rows use `timing_scope=cpu_baseline`.

## Summary Generation

Generate a Markdown summary:

```bash
python scripts/summarize_benchmarks.py bench/thr_gpu.csv bench/thr_cpu.csv -o bench/summary.md
```

The summary groups by device, cipher, operation, block size, and `timing_scope`, then reports count, min, mean, median, and max for `time_ms` and `GiB/s`.

## Methodology

- Build in Release mode.
- Run correctness tests first.
- Use explicit `--runs`, `--sizes`, and `--bench-dir`.
- Keep raw CSV and summary output together.
- Record fixed clocks and persistence mode manually when publishing results.
- Keep `kernel_only`, `cpu_baseline`, and any future `end_to_end` rows separate.

## Limitations

- GPU timing is currently kernel-only and excludes allocation and host/device transfers.
- CPU baseline uses OpenSSL EVP but does not pin CPU frequency or affinity.
- GCM benchmark scope follows Phase 2 correctness scope: 96-bit IV, empty AAD, full blocks.
- Runtime verification in the current shell is blocked until `cl.exe` is available to `nvcc`.

