# Benchmark Methodology

CUDA-AES benchmark results should be reproducible from raw artifacts, not copied from console output.

## Recommended Procedure

1. Build in Release mode and record the exact CMake command.
2. Run `ctest --test-dir build --output-on-failure`.
3. Run the benchmark with explicit parameters:

```bash
./build/CudaProject --runs 5 --sizes 1048576,10485760,104857600 --bench-dir bench/run-001
```

4. Record fixed clocks, persistence mode, GPU model, driver, CUDA Toolkit, OS, compiler, and command line.
5. Generate a summary:

```bash
python scripts/summarize_benchmarks.py bench/run-001/thr_gpu.csv bench/run-001/thr_cpu.csv -o bench/run-001/summary.md
```

6. Publish raw CSV files, `run_metadata.csv`, and `summary.md` together.

## Raw Output

The raw CSV schema is `phase3.v1`:

```text
schema_version,benchmark_run_id,timing_scope,device,cipher,block_size,run_index,run_count,time_ms,GiB/s,operation,command_line
```

`run_metadata.csv` records command, run count, selected sizes, CUDA versions, GPU properties, and environment hints.

## Timing Scope

GPU rows use `timing_scope=kernel_only`. This measures the CUDA kernel region with CUDA events and excludes allocation, transfers, validation, and summary generation.

CPU rows use `timing_scope=cpu_baseline`. They are OpenSSL EVP comparison rows, not a tuned CPU benchmark study.

Future end-to-end rows must use a distinct `timing_scope`.

## Limitations

- Do not compare kernel-only and end-to-end rows as the same metric.
- Do not publish throughput claims without raw artifacts.
- Do not treat the project as production cryptography software.
- GCM benchmark scope matches the current correctness scope: 96-bit IV, empty AAD, full blocks.

