# Results

This page defines how results should be packaged. It does not publish benchmark numbers yet.

## Result Package

A credible result package should include:

- exact commit hash
- build command and CMake cache-relevant options
- `ctest` result
- benchmark command line
- `run_metadata.csv`
- `thr_gpu.csv`
- `thr_cpu.csv`
- generated `summary.md`
- GPU clocks and persistence-mode note
- OS, compiler, CUDA Toolkit, driver, and GPU model

## Summary Tables

Generate tables from raw files:

```bash
python scripts/summarize_benchmarks.py bench/run-001/thr_gpu.csv bench/run-001/thr_cpu.csv -o bench/run-001/summary.md
```

The summary groups by device, cipher, operation, block size, and timing scope. Keep `kernel_only`, `cpu_baseline`, and any future `end_to_end` rows separate.

## Claims Policy

Avoid unsupported performance claims. Results should be presented as reproducible measurements for a named environment and commit, not as universal rankings.

