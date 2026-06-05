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

CBC, CFB, and OFB rows should be interpreted with their feedback dependencies in mind. They are included for reproducible mode coverage and comparison against OpenSSL baselines, not to imply the same parallelism profile as CTR.

CCM rows include both encryption and authentication tag computation for the current benchmark scope: 96-bit nonce, empty AAD, 16-byte tag, and full 16-byte blocks. Do not compare CCM rows against confidentiality-only modes without preserving that AEAD workload distinction.

## Claims Policy

Avoid unsupported performance claims. Results should be presented as reproducible measurements for a named environment and commit, not as universal rankings.
