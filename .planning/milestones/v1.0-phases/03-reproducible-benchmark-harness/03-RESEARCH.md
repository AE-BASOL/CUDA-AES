---
phase: 03-reproducible-benchmark-harness
researched_at: 2026-06-04T14:55:00+03:00
status: complete
---

# Phase 3 Research: Reproducible Benchmark Harness

## Research Complete

Phase 3 should turn the existing benchmark from an ad hoc console/CSV run into a reproducible benchmark workflow. The current executable already benchmarks `ecb-128`, `ecb-256`, `ctr-128`, `ctr-256`, `gcm-128`, and `gcm-256` across four sizes and five runs, prints results, and writes `bench/thr_gpu.csv` plus `bench/thr_cpu.csv`. It also performs benchmark-embedded round-trip checks before timed runs.

## Current Implementation Facts

- `main.cu` owns command-line parsing, input generation, round-trip validation, GPU timing, OpenSSL CPU baseline timing, console output, and CSV writing.
- Existing benchmark CSV columns are `device,cipher,block_size,run_id,time_ms,GiB/s,operation`.
- GPU timing uses CUDA events around the kernel launch only, so the current measurement is kernel-only timing.
- Host-to-device copy, device-to-host copy, allocation, pinned host allocation, CPU baseline setup, and round-trip checks are outside the timed GPU event region.
- CUDA events are created in `main.cu` and `gf_mult_bench()`, but they are not destroyed after use.
- `bench/` is the current artifact directory for generated benchmark data.
- `profiling_helpers.h` contains optional NVTX wrappers; benchmark harness changes should preserve NVTX compatibility.
- Phase 2 added `CudaAesKat` and CTest correctness coverage; Phase 3 should not replace correctness tests with benchmark round trips.

## Planning Implications

Phase 3 can be implemented conservatively without changing AES kernel algorithms:

1. Add run metadata and environment capture around the existing executable instead of rewriting kernels.
2. Make the current GPU measurement explicitly `kernel_only`; optionally add a host-clock `end_to_end` field or row if it can be implemented cleanly.
3. Keep raw benchmark output machine-readable and stable. CSV is acceptable, but the schema needs enough metadata for reproducibility.
4. Add a summary generator that reads raw CSV/JSON and computes clear tables from the raw artifact.
5. Document methodology separately from performance claims.
6. Wrap or centrally destroy CUDA events to close the resource leak without changing timed regions.

## Requirement Mapping

- BENCH-01 needs command-level environment capture: GPU name, driver/runtime/toolkit, OS, compiler, build type, clocks/persistence note, mode, size, run count, and command line.
- BENCH-02 needs timing scope labeling: current GPU timing is kernel-only; end-to-end timing should be separate or marked unavailable.
- BENCH-03 needs stable raw machine-readable output with reproducible run metadata.
- BENCH-04 needs summary tooling that reads raw output, not console text.
- BENCH-05 needs methodology documentation before claims.
- BENCH-06 needs CUDA event destruction or RAII.

## Validation Architecture

Validation should combine source-level checks and runtime checks:

- Source checks confirm CUDA events are destroyed or wrapped, raw output schema fields exist, and summary tooling reads raw artifacts.
- Help/README checks confirm the documented benchmark command exists and explains timing scope.
- Runtime checks, when a CUDA/MSVC environment is available, run a small benchmark configuration and generate a summary from raw output.
- Existing environment blocker remains: this shell cannot run CUDA CMake/CTest until `cl.exe` is discoverable.

## Recommended Plan Split

| Plan | Wave | Focus |
|------|------|-------|
| 03-01 | 1 | Add benchmark run metadata, configurable run parameters, stable raw output fields, timing scope labels, and CUDA event cleanup |
| 03-02 | 2 | Add summary generation from raw benchmark output |
| 03-03 | 3 | Add benchmark methodology documentation and verification notes |

Wave 2 depends on the raw schema from Wave 1. Wave 3 depends on both the command/schema and summary tooling so documentation can describe the real workflow.

