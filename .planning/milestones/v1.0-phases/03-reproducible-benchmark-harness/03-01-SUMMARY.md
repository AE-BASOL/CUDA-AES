---
phase: 03-reproducible-benchmark-harness
plan: 01
subsystem: benchmarking
tags: [cuda, benchmark, csv, reproducibility]

requires:
  - phase: 02-correctness-baseline
    provides: Correctness-gated AES benchmark modes and GCM tag semantics
provides:
  - Benchmark CLI flags for run count, selected sizes, and output directory
  - Phase 3 raw benchmark CSV schema with timing-scope labels
  - Benchmark run metadata sidecar
  - CUDA event cleanup in timed benchmark paths
affects: [benchmarking, documentation, phase-03-summary]

tech-stack:
  added: []
  patterns: [command-line benchmark configuration, metadata sidecar, stable raw CSV schema]

key-files:
  created: [.planning/phases/03-reproducible-benchmark-harness/03-01-USER-SETUP.md]
  modified: [main.cu, README.md, .planning/codebase/TESTING.md, .planning/codebase/CONCERNS.md]

key-decisions:
  - "Keep current GPU timing as kernel-only and label it with timing_scope=kernel_only instead of changing the measured region."
  - "Write run metadata to run_metadata.csv as a sidecar so raw result rows stay compact and parseable."

patterns-established:
  - "Phase 3 raw benchmark rows include schema_version, benchmark_run_id, timing_scope, run_count, and command_line."
  - "CUDA event resources are destroyed immediately after elapsed time is captured."

requirements-completed: [BENCH-01, BENCH-02, BENCH-03, BENCH-06]

duration: 20 min
completed: 2026-06-04
---

# Phase 3 Plan 01: Benchmark Metadata And Raw Schema Summary

**Benchmark CLI metadata, Phase 3 raw CSV schema, kernel-only timing labels, and CUDA event cleanup**

## Performance

- **Duration:** 20 min
- **Started:** 2026-06-04T15:05:00+03:00
- **Completed:** 2026-06-04T15:25:00+03:00
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Added `--runs`, `--sizes`, and `--bench-dir` benchmark flags while preserving the existing default benchmark behavior.
- Added `run_metadata.csv` with schema version, command line, selected parameters, OS/compiler hints, CUDA versions, GPU model, compute capability, and clocks/persistence note.
- Replaced raw CPU/GPU CSV rows with a stable Phase 3 schema that includes `timing_scope`, `benchmark_run_id`, `run_count`, and `command_line`.
- Labeled current GPU measurements as `kernel_only` and CPU rows as `cpu_baseline`.
- Destroyed CUDA events in both the main benchmark loop and `gf_mult_bench()`.

## Task Commits

Task work will be committed with this summary as `feat(03-01): add reproducible benchmark raw metadata`.

## Files Created/Modified

- `main.cu` - Adds benchmark config parsing, metadata capture, stable raw row writer, timing-scope labels, and CUDA event cleanup.
- `README.md` - Documents smoke benchmark command, raw files, metadata fields, and kernel-only timing scope.
- `.planning/codebase/TESTING.md` - Records Phase 3 benchmark schema and smoke command.
- `.planning/codebase/CONCERNS.md` - Updates event-leak and timing-scope concerns.
- `.planning/phases/03-reproducible-benchmark-harness/03-01-USER-SETUP.md` - Captures CUDA/MSVC shell requirement for runtime verification.

## Decisions Made

- Kept the current CUDA event timing region as kernel-only to avoid changing benchmark semantics in the same plan that labels raw output.
- Used a metadata sidecar rather than duplicating environment fields into every result row.
- Preserved `bench/thr_gpu.csv` and `bench/thr_cpu.csv` names, but updated their schema to Phase 3 `phase3.v1`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Runtime CMake verification remains environment-limited in this shell. `where.exe cl.exe` found no host compiler, and CMake configure failed with `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`.

## Verification

- PASS: `rg "runs|sizes|bench-dir|command_line|cudaRuntimeGetVersion|cudaDriverGetVersion|cudaGetDeviceProperties|compute" main.cu README.md`
- PASS: `rg "schema|timing_scope|kernel_only|end_to_end|GiB/s|run_id" main.cu .planning/codebase README.md`
- PASS: `rg "cudaEventCreate|cudaEventDestroy|GpuTimer|CudaEvent" main.cu profiling_helpers.h`
- ENV-LIMITED: `cmake -S . -B C:\tmp\cuda-aes-phase3-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86` failed because `cl.exe` is not in PATH.

## User Setup Required

CUDA/MSVC runtime verification is still required. See `.planning/phases/03-reproducible-benchmark-harness/03-01-USER-SETUP.md`.

## Next Phase Readiness

Plan 03-02 can read the Phase 3 raw CSV schema from `main.cu` and generate summary tables from `thr_gpu.csv` and `thr_cpu.csv`.

---
*Phase: 03-reproducible-benchmark-harness*
*Completed: 2026-06-04*

