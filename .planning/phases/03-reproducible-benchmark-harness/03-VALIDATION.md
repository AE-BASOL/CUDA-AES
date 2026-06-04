---
phase: 03-reproducible-benchmark-harness
status: planned
created_at: 2026-06-04T14:55:00+03:00
---

# Phase 3 Validation Strategy

## Scope

Validate that the benchmark harness is reproducible, inspectable, and resource-clean without relying on unsupported performance claims.

## Source-Level Checks

- `rg "cudaEventDestroy|CudaEvent|GpuTimer|Scoped" main.cu profiling_helpers.h` should show event cleanup or RAII.
- `rg "timing_scope|kernel_only|end_to_end|run_id|command_line|cuda|driver|compiler|build_type" main.cu scripts .planning/codebase README.md` should show metadata and timing labels.
- `rg "summary|median|mean|min|max|GiB/s|thr_gpu|thr_cpu" scripts .planning/codebase README.md` should show summary generation.
- `rg "benchmark methodology|warmup|repetitions|CPU baseline|kernel-only|end-to-end" README.md .planning/codebase` should show methodology documentation.

## Runtime Checks

When running from a CUDA/MSVC-ready shell:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase3-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build C:\tmp\cuda-aes-phase3-build --config Release
ctest --test-dir C:\tmp\cuda-aes-phase3-build --output-on-failure
```

Then run the smallest practical benchmark command documented by Phase 3 and generate the summary table from the raw output.

## Acceptance

Phase 3 passes when BENCH-01 through BENCH-06 are traceable to source artifacts and the runtime command path is either executed successfully or explicitly marked environment-limited with the same `cl.exe` blocker from Phase 2.

