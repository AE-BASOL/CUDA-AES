---
phase: 03-reproducible-benchmark-harness
plan: 03
status: incomplete
created: 2026-06-04
---

# Phase 3 Plan 03 User Setup

## Final Runtime Verification

A CUDA/MSVC-ready shell is required for final runtime benchmark verification.

Run from a Visual Studio Developer Command Prompt, or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase3-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build C:\tmp\cuda-aes-phase3-build --config Release
ctest --test-dir C:\tmp\cuda-aes-phase3-build --output-on-failure
C:\tmp\cuda-aes-phase3-build\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\smoke
python scripts\summarize_benchmarks.py bench\smoke\thr_gpu.csv bench\smoke\thr_cpu.csv -o bench\smoke\summary.md
```

If the executable is under `Release\`, adjust the path accordingly.

## Current Shell Status

Runtime verification is incomplete here because `cl.exe` is not available to `nvcc`.

