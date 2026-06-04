---
phase: 03-reproducible-benchmark-harness
plan: 01
status: incomplete
created: 2026-06-04
---

# Phase 3 Plan 01 User Setup

## CUDA/MSVC Build Shell

Runtime benchmark verification requires `nvcc` to find the Visual Studio host compiler `cl.exe`.

## Required Action

Run build and benchmark verification from a Visual Studio Developer Command Prompt, or pass the host compiler explicitly:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase3-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
cmake --build C:\tmp\cuda-aes-phase3-build --config Release
ctest --test-dir C:\tmp\cuda-aes-phase3-build --output-on-failure
C:\tmp\cuda-aes-phase3-build\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\smoke
```

If using a Visual Studio generator, adjust the executable path to the generated `Release\CudaProject.exe`.

## Current Shell Status

`where.exe cl.exe` returned no result in the current shell, and CMake configure failed with:

```text
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```
