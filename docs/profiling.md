# Profiling

CUDA-AES includes lightweight profiling support for inspecting kernel behavior.

## NVTX

`profiling_helpers.h` defines `NVTX_PUSH` and `NVTX_POP`. When `ENABLE_NVTX` is enabled, benchmark regions are annotated for NVIDIA tooling. Otherwise the macros compile to no-ops.

## Nsight Systems

Configure the Nsight Systems executable path, then build the helper target:

```bash
cmake -S . -B build -DNSYS_EXECUTABLE="/path/to/nsys"
cmake --build build --target nsight-profile
```

## PTX Dump

Generate a PTX dump for `aes128_ecb.cu`:

```bash
cmake --build build --target ptx-dump
```

## Methodology

Profiling runs should still follow the correctness and benchmark methodology rules:

- run CTest first
- record command line and environment
- keep raw benchmark artifacts
- distinguish profiling observations from reproducible throughput claims

