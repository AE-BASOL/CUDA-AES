# Benchmark Result Contribution

## Result Scope

- Commit:
- Mode(s):
- Operation(s): ENC / DEC
- Sizes:
- Runs:

## Environment

- GPU model:
- Compute capability:
- CUDA Toolkit:
- NVIDIA driver:
- OS:
- Compiler:
- CMake command:
- GPU clocks / persistence mode:

## Correctness Gate

Paste the result of:

```text
ctest --test-dir build --output-on-failure
```

## Benchmark Command

```text
./build/CudaProject --runs N --sizes ... --bench-dir ...
```

## Required Raw Artifacts

Attach:

- `run_metadata.csv`
- `thr_gpu.csv`
- `thr_cpu.csv`
- generated `summary.md`

Do not make ranking or fastest-in-world claims without controlled comparative evidence.

