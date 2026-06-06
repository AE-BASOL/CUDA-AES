---
name: Benchmark Result
about: Share reproducible CUDA-AES benchmark artifacts for a named environment
title: "Benchmark result: "
---

# Benchmark Result Contribution

Before submitting, review the canonical checklist at `docs/benchmark-result-contributions.md`:
[benchmark result contribution checklist](../../docs/benchmark-result-contributions.md).

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

## Claims Policy

Confirm that these results are environment-specific measurements for the named commit and environment, not universal rankings.

- [ ] I am not making ranking or fastest-in-world claims without controlled comparative evidence.
