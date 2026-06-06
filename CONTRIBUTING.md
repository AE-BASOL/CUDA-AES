# Contributing

CUDA-AES Benchmark welcomes contributions that improve build portability, correctness, reproducibility, documentation, and AES mode coverage.

This is benchmark and research software, not a production cryptography library.

## Before Opening a Pull Request

1. Build the project in Release mode.
2. Run correctness checks:

```bash
ctest --test-dir build --output-on-failure
```

3. If benchmark behavior changes, run a small benchmark and keep raw artifacts:

```bash
./build/CudaProject --runs 1 --sizes 1048576 --bench-dir bench/pr-check
python scripts/summarize_benchmarks.py bench/pr-check/thr_gpu.csv bench/pr-check/thr_cpu.csv -o bench/pr-check/summary.md
```

4. Document GPU model, CUDA Toolkit, driver, compiler, OS, CMake command, and benchmark command.

## Contribution Types

- Correctness tests and known-answer vectors
- Benchmark harness and summary tooling
- Documentation and methodology improvements
- New AES mode implementations with tests and benchmark rows
- Build portability fixes
- Profiling and observability improvements

## Benchmark Result Contributions

Benchmark result issues and PRs should follow the canonical checklist in
[docs/benchmark-result-contributions.md](docs/benchmark-result-contributions.md).

At minimum, include the commit hash, CTest status, configure/build command,
benchmark command, raw CSV artifacts, generated summary, environment metadata,
clocks/persistence note, and claims-policy acknowledgement.

Do not submit performance claims without raw artifacts. Accepted results are
reproducible measurements for a named environment and commit, not universal
performance rankings.

## Scope Boundaries

Security-sensitive API hardening and production cryptography guarantees are out of scope for v1. If a contribution changes cryptographic semantics, include deterministic tests and explain the standards scope.
