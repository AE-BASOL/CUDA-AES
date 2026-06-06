# CUDA AES Benchmark Documentation

This documentation index helps readers navigate the CUDA-AES Benchmark project as a GPU AES benchmark, AES GPU performance study, and reproducible cryptography benchmark. Start here when looking for CUDA AES mode coverage, correctness evidence, benchmark methodology, raw-result packaging, profiling support, or legacy provenance notes.

## Recommended Reading Path

1. [Mode Matrix](modes.md) - implemented, tested, benchmarked, documented, and planned CUDA AES modes, including ECB, CBC, CFB, OFB, CTR, GCM/GMAC, CCM, XTS-AES, AES-KW, and AES-KWP.
2. [Correctness](correctness.md) - known-answer tests, authentication rejection checks, AES-GCM CUDA scope, AES-128 CUDA and AES-256 CUDA coverage, and environment-limited verification notes.
3. [Benchmark Methodology](benchmark-methodology.md) - reproducible GPU AES benchmark procedure, raw CSV files, timing scopes, and limitations for AES GPU performance comparisons.
4. [Results](results.md) - how to package benchmark results with commit, environment, raw artifacts, generated summaries, and claims policy.
5. [Architecture](architecture.md) - canonical CUDA C++ source layout, benchmark runtime flow, module boundaries, and current implementation constraints.
6. [Profiling](profiling.md) - NVTX, Nsight Systems, PTX dump helpers, and profiling rules that preserve reproducibility.
7. [Legacy Tezcan Implementation](legacy-tezcan.md) - Legacy and Provenance notes for the original GPU AES optimization material and why it is not the canonical benchmark target.

## Scope Guardrails

CUDA-AES Benchmark is benchmark and research software. It is not a production cryptography library, and benchmark claims should stay tied to correctness checks, raw artifacts, environment metadata, and the timing scope recorded by the benchmark harness.

Return to the [top-level README](../README.md) for build, test, benchmark, governance, and roadmap information.
