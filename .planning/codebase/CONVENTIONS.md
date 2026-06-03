---
mapped_at: 2026-06-04
last_mapped_commit: unknown
focus: quality
---

# Conventions

## Summary

The codebase uses direct CUDA C++ with explicit source files per AES mode and key size. It favors performance-oriented low-level code, manual memory management, global constants, and benchmark-oriented output over framework abstractions.

## Code Style

- Top-level code uses C++17 and CUDA C++17.
- Braces are generally K&R style for functions, loops, and conditionals.
- Many helper functions are marked `static`, `inline`, `__device__`, or `__forceinline__` where locality or performance matters.
- CUDA kernels are declared `__global__` and exposed through `aes_common.h`.
- Constants in `main.cu` use `constexpr` or `static const`, such as `THREADS_PER_BLOCK`, `NUM_RUNS`, `SIZES`, and `MODES`.
- Legacy code relies heavily on `#define` constants in `cihangirTezcanAESimplementation/AES_final.h`.

## Error Handling

- Top-level CUDA calls are wrapped by the `CHECK_CUDA` macro in `main.cu`.
- `CHECK_CUDA` prints file, line, and `cudaGetErrorString(err)`, then exits.
- `aes_tables.cu` checks `cudaMemcpyToSymbol` failures and exits with `EXIT_FAILURE`.
- Filesystem directory creation in `main.cu` catches `std::filesystem::filesystem_error` and prints an error, but sometimes continues.
- CSV write failures in `main.cu` are reported with `fprintf(stderr, ...)`.
- OpenSSL EVP calls in `cpu_aes_throughput()` do not check return values.

## Memory Management

- `main.cu` uses explicit CUDA allocation and free calls.
- Host benchmark buffers use pinned memory through `cudaMallocHost`.
- Device buffers use `cudaMalloc` and are copied with `cudaMemcpy`.
- There is no RAII wrapper for CUDA resources, events, or pinned memory.
- Kernel modules use reinterpret casts to `uint4*`, `uint32_t*`, and `uint64_t*` for block-oriented processing.

## Performance Patterns

- AES tables and round keys are stored in device constant memory.
- ECB kernels process two 16-byte blocks per thread where possible.
- Kernels use `#pragma unroll` in AES rounds and table generation paths.
- CMake enables `--use_fast_math`, `-O3`, PTXAS verbose output, and register count limits.
- Benchmark timing uses CUDA events around kernel launches.
- Optional NVTX ranges are available through `profiling_helpers.h`.

## Output Patterns

- Console output uses `printf`, `fprintf`, and `std::cout`.
- Machine-readable benchmark rows are printed with prefixes such as `RESULT_GPU`, `RESULT_CPU`, `ROUND_TRIP_CHECK`, and `CTR_PREVIEW`.
- CSV files are opened in append mode under `bench/`.
- `ensure_csv_header()` in `main.cu` creates headers when output files are absent.

## Duplication Patterns

- AES-128 and AES-256 implementations are separate files with similar structure.
- Top-level and `v3/` implementations are largely duplicated.
- GCM code duplicates GF multiplication and AES counter encryption logic inside the GCM kernels rather than reusing CTR kernels.
- Legacy code contains multiple specialized kernels for optimization variants.

## Comments And Language

- Comments are mixed English and Turkish.
- Some comments contain mojibake/encoding artifacts in the current workspace rendering, especially in `profiling_helpers.h` and CMake comments.
- Several comments indicate local experimentation, such as register-count tuning, Windows getopt compatibility, and profiling targets.

## Build Conventions

- Source files are explicitly listed in CMake rather than discovered by globbing.
- Build assumptions are embedded in CMake files, including CUDA architecture, Windows paths, OpenSSL paths, and Nsight paths.
- `.gitignore` is configured for CMake, CUDA, CLion, Nsight, and benchmark outputs.

## Documentation Conventions

- `README.md` is minimal and focused on build/run instructions and benchmark output shape.
- `TEZCAN_README.md` documents provenance and claimed historical performance for the legacy code.
- There are no generated API docs or inline Doxygen-style public API contracts.

