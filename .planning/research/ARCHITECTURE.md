# Research: Architecture

## Question

How should the existing codebase be structured to support a living open-source benchmark suite?

## Recommended Components

- `src/` or top-level CUDA sources for production benchmark kernels.
- `include/` for shared headers such as `aes_common.h`.
- `benchmarks/` or `tools/` for benchmark runners and environment capture.
- `tests/` for known-answer tests and small deterministic regression checks.
- `docs/` for human-readable methodology and results.
- `.github/` for issue templates, PR template, and CI workflows.
- `legacy/` or `third_party/` for the Tezcan implementation, with clear provenance and license notes.

## Data Flow

1. Build config discovers CUDA, OpenSSL, and compiler settings.
2. Tests run deterministic vectors before benchmarks are trusted.
3. Benchmark script captures environment and runs modes/sizes/repetitions.
4. Raw output is saved as CSV/JSON.
5. Summary tables are generated from raw output.
6. README and docs link to versioned results.

## Build Order Implications

1. Clean repository and portable CMake first.
2. Add tests before changing benchmark behavior.
3. Add benchmark scripts after correctness is stable.
4. Add public docs after methodology is trustworthy.
5. Add SEO/GitHub metadata after public-facing wording is settled.

## Sources

- Current codebase map: `.planning/codebase/ARCHITECTURE.md`
- Current concerns map: `.planning/codebase/CONCERNS.md`

