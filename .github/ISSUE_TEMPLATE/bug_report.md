---
name: Bug Report
about: Report build, correctness, benchmark, documentation, or reproducibility issues
title: "Bug: "
---

# Bug Report

## Summary

Describe the bug and affected area.

## Environment

- Commit:
- OS:
- GPU model:
- CUDA Toolkit:
- NVIDIA driver:
- Compiler:
- CMake command:

## Commands

```text
ctest --test-dir build --output-on-failure
```

Include the configure/build command and any benchmark or debug commands if relevant.

## Expected Behavior

What should happen?

## Actual Behavior

What happened instead?

## Artifacts

Attach raw benchmark files when benchmark behavior is affected:

- `run_metadata.csv`
- `thr_gpu.csv`
- `thr_cpu.csv`
- `summary.md`

Do not include secrets or sensitive security details in a public bug report. Use the security policy for security-sensitive reports.
