---
phase: 05-confidentiality-mode-expansion
status: planned
nyquist_compliant: true
created: 2026-06-05T11:03:00+03:00
---

# Phase 5 Validation Strategy

## Source-Level Checks

- `rg "aes128_cbc|aes256_cbc|aes128_cfb|aes256_cfb|aes128_ofb|aes256_ofb" aes_common.h CMakeLists.txt *.cu tests`
- `rg "CBC|CFB|OFB|CFB-128|known|KAT|expected" tests/kat_main.cu docs README.md`
- `rg "cbc-128|cbc-256|cfb-128|cfb-256|ofb-128|ofb-256" main.cu`
- `rg "EVP_aes_128_cbc|EVP_aes_256_cbc|EVP_aes_128_cfb128|EVP_aes_256_cfb128|EVP_aes_128_ofb|EVP_aes_256_ofb" main.cu`
- `rg "confidentiality-only|feedback|dependency|CFB-128" docs README.md`

## Runtime Checks

Run from a CUDA/MSVC-ready shell:

```powershell
cmake -S . -B C:\tmp\cuda-aes-phase5-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build C:\tmp\cuda-aes-phase5-build --config Release
ctest --test-dir C:\tmp\cuda-aes-phase5-build --output-on-failure
.\C:\tmp\cuda-aes-phase5-build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir C:\tmp\cuda-aes-phase5-bench
```

If the shell cannot provide `cl.exe` to `nvcc`, record the configure failure in `05-VERIFICATION.md` and keep the source-level checks explicit.

## Per-Plan Verification Map

| Plan | Wave | Requirement | Automated Command | Status |
|------|------|-------------|-------------------|--------|
| 05-01 | 1 | MODE-02 | `rg "cbc-128|cbc-256|CBC" aes_common.h CMakeLists.txt tests main.cu docs` | pending |
| 05-02 | 2 | MODE-03, MODE-04 | `rg "cfb-128|cfb-256|ofb-128|ofb-256|CFB-128|OFB" aes_common.h CMakeLists.txt tests main.cu docs` | pending |
| 05-03 | 3 | MODE-02, MODE-03, MODE-04 | `rg "confidentiality-only|feedback|dependency|CBC|CFB|OFB" README.md docs .planning/phases/05-confidentiality-mode-expansion` | pending |

## Acceptance

Phase 5 passes when MODE-02, MODE-03, and MODE-04 each have:

- AES-128 and AES-256 source coverage.
- Deterministic correctness tests.
- Benchmark rows in the existing harness.
- Documentation that labels confidentiality-only behavior and dependency-chain limitations.
