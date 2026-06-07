---
phase: 06-authenticated-and-specialized-mode-expansion
status: complete
researched_at: 2026-06-05T15:55:00+03:00
requirements: [MODE-05, MODE-06, MODE-07, MODE-08]
---

# Phase 6 Research: Authenticated And Specialized Mode Expansion

## RESEARCH COMPLETE

Phase 6 adds CCM, XTS-AES, AES-KW, AES-KWP, and documentation boundaries for GMAC and CMAC. The canonical repo shape is already established: top-level CUDA sources implement each mode, `aes_common.h` declares kernel entry points, `CMakeLists.txt` wires kernel sources into both `CudaProject` and `CudaAesKat`, `tests/kat_main.cu` owns deterministic known-answer tests, `main.cu` owns benchmark enumeration/dispatch/CSV output, and public scope is described in `README.md` plus `docs/*.md`.

The phase should keep the project positioned as a reproducible benchmark suite, not a production cryptography library. Standards-sensitive modes need deterministic vectors before benchmark claims.

## Current Architecture Findings

- Existing canonical modes are top-level files, not `v3/` or `cihangirTezcanAESimplementation/`.
- AES block helpers are available in `aes_block_device.cuh` and are useful for modes that need one-block AES encryption/decryption inside a larger construction.
- `d_roundKeys` is global device constant memory, so multi-key constructions need careful key-schedule sequencing or explicit documentation of host orchestration.
- `tests/kat_main.cu` already supports OpenSSL EVP helpers for supplemental comparisons and small CUDA launch helpers for exact byte checks.
- `main.cu` already handles per-mode IV length differences for GCM versus feedback modes, round-trip checks, CPU baseline selection, and CSV row output.
- Runtime CMake/CTest remains environment-limited in this shell when `nvcc` cannot find `cl.exe`; plans must preserve source-level verification checks and record runtime debt clearly.

## Mode Semantics

### CCM

CCM combines CTR encryption with CBC-MAC authentication. It is an AEAD mode with nonce length, tag length, associated data, and payload length encoded into the authentication input. The implementation scope should be deliberately narrow for v1 benchmark credibility.

Planning implications:
- Lock a concrete benchmark/KAT scope: AES-128 and AES-256, 96-bit nonce unless vectors require another length, empty AAD or one explicit non-empty-AAD vector if feasible, 16-byte tag, and block-aligned payloads unless partial-block support is explicitly implemented.
- Add KATs that verify ciphertext and tag, not only round trips.
- Benchmark output and docs must expose nonce/tag/AAD assumptions so CCM is not treated as a generic AEAD API.

### XTS-AES

XTS-AES is a storage-oriented confidentiality mode using two AES keys. It encrypts data units with a tweak derived from a sector number; it is not an authenticated mode. Full XTS includes ciphertext stealing for non-block-multiple data units, but this benchmark can initially scope to full 16-byte blocks.

Planning implications:
- Add AES-128-XTS and AES-256-XTS coverage in terms of total key material: 256-bit total for AES-128-XTS and 512-bit total for AES-256-XTS.
- Document sector/tweak handling and block-aligned benchmark scope.
- Add deterministic vectors for full-block data units before benchmark rows.
- Be careful with two AES key schedules because the current device constant memory stores one active schedule at a time. A staged host/device implementation may be acceptable for source-level correctness and benchmark coverage if clearly documented.

### AES-KW and AES-KWP

AES-KW wraps key material using an integrity register and six rounds over 64-bit semiblocks. AES-KWP adds padding support for inputs that are not a multiple of 64 bits. These are key-management workloads, not bulk encryption modes.

Planning implications:
- Implement small-payload wrap/unwrap paths focused on key sizes and key-encryption-key sizes from standard vectors.
- Add exact vectors for AES-KW and AES-KWP, including unwrap failure semantics for tampered wrapped data.
- Benchmark with key-wrap-sized payload batches or explicitly label rows as key-wrap workloads. Avoid presenting GiB/s rows as comparable to streaming encryption modes unless the docs explain the workload shape.

### GMAC and CMAC Boundaries

GMAC is GCM authentication over AAD with no plaintext encryption. CMAC is a block-cipher MAC. Neither should be represented as bulk encryption throughput in v1. The current roadmap only requires documentation boundaries for Phase 6; standalone GMAC/CMAC benchmark coverage is v2 (`MODE-09`).

Planning implications:
- Update mode matrix and correctness/benchmark docs so readers distinguish AEAD, confidentiality-only, key-wrap, and MAC-only workloads.
- Preserve GCM as implemented and GMAC as not implemented unless a future phase adds dedicated rows.
- Mark CMAC as MAC-only and not part of encryption-mode throughput.

## Implementation Strategy

Use the existing top-level pattern:

- Add `aes128_ccm.cu`, `aes256_ccm.cu`, `aes128_xts.cu`, `aes256_xts.cu`, `aes128_kw.cu`, and `aes256_kw.cu` only if the implementation needs separate key-size modules. If KW/KWP can share a compact host/device helper, still expose clear AES-128/AES-256 entry points.
- Add prototypes to `aes_common.h` and sources to `CUDA_KERNEL_SOURCES`.
- Prefer correctness-first kernels and host orchestration over premature parallel optimization.
- Reuse `aes_block_device.cuh` for AES block encryption/decryption helpers.
- Keep full-block benchmark constraints explicit for CCM and XTS unless partial-block behavior is implemented and tested.
- Use OpenSSL EVP helpers in tests only where they strengthen evidence; fixed standard vectors remain the primary acceptance mechanism.

## Test Strategy

- CCM AES-128 and AES-256 ciphertext/tag KATs using fixed nonce/tag/AAD assumptions.
- CCM decrypt checks that tag mismatches are rejected before plaintext is accepted as valid.
- XTS-AES AES-128-XTS and AES-256-XTS full-block data-unit KATs.
- AES-KW and AES-KWP wrap/unwrap vectors for AES-128 and AES-256 KEKs.
- AES-KW/AES-KWP tamper tests where unwrap failure is observable.
- Existing ECB/CBC/CFB/OFB/CTR/GCM KATs remain in the same `CudaAesKat` executable.

## Benchmark Strategy

- Add benchmark labels that encode workload identity clearly: `ccm-128`, `ccm-256`, `xts-128`, `xts-256`, `kw-128`, `kw-256`, `kwp-128`, and `kwp-256`, or another consistent naming scheme documented in `docs/modes.md`.
- CCM rows should document nonce length, tag length, AAD scope, and message-size scope.
- XTS rows should document sector size, tweak/sector number convention, and block-aligned scope.
- KW/KWP rows should be treated as key-wrap workloads with small payload batches rather than generic streaming throughput.
- Preserve Phase 3 CSV schema and metadata fields unless a planned schema extension is required and documented.

## Documentation Strategy

- Update `docs/modes.md` status matrix after implementation and test coverage exists.
- Update `docs/correctness.md` to list CCM, XTS-AES, AES-KW, and AES-KWP KAT coverage and scope limits.
- Update `docs/benchmark-methodology.md` and `docs/results.md` to separate AEAD, storage, key-wrap, confidentiality-only, and MAC-only interpretation.
- Update README coverage table only after source-level evidence exists.
- Document GMAC and CMAC as authentication/MAC-only boundaries, with standalone benchmarking deferred to v2 unless implemented now.

## Validation Architecture

Validation should sample after every plan:

- Source-level `rg` checks prove declarations, CMake wiring, mode enumeration, KAT names, benchmark labels, and docs.
- Runtime verification is CMake configure/build plus `ctest --test-dir <build> --output-on-failure` when a CUDA/MSVC-ready shell is available.
- Small benchmark smoke should run with `--runs 1 --sizes 1048576 --bench-dir bench/smoke-phase6` if the build succeeds.
- If `cl.exe` remains unavailable, verification artifacts must record the exact blocker and commands needed to close it.

## Risks

- AEAD modes can pass ciphertext round trips while failing tag semantics.
- XTS two-key handling can be wrong if round keys are overwritten in device constant memory.
- AES-KW/KWP can be misleading if benchmarked as large-buffer encryption.
- GMAC/CMAC documentation can blur authentication/MAC-only workloads with encryption throughput.
- Adding many branches to `main.cu` can regress existing mode dispatch.

## Recommended Plan Shape

1. Add CCM source, KATs, benchmark dispatch, and AEAD parameter docs.
2. Add XTS-AES source, KATs, benchmark dispatch, and storage-sector docs.
3. Add AES-KW/AES-KWP source, KATs, benchmark dispatch, and key-wrap workload docs.
4. Finalize mode matrix, GMAC/CMAC boundaries, verification evidence, and handoff.
