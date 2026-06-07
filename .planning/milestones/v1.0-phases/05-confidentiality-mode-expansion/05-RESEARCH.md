---
phase: 05-confidentiality-mode-expansion
status: complete
researched_at: 2026-06-05T11:03:00+03:00
requirements: [MODE-02, MODE-03, MODE-04]
---

# Phase 5 Research: Confidentiality Mode Expansion

## RESEARCH COMPLETE

Phase 5 adds the missing SP 800-38A confidentiality modes: CBC, CFB, and OFB. The current canonical code implements ECB, CTR, and GCM through top-level CUDA files, exposes kernel prototypes in `aes_common.h`, wires kernel sources through `CMakeLists.txt`, validates deterministic known-answer tests in `tests/kat_main.cu`, and emits benchmark rows from `main.cu`.

## Current Architecture Findings

- Canonical implementation is top-level source, not `v3/` or `cihangirTezcanAESimplementation/`.
- Kernel declarations live in `aes_common.h`; concrete mode kernels live in per-key-size `.cu` files.
- `CMAKE_KERNEL_SOURCES` in `CMakeLists.txt` must include every kernel file used by both `CudaProject` and `CudaAesKat`.
- `tests/kat_main.cu` already has small helpers for loading keys, allocating device buffers, launching kernels, and exact byte comparisons.
- `main.cu` owns benchmark mode enumeration through `MODES[]`, mode classification, round-trip checks, GPU kernel timing, CPU OpenSSL baseline selection, and CSV row output.
- Documentation status is centered in `docs/modes.md`, `docs/correctness.md`, `docs/benchmark-methodology.md`, and `README.md`.

## Mode Semantics

### CBC

CBC encryption computes `C_i = AES_K(P_i xor C_{i-1})`, with `C_-1 = IV`. CBC decryption computes `P_i = AES_DEC_K(C_i) xor C_{i-1}`. Encryption has a true per-message dependency chain; decryption can be parallelized across blocks after reading the prior ciphertext block. For Phase 5, correctness matters more than pretending CBC is naturally parallel.

Planning implication:
- Implement a correct AES-128/AES-256 CBC path first, even if encryption uses a single chained kernel or host/device orchestration.
- Add NIST-style multi-block known-answer tests for encrypt and decrypt.
- Benchmark rows must make the dependency-chain limitation visible in docs if throughput is not comparable to CTR/GCM.

### CFB

CFB is a feedback mode built on AES encryption. CFB-128 uses full 128-bit segments: `C_i = P_i xor AES_K(C_{i-1})`, with `C_-1 = IV`; decrypt uses `P_i = C_i xor AES_K(C_{i-1})`. Other segment sizes exist, but they introduce byte/bit-shift feedback behavior and complicate benchmark comparability.

Planning implication:
- Phase 5 should explicitly choose CFB-128 segment size only.
- Add correctness tests and docs that name the segment-size scope.
- CFB encryption is chained; CFB-128 decryption can be parallelized per block using prior ciphertext.

### OFB

OFB generates a keystream by repeatedly encrypting the feedback state: `O_i = AES_K(O_{i-1})`, `C_i = P_i xor O_i`, with `O_-1 = IV`. Encryption and decryption are identical XOR operations once the keystream exists, but the keystream itself is chained.

Planning implication:
- Implement AES-128/AES-256 OFB with shared encrypt/decrypt path.
- Add multi-block known-answer tests for both directions.
- Benchmark docs must identify OFB as confidentiality-only and chained keystream generation.

## Implementation Strategy

Use the repo's existing top-level pattern:

- Add mode-specific CUDA sources, likely `aes128_cbc.cu`, `aes256_cbc.cu`, `aes128_cfb.cu`, `aes256_cfb.cu`, `aes128_ofb.cu`, and `aes256_ofb.cu`.
- Add prototypes to `aes_common.h`.
- Add sources to `CUDA_KERNEL_SOURCES` so both `CudaProject` and `CudaAesKat` link the new kernels.
- Reuse constant round keys and AES tables from `aes_tables.cu`.
- Prefer local device helpers that match existing ECB/CTR style unless a small shared helper naturally reduces duplication without destabilizing existing modes.
- Keep buffer scope to full 16-byte blocks, matching current benchmark and KAT assumptions.

## Test Strategy

Use deterministic vectors before benchmark claims:

- CBC AES-128 and AES-256 known-answer encrypt/decrypt tests.
- CFB-128 AES-128 and AES-256 known-answer encrypt/decrypt tests.
- OFB AES-128 and AES-256 known-answer encrypt/decrypt tests.
- Reuse `tests/kat_main.cu` helper style and keep small buffers.
- Add OpenSSL cross-checks only as supplemental evidence; fixed expected vectors should remain primary.

## Benchmark Strategy

Update `main.cu` in one place after correctness paths exist:

- Extend `MODES[]` with `cbc-128`, `cbc-256`, `cfb-128`, `cfb-256`, `ofb-128`, and `ofb-256`.
- Add mode classifiers alongside `isEcb`, `isCtr`, and `isGcm`.
- Use 16-byte IVs for CBC/CFB/OFB, while preserving 12-byte IV handling for CTR/GCM.
- Add OpenSSL CPU baseline selection for CBC, CFB-128, and OFB.
- Preserve the Phase 3 CSV schema and `kernel_only` timing scope.
- Keep round-trip checks before benchmark rows.

## Documentation Strategy

Update mode and correctness docs after source/test/benchmark wiring:

- `docs/modes.md` should mark CBC, CFB, and OFB implemented/tested/benchmarked after Phase 5.
- CFB documentation must state Phase 5 implements CFB-128 only.
- `docs/benchmark-methodology.md` should warn that CBC/CFB/OFB have feedback dependencies and should not be read as naturally parallel CTR-like modes.
- README and correctness docs should reflect the expanded confidentiality-only coverage.

## Validation Architecture

Validation should sample after every plan:

- Source-level grep checks prove declarations, CMake wiring, mode enumeration, and documentation updates.
- KAT build/test remains the primary runtime verification when a CUDA/MSVC-ready shell is available.
- If this shell still lacks `cl.exe`, runtime CMake/CTest/benchmark verification should be recorded as environment-limited, not silently omitted.

## Risks

- CBC/CFB/OFB may be incorrectly benchmarked as if they were embarrassingly parallel.
- CFB segment size ambiguity can cause mismatched vectors or misleading docs.
- Chained modes can pass random round-trip checks while failing standard vectors.
- Adding many mode branches in `main.cu` can accidentally break existing ECB/CTR/GCM benchmark behavior.

## Recommended Plan Shape

1. Add common wiring and CBC correctness path.
2. Add CFB-128 and OFB correctness paths.
3. Add benchmark rows, docs, and Phase 5 verification.
