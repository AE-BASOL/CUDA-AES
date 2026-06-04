---
phase: 02-correctness-baseline
researched_at: 2026-06-04
status: complete
sources:
  - NIST SP 800-38A, Recommendation for Block Cipher Modes of Operation: Methods and Techniques
  - NIST SP 800-38D, Recommendation for GCM and GMAC
  - NIST CAVP block cipher and GCM validation references
  - OpenSSL EVP AES and AEAD documentation
---

# Phase 2 Research: Correctness Baseline

## Research Question

What do we need to know to plan a correctness baseline for CUDA-AES that proves ECB, CTR, and GCM outputs before benchmark claims are trusted?

## Primary References

- NIST SP 800-38A defines ECB and CTR mode behavior and includes AES mode example vectors in Appendix F.
- NIST SP 800-38D defines GCM/GMAC as authenticated encryption with associated data. For 96-bit IVs, `J0 = IV || 0x00000001`; GCTR for plaintext/ciphertext uses `inc32(J0)`, and the authentication tag is `MSBt(GCTR_K(J0, GHASH_H(A, C)))`.
- NIST CAVP/ACVP references provide validation framing and vector sources for AES modes. Passing local vectors is not CAVP validation, but it is the minimum open-source correctness evidence.
- OpenSSL EVP is already a project dependency and can be used as an oracle for small deterministic checks. For GCM decrypt, the expected API shape is set expected tag, run decrypt, then treat finalization failure as authentication failure.

## Current Code Facts

- `main.cu` has embedded random round-trip checks but no deterministic known-answer test command.
- `main.cu` default sizes include 1 MiB, 10 MiB, 100 MiB, and 1 GiB, so a correctness smoke test needs a separate small path.
- `cpu_aes_throughput()` uses OpenSSL for timing but does not compare CPU and GPU bytes.
- `aes128_gcm.cu` and `aes256_gcm.cu` compute a GHASH-like value over ciphertext blocks only. They omit the GCM length block and final `E(K, J0)` XOR.
- GCM decrypt kernels accept `tag` but do not verify it before plaintext is accepted.
- GCM IV/counter setup initializes local variables in `threadIdx.x == 0` and broadcasts with `__shfl_sync`, which is warp-local. Threads outside the first warp can receive uninitialized per-warp lane 0 values.
- GCM kernels currently launch with one CUDA block and 256 threads for tag computation, so a multi-warp test is necessary to catch the IV broadcast issue.

## Planning Implications

1. Add test harness first.
   - This phase should not start by editing GCM kernels blindly.
   - The first plan should introduce deterministic KAT data and a fast `--kat` or test executable path.
   - The harness should cover AES-128 and AES-256 for ECB, CTR, and GCM.

2. Use both fixed vectors and OpenSSL oracle checks.
   - Fixed NIST vectors make regressions stable.
   - OpenSSL oracle checks make it easier to compare small buffers when kernel signatures or harness shape changes.
   - GCM negative tests must mutate ciphertext/tag and require rejection.

3. GCM implementation should be corrected in a contained step.
   - First fix IV/J0 handling with shared memory or reusable helper logic.
   - Then compute standard tag: GHASH over AAD and ciphertext with the final length block, XORed with `E(K, J0)`.
   - If full AAD API is too large for Phase 2, support empty AAD explicitly and structure helpers so AAD can be added later.

4. Decrypt API semantics must become explicit.
   - A matching plaintext alone is not success for GCM.
   - Host-side code should compare computed vs expected tag before accepting results.
   - Prefer a host wrapper or test helper that copies plaintext only after tag verification, or at minimum returns a verification flag and makes tests/round-trip fail on mismatch.

5. Build/test integration should be minimal and portable.
   - Add CTest and a small correctness target if practical.
   - The local machine may still lack `cl.exe` in PATH, so plans must distinguish source-level checks from environment-limited build verification.

## Validation Architecture

Phase 2 validation should be test-first:

- Wave 1 creates the deterministic correctness harness and vectors.
- Wave 2 fixes GCM internals and authentication behavior under the harness.
- Wave 3 documents and wires the correctness command into CMake/README.

Quick command target:

- Source-level: `cmake -S . -B C:\tmp\cuda-aes-phase2-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`
- Runtime when build environment is available: `ctest --test-dir C:\tmp\cuda-aes-phase2-build --output-on-failure`
- If local CUDA host compiler remains unavailable, executor must still run static checks and record the configure failure reason.

## Risks

- Existing GCM code has endianness-sensitive counter and GHASH operations; tests must catch byte-order mistakes.
- GCM current kernels are optimized around one-block-grid tag computation; plan should not silently generalize multi-block grid behavior.
- Adding a large test framework could distract from fixing correctness. A small native test executable or `--kat` path is enough for Phase 2.

## RESEARCH COMPLETE
