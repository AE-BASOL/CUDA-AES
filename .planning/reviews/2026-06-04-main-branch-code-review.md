# Main Branch Code Review

**Date:** 2026-06-04
**Scope:** `https://github.com/AE-BASOL/CUDA-AES.git`
**Local path:** `C:\Users\efebasol\Desktop\_Organized_2026-05-19\Projects\cuda\CUDA-AES`
**Reviewed HEAD:** `3d51a8b`
**Build verification:** `cmake -S . -B C:\tmp\cuda-aes-review-build -DCMAKE_BUILD_TYPE=Release` failed because `nvcc` could not find `cl.exe`; build/tests could not run.
**Git state at review:** clean after clone.

## Findings

### HIGH: GCM decrypt accepts unauthenticated ciphertext

**Files:** `aes128_gcm.cu:215`, `aes256_gcm.cu:177`, `main.cu:431`

GCM decrypt kernels compute a tag into `tagOut`, but the input tag is not checked before accepting plaintext. `main.cu` marks GCM round-trip as PASS when plaintext matches while skipping tag comparison.

**Impact:** AES-GCM is authenticated encryption. Decrypting without tag verification defeats the main security property of GCM.

**Planning implication:** Phase 2 must require tag verification before any public benchmark claim treats GCM as correct.

### HIGH: GCM IV broadcast is wrong outside the first warp

**Files:** `aes128_gcm.cu:88`, `aes256_gcm.cu:69`

`IV_lo` and `IV_hi` are local variables initialized only in `threadIdx.x == 0`, then shared through `__shfl_sync`. `__shfl_sync` is warp-local, not block-wide. Warps 1-7 can receive lane 0 values from their own warp where IV was never loaded.

**Impact:** GCM encryption/decryption with 256 threads can use wrong counters for most threads. Round-trip can still pass because encryption and decryption repeat the same bug.

**Planning implication:** Phase 2 must replace warp-local IV broadcast with shared-memory block-wide broadcast and test multi-warp behavior.

### HIGH: GCM tag is not standard AES-GCM

**Files:** `aes128_gcm.cu:152`, `aes256_gcm.cu:127`

The tag output is GHASH over ciphertext blocks only. Standard GCM also includes the length block and final XOR with `E(K, J0)`. AAD is not represented.

**Impact:** Output will not match OpenSSL or NIST AES-GCM vectors.

**Planning implication:** Phase 2 must add NIST/OpenSSL vectors and update GCM tag generation to the standard formula before expanding benchmark claims.

### MEDIUM: CMake contains maintainer-local absolute paths

**File:** `CMakeLists.txt:44`

CMake hardcodes CUDA include paths, OpenSSL paths under `C:/Users/efebasol/...`, Nsight path, GPU architecture `86`, and a placeholder `CMAKE_CUDA_HOST_COMPILER` after `project()`.

**Impact:** New contributors cannot build reliably. The reviewer configure attempt failed at CUDA host compiler detection.

**Planning implication:** Phase 1 must remove private paths, use CMake package discovery/cache variables, and document CUDA host compiler setup.

### LOW: CUDA events are never destroyed

**Files:** `main.cu:185`, `main.cu:490`

`cudaEventCreate` is called in benchmark paths without matching `cudaEventDestroy`.

**Impact:** Long benchmark runs can leak CUDA event resources and distort measurements.

**Planning implication:** Phase 3 should include resource cleanup as part of benchmark harness credibility.

## Recommended Actions

1. Fix AES-GCM correctness first: tag formula, tag verification, and IV broadcast.
2. Add NIST/OpenSSL known-answer tests before optimizing further.
3. Make CMake contributor-friendly with `find_package(OpenSSL REQUIRED)` and no personal absolute paths.
4. Remove or clearly separate duplicated `v3` sources so maintainers know which code is canonical.

