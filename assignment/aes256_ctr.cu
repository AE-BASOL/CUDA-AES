#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];

__global__ void aes256_ctr_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    const uint32_t *rk = d_roundKeys;  // 60 words for AES-256

    uint64_t ctr_lo = ctrLo + idx;
    uint64_t ctr_hi = ctrHi;
    if (ctr_lo < ctrLo) ctr_hi += 1ULL;

    uint32_t s0 = (uint32_t)(ctr_lo);
    uint32_t s1 = (uint32_t)(ctr_lo >> 32);
    uint32_t s2 = (uint32_t)(ctr_hi);
    uint32_t s3 = (uint32_t)(ctr_hi >> 32);

    // AES-256 encrypt counter
    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0, t1, t2, t3;
    #pragma unroll
    for (int r = 1; r <= 13; ++r) {
        t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1 >> 8) & 0xFF] ^
             d_T2[(s2 >> 16) & 0xFF] ^ d_T3[(s3 >> 24) & 0xFF] ^ rk[4*r + 0];
        t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2 >> 8) & 0xFF] ^
             d_T2[(s3 >> 16) & 0xFF] ^ d_T3[(s0 >> 24) & 0xFF] ^ rk[4*r + 1];
        t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3 >> 8) & 0xFF] ^
             d_T2[(s0 >> 16) & 0xFF] ^ d_T3[(s1 >> 24) & 0xFF] ^ rk[4*r + 2];
        t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0 >> 8) & 0xFF] ^
             d_T2[(s1 >> 16) & 0xFF] ^ d_T3[(s2 >> 24) & 0xFF] ^ rk[4*r + 3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }
    // Final round for AES-256
    uint8_t k[16];
    const uint8_t *sb = d_sbox;
    uint8_t *ko = k;
    ko[0]  = sb[s0 & 0xFF];        ko[1]  = sb[s1 & 0xFF];
    ko[2]  = sb[s2 & 0xFF];        ko[3]  = sb[s3 & 0xFF];
    ko[4]  = sb[(s1 >> 8) & 0xFF]; ko[5]  = sb[(s2 >> 8) & 0xFF];
    ko[6]  = sb[(s3 >> 8) & 0xFF]; ko[7]  = sb[(s0 >> 8) & 0xFF];
    ko[8]  = sb[(s2 >> 16) & 0xFF];ko[9]  = sb[(s3 >> 16) & 0xFF];
    ko[10] = sb[(s0 >> 16) & 0xFF];ko[11] = sb[(s1 >> 16) & 0xFF];
    ko[12] = sb[(s3 >> 24) & 0xFF];ko[13] = sb[(s0 >> 24) & 0xFF];
    ko[14] = sb[(s1 >> 24) & 0xFF];ko[15] = sb[(s2 >> 24) & 0xFF];
    ((uint32_t*)k)[0] ^= rk[56];
    ((uint32_t*)k)[1] ^= rk[57];
    ((uint32_t*)k)[2] ^= rk[58];
    ((uint32_t*)k)[3] ^= rk[59];

    // XOR keystream with input data
    const uint8_t *pt = in + idx * 16;
    uint8_t *ct = out + idx * 16;
    ((uint32_t*)ct)[0] = ((const uint32_t*)pt)[0] ^ ((const uint32_t*)k)[0];
    ((uint32_t*)ct)[1] = ((const uint32_t*)pt)[1] ^ ((const uint32_t*)k)[1];
    ((uint32_t*)ct)[2] = ((const uint32_t*)pt)[2] ^ ((const uint32_t*)k)[2];
    ((uint32_t*)ct)[3] = ((const uint32_t*)pt)[3] ^ ((const uint32_t*)k)[3];
}

__global__ void aes256_ctr_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    // Same as encryption
    aes256_ctr_encrypt<<<gridDim, blockDim>>>(in, out, nBlocks, ctrLo, ctrHi);
    // (See note in aes128_ctr_decrypt regarding actual implementation.)
}
