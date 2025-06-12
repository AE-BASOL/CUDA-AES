#include "aes_common.h"
#include <stdint.h>

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint64_t d_H_pow_hi[32];
extern __device__ __constant__ uint64_t d_H_pow_lo[32];

// Reuse the gf_mul128 device function from aes128_gcm.cu (same implementation)
static __device__ inline void gf_mul128(uint64_t &Ah, uint64_t &Al, uint64_t Bh, uint64_t Bl) {
    uint64_t Zh = 0ull, Zl = 0ull;
    uint64_t Vh = Bh, Vl = Bl;
    const uint64_t R = 0xE100000000000000ULL;
    for (int i = 0; i < 128; ++i) {
        if (Al & 1ULL) {
            Zl ^= Vl; Zh ^= Vh;
        }
        bool carry = (Vl & 1ULL);
        Vl = (Vl >> 1) | (Vh << 63);
        Vh = (Vh >> 1);
        if (carry) Vh ^= R;
        Al = (Al >> 1) | (Ah << 63);
        Ah >>= 1;
    }
    Ah = Zh;
    Al = Zl;
}

__global__ void aes256_gcm_encrypt(const uint8_t * __restrict__ plain, uint8_t * __restrict__ cipher, size_t nBlocks, const uint8_t * __restrict__ iv, uint8_t * __restrict__ tagOut) {
    // Implementation is analogous to aes128_gcm_encrypt, but using AES-256 (14 rounds).
    // GHASH uses precomputed H powers in constant memory

    // Setup IV counters
    uint64_t IV_lo = 0ull, IV_hi = 0ull;
    if (threadIdx.x == 0) {
        uint32_t w0 = 0, w1 = 0, w2 = 0;
        memcpy(&w0, iv, 4);
        memcpy(&w1, iv + 4, 4);
        memcpy(&w2, iv + 8, 4);
        uint32_t w3 = 0x01000000u;
        IV_lo = (uint64_t)w0 | ((uint64_t)w1 << 32);
        IV_hi = (uint64_t)w2 | ((uint64_t)w3 << 32);
    }
    __syncthreads();
    IV_lo = __shfl_sync(0xFFFFFFFF, IV_lo, 0);
    IV_hi = __shfl_sync(0xFFFFFFFF, IV_hi, 0);

    // CTR encryption (each thread processes blocks in stride)
    for (size_t i = threadIdx.x; i < nBlocks; i += blockDim.x) {
        uint64_t ctr_lo = IV_lo + i;
        uint64_t ctr_hi = IV_hi;
        if (ctr_lo < IV_lo) ctr_hi += 1ULL;
        uint32_t s0 = (uint32_t)ctr_lo;
        uint32_t s1 = (uint32_t)(ctr_lo >> 32);
        uint32_t s2 = (uint32_t)ctr_hi;
        uint32_t s3 = (uint32_t)(ctr_hi >> 32);
        const uint32_t *rk = d_roundKeys;
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        uint32_t t0,t1,t2,t3;
        #pragma unroll
        for (int r = 1; r <= 13; ++r) {
            t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1>>8)&0xFF] ^ d_T2[(s2>>16)&0xFF] ^ d_T3[(s3>>24)&0xFF] ^ rk[4*r+0];
            t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2>>8)&0xFF] ^ d_T2[(s3>>16)&0xFF] ^ d_T3[(s0>>24)&0xFF] ^ rk[4*r+1];
            t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3>>8)&0xFF] ^ d_T2[(s0>>16)&0xFF] ^ d_T3[(s1>>24)&0xFF] ^ rk[4*r+2];
            t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0>>8)&0xFF] ^ d_T2[(s1>>16)&0xFF] ^ d_T3[(s2>>24)&0xFF] ^ rk[4*r+3];
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        uint8_t keystream[16];
        uint8_t *ko = keystream;
        const uint8_t *sb = d_sbox;
        ko[0]  = sb[s0 & 0xFF];       ko[4]  = sb[(s1>>8)&0xFF];
        ko[8]  = sb[(s2>>16)&0xFF];   ko[12] = sb[(s3>>24)&0xFF];
        ko[1]  = sb[s1 & 0xFF];       ko[5]  = sb[(s2>>8)&0xFF];
        ko[9]  = sb[(s3>>16)&0xFF];   ko[13] = sb[(s0>>24)&0xFF];
        ko[2]  = sb[s2 & 0xFF];       ko[6]  = sb[(s3>>8)&0xFF];
        ko[10] = sb[(s0>>16)&0xFF];   ko[14] = sb[(s1>>24)&0xFF];
        ko[3]  = sb[s3 & 0xFF];       ko[7]  = sb[(s0>>8)&0xFF];
        ko[11] = sb[(s1>>16)&0xFF];   ko[15] = sb[(s2>>24)&0xFF];
        ((uint32_t*)keystream)[0] ^= rk[56];
        ((uint32_t*)keystream)[1] ^= rk[57];
        ((uint32_t*)keystream)[2] ^= rk[58];
        ((uint32_t*)keystream)[3] ^= rk[59];
        const uint8_t *ct_in = plain + i * 16;
        uint8_t *ct_out = cipher + i * 16;
        ((uint32_t*)ct_out)[0] = ((const uint32_t*)ct_in)[0] ^ ((uint32_t*)keystream)[0];
        ((uint32_t*)ct_out)[1] = ((const uint32_t*)ct_in)[1] ^ ((uint32_t*)keystream)[1];
        ((uint32_t*)ct_out)[2] = ((const uint32_t*)ct_in)[2] ^ ((uint32_t*)keystream)[2];
        ((uint32_t*)ct_out)[3] = ((const uint32_t*)ct_in)[3] ^ ((uint32_t*)keystream)[3];
    }
    __syncthreads();

    // GHASH via warp-level XOR reduction
    uint32_t tid = threadIdx.x;
    if (tid < 32) {
        uint64_t step_hi = 0ull, step_lo = 1ull;
        for (int b = 0; b < 32; ++b)
            if (32u & (1u << b))
                gf_mul128(step_hi, step_lo, d_H_pow_hi[b], d_H_pow_lo[b]);

        uint64_t pow_hi = 0ull, pow_lo = 1ull;
        uint32_t exp = (uint32_t)(nBlocks - 1 - tid);
        for (int b = 0; b < 32; ++b)
            if (exp & (1u << b))
                gf_mul128(pow_hi, pow_lo, d_H_pow_hi[b], d_H_pow_lo[b]);

        uint64_t accum_hi = 0ull, accum_lo = 0ull;
        for (size_t j = tid; j < nBlocks; j += 32) {
            uint64_t c_lo = ((const uint64_t*)cipher)[2*j + 0];
            uint64_t c_hi = ((const uint64_t*)cipher)[2*j + 1];
            uint64_t tmp_hi = c_hi, tmp_lo = c_lo;
            gf_mul128(tmp_hi, tmp_lo, pow_hi, pow_lo);
            accum_hi ^= tmp_hi;
            accum_lo ^= tmp_lo;
            gf_mul128(pow_hi, pow_lo, step_hi, step_lo);
        }

        for (int off = 16; off > 0; off >>= 1) {
            accum_hi ^= __shfl_xor_sync(0xFFFFFFFF, accum_hi, off);
            accum_lo ^= __shfl_xor_sync(0xFFFFFFFF, accum_lo, off);
        }
        if (tid == 0) {
            uint64_t lenBlock_lo = (uint64_t)nBlocks * 16ull * 8ull;
            uint64_t lenBlock_hi = 0ull;
            accum_lo ^= lenBlock_lo;
            accum_hi ^= lenBlock_hi;
            gf_mul128(accum_hi, accum_lo, d_H_pow_hi[0], d_H_pow_lo[0]);
            ((uint64_t*)tagOut)[0] = accum_lo;
            ((uint64_t*)tagOut)[1] = accum_hi;
        }
    }
}

__global__ void aes256_gcm_decrypt(const uint8_t * __restrict__ cipher, uint8_t * __restrict__ plain, size_t nBlocks, const uint8_t * __restrict__ iv, const uint8_t * __restrict__ tag, uint8_t * __restrict__ tagOut) {
    // Mirror aes256_gcm_encrypt but with ciphertext as input and plaintext as
    // output.  The authentication tag for the ciphertext is produced in
    // tagOut; the provided 'tag' parameter is expected to be checked by the
    // host.

    // GHASH uses precomputed powers of H

    if (threadIdx.x == 0) {
        uint32_t s0=0, s1=0, s2=0, s3=0;
        const uint32_t *rk = d_roundKeys;
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        uint32_t t0,t1,t2,t3;
#pragma unroll
        for (int r = 1; r <= 13; ++r) {
            t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1>>8)&0xFF] ^ d_T2[(s2>>16)&0xFF] ^ d_T3[(s3>>24)&0xFF] ^ rk[4*r+0];
            t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2>>8)&0xFF] ^ d_T2[(s3>>16)&0xFF] ^ d_T3[(s0>>24)&0xFF] ^ rk[4*r+1];
            t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3>>8)&0xFF] ^ d_T2[(s0>>16)&0xFF] ^ d_T3[(s1>>24)&0xFF] ^ rk[4*r+2];
            t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0>>8)&0xFF] ^ d_T2[(s1>>16)&0xFF] ^ d_T3[(s2>>24)&0xFF] ^ rk[4*r+3];
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        uint8_t buf[16];
        const uint8_t *sb = d_sbox;
        buf[0]  = sb[s0 & 0xFF];      buf[4]  = sb[(s1>>8)&0xFF];
        // H subkey handled on host

    uint64_t IV_lo = 0ull, IV_hi = 0ull;
    if (threadIdx.x == 0) {
        uint32_t w0 = 0, w1 = 0, w2 = 0;
        memcpy(&w0, iv, 4);
        memcpy(&w1, iv + 4, 4);
        memcpy(&w2, iv + 8, 4);
        uint32_t w3 = 0x01000000u;
        IV_lo = (uint64_t)w0 | ((uint64_t)w1 << 32);
        IV_hi = (uint64_t)w2 | ((uint64_t)w3 << 32);
    }
    __syncthreads();
    IV_lo = __shfl_sync(0xFFFFFFFF, IV_lo, 0);
    IV_hi = __shfl_sync(0xFFFFFFFF, IV_hi, 0);

    for (size_t i = threadIdx.x; i < nBlocks; i += blockDim.x) {
        uint64_t ctr_lo = IV_lo + i;
        uint64_t ctr_hi = IV_hi;
        if (ctr_lo < IV_lo) ctr_hi += 1ULL;
        uint32_t s0 = (uint32_t)ctr_lo;
        uint32_t s1 = (uint32_t)(ctr_lo >> 32);
        uint32_t s2 = (uint32_t)ctr_hi;
        uint32_t s3 = (uint32_t)(ctr_hi >> 32);
        const uint32_t *rk = d_roundKeys;
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        uint32_t t0,t1,t2,t3;
#pragma unroll
        for (int r = 1; r <= 13; ++r) {
            t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1>>8)&0xFF] ^ d_T2[(s2>>16)&0xFF] ^ d_T3[(s3>>24)&0xFF] ^ rk[4*r+0];
            t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2>>8)&0xFF] ^ d_T2[(s3>>16)&0xFF] ^ d_T3[(s0>>24)&0xFF] ^ rk[4*r+1];
            t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3>>8)&0xFF] ^ d_T2[(s0>>16)&0xFF] ^ d_T3[(s1>>24)&0xFF] ^ rk[4*r+2];
            t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0>>8)&0xFF] ^ d_T2[(s1>>16)&0xFF] ^ d_T3[(s2>>24)&0xFF] ^ rk[4*r+3];
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        uint8_t keystream[16];
        uint8_t *ko = keystream;
        const uint8_t *sb = d_sbox;
        ko[0]  = sb[s0 & 0xFF];       ko[4]  = sb[(s1>>8)&0xFF];
        ko[8]  = sb[(s2>>16)&0xFF];   ko[12] = sb[(s3>>24)&0xFF];
        ko[1]  = sb[s1 & 0xFF];       ko[5]  = sb[(s2>>8)&0xFF];
        ko[9]  = sb[(s3>>16)&0xFF];   ko[13] = sb[(s0>>24)&0xFF];
        ko[2]  = sb[s2 & 0xFF];       ko[6]  = sb[(s3>>8)&0xFF];
        ko[10] = sb[(s0>>16)&0xFF];   ko[14] = sb[(s1>>24)&0xFF];
        ko[3]  = sb[s3 & 0xFF];       ko[7]  = sb[(s0>>8)&0xFF];
        ko[11] = sb[(s1>>16)&0xFF];   ko[15] = sb[(s2>>24)&0xFF];
        ((uint32_t*)keystream)[0] ^= rk[56];
        ((uint32_t*)keystream)[1] ^= rk[57];
        ((uint32_t*)keystream)[2] ^= rk[58];
        ((uint32_t*)keystream)[3] ^= rk[59];
        const uint8_t *ct_block = cipher + i * 16;
        uint8_t *pt_block = plain + i * 16;
        ((uint32_t*)pt_block)[0] = ((const uint32_t*)ct_block)[0] ^ ((uint32_t*)keystream)[0];
        ((uint32_t*)pt_block)[1] = ((const uint32_t*)ct_block)[1] ^ ((uint32_t*)keystream)[1];
        ((uint32_t*)pt_block)[2] = ((const uint32_t*)ct_block)[2] ^ ((uint32_t*)keystream)[2];
        ((uint32_t*)pt_block)[3] = ((const uint32_t*)ct_block)[3] ^ ((uint32_t*)keystream)[3];
    }
    __syncthreads();

    uint32_t tid = threadIdx.x;
    if (tid < 32) {
        uint64_t step_hi = 0ull, step_lo = 1ull;
        for (int b = 0; b < 32; ++b)
            if (32u & (1u << b))
                gf_mul128(step_hi, step_lo, d_H_pow_hi[b], d_H_pow_lo[b]);

        uint64_t pow_hi = 0ull, pow_lo = 1ull;
        uint32_t exp = (uint32_t)(nBlocks - 1 - tid);
        for (int b = 0; b < 32; ++b)
            if (exp & (1u << b))
                gf_mul128(pow_hi, pow_lo, d_H_pow_hi[b], d_H_pow_lo[b]);

        uint64_t accum_hi = 0ull, accum_lo = 0ull;
        for (size_t j = tid; j < nBlocks; j += 32) {
            uint64_t c_lo = ((const uint64_t*)cipher)[2*j + 0];
            uint64_t c_hi = ((const uint64_t*)cipher)[2*j + 1];
            uint64_t tmp_hi = c_hi, tmp_lo = c_lo;
            gf_mul128(tmp_hi, tmp_lo, pow_hi, pow_lo);
            accum_hi ^= tmp_hi;
            accum_lo ^= tmp_lo;
            gf_mul128(pow_hi, pow_lo, step_hi, step_lo);
        }

        for (int off = 16; off > 0; off >>= 1) {
            accum_hi ^= __shfl_xor_sync(0xFFFFFFFF, accum_hi, off);
            accum_lo ^= __shfl_xor_sync(0xFFFFFFFF, accum_lo, off);
        }
        if (tid == 0) {
            uint64_t lenBlock_lo = (uint64_t)nBlocks * 16ull * 8ull;
            uint64_t lenBlock_hi = 0ull;
            accum_lo ^= lenBlock_lo;
            accum_hi ^= lenBlock_hi;
            gf_mul128(accum_hi, accum_lo, d_H_pow_hi[0], d_H_pow_lo[0]);
            ((uint64_t*)tagOut)[0] = accum_lo;
            ((uint64_t*)tagOut)[1] = accum_hi;
        }
    }
}
