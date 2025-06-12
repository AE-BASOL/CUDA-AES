#include "aes_common.h"
#include <stdint.h>

// Extern constants
extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint64_t d_H_pow_hi[32];
extern __device__ __constant__ uint64_t d_H_pow_lo[32];

// Multiply two 128-bit values (A * B) in GF(2^128) with the GCM reduction polynomial.
// Inputs/outputs are in 64-bit high/low parts.
static __device__ inline void gf_mul128(uint64_t &Ah, uint64_t &Al, uint64_t Bh, uint64_t Bl) {
    // Implements bitwise multiplication with reduction: O(128) steps
    uint64_t Zh = 0ull, Zl = 0ull;
    uint64_t Vh = Bh, Vl = Bl;
    // GF(2^128) irreducible polynomial: x^128 + x^7 + x^2 + x + 1 (0xE100000000000000 as 128-bit)
    const uint64_t R = 0xE100000000000000ULL;  // Represents polynomial (1 followed by 0xE1)
    for (int i = 0; i < 128; ++i) {
        if (Al & 1ULL) {  // if LSB of A is 1, add V to Z
            Zl ^= Vl;
            Zh ^= Vh;
        }
        // Carry for reduction = LSB of V
        bool carry = (Vl & 1ULL);
        // Shift V right by 1
        Vl = (Vl >> 1) | (Vh << 63);
        Vh = (Vh >> 1);
        if (carry) {
            Vh ^= R;  // XOR high part with reduction constant if carry out
        }
        // Shift A right by 1 (to process next bit)
        bool a_carry = (Al & 1ULL);
        Al = (Al >> 1) | (Ah << 63);
        Ah = (Ah >> 1);
        // (We don't actually use a_carry here because we already used A's LSB for if)
    }
    Ah = Zh;
    Al = Zl;
}

// AES-128-GCM encryption kernel
__global__ void aes128_gcm_encrypt(const uint8_t * __restrict__ plain, uint8_t * __restrict__ cipher, size_t nBlocks, const uint8_t * __restrict__ iv, uint8_t * __restrict__ tagOut) {
    // We will use one thread block to process the entire message for tag calculation correctness (no inter-block sync on device).
    // Use 256 threads: parallelize CTR encryption, then do GHASH in a single warp for tag.
    // nBlocks is number of 16-byte blocks of plaintext.
    // GHASH uses precomputed H powers in constant memory; no setup needed

    // CTR mode encryption of plaintext
    // Prepare IV counter (96-bit IV + 32-bit initial counter=1)
    uint64_t IV_lo = 0ull;
    uint64_t IV_hi = 0ull;
    if (threadIdx.x == 0) {
        uint32_t w0 = 0, w1 = 0, w2 = 0;
        memcpy(&w0, iv, 4);
        memcpy(&w1, iv + 4, 4);
        memcpy(&w2, iv + 8, 4);
        uint32_t w3 = 0x01000000u; // counter = 1 in big-endian
        IV_lo = (uint64_t)w0 | ((uint64_t)w1 << 32);
        IV_hi = (uint64_t)w2 | ((uint64_t)w3 << 32);
    }
    __syncthreads();
    // Broadcast IV values to all threads
    IV_lo = __shfl_sync(0xFFFFFFFF, IV_lo, 0);
    IV_hi = __shfl_sync(0xFFFFFFFF, IV_hi, 0);

    // Parallel encryption: each thread handles multiple blocks in strides
    for (size_t i = threadIdx.x; i < nBlocks; i += blockDim.x) {
        // Compute counter for block i
        uint64_t ctr_lo = IV_lo + i;
        uint64_t ctr_hi = IV_hi;
        if (ctr_lo < IV_lo) ctr_hi += 1ULL;
        // Encrypt counter (like in CTR encryption)
        uint32_t s0 = (uint32_t) ctr_lo;
        uint32_t s1 = (uint32_t)(ctr_lo >> 32);
        uint32_t s2 = (uint32_t) ctr_hi;
        uint32_t s3 = (uint32_t)(ctr_hi >> 32);
        const uint32_t *rk = d_roundKeys;
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        uint32_t t0,t1,t2,t3;
        #pragma unroll
        for (int r = 1; r <= 9; ++r) {
            t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1>>8)&0xFF] ^ d_T2[(s2>>16)&0xFF] ^ d_T3[(s3>>24)&0xFF] ^ rk[4*r+0];
            t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2>>8)&0xFF] ^ d_T2[(s3>>16)&0xFF] ^ d_T3[(s0>>24)&0xFF] ^ rk[4*r+1];
            t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3>>8)&0xFF] ^ d_T2[(s0>>16)&0xFF] ^ d_T3[(s1>>24)&0xFF] ^ rk[4*r+2];
            t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0>>8)&0xFF] ^ d_T2[(s1>>16)&0xFF] ^ d_T3[(s2>>24)&0xFF] ^ rk[4*r+3];
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        // Final round
        uint8_t keystream[16];
        uint8_t *ko = keystream;
        const uint8_t *sb = d_sbox;
        ko[0]  = sb[s0 & 0xFF];        ko[4]  = sb[(s1>>8)&0xFF];
        ko[8]  = sb[(s2>>16)&0xFF];    ko[12] = sb[(s3>>24)&0xFF];
        ko[1]  = sb[s1 & 0xFF];        ko[5]  = sb[(s2>>8)&0xFF];
        ko[9]  = sb[(s3>>16)&0xFF];    ko[13] = sb[(s0>>24)&0xFF];
        ko[2]  = sb[s2 & 0xFF];        ko[6]  = sb[(s3>>8)&0xFF];
        ko[10] = sb[(s0>>16)&0xFF];    ko[14] = sb[(s1>>24)&0xFF];
        ko[3]  = sb[s3 & 0xFF];        ko[7]  = sb[(s0>>8)&0xFF];
        ko[11] = sb[(s1>>16)&0xFF];    ko[15] = sb[(s2>>24)&0xFF];
        ((uint32_t*)keystream)[0] ^= rk[40];
        ((uint32_t*)keystream)[1] ^= rk[41];
        ((uint32_t*)keystream)[2] ^= rk[42];
        ((uint32_t*)keystream)[3] ^= rk[43];
        // XOR plaintext with keystream to get ciphertext
        const uint8_t *pt_block = plain + i * 16;
        uint8_t *ct_block = cipher + i * 16;
        ((uint32_t*)ct_block)[0] = ((const uint32_t*)pt_block)[0] ^ ((uint32_t*)keystream)[0];
        ((uint32_t*)ct_block)[1] = ((const uint32_t*)pt_block)[1] ^ ((uint32_t*)keystream)[1];
        ((uint32_t*)ct_block)[2] = ((const uint32_t*)pt_block)[2] ^ ((uint32_t*)keystream)[2];
        ((uint32_t*)ct_block)[3] = ((const uint32_t*)pt_block)[3] ^ ((uint32_t*)keystream)[3];
    }
    __syncthreads();

    // GHASH computation using warp-level XOR reduction
    uint32_t tid = threadIdx.x;
    if (tid < 32) {
        // Precompute H^32 for stepping between iterations
        uint64_t step_hi = 0ull, step_lo = 1ull;
        for (int b = 0; b < 32; ++b)
            if (32u & (1u << b))
                gf_mul128(step_hi, step_lo, d_H_pow_hi[b], d_H_pow_lo[b]);

        // Starting power for this thread corresponds to H^(nBlocks-1-tid)
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
            gf_mul128(pow_hi, pow_lo, step_hi, step_lo);  // advance power by 32
        }

        for (int off = 16; off > 0; off >>= 1) {
            accum_hi ^= __shfl_xor_sync(0xFFFFFFFF, accum_hi, off);
            accum_lo ^= __shfl_xor_sync(0xFFFFFFFF, accum_lo, off);
        }
        if (tid == 0) {
            uint64_t lenBlock_lo = (uint64_t)nBlocks * 16ull * 8ull;
            uint64_t lenBlock_hi = 0ull; // no AAD
            accum_lo ^= lenBlock_lo;
            accum_hi ^= lenBlock_hi;
            gf_mul128(accum_hi, accum_lo, d_H_pow_hi[0], d_H_pow_lo[0]);
            ((uint64_t*)tagOut)[0] = accum_lo;
            ((uint64_t*)tagOut)[1] = accum_hi;
        }
    }
}

__global__ void aes128_gcm_decrypt(const uint8_t * __restrict__ cipher, uint8_t * __restrict__ plain, size_t nBlocks, const uint8_t * __restrict__ iv, const uint8_t * __restrict__ tag, uint8_t * __restrict__ tagOut) {
    // Implement decryption directly rather than launching a nested kernel.
    // The operations mirror aes128_gcm_encrypt with plaintext and ciphertext
    // swapped.  The 'tag' parameter is provided for host-side verification and
    // is unused here.

    // GHASH uses precomputed powers of H

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
        uint32_t s0 = (uint32_t) ctr_lo;
        uint32_t s1 = (uint32_t)(ctr_lo >> 32);
        uint32_t s2 = (uint32_t) ctr_hi;
        uint32_t s3 = (uint32_t)(ctr_hi >> 32);
        const uint32_t *rk = d_roundKeys;
        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        uint32_t t0,t1,t2,t3;
#pragma unroll
        for (int r = 1; r <= 9; ++r) {
            t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1>>8)&0xFF] ^ d_T2[(s2>>16)&0xFF] ^ d_T3[(s3>>24)&0xFF] ^ rk[4*r+0];
            t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2>>8)&0xFF] ^ d_T2[(s3>>16)&0xFF] ^ d_T3[(s0>>24)&0xFF] ^ rk[4*r+1];
            t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3>>8)&0xFF] ^ d_T2[(s0>>16)&0xFF] ^ d_T3[(s1>>24)&0xFF] ^ rk[4*r+2];
            t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0>>8)&0xFF] ^ d_T2[(s1>>16)&0xFF] ^ d_T3[(s2>>24)&0xFF] ^ rk[4*r+3];
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        uint8_t keystream[16];
        uint8_t *ko = keystream;
        const uint8_t *sb = d_sbox;
        ko[0]  = sb[s0 & 0xFF];        ko[4]  = sb[(s1>>8)&0xFF];
        ko[8]  = sb[(s2>>16)&0xFF];    ko[12] = sb[(s3>>24)&0xFF];
        ko[1]  = sb[s1 & 0xFF];        ko[5]  = sb[(s2>>8)&0xFF];
        ko[9]  = sb[(s3>>16)&0xFF];    ko[13] = sb[(s0>>24)&0xFF];
        ko[2]  = sb[s2 & 0xFF];        ko[6]  = sb[(s3>>8)&0xFF];
        ko[10] = sb[(s0>>16)&0xFF];    ko[14] = sb[(s1>>24)&0xFF];
        ko[3]  = sb[s3 & 0xFF];        ko[7]  = sb[(s0>>8)&0xFF];
        ko[11] = sb[(s1>>16)&0xFF];    ko[15] = sb[(s2>>24)&0xFF];
        ((uint32_t*)keystream)[0] ^= rk[40];
        ((uint32_t*)keystream)[1] ^= rk[41];
        ((uint32_t*)keystream)[2] ^= rk[42];
        ((uint32_t*)keystream)[3] ^= rk[43];
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
