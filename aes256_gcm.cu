#include "aes_common.h"
#include <stdint.h>

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];

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

__global__ void aes256_gcm_encrypt(const uint8_t *plain, uint8_t *cipher, size_t nBlocks, const uint8_t *iv, uint8_t *tagOut) {
    // Implementation is analogous to aes128_gcm_encrypt, but using AES-256 (14 rounds).
    __shared__ uint64_t sh_H_hi, sh_H_lo;
    __shared__ uint64_t partial_tag_hi[32];
    __shared__ uint64_t partial_tag_lo[32];
    __shared__ uint32_t partial_len[32];

    if (threadIdx.x == 0) {
        // Compute H = AES-256 encrypt of all-zero block
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
        buf[8]  = sb[(s2>>16)&0xFF]; buf[12] = sb[(s3>>24)&0xFF];
        buf[1]  = sb[s1 & 0xFF];      buf[5]  = sb[(s2>>8)&0xFF];
        buf[9]  = sb[(s3>>16)&0xFF]; buf[13] = sb[(s0>>24)&0xFF];
        buf[2]  = sb[s2 & 0xFF];      buf[6]  = sb[(s3>>8)&0xFF];
        buf[10] = sb[(s0>>16)&0xFF]; buf[14] = sb[(s1>>24)&0xFF];
        buf[3]  = sb[s3 & 0xFF];      buf[7]  = sb[(s0>>8)&0xFF];
        buf[11] = sb[(s1>>16)&0xFF]; buf[15] = sb[(s2>>24)&0xFF];
        ((uint32_t*)buf)[0] ^= rk[56];
        ((uint32_t*)buf)[1] ^= rk[57];
        ((uint32_t*)buf)[2] ^= rk[58];
        ((uint32_t*)buf)[3] ^= rk[59];
        sh_H_lo = ((uint64_t*)buf)[0];
        sh_H_hi = ((uint64_t*)buf)[1];
    }
    __syncthreads();

    // Setup IV counters
    uint64_t IV_lo = 0ull, IV_hi = 0ull;
    if (threadIdx.x == 0) {
        IV_hi = ((const uint64_t*)iv)[0];
        uint32_t iv_low32 = 0;
        memcpy(&iv_low32, iv + 8, 4);
        IV_lo = ((uint64_t)iv_low32 << 32) | 1ULL;
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

    // GHASH (same pattern as AES-128 GCM, with same polynomial multiply)
    uint32_t tid = threadIdx.x;
    if (tid < 32) {
        size_t seg_size = (nBlocks + 31) / 32;
        size_t start = seg_size * tid;
        size_t end = (start + seg_size < nBlocks) ? (start + seg_size) : nBlocks;
        uint64_t Xh = 0ull, Xl = 0ull;
        for (size_t j = start; j < end; ++j) {
            uint64_t c_l = ((uint64_t*)cipher)[2*j + 0];
            uint64_t c_h = ((uint64_t*)cipher)[2*j + 1];
            Xl ^= c_l;
            Xh ^= c_h;
            gf_mul128(Xh, Xl, sh_H_hi, sh_H_lo);
        }
        partial_tag_hi[tid] = Xh;
        partial_tag_lo[tid] = Xl;
        partial_len[tid] = (uint32_t)(end - start);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t tag_h = partial_tag_hi[0];
        uint64_t tag_l = partial_tag_lo[0];
        uint64_t pow_hi[27], pow_lo[27];
        pow_hi[0] = sh_H_hi;
        pow_lo[0] = sh_H_lo;
        for (int i = 1; i < 27; ++i) {
            pow_hi[i] = pow_hi[i-1];
            pow_lo[i] = pow_lo[i-1];
            gf_mul128(pow_hi[i], pow_lo[i], pow_hi[i-1], pow_lo[i-1]);
        }
        size_t done_blocks = partial_len[0];
        for (int seg = 1; seg < 32; ++seg) {
            uint64_t exp_hi = 0ull, exp_lo = 0ull;
            exp_lo = 1ull; exp_hi = 0ull;
            uint32_t count = partial_len[seg];
            for (int b = 0; b < 27; ++b) {
                if (count & (1u << b)) {
                    gf_mul128(exp_hi, exp_lo, pow_hi[b], pow_lo[b]);
                }
            }
            gf_mul128(tag_h, tag_l, exp_hi, exp_lo);
            tag_h ^= partial_tag_hi[seg];
            tag_l ^= partial_tag_lo[seg];
            done_blocks += partial_len[seg];
        }
        ((uint64_t*)tagOut)[0] = tag_l;
        ((uint64_t*)tagOut)[1] = tag_h;
    }
}

__global__ void aes256_gcm_decrypt(const uint8_t *cipher, uint8_t *plain, size_t nBlocks, const uint8_t *iv, const uint8_t *tag, uint8_t *tagOut) {
    // Mirror aes256_gcm_encrypt but with ciphertext as input and plaintext as
    // output.  The authentication tag for the ciphertext is produced in
    // tagOut; the provided 'tag' parameter is expected to be checked by the
    // host.

    __shared__ uint64_t sh_H_hi, sh_H_lo;
    __shared__ uint64_t partial_tag_hi[32];
    __shared__ uint64_t partial_tag_lo[32];
    __shared__ uint32_t partial_len[32];

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
        buf[8]  = sb[(s2>>16)&0xFF];  buf[12] = sb[(s3>>24)&0xFF];
        buf[1]  = sb[s1 & 0xFF];      buf[5]  = sb[(s2>>8)&0xFF];
        buf[9]  = sb[(s3>>16)&0xFF];  buf[13] = sb[(s0>>24)&0xFF];
        buf[2]  = sb[s2 & 0xFF];      buf[6]  = sb[(s3>>8)&0xFF];
        buf[10] = sb[(s0>>16)&0xFF];  buf[14] = sb[(s1>>24)&0xFF];
        buf[3]  = sb[s3 & 0xFF];      buf[7]  = sb[(s0>>8)&0xFF];
        buf[11] = sb[(s1>>16)&0xFF];  buf[15] = sb[(s2>>24)&0xFF];
        ((uint32_t*)buf)[0] ^= rk[56];
        ((uint32_t*)buf)[1] ^= rk[57];
        ((uint32_t*)buf)[2] ^= rk[58];
        ((uint32_t*)buf)[3] ^= rk[59];
        sh_H_lo = ((uint64_t*)buf)[0];
        sh_H_hi = ((uint64_t*)buf)[1];
    }
    __syncthreads();

    uint64_t IV_lo = 0ull, IV_hi = 0ull;
    if (threadIdx.x == 0) {
        IV_hi = ((const uint64_t*)iv)[0];
        uint32_t iv_low32 = 0;
        memcpy(&iv_low32, iv + 8, 4);
        IV_lo = ((uint64_t)iv_low32 << 32) | 1ULL;
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
        size_t seg_size = (nBlocks + 31) / 32;
        size_t start = seg_size * tid;
        size_t end = (start + seg_size < nBlocks) ? (start + seg_size) : nBlocks;
        uint64_t Xh = 0ull, Xl = 0ull;
        for (size_t j = start; j < end; ++j) {
            uint64_t c_l = ((uint64_t*)cipher)[2*j + 0];
            uint64_t c_h = ((uint64_t*)cipher)[2*j + 1];
            Xl ^= c_l;
            Xh ^= c_h;
            gf_mul128(Xh, Xl, sh_H_hi, sh_H_lo);
        }
        partial_tag_hi[tid] = Xh;
        partial_tag_lo[tid] = Xl;
        partial_len[tid] = (uint32_t)(end - start);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t tag_h = partial_tag_hi[0];
        uint64_t tag_l = partial_tag_lo[0];
        uint64_t pow_hi[27], pow_lo[27];
        pow_hi[0] = sh_H_hi;
        pow_lo[0] = sh_H_lo;
        for (int i = 1; i < 27; ++i) {
            pow_hi[i] = pow_hi[i-1];
            pow_lo[i] = pow_lo[i-1];
            gf_mul128(pow_hi[i], pow_lo[i], pow_hi[i-1], pow_lo[i-1]);
        }
        size_t done_blocks = partial_len[0];
        for (int seg = 1; seg < 32; ++seg) {
            uint64_t exp_hi = 0ull, exp_lo = 0ull;
            exp_lo = 1ull; exp_hi = 0ull;
            uint32_t count = partial_len[seg];
            for (int b = 0; b < 27; ++b) {
                if (count & (1u << b)) {
                    gf_mul128(exp_hi, exp_lo, pow_hi[b], pow_lo[b]);
                }
            }
            gf_mul128(tag_h, tag_l, exp_hi, exp_lo);
            tag_h ^= partial_tag_hi[seg];
            tag_l ^= partial_tag_lo[seg];
            done_blocks += partial_len[seg];
        }
        ((uint64_t*)tagOut)[0] = tag_l;
        ((uint64_t*)tagOut)[1] = tag_h;
    }
}
