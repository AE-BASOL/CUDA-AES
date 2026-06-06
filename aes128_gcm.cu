#include "aes_common.h"

#include <stdint.h>

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t d_sbox[256];

namespace {

__device__ __forceinline__ uint32_t load_le32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void store_le32(uint8_t *p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}

__device__ __forceinline__ void aes128_encrypt_block(const uint8_t in[16], uint8_t out[16]) {
    const uint32_t *rk = d_roundKeys;
    uint32_t s0 = load_le32(in) ^ rk[0];
    uint32_t s1 = load_le32(in + 4) ^ rk[1];
    uint32_t s2 = load_le32(in + 8) ^ rk[2];
    uint32_t s3 = load_le32(in + 12) ^ rk[3];
    uint32_t t0, t1, t2, t3;

#pragma unroll
    for (int r = 1; r <= 9; ++r) {
        t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1 >> 8) & 0xFF] ^ d_T2[(s2 >> 16) & 0xFF] ^ d_T3[(s3 >> 24) & 0xFF] ^ rk[4 * r + 0];
        t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2 >> 8) & 0xFF] ^ d_T2[(s3 >> 16) & 0xFF] ^ d_T3[(s0 >> 24) & 0xFF] ^ rk[4 * r + 1];
        t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3 >> 8) & 0xFF] ^ d_T2[(s0 >> 16) & 0xFF] ^ d_T3[(s1 >> 24) & 0xFF] ^ rk[4 * r + 2];
        t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0 >> 8) & 0xFF] ^ d_T2[(s1 >> 16) & 0xFF] ^ d_T3[(s2 >> 24) & 0xFF] ^ rk[4 * r + 3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
    }

    uint32_t k0 = static_cast<uint32_t>(d_sbox[s0 & 0xFF]) |
                  (static_cast<uint32_t>(d_sbox[(s1 >> 8) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(d_sbox[(s2 >> 16) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(d_sbox[(s3 >> 24) & 0xFF]) << 24);
    uint32_t k1 = static_cast<uint32_t>(d_sbox[s1 & 0xFF]) |
                  (static_cast<uint32_t>(d_sbox[(s2 >> 8) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(d_sbox[(s3 >> 16) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(d_sbox[(s0 >> 24) & 0xFF]) << 24);
    uint32_t k2 = static_cast<uint32_t>(d_sbox[s2 & 0xFF]) |
                  (static_cast<uint32_t>(d_sbox[(s3 >> 8) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(d_sbox[(s0 >> 16) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(d_sbox[(s1 >> 24) & 0xFF]) << 24);
    uint32_t k3 = static_cast<uint32_t>(d_sbox[s3 & 0xFF]) |
                  (static_cast<uint32_t>(d_sbox[(s0 >> 8) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(d_sbox[(s1 >> 16) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(d_sbox[(s2 >> 24) & 0xFF]) << 24);

    store_le32(out, k0 ^ rk[40]);
    store_le32(out + 4, k1 ^ rk[41]);
    store_le32(out + 8, k2 ^ rk[42]);
    store_le32(out + 12, k3 ^ rk[43]);
}

__device__ __forceinline__ void make_j0(const uint8_t *iv, uint8_t out[16]) {
#pragma unroll
    for (int i = 0; i < 12; ++i) out[i] = iv[i];
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
}

__device__ __forceinline__ void make_counter(const uint8_t *iv, uint32_t counter, uint8_t out[16]) {
#pragma unroll
    for (int i = 0; i < 12; ++i) out[i] = iv[i];
    out[12] = static_cast<uint8_t>(counter >> 24);
    out[13] = static_cast<uint8_t>(counter >> 16);
    out[14] = static_cast<uint8_t>(counter >> 8);
    out[15] = static_cast<uint8_t>(counter);
}

__device__ void shift_right_one(uint8_t v[16]) {
    uint8_t carry = 0;
    for (int i = 0; i < 16; ++i) {
        const uint8_t next_carry = static_cast<uint8_t>(v[i] & 1u);
        v[i] = static_cast<uint8_t>((v[i] >> 1) | (carry << 7));
        carry = next_carry;
    }
}

__device__ void gf_mul_gcm(const uint8_t x[16], const uint8_t y[16], uint8_t out[16]) {
    uint8_t z[16] = {};
    uint8_t v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = y[i];

    for (int byte = 0; byte < 16; ++byte) {
        for (int bit = 7; bit >= 0; --bit) {
            if ((x[byte] >> bit) & 1u) {
#pragma unroll
                for (int i = 0; i < 16; ++i) z[i] ^= v[i];
            }
            const bool lsb = (v[15] & 1u) != 0;
            shift_right_one(v);
            if (lsb) v[0] ^= 0xE1u;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = z[i];
}

__device__ void ghash_update(uint8_t y[16], const uint8_t block[16], const uint8_t h[16]) {
    uint8_t x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = y[i] ^ block[i];
    gf_mul_gcm(x, h, y);
}

// Phase 2 supports the standard 96-bit-IV, empty-AAD GCM shape:
// tag = GHASH_H(AAD="", C, length_block) XOR E(K, J0).
__device__ void compute_gcm_tag(const uint8_t *cipher, size_t nBlocks, const uint8_t *iv, uint8_t *tagOut) {
    uint8_t zero[16] = {};
    uint8_t h[16];
    aes128_encrypt_block(zero, h);

    uint8_t y[16] = {};
    for (size_t block = 0; block < nBlocks; ++block) {
        uint8_t c[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) c[i] = cipher[block * 16 + i];
        ghash_update(y, c, h);
    }

    uint8_t length_block[16] = {};
    const uint64_t ciphertext_bits = static_cast<uint64_t>(nBlocks) * 128ull;
    for (int i = 0; i < 8; ++i) {
        length_block[15 - i] = static_cast<uint8_t>(ciphertext_bits >> (8 * i));
    }
    ghash_update(y, length_block, h);

    uint8_t j0[16];
    uint8_t encrypted_j0[16];
    make_j0(iv, j0);
    aes128_encrypt_block(j0, encrypted_j0);
#pragma unroll
    for (int i = 0; i < 16; ++i) tagOut[i] = y[i] ^ encrypted_j0[i];
}

}  // namespace

__global__ void aes128_gcm_encrypt(const uint8_t *plain, uint8_t *cipher, size_t nBlocks, const uint8_t *iv, uint8_t *tagOut) {
    __shared__ uint8_t sh_iv[12];
    if (threadIdx.x < 12) sh_iv[threadIdx.x] = iv[threadIdx.x];
    __syncthreads();

    for (size_t block = threadIdx.x; block < nBlocks; block += blockDim.x) {
        uint8_t counter[16];
        uint8_t stream[16];
        make_counter(sh_iv, static_cast<uint32_t>(block) + 2u, counter);
        aes128_encrypt_block(counter, stream);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            cipher[block * 16 + i] = plain[block * 16 + i] ^ stream[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        compute_gcm_tag(cipher, nBlocks, sh_iv, tagOut);
    }
}

__global__ void aes128_gcm_decrypt(const uint8_t *cipher, uint8_t *plain, size_t nBlocks, const uint8_t *iv, const uint8_t *tag, uint8_t *tagOut) {
    (void)tag;
    __shared__ uint8_t sh_iv[12];
    if (threadIdx.x < 12) sh_iv[threadIdx.x] = iv[threadIdx.x];
    __syncthreads();

    for (size_t block = threadIdx.x; block < nBlocks; block += blockDim.x) {
        uint8_t counter[16];
        uint8_t stream[16];
        make_counter(sh_iv, static_cast<uint32_t>(block) + 2u, counter);
        aes128_encrypt_block(counter, stream);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            plain[block * 16 + i] = cipher[block * 16 + i] ^ stream[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        compute_gcm_tag(cipher, nBlocks, sh_iv, tagOut);
    }
}
