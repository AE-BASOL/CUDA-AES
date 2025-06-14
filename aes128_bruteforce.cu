#include "aes_common.h"
#include <cuda_runtime.h>

// Constant Rcon values for AES-128 key expansion (LSB layout)
__device__ __constant__ uint32_t d_rcon[10] = {
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
};

union Block { uint32_t w[4]; uint8_t b[16]; };

__device__ __forceinline__ void key_expand(uint32_t &rk0, uint32_t &rk1,
                                            uint32_t &rk2, uint32_t &rk3,
                                            int round)
{
    uint32_t temp = rk3;
    temp = (temp << 8) | (temp >> 24);
    temp = ((uint32_t)d_sbox[temp & 0xFF]) |
           ((uint32_t)d_sbox[(temp >> 8) & 0xFF] << 8) |
           ((uint32_t)d_sbox[(temp >> 16) & 0xFF] << 16) |
           ((uint32_t)d_sbox[(temp >> 24) & 0xFF] << 24);
    temp ^= d_rcon[round];
    rk0 ^= temp;
    rk1 ^= rk0;
    rk2 ^= rk1;
    rk3 ^= rk2;
}

__device__ __forceinline__ void encrypt_round(uint32_t &s0, uint32_t &s1,
                                              uint32_t &s2, uint32_t &s3,
                                              const uint32_t *rk)
{
    uint32_t t0 = d_T0[s0 & 0xFF] ^ d_T1[(s1 >> 8) & 0xFF] ^
                  d_T2[(s2 >> 16) & 0xFF] ^ d_T3[(s3 >> 24) & 0xFF] ^ rk[0];
    uint32_t t1 = d_T0[s1 & 0xFF] ^ d_T1[(s2 >> 8) & 0xFF] ^
                  d_T2[(s3 >> 16) & 0xFF] ^ d_T3[(s0 >> 24) & 0xFF] ^ rk[1];
    uint32_t t2 = d_T0[s2 & 0xFF] ^ d_T1[(s3 >> 8) & 0xFF] ^
                  d_T2[(s0 >> 16) & 0xFF] ^ d_T3[(s1 >> 24) & 0xFF] ^ rk[2];
    uint32_t t3 = d_T0[s3 & 0xFF] ^ d_T1[(s0 >> 8) & 0xFF] ^
                  d_T2[(s1 >> 16) & 0xFF] ^ d_T3[(s2 >> 24) & 0xFF] ^ rk[3];
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
}

__device__ __forceinline__ void encrypt_last(uint32_t &s0, uint32_t &s1,
                                             uint32_t &s2, uint32_t &s3,
                                             const uint32_t *rk)
{
    const uint8_t *sb = d_sbox;
    uint32_t t0 = ((uint32_t)sb[s0 & 0xFF]) |
                  ((uint32_t)sb[s1 & 0xFF] << 8) |
                  ((uint32_t)sb[s2 & 0xFF] << 16) |
                  ((uint32_t)sb[s3 & 0xFF] << 24);
    uint32_t t1 = ((uint32_t)sb[(s1 >> 8) & 0xFF]) |
                  ((uint32_t)sb[(s2 >> 8) & 0xFF] << 8) |
                  ((uint32_t)sb[(s3 >> 8) & 0xFF] << 16) |
                  ((uint32_t)sb[(s0 >> 8) & 0xFF] << 24);
    uint32_t t2 = ((uint32_t)sb[(s2 >> 16) & 0xFF]) |
                  ((uint32_t)sb[(s3 >> 16) & 0xFF] << 8) |
                  ((uint32_t)sb[(s0 >> 16) & 0xFF] << 16) |
                  ((uint32_t)sb[(s1 >> 16) & 0xFF] << 24);
    uint32_t t3 = ((uint32_t)sb[(s3 >> 24) & 0xFF]) |
                  ((uint32_t)sb[(s0 >> 24) & 0xFF] << 8) |
                  ((uint32_t)sb[(s1 >> 24) & 0xFF] << 16) |
                  ((uint32_t)sb[(s2 >> 24) & 0xFF] << 24);
    s0 = t0 ^ rk[0];
    s1 = t1 ^ rk[1];
    s2 = t2 ^ rk[2];
    s3 = t3 ^ rk[3];
}

__device__ bool aes128_match(Block pt, Block ct,
                             uint32_t rk0, uint32_t rk1,
                             uint32_t rk2, uint32_t rk3)
{
    uint32_t s0 = pt.w[0] ^ rk0;
    uint32_t s1 = pt.w[1] ^ rk1;
    uint32_t s2 = pt.w[2] ^ rk2;
    uint32_t s3 = pt.w[3] ^ rk3;

#pragma unroll
    for (int round = 0; round < 9; ++round) {
        key_expand(rk0, rk1, rk2, rk3, round);
        uint32_t rk_round[4] = {rk0, rk1, rk2, rk3};
        encrypt_round(s0, s1, s2, s3, rk_round);
    }
    key_expand(rk0, rk1, rk2, rk3, 9);
    uint32_t rk_last[4] = {rk0, rk1, rk2, rk3};
    encrypt_last(s0, s1, s2, s3, rk_last);

    return (s0 == ct.w[0] && s1 == ct.w[1] &&
            s2 == ct.w[2] && s3 == ct.w[3]);
}

__global__ void aes128_bruteforce(const uint8_t *pt, const uint8_t *ct,
                                  const uint32_t *keySeed, uint64_t range,
                                  uint32_t *foundKey, int *found)
{
    Block P, C;
    const uint32_t *p32 = reinterpret_cast<const uint32_t*>(pt);
    P.w[0]=p32[0]; P.w[1]=p32[1]; P.w[2]=p32[2]; P.w[3]=p32[3];
    const uint32_t *c32 = reinterpret_cast<const uint32_t*>(ct);
    C.w[0]=c32[0]; C.w[1]=c32[1]; C.w[2]=c32[2]; C.w[3]=c32[3];

    uint32_t rk0=keySeed[0];
    uint32_t rk1=keySeed[1];
    uint32_t rk2=keySeed[2];
    uint32_t rk3=keySeed[3];

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = idx * range;
    rk2 += (uint32_t)(start >> 32);
    rk3 += (uint32_t)start;

    for (uint64_t i=0; i<range && !(*found); ++i) {
        if (aes128_match(P, C, rk0, rk1, rk2, rk3)) {
            if (atomicCAS(found, 0, 1) == 0) {
                foundKey[0]=rk0; foundKey[1]=rk1;
                foundKey[2]=rk2; foundKey[3]=rk3;
            }
            return;
        }
        rk3 += 1;
        if (rk3 == 0) rk2 += 1;
    }
}

