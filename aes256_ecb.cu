// Optimised AES-256 ECB kernels using only registers.
// Each thread handles two consecutive 16-byte blocks.
// Compatible with main.cu and aes_common.h

#include <cuda_runtime.h>
#include <stdint.h>
#include "aes_common.h"

extern __constant__ uint32_t d_roundKeys[];     // 60 words
extern __constant__ uint8_t  d_sbox[];
extern __constant__ uint8_t  d_inv_sbox[];

union Block {
    uint32_t w[4];
    uint8_t  b[16];
};

// ─────────────────────────── GF(2^8) helpers ────────────────────────────────
__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x & 0x80u) ? 0x1bu : 0));
}
__device__ __forceinline__ uint8_t gmul2(uint8_t x) { return xtime(x); }
__device__ __forceinline__ uint8_t gmul3(uint8_t x) { return static_cast<uint8_t>(xtime(x) ^ x); }
__device__ __forceinline__ uint8_t gmul9(uint8_t x)  { uint8_t x2=xtime(x),x4=xtime(x2),x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x); }
__device__ __forceinline__ uint8_t gmul11(uint8_t x) { uint8_t x2=xtime(x),x4=xtime(x2),x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x2 ^ x); }
__device__ __forceinline__ uint8_t gmul13(uint8_t x) { uint8_t x2=xtime(x),x4=xtime(x2),x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x4 ^ x); }
__device__ __forceinline__ uint8_t gmul14(uint8_t x){ uint8_t x2=xtime(x),x4=xtime(x2),x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x4 ^ x2); }

// ─────────────────────────── Round helpers ──────────────────────────────────
__device__ __forceinline__ void addRoundKey(Block &st, const uint32_t *rk) {
    st.w[0] ^= rk[0];
    st.w[1] ^= rk[1];
    st.w[2] ^= rk[2];
    st.w[3] ^= rk[3];
}

__device__ __forceinline__ void subBytesShiftRows(Block &st) {
    uint8_t tmp[16];
    const uint8_t *sb = d_sbox;
    tmp[0]  = sb[st.b[0]];  tmp[1]  = sb[st.b[5]];  tmp[2]  = sb[st.b[10]]; tmp[3]  = sb[st.b[15]];
    tmp[4]  = sb[st.b[4]];  tmp[5]  = sb[st.b[9]];  tmp[6]  = sb[st.b[14]]; tmp[7]  = sb[st.b[3]];
    tmp[8]  = sb[st.b[8]];  tmp[9]  = sb[st.b[13]]; tmp[10] = sb[st.b[2]];  tmp[11] = sb[st.b[7]];
    tmp[12] = sb[st.b[12]]; tmp[13] = sb[st.b[1]];  tmp[14] = sb[st.b[6]];  tmp[15] = sb[st.b[11]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st.b[i] = tmp[i];
}

__device__ __forceinline__ void invShiftRowsSubBytes(Block &st) {
    uint8_t tmp[16];
    const uint8_t *sb = d_inv_sbox;
    tmp[0]  = sb[st.b[0]];  tmp[1]  = sb[st.b[13]]; tmp[2]  = sb[st.b[10]]; tmp[3]  = sb[st.b[7]];
    tmp[4]  = sb[st.b[4]];  tmp[5]  = sb[st.b[1]];  tmp[6]  = sb[st.b[14]]; tmp[7]  = sb[st.b[11]];
    tmp[8]  = sb[st.b[8]];  tmp[9]  = sb[st.b[5]];  tmp[10] = sb[st.b[2]];  tmp[11] = sb[st.b[15]];
    tmp[12] = sb[st.b[12]]; tmp[13] = sb[st.b[9]];  tmp[14] = sb[st.b[6]];  tmp[15] = sb[st.b[3]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st.b[i] = tmp[i];
}

__device__ __forceinline__ void mixColumns(Block &st) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = st.b[c];
        uint8_t a1 = st.b[4 + c];
        uint8_t a2 = st.b[8 + c];
        uint8_t a3 = st.b[12 + c];
        st.b[c]        = gmul2(a0) ^ gmul3(a1) ^ a2 ^ a3;
        st.b[4 + c]    = a0 ^ gmul2(a1) ^ gmul3(a2) ^ a3;
        st.b[8 + c]    = a0 ^ a1 ^ gmul2(a2) ^ gmul3(a3);
        st.b[12 + c]   = gmul3(a0) ^ a1 ^ a2 ^ gmul2(a3);
    }
}

__device__ __forceinline__ void invMixColumns(Block &st) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = st.b[c];
        uint8_t a1 = st.b[4 + c];
        uint8_t a2 = st.b[8 + c];
        uint8_t a3 = st.b[12 + c];
        st.b[c]        = gmul14(a0) ^ gmul11(a1) ^ gmul13(a2) ^ gmul9(a3);
        st.b[4 + c]    = gmul9(a0) ^ gmul14(a1) ^ gmul11(a2) ^ gmul13(a3);
        st.b[8 + c]    = gmul13(a0) ^ gmul9(a1) ^ gmul14(a2) ^ gmul11(a3);
        st.b[12 + c]   = gmul11(a0) ^ gmul13(a1) ^ gmul9(a2) ^ gmul14(a3);
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// AES-256 ECB encryption kernel
__global__ void aes256_ecb_encrypt(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const uint4* in4  = reinterpret_cast<const uint4*>(in);
    uint4* out4       = reinterpret_cast<uint4*>(out);
    const uint32_t* rk = d_roundKeys;

    for (size_t blk = tid * 2; blk < nBlocks; blk += stride * 2) {
        Block s0, s1; bool second = (blk + 1 < nBlocks);
        uint4 v0 = in4[blk];
        s0.w[0]=v0.x; s0.w[1]=v0.y; s0.w[2]=v0.z; s0.w[3]=v0.w;
        if (second) {
            uint4 v1 = in4[blk + 1];
            s1.w[0]=v1.x; s1.w[1]=v1.y; s1.w[2]=v1.z; s1.w[3]=v1.w;
        }

        addRoundKey(s0, rk);
        if (second) addRoundKey(s1, rk);

#pragma unroll
        for (int round = 1; round <= 13; ++round) {
            subBytesShiftRows(s0);  mixColumns(s0);  addRoundKey(s0, rk + round * 4);
            if (second) { subBytesShiftRows(s1); mixColumns(s1); addRoundKey(s1, rk + round * 4); }
        }
        subBytesShiftRows(s0);  addRoundKey(s0, rk + 56);
        out4[blk] = make_uint4(s0.w[0], s0.w[1], s0.w[2], s0.w[3]);
        if (second) {
            subBytesShiftRows(s1); addRoundKey(s1, rk + 56);
            out4[blk + 1] = make_uint4(s1.w[0], s1.w[1], s1.w[2], s1.w[3]);
        }
    }
}

// AES-256 ECB decryption kernel
__global__ void aes256_ecb_decrypt(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const uint4* in4  = reinterpret_cast<const uint4*>(in);
    uint4* out4       = reinterpret_cast<uint4*>(out);
    const uint32_t* rk = d_roundKeys;

    for (size_t blk = tid * 2; blk < nBlocks; blk += stride * 2) {
        Block s0, s1; bool second = (blk + 1 < nBlocks);
        uint4 v0 = in4[blk];
        s0.w[0]=v0.x; s0.w[1]=v0.y; s0.w[2]=v0.z; s0.w[3]=v0.w;
        if (second) {
            uint4 v1 = in4[blk + 1];
            s1.w[0]=v1.x; s1.w[1]=v1.y; s1.w[2]=v1.z; s1.w[3]=v1.w;
        }

        addRoundKey(s0, rk + 56);
        if (second) addRoundKey(s1, rk + 56);

#pragma unroll
        for (int round = 13; round >= 1; --round) {
            invShiftRowsSubBytes(s0);  addRoundKey(s0, rk + round * 4);  invMixColumns(s0);
            if (second) { invShiftRowsSubBytes(s1); addRoundKey(s1, rk + round * 4); invMixColumns(s1); }
        }
        invShiftRowsSubBytes(s0);  addRoundKey(s0, rk);
        out4[blk] = make_uint4(s0.w[0], s0.w[1], s0.w[2], s0.w[3]);
        if (second) {
            invShiftRowsSubBytes(s1); addRoundKey(s1, rk);
            out4[blk + 1] = make_uint4(s1.w[0], s1.w[1], s1.w[2], s1.w[3]);
        }
    }
}
