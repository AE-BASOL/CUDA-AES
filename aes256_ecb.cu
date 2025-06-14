// aes256_ecb.cu – standalone CUDA ECB kernels for AES-256, compatible with main.cu
// Build together with your existing sources (e.g., main.cu, aes_common.h)
// Author: ChatGPT (OpenAI o4-mini)

#include <cuda_runtime.h>
#include <stdint.h>
#include "aes_common.h"   // provides d_roundKeys (60 words) and tables

// Constant memory from aes_common.h
extern __constant__ uint32_t d_roundKeys[];  // 60 words for AES-256
extern __constant__ uint8_t d_sbox[];
extern __constant__ uint8_t d_rsbox[];

// GF(2^8) multiplication helpers (same as in aes128_ecb.cu)
__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x & 0x80u) ? 0x1bu : 0));
}
__device__ __forceinline__ uint8_t gmul2(uint8_t x) { return xtime(x); }
__device__ __forceinline__ uint8_t gmul3(uint8_t x) { return (uint8_t)(xtime(x) ^ x); }
__device__ __forceinline__ uint8_t gmul9(uint8_t x) { uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return (uint8_t)(x8 ^ x); }
__device__ __forceinline__ uint8_t gmul11(uint8_t x){ uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return (uint8_t)(x8 ^ x2 ^ x); }
__device__ __forceinline__ uint8_t gmul13(uint8_t x){ uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return (uint8_t)(x8 ^ x4 ^ x); }
__device__ __forceinline__ uint8_t gmul14(uint8_t x){ uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return (uint8_t)(x8 ^ x4 ^ x2); }

// Round transformations
__device__ __forceinline__ void addRoundKey(uint8_t st[16], const uint8_t *rk) {
#pragma unroll
    for (int i = 0; i < 16; ++i) st[i] ^= rk[i];
}

__device__ __forceinline__ void subBytesShiftRows(uint8_t st[16]) {
    uint8_t tmp[16];
    tmp[0]=d_sbox[st[0]]; tmp[1]=d_sbox[st[5]]; tmp[2]=d_sbox[st[10]]; tmp[3]=d_sbox[st[15]];
    tmp[4]=d_sbox[st[4]]; tmp[5]=d_sbox[st[9]]; tmp[6]=d_sbox[st[14]]; tmp[7]=d_sbox[st[3]];
    tmp[8]=d_sbox[st[8]]; tmp[9]=d_sbox[st[13]]; tmp[10]=d_sbox[st[2]]; tmp[11]=d_sbox[st[7]];
    tmp[12]=d_sbox[st[12]]; tmp[13]=d_sbox[st[1]]; tmp[14]=d_sbox[st[6]]; tmp[15]=d_sbox[st[11]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st[i] = tmp[i];
}

__device__ __forceinline__ void invShiftRowsSubBytes(uint8_t st[16]) {
    uint8_t tmp[16];
    tmp[0]=d_rsbox[st[0]]; tmp[1]=d_rsbox[st[13]]; tmp[2]=d_rsbox[st[10]]; tmp[3]=d_rsbox[st[7]];
    tmp[4]=d_rsbox[st[4]]; tmp[5]=d_rsbox[st[1]]; tmp[6]=d_rsbox[st[14]]; tmp[7]=d_rsbox[st[11]];
    tmp[8]=d_rsbox[st[8]]; tmp[9]=d_rsbox[st[5]]; tmp[10]=d_rsbox[st[2]]; tmp[11]=d_rsbox[st[15]];
    tmp[12]=d_rsbox[st[12]]; tmp[13]=d_rsbox[st[9]]; tmp[14]=d_rsbox[st[6]]; tmp[15]=d_rsbox[st[3]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st[i] = tmp[i];
}

__device__ __forceinline__ void mixColumns(uint8_t st[16]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=st[c], a1=st[4+c], a2=st[8+c], a3=st[12+c];
        st[c]    = gmul2(a0)^gmul3(a1)^a2^a3;
        st[4+c]  = a0^gmul2(a1)^gmul3(a2)^a3;
        st[8+c]  = a0^a1^gmul2(a2)^gmul3(a3);
        st[12+c] = gmul3(a0)^a1^a2^gmul2(a3);
    }
}

__device__ __forceinline__ void invMixColumns(uint8_t st[16]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=st[c], a1=st[4+c], a2=st[8+c], a3=st[12+c];
        st[c]    = gmul14(a0)^gmul11(a1)^gmul13(a2)^gmul9(a3);
        st[4+c]  = gmul9(a0)^gmul14(a1)^gmul11(a2)^gmul13(a3);
        st[8+c]  = gmul13(a0)^gmul9(a1)^gmul14(a2)^gmul11(a3);
        st[12+c] = gmul11(a0)^gmul13(a1)^gmul9(a2)^gmul14(a3);
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// AES-256 ECB encryption kernel
__global__ void aes256_ecb_encrypt(const uint8_t *__restrict__ in,
                                   uint8_t *__restrict__ out,
                                   size_t nBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    uint8_t state[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) state[i] = in[idx*16 + i];
    const uint8_t *rk = reinterpret_cast<const uint8_t*>(d_roundKeys);
    // initial round
    addRoundKey(state, rk);
    // rounds 1-13
    for (int round = 1; round <= 13; ++round) {
        subBytesShiftRows(state);
        mixColumns(state);
        addRoundKey(state, rk + round*16);
    }
    // final round
    subBytesShiftRows(state);
    addRoundKey(state, rk + 14*16);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[idx*16 + i] = state[i];
}

// AES-256 ECB decryption kernel
__global__ void aes256_ecb_decrypt(const uint8_t *__restrict__ in,
                                   uint8_t *__restrict__ out,
                                   size_t nBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    uint8_t state[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) state[i] = in[idx*16 + i];
    const uint8_t *rk = reinterpret_cast<const uint8_t*>(d_roundKeys);
    // initial inverse round
    addRoundKey(state, rk + 14*16);
    // rounds 13-1
    for (int round = 13; round >= 1; --round) {
        invShiftRowsSubBytes(state);
        addRoundKey(state, rk + round*16);
        invMixColumns(state);
    }
    // final inverse round
    invShiftRowsSubBytes(state);
    addRoundKey(state, rk);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[idx*16 + i] = state[i];
}
