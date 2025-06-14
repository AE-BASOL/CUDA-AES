// aes128_ecb.cu – standalone CUDA ECB kernels compatible with main.cu
// Build together with your existing sources (e.g., main.cu, aes_common.h)
// Author: ChatGPT (OpenAI o3)

#include <cuda_runtime.h>
#include <stdint.h>
#include "aes_common.h"   // provides d_roundKeys and tables

// Constant memory helpers (as in aes_common.h)
extern __constant__ uint32_t d_roundKeys[];  // 44 words for AES-128
extern __constant__ uint8_t d_sbox[];
__constant__ uint8_t d_rsbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

// GF(2^8) multiplication helpers
__device__ __forceinline__ uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x & 0x80u) ? 0x1bu : 0));
}
__device__ __forceinline__ uint8_t gmul2(uint8_t x) { return xtime(x); }
__device__ __forceinline__ uint8_t gmul3(uint8_t x) { return static_cast<uint8_t>(xtime(x) ^ x); }
__device__ __forceinline__ uint8_t gmul9(uint8_t x) { uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x); }
__device__ __forceinline__ uint8_t gmul11(uint8_t x){ uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x2 ^ x); }
__device__ __forceinline__ uint8_t gmul13(uint8_t x){ uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x4 ^ x); }
__device__ __forceinline__ uint8_t gmul14(uint8_t x){ uint8_t x2=xtime(x), x4=xtime(x2), x8=xtime(x4); return static_cast<uint8_t>(x8 ^ x4 ^ x2); }

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
    for (int i = 0; i < 16; ++i) st[i]=tmp[i];
}

__device__ __forceinline__ void invShiftRowsSubBytes(uint8_t st[16]) {
    uint8_t tmp[16];
    tmp[0]=d_rsbox[st[0]]; tmp[1]=d_rsbox[st[13]]; tmp[2]=d_rsbox[st[10]]; tmp[3]=d_rsbox[st[7]];
    tmp[4]=d_rsbox[st[4]]; tmp[5]=d_rsbox[st[1]]; tmp[6]=d_rsbox[st[14]]; tmp[7]=d_rsbox[st[11]];
    tmp[8]=d_rsbox[st[8]]; tmp[9]=d_rsbox[st[5]]; tmp[10]=d_rsbox[st[2]]; tmp[11]=d_rsbox[st[15]];
    tmp[12]=d_rsbox[st[12]]; tmp[13]=d_rsbox[st[9]]; tmp[14]=d_rsbox[st[6]]; tmp[15]=d_rsbox[st[3]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st[i]=tmp[i];
}

__device__ __forceinline__ void mixColumns(uint8_t st[16]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=st[c], a1=st[4+c], a2=st[8+c], a3=st[12+c];
        st[c]=gmul2(a0)^gmul3(a1)^a2^a3;
        st[4+c]=a0^gmul2(a1)^gmul3(a2)^a3;
        st[8+c]=a0^a1^gmul2(a2)^gmul3(a3);
        st[12+c]=gmul3(a0)^a1^a2^gmul2(a3);
    }
}

__device__ __forceinline__ void invMixColumns(uint8_t st[16]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0=st[c], a1=st[4+c], a2=st[8+c], a3=st[12+c];
        st[c]=gmul14(a0)^gmul11(a1)^gmul13(a2)^gmul9(a3);
        st[4+c]=gmul9(a0)^gmul14(a1)^gmul11(a2)^gmul13(a3);
        st[8+c]=gmul13(a0)^gmul9(a1)^gmul14(a2)^gmul11(a3);
        st[12+c]=gmul11(a0)^gmul13(a1)^gmul9(a2)^gmul14(a3);
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Kernels
__global__ void aes128_ecb_encrypt(const uint8_t *__restrict__ in,
                                   uint8_t *__restrict__ out,
                                   size_t nBlocks) {
    size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nBlocks) return;
    uint8_t state[16];
#pragma unroll
    for(int i=0;i<16;++i) state[i]=in[idx*16+i];
    const uint8_t *rk=reinterpret_cast<const uint8_t*>(d_roundKeys);
    addRoundKey(state,rk);
    for(int round=1;round<=9;++round) {
        subBytesShiftRows(state);
        mixColumns(state);
        addRoundKey(state,rk+round*16);
    }
    subBytesShiftRows(state);
    addRoundKey(state,rk+10*16);
#pragma unroll
    for(int i=0;i<16;++i) out[idx*16+i]=state[i];
}

__global__ void aes128_ecb_decrypt(const uint8_t *__restrict__ in,
                                   uint8_t *__restrict__ out,
                                   size_t nBlocks) {
    size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=nBlocks) return;
    uint8_t state[16];
#pragma unroll
    for(int i=0;i<16;++i) state[i]=in[idx*16+i];
    const uint8_t *rk=reinterpret_cast<const uint8_t*>(d_roundKeys);
    addRoundKey(state,rk+10*16);
    for(int round=9;round>=1;--round) {
        invShiftRowsSubBytes(state);
        addRoundKey(state,rk+round*16);
        invMixColumns(state);
    }
    invShiftRowsSubBytes(state);
    addRoundKey(state,rk);
#pragma unroll
    for(int i=0;i<16;++i) out[idx*16+i]=state[i];
}