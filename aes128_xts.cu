#include "aes_block_device.cuh"

extern __device__ __constant__ uint32_t d_xtsTweakRoundKeys[60];

namespace {

__device__ __forceinline__ void xts_mul_alpha(uint8_t tweak[16]) {
    uint8_t carry = 0;
    for (int i = 0; i < 16; ++i) {
        const uint8_t next = static_cast<uint8_t>(tweak[i] >> 7);
        tweak[i] = static_cast<uint8_t>((tweak[i] << 1) | carry);
        carry = next;
    }
    if (carry) tweak[0] ^= 0x87;
}

__device__ __forceinline__ void xts_tweak_for_block(const uint8_t *sector, size_t block_index, uint8_t tweak[16]) {
    AesBlock tw;
    aes_load_block(sector, tw);
    aes_encrypt_block(tw, d_xtsTweakRoundKeys, 10);
#pragma unroll
    for (int i = 0; i < 16; ++i) tweak[i] = tw.b[i];
    for (size_t i = 0; i < block_index; ++i) xts_mul_alpha(tweak);
}

}  // namespace

__global__ void aes128_xts_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *tweak) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t blk = tid; blk < nBlocks; blk += stride) {
        uint8_t tw[16];
        xts_tweak_for_block(tweak, blk, tw);
        AesBlock st;
        aes_load_block(in + blk * 16, st);
#pragma unroll
        for (int i = 0; i < 16; ++i) st.b[i] ^= tw[i];
        aes_encrypt_block(st, d_roundKeys, 10);
#pragma unroll
        for (int i = 0; i < 16; ++i) st.b[i] ^= tw[i];
        aes_store_block(out + blk * 16, st);
    }
}

__global__ void aes128_xts_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *tweak) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t blk = tid; blk < nBlocks; blk += stride) {
        uint8_t tw[16];
        xts_tweak_for_block(tweak, blk, tw);
        AesBlock st;
        aes_load_block(in + blk * 16, st);
#pragma unroll
        for (int i = 0; i < 16; ++i) st.b[i] ^= tw[i];
        aes_decrypt_block(st, d_roundKeys, 10);
#pragma unroll
        for (int i = 0; i < 16; ++i) st.b[i] ^= tw[i];
        aes_store_block(out + blk * 16, st);
    }
}
