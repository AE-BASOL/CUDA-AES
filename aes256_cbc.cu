#include "aes_block_device.cuh"

__global__ void aes256_cbc_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint8_t feedback[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) feedback[i] = iv[i];

    const uint32_t *rk = d_roundKeys;
    for (size_t blk = 0; blk < nBlocks; ++blk) {
        AesBlock st;
        aes_load_block(in + blk * 16, st);
#pragma unroll
        for (int i = 0; i < 16; ++i) st.b[i] ^= feedback[i];
        aes_encrypt_block(st, rk, 14);
        aes_store_block(out + blk * 16, st);
#pragma unroll
        for (int i = 0; i < 16; ++i) feedback[i] = st.b[i];
    }
}

__global__ void aes256_cbc_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const uint32_t *rk = d_roundKeys;

    for (size_t blk = tid; blk < nBlocks; blk += stride) {
        AesBlock st;
        aes_load_block(in + blk * 16, st);
        aes_decrypt_block(st, rk, 14);
        const uint8_t *feedback = (blk == 0) ? iv : in + (blk - 1) * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) st.b[i] ^= feedback[i];
        aes_store_block(out + blk * 16, st);
    }
}
