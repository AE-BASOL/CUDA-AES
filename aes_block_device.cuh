#ifndef AES_BLOCK_DEVICE_CUH
#define AES_BLOCK_DEVICE_CUH

#include "aes_common.h"

struct AesBlock {
    uint32_t w[4];
    uint8_t b[16];
};

static __device__ __forceinline__ uint8_t aes_xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x & 0x80u) ? 0x1bu : 0));
}

static __device__ __forceinline__ uint8_t aes_gmul2(uint8_t x) { return aes_xtime(x); }
static __device__ __forceinline__ uint8_t aes_gmul3(uint8_t x) { return static_cast<uint8_t>(aes_xtime(x) ^ x); }
static __device__ __forceinline__ uint8_t aes_gmul9(uint8_t x) {
    uint8_t x2 = aes_xtime(x), x4 = aes_xtime(x2), x8 = aes_xtime(x4);
    return static_cast<uint8_t>(x8 ^ x);
}
static __device__ __forceinline__ uint8_t aes_gmul11(uint8_t x) {
    uint8_t x2 = aes_xtime(x), x4 = aes_xtime(x2), x8 = aes_xtime(x4);
    return static_cast<uint8_t>(x8 ^ x2 ^ x);
}
static __device__ __forceinline__ uint8_t aes_gmul13(uint8_t x) {
    uint8_t x2 = aes_xtime(x), x4 = aes_xtime(x2), x8 = aes_xtime(x4);
    return static_cast<uint8_t>(x8 ^ x4 ^ x);
}
static __device__ __forceinline__ uint8_t aes_gmul14(uint8_t x) {
    uint8_t x2 = aes_xtime(x), x4 = aes_xtime(x2), x8 = aes_xtime(x4);
    return static_cast<uint8_t>(x8 ^ x4 ^ x2);
}

static __device__ __forceinline__ void aes_load_block(const uint8_t *src, AesBlock &st) {
#pragma unroll
    for (int i = 0; i < 16; ++i) st.b[i] = src[i];
}

static __device__ __forceinline__ void aes_store_block(uint8_t *dst, const AesBlock &st) {
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[i] = st.b[i];
}

static __device__ __forceinline__ void aes_add_round_key(AesBlock &st, const uint32_t *rk) {
    st.w[0] ^= rk[0];
    st.w[1] ^= rk[1];
    st.w[2] ^= rk[2];
    st.w[3] ^= rk[3];
}

static __device__ __forceinline__ void aes_sub_bytes_shift_rows(AesBlock &st) {
    uint8_t tmp[16];
    const uint8_t *sb = d_sbox;
    tmp[0]  = sb[st.b[0]];  tmp[1]  = sb[st.b[5]];  tmp[2]  = sb[st.b[10]]; tmp[3]  = sb[st.b[15]];
    tmp[4]  = sb[st.b[4]];  tmp[5]  = sb[st.b[9]];  tmp[6]  = sb[st.b[14]]; tmp[7]  = sb[st.b[3]];
    tmp[8]  = sb[st.b[8]];  tmp[9]  = sb[st.b[13]]; tmp[10] = sb[st.b[2]];  tmp[11] = sb[st.b[7]];
    tmp[12] = sb[st.b[12]]; tmp[13] = sb[st.b[1]];  tmp[14] = sb[st.b[6]];  tmp[15] = sb[st.b[11]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st.b[i] = tmp[i];
}

static __device__ __forceinline__ void aes_inv_shift_rows_sub_bytes(AesBlock &st) {
    uint8_t tmp[16];
    const uint8_t *sb = d_inv_sbox;
    tmp[0]  = sb[st.b[0]];  tmp[1]  = sb[st.b[13]]; tmp[2]  = sb[st.b[10]]; tmp[3]  = sb[st.b[7]];
    tmp[4]  = sb[st.b[4]];  tmp[5]  = sb[st.b[1]];  tmp[6]  = sb[st.b[14]]; tmp[7]  = sb[st.b[11]];
    tmp[8]  = sb[st.b[8]];  tmp[9]  = sb[st.b[5]];  tmp[10] = sb[st.b[2]];  tmp[11] = sb[st.b[15]];
    tmp[12] = sb[st.b[12]]; tmp[13] = sb[st.b[9]];  tmp[14] = sb[st.b[6]];  tmp[15] = sb[st.b[3]];
#pragma unroll
    for (int i = 0; i < 16; ++i) st.b[i] = tmp[i];
}

static __device__ __forceinline__ void aes_mix_columns(AesBlock &st) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = st.b[c];
        uint8_t a1 = st.b[4 + c];
        uint8_t a2 = st.b[8 + c];
        uint8_t a3 = st.b[12 + c];
        st.b[c]      = aes_gmul2(a0) ^ aes_gmul3(a1) ^ a2 ^ a3;
        st.b[4 + c]  = a0 ^ aes_gmul2(a1) ^ aes_gmul3(a2) ^ a3;
        st.b[8 + c]  = a0 ^ a1 ^ aes_gmul2(a2) ^ aes_gmul3(a3);
        st.b[12 + c] = aes_gmul3(a0) ^ a1 ^ a2 ^ aes_gmul2(a3);
    }
}

static __device__ __forceinline__ void aes_inv_mix_columns(AesBlock &st) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t a0 = st.b[c];
        uint8_t a1 = st.b[4 + c];
        uint8_t a2 = st.b[8 + c];
        uint8_t a3 = st.b[12 + c];
        st.b[c]      = aes_gmul14(a0) ^ aes_gmul11(a1) ^ aes_gmul13(a2) ^ aes_gmul9(a3);
        st.b[4 + c]  = aes_gmul9(a0) ^ aes_gmul14(a1) ^ aes_gmul11(a2) ^ aes_gmul13(a3);
        st.b[8 + c]  = aes_gmul13(a0) ^ aes_gmul9(a1) ^ aes_gmul14(a2) ^ aes_gmul11(a3);
        st.b[12 + c] = aes_gmul11(a0) ^ aes_gmul13(a1) ^ aes_gmul9(a2) ^ aes_gmul14(a3);
    }
}

static __device__ __forceinline__ void aes_encrypt_block(AesBlock &st, const uint32_t *rk, int rounds) {
    aes_add_round_key(st, rk);
    for (int round = 1; round < rounds; ++round) {
        aes_sub_bytes_shift_rows(st);
        aes_mix_columns(st);
        aes_add_round_key(st, rk + round * 4);
    }
    aes_sub_bytes_shift_rows(st);
    aes_add_round_key(st, rk + rounds * 4);
}

static __device__ __forceinline__ void aes_decrypt_block(AesBlock &st, const uint32_t *rk, int rounds) {
    aes_add_round_key(st, rk + rounds * 4);
    for (int round = rounds - 1; round >= 1; --round) {
        aes_inv_shift_rows_sub_bytes(st);
        aes_add_round_key(st, rk + round * 4);
        aes_inv_mix_columns(st);
    }
    aes_inv_shift_rows_sub_bytes(st);
    aes_add_round_key(st, rk);
}

#endif  // AES_BLOCK_DEVICE_CUH
