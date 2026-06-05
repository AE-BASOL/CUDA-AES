#include "aes_block_device.cuh"

namespace {

__device__ __forceinline__ void xor_t(uint8_t a[8], unsigned t) {
    for (int i = 7; i >= 0 && t; --i) {
        a[i] ^= static_cast<uint8_t>(t);
        t >>= 8;
    }
}

__device__ __forceinline__ bool eq8(const uint8_t a[8], const uint8_t b[8]) {
    uint8_t diff = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) diff |= static_cast<uint8_t>(a[i] ^ b[i]);
    return diff == 0;
}

__device__ void kw_wrap_record(const uint8_t *in, uint8_t *out, const uint8_t a0[8]) {
    uint8_t a[8];
    uint8_t r[3][8] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = a0[i];
        r[1][i] = in[i];
        r[2][i] = in[8 + i];
    }
    for (int j = 0; j <= 5; ++j) {
        for (int i = 1; i <= 2; ++i) {
            AesBlock b;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                b.b[k] = a[k];
                b.b[8 + k] = r[i][k];
            }
            aes_encrypt_block(b, d_roundKeys, 10);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                a[k] = b.b[k];
                r[i][k] = b.b[8 + k];
            }
            xor_t(a, static_cast<unsigned>(2 * j + i));
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = a[i];
        out[8 + i] = r[1][i];
        out[16 + i] = r[2][i];
    }
}

__device__ bool kw_unwrap_record(const uint8_t *in, uint8_t *out, const uint8_t expected_a[8]) {
    uint8_t a[8];
    uint8_t r[3][8] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = in[i];
        r[1][i] = in[8 + i];
        r[2][i] = in[16 + i];
    }
    for (int j = 5; j >= 0; --j) {
        for (int i = 2; i >= 1; --i) {
            uint8_t ax[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) ax[k] = a[k];
            xor_t(ax, static_cast<unsigned>(2 * j + i));
            AesBlock b;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                b.b[k] = ax[k];
                b.b[8 + k] = r[i][k];
            }
            aes_decrypt_block(b, d_roundKeys, 10);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                a[k] = b.b[k];
                r[i][k] = b.b[8 + k];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = r[1][i];
        out[8 + i] = r[2][i];
    }
    return eq8(a, expected_a);
}

}  // namespace

__global__ void aes128_kw_wrap(const uint8_t *in, uint8_t *out, size_t nRecords) {
    const uint8_t a0[8] = {0xA6,0xA6,0xA6,0xA6,0xA6,0xA6,0xA6,0xA6};
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t rec = tid; rec < nRecords; rec += stride) kw_wrap_record(in + rec * 16, out + rec * 24, a0);
}

__global__ void aes128_kw_unwrap(const uint8_t *in, uint8_t *out, size_t nRecords, uint8_t *status) {
    const uint8_t a0[8] = {0xA6,0xA6,0xA6,0xA6,0xA6,0xA6,0xA6,0xA6};
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t rec = tid; rec < nRecords; rec += stride) status[rec] = kw_unwrap_record(in + rec * 24, out + rec * 16, a0) ? 1 : 0;
}

__global__ void aes128_kwp_wrap(const uint8_t *in, uint8_t *out, size_t nRecords) {
    const uint8_t aiv[8] = {0xA6,0x59,0x59,0xA6,0x00,0x00,0x00,0x14};
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t rec = tid; rec < nRecords; rec += stride) {
        uint8_t padded[24];
#pragma unroll
        for (int i = 0; i < 20; ++i) padded[i] = in[rec * 20 + i];
#pragma unroll
        for (int i = 20; i < 24; ++i) padded[i] = 0;
        uint8_t a[8];
        uint8_t r[4][8] = {};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            a[i] = aiv[i];
            r[1][i] = padded[i];
            r[2][i] = padded[8 + i];
            r[3][i] = padded[16 + i];
        }
        for (int j = 0; j <= 5; ++j) {
            for (int i = 1; i <= 3; ++i) {
                AesBlock b;
#pragma unroll
                for (int k = 0; k < 8; ++k) { b.b[k] = a[k]; b.b[8 + k] = r[i][k]; }
                aes_encrypt_block(b, d_roundKeys, 10);
#pragma unroll
                for (int k = 0; k < 8; ++k) { a[k] = b.b[k]; r[i][k] = b.b[8 + k]; }
                xor_t(a, static_cast<unsigned>(3 * j + i));
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[rec * 32 + i] = a[i];
            out[rec * 32 + 8 + i] = r[1][i];
            out[rec * 32 + 16 + i] = r[2][i];
            out[rec * 32 + 24 + i] = r[3][i];
        }
    }
}

__global__ void aes128_kwp_unwrap(const uint8_t *in, uint8_t *out, size_t nRecords, uint8_t *status) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t rec = tid; rec < nRecords; rec += stride) {
        uint8_t a[8];
        uint8_t r[4][8] = {};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            a[i] = in[rec * 32 + i];
            r[1][i] = in[rec * 32 + 8 + i];
            r[2][i] = in[rec * 32 + 16 + i];
            r[3][i] = in[rec * 32 + 24 + i];
        }
        for (int j = 5; j >= 0; --j) {
            for (int i = 3; i >= 1; --i) {
                uint8_t ax[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) ax[k] = a[k];
                xor_t(ax, static_cast<unsigned>(3 * j + i));
                AesBlock b;
#pragma unroll
                for (int k = 0; k < 8; ++k) { b.b[k] = ax[k]; b.b[8 + k] = r[i][k]; }
                aes_decrypt_block(b, d_roundKeys, 10);
#pragma unroll
                for (int k = 0; k < 8; ++k) { a[k] = b.b[k]; r[i][k] = b.b[8 + k]; }
            }
        }
        const bool aiv_ok = a[0] == 0xA6 && a[1] == 0x59 && a[2] == 0x59 && a[3] == 0xA6 && a[4] == 0 && a[5] == 0 && a[6] == 0 && a[7] == 0x14;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[rec * 20 + i] = r[1][i];
            out[rec * 20 + 8 + i] = r[2][i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) out[rec * 20 + 16 + i] = r[3][i];
        const bool pad_ok = r[3][4] == 0 && r[3][5] == 0 && r[3][6] == 0 && r[3][7] == 0;
        status[rec] = (aiv_ok && pad_ok) ? 1 : 0;
    }
}
