#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <openssl/evp.h>
#include <iostream>
#include <sstream>
#include <iterator>
#include "aes_common.h"
#include "profiling_helpers.h"
#include <iomanip> // For formatting output

// -------------------------------
// Error handling macro
// -------------------------------
#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// -------------------------------
// Constants and parameters
// -------------------------------
constexpr int THREADS_PER_BLOCK = 256;
constexpr int DEFAULT_NUM_RUNS  = 5;
static const size_t DEFAULT_SIZES[] = {1ull<<20, 10ull<<20, 100ull<<20, 1ull<<30};
static const char*  MODES[]     = {
    "ecb-128","ecb-256",
    "cbc-128","cbc-256",
    "cfb-128","cfb-256",
    "ofb-128","ofb-256",
    "ctr-128","ctr-256",
    "gcm-128","gcm-256",
    "ccm-128","ccm-256"
};
static const char*  BENCH_SCHEMA_VERSION = "phase3.v1";

// -------------------------------
// Print header helper
// -------------------------------
inline void print_header() {
    printf("%-10s %-10s %-12s %-5s %-10s %-10s %-5s\n",
           "TYPE", "MODE", "SIZE_BYTES", "RUN", "MS", "GiB/s", "OP");
    printf("-------------------------------------------------------------------------------------\n");
}

// -------------------------------
// CTR helper
// -------------------------------
static void packCtr(const uint8_t iv[12], uint64_t &lo, uint64_t &hi) {
    uint32_t w0=0,w1=0,w2=0; memcpy(&w0,iv,4); memcpy(&w1,iv+4,4); memcpy(&w2,iv+8,4);
    uint32_t w3=0x01000000u; lo = (uint64_t)w0 | ((uint64_t)w1<<32); hi = (uint64_t)w2 | ((uint64_t)w3<<32);
}

// -------------------------------
// Device GF multiply used for --gf-mult and GCM debug
// -------------------------------
__device__ inline void gf_mul128_dev(uint64_t &Ah, uint64_t &Al, uint64_t Bh, uint64_t Bl) {
    uint64_t Zh=0, Zl=0, Vh=Bh, Vl=Bl; const uint64_t R=0xE100000000000000ULL;
    for(int i=0;i<128;++i){
        if(Al & 1ULL){ Zl^=Vl; Zh^=Vh; }
        bool carry = Vl & 1ULL;
        Vl = (Vl>>1) | (Vh<<63); Vh >>=1; if(carry) Vh^=R;
        Al = (Al>>1) | (Ah<<63); Ah >>=1;
    }
    Ah=Zh; Al=Zl;
}

// Kernel performing many GF multiplies per thread
__global__ void gf_mult_kernel(uint64_t *out) {
    uint64_t Ah=0x0123456789abcdefULL, Al=0xfedcba9876543210ULL;
    uint64_t Bh=0x0fedcba987654321ULL, Bl=0x1234567890abcdefULL;
    for(int i=0;i<1000000;i++) {
        gf_mul128_dev(Ah,Al,Bh,Bl);
        Bh += 1; Bl += 1;
    }
    out[threadIdx.x] = Ah ^ Al ^ Bh ^ Bl;
}

// Kernel computing per-thread partial GHASH
__global__ void gcm_partial_kernel(const uint8_t *cipher, size_t nBlocks,
                                   uint64_t Hh, uint64_t Hl,
                                   uint64_t *outH, uint64_t *outL) {
    int tid = threadIdx.x;
    size_t start = tid * nBlocks / blockDim.x;
    size_t end   = (tid+1) * nBlocks / blockDim.x;
    uint64_t Xh=0, Xl=0;
    for(size_t i=start;i<end;++i){
        uint64_t cl=((const uint64_t*)cipher)[2*i];
        uint64_t ch=((const uint64_t*)cipher)[2*i+1];
        Xl ^= cl; Xh ^= ch; gf_mul128_dev(Xh,Xl,Hh,Hl);
    }
    outH[tid]=Xh; outL[tid]=Xl;
}

// -------------------------------
// OpenSSL throughput helper
// -------------------------------
static double cpu_aes_throughput(const void* src, size_t bytes,
                                 const unsigned char* key, int bits,
                                 bool decrypt, const EVP_CIPHER* (*cipherSel)()) {
    std::vector<unsigned char> buf(bytes);
    std::vector<unsigned char> iv(16,0);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    const EVP_CIPHER *cipher = cipherSel();
    if(decrypt) EVP_DecryptInit_ex(ctx, cipher, nullptr, key, iv.data());
    else        EVP_EncryptInit_ex(ctx, cipher, nullptr, key, iv.data());
    EVP_CIPHER_CTX_set_padding(ctx,0);
    auto t0=std::chrono::high_resolution_clock::now();
    int outLen=0,total=0;
    if(decrypt) EVP_DecryptUpdate(ctx, buf.data(), &outLen, (const unsigned char*)src, (int)bytes);
    else        EVP_EncryptUpdate(ctx, buf.data(), &outLen, (const unsigned char*)src, (int)bytes);
    total += outLen;
    if(decrypt) EVP_DecryptFinal_ex(ctx, buf.data()+total, &outLen);
    else        EVP_EncryptFinal_ex(ctx, buf.data()+total, &outLen);
    total += outLen;
    auto t1=std::chrono::high_resolution_clock::now();
    EVP_CIPHER_CTX_free(ctx);
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double gib=(double)bytes/(double)(1ull<<30);
    return gib/(ms/1000.0);
}

static double cpu_ccm_throughput(const void* src, size_t bytes,
                                 const unsigned char* key, int bits,
                                 bool decrypt) {
    const EVP_CIPHER *cipher = bits == 128 ? EVP_aes_128_ccm() : EVP_aes_256_ccm();
    std::vector<unsigned char> buf(bytes);
    std::vector<unsigned char> iv(12, 0);
    std::vector<unsigned char> tag(16, 0);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    int outLen = 0;
    if (decrypt) {
        EVP_DecryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_SET_IVLEN, static_cast<int>(iv.size()), nullptr);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_SET_TAG, static_cast<int>(tag.size()), tag.data());
        EVP_DecryptInit_ex(ctx, nullptr, nullptr, key, iv.data());
        EVP_DecryptUpdate(ctx, nullptr, &outLen, nullptr, static_cast<int>(bytes));
        EVP_DecryptUpdate(ctx, buf.data(), &outLen, static_cast<const unsigned char*>(src), static_cast<int>(bytes));
    } else {
        EVP_EncryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_SET_IVLEN, static_cast<int>(iv.size()), nullptr);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_SET_TAG, static_cast<int>(tag.size()), nullptr);
        EVP_EncryptInit_ex(ctx, nullptr, nullptr, key, iv.data());
        EVP_EncryptUpdate(ctx, nullptr, &outLen, nullptr, static_cast<int>(bytes));
        EVP_EncryptUpdate(ctx, buf.data(), &outLen, static_cast<const unsigned char*>(src), static_cast<int>(bytes));
        EVP_EncryptFinal_ex(ctx, buf.data() + outLen, &outLen);
        EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_CCM_GET_TAG, static_cast<int>(tag.size()), tag.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    EVP_CIPHER_CTX_free(ctx);
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double gib = static_cast<double>(bytes) / static_cast<double>(1ull << 30);
    return gib / (ms / 1000.0);
}

// -------------------------------
// Helper to generate random bytes
// -------------------------------
static void fill_random(uint8_t *buf, size_t n, std::mt19937_64 &rng) {
    for(size_t i=0;i<n;++i) buf[i] = static_cast<uint8_t>(rng() & 0xFF);
}

struct BenchmarkConfig {
    int runs = DEFAULT_NUM_RUNS;
    std::vector<size_t> sizes = std::vector<size_t>(std::begin(DEFAULT_SIZES), std::end(DEFAULT_SIZES));
    std::string bench_dir = "bench";
    std::string command_line;
};

static std::string join_command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i) out << ' ';
        std::string arg = argv[i] ? argv[i] : "";
        if (arg.find_first_of(" \t\"") != std::string::npos) {
            out << '"';
            for (char ch : arg) {
                if (ch == '"') out << '\\';
                out << ch;
            }
            out << '"';
        } else {
            out << arg;
        }
    }
    return out.str();
}

static bool parse_size_list(const std::string& value, std::vector<size_t>& sizes) {
    std::vector<size_t> parsed;
    std::stringstream ss(value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) return false;
        char* end = nullptr;
        unsigned long long raw = std::strtoull(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0' || raw == 0) return false;
        parsed.push_back(static_cast<size_t>(raw));
    }
    if (parsed.empty()) return false;
    sizes = parsed;
    return true;
}

static std::string csv_escape(const std::string& value) {
    bool quote = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!quote) return value;
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') escaped += "\"\"";
        else escaped += ch;
    }
    escaped += "\"";
    return escaped;
}

static std::string benchmark_run_id() {
    auto now = std::chrono::system_clock::now();
    auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return std::to_string(epoch_ms);
}

static void write_metadata_row(std::ofstream& out, const std::string& key, const std::string& value) {
    out << csv_escape(key) << ',' << csv_escape(value) << '\n';
}

static void write_benchmark_metadata(const BenchmarkConfig& config, const std::string& run_id) {
    std::filesystem::create_directories(config.bench_dir);
    std::ofstream out(std::filesystem::path(config.bench_dir) / "run_metadata.csv", std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        fprintf(stderr, "Error: Could not write benchmark metadata\n");
        return;
    }

    out << "key,value\n";
    write_metadata_row(out, "schema_version", BENCH_SCHEMA_VERSION);
    write_metadata_row(out, "benchmark_run_id", run_id);
    write_metadata_row(out, "command_line", config.command_line);
    write_metadata_row(out, "run_count", std::to_string(config.runs));
    write_metadata_row(out, "bench_dir", config.bench_dir);
    std::ostringstream size_list;
    for (size_t i = 0; i < config.sizes.size(); ++i) {
        if (i) size_list << ';';
        size_list << config.sizes[i];
    }
    write_metadata_row(out, "sizes_bytes", size_list.str());

#ifdef _WIN32
    write_metadata_row(out, "os", "windows");
#elif defined(__linux__)
    write_metadata_row(out, "os", "linux");
#elif defined(__APPLE__)
    write_metadata_row(out, "os", "macos");
#else
    write_metadata_row(out, "os", "unknown");
#endif

#ifdef _MSC_VER
    write_metadata_row(out, "compiler", "msvc " + std::to_string(_MSC_VER));
#elif defined(__GNUC__)
    write_metadata_row(out, "compiler", "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__));
#else
    write_metadata_row(out, "compiler", "unknown");
#endif
    write_metadata_row(out, "build_type", "unknown");

    int runtime_version = 0;
    int driver_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        write_metadata_row(out, "cuda_runtime_version", std::to_string(runtime_version));
    } else {
        write_metadata_row(out, "cuda_runtime_version", "unknown");
    }
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        write_metadata_row(out, "cuda_driver_version", std::to_string(driver_version));
    } else {
        write_metadata_row(out, "cuda_driver_version", "unknown");
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess) {
        write_metadata_row(out, "device_count", std::to_string(device_count));
        for (int device = 0; device < device_count; ++device) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
                std::string prefix = "device_" + std::to_string(device) + "_";
                write_metadata_row(out, prefix + "name", prop.name);
                write_metadata_row(out, prefix + "compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
                write_metadata_row(out, prefix + "global_mem_bytes", std::to_string(static_cast<unsigned long long>(prop.totalGlobalMem)));
            }
        }
    } else {
        write_metadata_row(out, "device_count", "unknown");
    }
    write_metadata_row(out, "clocks_persistence_note", "not captured; record fixed clocks and persistence mode manually when publishing results");
}

static void append_benchmark_row(const std::string& path,
                                 const std::string& run_id,
                                 const std::string& timing_scope,
                                 const std::string& device,
                                 const std::string& mode,
                                 size_t bytes,
                                 int run_index,
                                 int run_count,
                                 double ms,
                                 double throughput,
                                 const std::string& operation,
                                 const std::string& command_line) {
    std::ofstream csv(path, std::ios::app);
    if (!csv.is_open()) {
        fprintf(stderr, "Error: Could not write to %s\n", path.c_str());
        return;
    }
    csv << BENCH_SCHEMA_VERSION << ','
        << csv_escape(run_id) << ','
        << csv_escape(timing_scope) << ','
        << csv_escape(device) << ','
        << csv_escape(mode) << ','
        << bytes << ','
        << run_index << ','
        << run_count << ','
        << std::fixed << std::setprecision(3) << ms << ','
        << throughput << ','
        << csv_escape(operation) << ','
        << csv_escape(command_line) << '\n';
}

// -------------------------------
// CTR preview routine
// -------------------------------
static int ctr_preview() {
    std::mt19937_64 rng(42);
    std::vector<uint8_t> key(16); fill_random(key.data(),16,rng);
    std::vector<uint8_t> iv(12);  fill_random(iv.data(),12,rng);

    std::vector<uint32_t> rk(44); expandKey128(key.data(), rk.data());
    init_roundKeys(rk.data(), (int)rk.size());

    uint8_t *d_in,*d_out; CHECK_CUDA(cudaMalloc(&d_in,32)); CHECK_CUDA(cudaMalloc(&d_out,32));
    CHECK_CUDA(cudaMemset(d_in,0,32));
    uint64_t lo=0,hi=0; packCtr(iv.data(),lo,hi);
    NVTX_PUSH("CTR_PREVIEW");
    aes128_ctr_encrypt<<<1,1>>>(d_in,d_out,2,lo,hi);
    CHECK_CUDA(cudaDeviceSynchronize());
    NVTX_POP();
    uint8_t h_out[32]; CHECK_CUDA(cudaMemcpy(h_out,d_out,32,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));

    printf("CTR_PREVIEW,");
    for(int i=0;i<32;i++){ printf("%02x", h_out[i]); if(i==15) printf(","); }
    printf("\n");
    return 0;
}

// -------------------------------
// GF multiply benchmark
// -------------------------------
static int gf_mult_bench() {
    // Ensure directory exists, handle potential errors if std::filesystem is problematic
    try {
        std::filesystem::create_directories("bench");
    } catch (const std::filesystem::filesystem_error& e) {
        fprintf(stderr, "Filesystem error: %s\n", e.what());
        // Decide if this is a fatal error or if the program can continue
    }
    // CPU part
    double ms_cpu=0.0; {
        __m128i a = _mm_set_epi64x(0x0123456789abcdefULL,0xfedcba9876543210ULL);
        __m128i b = _mm_set_epi64x(0x0fedcba987654321ULL,0x1234567890abcdefULL);
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<1000000;i++) {
            __m128i r = _mm_clmulepi64_si128(a,b,0x00);
            a = _mm_xor_si128(a,r);
            b = _mm_xor_si128(b,r);
        }
        auto t1=std::chrono::high_resolution_clock::now();
        ms_cpu=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    double gbps_cpu = (1000000.0*128/1e9) / (ms_cpu/1000.0);

    // GPU part
    float ms_gpu=0.0; double gbps_gpu=0.0; {
        uint64_t *d_out; CHECK_CUDA(cudaMalloc(&d_out, THREADS_PER_BLOCK*sizeof(uint64_t)));
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        gf_mult_kernel<<<1,THREADS_PER_BLOCK>>>(d_out);
        cudaEventRecord(e); CHECK_CUDA(cudaEventSynchronize(e));
        cudaEventElapsedTime(&ms_gpu,s,e);
        CHECK_CUDA(cudaEventDestroy(s));
        CHECK_CUDA(cudaEventDestroy(e));
        CHECK_CUDA(cudaFree(d_out));
        gbps_gpu = (1000000.0*THREADS_PER_BLOCK*128/1e9) / (ms_gpu/1000.0);
    }

    std::ofstream f("bench/gf_mult.csv", std::ios::app);
    f << "SRC,CPU,1000000," << ms_cpu << ',' << gbps_cpu << "\n";
    f << "SRC,GPU," << (1000000*THREADS_PER_BLOCK) << ',' << ms_gpu << ',' << gbps_gpu << "\n";
    std::cout << "GF_MULT CPU "<<gbps_cpu<<" Gbps\n";
    std::cout << "GF_MULT GPU "<<gbps_gpu<<" Gbps\n";
    return 0;
}

// -------------------------------
// GCM debug routine: encrypt 64B and dump partial GHASH
// -------------------------------
static int gcm_debug_run() {
    // Ensure directory exists
    try {
        std::filesystem::create_directories("bench");
    } catch (const std::filesystem::filesystem_error& e) {
        fprintf(stderr, "Filesystem error: %s\n", e.what());
    }
    std::mt19937_64 rng(123);
    const size_t bytes=64; size_t nBlocks=bytes/16;
    uint8_t *h_plain,*h_cipher; CHECK_CUDA(cudaMallocHost(&h_plain,bytes)); CHECK_CUDA(cudaMallocHost(&h_cipher,bytes));
    fill_random(h_plain,bytes,rng);
    std::vector<uint8_t> key(16); fill_random(key.data(),16,rng);
    std::vector<uint8_t> iv(12);  fill_random(iv.data(),12,rng);
    std::vector<uint32_t> rk(44); expandKey128(key.data(), rk.data());
    init_roundKeys(rk.data(), (int)rk.size());
    uint8_t *d_plain,*d_cipher,*d_tag,*d_iv; CHECK_CUDA(cudaMalloc(&d_plain,bytes)); CHECK_CUDA(cudaMalloc(&d_cipher,bytes)); CHECK_CUDA(cudaMalloc(&d_tag,16)); CHECK_CUDA(cudaMalloc(&d_iv,12));
    CHECK_CUDA(cudaMemcpy(d_plain,h_plain,bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv,iv.data(),12,cudaMemcpyHostToDevice));
    aes128_gcm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_plain,d_cipher,nBlocks,d_iv,d_tag);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_cipher,d_cipher,bytes,cudaMemcpyDeviceToHost));

    // compute H = AES_k(0)
    uint8_t *d_zero,*d_h; CHECK_CUDA(cudaMalloc(&d_zero,16)); CHECK_CUDA(cudaMalloc(&d_h,16));
    CHECK_CUDA(cudaMemset(d_zero,0,16));
    aes128_ecb_encrypt<<<1,1>>>(d_zero,d_h,1);
    CHECK_CUDA(cudaDeviceSynchronize());
    uint8_t hbuf[16]; CHECK_CUDA(cudaMemcpy(hbuf,d_h,16,cudaMemcpyDeviceToHost));
    uint64_t Hl=((uint64_t*)hbuf)[0]; uint64_t Hh=((uint64_t*)hbuf)[1];
    CHECK_CUDA(cudaFree(d_zero)); CHECK_CUDA(cudaFree(d_h));

    // partial GHASH
    uint64_t *d_ph,*d_pl; CHECK_CUDA(cudaMalloc(&d_ph,THREADS_PER_BLOCK*sizeof(uint64_t))); CHECK_CUDA(cudaMalloc(&d_pl,THREADS_PER_BLOCK*sizeof(uint64_t)));
    gcm_partial_kernel<<<1,THREADS_PER_BLOCK>>>(d_cipher,nBlocks,Hh,Hl,d_ph,d_pl);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<uint64_t> ph(THREADS_PER_BLOCK), pl(THREADS_PER_BLOCK);
    CHECK_CUDA(cudaMemcpy(ph.data(),d_ph,THREADS_PER_BLOCK*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(pl.data(),d_pl,THREADS_PER_BLOCK*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_ph)); CHECK_CUDA(cudaFree(d_pl));
    std::ofstream out("bench/ghash_partials.txt");
    for(int i=0;i<THREADS_PER_BLOCK;i++)
        out << i << "," << std::hex << ph[i] << "," << pl[i] << std::dec << "\n";

    CHECK_CUDA(cudaFree(d_plain)); CHECK_CUDA(cudaFree(d_cipher)); CHECK_CUDA(cudaFree(d_tag)); CHECK_CUDA(cudaFree(d_iv));
    CHECK_CUDA(cudaFreeHost(h_plain)); CHECK_CUDA(cudaFreeHost(h_cipher));
    std::cout << "GHASH partials written to bench/ghash_partials.txt\n";
    return 0;
}

// -------------------------------
// [PYTHON_EXPORT] Helper function to ensure correct CSV headers
// -------------------------------
void ensure_csv_header(const std::string& filename, const std::string& header) {
    std::ofstream file;
    bool write_header = false;

    if (!std::filesystem::exists(filename) || std::filesystem::file_size(filename) == 0) {
        write_header = true;
    }

    if (write_header) {
        file.open(filename, std::ios::out | std::ios::trunc);
        if (file.is_open()) {
            file << header << std::endl;
            file.close();
        } else {
            fprintf(stderr, "Error: Could not open %s to write header.\n", filename.c_str());
        }
    }
}

// -------------------------------
// Main benchmark loop
// -------------------------------
int main(int argc, char** argv) {
    int blockOverride = THREADS_PER_BLOCK;
    bool decrypt=false, doCtrPreview=false, doGcmDebug=false, doGfMult=false;
    BenchmarkConfig bench_config;
    bench_config.command_line = join_command_line(argc, argv);

    // getopt_long related code will be commented out for Windows compatibility
    /*
    enum { OPT_CTR_PREVIEW=1000, OPT_GCM_DEBUG, OPT_GF_MULT };
    static struct option opts[] = {
        {\"block\", required_argument, nullptr, \'b\'},
        {\"decrypt\", no_argument, nullptr, \'d\'},
        {\"ctr-preview\", no_argument, nullptr, OPT_CTR_PREVIEW},
        {\"gcm-debug\", no_argument, nullptr, OPT_GCM_DEBUG},
        {\"gf-mult\", no_argument, nullptr, OPT_GF_MULT},
        {\"help\", no_argument, nullptr, \'h\'},
        {0,0,0,0}
    };
    while(true){
        int idx=0; int c=getopt_long(argc,argv,\"b:dh\",opts,&idx); if(c==-1) break;
        switch(c){
            case \'b\': blockOverride=atoi(optarg); break;
            case \'d\': decrypt=true; break;
            case OPT_CTR_PREVIEW: doCtrPreview=true; break;
            case OPT_GCM_DEBUG: doGcmDebug=true; break;
            case OPT_GF_MULT: doGfMult=true; break;
            case \'h\':
            default:
                std::cout << \"Usage: \"<<argv[0]<<\" [--block N] [--decrypt] [--ctr-preview] [--gcm-debug] [--gf-mult]\\n\";
                return 0;
        }
    }
    */
    // Manual parsing for essential flags as a temporary measure
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--decrypt") {
            decrypt = true;
        } else if (arg == "--ctr-preview") {
            doCtrPreview = true;
        } else if (arg == "--gcm-debug") {
            doGcmDebug = true;
        } else if (arg == "--gf-mult") {
            doGfMult = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            bench_config.runs = std::atoi(argv[++i]);
            if (bench_config.runs <= 0) {
                fprintf(stderr, "Error: --runs must be a positive integer\n");
                return EXIT_FAILURE;
            }
        } else if (arg == "--sizes" && i + 1 < argc) {
            if (!parse_size_list(argv[++i], bench_config.sizes)) {
                fprintf(stderr, "Error: --sizes expects comma-separated positive byte counts\n");
                return EXIT_FAILURE;
            }
        } else if (arg == "--bench-dir" && i + 1 < argc) {
            bench_config.bench_dir = argv[++i];
            if (bench_config.bench_dir.empty()) {
                fprintf(stderr, "Error: --bench-dir must not be empty\n");
                return EXIT_FAILURE;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--decrypt] [--runs N] [--sizes bytes[,bytes]] [--bench-dir PATH] [--ctr-preview] [--gcm-debug] [--gf-mult]\n";
            std::cout << "Benchmark rows are kernel_only timings. Run metadata is written to bench/run_metadata.csv by default.\n";
            std::cout << "Note: --block N is currently disabled due to getopt_long incompatibility on Windows.\n";
            return 0;
        } else if (arg == "--runs" || arg == "--sizes" || arg == "--bench-dir") {
            fprintf(stderr, "Error: %s requires a value\n", arg.c_str());
            return EXIT_FAILURE;
        }
        // blockOverride is not parsed in this simplified version
    }


    // Ensure directory exists
    try {
        std::filesystem::create_directories(bench_config.bench_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        fprintf(stderr, "Filesystem error: %s\n", e.what());
    }
    init_T_tables();

    // [PYTHON_EXPORT] Define CSV headers
    const std::string csv_header = "schema_version,benchmark_run_id,timing_scope,device,cipher,block_size,run_index,run_count,time_ms,GiB/s,operation,command_line";
    const std::string run_id = benchmark_run_id();
    const std::string gpu_csv_path = (std::filesystem::path(bench_config.bench_dir) / "thr_gpu.csv").string();
    const std::string cpu_csv_path = (std::filesystem::path(bench_config.bench_dir) / "thr_cpu.csv").string();

    // Ensure CSV headers are correct
    ensure_csv_header(gpu_csv_path, csv_header);
    ensure_csv_header(cpu_csv_path, csv_header);
    write_benchmark_metadata(bench_config, run_id);

    if(doCtrPreview) return ctr_preview();
    if(doGcmDebug)   return gcm_debug_run();
    if(doGfMult)     return gf_mult_bench();

    // Print header for benchmark results ONCE before all results
    print_header();

    std::mt19937_64 rng(12345);
    for(const char* modeStr : MODES){
        std::string mode(modeStr);
        bool isEcb = mode.find("ecb")==0;
        bool isCbc = mode.find("cbc")==0;
        bool isCfb = mode.find("cfb")==0;
        bool isOfb = mode.find("ofb")==0;
        bool isCtr = mode.find("ctr")==0;
        bool isGcm = mode.find("gcm")==0;
        bool isCcm = mode.find("ccm")==0;
        int bits = mode.find("256")!=std::string::npos ? 256 : 128;
        size_t keyBytes = bits/8;
        std::vector<uint8_t> key(keyBytes); fill_random(key.data(),keyBytes,rng);
        std::vector<uint32_t> rk(bits==128?44:60);
        if(bits==128) expandKey128(key.data(),rk.data()); else expandKey256(key.data(),rk.data());
        init_roundKeys(rk.data(), (int)rk.size());
        const bool usesFeedbackIv = isCbc || isCfb || isOfb;
        const bool usesAuthTag = isGcm || isCcm;
        const size_t iv_bytes = usesFeedbackIv ? 16 : 12;
        std::vector<uint8_t> iv(iv_bytes);
        if(!isEcb) fill_random(iv.data(), iv.size(), rng);

        for(size_t sz : bench_config.sizes){
            size_t nBlocks=(sz+15)/16; size_t bytes=nBlocks*16;

            printf("\n================ ROUND %zu ================\n", sz);
            printf("ROUND_TRIP_CHECK %-10s %-12zu\n", mode.c_str(), bytes);

            // Print Key and IV (first 8 bytes for brevity)
            printf("  Key Used:         ");
            for(int k_idx = 0; k_idx < std::min((size_t)8, keyBytes); ++k_idx) printf("%02x", key[k_idx]);
            printf("...\n");
            if(!isEcb){
                printf("  IV Used:          ");
                for(int iv_idx = 0; iv_idx < std::min((size_t)8, iv.size()); ++iv_idx) printf("%02x", iv[iv_idx]);
                printf("...\n");
            }

            uint8_t *h_rt_original, *h_rt_decrypted_gpu, *h_rt_cipher_gpu;
            CHECK_CUDA(cudaMallocHost(&h_rt_original, bytes));
            CHECK_CUDA(cudaMallocHost(&h_rt_decrypted_gpu, bytes));
            CHECK_CUDA(cudaMallocHost(&h_rt_cipher_gpu, bytes)); // For storing GPU ciphertext

            fill_random(h_rt_original, bytes, rng);

            std::vector<uint8_t> h_rt_cipher(bytes); // Host buffer for ciphertext (if needed)

            uint8_t *d_rt_plain, *d_rt_cipher, *d_rt_decrypted_final;
            uint8_t *d_rt_iv = nullptr, *d_rt_tag_encrypt = nullptr, *d_rt_tag_decrypt_out = nullptr;

            CHECK_CUDA(cudaMalloc(&d_rt_plain, bytes));
            CHECK_CUDA(cudaMalloc(&d_rt_cipher, bytes));
            CHECK_CUDA(cudaMalloc(&d_rt_decrypted_final, bytes));

            CHECK_CUDA(cudaMemcpy(d_rt_plain, h_rt_original, bytes, cudaMemcpyHostToDevice));

            dim3 rt_kernel_block_dim(blockOverride);
            if (usesAuthTag) rt_kernel_block_dim.x = THREADS_PER_BLOCK; // AEAD kernels use fixed block size
            dim3 rt_kernel_grid_dim((unsigned)((nBlocks + rt_kernel_block_dim.x - 1) / rt_kernel_block_dim.x));
            if (usesAuthTag) rt_kernel_grid_dim.x = 1; // AEAD kernels compute tags within one block

            if (usesFeedbackIv || usesAuthTag) {
                CHECK_CUDA(cudaMalloc(&d_rt_iv, iv.size()));
                CHECK_CUDA(cudaMemcpy(d_rt_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
            }
            if (usesAuthTag) {
                CHECK_CUDA(cudaMalloc(&d_rt_tag_encrypt, 16));
                CHECK_CUDA(cudaMalloc(&d_rt_tag_decrypt_out, 16));
            }

            // Perform Encryption
            if(isEcb && bits==128) aes128_ecb_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks);
            else if(isEcb && bits==256) aes256_ecb_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks);
            else if(isCbc && bits==128) aes128_cbc_encrypt<<<1,1>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv);
            else if(isCbc && bits==256) aes256_cbc_encrypt<<<1,1>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv);
            else if(isCfb && bits==128) aes128_cfb_encrypt<<<1,1>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv);
            else if(isCfb && bits==256) aes256_cfb_encrypt<<<1,1>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv);
            else if(isOfb && bits==128) aes128_ofb_encrypt<<<1,1>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv);
            else if(isOfb && bits==256) aes256_ofb_encrypt<<<1,1>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv);
            else if(isCtr && bits==128){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); aes128_ctr_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks,lo,hi); }
            else if(isCtr && bits==256){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); aes256_ctr_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks,lo,hi); }
            else if(isGcm && bits==128) aes128_gcm_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv, d_rt_tag_encrypt);
            else if(isGcm && bits==256) aes256_gcm_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv, d_rt_tag_encrypt);
            else if(isCcm && bits==128) aes128_ccm_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv, d_rt_tag_encrypt);
            else if(isCcm && bits==256) aes256_ccm_encrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_plain, d_rt_cipher, nBlocks, d_rt_iv, d_rt_tag_encrypt);
            CHECK_CUDA(cudaDeviceSynchronize());

            // Copy ciphertext from GPU to host for printing a sample
            CHECK_CUDA(cudaMemcpy(h_rt_cipher_gpu, d_rt_cipher, bytes, cudaMemcpyDeviceToHost));
            printf("  Ciphertext (GPU): ");
            for(int c_idx = 0; c_idx < std::min((size_t)16, bytes); ++c_idx) printf("%02x", h_rt_cipher_gpu[c_idx]);
            printf("...\n");

            // Perform Decryption
            if(isEcb && bits==128) aes128_ecb_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks);
            else if(isEcb && bits==256) aes256_ecb_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks);
            else if(isCbc && bits==128) aes128_cbc_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv);
            else if(isCbc && bits==256) aes256_cbc_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv);
            else if(isCfb && bits==128) aes128_cfb_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv);
            else if(isCfb && bits==256) aes256_cfb_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv);
            else if(isOfb && bits==128) aes128_ofb_decrypt<<<1,1>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv);
            else if(isOfb && bits==256) aes256_ofb_decrypt<<<1,1>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv);
            else if(isCtr && bits==128){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); aes128_ctr_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks,lo,hi); }
            else if(isCtr && bits==256){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); aes256_ctr_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks,lo,hi); }
            else if(isGcm && bits==128) aes128_gcm_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv, d_rt_tag_encrypt, d_rt_tag_decrypt_out);
            else if(isGcm && bits==256) aes256_gcm_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv, d_rt_tag_encrypt, d_rt_tag_decrypt_out);
            else if(isCcm && bits==128) aes128_ccm_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv, d_rt_tag_encrypt, d_rt_tag_decrypt_out);
            else if(isCcm && bits==256) aes256_ccm_decrypt<<<rt_kernel_grid_dim,rt_kernel_block_dim>>>(d_rt_cipher, d_rt_decrypted_final, nBlocks, d_rt_iv, d_rt_tag_encrypt, d_rt_tag_decrypt_out);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_rt_decrypted_gpu, d_rt_decrypted_final, bytes, cudaMemcpyDeviceToHost));

            bool tag_match = true;
            if (usesAuthTag) {
                uint8_t h_tag_encrypt[16];
                uint8_t h_tag_decrypt[16];
                CHECK_CUDA(cudaMemcpy(h_tag_encrypt, d_rt_tag_encrypt, sizeof(h_tag_encrypt), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_tag_decrypt, d_rt_tag_decrypt_out, sizeof(h_tag_decrypt), cudaMemcpyDeviceToHost));
                tag_match = (std::memcmp(h_tag_encrypt, h_tag_decrypt, sizeof(h_tag_encrypt)) == 0);
                if (!tag_match) {
                    printf("FAIL - AEAD authentication tag mismatch\n");
                }
            }

            bool match = true;
            for (size_t i = 0; i < bytes; ++i) {
                if (h_rt_original[i] != h_rt_decrypted_gpu[i]) {
                    match = false;
                    printf("FAIL - Mismatch at byte %zu: original %02x, decrypted %02x\n", i, h_rt_original[i], h_rt_decrypted_gpu[i]);
                    // Optionally print more context around mismatch
                    // size_t start_print = (i > 5) ? (i - 5) : 0;
                    // size_t end_print = (i + 5 < bytes) ? (i + 5) : bytes -1;
                    // printf("Original:  "); for(size_t k=start_print; k<=end_print; ++k) printf("%02x ", h_rt_original[k]); printf("\n");
                    // printf("Decrypted: "); for(size_t k=start_print; k<=end_print; ++k) printf("%02x ", h_rt_decrypted_gpu[k]); printf("\n");
                    break;
                }
            }
            if (match && tag_match) {
                printf("  Result:           PASS\n");
            } else {
                printf("  Result:           FAIL\n"); // Ensure FAIL is also followed by a newline and indented
            }

            CHECK_CUDA(cudaFreeHost(h_rt_original));
            CHECK_CUDA(cudaFreeHost(h_rt_decrypted_gpu));
            CHECK_CUDA(cudaFreeHost(h_rt_cipher_gpu)); // Free the new host buffer
            CHECK_CUDA(cudaFree(d_rt_plain));
            CHECK_CUDA(cudaFree(d_rt_cipher));
            CHECK_CUDA(cudaFree(d_rt_decrypted_final));
            if (d_rt_iv) CHECK_CUDA(cudaFree(d_rt_iv));
            if (d_rt_tag_encrypt) CHECK_CUDA(cudaFree(d_rt_tag_encrypt));
            if (d_rt_tag_decrypt_out) CHECK_CUDA(cudaFree(d_rt_tag_decrypt_out));
            // <<< END OF ROUND-TRIP CHECK >>>

            // Original benchmarking loop for configured run count
            for(int run=1; run<=bench_config.runs; ++run){
                if(run == 1)                   // ➋ ilk RESULT_GPU/CPU’dan önce
                    print_header();
                uint8_t *h_in,*h_out; CHECK_CUDA(cudaMallocHost(&h_in,bytes)); CHECK_CUDA(cudaMallocHost(&h_out,bytes));
                fill_random(h_in,bytes,rng);
                uint8_t *d_in,*d_out,*d_tag=nullptr,*d_iv=nullptr;
                CHECK_CUDA(cudaMalloc(&d_in,bytes)); CHECK_CUDA(cudaMalloc(&d_out,bytes));
                if(usesFeedbackIv || usesAuthTag) { CHECK_CUDA(cudaMalloc(&d_iv,iv.size())); CHECK_CUDA(cudaMemcpy(d_iv,iv.data(),iv.size(),cudaMemcpyHostToDevice)); }
                if(usesAuthTag) { CHECK_CUDA(cudaMalloc(&d_tag,16)); }
                CHECK_CUDA(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));
                dim3 block(blockOverride); dim3 grid((unsigned)((nBlocks+block.x-1)/block.x));

                // Create a descriptive NVTX range name for the entire benchmark iteration
                char nvtx_benchmark_range_name[128];
                snprintf(nvtx_benchmark_range_name, sizeof(nvtx_benchmark_range_name),
                         "%s-%d %s Run %d",
                         mode.c_str(), bits, decrypt ? "DEC" : "ENC", run);

                NVTX_PUSH(nvtx_benchmark_range_name); // Push NVTX range for the entire iteration

                cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
                cudaEventRecord(s);
                if(!decrypt){
                    if(isEcb && bits==128){ NVTX_PUSH("ECB-128 ENC kernel"); aes128_ecb_encrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isEcb && bits==256){ NVTX_PUSH("ECB-256 ENC kernel"); aes256_ecb_encrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isCbc && bits==128){ NVTX_PUSH("CBC-128 ENC kernel"); aes128_cbc_encrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCbc && bits==256){ NVTX_PUSH("CBC-256 ENC kernel"); aes256_cbc_encrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCfb && bits==128){ NVTX_PUSH("CFB-128 ENC kernel"); aes128_cfb_encrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCfb && bits==256){ NVTX_PUSH("CFB-256 ENC kernel"); aes256_cfb_encrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isOfb && bits==128){ NVTX_PUSH("OFB-128 ENC kernel"); aes128_ofb_encrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isOfb && bits==256){ NVTX_PUSH("OFB-256 ENC kernel"); aes256_ofb_encrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCtr && bits==128){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("CTR-128 ENC kernel"); aes128_ctr_encrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isCtr && bits==256){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("CTR-256 ENC kernel"); aes256_ctr_encrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isGcm && bits==128){ NVTX_PUSH("GCM-128 ENC kernel"); aes128_gcm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag); NVTX_POP(); }
                    else if(isGcm && bits==256){ NVTX_PUSH("GCM-256 ENC kernel"); aes256_gcm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag); NVTX_POP(); }
                    else if(isCcm && bits==128){ NVTX_PUSH("CCM-128 ENC kernel"); aes128_ccm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag); NVTX_POP(); }
                    else if(isCcm && bits==256){ NVTX_PUSH("CCM-256 ENC kernel"); aes256_ccm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag); NVTX_POP(); }
                } else {
                    if(isEcb && bits==128){ NVTX_PUSH("ECB-128 DEC kernel"); aes128_ecb_decrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isEcb && bits==256){ NVTX_PUSH("ECB-256 DEC kernel"); aes256_ecb_decrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isCbc && bits==128){ NVTX_PUSH("CBC-128 DEC kernel"); aes128_cbc_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCbc && bits==256){ NVTX_PUSH("CBC-256 DEC kernel"); aes256_cbc_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCfb && bits==128){ NVTX_PUSH("CFB-128 DEC kernel"); aes128_cfb_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCfb && bits==256){ NVTX_PUSH("CFB-256 DEC kernel"); aes256_cfb_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isOfb && bits==128){ NVTX_PUSH("OFB-128 DEC kernel"); aes128_ofb_decrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isOfb && bits==256){ NVTX_PUSH("OFB-256 DEC kernel"); aes256_ofb_decrypt<<<1,1>>>(d_in,d_out,nBlocks,d_iv); NVTX_POP(); }
                    else if(isCtr && bits==128){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("CTR-128 DEC kernel"); aes128_ctr_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isCtr && bits==256){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("CTR-256 DEC kernel"); aes256_ctr_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isGcm && bits==128){ NVTX_PUSH("GCM-128 DEC kernel"); aes128_gcm_decrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag,d_tag); NVTX_POP(); }
                    else if(isGcm && bits==256){ NVTX_PUSH("GCM-256 DEC kernel"); aes256_gcm_decrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag,d_tag); NVTX_POP(); }
                    else if(isCcm && bits==128){ NVTX_PUSH("CCM-128 DEC kernel"); aes128_ccm_decrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag,d_tag); NVTX_POP(); }
                    else if(isCcm && bits==256){ NVTX_PUSH("CCM-256 DEC kernel"); aes256_ccm_decrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag,d_tag); NVTX_POP(); }
                }
                cudaEventRecord(e); CHECK_CUDA(cudaEventSynchronize(e));
                NVTX_POP(); // Pop NVTX range for the entire iteration

                float ms=0.0f; cudaEventElapsedTime(&ms,s,e);
                CHECK_CUDA(cudaEventDestroy(s));
                CHECK_CUDA(cudaEventDestroy(e));
                double gib=(double)bytes/(double)(1ull<<30); double thr=gib/(ms/1000.0);
                printf("RESULT_GPU %-10s %-12zu %-5d %-10.3f %-10.3f %-5s\n", mode.c_str(), bytes, run, ms, thr, decrypt?"DEC":"ENC");
                append_benchmark_row(gpu_csv_path, run_id, "kernel_only", "GPU", mode, bytes, run, bench_config.runs, ms, thr, decrypt ? "DEC" : "ENC", bench_config.command_line);

                std::vector<uint8_t> host_in(bytes); CHECK_CUDA(cudaMemcpy(host_in.data(),d_in,bytes,cudaMemcpyDeviceToHost));
                const EVP_CIPHER* (*sel)();
                double cpu_thr = 0.0;
                if(isEcb&&bits==128) sel=&EVP_aes_128_ecb; else if(isEcb&&bits==256) sel=&EVP_aes_256_ecb;
                else if(isCbc&&bits==128) sel=&EVP_aes_128_cbc; else if(isCbc&&bits==256) sel=&EVP_aes_256_cbc;
                else if(isCfb&&bits==128) sel=&EVP_aes_128_cfb128; else if(isCfb&&bits==256) sel=&EVP_aes_256_cfb128;
                else if(isOfb&&bits==128) sel=&EVP_aes_128_ofb; else if(isOfb&&bits==256) sel=&EVP_aes_256_ofb;
                else if(isCtr&&bits==128) sel=&EVP_aes_128_ctr; else if(isCtr&&bits==256) sel=&EVP_aes_256_ctr;
                else if(isGcm&&bits==128) sel=&EVP_aes_128_gcm; else if(isGcm&&bits==256) sel=&EVP_aes_256_gcm;
                else sel=nullptr;
                if (isCcm) cpu_thr = cpu_ccm_throughput(host_in.data(), bytes, key.data(), bits, decrypt);
                else cpu_thr = cpu_aes_throughput(host_in.data(), bytes, key.data(), bits, decrypt, sel);
                double ms_cpu = (double)bytes/(cpu_thr*(1ull<<30))*1000.0;
                printf("RESULT_CPU %-10s %-12zu %-5d %-10.3f %-10.3f %-5s\n", mode.c_str(), bytes, run, ms_cpu, cpu_thr, decrypt?"DEC":"ENC");
                append_benchmark_row(cpu_csv_path, run_id, "cpu_baseline", "CPU", mode, bytes, run, bench_config.runs, ms_cpu, cpu_thr, decrypt ? "DEC" : "ENC", bench_config.command_line);

                CHECK_CUDA(cudaFreeHost(h_in)); CHECK_CUDA(cudaFreeHost(h_out));
                CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out)); if(d_tag) CHECK_CUDA(cudaFree(d_tag)); if(d_iv) CHECK_CUDA(cudaFree(d_iv));
            }
        }
    }
    return 0;
}

