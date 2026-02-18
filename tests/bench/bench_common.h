#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../cactus/cactus.h"

namespace bench {

constexpr size_t kGroupSize = 32;
constexpr size_t kBlockSize = 4;

struct ProjectionSpec {
    const char* name;
    size_t K;
    size_t N;
};

constexpr std::array<ProjectionSpec, 7> kProjectionSpecs = {{
    {"attn_q",   640, 1024},
    {"attn_k",   640,  256},
    {"attn_v",   640,  256},
    {"attn_o",  1024,  640},
    {"ffn_gate", 640, 2048},
    {"ffn_up",   640, 2048},
    {"ffn_down",2048,  640},
}};

struct BenchResult {
    double avg_us = 0.0;
    double gops = 0.0;
};

struct BenchOptions {
    int warmup = 100;
    int iterations = 1000;
    int num_threads = 0; // 0 = all available
    int layers = 0;    // >0 enables stack mode (layer-cycling benchmark)
    std::vector<size_t> batch_sizes = {1, 13, 34};
    std::string backends_filter;
    std::string mode;  // "comparison", "stack", or "both" (empty = auto)
    float* capture_output = nullptr;    // bench_fn copies fp32 kernel output here after warmup
    float* capture_reference = nullptr; // bench_fn writes its own scalar reference here (optional)
};

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline double compute_gops(size_t M, size_t K, size_t N, int iters, double total_ms) {
    if (total_ms <= 0.0) return 0.0;
    return (2.0 * M * K * N * iters) / (total_ms * 1e6);
}

// INT8 per-group quantization [N, K] -> int8 weights + float scales [N, K/group_size]
void quantize_int8_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales);

// INT4 per-group quantization [N, K] -> int4 weights (stored as int8) + float scales
void quantize_int4_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales);

// INT4 per-channel quantization [N, K] -> int4 weights + one scale per output row
void quantize_int4_per_channel(const std::vector<float>& src, size_t N, size_t K,
                                std::vector<int8_t>& dst, std::vector<float>& scales);

// Interleave weights into N-K 4x4 block layout for Cactus SIMD kernels
std::vector<int8_t> interleave_weights_nk4(const std::vector<int8_t>& rowmajor, size_t N, size_t K);

// Interleave per-group scales to match 4x4 block layout
std::vector<__fp16> interleave_scales_n4(const std::vector<float>& scales, size_t N, size_t num_groups);

// Pack signed INT4 pairs into nibble-packed uint8 format
std::vector<uint8_t> pack_int4_pairs(const std::vector<int8_t>& interleaved);

// Standard Cactus activation data: int8 quantized with per-row scales
struct CactusActivations {
    std::vector<int8_t> int8;
    std::vector<float> scales;
    std::vector<__fp16> fp16;
    std::vector<float> fp32;
};

CactusActivations prepare_cactus_activations(size_t M, size_t K, std::mt19937& gen);

// Naive fp32 reference matmul: C[M,N] = A[M,K] * B[N,K]^T
void reference_matmul_fp32(const float* A, const float* B_rowmajor_NK,
                            float* C, size_t M, size_t K, size_t N);

struct AccuracyResult {
    float max_abs_error = 0.0f;
    float nrmse = 0.0f;
    bool passed = false;
};

AccuracyResult check_accuracy(const float* reference, const float* actual,
                               size_t count, float nrmse_tolerance);

// CLI argument parsing
bool parse_bench_args(int argc, char** argv, BenchOptions& opt, std::string& err);

} // namespace bench

#endif
