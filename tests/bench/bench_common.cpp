#include "bench_common.h"

#include <algorithm>
#include <cmath>

namespace bench {

void quantize_int8_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales) {
    const size_t num_groups = K / kGroupSize;
    dst.resize(N * K);
    scales.resize(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            const size_t base = n * K + g * kGroupSize;
            for (size_t k = 0; k < kGroupSize; ++k)
                max_abs = std::max(max_abs, std::abs(src[base + k]));
            float scale = std::max(max_abs / 127.0f, 1e-10f);
            scales[n * num_groups + g] = scale;
            for (size_t k = 0; k < kGroupSize; ++k) {
                int q = static_cast<int>(std::round(src[base + k] / scale));
                dst[base + k] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
            }
        }
    }
}

void quantize_int4_per_group(const std::vector<float>& src, size_t N, size_t K,
                              std::vector<int8_t>& dst, std::vector<float>& scales) {
    const size_t num_groups = K / kGroupSize;
    dst.resize(N * K);
    scales.resize(N * num_groups);
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_groups; ++g) {
            float max_abs = 0.0f;
            const size_t base = n * K + g * kGroupSize;
            for (size_t k = 0; k < kGroupSize; ++k)
                max_abs = std::max(max_abs, std::abs(src[base + k]));
            float scale = std::max(max_abs / 7.0f, 1e-10f);
            scales[n * num_groups + g] = scale;
            for (size_t k = 0; k < kGroupSize; ++k) {
                int q = static_cast<int>(std::round(src[base + k] / scale));
                dst[base + k] = static_cast<int8_t>(std::max(-8, std::min(7, q)));
            }
        }
    }
}

void quantize_int4_per_channel(const std::vector<float>& src, size_t N, size_t K,
                                std::vector<int8_t>& dst, std::vector<float>& scales) {
    dst.resize(N * K);
    scales.resize(N);
    for (size_t n = 0; n < N; ++n) {
        float max_abs = 0.0f;
        for (size_t k = 0; k < K; ++k)
            max_abs = std::max(max_abs, std::abs(src[n * K + k]));
        float scale = std::max(max_abs / 7.0f, 1e-10f);
        scales[n] = scale;
        for (size_t k = 0; k < K; ++k) {
            int q = static_cast<int>(std::round(src[n * K + k] / scale));
            dst[n * K + k] = static_cast<int8_t>(std::max(-7, std::min(7, q)));
        }
    }
}

std::vector<int8_t> interleave_weights_nk4(const std::vector<int8_t>& rowmajor, size_t N, size_t K) {
    const size_t N_blocks = N / kBlockSize;
    const size_t K_blocks = K / kBlockSize;
    std::vector<int8_t> out(N * K);
    for (size_t nb = 0; nb < N_blocks; ++nb)
        for (size_t kb = 0; kb < K_blocks; ++kb)
            for (size_t ni = 0; ni < kBlockSize; ++ni)
                for (size_t ki = 0; ki < kBlockSize; ++ki)
                    out[(nb * K_blocks + kb) * kBlockSize * kBlockSize + ni * kBlockSize + ki] =
                        rowmajor[(nb * kBlockSize + ni) * K + kb * kBlockSize + ki];
    return out;
}

std::vector<__fp16> interleave_scales_n4(const std::vector<float>& scales, size_t N, size_t num_groups) {
    const size_t N_blocks = N / kBlockSize;
    std::vector<__fp16> out(N * num_groups);
    for (size_t nb = 0; nb < N_blocks; ++nb)
        for (size_t g = 0; g < num_groups; ++g)
            for (size_t ni = 0; ni < kBlockSize; ++ni)
                out[(nb * num_groups + g) * kBlockSize + ni] =
                    static_cast<__fp16>(scales[(nb * kBlockSize + ni) * num_groups + g]);
    return out;
}

std::vector<uint8_t> pack_int4_pairs(const std::vector<int8_t>& interleaved) {
    std::vector<uint8_t> packed(interleaved.size() / 2);
    for (size_t i = 0; i < interleaved.size(); i += 32) {
        for (size_t j = 0; j < 16; ++j) {
            const uint8_t lo = static_cast<uint8_t>((static_cast<int16_t>(interleaved[i + j]) + 8) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>(((static_cast<int16_t>(interleaved[i + 16 + j]) + 8) & 0x0F) << 4);
            packed[i / 2 + j] = lo | hi;
        }
    }
    return packed;
}

CactusActivations prepare_cactus_activations(size_t M, size_t K, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    CactusActivations a;

    a.fp32.resize(M * K);
    for (auto& v : a.fp32) v = dist(gen);

    a.fp16.resize(M * K);
    for (size_t i = 0; i < M * K; ++i) a.fp16[i] = static_cast<__fp16>(a.fp32[i]);

    a.int8.resize(M * K);
    a.scales.resize(M);
    for (size_t m = 0; m < M; ++m) {
        float max_abs = cactus_fp16_max_abs(a.fp16.data() + m * K, K);
        float scale = std::max(max_abs / 127.0f, 1e-10f);
        a.scales[m] = scale;
        cactus_fp16_to_int8(a.fp16.data() + m * K, a.int8.data() + m * K, K, scale);
    }
    return a;
}

void reference_matmul_fp32(const float* A, const float* B_rowmajor_NK,
                            float* C, size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; m++)
        for (size_t n = 0; n < N; n++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++)
                sum += static_cast<double>(A[m * K + k]) * static_cast<double>(B_rowmajor_NK[n * K + k]);
            C[m * N + n] = static_cast<float>(sum);
        }
}

AccuracyResult check_accuracy(const float* reference, const float* actual,
                               size_t count, float nrmse_tolerance) {
    AccuracyResult r;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    for (size_t i = 0; i < count; i++) {
        float err = std::abs(reference[i] - actual[i]);
        if (err > r.max_abs_error) r.max_abs_error = err;
        sum_sq_err += static_cast<double>(err) * err;
        sum_sq_ref += static_cast<double>(reference[i]) * reference[i];
    }
    float rms_ref = static_cast<float>(std::sqrt(sum_sq_ref / std::max(count, size_t(1))));
    float rms_err = static_cast<float>(std::sqrt(sum_sq_err / std::max(count, size_t(1))));
    r.nrmse = (rms_ref > 1e-6f) ? rms_err / rms_ref : rms_err;
    r.passed = r.nrmse <= nrmse_tolerance;
    return r;
}

bool parse_bench_args(int argc, char** argv, BenchOptions& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--iterations") {
            if (++i >= argc) { err = "Missing --iterations value"; return false; }
            opt.iterations = std::max(1, std::stoi(argv[i]));
        } else if (a == "--warmup") {
            if (++i >= argc) { err = "Missing --warmup value"; return false; }
            opt.warmup = std::max(0, std::stoi(argv[i]));
        } else if (a == "--m") {
            if (++i >= argc) { err = "Missing --m value"; return false; }
            opt.batch_sizes = {static_cast<size_t>(std::max(1, std::stoi(argv[i])))};
        } else if (a == "--threads") {
            if (++i >= argc) { err = "Missing --threads value"; return false; }
            opt.num_threads = std::max(1, std::stoi(argv[i]));
        } else if (a == "--backends") {
            if (++i >= argc) { err = "Missing --backends value"; return false; }
            opt.backends_filter = argv[i];
        } else if (a == "--layers") {
            if (++i >= argc) { err = "Missing --layers value"; return false; }
            opt.layers = std::max(1, std::stoi(argv[i]));
        } else if (a == "--mode") {
            if (++i >= argc) { err = "Missing --mode value"; return false; }
            opt.mode = argv[i];
            if (opt.mode != "comparison" && opt.mode != "stack" && opt.mode != "both") {
                err = "Invalid --mode, expected comparison|stack|both";
                return false;
            }
        } else {
            err = "Unknown argument: " + a;
            return false;
        }
    }
    return true;
}

} // namespace bench
