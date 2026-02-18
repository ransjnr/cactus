#include "bench_driver.h"

#include <arm_neon.h>
#include <cstdio>

namespace {

// Shared weight structures for INT4 variants

struct CactusInt4PerGroupWeights {
    size_t K, N;
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;
};

struct CactusInt4PerChannelWeights {
    size_t K, N;
    std::vector<uint8_t> packed;
    std::vector<float> scales;
    std::vector<float> output_buf;
};

// Per-group INT4

void* pg_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;
    bench::quantize_int4_per_group(src, N, K, rowmajor, raw_scales);

    auto interleaved = bench::interleave_weights_nk4(rowmajor, N, K);

    auto* w = new CactusInt4PerGroupWeights();
    w->K = K;
    w->N = N;
    w->packed = bench::pack_int4_pairs(interleaved);
    w->scales = bench::interleave_scales_n4(raw_scales, N, K / bench::kGroupSize);
    return w;
}

bench::BenchResult pg_run(size_t M, void* weights, void*,
                          const int8_t* act_int8, const float* act_scales,
                          const bench::BenchOptions& opt) {
    auto* w = static_cast<CactusInt4PerGroupWeights*>(weights);
    std::vector<__fp16> output(M * w->N);

    CactusThreading::set_gemm_threads(opt.num_threads);

    for (int i = 0; i < opt.warmup; ++i)
        cactus_matmul_int4(act_int8, act_scales,
                           reinterpret_cast<const int8_t*>(w->packed.data()),
                           w->scales.data(),
                           output.data(), M, w->K, w->N, bench::kGroupSize);

    if (opt.capture_output)
        for (size_t i = 0; i < M * w->N; i++)
            opt.capture_output[i] = static_cast<float>(output[i]);

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        cactus_matmul_int4(act_int8, act_scales,
                           reinterpret_cast<const int8_t*>(w->packed.data()),
                           w->scales.data(),
                           output.data(), M, w->K, w->N, bench::kGroupSize);
        total_ms += bench::now_ms() - t0;
    }

    CactusThreading::reset_gemm_threads();

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void pg_once(size_t M, void* weights, void*,
             const int8_t* act_int8, const float* act_scales) {
    auto* w = static_cast<CactusInt4PerGroupWeights*>(weights);
    w->output_buf.resize(M * w->N);
    cactus_matmul_int4(act_int8, act_scales,
                       reinterpret_cast<const int8_t*>(w->packed.data()),
                       w->scales.data(),
                       w->output_buf.data(), M, w->K, w->N, bench::kGroupSize);
}

void pg_cleanup(void* weights, void*) {
    delete static_cast<CactusInt4PerGroupWeights*>(weights);
}

// Per-channel INT4 (inline SDOT kernel)

static inline void int4_unpack_signed(const uint8_t* ptr, int8x16_t& hi, int8x16_t& lo) {
    uint8x16_t packed = vld1q_u8(ptr);
    hi = vreinterpretq_s8_u8(vsubq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(8)));
    lo = vreinterpretq_s8_u8(vsubq_u8(vandq_u8(packed, vdupq_n_u8(0x0F)), vdupq_n_u8(8)));
}

#define PC_DOT_GROUP(b_ptr, acc, a_lo, a_hi)                              \
    {                                                                      \
        int8x16_t lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;                 \
        int4_unpack_signed(b_ptr, hi0, lo0);                               \
        int4_unpack_signed(b_ptr + 16, hi1, lo1);                          \
        acc = vdotq_laneq_s32(acc, lo0, a_lo, 0);                         \
        acc = vdotq_laneq_s32(acc, hi0, a_lo, 1);                         \
        acc = vdotq_laneq_s32(acc, lo1, a_lo, 2);                         \
        acc = vdotq_laneq_s32(acc, hi1, a_lo, 3);                         \
        int4_unpack_signed(b_ptr + 32, hi2, lo2);                          \
        int4_unpack_signed(b_ptr + 48, hi3, lo3);                          \
        acc = vdotq_laneq_s32(acc, lo2, a_hi, 0);                         \
        acc = vdotq_laneq_s32(acc, hi2, a_hi, 1);                         \
        acc = vdotq_laneq_s32(acc, lo3, a_hi, 2);                         \
        acc = vdotq_laneq_s32(acc, hi3, a_hi, 3);                         \
    }

void* pc_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;
    bench::quantize_int4_per_channel(src, N, K, rowmajor, raw_scales);

    auto interleaved = bench::interleave_weights_nk4(rowmajor, N, K);

    auto* w = new CactusInt4PerChannelWeights();
    w->K = K;
    w->N = N;
    w->packed = bench::pack_int4_pairs(interleaved);

    const size_t N_blocks = N / 4;
    w->scales.resize(N_blocks * 4, 0.0f);
    for (size_t n = 0; n < N; n++)
        w->scales[n] = raw_scales[n];
    return w;
}

bench::BenchResult pc_run(size_t M, void* weights, void*,
                          const int8_t* act_int8, const float* act_scales,
                          const bench::BenchOptions& opt) {
    if (M != 1) {
        fprintf(stderr, "[cactus_int4_pc] skipping M=%zu (only M=1 supported)\n", M);
        return {};
    }
    auto* w = static_cast<CactusInt4PerChannelWeights*>(weights);
    const size_t K = w->K, N = w->N;
    const size_t N_blocks = (N + 3) / 4;
    std::vector<float> output(M * N, 0.0f);

    const uint8_t* B = w->packed.data();
    const float* scales = w->scales.data();
    const int8_t* A = act_int8;
    const float A_scale = act_scales[0];

    auto run_once = [&]() {
        size_t nb = 0;
        for (; nb + 1 < N_blocks; nb += 2) {
            int32x4_t sum_a = vdupq_n_s32(0);
            int32x4_t sum_b = vdupq_n_s32(0);

            for (size_t k = 0; k < K; k += 32) {
                int8x16_t a_lo = vld1q_s8(A + k);
                int8x16_t a_hi = vld1q_s8(A + k + 16);
                PC_DOT_GROUP(B + (nb * K + k) * 2, sum_a, a_lo, a_hi)
                PC_DOT_GROUP(B + ((nb + 1) * K + k) * 2, sum_b, a_lo, a_hi)
            }

            float32x4_t sa = vld1q_f32(scales + nb * 4);
            float32x4_t sb = vld1q_f32(scales + (nb + 1) * 4);
            vst1q_f32(output.data() + nb * 4,
                      vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(sum_a), sa), A_scale));
            vst1q_f32(output.data() + (nb + 1) * 4,
                      vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(sum_b), sb), A_scale));
        }
        for (; nb < N_blocks; nb++) {
            int32x4_t acc = vdupq_n_s32(0);
            for (size_t k = 0; k < K; k += 32) {
                int8x16_t a_lo = vld1q_s8(A + k);
                int8x16_t a_hi = vld1q_s8(A + k + 16);
                PC_DOT_GROUP(B + (nb * K + k) * 2, acc, a_lo, a_hi)
            }
            float32x4_t s = vld1q_f32(scales + nb * 4);
            float32x4_t result = vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(acc), s), A_scale);
            size_t actual_n = std::min(size_t(4), N - nb * 4);
            if (actual_n == 4) {
                vst1q_f32(output.data() + nb * 4, result);
            } else {
                float tmp[4];
                vst1q_f32(tmp, result);
                for (size_t ni = 0; ni < actual_n; ni++)
                    output[nb * 4 + ni] = tmp[ni];
            }
        }
    };

    for (int i = 0; i < opt.warmup; ++i) run_once();

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * N * sizeof(float));

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        run_once();
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, K, N, opt.iterations, total_ms)};
}

void pc_once(size_t M, void* weights, void*,
             const int8_t* act_int8, const float* act_scales) {
    if (M != 1) return;
    auto* w = static_cast<CactusInt4PerChannelWeights*>(weights);
    const size_t K = w->K, N = w->N;
    const size_t N_blocks = (N + 3) / 4;
    w->output_buf.resize(M * N, 0.0f);

    const uint8_t* B = w->packed.data();
    const float* wscales = w->scales.data();
    const int8_t* A = act_int8;
    const float A_scale = act_scales[0];

    size_t nb = 0;
    for (; nb + 1 < N_blocks; nb += 2) {
        int32x4_t sum_a = vdupq_n_s32(0);
        int32x4_t sum_b = vdupq_n_s32(0);
        for (size_t k = 0; k < K; k += 32) {
            int8x16_t a_lo = vld1q_s8(A + k);
            int8x16_t a_hi = vld1q_s8(A + k + 16);
            PC_DOT_GROUP(B + (nb * K + k) * 2, sum_a, a_lo, a_hi)
            PC_DOT_GROUP(B + ((nb + 1) * K + k) * 2, sum_b, a_lo, a_hi)
        }
        float32x4_t sa = vld1q_f32(wscales + nb * 4);
        float32x4_t sb = vld1q_f32(wscales + (nb + 1) * 4);
        vst1q_f32(w->output_buf.data() + nb * 4,
                  vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(sum_a), sa), A_scale));
        vst1q_f32(w->output_buf.data() + (nb + 1) * 4,
                  vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(sum_b), sb), A_scale));
    }
    for (; nb < N_blocks; nb++) {
        int32x4_t acc = vdupq_n_s32(0);
        for (size_t k = 0; k < K; k += 32) {
            int8x16_t a_lo = vld1q_s8(A + k);
            int8x16_t a_hi = vld1q_s8(A + k + 16);
            PC_DOT_GROUP(B + (nb * K + k) * 2, acc, a_lo, a_hi)
        }
        float32x4_t s = vld1q_f32(wscales + nb * 4);
        float32x4_t result = vmulq_n_f32(vmulq_f32(vcvtq_f32_s32(acc), s), A_scale);
        size_t actual_n = std::min(size_t(4), N - nb * 4);
        if (actual_n == 4) {
            vst1q_f32(w->output_buf.data() + nb * 4, result);
        } else {
            float tmp[4];
            vst1q_f32(tmp, result);
            for (size_t ni = 0; ni < actual_n; ni++)
                w->output_buf[nb * 4 + ni] = tmp[ni];
        }
    }
}

#undef PC_DOT_GROUP

void pc_cleanup(void* weights, void*) {
    delete static_cast<CactusInt4PerChannelWeights*>(weights);
}

// Registration

static int reg = [] {
    bench::register_backend({
        "cactus_int4", "cactus", bench::QuantCategory::INT4,
        pg_prepare, nullptr, pg_run, pg_cleanup, 0, pg_once
    });
    bench::register_backend({
        "cactus_int4_pc", "cactus", bench::QuantCategory::INT4,
        pc_prepare, nullptr, pc_run, pc_cleanup, 1, pc_once
    });
    return 0;
}();

} // namespace
