#include "bench_driver.h"

#include <cstdlib>

#include "tflite/kernels/internal/optimized/fully_connected_4bit.h"
#include "tflite/kernels/internal/optimized/4bit/fully_connected_reference_impl.h"

namespace {

// LiteRT 4-bit NEON FC kernel (per-channel, optimized)

struct LiteRTInt4FcWeights {
    size_t K, N;
    uint8_t* prepacked = nullptr;
    uint8_t* ref_prepacked = nullptr;
    std::vector<float> filter_scales;
    int lhs_layout_rows;
    int lhs_layout_cols;
    std::vector<int32_t> dst_buf;
    std::vector<float> output_buf;

    ~LiteRTInt4FcWeights() { free(prepacked); free(ref_prepacked); }
};

struct LiteRTInt4FcActivations {
    std::vector<int8_t> rhs;
    std::vector<float> scales;
    std::vector<int32_t> input_offsets;
    std::vector<float> fp32;
    int rhs_width;
    int rhs_layout_rows;
    int rhs_layout_cols;
};

std::vector<int8_t> pack_litert_source(const std::vector<int8_t>& int4_rowmajor, size_t N, size_t K) {
    std::vector<int8_t> packed(N * K / 2);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; k += 2) {
            int8_t upper = int4_rowmajor[n * K + k];
            int8_t lower = int4_rowmajor[n * K + k + 1];
            uint8_t byte = (static_cast<uint8_t>(upper) << 4) | (static_cast<uint8_t>(lower) & 0x0F);
            packed[n * (K / 2) + k / 2] = static_cast<int8_t>(byte);
        }
    }
    return packed;
}

void* fc_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    std::vector<int8_t> int4_rowmajor;
    std::vector<float> filter_scales;
    bench::quantize_int4_per_channel(src, N, K, int4_rowmajor, filter_scales);

    auto litert_source = pack_litert_source(int4_rowmajor, N, K);

    auto* w = new LiteRTInt4FcWeights();
    w->K = K;
    w->N = N;
    w->filter_scales = filter_scales;

    w->lhs_layout_rows = (static_cast<int>(N) + tflite::optimized_4bit::FilterWidth - 1)
                         & ~(tflite::optimized_4bit::FilterWidth - 1);
    w->lhs_layout_cols = (static_cast<int>(K) + tflite::optimized_4bit::FilterDepth - 1)
                         & ~(tflite::optimized_4bit::FilterDepth - 1);

    size_t prepacked_size = static_cast<size_t>(w->lhs_layout_rows) * w->lhs_layout_cols / 2
                          + tflite::optimized_4bit::kDefaultAlignmentPadding;
    void* raw = nullptr;
    posix_memalign(&raw, 64, prepacked_size);
    w->prepacked = static_cast<uint8_t*>(raw);

    tflite::optimized_4bit::api::Prepack(
        w->prepacked, litert_source.data(),
        w->lhs_layout_rows, w->lhs_layout_cols,
        static_cast<int>(N), static_cast<int>(K),
        tflite::optimized_4bit::FilterWidth,
        tflite::optimized_4bit::FilterDepth);

    posix_memalign(reinterpret_cast<void**>(&raw), 64, prepacked_size);
    w->ref_prepacked = static_cast<uint8_t*>(raw);
    tflite::optimized_4bit::ReferencePrepack(
        w->ref_prepacked, litert_source.data(),
        w->lhs_layout_rows, w->lhs_layout_cols,
        static_cast<int>(N), static_cast<int>(K),
        tflite::optimized_4bit::FilterWidth,
        tflite::optimized_4bit::FilterDepth);

    return w;
}

void* fc_prepare_activations(const float* fp32, size_t M, size_t K, void* weights) {
    auto* w = static_cast<LiteRTInt4FcWeights*>(weights);
    auto* a = new LiteRTInt4FcActivations();

    a->rhs_width = std::min(static_cast<int>(M),
                            tflite::optimized_4bit::GetMaxSupportedRows());
    a->rhs_layout_rows = (static_cast<int>(M) + a->rhs_width - 1) & ~(a->rhs_width - 1);
    a->rhs_layout_cols = (static_cast<int>(K) + tflite::optimized_4bit::FilterDepth - 1)
                        & ~(tflite::optimized_4bit::FilterDepth - 1);

    a->rhs.resize(static_cast<size_t>(a->rhs_layout_rows) * a->rhs_layout_cols, 0);
    a->scales.resize(a->rhs_layout_rows, 1.0f);
    a->input_offsets.resize(a->rhs_layout_rows, 0);

    a->fp32.assign(fp32, fp32 + M * K);

    tflite::optimized_4bit::api::BatchQuantizeFloats4Bit(
        fp32, static_cast<int>(M), static_cast<int>(K),
        a->rhs.data(), a->scales.data(),
        a->rhs_width, tflite::optimized_4bit::FilterDepth,
        a->input_offsets.data());

    (void)w;
    return a;
}

bench::BenchResult fc_run(size_t M, void* weights, void* activations,
                          const int8_t*, const float*,
                          const bench::BenchOptions& opt) {
    auto* w = static_cast<LiteRTInt4FcWeights*>(weights);
    auto* a = static_cast<LiteRTInt4FcActivations*>(activations);

    const int output_depth = static_cast<int>(w->N);
    const int batch_size = static_cast<int>(M);
    const int dst_layout_rows = a->rhs_layout_rows;
    const int dst_layout_cols = w->lhs_layout_rows;

    std::vector<int32_t> dst(static_cast<size_t>(dst_layout_rows) * dst_layout_cols, 0);
    std::vector<float> output(M * w->N, 0.0f);
    std::vector<float> filter_scales = w->filter_scales;

    for (int i = 0; i < opt.warmup; ++i) {
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(dst.begin(), dst.end(), 0);
        tflite::optimized_4bit::api::AssignBiasAndComputeOffsets(
            a->input_offsets.data(), a->scales.data(),
            filter_scales.data(), nullptr, output.data(), output_depth, batch_size);
        tflite::optimized_4bit::api::RunAndUnpack(
            a->rhs_width, w->prepacked, a->rhs.data(),
            dst.data(), output_depth, batch_size,
            w->lhs_layout_rows, w->lhs_layout_cols,
            a->rhs_layout_rows, a->rhs_layout_cols,
            dst_layout_rows, dst_layout_cols,
            output.data(), a->scales.data(), filter_scales.data());
    }

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * w->N * sizeof(float));

    if (opt.capture_reference) {
        int ref_rw = 1;
        int ref_rhs_lr = (static_cast<int>(M) + ref_rw - 1) & ~(ref_rw - 1);
        int ref_dst_lr = ref_rhs_lr;

        std::vector<int8_t> ref_act(static_cast<size_t>(ref_rhs_lr) * a->rhs_layout_cols, 0);
        std::vector<float> ref_scales(ref_rhs_lr, 1.0f);
        std::vector<int32_t> ref_offsets(ref_rhs_lr, 0);
        tflite::optimized_4bit::ReferenceBatchQuantizeFloats4Bit(
            a->fp32.data(), static_cast<int>(M), static_cast<int>(w->K),
            ref_act.data(), ref_scales.data(), ref_rw,
            tflite::optimized_4bit::FilterDepth, ref_offsets.data());

        std::vector<int32_t> ref_dst_buf(static_cast<size_t>(ref_dst_lr) * dst_layout_cols, 0);
        std::vector<float> fs_ref = filter_scales;
        std::memset(opt.capture_reference, 0, M * w->N * sizeof(float));
        tflite::optimized_4bit::ReferenceAssignBiasAndComputeOffsets(
            ref_offsets.data(), ref_scales.data(), fs_ref.data(),
            nullptr, opt.capture_reference, output_depth, batch_size);
        tflite::optimized_4bit::ReferenceRunKernel<4, 1, 32>(
            w->ref_prepacked, ref_act.data(), ref_dst_buf.data(),
            w->lhs_layout_rows, w->lhs_layout_cols,
            ref_rhs_lr, a->rhs_layout_cols, ref_dst_lr, dst_layout_cols);
        tflite::optimized_4bit::ReferenceUnpack<4, 1>(
            opt.capture_reference, ref_dst_buf.data(), batch_size, output_depth,
            ref_scales.data(), fs_ref.data(), ref_dst_lr, dst_layout_cols);
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(dst.begin(), dst.end(), 0);
        double t0 = bench::now_ms();
        tflite::optimized_4bit::api::AssignBiasAndComputeOffsets(
            a->input_offsets.data(), a->scales.data(),
            filter_scales.data(), nullptr, output.data(), output_depth, batch_size);
        tflite::optimized_4bit::api::RunAndUnpack(
            a->rhs_width, w->prepacked, a->rhs.data(),
            dst.data(), output_depth, batch_size,
            w->lhs_layout_rows, w->lhs_layout_cols,
            a->rhs_layout_rows, a->rhs_layout_cols,
            dst_layout_rows, dst_layout_cols,
            output.data(), a->scales.data(), filter_scales.data());
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void fc_once(size_t M, void* weights, void* activations,
             const int8_t*, const float*) {
    auto* w = static_cast<LiteRTInt4FcWeights*>(weights);
    auto* a = static_cast<LiteRTInt4FcActivations*>(activations);

    const int output_depth = static_cast<int>(w->N);
    const int batch_size = static_cast<int>(M);
    const int dst_layout_rows = a->rhs_layout_rows;
    const int dst_layout_cols = w->lhs_layout_rows;

    const size_t dst_count = static_cast<size_t>(dst_layout_rows) * dst_layout_cols;
    w->dst_buf.resize(dst_count);
    std::memset(w->dst_buf.data(), 0, dst_count * sizeof(int32_t));
    w->output_buf.resize(M * w->N);
    std::memset(w->output_buf.data(), 0, M * w->N * sizeof(float));

    tflite::optimized_4bit::api::AssignBiasAndComputeOffsets(
        a->input_offsets.data(), a->scales.data(),
        w->filter_scales.data(), nullptr, w->output_buf.data(), output_depth, batch_size);
    tflite::optimized_4bit::api::RunAndUnpack(
        a->rhs_width, w->prepacked, a->rhs.data(),
        w->dst_buf.data(), output_depth, batch_size,
        w->lhs_layout_rows, w->lhs_layout_cols,
        a->rhs_layout_rows, a->rhs_layout_cols,
        dst_layout_rows, dst_layout_cols,
        w->output_buf.data(), a->scales.data(), w->filter_scales.data());
}

void fc_cleanup(void* weights, void* activations) {
    delete static_cast<LiteRTInt4FcWeights*>(weights);
    delete static_cast<LiteRTInt4FcActivations*>(activations);
}

// LiteRT blockwise reference (scalar C, per-group INT4)

struct LiteRTBlockwiseWeights {
    size_t K, N;
    std::vector<uint8_t> packed;
    std::vector<float> scales;
    std::vector<float> output_buf;
};

static inline int32_t SignExtendInt4(int8_t v) {
    return (v & 0x08) ? (v | 0xFFFFFFF0) : (v & 0x0F);
}

void* bw_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    std::vector<int8_t> int4_rowmajor;
    std::vector<float> scales;
    bench::quantize_int4_per_group(src, N, K, int4_rowmajor, scales);

    auto* w = new LiteRTBlockwiseWeights();
    w->K = K;
    w->N = N;
    w->scales = scales;

    const size_t k2 = (K + 1) & ~size_t(1);
    w->packed.resize(N * k2 / 2);
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; k += 2) {
            uint8_t lo = static_cast<uint8_t>(int4_rowmajor[n * K + k] + 8) & 0x0F;
            uint8_t hi = (static_cast<uint8_t>(int4_rowmajor[n * K + k + 1] + 8) & 0x0F) << 4;
            w->packed[n * (k2 / 2) + k / 2] = lo | hi;
        }

    return w;
}

bench::BenchResult bw_run(size_t M, void* weights, void*,
                          const int8_t* act_int8, const float* act_scales,
                          const bench::BenchOptions& opt) {
    auto* w = static_cast<LiteRTBlockwiseWeights*>(weights);
    const size_t N = w->N, K = w->K;
    const size_t num_groups = K / bench::kGroupSize;
    const size_t k2 = (K + 1) & ~size_t(1);
    std::vector<float> output(M * N, 0.0f);

    auto run_once = [&]() {
        std::memset(output.data(), 0, M * N * sizeof(float));
        for (size_t mi = 0; mi < M; mi++) {
            for (size_t ni = 0; ni < N; ni++) {
                for (size_t bi = 0; bi < num_groups; bi++) {
                    int32_t c_ref_acc = 0;
                    for (size_t ki = 0; ki < bench::kGroupSize; ki++) {
                        size_t k_index = bi * bench::kGroupSize + ki;
                        size_t nb_index = (ni * k2 + k_index) / 2;
                        uint8_t byte = w->packed[nb_index];
                        int8_t k_value_raw = static_cast<int8_t>(
                            (k_index % 2 == 0) ? (byte & 0x0F) : (byte >> 4));
                        int32_t kernel_value = SignExtendInt4(k_value_raw);
                        c_ref_acc += static_cast<int32_t>(act_int8[mi * K + k_index]) * kernel_value;
                    }
                    size_t scale_index = ni * num_groups + bi;
                    float scale = w->scales[scale_index];
                    output[mi * N + ni] += c_ref_acc * scale;
                }
                output[mi * N + ni] *= act_scales[mi];
            }
        }
    };

    for (int i = 0; i < opt.warmup; ++i) run_once();

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * N * sizeof(float));

    if (opt.capture_reference) {
        std::vector<float> deq_w(N * K);
        for (size_t ni = 0; ni < N; ni++) {
            for (size_t ki = 0; ki < K; ki++) {
                size_t gi = ki / bench::kGroupSize;
                size_t nb_index = (ni * k2 + ki) / 2;
                uint8_t byte = w->packed[nb_index];
                int8_t raw = static_cast<int8_t>((ki % 2 == 0) ? (byte & 0x0F) : (byte >> 4));
                deq_w[ni * K + ki] = static_cast<float>(SignExtendInt4(raw)) * w->scales[ni * num_groups + gi];
            }
        }

        std::vector<float> deq_a(M * K);
        for (size_t mi = 0; mi < M; mi++)
            for (size_t ki = 0; ki < K; ki++)
                deq_a[mi * K + ki] = static_cast<float>(act_int8[mi * K + ki]) * act_scales[mi];

        bench::reference_matmul_fp32(deq_a.data(), deq_w.data(), opt.capture_reference, M, K, N);
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        run_once();
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, K, N, opt.iterations, total_ms)};
}

void bw_once(size_t M, void* weights, void*,
             const int8_t* act_int8, const float* act_scales) {
    auto* w = static_cast<LiteRTBlockwiseWeights*>(weights);
    const size_t N = w->N, K = w->K;
    const size_t num_groups = K / bench::kGroupSize;
    const size_t k2 = (K + 1) & ~size_t(1);
    w->output_buf.resize(M * N);
    std::memset(w->output_buf.data(), 0, M * N * sizeof(float));

    for (size_t mi = 0; mi < M; mi++) {
        for (size_t ni = 0; ni < N; ni++) {
            for (size_t bi = 0; bi < num_groups; bi++) {
                int32_t c_ref_acc = 0;
                for (size_t ki = 0; ki < bench::kGroupSize; ki++) {
                    size_t k_index = bi * bench::kGroupSize + ki;
                    size_t nb_index = (ni * k2 + k_index) / 2;
                    uint8_t byte = w->packed[nb_index];
                    int8_t k_value_raw = static_cast<int8_t>(
                        (k_index % 2 == 0) ? (byte & 0x0F) : (byte >> 4));
                    int32_t kernel_value = SignExtendInt4(k_value_raw);
                    c_ref_acc += static_cast<int32_t>(act_int8[mi * K + k_index]) * kernel_value;
                }
                size_t scale_index = ni * num_groups + bi;
                w->output_buf[mi * N + ni] += c_ref_acc * w->scales[scale_index];
            }
            w->output_buf[mi * N + ni] *= act_scales[mi];
        }
    }
}

void bw_cleanup(void* weights, void*) {
    delete static_cast<LiteRTBlockwiseWeights*>(weights);
}

static int reg = [] {
    bench::register_backend({
        "litert_4bit_neon", "litert", bench::QuantCategory::INT4,
        fc_prepare, fc_prepare_activations, fc_run, fc_cleanup, 0, fc_once
    });
    // bench::register_backend({
    //     "litert_bw_ref", "litert", bench::QuantCategory::INT4,
    //     bw_prepare, nullptr, bw_run, bw_cleanup, 0, bw_once
    // });
    return 0;
}();

} // namespace
