#include "bench_driver.h"

#include "tflite/kernels/internal/optimized/neon_tensor_utils.h"
#include "ruy/ruy.h"

#include <memory>
#include <thread>

namespace {

// LiteRT NEON kernel (single-threaded LSTM-path GEMV)

struct LiteRTNeonWeights {
    size_t K, N;
    std::vector<int8_t> int8_rowmajor;
    std::vector<float> output_buf;
};

void* neon_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    auto* w = new LiteRTNeonWeights();
    w->K = K;
    w->N = N;
    w->int8_rowmajor.resize(N * K);

    std::vector<float> scales;
    bench::quantize_int8_per_group(src, N, K, w->int8_rowmajor, scales);
    return w;
}

bench::BenchResult neon_run(size_t M, void* weights, void*,
                            const int8_t* act_int8, const float* act_scales,
                            const bench::BenchOptions& opt) {
    auto* w = static_cast<LiteRTNeonWeights*>(weights);
    std::vector<float> output(M * w->N);
    const int m_rows = static_cast<int>(w->N);
    const int m_cols = static_cast<int>(w->K);
    const int n_batch = static_cast<int>(M);

    for (int i = 0; i < opt.warmup; ++i) {
        std::fill(output.begin(), output.end(), 0.0f);
        tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            w->int8_rowmajor.data(), m_rows, m_cols,
            act_int8, act_scales, n_batch, output.data());
    }

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * w->N * sizeof(float));

    if (opt.capture_reference) {
        const size_t K = w->K, N = w->N;
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                int32_t dot = 0;
                for (size_t k = 0; k < K; k++)
                    dot += static_cast<int32_t>(w->int8_rowmajor[n * K + k])
                         * static_cast<int32_t>(act_int8[m * K + k]);
                opt.capture_reference[m * N + n] = static_cast<float>(dot) * act_scales[m];
            }
        }
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        std::fill(output.begin(), output.end(), 0.0f);
        double t0 = bench::now_ms();
        tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            w->int8_rowmajor.data(), m_rows, m_cols,
            act_int8, act_scales, n_batch, output.data());
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void neon_once(size_t M, void* weights, void*,
               const int8_t* act_int8, const float* act_scales) {
    auto* w = static_cast<LiteRTNeonWeights*>(weights);
    w->output_buf.resize(M * w->N);
    std::memset(w->output_buf.data(), 0, M * w->N * sizeof(float));
    tflite::tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        w->int8_rowmajor.data(), static_cast<int>(w->N), static_cast<int>(w->K),
        act_int8, act_scales, static_cast<int>(M), w->output_buf.data());
}

void neon_cleanup(void* weights, void*) {
    delete static_cast<LiteRTNeonWeights*>(weights);
}

// Ruy GEMM (LiteRT's production int8 matmul engine)

struct RuyWeights {
    size_t K, N;
    std::vector<int8_t> int8_rowmajor;
    std::vector<int32_t> output_buf;
    std::unique_ptr<ruy::Context> ctx_mc;
    std::unique_ptr<ruy::Context> ctx_1c;
};

void* ruy_prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    auto* w = new RuyWeights();
    w->K = K;
    w->N = N;
    w->int8_rowmajor.resize(N * K);

    std::vector<float> scales;
    bench::quantize_int8_per_group(src, N, K, w->int8_rowmajor, scales);
    return w;
}

static bench::BenchResult ruy_run_impl(size_t M, RuyWeights* w,
                                        const int8_t* act_int8,
                                        const bench::BenchOptions& opt,
                                        int max_threads) {
    ruy::Context ctx;
    ctx.set_max_num_threads(max_threads);

    std::vector<int32_t> output(M * w->N);

    ruy::Matrix<int8_t> lhs;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->K),
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(act_int8);
    lhs.set_zero_point(0);

    ruy::Matrix<int8_t> rhs;
    ruy::MakeSimpleLayout(static_cast<int>(w->K), static_cast<int>(w->N),
                          ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(w->int8_rowmajor.data());
    rhs.set_zero_point(0);
    rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ruy::Matrix<int32_t> dst;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->N),
                          ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(output.data());

    ruy::MulParams<int32_t, int32_t> mul_params;

    for (int i = 0; i < opt.warmup; ++i)
        ruy::Mul(lhs, rhs, mul_params, &ctx, &dst);

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        ruy::Mul(lhs, rhs, mul_params, &ctx, &dst);
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

bench::BenchResult ruy_mc_run(size_t M, void* weights, void*,
                              const int8_t* act_int8, const float*,
                              const bench::BenchOptions& opt) {
    return ruy_run_impl(M, static_cast<RuyWeights*>(weights), act_int8, opt, opt.num_threads);
}

bench::BenchResult ruy_1c_run(size_t M, void* weights, void*,
                              const int8_t* act_int8, const float*,
                              const bench::BenchOptions& opt) {
    return ruy_run_impl(M, static_cast<RuyWeights*>(weights), act_int8, opt, 1);
}

static void ruy_once_impl(size_t M, RuyWeights* w,
                          const int8_t* act_int8, ruy::Context* ctx) {
    w->output_buf.resize(M * w->N);

    ruy::Matrix<int8_t> lhs;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->K),
                          ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(act_int8);
    lhs.set_zero_point(0);

    ruy::Matrix<int8_t> rhs;
    ruy::MakeSimpleLayout(static_cast<int>(w->K), static_cast<int>(w->N),
                          ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(w->int8_rowmajor.data());
    rhs.set_zero_point(0);
    rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

    ruy::Matrix<int32_t> dst;
    ruy::MakeSimpleLayout(static_cast<int>(M), static_cast<int>(w->N),
                          ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(w->output_buf.data());

    ruy::MulParams<int32_t, int32_t> mul_params;
    ruy::Mul(lhs, rhs, mul_params, ctx, &dst);
}

void ruy_mc_once(size_t M, void* weights, void*,
                 const int8_t* act_int8, const float*) {
    auto* w = static_cast<RuyWeights*>(weights);
    if (!w->ctx_mc) {
        w->ctx_mc = std::make_unique<ruy::Context>();
        w->ctx_mc->set_max_num_threads(static_cast<int>(std::thread::hardware_concurrency()));
    }
    ruy_once_impl(M, w, act_int8, w->ctx_mc.get());
}

void ruy_1c_once(size_t M, void* weights, void*,
                 const int8_t* act_int8, const float*) {
    auto* w = static_cast<RuyWeights*>(weights);
    if (!w->ctx_1c) {
        w->ctx_1c = std::make_unique<ruy::Context>();
        w->ctx_1c->set_max_num_threads(1);
    }
    ruy_once_impl(M, w, act_int8, w->ctx_1c.get());
}

void ruy_cleanup(void* weights, void*) {
    delete static_cast<RuyWeights*>(weights);
}

static int reg = [] {
    bench::register_backend({
        "litert_neon", "litert", bench::QuantCategory::INT8,
        neon_prepare, nullptr, neon_run, neon_cleanup, 0, neon_once
    });
    bench::register_backend({
        "ruy_mc", "litert", bench::QuantCategory::INT8,
        ruy_prepare, nullptr, ruy_mc_run, ruy_cleanup, 0, ruy_mc_once
    });
    bench::register_backend({
        "ruy_1c", "litert", bench::QuantCategory::INT8,
        ruy_prepare, nullptr, ruy_1c_run, ruy_cleanup, 0, ruy_1c_once
    });
    return 0;
}();

} // namespace
