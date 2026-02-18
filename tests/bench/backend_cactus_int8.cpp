#include "bench_driver.h"

namespace {

struct CactusInt8Weights {
    size_t K, N;
    std::vector<int8_t> interleaved;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;
};

void* prepare(const float* fp32, size_t N, size_t K) {
    std::vector<float> src(fp32, fp32 + N * K);

    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;
    bench::quantize_int8_per_group(src, N, K, rowmajor, raw_scales);

    auto* w = new CactusInt8Weights();
    w->K = K;
    w->N = N;
    w->interleaved = bench::interleave_weights_nk4(rowmajor, N, K);
    w->scales = bench::interleave_scales_n4(raw_scales, N, K / bench::kGroupSize);
    return w;
}

bench::BenchResult run(size_t M, void* weights, void*,
                       const int8_t* act_int8, const float* act_scales,
                       const bench::BenchOptions& opt) {
    auto* w = static_cast<CactusInt8Weights*>(weights);
    std::vector<__fp16> output(M * w->N);

    CactusThreading::set_gemm_threads(opt.num_threads);

    for (int i = 0; i < opt.warmup; ++i)
        cactus_matmul_int8(act_int8, act_scales,
                           w->interleaved.data(), w->scales.data(),
                           output.data(), M, w->K, w->N, bench::kGroupSize);

    if (opt.capture_output)
        for (size_t i = 0; i < M * w->N; i++)
            opt.capture_output[i] = static_cast<float>(output[i]);

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        cactus_matmul_int8(act_int8, act_scales,
                           w->interleaved.data(), w->scales.data(),
                           output.data(), M, w->K, w->N, bench::kGroupSize);
        total_ms += bench::now_ms() - t0;
    }

    CactusThreading::reset_gemm_threads();

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void once(size_t M, void* weights, void*,
          const int8_t* act_int8, const float* act_scales) {
    auto* w = static_cast<CactusInt8Weights*>(weights);
    w->output_buf.resize(M * w->N);
    cactus_matmul_int8(act_int8, act_scales,
                       w->interleaved.data(), w->scales.data(),
                       w->output_buf.data(), M, w->K, w->N, bench::kGroupSize);
}

void cleanup(void* weights, void*) {
    delete static_cast<CactusInt8Weights*>(weights);
}

static int reg = [] {
    bench::register_backend({
        "cactus_int8", "cactus", bench::QuantCategory::INT8,
        prepare, nullptr, run, cleanup, 0, once
    });
    return 0;
}();

} // namespace
