#include "bench_driver.h"

#ifndef WITH_MLX

namespace {
[[maybe_unused]] static int reg = [] {
    return 0;
}();
} // namespace

#else

#include <mlx/mlx.h>
#include <cstring>

namespace mx = mlx::core;

namespace {

struct MLXWeights {
    size_t K, N;
    int bits;
    mx::array w_q;
    mx::array scales;
    mx::array biases;
};

struct MLXActivations {
    mx::array x;
};

void* prepare_impl(const float* fp32, size_t N, size_t K, int bits) {
    auto w = mx::array(fp32, {static_cast<int>(N), static_cast<int>(K)}, mx::float32);
    auto parts = mx::quantize(w, static_cast<int>(bench::kGroupSize), bits);
    mx::eval(parts);
    return new MLXWeights{K, N, bits,
        std::move(parts[0]), std::move(parts[1]), std::move(parts[2])};
}

void* prepare_q4(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, 4); }
void* prepare_q8(const float* fp32, size_t N, size_t K) { return prepare_impl(fp32, N, K, 8); }

void* prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto x = mx::astype(
        mx::array(fp32, {static_cast<int>(M), static_cast<int>(K)}, mx::float32),
        mx::float16);
    mx::eval(x);
    return new MLXActivations{std::move(x)};
}

bench::BenchResult run(size_t M, void* weights, void* activations,
                       const int8_t*, const float*,
                       const bench::BenchOptions& opt) {
    auto* w = static_cast<MLXWeights*>(weights);
    auto* a = static_cast<MLXActivations*>(activations);
    const int gs = static_cast<int>(bench::kGroupSize);

    auto qmatmul = [&]() {
        return mx::quantized_matmul(a->x, w->w_q, w->scales, w->biases, true, gs, w->bits);
    };

    for (int i = 0; i < opt.warmup; ++i) {
        auto y = qmatmul();
        mx::eval(y);
    }
    mx::synchronize();

    if (opt.capture_output) {
        auto y = mx::astype(qmatmul(), mx::float32);
        mx::eval(y);
        std::memcpy(opt.capture_output, y.data<float>(), M * w->N * sizeof(float));
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        auto y = qmatmul();
        mx::eval(y);
        mx::synchronize();
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void once(size_t, void* weights, void* activations,
          const int8_t*, const float*) {
    auto* w = static_cast<MLXWeights*>(weights);
    auto* a = static_cast<MLXActivations*>(activations);
    auto y = mx::quantized_matmul(a->x, w->w_q, w->scales, w->biases,
                                   true, static_cast<int>(bench::kGroupSize), w->bits);
    mx::eval(y);
}

void cleanup(void* weights, void* activations) {
    delete static_cast<MLXWeights*>(weights);
    if (activations) delete static_cast<MLXActivations*>(activations);
}

static int reg = [] {
    bench::register_backend({
        "mlx_q4", "mlx", bench::QuantCategory::INT4,
        prepare_q4, prepare_act, run, cleanup, 0, once
    });
    bench::register_backend({
        "mlx_q8", "mlx", bench::QuantCategory::INT8,
        prepare_q8, prepare_act, run, cleanup, 0, once
    });
    return 0;
}();

} // namespace

#endif
