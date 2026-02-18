#include "bench_driver.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstring>

namespace {

struct GgmlQ8Weights {
    size_t K, N;
    std::vector<uint8_t> q8_data;
    size_t row_stride;

    size_t once_M = 0;
    std::vector<uint8_t> q8_input;
    size_t q8_row_stride = 0;
    std::vector<float> output;
    ggml_vec_dot_t vec_dot = nullptr;
    int64_t nrows = 0;
};

void* prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new GgmlQ8Weights();
    w->K = K;
    w->N = N;
    w->row_stride = ggml_row_size(GGML_TYPE_Q8_0, K);
    w->q8_data.resize(w->row_stride * N);

    const auto* cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    auto from_float = cpu_traits->from_float;
    if (!from_float) {
        const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        from_float = traits->from_float_ref;
    }

    for (size_t n = 0; n < N; n++)
        from_float(fp32 + n * K, w->q8_data.data() + n * w->row_stride, static_cast<int64_t>(K));

    return w;
}

static void gemv_q8_slice(ggml_vec_dot_t vec_dot, int64_t nrows,
                           const uint8_t* wdata, size_t w_stride,
                           const uint8_t* q8_input, size_t q8_row_stride,
                           float* output, size_t M, size_t K, size_t N,
                           size_t n_begin, size_t n_end) {
    const int Kint = static_cast<int>(K);
    for (size_t m = 0; m < M; m++) {
        const uint8_t* act_row = q8_input + m * q8_row_stride;
        float* out_row = output + m * N;
        bool can_pair_m = (nrows >= 2 && m + 1 < M);

        size_t n = n_begin;
        if (can_pair_m) {
            float* out_row_next = output + (m + 1) * N;
            for (; n + 1 < n_end; n += 2) {
                float tmp[4];
                vec_dot(Kint, tmp, 2,
                        wdata + n * w_stride, w_stride,
                        act_row, q8_row_stride, 2);
                out_row[n]          = tmp[0];
                out_row[n + 1]      = tmp[1];
                out_row_next[n]     = tmp[2];
                out_row_next[n + 1] = tmp[3];
            }
            for (; n < n_end; n++) {
                vec_dot(Kint, out_row + n, 0,
                        wdata + n * w_stride, 0,
                        act_row, 0, 1);
                vec_dot(Kint, out_row_next + n, 0,
                        wdata + n * w_stride, 0,
                        act_row + q8_row_stride, 0, 1);
            }
            m++;
        } else {
            for (; n < n_end; n++) {
                vec_dot(Kint, out_row + n, 0,
                        wdata + n * w_stride, 0,
                        act_row, 0, 1);
            }
        }
    }
}

bench::BenchResult run(size_t M, void* weights, void*,
                       const int8_t*, const float*,
                       const bench::BenchOptions& opt) {
    auto* w = static_cast<GgmlQ8Weights*>(weights);
    const size_t K = w->K, N = w->N;

    const auto* cpu_traits_q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    auto vec_dot = cpu_traits_q8->vec_dot;
    int64_t nrows = cpu_traits_q8->nrows;

    auto quantize_act = cpu_traits_q8->from_float;

    size_t q8_row_stride = ggml_row_size(GGML_TYPE_Q8_0, K);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> fp32_input(M * K);
    for (auto& v : fp32_input) v = dist(gen);

    std::vector<uint8_t> q8_input(q8_row_stride * M);
    for (size_t m = 0; m < M; m++)
        quantize_act(fp32_input.data() + m * K,
                     q8_input.data() + m * q8_row_stride,
                     static_cast<int64_t>(K));

    std::vector<float> output(M * N);
    const uint8_t* wdata = w->q8_data.data();
    size_t w_stride = w->row_stride;

    CactusThreading::set_gemm_threads(opt.num_threads);

    CactusThreading::ParallelConfig par_cfg{256, 64};

    auto run_once = [&]() {
        CactusThreading::parallel_for(N, par_cfg,
            [&](size_t n_begin, size_t n_end) {
                gemv_q8_slice(vec_dot, nrows, wdata, w_stride,
                              q8_input.data(), q8_row_stride,
                              output.data(), M, K, N, n_begin, n_end);
            });
    };

    for (int i = 0; i < opt.warmup; ++i)
        run_once();

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * N * sizeof(float));

    if (opt.capture_reference) {
        auto to_float_q8 = ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float;

        std::vector<float> deq_w(N * K);
        for (size_t n = 0; n < N; n++)
            to_float_q8(wdata + n * w_stride, deq_w.data() + n * K, static_cast<int64_t>(K));

        std::vector<float> deq_a(M * K);
        for (size_t m = 0; m < M; m++)
            to_float_q8(q8_input.data() + m * q8_row_stride, deq_a.data() + m * K, static_cast<int64_t>(K));

        bench::reference_matmul_fp32(deq_a.data(), deq_w.data(),
                                     opt.capture_reference, M, K, N);
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        run_once();
        total_ms += bench::now_ms() - t0;
    }

    CactusThreading::reset_gemm_threads();

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, K, N, opt.iterations, total_ms)};
}

void once(size_t M, void* weights, void*,
          const int8_t*, const float*) {
    auto* w = static_cast<GgmlQ8Weights*>(weights);

    if (w->once_M != M) {
        w->once_M = M;
        const auto* cpu_traits_q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        w->vec_dot = cpu_traits_q8->vec_dot;
        w->nrows = cpu_traits_q8->nrows;

        auto quantize_act = cpu_traits_q8->from_float;
        w->q8_row_stride = ggml_row_size(GGML_TYPE_Q8_0, w->K);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32_input(M * w->K);
        for (auto& v : fp32_input) v = dist(gen);

        w->q8_input.resize(w->q8_row_stride * M);
        for (size_t m = 0; m < M; m++)
            quantize_act(fp32_input.data() + m * w->K,
                         w->q8_input.data() + m * w->q8_row_stride,
                         static_cast<int64_t>(w->K));

        w->output.resize(M * w->N);
    }

    CactusThreading::ParallelConfig par_cfg{256, 64};
    CactusThreading::parallel_for(w->N, par_cfg,
        [&](size_t n_begin, size_t n_end) {
            gemv_q8_slice(w->vec_dot, w->nrows,
                          w->q8_data.data(), w->row_stride,
                          w->q8_input.data(), w->q8_row_stride,
                          w->output.data(), M, w->K, w->N, n_begin, n_end);
        });
}

void cleanup(void* weights, void*) {
    delete static_cast<GgmlQ8Weights*>(weights);
}

static int reg = [] {
    bench::register_backend({
        "ggml_q8_0", "ggml", bench::QuantCategory::INT8,
        prepare, nullptr, run, cleanup, 0, once
    });
    return 0;
}();

} // namespace
