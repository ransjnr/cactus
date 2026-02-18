#include "bench_driver.h"

#ifdef WITH_EXECUTORCH

#include <xnnpack.h>
#include <pthreadpool.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

static bool s_initialized = false;
static pthreadpool_t s_threadpool = nullptr;
static size_t s_threadpool_threads = 0;

static bool ensure_init() {
    if (!s_initialized) {
        if (xnn_initialize(nullptr) != xnn_status_success) {
            fprintf(stderr, "[executorch] xnn_initialize failed\n");
            return false;
        }
        s_threadpool = pthreadpool_create(0);
        s_threadpool_threads = 0;
        s_initialized = true;
    }
    return true;
}

static void ensure_threadpool(int num_threads) {
    size_t target = (num_threads > 0) ? static_cast<size_t>(num_threads) : 0;
    if (target == s_threadpool_threads) return;
    if (s_threadpool) pthreadpool_destroy(s_threadpool);
    s_threadpool = pthreadpool_create(target);
    s_threadpool_threads = target;
}

static void* aligned_alloc_workspace(size_t size) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
}

static void fill_qparams(struct xnn_quantization_params* qp, size_t M,
                          const float* act_scales) {
    for (size_t m = 0; m < M; ++m) {
        qp[m].zero_point = 0;
        qp[m].scale = act_scales[m];
    }
    for (size_t m = M; m < M + XNN_EXTRA_QUANTIZATION_PARAMS; ++m)
        qp[m] = qp[M - 1];
}

// ──── INT8: qd8-f32-qc8w (dynamic quant activations, per-channel INT8 weights) ────

struct XnnInt8Weights {
    size_t K, N;
    xnn_operator_t op = nullptr;
    size_t current_M = 0;
    size_t workspace_size = 0;
    void* workspace = nullptr;
    std::vector<struct xnn_quantization_params> qp_buf;
    std::vector<float> output_buf;
};

void* int8_prepare(const float* fp32, size_t N, size_t K) {
    if (!ensure_init()) return nullptr;

    std::vector<int8_t> qw(N * K);
    std::vector<float> scales(N);
    for (size_t n = 0; n < N; ++n) {
        float mx = 0.0f;
        for (size_t k = 0; k < K; ++k)
            mx = std::max(mx, std::abs(fp32[n * K + k]));
        float s = std::max(mx / 127.0f, 1e-10f);
        scales[n] = s;
        for (size_t k = 0; k < K; ++k) {
            int q = static_cast<int>(std::round(fp32[n * K + k] / s));
            qw[n * K + k] = static_cast<int8_t>(std::clamp(q, -128, 127));
        }
    }

    auto* w = new XnnInt8Weights();
    w->K = K;
    w->N = N;
    xnn_status st = xnn_create_fully_connected_nc_qd8_f32_qc8w(
        K, N, K, N,
        scales.data(), qw.data(), nullptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0, nullptr, &w->op);
    if (st != xnn_status_success) {
        fprintf(stderr, "[executorch_int8] create failed (%d)\n", static_cast<int>(st));
        delete w;
        return nullptr;
    }
    return w;
}

static void int8_reshape(XnnInt8Weights* w, size_t M) {
    size_t ws = 0;
    xnn_reshape_fully_connected_nc_qd8_f32_qc8w(w->op, M, &ws, s_threadpool);
    if (ws > w->workspace_size) {
        free(w->workspace);
        w->workspace = aligned_alloc_workspace(ws);
        w->workspace_size = ws;
    }
    if (w->qp_buf.size() < M + XNN_EXTRA_QUANTIZATION_PARAMS)
        w->qp_buf.resize(M + XNN_EXTRA_QUANTIZATION_PARAMS);
    w->current_M = M;
}

bench::BenchResult int8_run(size_t M, void* weights, void*,
                             const int8_t* act, const float* act_scales,
                             const bench::BenchOptions& opt) {
    auto* w = static_cast<XnnInt8Weights*>(weights);
    if (!w || !w->op) return {};

    ensure_threadpool(opt.num_threads);

    std::vector<float> output(M * w->N);
    int8_reshape(w, M);
    fill_qparams(w->qp_buf.data(), M, act_scales);

    xnn_setup_fully_connected_nc_qd8_f32_qc8w(
        w->op, act, output.data(), w->workspace, w->qp_buf.data());

    for (int i = 0; i < opt.warmup; ++i)
        xnn_run_operator(w->op, s_threadpool);

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * w->N * sizeof(float));

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        xnn_run_operator(w->op, s_threadpool);
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void int8_once(size_t M, void* weights, void*,
               const int8_t* act, const float* act_scales) {
    auto* w = static_cast<XnnInt8Weights*>(weights);
    if (!w || !w->op) return;

    w->output_buf.resize(M * w->N);
    if (w->current_M != M) int8_reshape(w, M);
    fill_qparams(w->qp_buf.data(), M, act_scales);

    xnn_setup_fully_connected_nc_qd8_f32_qc8w(
        w->op, act, w->output_buf.data(), w->workspace, w->qp_buf.data());
    xnn_run_operator(w->op, s_threadpool);
}

void int8_cleanup(void* weights, void*) {
    auto* w = static_cast<XnnInt8Weights*>(weights);
    if (w) {
        if (w->op) xnn_delete_operator(w->op);
        free(w->workspace);
        delete w;
    }
}

// ──── INT4: qd8-f32-qb4w (dynamic quant activations, block-wise INT4 weights) ────

struct XnnInt4Weights {
    size_t K, N;
    xnn_operator_t op = nullptr;
    size_t current_M = 0;
    size_t workspace_size = 0;
    void* workspace = nullptr;
    std::vector<struct xnn_quantization_params> qp_buf;
    std::vector<float> output_buf;
};

void* int4_prepare(const float* fp32, size_t N, size_t K) {
    if (!ensure_init()) return nullptr;

    const size_t block_size = bench::kGroupSize;
    const size_t num_blocks = K / block_size;

    std::vector<uint8_t> packed(N * K / 2);
    std::vector<uint16_t> scales_u16(N * num_blocks);

    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < num_blocks; ++g) {
            float mx = 0.0f;
            size_t base = n * K + g * block_size;
            for (size_t k = 0; k < block_size; ++k)
                mx = std::max(mx, std::abs(fp32[base + k]));
            float s = std::max(mx / 7.0f, 1e-10f);

            uint32_t s_u32;
            std::memcpy(&s_u32, &s, sizeof(uint32_t));
            scales_u16[n * num_blocks + g] = static_cast<uint16_t>(s_u32 >> 16);

            for (size_t k = 0; k < block_size; k += 2) {
                int q0 = static_cast<int>(std::round(fp32[base + k] / s));
                int q1 = static_cast<int>(std::round(fp32[base + k + 1] / s));
                q0 = std::clamp(q0, -8, 7);
                q1 = std::clamp(q1, -8, 7);
                uint8_t lo = static_cast<uint8_t>(q0 + 8) & 0x0F;
                uint8_t hi = static_cast<uint8_t>(q1 + 8) & 0x0F;
                packed[(base + k) / 2] = lo | (hi << 4);
            }
        }
    }

    auto* w = new XnnInt4Weights();
    w->K = K;
    w->N = N;
    xnn_status st = xnn_create_fully_connected_nc_qd8_f32_qb4w(
        K, N, K, N,
        block_size, 8,
        scales_u16.data(), packed.data(), nullptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        0, nullptr, &w->op);
    if (st != xnn_status_success) {
        fprintf(stderr, "[executorch_int4] create failed (%d)\n", static_cast<int>(st));
        delete w;
        return nullptr;
    }
    return w;
}

static void int4_reshape(XnnInt4Weights* w, size_t M) {
    size_t ws = 0;
    xnn_reshape_fully_connected_nc_qd8_f32_qb4w(w->op, M, &ws, s_threadpool);
    if (ws > w->workspace_size) {
        free(w->workspace);
        w->workspace = aligned_alloc_workspace(ws);
        w->workspace_size = ws;
    }
    if (w->qp_buf.size() < M + XNN_EXTRA_QUANTIZATION_PARAMS)
        w->qp_buf.resize(M + XNN_EXTRA_QUANTIZATION_PARAMS);
    w->current_M = M;
}

bench::BenchResult int4_run(size_t M, void* weights, void*,
                             const int8_t* act, const float* act_scales,
                             const bench::BenchOptions& opt) {
    auto* w = static_cast<XnnInt4Weights*>(weights);
    if (!w || !w->op) return {};

    ensure_threadpool(opt.num_threads);

    std::vector<float> output(M * w->N);
    int4_reshape(w, M);
    fill_qparams(w->qp_buf.data(), M, act_scales);

    xnn_setup_fully_connected_nc_qd8_f32_qb4w(
        w->op, act, output.data(), w->workspace, w->qp_buf.data());

    for (int i = 0; i < opt.warmup; ++i)
        xnn_run_operator(w->op, s_threadpool);

    if (opt.capture_output)
        std::memcpy(opt.capture_output, output.data(), M * w->N * sizeof(float));

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        xnn_run_operator(w->op, s_threadpool);
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void int4_once(size_t M, void* weights, void*,
               const int8_t* act, const float* act_scales) {
    auto* w = static_cast<XnnInt4Weights*>(weights);
    if (!w || !w->op) return;

    w->output_buf.resize(M * w->N);
    if (w->current_M != M) int4_reshape(w, M);
    fill_qparams(w->qp_buf.data(), M, act_scales);

    xnn_setup_fully_connected_nc_qd8_f32_qb4w(
        w->op, act, w->output_buf.data(), w->workspace, w->qp_buf.data());
    xnn_run_operator(w->op, s_threadpool);
}

void int4_cleanup(void* weights, void*) {
    auto* w = static_cast<XnnInt4Weights*>(weights);
    if (w) {
        if (w->op) xnn_delete_operator(w->op);
        free(w->workspace);
        delete w;
    }
}

// ──── Registration ────

static int reg = [] {
    bench::register_backend({
        "executorch_int8", "executorch", bench::QuantCategory::INT8,
        int8_prepare, nullptr, int8_run, int8_cleanup, 0, int8_once
    });
    bench::register_backend({
        "executorch_int4", "executorch", bench::QuantCategory::INT4,
        int4_prepare, nullptr, int4_run, int4_cleanup, 0, int4_once
    });
    return 0;
}();

} // namespace

#else // !WITH_EXECUTORCH

namespace {
[[maybe_unused]] static int reg = [] { return 0; }();
} // namespace

#endif // WITH_EXECUTORCH
