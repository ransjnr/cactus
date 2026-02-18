#include "bench_driver.h"

#ifdef WITH_ONNXRT

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace {

// ─── Minimal protobuf encoder for ONNX models ──────────────────────────────

struct PBuf {
    std::vector<uint8_t> d;
    void varint(uint64_t v) {
        while (v > 0x7F) { d.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80)); v >>= 7; }
        d.push_back(static_cast<uint8_t>(v));
    }
    void fld_vi(int f, uint64_t v) {
        varint(static_cast<uint64_t>(f) << 3);
        varint(v);
    }
    void fld_ld(int f, const PBuf& sub) {
        varint(static_cast<uint64_t>(f) << 3 | 2);
        varint(sub.d.size());
        d.insert(d.end(), sub.d.begin(), sub.d.end());
    }
    void fld_str(int f, const char* s) {
        size_t n = std::strlen(s);
        varint(static_cast<uint64_t>(f) << 3 | 2);
        varint(n);
        d.insert(d.end(), s, s + n);
    }
};

static PBuf make_dim_param(const char* name) { PBuf d; d.fld_str(2, name); return d; }
static PBuf make_dim_value(size_t v) { PBuf d; d.fld_vi(1, v); return d; }

template<typename... Dims>
static PBuf make_value_info(const char* name, int elem_type, const Dims&... dims) {
    PBuf shape; (shape.fld_ld(1, dims), ...);
    PBuf tensor; tensor.fld_vi(1, elem_type); tensor.fld_ld(2, shape);
    PBuf type; type.fld_ld(1, tensor);
    PBuf vi; vi.fld_str(1, name); vi.fld_ld(2, type);
    return vi;
}

static PBuf make_attr_int(const char* name, int64_t val) {
    PBuf a;
    a.fld_str(1, name);
    a.fld_vi(3, static_cast<uint64_t>(val));
    a.fld_vi(20, 2); // AttributeType.INT = 2
    return a;
}

static Ort::Env& get_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bench");
    return env;
}

static Ort::MemoryInfo& get_cpu_mem() {
    static Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return mem;
}

// ─── INT8 backend (MatMulInteger) ───────────────────────────────────────────

static const std::vector<uint8_t>& get_int8_model() {
    static std::vector<uint8_t> bytes = [] {
        PBuf node;
        node.fld_str(1, "A"); node.fld_str(1, "B");
        node.fld_str(2, "Y");
        node.fld_str(4, "MatMulInteger");

        auto a = make_value_info("A", 3, make_dim_param("M"), make_dim_param("K"));
        auto b = make_value_info("B", 3, make_dim_param("K"), make_dim_param("N"));
        auto y = make_value_info("Y", 6, make_dim_param("M"), make_dim_param("N"));

        PBuf graph;
        graph.fld_ld(1, node); graph.fld_str(2, "bench_int8");
        graph.fld_ld(11, a); graph.fld_ld(11, b); graph.fld_ld(12, y);

        PBuf opset; opset.fld_str(1, ""); opset.fld_vi(2, 13);

        PBuf model; model.fld_vi(1, 7); model.fld_ld(8, opset); model.fld_ld(7, graph);
        return model.d;
    }();
    return bytes;
}

struct OrtInt8Weights {
    size_t K, N;
    std::vector<int8_t> B_KN;
    float w_scale;
    std::unique_ptr<Ort::Session> session;
    int session_threads = 0;
    std::vector<float> output_buf;

    void ensure_session(int threads) {
        int target = threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
        if (session && session_threads == target) return;
        const auto& bytes = get_int8_model();
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(target);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(get_env(), bytes.data(), bytes.size(), opts);
        session_threads = target;
    }
};

void* i8_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new OrtInt8Weights();
    w->K = K; w->N = N;

    float max_abs = 0.0f;
    for (size_t i = 0; i < N * K; i++)
        max_abs = std::max(max_abs, std::abs(fp32[i]));
    w->w_scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    float inv = 127.0f / std::max(max_abs, 1e-10f);

    std::vector<int8_t> row(N * K);
    for (size_t i = 0; i < N * K; i++)
        row[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(fp32[i] * inv))));

    w->B_KN.resize(K * N);
    for (size_t n = 0; n < N; n++)
        for (size_t k = 0; k < K; k++)
            w->B_KN[k * N + n] = row[n * K + k];
    return w;
}

bench::BenchResult i8_run(size_t M, void* weights, void*,
                          const int8_t* act_int8, const float* act_scales,
                          const bench::BenchOptions& opt) {
    auto* w = static_cast<OrtInt8Weights*>(weights);
    w->ensure_session(opt.num_threads);

    auto& mem = get_cpu_mem();
    int64_t a_shape[] = {(int64_t)M, (int64_t)w->K};
    int64_t b_shape[] = {(int64_t)w->K, (int64_t)w->N};

    Ort::Value inputs[2] = {
        Ort::Value::CreateTensor<int8_t>(mem, const_cast<int8_t*>(act_int8), M * w->K, a_shape, 2),
        Ort::Value::CreateTensor<int8_t>(mem, w->B_KN.data(), w->K * w->N, b_shape, 2)
    };
    const char* in_names[] = {"A", "B"};
    const char* out_names[] = {"Y"};

    for (int i = 0; i < opt.warmup; ++i)
        w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);

    if (opt.capture_output) {
        auto out = w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);
        const int32_t* y = out[0].GetTensorData<int32_t>();
        for (size_t m = 0; m < M; m++) {
            float scale = act_scales[m] * w->w_scale;
            for (size_t n = 0; n < w->N; n++)
                opt.capture_output[m * w->N + n] = static_cast<float>(y[m * w->N + n]) * scale;
        }
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);
        total_ms += bench::now_ms() - t0;
    }
    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void i8_once(size_t M, void* weights, void*,
             const int8_t* act_int8, const float* act_scales) {
    auto* w = static_cast<OrtInt8Weights*>(weights);
    if (!w->session) return;

    auto& mem = get_cpu_mem();
    int64_t a_shape[] = {(int64_t)M, (int64_t)w->K};
    int64_t b_shape[] = {(int64_t)w->K, (int64_t)w->N};

    Ort::Value inputs[2] = {
        Ort::Value::CreateTensor<int8_t>(mem, const_cast<int8_t*>(act_int8), M * w->K, a_shape, 2),
        Ort::Value::CreateTensor<int8_t>(mem, w->B_KN.data(), w->K * w->N, b_shape, 2)
    };
    const char* in_names[] = {"A", "B"};
    const char* out_names[] = {"Y"};

    auto out = w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);
    const int32_t* y = out[0].GetTensorData<int32_t>();
    w->output_buf.resize(M * w->N);
    for (size_t m = 0; m < M; m++) {
        float scale = act_scales[m] * w->w_scale;
        for (size_t n = 0; n < w->N; n++)
            w->output_buf[m * w->N + n] = static_cast<float>(y[m * w->N + n]) * scale;
    }
}

void i8_cleanup(void* weights, void*) { delete static_cast<OrtInt8Weights*>(weights); }

// ─── INT4 backend (MatMulNBits, com.microsoft contrib op) ───────────────────

static std::vector<uint8_t> build_int4_model(size_t K, size_t N) {
    const size_t n_blocks = K / bench::kGroupSize;
    const size_t blob_size = bench::kGroupSize / 2;

    PBuf node;
    node.fld_str(1, "A");
    node.fld_str(1, "B");
    node.fld_str(1, "scales");
    node.fld_str(1, "zero_points");
    node.fld_str(2, "Y");
    node.fld_str(4, "MatMulNBits");
    node.fld_str(7, "com.microsoft");
    node.fld_ld(5, make_attr_int("K", static_cast<int64_t>(K)));
    node.fld_ld(5, make_attr_int("N", static_cast<int64_t>(N)));
    node.fld_ld(5, make_attr_int("bits", 4));
    node.fld_ld(5, make_attr_int("block_size", static_cast<int64_t>(bench::kGroupSize)));
    node.fld_ld(5, make_attr_int("accuracy_level", 4)); // int8 accumulation — fastest path

    // FLOAT=1, UINT8=2
    auto a_vi  = make_value_info("A", 1, make_dim_param("M"), make_dim_value(K));
    auto b_vi  = make_value_info("B", 2, make_dim_value(N), make_dim_value(n_blocks),
                                 make_dim_value(blob_size));
    auto s_vi  = make_value_info("scales", 1, make_dim_value(N), make_dim_value(n_blocks));
    auto zp_vi = make_value_info("zero_points", 2,
                                 make_dim_value(N), make_dim_value((n_blocks + 1) / 2));
    auto y_vi  = make_value_info("Y", 1, make_dim_param("M"), make_dim_value(N));

    PBuf graph;
    graph.fld_ld(1, node);
    graph.fld_str(2, "bench_int4");
    graph.fld_ld(11, a_vi);
    graph.fld_ld(11, b_vi);
    graph.fld_ld(11, s_vi);
    graph.fld_ld(11, zp_vi);
    graph.fld_ld(12, y_vi);

    PBuf opset1; opset1.fld_str(1, ""); opset1.fld_vi(2, 13);
    PBuf opset2; opset2.fld_str(1, "com.microsoft"); opset2.fld_vi(2, 1);

    PBuf model;
    model.fld_vi(1, 7);
    model.fld_ld(8, opset1);
    model.fld_ld(8, opset2);
    model.fld_ld(7, graph);
    return model.d;
}

struct OrtInt4Weights {
    size_t K, N;
    std::vector<uint8_t> B_packed;
    std::vector<float> scales;
    std::vector<uint8_t> zero_points;
    std::unique_ptr<Ort::Session> session;
    int session_threads = 0;
    std::vector<float> output_buf;

    void ensure_session(int threads) {
        int target = threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
        if (session && session_threads == target) return;
        auto bytes = build_int4_model(K, N);
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(target);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(get_env(), bytes.data(), bytes.size(), opts);
        session_threads = target;
    }
};

struct OrtInt4Activations {
    std::vector<float> fp32;
};

void* i4_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new OrtInt4Weights();
    w->K = K; w->N = N;

    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> int4_vals;
    std::vector<float> raw_scales;
    bench::quantize_int4_per_group(src, N, K, int4_vals, raw_scales);

    w->B_packed.resize(N * K / 2);
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k += 2) {
            uint8_t lo = static_cast<uint8_t>(int4_vals[n * K + k] + 8);
            uint8_t hi = static_cast<uint8_t>(int4_vals[n * K + k + 1] + 8);
            w->B_packed[n * (K / 2) + k / 2] = (hi << 4) | lo;
        }
    }

    w->scales = raw_scales;

    const size_t n_blocks = K / bench::kGroupSize;
    const size_t zp_cols = (n_blocks + 1) / 2;
    w->zero_points.resize(N * zp_cols, 0x88);

    return w;
}

void* i4_prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new OrtInt4Activations();
    a->fp32.assign(fp32, fp32 + M * K);
    return a;
}

bench::BenchResult i4_run(size_t M, void* weights, void* activations,
                          const int8_t*, const float*,
                          const bench::BenchOptions& opt) {
    auto* w = static_cast<OrtInt4Weights*>(weights);
    auto* a = static_cast<OrtInt4Activations*>(activations);
    w->ensure_session(opt.num_threads);

    auto& mem = get_cpu_mem();
    const size_t n_blocks = w->K / bench::kGroupSize;
    const size_t blob_size = bench::kGroupSize / 2;
    const size_t zp_cols = (n_blocks + 1) / 2;

    int64_t a_shape[]  = {(int64_t)M, (int64_t)w->K};
    int64_t b_shape[]  = {(int64_t)w->N, (int64_t)n_blocks, (int64_t)blob_size};
    int64_t s_shape[]  = {(int64_t)w->N, (int64_t)n_blocks};
    int64_t zp_shape[] = {(int64_t)w->N, (int64_t)zp_cols};

    Ort::Value inputs[4] = {
        Ort::Value::CreateTensor<float>(mem, a->fp32.data(), M * w->K, a_shape, 2),
        Ort::Value::CreateTensor<uint8_t>(mem, w->B_packed.data(), w->N * w->K / 2, b_shape, 3),
        Ort::Value::CreateTensor<float>(mem, w->scales.data(), w->N * n_blocks, s_shape, 2),
        Ort::Value::CreateTensor<uint8_t>(mem, w->zero_points.data(), w->N * zp_cols, zp_shape, 2)
    };
    const char* in_names[] = {"A", "B", "scales", "zero_points"};
    const char* out_names[] = {"Y"};

    for (int i = 0; i < opt.warmup; ++i)
        w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 4, out_names, 1);

    if (opt.capture_output) {
        auto out = w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 4, out_names, 1);
        const float* y = out[0].GetTensorData<float>();
        std::memcpy(opt.capture_output, y, M * w->N * sizeof(float));
    }

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; ++i) {
        double t0 = bench::now_ms();
        w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 4, out_names, 1);
        total_ms += bench::now_ms() - t0;
    }
    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void i4_once(size_t M, void* weights, void* activations,
             const int8_t*, const float*) {
    auto* w = static_cast<OrtInt4Weights*>(weights);
    auto* a = static_cast<OrtInt4Activations*>(activations);
    if (!w->session) return;

    auto& mem = get_cpu_mem();
    const size_t n_blocks = w->K / bench::kGroupSize;
    const size_t blob_size = bench::kGroupSize / 2;
    const size_t zp_cols = (n_blocks + 1) / 2;

    int64_t a_shape[]  = {(int64_t)M, (int64_t)w->K};
    int64_t b_shape[]  = {(int64_t)w->N, (int64_t)n_blocks, (int64_t)blob_size};
    int64_t s_shape[]  = {(int64_t)w->N, (int64_t)n_blocks};
    int64_t zp_shape[] = {(int64_t)w->N, (int64_t)zp_cols};

    Ort::Value inputs[4] = {
        Ort::Value::CreateTensor<float>(mem, a->fp32.data(), M * w->K, a_shape, 2),
        Ort::Value::CreateTensor<uint8_t>(mem, w->B_packed.data(), w->N * w->K / 2, b_shape, 3),
        Ort::Value::CreateTensor<float>(mem, w->scales.data(), w->N * n_blocks, s_shape, 2),
        Ort::Value::CreateTensor<uint8_t>(mem, w->zero_points.data(), w->N * zp_cols, zp_shape, 2)
    };
    const char* in_names[] = {"A", "B", "scales", "zero_points"};
    const char* out_names[] = {"Y"};

    auto out = w->session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 4, out_names, 1);
    const float* y = out[0].GetTensorData<float>();
    w->output_buf.resize(M * w->N);
    std::memcpy(w->output_buf.data(), y, M * w->N * sizeof(float));
}

void i4_cleanup(void* weights, void* activations) {
    delete static_cast<OrtInt4Weights*>(weights);
    if (activations) delete static_cast<OrtInt4Activations*>(activations);
}

// ─── Registration ───────────────────────────────────────────────────────────

static int reg = [] {
    bench::register_backend({
        "onnxrt_int8", "onnxrt", bench::QuantCategory::INT8,
        i8_prepare, nullptr, i8_run, i8_cleanup, 0, i8_once
    });
    bench::register_backend({
        "onnxrt_int4", "onnxrt", bench::QuantCategory::INT4,
        i4_prepare, i4_prepare_act, i4_run, i4_cleanup, 0, i4_once
    });
    return 0;
}();

} // namespace

#else

namespace {
[[maybe_unused]] static int reg = [] { return 0; }();
} // namespace

#endif
