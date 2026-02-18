#include "bench_driver.h"

#ifdef WITH_MLC

#include <tvm/ffi/c_api.h>
#include <dlpack/dlpack.h>
#include <arm_neon.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// MLC backend: uses the TVM FFI runtime to dispatch a quantized INT4 matmul
// kernel through the same PackedFunc/DLTensor interface that MLC-LLM uses.
//
// The kernel is registered as a TVM global function via TVMFFIFunctionCreate,
// matching the ABI that TVM's codegen produces for ARM targets. If
// MLC_MATMUL_LIB is set, the backend loads that .so first (for externally
// compiled kernels from `mlc_llm compile`).
//
// Setup:
//   1. git clone https://github.com/mlc-ai/mlc-llm third_party/mlc
//   2. cd third_party/mlc/3rdparty/tvm
//      git submodule update --init 3rdparty/tvm-ffi
//      cd 3rdparty/tvm-ffi && git submodule update --init && cd ../..
//      mkdir build && cd build
//      cmake .. -DUSE_LLVM=OFF -DUSE_CUDA=OFF -DUSE_METAL=OFF
//      make tvm_runtime -j
//   3. cmake -B build -DWITH_MLC=ON && cmake --build build --target test_matmul_bench

namespace {

// ── TVM-dispatched INT4 dequant-to-fp16 kernel ─────────────────────────

static inline __fp16 hsum_f16x8(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t s4 = vadd_f16(lo, hi);
    float16x4_t s2 = vadd_f16(s4, vext_f16(s4, s4, 2));
    float16x4_t s1 = vadd_f16(s2, vext_f16(s2, s2, 1));
    return vget_lane_f16(s1, 0);
}

static void mlc_q4_matmul_impl(
    const __fp16* A, const uint8_t* B_packed, const __fp16* B_scales,
    __fp16* C, size_t M, size_t K, size_t N, size_t group_size
) {
    const size_t num_groups = K / group_size;
    const size_t half_K = K / 2;
    const uint8x16_t eight_u8 = vdupq_n_u8(8);

    for (size_t m = 0; m < M; m++) {
        const __fp16* a_row = A + m * K;
        __fp16* c_row = C + m * N;

        size_t n = 0;
        for (; n + 3 < N; n += 4) {
            float16x8_t acc[4] = {
                vdupq_n_f16(0), vdupq_n_f16(0),
                vdupq_n_f16(0), vdupq_n_f16(0)
            };

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                float16x8_t a0 = vld1q_f16(a_row + k_base);
                float16x8_t a1 = vld1q_f16(a_row + k_base + 8);
                float16x8_t a2 = vld1q_f16(a_row + k_base + 16);
                float16x8_t a3 = vld1q_f16(a_row + k_base + 24);

                for (size_t ni = 0; ni < 4; ni++) {
                    float16x8_t sv = vdupq_n_f16(B_scales[(n + ni) * num_groups + g]);
                    const uint8_t* p = B_packed + (n + ni) * half_K + k_base / 2;

                    uint8x16_t raw = vld1q_u8(p);
                    uint8x16_t lo_u = vandq_u8(raw, vdupq_n_u8(0x0F));
                    uint8x16_t hi_u = vshrq_n_u8(raw, 4);
                    uint8x16x2_t z = vzipq_u8(lo_u, hi_u);

                    int8x16_t s0 = vreinterpretq_s8_u8(vsubq_u8(z.val[0], eight_u8));
                    int8x16_t s1 = vreinterpretq_s8_u8(vsubq_u8(z.val[1], eight_u8));

                    float16x8_t w0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(s0))), sv);
                    float16x8_t w1 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(s0))), sv);
                    float16x8_t w2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(s1))), sv);
                    float16x8_t w3 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(s1))), sv);

                    acc[ni] = vfmaq_f16(acc[ni], w0, a0);
                    acc[ni] = vfmaq_f16(acc[ni], w1, a1);
                    acc[ni] = vfmaq_f16(acc[ni], w2, a2);
                    acc[ni] = vfmaq_f16(acc[ni], w3, a3);
                }
            }
            for (size_t ni = 0; ni < 4; ni++)
                c_row[n + ni] = hsum_f16x8(acc[ni]);
        }
        for (; n < N; n++) {
            float16x8_t acc = vdupq_n_f16(0);
            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                float16x8_t sv = vdupq_n_f16(B_scales[n * num_groups + g]);
                const uint8_t* p = B_packed + n * half_K + k_base / 2;

                uint8x16_t raw = vld1q_u8(p);
                uint8x16_t lo_u = vandq_u8(raw, vdupq_n_u8(0x0F));
                uint8x16_t hi_u = vshrq_n_u8(raw, 4);
                uint8x16x2_t z = vzipq_u8(lo_u, hi_u);

                int8x16_t s0 = vreinterpretq_s8_u8(vsubq_u8(z.val[0], eight_u8));
                int8x16_t s1 = vreinterpretq_s8_u8(vsubq_u8(z.val[1], eight_u8));

                float16x8_t w0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(s0))), sv);
                float16x8_t w1 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(s0))), sv);
                float16x8_t w2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(s1))), sv);
                float16x8_t w3 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(s1))), sv);

                acc = vfmaq_f16(acc, w0, vld1q_f16(a_row + k_base));
                acc = vfmaq_f16(acc, w1, vld1q_f16(a_row + k_base + 8));
                acc = vfmaq_f16(acc, w2, vld1q_f16(a_row + k_base + 16));
                acc = vfmaq_f16(acc, w3, vld1q_f16(a_row + k_base + 24));
            }
            c_row[n] = hsum_f16x8(acc);
        }
    }
}

// TVMFFISafeCallType callback: bridges TVM PackedFunc dispatch to our kernel
static int tvm_q4_matmul_callback(void*, const TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* result) {
    if (num_args != 4) {
        TVMFFIErrorSetRaisedFromCStr("ValueError", "quantized_matmul_int4 expects 4 args");
        return -1;
    }

    auto* A_dl = static_cast<DLTensor*>(args[0].v_ptr);
    auto* B_dl = static_cast<DLTensor*>(args[1].v_ptr);
    auto* S_dl = static_cast<DLTensor*>(args[2].v_ptr);
    auto* C_dl = static_cast<DLTensor*>(args[3].v_ptr);

    size_t M = static_cast<size_t>(A_dl->shape[0]);
    size_t K = static_cast<size_t>(A_dl->shape[1]);
    size_t N = static_cast<size_t>(B_dl->shape[0]);
    size_t num_groups = static_cast<size_t>(S_dl->shape[1]);
    size_t group_size = K / num_groups;

    mlc_q4_matmul_impl(
        static_cast<const __fp16*>(A_dl->data),
        static_cast<const uint8_t*>(B_dl->data),
        static_cast<const __fp16*>(S_dl->data),
        static_cast<__fp16*>(C_dl->data),
        M, K, N, group_size);

    result->type_index = kTVMFFINone;
    return 0;
}

// ── TVM-dispatched INT8 dequant-to-fp16 kernel ─────────────────────────

static void mlc_q8_matmul_impl(
    const __fp16* A, const int8_t* B, const __fp16* B_scales,
    __fp16* C, size_t M, size_t K, size_t N, size_t group_size
) {
    const size_t num_groups = K / group_size;

    for (size_t m = 0; m < M; m++) {
        const __fp16* a_row = A + m * K;
        __fp16* c_row = C + m * N;

        size_t n = 0;
        for (; n + 3 < N; n += 4) {
            float16x8_t acc[4] = {
                vdupq_n_f16(0), vdupq_n_f16(0),
                vdupq_n_f16(0), vdupq_n_f16(0)
            };

            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                float16x8_t a0 = vld1q_f16(a_row + k_base);
                float16x8_t a1 = vld1q_f16(a_row + k_base + 8);
                float16x8_t a2 = vld1q_f16(a_row + k_base + 16);
                float16x8_t a3 = vld1q_f16(a_row + k_base + 24);

                for (size_t ni = 0; ni < 4; ni++) {
                    float16x8_t sv = vdupq_n_f16(B_scales[(n + ni) * num_groups + g]);
                    const int8_t* p = B + (n + ni) * K + k_base;

                    int8x16_t b0 = vld1q_s8(p);
                    int8x16_t b1 = vld1q_s8(p + 16);

                    float16x8_t w0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(b0))), sv);
                    float16x8_t w1 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(b0))), sv);
                    float16x8_t w2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(b1))), sv);
                    float16x8_t w3 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(b1))), sv);

                    acc[ni] = vfmaq_f16(acc[ni], w0, a0);
                    acc[ni] = vfmaq_f16(acc[ni], w1, a1);
                    acc[ni] = vfmaq_f16(acc[ni], w2, a2);
                    acc[ni] = vfmaq_f16(acc[ni], w3, a3);
                }
            }
            for (size_t ni = 0; ni < 4; ni++)
                c_row[n + ni] = hsum_f16x8(acc[ni]);
        }
        for (; n < N; n++) {
            float16x8_t acc = vdupq_n_f16(0);
            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                float16x8_t sv = vdupq_n_f16(B_scales[n * num_groups + g]);
                const int8_t* p = B + n * K + k_base;

                int8x16_t b0 = vld1q_s8(p);
                int8x16_t b1 = vld1q_s8(p + 16);

                float16x8_t w0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(b0))), sv);
                float16x8_t w1 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(b0))), sv);
                float16x8_t w2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(b1))), sv);
                float16x8_t w3 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_high_s8(b1))), sv);

                acc = vfmaq_f16(acc, w0, vld1q_f16(a_row + k_base));
                acc = vfmaq_f16(acc, w1, vld1q_f16(a_row + k_base + 8));
                acc = vfmaq_f16(acc, w2, vld1q_f16(a_row + k_base + 16));
                acc = vfmaq_f16(acc, w3, vld1q_f16(a_row + k_base + 24));
            }
            c_row[n] = hsum_f16x8(acc);
        }
    }
}

static int tvm_q8_matmul_callback(void*, const TVMFFIAny* args, int32_t num_args,
                                   TVMFFIAny* result) {
    if (num_args != 4) {
        TVMFFIErrorSetRaisedFromCStr("ValueError", "quantized_matmul_int8 expects 4 args");
        return -1;
    }

    auto* A_dl = static_cast<DLTensor*>(args[0].v_ptr);
    auto* B_dl = static_cast<DLTensor*>(args[1].v_ptr);
    auto* S_dl = static_cast<DLTensor*>(args[2].v_ptr);
    auto* C_dl = static_cast<DLTensor*>(args[3].v_ptr);

    size_t M = static_cast<size_t>(A_dl->shape[0]);
    size_t K = static_cast<size_t>(A_dl->shape[1]);
    size_t N = static_cast<size_t>(B_dl->shape[0]);
    size_t num_groups = static_cast<size_t>(S_dl->shape[1]);
    size_t group_size = K / num_groups;

    mlc_q8_matmul_impl(
        static_cast<const __fp16*>(A_dl->data),
        static_cast<const int8_t*>(B_dl->data),
        static_cast<const __fp16*>(S_dl->data),
        static_cast<__fp16*>(C_dl->data),
        M, K, N, group_size);

    result->type_index = kTVMFFINone;
    return 0;
}

// ── MLC backend state ───────────────────────────────────────────────────

static TVMFFIObjectHandle register_tvm_func(const char* name, size_t name_len,
                                             TVMFFISafeCallType callback) {
    const char* lib_path = std::getenv("MLC_MATMUL_LIB");
    if (lib_path) {
        static bool lib_loaded = false;
        if (!lib_loaded) {
            void* handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
            if (!handle)
                fprintf(stderr, "[mlc] Warning: cannot load %s: %s\n", lib_path, dlerror());
            lib_loaded = true;
        }
    }

    TVMFFIByteArray ba;
    ba.data = name;
    ba.size = name_len;

    TVMFFIObjectHandle func = nullptr;
    TVMFFIFunctionGetGlobal(&ba, &func);

    if (!func) {
        if (TVMFFIFunctionCreate(nullptr, callback, nullptr, &func) != 0 || !func)
            return nullptr;
        if (TVMFFIFunctionSetGlobal(&ba, func, 0) != 0)
            return nullptr;
    }

    TVMFFIObjectIncRef(func);
    return func;
}

static TVMFFIObjectHandle s_q4_fn = nullptr;
static TVMFFIObjectHandle s_q8_fn = nullptr;

// ── Shared helpers ──────────────────────────────────────────────────────

struct MlcActivations {
    std::vector<__fp16> fp16;
};

struct MlcInt4Weights {
    size_t K, N;
    std::vector<uint8_t> packed;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;
    int64_t b_shape[2];
    int64_t s_shape[2];
};

struct MlcInt8Weights {
    size_t K, N;
    std::vector<int8_t> quantized;
    std::vector<__fp16> scales;
    std::vector<__fp16> output_buf;
    int64_t b_shape[2];
    int64_t s_shape[2];
};

static inline DLTensor make_dl(void* data, int64_t* shape, int ndim,
                                uint8_t type_code, uint8_t bits) {
    DLTensor t{};
    t.data = data;
    t.device = {kDLCPU, 0};
    t.ndim = ndim;
    t.dtype = {type_code, bits, 1};
    t.shape = shape;
    return t;
}

static void call_tvm_fn(TVMFFIObjectHandle fn, DLTensor* a, DLTensor* b,
                         DLTensor* s, DLTensor* c) {
    TVMFFIAny args[4];
    for (int i = 0; i < 4; i++) {
        args[i].type_index = kTVMFFIDLTensorPtr;
        args[i].zero_padding = 0;
    }
    args[0].v_ptr = a;
    args[1].v_ptr = b;
    args[2].v_ptr = s;
    args[3].v_ptr = c;

    TVMFFIAny result;
    result.type_index = kTVMFFINone;
    result.zero_padding = 0;
    result.v_int64 = 0;

    auto* cell = TVMFFIFunctionGetCellPtr(fn);
    cell->safe_call(fn, args, 4, &result);
}

// ── INT4 benchmark plumbing ─────────────────────────────────────────────

static void q4_run_kernel(size_t M, MlcInt4Weights* w, MlcActivations* act,
                           __fp16* output) {
    int64_t a_shape[] = {static_cast<int64_t>(M), static_cast<int64_t>(w->K)};
    int64_t c_shape[] = {static_cast<int64_t>(M), static_cast<int64_t>(w->N)};

    DLTensor a_dl = make_dl(act->fp16.data(), a_shape, 2, kDLFloat, 16);
    DLTensor b_dl = make_dl(w->packed.data(), w->b_shape, 2, kDLUInt, 8);
    DLTensor s_dl = make_dl(w->scales.data(), w->s_shape, 2, kDLFloat, 16);
    DLTensor c_dl = make_dl(output, c_shape, 2, kDLFloat, 16);
    call_tvm_fn(s_q4_fn, &a_dl, &b_dl, &s_dl, &c_dl);
}

void* q4_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new MlcInt4Weights();
    w->K = K;
    w->N = N;

    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> int4_vals;
    std::vector<float> raw_scales;
    bench::quantize_int4_per_group(src, N, K, int4_vals, raw_scales);

    w->packed.resize(N * K / 2);
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k += 2) {
            uint8_t lo = static_cast<uint8_t>(int4_vals[n * K + k] + 8);
            uint8_t hi = static_cast<uint8_t>(int4_vals[n * K + k + 1] + 8);
            w->packed[n * (K / 2) + k / 2] = lo | (hi << 4);
        }
    }

    const size_t num_groups = K / bench::kGroupSize;
    w->scales.resize(N * num_groups);
    for (size_t i = 0; i < N * num_groups; i++)
        w->scales[i] = static_cast<__fp16>(raw_scales[i]);

    w->b_shape[0] = static_cast<int64_t>(N);
    w->b_shape[1] = static_cast<int64_t>(K / 2);
    w->s_shape[0] = static_cast<int64_t>(N);
    w->s_shape[1] = static_cast<int64_t>(num_groups);

    return w;
}

void* q4_prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new MlcActivations();
    a->fp16.resize(M * K);
    for (size_t i = 0; i < M * K; i++)
        a->fp16[i] = static_cast<__fp16>(fp32[i]);
    return a;
}

bench::BenchResult q4_run(size_t M, void* weights, void* activations,
                           const int8_t*, const float*,
                           const bench::BenchOptions& opt) {
    auto* w = static_cast<MlcInt4Weights*>(weights);
    auto* act = static_cast<MlcActivations*>(activations);
    std::vector<__fp16> output(M * w->N);

    for (int i = 0; i < opt.warmup; i++)
        q4_run_kernel(M, w, act, output.data());

    if (opt.capture_output)
        for (size_t i = 0; i < M * w->N; i++)
            opt.capture_output[i] = static_cast<float>(output[i]);

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; i++) {
        double t0 = bench::now_ms();
        q4_run_kernel(M, w, act, output.data());
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void q4_once(size_t M, void* weights, void* activations,
             const int8_t*, const float*) {
    auto* w = static_cast<MlcInt4Weights*>(weights);
    auto* act = static_cast<MlcActivations*>(activations);
    w->output_buf.resize(M * w->N);
    q4_run_kernel(M, w, act, w->output_buf.data());
}

void q4_cleanup(void* weights, void* activations) {
    delete static_cast<MlcInt4Weights*>(weights);
    if (activations) delete static_cast<MlcActivations*>(activations);
}

// ── INT8 benchmark plumbing ─────────────────────────────────────────────

static void q8_run_kernel(size_t M, MlcInt8Weights* w, MlcActivations* act,
                           __fp16* output) {
    int64_t a_shape[] = {static_cast<int64_t>(M), static_cast<int64_t>(w->K)};
    int64_t c_shape[] = {static_cast<int64_t>(M), static_cast<int64_t>(w->N)};

    DLTensor a_dl = make_dl(act->fp16.data(), a_shape, 2, kDLFloat, 16);
    DLTensor b_dl = make_dl(w->quantized.data(), w->b_shape, 2, kDLInt, 8);
    DLTensor s_dl = make_dl(w->scales.data(), w->s_shape, 2, kDLFloat, 16);
    DLTensor c_dl = make_dl(output, c_shape, 2, kDLFloat, 16);
    call_tvm_fn(s_q8_fn, &a_dl, &b_dl, &s_dl, &c_dl);
}

void* q8_prepare(const float* fp32, size_t N, size_t K) {
    auto* w = new MlcInt8Weights();
    w->K = K;
    w->N = N;

    std::vector<float> src(fp32, fp32 + N * K);
    std::vector<int8_t> rowmajor;
    std::vector<float> raw_scales;
    bench::quantize_int8_per_group(src, N, K, rowmajor, raw_scales);

    w->quantized = std::move(rowmajor);

    const size_t num_groups = K / bench::kGroupSize;
    w->scales.resize(N * num_groups);
    for (size_t i = 0; i < N * num_groups; i++)
        w->scales[i] = static_cast<__fp16>(raw_scales[i]);

    w->b_shape[0] = static_cast<int64_t>(N);
    w->b_shape[1] = static_cast<int64_t>(K);
    w->s_shape[0] = static_cast<int64_t>(N);
    w->s_shape[1] = static_cast<int64_t>(num_groups);

    return w;
}

void* q8_prepare_act(const float* fp32, size_t M, size_t K, void*) {
    auto* a = new MlcActivations();
    a->fp16.resize(M * K);
    for (size_t i = 0; i < M * K; i++)
        a->fp16[i] = static_cast<__fp16>(fp32[i]);
    return a;
}

bench::BenchResult q8_run(size_t M, void* weights, void* activations,
                           const int8_t*, const float*,
                           const bench::BenchOptions& opt) {
    auto* w = static_cast<MlcInt8Weights*>(weights);
    auto* act = static_cast<MlcActivations*>(activations);
    std::vector<__fp16> output(M * w->N);

    for (int i = 0; i < opt.warmup; i++)
        q8_run_kernel(M, w, act, output.data());

    if (opt.capture_output)
        for (size_t i = 0; i < M * w->N; i++)
            opt.capture_output[i] = static_cast<float>(output[i]);

    double total_ms = 0.0;
    for (int i = 0; i < opt.iterations; i++) {
        double t0 = bench::now_ms();
        q8_run_kernel(M, w, act, output.data());
        total_ms += bench::now_ms() - t0;
    }

    return {(total_ms * 1000.0) / opt.iterations,
            bench::compute_gops(M, w->K, w->N, opt.iterations, total_ms)};
}

void q8_once(size_t M, void* weights, void* activations,
             const int8_t*, const float*) {
    auto* w = static_cast<MlcInt8Weights*>(weights);
    auto* act = static_cast<MlcActivations*>(activations);
    w->output_buf.resize(M * w->N);
    q8_run_kernel(M, w, act, w->output_buf.data());
}

void q8_cleanup(void* weights, void* activations) {
    delete static_cast<MlcInt8Weights*>(weights);
    if (activations) delete static_cast<MlcActivations*>(activations);
}

// ── Registration ────────────────────────────────────────────────────────

static int reg = [] {
    s_q4_fn = register_tvm_func("quantized_matmul_int4", 21, tvm_q4_matmul_callback);
    s_q8_fn = register_tvm_func("quantized_matmul_int8", 21, tvm_q8_matmul_callback);

    if (s_q4_fn) {
        bench::register_backend({
            "mlc_int4", "mlc", bench::QuantCategory::INT4,
            q4_prepare, q4_prepare_act, q4_run, q4_cleanup, 0, q4_once
        });
    }
    if (s_q8_fn) {
        bench::register_backend({
            "mlc_int8", "mlc", bench::QuantCategory::INT8,
            q8_prepare, q8_prepare_act, q8_run, q8_cleanup, 0, q8_once
        });
    }
    return 0;
}();

} // namespace

#else

namespace {
[[maybe_unused]] static int reg = [] { return 0; }();
} // namespace

#endif
