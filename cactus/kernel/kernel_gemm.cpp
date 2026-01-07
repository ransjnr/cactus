#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace {
    constexpr size_t CACHE_LINE = 64;

    struct alignas(CACHE_LINE) PaddedRowData {
        float scale;
        std::atomic<uint8_t> state;
    };

    struct GemmBuffers {
        std::vector<int8_t> A_quant;
        std::unique_ptr<PaddedRowData[]> row_data;
        size_t row_data_size = 0;
    };
    static GemmBuffers gemm_buffers;

    enum class RowState : uint8_t { NOT_STARTED = 0, IN_PROGRESS = 1, DONE = 2 };
}

static void cactus_matmul_f16_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int VECTOR_WIDTH = 8;
    const size_t K_aligned = (K / (VECTOR_WIDTH * 2)) * (VECTOR_WIDTH * 2);

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            float16x8_t accumulators[TILE_M][TILE_N];
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n)
                    accumulators[m][n] = vdupq_n_f16(0.0);

            for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * 2) {
                float16x8_t a_vec_low[TILE_M], a_vec_high[TILE_M];
                float16x8_t b_vec_low[TILE_N], b_vec_high[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        a_vec_low[m] = vld1q_f16(&a[row * K + k_block]);
                        a_vec_high[m] = vld1q_f16(&a[row * K + k_block + VECTOR_WIDTH]);
                    } else {
                        a_vec_low[m] = vdupq_n_f16(0.0);
                        a_vec_high[m] = vdupq_n_f16(0.0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        b_vec_low[n] = vld1q_f16(&b_transposed[col * K + k_block]);
                        b_vec_high[n] = vld1q_f16(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                    } else {
                        b_vec_low[n] = vdupq_n_f16(0.0);
                        b_vec_high[n] = vdupq_n_f16(0.0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n) {
                        accumulators[m][n] = accum_f16_dot(accumulators[m][n], 
                                                          a_vec_low[m], a_vec_high[m],
                                                          b_vec_low[n], b_vec_high[n]);
                    }
            }

            for (size_t k_block = K_aligned; k_block < K; k_block += VECTOR_WIDTH) {
                size_t remaining = K - k_block;
                float16x8_t a_vec[TILE_M], b_vec[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        if (remaining >= VECTOR_WIDTH) {
                            a_vec[m] = vld1q_f16(&a[row * K + k_block]);
                        } else {
                            __fp16 tmp[VECTOR_WIDTH] = {0.0};
                            memcpy(tmp, &a[row * K + k_block], remaining * sizeof(__fp16));
                            a_vec[m] = vld1q_f16(tmp);
                        }
                    } else {
                        a_vec[m] = vdupq_n_f16(0.0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        if (remaining >= VECTOR_WIDTH) {
                            b_vec[n] = vld1q_f16(&b_transposed[col * K + k_block]);
                        } else {
                            __fp16 tmp[VECTOR_WIDTH] = {0.0};
                            memcpy(tmp, &b_transposed[col * K + k_block], remaining * sizeof(__fp16));
                            b_vec[n] = vld1q_f16(tmp);
                        }
                    } else {
                        b_vec[n] = vdupq_n_f16(0.0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n)
                        accumulators[m][n] = vfmaq_f16(accumulators[m][n], a_vec[m], b_vec[n]);
            }

            for (int m = 0; m < TILE_M; ++m) {
                size_t row = row_block + m;
                if (row >= M) continue;
                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col >= N) continue;
                    float16x4_t low = vget_low_f16(accumulators[m][n]);
                    float16x4_t high = vget_high_f16(accumulators[m][n]);
                    float16x4_t sum_vec = vadd_f16(low, high);
                    __fp16 sum = vget_lane_f16(sum_vec, 0) + vget_lane_f16(sum_vec, 1) + 
                                vget_lane_f16(sum_vec, 2) + vget_lane_f16(sum_vec, 3);
                    c[row * N + col] = sum;
                }
            }
        }
    }
}

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {
    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);

                cactus_matmul_f16_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row
                );
            }
        });
}

static inline float quantize_row_fp16_to_int8(const __fp16* src, int8_t* dst, size_t K) {
    float32x4_t max_vec0 = vdupq_n_f32(0.0f);
    float32x4_t max_vec1 = vdupq_n_f32(0.0f);
    size_t k = 0;

    for (; k + 16 <= K; k += 16) {
        float16x8_t v0 = vld1q_f16(src + k);
        float16x8_t v1 = vld1q_f16(src + k + 8);
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_low_f16(v0))));
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_high_f16(v0))));
        max_vec1 = vmaxq_f32(max_vec1, vabsq_f32(vcvt_f32_f16(vget_low_f16(v1))));
        max_vec1 = vmaxq_f32(max_vec1, vabsq_f32(vcvt_f32_f16(vget_high_f16(v1))));
    }

    max_vec0 = vmaxq_f32(max_vec0, max_vec1);

    for (; k + 8 <= K; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_low_f16(vals))));
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_high_f16(vals))));
    }

    float max_abs = vmaxvq_f32(max_vec0);

    for (; k < K; k++) {
        float val = fabsf((float)src[k]);
        if (val > max_abs) max_abs = val;
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);

    k = 0;

    for (; k + 16 <= K; k += 16) {
        float16x8_t v0 = vld1q_f16(src + k);
        float16x8_t v1 = vld1q_f16(src + k + 8);

        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v0)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v0)), inv_scale_vec));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v1)), inv_scale_vec));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v1)), inv_scale_vec));

        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);
        int16x8_t s01 = vcombine_s16(s0, s1);
        int16x8_t s23 = vcombine_s16(s2, s3);
        int8x8_t r0 = vqmovn_s16(s01);
        int8x8_t r1 = vqmovn_s16(s23);
        vst1q_s8(dst + k, vcombine_s8(r0, r1));
    }

    for (; k + 8 <= K; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(vals)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(vals)), inv_scale_vec));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int8x8_t result = vqmovn_s16(vcombine_s16(s0, s1));
        vst1_s8(dst + k, result);
    }

    for (; k < K; k++) {
        float val = (float)src[k] * inv_scale;
        int32_t q = (int32_t)roundf(val);
        q = std::max(-128, std::min(127, q));
        dst[k] = (int8_t)q;
    }

    return scale;
}

void cactus_matmul_int8(
    const __fp16* A,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    #if defined(__APPLE__) && defined(__arm64__)
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 8;
    #else
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 4;
    #endif

    const size_t num_groups = K / group_size;
    const size_t K_aligned = ((K + group_size - 1) / group_size) * group_size;

    const size_t quant_size = M * K_aligned;
    if (gemm_buffers.A_quant.size() < quant_size) {
        gemm_buffers.A_quant.resize(quant_size);
    }
    if (gemm_buffers.row_data_size < M) {
        gemm_buffers.row_data = std::make_unique<PaddedRowData[]>(M);
        gemm_buffers.row_data_size = M;
    }

    for (size_t m = 0; m < M; m++) {
        gemm_buffers.row_data[m].state.store(static_cast<uint8_t>(RowState::NOT_STARTED),
            std::memory_order_relaxed);
    }

    int8_t* A_quant = gemm_buffers.A_quant.data();
    auto* row_data = gemm_buffers.row_data.get();

    constexpr size_t MAX_GROUPS = 64;
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t num_col_tiles = (N + TILE_N - 1) / TILE_N;
    const size_t total_tiles = num_row_tiles * num_col_tiles;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const size_t tile_row = tile_idx / num_col_tiles;
            const size_t tile_col = tile_idx % num_col_tiles;
            const size_t m_start = tile_row * TILE_M;
            const size_t m_end = std::min(m_start + TILE_M, M);
            const size_t n_start = tile_col * TILE_N;
            const size_t n_end = std::min(n_start + TILE_N, N);
            const size_t actual_m = m_end - m_start;
            const size_t actual_n = n_end - n_start;

            for (size_t m = m_start; m < m_end; m++) {
                uint8_t expected = static_cast<uint8_t>(RowState::NOT_STARTED);

                if (row_data[m].state.compare_exchange_strong(expected,
                        static_cast<uint8_t>(RowState::IN_PROGRESS),
                        std::memory_order_acq_rel)) {

                    row_data[m].scale = quantize_row_fp16_to_int8(
                        A + m * K, A_quant + m * K_aligned, K);

                    row_data[m].state.store(static_cast<uint8_t>(RowState::DONE),
                        std::memory_order_release);

                } else {
                    while (row_data[m].state.load(std::memory_order_acquire) !=
                           static_cast<uint8_t>(RowState::DONE)) {
                        #if defined(__arm__) || defined(__aarch64__)
                            __asm__ volatile("yield");
                        #endif
                    }
                }
            }

            int32_t all_group_acc[MAX_GROUPS][TILE_M][TILE_N] = {{{0}}};

            for (size_t ni = 0; ni < actual_n; ni++) {
                __builtin_prefetch(&B_scales[(n_start + ni) * num_groups], 0, 3);
            }

#if defined(CACTUS_HAS_I8MM)
            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;

                size_t mi = 0;
                for (; mi + 1 < actual_m; mi += 2) {
                    size_t ni = 0;
                    for (; ni + 3 < actual_n; ni += 4) {
                        int32x4_t acc01 = vdupq_n_s32(0);  
                        int32x4_t acc23 = vdupq_n_s32(0); 

                        const int8_t* a_base0 = A_quant + (m_start + mi) * K_aligned + k_base;
                        const int8_t* a_base1 = A_quant + (m_start + mi + 1) * K_aligned + k_base;
                        const int8_t* b_base0 = B + (n_start + ni) * K + k_base;
                        const int8_t* b_base1 = B + (n_start + ni + 1) * K + k_base;
                        const int8_t* b_base2 = B + (n_start + ni + 2) * K + k_base;
                        const int8_t* b_base3 = B + (n_start + ni + 3) * K + k_base;

                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                            #pragma unroll
                            for (int kk = 0; kk < 64; kk += 16) {
                                int8x16_t a_lo = vcombine_s8(vld1_s8(a_base0 + k_offset + kk),
                                                              vld1_s8(a_base1 + k_offset + kk));
                                int8x16_t a_hi = vcombine_s8(vld1_s8(a_base0 + k_offset + kk + 8),
                                                              vld1_s8(a_base1 + k_offset + kk + 8));

                                int8x16_t b01_lo = vcombine_s8(vld1_s8(b_base0 + k_offset + kk),
                                                                vld1_s8(b_base1 + k_offset + kk));
                                int8x16_t b01_hi = vcombine_s8(vld1_s8(b_base0 + k_offset + kk + 8),
                                                                vld1_s8(b_base1 + k_offset + kk + 8));
                                acc01 = accum_matmul(acc01, a_lo, b01_lo);
                                acc01 = accum_matmul(acc01, a_hi, b01_hi);

                                int8x16_t b23_lo = vcombine_s8(vld1_s8(b_base2 + k_offset + kk),
                                                                vld1_s8(b_base3 + k_offset + kk));
                                int8x16_t b23_hi = vcombine_s8(vld1_s8(b_base2 + k_offset + kk + 8),
                                                                vld1_s8(b_base3 + k_offset + kk + 8));
                                acc23 = accum_matmul(acc23, a_lo, b23_lo);
                                acc23 = accum_matmul(acc23, a_hi, b23_hi);
                            }
                        }

                        all_group_acc[g][mi][ni] += vgetq_lane_s32(acc01, 0);
                        all_group_acc[g][mi][ni + 1] += vgetq_lane_s32(acc01, 1);
                        all_group_acc[g][mi + 1][ni] += vgetq_lane_s32(acc01, 2);
                        all_group_acc[g][mi + 1][ni + 1] += vgetq_lane_s32(acc01, 3);
                        all_group_acc[g][mi][ni + 2] += vgetq_lane_s32(acc23, 0);
                        all_group_acc[g][mi][ni + 3] += vgetq_lane_s32(acc23, 1);
                        all_group_acc[g][mi + 1][ni + 2] += vgetq_lane_s32(acc23, 2);
                        all_group_acc[g][mi + 1][ni + 3] += vgetq_lane_s32(acc23, 3);
                    }

                    for (; ni + 1 < actual_n; ni += 2) {
                        int32x4_t acc0 = vdupq_n_s32(0);

                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                            const int8_t* a_ptr0 = A_quant + (m_start + mi) * K_aligned + k_base + k_offset;
                            const int8_t* a_ptr1 = A_quant + (m_start + mi + 1) * K_aligned + k_base + k_offset;
                            const int8_t* b_ptr0 = B + (n_start + ni) * K + k_base + k_offset;
                            const int8_t* b_ptr1 = B + (n_start + ni + 1) * K + k_base + k_offset;

                            for (int kk = 0; kk < 64; kk += 8) {
                                int8x16_t a_vec = vcombine_s8(vld1_s8(a_ptr0 + kk), vld1_s8(a_ptr1 + kk));
                                int8x16_t b_vec = vcombine_s8(vld1_s8(b_ptr0 + kk), vld1_s8(b_ptr1 + kk));
                                acc0 = accum_matmul(acc0, a_vec, b_vec);
                            }
                        }

                        all_group_acc[g][mi][ni] += vgetq_lane_s32(acc0, 0);
                        all_group_acc[g][mi][ni + 1] += vgetq_lane_s32(acc0, 1);
                        all_group_acc[g][mi + 1][ni] += vgetq_lane_s32(acc0, 2);
                        all_group_acc[g][mi + 1][ni + 1] += vgetq_lane_s32(acc0, 3);
                    }

                    for (; ni < actual_n; ni++) {
                        for (size_t mii = mi; mii < mi + 2; mii++) {
                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                                const int8_t* a_ptr = A_quant + (m_start + mii) * K_aligned + k_base + k_offset;
                                const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                                int32x4_t sum = vdupq_n_s32(0);
                                sum = accum_dot(sum, vld1q_s8(a_ptr), vld1q_s8(b_ptr));
                                sum = accum_dot(sum, vld1q_s8(a_ptr + 16), vld1q_s8(b_ptr + 16));
                                sum = accum_dot(sum, vld1q_s8(a_ptr + 32), vld1q_s8(b_ptr + 32));
                                sum = accum_dot(sum, vld1q_s8(a_ptr + 48), vld1q_s8(b_ptr + 48));
                                all_group_acc[g][mii][ni] += vaddvq_s32(sum);
                            }
                        }
                    }
                }

                for (; mi < actual_m; mi++) {
                    for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                        const int8_t* a_ptr = A_quant + (m_start + mi) * K_aligned + k_base + k_offset;
                        int8x16_t a_vec0 = vld1q_s8(a_ptr);
                        int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                        int8x16_t a_vec2 = vld1q_s8(a_ptr + 32);
                        int8x16_t a_vec3 = vld1q_s8(a_ptr + 48);

                        for (size_t ni = 0; ni < actual_n; ni++) {
                            const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                            int32x4_t sum = vdupq_n_s32(0);
                            sum = accum_dot(sum, a_vec0, vld1q_s8(b_ptr));
                            sum = accum_dot(sum, a_vec1, vld1q_s8(b_ptr + 16));
                            sum = accum_dot(sum, a_vec2, vld1q_s8(b_ptr + 32));
                            sum = accum_dot(sum, a_vec3, vld1q_s8(b_ptr + 48));
                            all_group_acc[g][mi][ni] += vaddvq_s32(sum);
                        }
                    }
                }
            }
#else
            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;

                for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                    int8x16_t b_vec0[TILE_N], b_vec1[TILE_N], b_vec2[TILE_N], b_vec3[TILE_N];
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                        b_vec0[ni] = vld1q_s8(b_ptr);
                        b_vec1[ni] = vld1q_s8(b_ptr + 16);
                        b_vec2[ni] = vld1q_s8(b_ptr + 32);
                        b_vec3[ni] = vld1q_s8(b_ptr + 48);
                    }

                    for (size_t mi = 0; mi < actual_m; mi++) {
                        const int8_t* a_ptr = A_quant + (m_start + mi) * K_aligned + k_base + k_offset;
                        int8x16_t a_vec0 = vld1q_s8(a_ptr);
                        int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                        int8x16_t a_vec2 = vld1q_s8(a_ptr + 32);
                        int8x16_t a_vec3 = vld1q_s8(a_ptr + 48);

                        for (size_t ni = 0; ni < actual_n; ni++) {
                            int32x4_t sum = vdupq_n_s32(0);
                            sum = accum_dot(sum, a_vec0, b_vec0[ni]);
                            sum = accum_dot(sum, a_vec1, b_vec1[ni]);
                            sum = accum_dot(sum, a_vec2, b_vec2[ni]);
                            sum = accum_dot(sum, a_vec3, b_vec3[ni]);
                            all_group_acc[g][mi][ni] += vaddvq_s32(sum);
                        }
                    }
                }
            }
#endif

            for (size_t mi = 0; mi < actual_m; mi++) {
                const float a_scale = row_data[m_start + mi].scale;
                for (size_t ni = 0; ni < actual_n; ni++) {
                    const __fp16* col_scales = &B_scales[(n_start + ni) * num_groups];
                    float sum = 0.0f;
                    for (size_t g = 0; g < num_groups; g++) {
                        sum += (float)all_group_acc[g][mi][ni] * (float)col_scales[g];
                    }
                    C[(m_start + mi) * N + (n_start + ni)] = (__fp16)(sum * a_scale);
                }
            }
            } // tile_idx
        });
}

void cactus_matmul_int4(
    const __fp16* A,
    const uint8_t* B_packed, 
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    #if defined(__APPLE__) && defined(__arm64__)
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 8;
    #else
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 4;
    #endif

    const size_t num_groups = K / group_size;
    const size_t K_aligned = ((K + group_size - 1) / group_size) * group_size;
    const size_t K_packed = K / 2;  

    const size_t quant_size = M * K_aligned;
    if (gemm_buffers.A_quant.size() < quant_size) {
        gemm_buffers.A_quant.resize(quant_size);
    }
    if (gemm_buffers.row_data_size < M) {
        gemm_buffers.row_data = std::make_unique<PaddedRowData[]>(M);
        gemm_buffers.row_data_size = M;
    }

    for (size_t m = 0; m < M; m++) {
        gemm_buffers.row_data[m].state.store(static_cast<uint8_t>(RowState::NOT_STARTED),
            std::memory_order_relaxed);
    }

    int8_t* A_quant = gemm_buffers.A_quant.data();
    auto* row_data = gemm_buffers.row_data.get();

    constexpr size_t MAX_GROUPS = 64;
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t num_col_tiles = (N + TILE_N - 1) / TILE_N;
    const size_t total_tiles = num_row_tiles * num_col_tiles;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
            const size_t tile_row = tile_idx / num_col_tiles;
            const size_t tile_col = tile_idx % num_col_tiles;
            const size_t m_start = tile_row * TILE_M;
            const size_t m_end = std::min(m_start + TILE_M, M);
            const size_t n_start = tile_col * TILE_N;
            const size_t n_end = std::min(n_start + TILE_N, N);
            const size_t actual_m = m_end - m_start;
            const size_t actual_n = n_end - n_start;

            for (size_t m = m_start; m < m_end; m++) {
                uint8_t expected = static_cast<uint8_t>(RowState::NOT_STARTED);

                if (row_data[m].state.compare_exchange_strong(expected,
                        static_cast<uint8_t>(RowState::IN_PROGRESS),
                        std::memory_order_acq_rel)) {

                    row_data[m].scale = quantize_row_fp16_to_int8(
                        A + m * K, A_quant + m * K_aligned, K);

                    row_data[m].state.store(static_cast<uint8_t>(RowState::DONE),
                        std::memory_order_release);

                } else {
                    while (row_data[m].state.load(std::memory_order_acquire) !=
                           static_cast<uint8_t>(RowState::DONE)) {
                        #if defined(__arm__) || defined(__aarch64__)
                            __asm__ volatile("yield");
                        #endif
                    }
                }
            }

            int32_t all_group_acc[MAX_GROUPS][TILE_M][TILE_N] = {{{0}}};

            for (size_t ni = 0; ni < actual_n; ni++) {
                __builtin_prefetch(&B_scales[(n_start + ni) * num_groups], 0, 3);
            }

#if defined(CACTUS_HAS_I8MM)
            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const size_t k_base_packed = k_base / 2;

                size_t mi = 0;
                for (; mi + 1 < actual_m; mi += 2) {
                    const int8_t* a_base0 = A_quant + (m_start + mi) * K_aligned + k_base;
                    const int8_t* a_base1 = A_quant + (m_start + mi + 1) * K_aligned + k_base;

                    size_t ni = 0;
                    for (; ni + 3 < actual_n; ni += 4) {
                        int32x4_t acc01 = vdupq_n_s32(0);
                        int32x4_t acc23 = vdupq_n_s32(0);

                        const uint8_t* b_base0 = B_packed + (n_start + ni) * K_packed + k_base_packed;
                        const uint8_t* b_base1 = B_packed + (n_start + ni + 1) * K_packed + k_base_packed;
                        const uint8_t* b_base2 = B_packed + (n_start + ni + 2) * K_packed + k_base_packed;
                        const uint8_t* b_base3 = B_packed + (n_start + ni + 3) * K_packed + k_base_packed;

                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 32) {
                            const size_t k_offset_packed = k_offset / 2;

                            uint8x16_t p0 = vld1q_u8(b_base0 + k_offset_packed);
                            uint8x16_t p1 = vld1q_u8(b_base1 + k_offset_packed);
                            uint8x16_t p2 = vld1q_u8(b_base2 + k_offset_packed);
                            uint8x16_t p3 = vld1q_u8(b_base3 + k_offset_packed);

                            int8x16_t b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi, b3_lo, b3_hi;
                            unpack_int4_to_int8x32(p0, b0_lo, b0_hi);
                            unpack_int4_to_int8x32(p1, b1_lo, b1_hi);
                            unpack_int4_to_int8x32(p2, b2_lo, b2_hi);
                            unpack_int4_to_int8x32(p3, b3_lo, b3_hi);

                            #pragma unroll
                            for (int chunk = 0; chunk < 4; chunk++) {
                                size_t k_pos = k_offset + chunk * 8;

                                int8x8_t b0_8 = (chunk < 2) ?
                                    ((chunk == 0) ? vget_low_s8(b0_lo) : vget_high_s8(b0_lo)) :
                                    ((chunk == 2) ? vget_low_s8(b0_hi) : vget_high_s8(b0_hi));
                                int8x8_t b1_8 = (chunk < 2) ?
                                    ((chunk == 0) ? vget_low_s8(b1_lo) : vget_high_s8(b1_lo)) :
                                    ((chunk == 2) ? vget_low_s8(b1_hi) : vget_high_s8(b1_hi));
                                int8x8_t b2_8 = (chunk < 2) ?
                                    ((chunk == 0) ? vget_low_s8(b2_lo) : vget_high_s8(b2_lo)) :
                                    ((chunk == 2) ? vget_low_s8(b2_hi) : vget_high_s8(b2_hi));
                                int8x8_t b3_8 = (chunk < 2) ?
                                    ((chunk == 0) ? vget_low_s8(b3_lo) : vget_high_s8(b3_lo)) :
                                    ((chunk == 2) ? vget_low_s8(b3_hi) : vget_high_s8(b3_hi));

                                int8x8_t a0_8 = vld1_s8(a_base0 + k_pos);
                                int8x8_t a1_8 = vld1_s8(a_base1 + k_pos);

                                int8x16_t a_combined = vcombine_s8(a0_8, a1_8);
                                int8x16_t b01_combined = vcombine_s8(b0_8, b1_8);
                                int8x16_t b23_combined = vcombine_s8(b2_8, b3_8);

                                acc01 = accum_matmul(acc01, a_combined, b01_combined);
                                acc23 = accum_matmul(acc23, a_combined, b23_combined);
                            }
                        }

                        all_group_acc[g][mi][ni] += vgetq_lane_s32(acc01, 0);
                        all_group_acc[g][mi][ni + 1] += vgetq_lane_s32(acc01, 1);
                        all_group_acc[g][mi + 1][ni] += vgetq_lane_s32(acc01, 2);
                        all_group_acc[g][mi + 1][ni + 1] += vgetq_lane_s32(acc01, 3);
                        all_group_acc[g][mi][ni + 2] += vgetq_lane_s32(acc23, 0);
                        all_group_acc[g][mi][ni + 3] += vgetq_lane_s32(acc23, 1);
                        all_group_acc[g][mi + 1][ni + 2] += vgetq_lane_s32(acc23, 2);
                        all_group_acc[g][mi + 1][ni + 3] += vgetq_lane_s32(acc23, 3);
                    }

                    for (; ni + 1 < actual_n; ni += 2) {
                        int32x4_t acc = vdupq_n_s32(0);
                        const uint8_t* b_base0 = B_packed + (n_start + ni) * K_packed + k_base_packed;
                        const uint8_t* b_base1 = B_packed + (n_start + ni + 1) * K_packed + k_base_packed;

                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 32) {
                            const size_t k_offset_packed = k_offset / 2;
                            uint8x16_t p0 = vld1q_u8(b_base0 + k_offset_packed);
                            uint8x16_t p1 = vld1q_u8(b_base1 + k_offset_packed);

                            int8x16_t b0_lo, b0_hi, b1_lo, b1_hi;
                            unpack_int4_to_int8x32(p0, b0_lo, b0_hi);
                            unpack_int4_to_int8x32(p1, b1_lo, b1_hi);

                            for (int chunk = 0; chunk < 4; chunk++) {
                                size_t k_pos = k_offset + chunk * 8;
                                int8x8_t b0_8 = (chunk < 2) ?
                                    ((chunk == 0) ? vget_low_s8(b0_lo) : vget_high_s8(b0_lo)) :
                                    ((chunk == 2) ? vget_low_s8(b0_hi) : vget_high_s8(b0_hi));
                                int8x8_t b1_8 = (chunk < 2) ?
                                    ((chunk == 0) ? vget_low_s8(b1_lo) : vget_high_s8(b1_lo)) :
                                    ((chunk == 2) ? vget_low_s8(b1_hi) : vget_high_s8(b1_hi));

                                int8x8_t a0_8 = vld1_s8(a_base0 + k_pos);
                                int8x8_t a1_8 = vld1_s8(a_base1 + k_pos);

                                acc = accum_matmul(acc, vcombine_s8(a0_8, a1_8), vcombine_s8(b0_8, b1_8));
                            }
                        }

                        all_group_acc[g][mi][ni] += vgetq_lane_s32(acc, 0);
                        all_group_acc[g][mi][ni + 1] += vgetq_lane_s32(acc, 1);
                        all_group_acc[g][mi + 1][ni] += vgetq_lane_s32(acc, 2);
                        all_group_acc[g][mi + 1][ni + 1] += vgetq_lane_s32(acc, 3);
                    }

                    // Single remaining column - fallback to dot
                    for (; ni < actual_n; ni++) {
                        const uint8_t* b_ptr = B_packed + (n_start + ni) * K_packed + k_base_packed;
                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                            const size_t k_offset_packed = k_offset / 2;
                            uint8x16_t packed0 = vld1q_u8(b_ptr + k_offset_packed);
                            uint8x16_t packed1 = vld1q_u8(b_ptr + k_offset_packed + 16);
                            int8x16_t b0, b1, b2, b3;
                            unpack_int4_to_int8x32(packed0, b0, b1);
                            unpack_int4_to_int8x32(packed1, b2, b3);

                            for (size_t mii = mi; mii < mi + 2; mii++) {
                                const int8_t* a_ptr = A_quant + (m_start + mii) * K_aligned + k_base + k_offset;
                                int32x4_t sum = vdupq_n_s32(0);
                                sum = accum_dot(sum, vld1q_s8(a_ptr), b0);
                                sum = accum_dot(sum, vld1q_s8(a_ptr + 16), b1);
                                sum = accum_dot(sum, vld1q_s8(a_ptr + 32), b2);
                                sum = accum_dot(sum, vld1q_s8(a_ptr + 48), b3);
                                all_group_acc[g][mii][ni] += vaddvq_s32(sum);
                            }
                        }
                    }
                }

                for (; mi < actual_m; mi++) {
                    const int8_t* a_base = A_quant + (m_start + mi) * K_aligned + k_base;
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        const uint8_t* b_ptr = B_packed + (n_start + ni) * K_packed + k_base_packed;
                        for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                            const size_t k_offset_packed = k_offset / 2;
                            uint8x16_t packed0 = vld1q_u8(b_ptr + k_offset_packed);
                            uint8x16_t packed1 = vld1q_u8(b_ptr + k_offset_packed + 16);
                            int8x16_t b0, b1, b2, b3;
                            unpack_int4_to_int8x32(packed0, b0, b1);
                            unpack_int4_to_int8x32(packed1, b2, b3);

                            const int8_t* a_ptr = a_base + k_offset;
                            int32x4_t sum = vdupq_n_s32(0);
                            sum = accum_dot(sum, vld1q_s8(a_ptr), b0);
                            sum = accum_dot(sum, vld1q_s8(a_ptr + 16), b1);
                            sum = accum_dot(sum, vld1q_s8(a_ptr + 32), b2);
                            sum = accum_dot(sum, vld1q_s8(a_ptr + 48), b3);
                            all_group_acc[g][mi][ni] += vaddvq_s32(sum);
                        }
                    }
                }
            }
#else
            for (size_t g = 0; g < num_groups; g++) {
                const size_t k_base = g * group_size;
                const size_t k_base_packed = k_base / 2;

                for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                    const size_t k_offset_packed = k_offset / 2;

                    int8x16_t b_unpacked[TILE_N][4];

                    for (size_t ni = 0; ni < actual_n; ni++) {
                        const uint8_t* b_ptr = B_packed + (n_start + ni) * K_packed + k_base_packed + k_offset_packed;

                        uint8x16_t packed0 = vld1q_u8(b_ptr);
                        uint8x16_t packed1 = vld1q_u8(b_ptr + 16);

                        unpack_int4_to_int8x32(packed0, b_unpacked[ni][0], b_unpacked[ni][1]);
                        unpack_int4_to_int8x32(packed1, b_unpacked[ni][2], b_unpacked[ni][3]);
                    }

                    for (size_t mi = 0; mi < actual_m; mi++) {
                        const int8_t* a_ptr = A_quant + (m_start + mi) * K_aligned + k_base + k_offset;
                        int8x16_t a_vec0 = vld1q_s8(a_ptr);
                        int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                        int8x16_t a_vec2 = vld1q_s8(a_ptr + 32);
                        int8x16_t a_vec3 = vld1q_s8(a_ptr + 48);

                        for (size_t ni = 0; ni < actual_n; ni++) {
                            int32x4_t sum = vdupq_n_s32(0);
                            sum = accum_dot(sum, a_vec0, b_unpacked[ni][0]);
                            sum = accum_dot(sum, a_vec1, b_unpacked[ni][1]);
                            sum = accum_dot(sum, a_vec2, b_unpacked[ni][2]);
                            sum = accum_dot(sum, a_vec3, b_unpacked[ni][3]);
                            all_group_acc[g][mi][ni] += vaddvq_s32(sum);
                        }
                    }
                }
            }
#endif

            for (size_t mi = 0; mi < actual_m; mi++) {
                const float a_scale = row_data[m_start + mi].scale;
                for (size_t ni = 0; ni < actual_n; ni++) {
                    const __fp16* col_scales = &B_scales[(n_start + ni) * num_groups];
                    float sum = 0.0f;
                    for (size_t g = 0; g < num_groups; g++) {
                        sum += (float)all_group_acc[g][mi][ni] * (float)col_scales[g];
                    }
                    C[(m_start + mi) * N + (n_start + ni)] = (__fp16)(sum * a_scale);
                }
            }
            } // tile_idx
        });
}