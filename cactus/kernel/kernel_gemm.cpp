#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

static inline __fp16 hsum_f16x8(float16x8_t v) {
    float16x4_t lo = vget_low_f16(v);
    float16x4_t hi = vget_high_f16(v);
    float16x4_t sum4 = vadd_f16(lo, hi);
    float16x4_t sum2 = vadd_f16(sum4, vext_f16(sum4, sum4, 2));
    float16x4_t sum1 = vadd_f16(sum2, vext_f16(sum2, sum2, 1));
    return vget_lane_f16(sum1, 0);
}

static void cactus_matmul_f16_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t /*M*/,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    const size_t K16 = (K / 16) * 16;
    const size_t K8 = (K / 8) * 8;

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        const size_t m_end = std::min(row_block + TILE_M, end_row);

        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            const size_t n_end = std::min(col_block + TILE_N, N);

            float16x8_t acc[TILE_M][TILE_N];
            for (size_t m = 0; m < TILE_M; ++m)
                for (size_t n = 0; n < TILE_N; ++n)
                    acc[m][n] = vdupq_n_f16(0);

            for (size_t k = 0; k < K16; k += 16) {
                float16x8_t a0_lo = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a0_hi = (row_block < m_end) ? vld1q_f16(a + row_block * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a1_lo = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a1_hi = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a2_lo = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_hi = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k + 8) : vdupq_n_f16(0);
                float16x8_t a3_lo = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_hi = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k + 8) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_lo = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    float16x8_t b_hi = vld1q_f16(b_transposed + (col_block + ni) * K + k + 8);

                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_lo, b_lo);
                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_hi, b_hi);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_lo, b_lo);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_hi, b_hi);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_lo, b_lo);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_hi, b_hi);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_lo, b_lo);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_hi, b_hi);
                }
            }

            for (size_t k = K16; k < K8; k += 8) {
                float16x8_t a0_v = (row_block < m_end) ? vld1q_f16(a + row_block * K + k) : vdupq_n_f16(0);
                float16x8_t a1_v = (row_block + 1 < m_end) ? vld1q_f16(a + (row_block + 1) * K + k) : vdupq_n_f16(0);
                float16x8_t a2_v = (row_block + 2 < m_end) ? vld1q_f16(a + (row_block + 2) * K + k) : vdupq_n_f16(0);
                float16x8_t a3_v = (row_block + 3 < m_end) ? vld1q_f16(a + (row_block + 3) * K + k) : vdupq_n_f16(0);

                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    float16x8_t b_v = vld1q_f16(b_transposed + (col_block + ni) * K + k);
                    acc[0][ni] = vfmaq_f16(acc[0][ni], a0_v, b_v);
                    acc[1][ni] = vfmaq_f16(acc[1][ni], a1_v, b_v);
                    acc[2][ni] = vfmaq_f16(acc[2][ni], a2_v, b_v);
                    acc[3][ni] = vfmaq_f16(acc[3][ni], a3_v, b_v);
                }
            }

            for (size_t k = K8; k < K; ++k) {
                for (size_t mi = 0; mi < TILE_M && row_block + mi < m_end; ++mi) {
                    __fp16 av = a[(row_block + mi) * K + k];
                    for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                        __fp16 bv = b_transposed[(col_block + ni) * K + k];
                        acc[mi][ni] = vsetq_lane_f16(vgetq_lane_f16(acc[mi][ni], 0) + av * bv, acc[mi][ni], 0);
                    }
                }
            }

            for (size_t mi = 0; mi < TILE_M && row_block + mi < m_end; ++mi) {
                for (size_t ni = 0; ni < TILE_N && col_block + ni < n_end; ++ni) {
                    c[(row_block + mi) * N + col_block + ni] = hsum_f16x8(acc[mi][ni]);
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


void cactus_matmul_int8(
    const int8_t* A, 
    const float* A_scales, 
    const int8_t* B, 
    const __fp16* B_scales, 
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;  

    const size_t num_groups = K / group_size;
    const size_t N_blocks = (N + 3) / 4;  
    const size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
    const size_t total_tiles = num_row_tiles * N_blocks;

    CactusThreading::parallel_gemm_tiles(M, total_tiles,
        [=](size_t tile_start, size_t tile_end) {
            for (size_t tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                const size_t tile_row = tile_idx / N_blocks;
                const size_t n_block = tile_idx % N_blocks;
                const size_t m_start = tile_row * TILE_M;
                const size_t m_end = std::min(m_start + TILE_M, M);
                const size_t n_start = n_block * 4;
                const size_t n_end = std::min(n_start + 4, N);
                const size_t actual_m = m_end - m_start;
                const size_t actual_n = n_end - n_start;

                float running_sum[TILE_M][TILE_N] = {{0.0f}};
                __builtin_prefetch(&B_scales[n_block * num_groups * 4], 0, 3);


                for (size_t g = 0; g < num_groups; g++) {
                    const size_t k_base = g * group_size;

                    int32_t group_acc[TILE_M][TILE_N] = {{0}};

                    const int8_t* b_block_base = B + (n_block * K + k_base) * 4;

                    __builtin_prefetch(b_block_base, 0, 3);
                    __builtin_prefetch(b_block_base + 64, 0, 3);

                    if (cactus_has_i8mm() && actual_m >= 2) {
                        size_t mi = 0;
                        for (; mi + 1 < actual_m; mi += 2) {
                            const int8_t* a_base0 = A + (m_start + mi) * K + k_base;
                            const int8_t* a_base1 = A + (m_start + mi + 1) * K + k_base;

                            int32x4_t acc01 = vdupq_n_s32(0);
                            int32x4_t acc23 = vdupq_n_s32(0);

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 16) {
                                int8x16x4_t b_cols = vld4q_s8(b_block_base + k_offset * 4);

                                int8x8_t a0_lo = vld1_s8(a_base0 + k_offset);
                                int8x8_t a1_lo = vld1_s8(a_base1 + k_offset);

                                int8x8_t a0_hi = vld1_s8(a_base0 + k_offset + 8);
                                int8x8_t a1_hi = vld1_s8(a_base1 + k_offset + 8);

                                int8x16_t a_combined_lo = vcombine_s8(a0_lo, a1_lo);
                                int8x16_t a_combined_hi = vcombine_s8(a0_hi, a1_hi);

                                int8x16_t b01_lo = vcombine_s8(vget_low_s8(b_cols.val[0]), vget_low_s8(b_cols.val[1]));
                                int8x16_t b01_hi = vcombine_s8(vget_high_s8(b_cols.val[0]), vget_high_s8(b_cols.val[1]));

                                int8x16_t b23_lo = vcombine_s8(vget_low_s8(b_cols.val[2]), vget_low_s8(b_cols.val[3]));
                                int8x16_t b23_hi = vcombine_s8(vget_high_s8(b_cols.val[2]), vget_high_s8(b_cols.val[3]));

                                acc01 = accum_matmul(acc01, a_combined_lo, b01_lo);
                                acc01 = accum_matmul(acc01, a_combined_hi, b01_hi);

                                acc23 = accum_matmul(acc23, a_combined_lo, b23_lo);
                                acc23 = accum_matmul(acc23, a_combined_hi, b23_hi);
                            }

                            group_acc[mi][0] += vgetq_lane_s32(acc01, 0);
                            group_acc[mi][1] += vgetq_lane_s32(acc01, 1);

                            group_acc[mi + 1][0] += vgetq_lane_s32(acc01, 2);
                            group_acc[mi + 1][1] += vgetq_lane_s32(acc01, 3);

                            if (actual_n > 2) {
                                group_acc[mi][2] += vgetq_lane_s32(acc23, 0);
                                group_acc[mi][3] += vgetq_lane_s32(acc23, 1);

                                group_acc[mi + 1][2] += vgetq_lane_s32(acc23, 2);
                                group_acc[mi + 1][3] += vgetq_lane_s32(acc23, 3);
                            }
                        }

                        for (; mi < actual_m; mi++) {
                            const int8_t* a_ptr = A + (m_start + mi) * K + k_base;
                            int32x4_t acc[TILE_N] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 16) {
                                int8x16_t a_vec = vld1q_s8(a_ptr + k_offset);
                                int8x16x4_t b_cols = vld4q_s8(b_block_base + k_offset * 4);

                                acc[0] = accum_dot(acc[0], a_vec, b_cols.val[0]);
                                acc[1] = accum_dot(acc[1], a_vec, b_cols.val[1]);
                                acc[2] = accum_dot(acc[2], a_vec, b_cols.val[2]);
                                acc[3] = accum_dot(acc[3], a_vec, b_cols.val[3]);
                            }

                            for (size_t ni = 0; ni < actual_n; ni++) {
                                group_acc[mi][ni] += vaddvq_s32(acc[ni]);
                            }
                        }
                    } else {
                        size_t mi = 0;
                        for (; mi + 1 < actual_m; mi += 2) {
                            const int8_t* a_ptr0 = A + (m_start + mi) * K + k_base;
                            const int8_t* a_ptr1 = A + (m_start + mi + 1) * K + k_base;

                            int32x4_t acc0[TILE_N] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
                            int32x4_t acc1[TILE_N] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 16) {
                                int8x16x4_t b_cols = vld4q_s8(b_block_base + k_offset * 4);
                                int8x16_t a_vec0 = vld1q_s8(a_ptr0 + k_offset);
                                int8x16_t a_vec1 = vld1q_s8(a_ptr1 + k_offset);

                                acc0[0] = accum_dot(acc0[0], a_vec0, b_cols.val[0]);
                                acc0[1] = accum_dot(acc0[1], a_vec0, b_cols.val[1]);
                                acc0[2] = accum_dot(acc0[2], a_vec0, b_cols.val[2]);
                                acc0[3] = accum_dot(acc0[3], a_vec0, b_cols.val[3]);

                                acc1[0] = accum_dot(acc1[0], a_vec1, b_cols.val[0]);
                                acc1[1] = accum_dot(acc1[1], a_vec1, b_cols.val[1]);
                                acc1[2] = accum_dot(acc1[2], a_vec1, b_cols.val[2]);
                                acc1[3] = accum_dot(acc1[3], a_vec1, b_cols.val[3]);
                            }

                            for (size_t ni = 0; ni < actual_n; ni++) {
                                group_acc[mi][ni] += vaddvq_s32(acc0[ni]);
                                group_acc[mi + 1][ni] += vaddvq_s32(acc1[ni]);
                            }
                        }

                        for (; mi < actual_m; mi++) {
                            const int8_t* a_ptr = A + (m_start + mi) * K + k_base;
                            int32x4_t acc[TILE_N] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};

                            for (size_t k_offset = 0; k_offset < group_size; k_offset += 16) {
                                int8x16x4_t b_cols = vld4q_s8(b_block_base + k_offset * 4);
                                int8x16_t a_vec = vld1q_s8(a_ptr + k_offset);

                                acc[0] = accum_dot(acc[0], a_vec, b_cols.val[0]);
                                acc[1] = accum_dot(acc[1], a_vec, b_cols.val[1]);
                                acc[2] = accum_dot(acc[2], a_vec, b_cols.val[2]);
                                acc[3] = accum_dot(acc[3], a_vec, b_cols.val[3]);
                            }

                            for (size_t ni = 0; ni < actual_n; ni++) {
                                group_acc[mi][ni] += vaddvq_s32(acc[ni]);
                            }
                        }
                    }

                    const __fp16* scale_ptr = B_scales + (n_block * num_groups + g) * 4;
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        const float b_scale = (float)scale_ptr[ni];
                        for (size_t mi = 0; mi < actual_m; mi++) {
                            running_sum[mi][ni] += (float)group_acc[mi][ni] * b_scale;
                        }
                    }
                }

                for (size_t mi = 0; mi < actual_m; mi++) {
                    const float a_scale = A_scales[m_start + mi];
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        C[(m_start + mi) * N + (n_start + ni)] = (__fp16)(running_sum[mi][ni] * a_scale);
                    }
                }
            }
        });
}

