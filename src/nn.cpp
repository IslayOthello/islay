#include "nn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "common.hpp"

namespace islay {

  namespace {

    // Spatial shift for 8x8 plane with zero-padding.
    // Tap order: dy in {-1, 0, 1}, dx in {-1, 0, 1} -> t = (dy + 1) * 3 + (dx + 1).
    ISLAY_FORCEINLINE void shift_plane(const float *src, float *dst, int dy, int dx) noexcept {
#if defined(__ARM_NEON)
      const float32x4_t zero = vdupq_n_f32(0.0f);
      for (int y = 0; y < 8; ++y) {
        const int sy      = y + dy;
        float    *out_row = dst + y * 8;
        if (sy < 0 || sy >= 8) {
          vst1q_f32(out_row, zero);
          vst1q_f32(out_row + 4, zero);
        } else {
          const float      *in_row = src + sy * 8;
          const float32x4_t v0     = vld1q_f32(in_row);
          const float32x4_t v1     = vld1q_f32(in_row + 4);
          if (dx == 0) {
            vst1q_f32(out_row, v0);
            vst1q_f32(out_row + 4, v1);
          } else if (dx == -1) {
            // Shift right by 1 element (pixel 0 becomes 0)
            const float32x4_t d0 = vextq_f32(zero, v0, 3);
            const float32x4_t d1 = vextq_f32(v0, v1, 3);
            vst1q_f32(out_row, d0);
            vst1q_f32(out_row + 4, d1);
          } else { // dx == 1
            // Shift left by 1 element (pixel 7 becomes 0)
            const float32x4_t d0 = vextq_f32(v0, v1, 1);
            const float32x4_t d1 = vextq_f32(v1, zero, 1);
            vst1q_f32(out_row, d0);
            vst1q_f32(out_row + 4, d1);
          }
        }
      }
#else
      for (int y = 0; y < 8; ++y) {
        const int sy = y + dy;
        for (int x = 0; x < 8; ++x) {
          const int sx   = x + dx;
          dst[y * 8 + x] = (sy >= 0 && sy < 8 && sx >= 0 && sx < 8) ? src[sy * 8 + sx] : 0.0f;
        }
      }
#endif
    }

    // In-place vectorized ReLU
    ISLAY_FORCEINLINE void apply_relu(float *data, int count) noexcept {
      int i = 0;
#if defined(__ARM_NEON)
      const float32x4_t zero = vdupq_n_f32(0.0f);
      for (; i + 4 <= count; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        v             = vmaxq_f32(v, zero);
        vst1q_f32(data + i, v);
      }
#endif
      for (; i < count; ++i)
        data[i] = std::max(data[i], 0.0f);
    }

    // In-place vectorized elementwise add: dst += src
    ISLAY_FORCEINLINE void add_inplace(float *dst, const float *src, int count) noexcept {
      int i = 0;
#if defined(__ARM_NEON)
      for (; i + 4 <= count; i += 4) {
        float32x4_t vd = vld1q_f32(dst + i);
        float32x4_t vs = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(vd, vs));
      }
#endif
      for (; i < count; ++i)
        dst[i] += src[i];
    }

#if defined(__ARM_NEON)
    inline void preshift_all_9(const float *src, float *dst_9) noexcept {
      const float32x4_t zero = vdupq_n_f32(0.0f);
      float32x4_t       r_lo[8], r_hi[8];
      for (int r = 0; r < 8; ++r) {
        r_lo[r] = vld1q_f32(src + r * 8);
        r_hi[r] = vld1q_f32(src + r * 8 + 4);
      }

      float32x4_t h_lo[3][8], h_hi[3][8];
      for (int r = 0; r < 8; ++r) {
        h_lo[0][r] = vextq_f32(r_lo[r], r_hi[r], 1);
        h_hi[0][r] = vextq_f32(r_hi[r], zero, 1);
        h_lo[1][r] = r_lo[r];
        h_hi[1][r] = r_hi[r];
        h_lo[2][r] = vextq_f32(zero, r_lo[r], 3);
        h_hi[2][r] = vextq_f32(r_lo[r], r_hi[r], 3);
      }

      for (int t = 0; t < 9; ++t) {
        const int dy    = (t / 3) - 1;
        const int h_idx = t % 3;
        float    *dst   = dst_9 + t * 64;

        for (int r = 0; r < 8; ++r) {
          const int sr = r + dy;
          if (sr >= 0 && sr < 8) {
            vst1q_f32(dst + r * 8, h_lo[h_idx][sr]);
            vst1q_f32(dst + r * 8 + 4, h_hi[h_idx][sr]);
          } else {
            vst1q_f32(dst + r * 8, zero);
            vst1q_f32(dst + r * 8 + 4, zero);
          }
        }
      }
    }
#endif

    // Fast packed Conv3x3 on 8x8 with bias
    // w_packed layout: [cout / 4, cin, 9, 4]
    void conv3x3_bias_packed(const float *in, int cin, const float *w_packed, const float *b, float *out,
                             int cout) noexcept {
#if defined(__ARM_NEON)
      alignas(64) thread_local float shifted[64 * 9 * 64];
      for (int ci = 0; ci < cin; ++ci)
        preshift_all_9(in + ci * 64, shifted + ci * 9 * 64);

      for (int co_blk = 0; co_blk < cout / 4; ++co_blk) {
        const int         co    = co_blk * 4;
        const float      *w_blk = w_packed + co_blk * cin * 36;
        const float32x4_t vb0   = vdupq_n_f32(b ? b[co + 0] : 0.0f);
        const float32x4_t vb1   = vdupq_n_f32(b ? b[co + 1] : 0.0f);
        const float32x4_t vb2   = vdupq_n_f32(b ? b[co + 2] : 0.0f);
        const float32x4_t vb3   = vdupq_n_f32(b ? b[co + 3] : 0.0f);

        for (int v_blk = 0; v_blk < 16; v_blk += 4) {
          float32x4_t a0_0 = vb0, a0_1 = vb0, a0_2 = vb0, a0_3 = vb0;
          float32x4_t a1_0 = vb1, a1_1 = vb1, a1_2 = vb1, a1_3 = vb1;
          float32x4_t a2_0 = vb2, a2_1 = vb2, a2_2 = vb2, a2_3 = vb2;
          float32x4_t a3_0 = vb3, a3_1 = vb3, a3_2 = vb3, a3_3 = vb3;

          for (int ci = 0; ci < cin; ++ci) {
            const float *w_ci = w_blk + ci * 36;
            const float *s_ci = shifted + ci * 576 + v_blk * 4;

            for (int t = 0; t < 9; ++t) {
              const float32x4_t vw_quad = vld1q_f32(w_ci + t * 4);
              const float      *s_ptr   = s_ci + t * 64;

              const float32x4_t s0 = vld1q_f32(s_ptr + 0 * 4);
              const float32x4_t s1 = vld1q_f32(s_ptr + 1 * 4);
              const float32x4_t s2 = vld1q_f32(s_ptr + 2 * 4);
              const float32x4_t s3 = vld1q_f32(s_ptr + 3 * 4);

              a0_0 = vfmaq_laneq_f32(a0_0, s0, vw_quad, 0);
              a1_0 = vfmaq_laneq_f32(a1_0, s0, vw_quad, 1);
              a2_0 = vfmaq_laneq_f32(a2_0, s0, vw_quad, 2);
              a3_0 = vfmaq_laneq_f32(a3_0, s0, vw_quad, 3);

              a0_1 = vfmaq_laneq_f32(a0_1, s1, vw_quad, 0);
              a1_1 = vfmaq_laneq_f32(a1_1, s1, vw_quad, 1);
              a2_1 = vfmaq_laneq_f32(a2_1, s1, vw_quad, 2);
              a3_1 = vfmaq_laneq_f32(a3_1, s1, vw_quad, 3);

              a0_2 = vfmaq_laneq_f32(a0_2, s2, vw_quad, 0);
              a1_2 = vfmaq_laneq_f32(a1_2, s2, vw_quad, 1);
              a2_2 = vfmaq_laneq_f32(a2_2, s2, vw_quad, 2);
              a3_2 = vfmaq_laneq_f32(a3_2, s2, vw_quad, 3);

              a0_3 = vfmaq_laneq_f32(a0_3, s3, vw_quad, 0);
              a1_3 = vfmaq_laneq_f32(a1_3, s3, vw_quad, 1);
              a2_3 = vfmaq_laneq_f32(a2_3, s3, vw_quad, 2);
              a3_3 = vfmaq_laneq_f32(a3_3, s3, vw_quad, 3);
            }
          }

          vst1q_f32(out + (co + 0) * 64 + (v_blk + 0) * 4, a0_0);
          vst1q_f32(out + (co + 0) * 64 + (v_blk + 1) * 4, a0_1);
          vst1q_f32(out + (co + 0) * 64 + (v_blk + 2) * 4, a0_2);
          vst1q_f32(out + (co + 0) * 64 + (v_blk + 3) * 4, a0_3);

          vst1q_f32(out + (co + 1) * 64 + (v_blk + 0) * 4, a1_0);
          vst1q_f32(out + (co + 1) * 64 + (v_blk + 1) * 4, a1_1);
          vst1q_f32(out + (co + 1) * 64 + (v_blk + 2) * 4, a1_2);
          vst1q_f32(out + (co + 1) * 64 + (v_blk + 3) * 4, a1_3);

          vst1q_f32(out + (co + 2) * 64 + (v_blk + 0) * 4, a2_0);
          vst1q_f32(out + (co + 2) * 64 + (v_blk + 1) * 4, a2_1);
          vst1q_f32(out + (co + 2) * 64 + (v_blk + 2) * 4, a2_2);
          vst1q_f32(out + (co + 2) * 64 + (v_blk + 3) * 4, a2_3);

          vst1q_f32(out + (co + 3) * 64 + (v_blk + 0) * 4, a3_0);
          vst1q_f32(out + (co + 3) * 64 + (v_blk + 1) * 4, a3_1);
          vst1q_f32(out + (co + 3) * 64 + (v_blk + 2) * 4, a3_2);
          vst1q_f32(out + (co + 3) * 64 + (v_blk + 3) * 4, a3_3);
        }
      }
#else
      for (int co = 0; co < cout; ++co) {
        const int co_blk = co / 4;
        const int lane   = co % 4;
        for (int p = 0; p < 64; ++p)
          out[co * 64 + p] = b ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
          const float *w_ci = w_packed + (co_blk * cin + ci) * 36;
          for (int t = 0; t < 9; ++t) {
            const float weight = w_ci[t * 4 + lane];
            const int   dy     = (t / 3) - 1;
            const int   dx     = (t % 3) - 1;
            for (int r = 0; r < 8; ++r) {
              const int sr = r + dy;
              if (sr < 0 || sr >= 8)
                continue;
              for (int c = 0; c < 8; ++c) {
                const int sc = c + dx;
                if (sc < 0 || sc >= 8)
                  continue;
                out[co * 64 + r * 8 + c] += in[ci * 64 + sr * 8 + sc] * weight;
              }
            }
          }
        }
      }
#endif
    }

    // Vectorized Conv3x3 on 8x8 with bias (register-accumulated, zero memory store in inner loop)
    [[maybe_unused]] void conv3x3_bias(const float *in, int cin, const float *w, const float *b, float *out,
                                       int cout) noexcept {
      alignas(64) thread_local float shifted[64 * 9 * 64];

      // 1. Pre-shift all input channels
      for (int ci = 0; ci < cin; ++ci) {
#if defined(__ARM_NEON)
        preshift_all_9(in + ci * 64, shifted + ci * 9 * 64);
#else
        const float *in_ci = in + ci * 64;
        float       *s_ci  = shifted + ci * 9 * 64;
        for (int t = 0; t < 9; ++t) {
          const int dy = (t / 3) - 1;
          const int dx = (t % 3) - 1;
          shift_plane(in_ci, s_ci + t * 64, dy, dx);
        }
#endif
      }

      // 2. Accumulate with output channels in outer loop (keeps out in registers)
#if defined(__ARM_NEON)
      int co = 0;
      for (; co + 4 <= cout; co += 4) {
        const float *w0 = w + (co + 0) * cin * 9;
        const float *w1 = w + (co + 1) * cin * 9;
        const float *w2 = w + (co + 2) * cin * 9;
        const float *w3 = w + (co + 3) * cin * 9;

        const float32x4_t vb0 = vdupq_n_f32(b ? b[co + 0] : 0.0f);
        const float32x4_t vb1 = vdupq_n_f32(b ? b[co + 1] : 0.0f);
        const float32x4_t vb2 = vdupq_n_f32(b ? b[co + 2] : 0.0f);
        const float32x4_t vb3 = vdupq_n_f32(b ? b[co + 3] : 0.0f);

        float32x4_t a0[16], a1[16], a2[16], a3[16];
        for (int v = 0; v < 16; ++v) {
          a0[v] = vb0;
          a1[v] = vb1;
          a2[v] = vb2;
          a3[v] = vb3;
        }

        for (int ci = 0; ci < cin; ++ci) {
          const float *w_ci0 = w0 + ci * 9;
          const float *w_ci1 = w1 + ci * 9;
          const float *w_ci2 = w2 + ci * 9;
          const float *w_ci3 = w3 + ci * 9;
          const float *s_ci  = shifted + ci * 9 * 64;

          for (int t = 0; t < 9; ++t) {
            const float32x4_t vw0   = vdupq_n_f32(w_ci0[t]);
            const float32x4_t vw1   = vdupq_n_f32(w_ci1[t]);
            const float32x4_t vw2   = vdupq_n_f32(w_ci2[t]);
            const float32x4_t vw3   = vdupq_n_f32(w_ci3[t]);
            const float      *s_ptr = s_ci + t * 64;

            for (int v = 0; v < 16; ++v) {
              const float32x4_t s = vld1q_f32(s_ptr + v * 4);
              a0[v]               = vfmaq_f32(a0[v], s, vw0);
              a1[v]               = vfmaq_f32(a1[v], s, vw1);
              a2[v]               = vfmaq_f32(a2[v], s, vw2);
              a3[v]               = vfmaq_f32(a3[v], s, vw3);
            }
          }
        }

        for (int v = 0; v < 16; ++v) {
          vst1q_f32(out + (co + 0) * 64 + v * 4, a0[v]);
          vst1q_f32(out + (co + 1) * 64 + v * 4, a1[v]);
          vst1q_f32(out + (co + 2) * 64 + v * 4, a2[v]);
          vst1q_f32(out + (co + 3) * 64 + v * 4, a3[v]);
        }
      }

      for (; co < cout; ++co) {
        const float      *w_co = w + co * cin * 9;
        const float32x4_t vb   = vdupq_n_f32(b ? b[co] : 0.0f);
        float32x4_t       acc[16];
        for (int v = 0; v < 16; ++v)
          acc[v] = vb;

        for (int ci = 0; ci < cin; ++ci) {
          const float *w_ci = w_co + ci * 9;
          const float *s_ci = shifted + ci * 9 * 64;
          for (int t = 0; t < 9; ++t) {
            const float32x4_t vw    = vdupq_n_f32(w_ci[t]);
            const float      *s_ptr = s_ci + t * 64;
            for (int v = 0; v < 16; ++v)
              acc[v] = vfmaq_f32(acc[v], vld1q_f32(s_ptr + v * 4), vw);
          }
        }
        for (int v = 0; v < 16; ++v)
          vst1q_f32(out + co * 64 + v * 4, acc[v]);
      }
#else
      for (int co = 0; co < cout; ++co) {
        const float *w_co = w + co * cin * 9;
        for (int p = 0; p < 64; ++p)
          out[co * 64 + p] = b ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
          const float *w_ci = w_co + ci * 9;
          const float *s_ci = shifted + ci * 9 * 64;
          for (int t = 0; t < 9; ++t) {
            const float  weight = w_ci[t];
            const float *s_ptr  = s_ci + t * 64;
            for (int p = 0; p < 64; ++p)
              out[co * 64 + p] += s_ptr[p] * weight;
          }
        }
      }
#endif
    }

    // Vectorized Conv1x1 on 8x8 with bias
    void conv1x1_bias(const float *in, int cin, const float *w, const float *b, float *out, int cout) noexcept {
      for (int co = 0; co < cout; ++co) {
        float *out_co = out + co * 64;
#if defined(__ARM_NEON)
        const float32x4_t vb = vdupq_n_f32(b ? b[co] : 0.0f);
        float32x4_t       acc[16];
        for (int v = 0; v < 16; ++v)
          acc[v] = vb;
        for (int ci = 0; ci < cin; ++ci) {
          const float32x4_t vw    = vdupq_n_f32(w[co * cin + ci]);
          const float      *in_ci = in + ci * 64;
          for (int v = 0; v < 16; ++v) {
            float32x4_t vi = vld1q_f32(in_ci + v * 4);
            acc[v]         = vfmaq_f32(acc[v], vi, vw);
          }
        }
        for (int v = 0; v < 16; ++v)
          vst1q_f32(out_co + v * 4, acc[v]);
#else
        for (int p = 0; p < 64; ++p)
          out_co[p] = b ? b[co] : 0.0f;
        for (int ci = 0; ci < cin; ++ci) {
          const float  weight = w[co * cin + ci];
          const float *in_ci  = in + ci * 64;
          for (int p = 0; p < 64; ++p)
            out_co[p] += in_ci[p] * weight;
        }
#endif
      }
    }

    // Vectorized Linear (Dense) layer
    void linear(const float *in, int nin, const float *w, const float *b, float *out, int nout) noexcept {
      for (int i = 0; i < nout; ++i) {
        float        sum   = b ? b[i] : 0.0f;
        const float *w_row = w + i * nin;
        int          j     = 0;
#if defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; j + 4 <= nin; j += 4) {
          float32x4_t vi = vld1q_f32(in + j);
          float32x4_t vw = vld1q_f32(w_row + j);
          acc            = vfmaq_f32(acc, vi, vw);
        }
        sum += vaddvq_f32(acc);
#endif
        for (; j < nin; ++j)
          sum += in[j] * w_row[j];
        out[i] = sum;
      }
    }

  } // namespace

  // ---------------------------------------------------------------------------
  //  ModelWeights & FusedWeights initialization
  // ---------------------------------------------------------------------------

  ModelWeights::ModelWeights() {
    conv_stem_w.resize(64 * 2 * 9);
    bn_stem_gamma.assign(64, 1.0f);
    bn_stem_beta.assign(64, 0.0f);
    bn_stem_mean.assign(64, 0.0f);
    bn_stem_var.assign(64, 1.0f);

    for (int b = 0; b < kNumResBlocks; ++b) {
      blocks[b].conv1_w.resize(64 * 64 * 9);
      blocks[b].bn1_gamma.assign(64, 1.0f);
      blocks[b].bn1_beta.assign(64, 0.0f);
      blocks[b].bn1_mean.assign(64, 0.0f);
      blocks[b].bn1_var.assign(64, 1.0f);

      blocks[b].conv2_w.resize(64 * 64 * 9);
      blocks[b].bn2_gamma.assign(64, 1.0f);
      blocks[b].bn2_beta.assign(64, 0.0f);
      blocks[b].bn2_mean.assign(64, 0.0f);
      blocks[b].bn2_var.assign(64, 1.0f);
    }

    conv_pol_w.resize(2 * 64 * 1);
    bn_pol_gamma.assign(2, 1.0f);
    bn_pol_beta.assign(2, 0.0f);
    bn_pol_mean.assign(2, 0.0f);
    bn_pol_var.assign(2, 1.0f);
    fc_pol_w.resize(65 * 128);
    fc_pol_b.assign(65, 0.0f);

    conv_val_w.resize(1 * 64 * 1);
    bn_val_gamma.assign(1, 1.0f);
    bn_val_beta.assign(1, 0.0f);
    bn_val_mean.assign(1, 0.0f);
    bn_val_var.assign(1, 1.0f);
    fc_val1_w.resize(64 * 64);
    fc_val1_b.assign(64, 0.0f);
    fc_val2_w.resize(2 * 64);
    fc_val2_b.assign(2, 0.0f);
  }

  void ModelWeights::init_random(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto      kaiming = [&](std::vector<float> &w, int fan_in) {
      const float                     std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));
      std::normal_distribution<float> dist(0.0f, std_dev);
      for (auto &x: w)
        x = dist(rng);
    };

    kaiming(conv_stem_w, 2 * 9);
    for (int b = 0; b < kNumResBlocks; ++b) {
      kaiming(blocks[b].conv1_w, 64 * 9);
      kaiming(blocks[b].conv2_w, 64 * 9);
    }
    kaiming(conv_pol_w, 64 * 1);
    kaiming(fc_pol_w, 128);
    kaiming(conv_val_w, 64 * 1);
    kaiming(fc_val1_w, 64);
    kaiming(fc_val2_w, 64);
  }

  std::size_t ModelWeights::total_trainable_params() const noexcept {
    std::size_t count = conv_stem_w.size() + bn_stem_gamma.size() + bn_stem_beta.size();
    for (int b = 0; b < kNumResBlocks; ++b) {
      count += blocks[b].conv1_w.size() + blocks[b].bn1_gamma.size() + blocks[b].bn1_beta.size();
      count += blocks[b].conv2_w.size() + blocks[b].bn2_gamma.size() + blocks[b].bn2_beta.size();
    }
    count += conv_pol_w.size() + bn_pol_gamma.size() + bn_pol_beta.size() + fc_pol_w.size() + fc_pol_b.size();
    count += conv_val_w.size() + bn_val_gamma.size() + bn_val_beta.size() + fc_val1_w.size() + fc_val1_b.size() +
             fc_val2_w.size() + fc_val2_b.size();
    return count;
  }

  void ModelWeights::apply_ema(const ModelWeights &current, float decay) noexcept {
    const auto update = [decay](std::vector<float> &ema, const std::vector<float> &cur) {
      for (std::size_t i = 0; i < ema.size(); ++i)
        ema[i] = decay * ema[i] + (1.0f - decay) * cur[i];
    };
    update(conv_stem_w, current.conv_stem_w);
    update(bn_stem_gamma, current.bn_stem_gamma);
    update(bn_stem_beta, current.bn_stem_beta);
    update(bn_stem_mean, current.bn_stem_mean);
    update(bn_stem_var, current.bn_stem_var);
    for (int b = 0; b < kNumResBlocks; ++b) {
      update(blocks[b].conv1_w, current.blocks[b].conv1_w);
      update(blocks[b].bn1_gamma, current.blocks[b].bn1_gamma);
      update(blocks[b].bn1_beta, current.blocks[b].bn1_beta);
      update(blocks[b].bn1_mean, current.blocks[b].bn1_mean);
      update(blocks[b].bn1_var, current.blocks[b].bn1_var);
      update(blocks[b].conv2_w, current.blocks[b].conv2_w);
      update(blocks[b].bn2_gamma, current.blocks[b].bn2_gamma);
      update(blocks[b].bn2_beta, current.blocks[b].bn2_beta);
      update(blocks[b].bn2_mean, current.blocks[b].bn2_mean);
      update(blocks[b].bn2_var, current.blocks[b].bn2_var);
    }
    update(conv_pol_w, current.conv_pol_w);
    update(bn_pol_gamma, current.bn_pol_gamma);
    update(bn_pol_beta, current.bn_pol_beta);
    update(bn_pol_mean, current.bn_pol_mean);
    update(bn_pol_var, current.bn_pol_var);
    update(fc_pol_w, current.fc_pol_w);
    update(fc_pol_b, current.fc_pol_b);
    update(conv_val_w, current.conv_val_w);
    update(bn_val_gamma, current.bn_val_gamma);
    update(bn_val_beta, current.bn_val_beta);
    update(bn_val_mean, current.bn_val_mean);
    update(bn_val_var, current.bn_val_var);
    update(fc_val1_w, current.fc_val1_w);
    update(fc_val1_b, current.fc_val1_b);
    update(fc_val2_w, current.fc_val2_w);
    update(fc_val2_b, current.fc_val2_b);
  }

  FusedWeights::FusedWeights() {
    stem_w.resize(64 * 2 * 9);
    stem_w_packed.resize(64 * 2 * 9);
    stem_b.resize(64);
    for (int b = 0; b < kNumResBlocks; ++b) {
      blocks[b].conv1_w.resize(64 * 64 * 9);
      blocks[b].conv1_w_packed.resize(64 * 64 * 9);
      blocks[b].conv1_b.resize(64);
      blocks[b].conv2_w.resize(64 * 64 * 9);
      blocks[b].conv2_w_packed.resize(64 * 64 * 9);
      blocks[b].conv2_b.resize(64);
    }
    pol_conv_w.resize(2 * 64 * 1);
    pol_conv_b.resize(2);
    pol_fc_w.resize(65 * 128);
    pol_fc_b.resize(65);

    val_conv_w.resize(1 * 64 * 1);
    val_conv_b.resize(1);
    val_fc1_w.resize(64 * 64);
    val_fc1_b.resize(64);
    val_fc2_w.resize(2 * 64);
    val_fc2_b.resize(2);
  }

  void FusedWeights::fuse_from(const ModelWeights &src) {
    const auto fuse_conv_bn = [](const std::vector<float> &cw, int cout, int cin, int ksize,
                                 const std::vector<float> &gamma, const std::vector<float> &beta,
                                 const std::vector<float> &mean, const std::vector<float> &var, std::vector<float> &fw,
                                 std::vector<float> &fb) {
      constexpr float eps = 1e-5f;
      for (int c = 0; c < cout; ++c) {
        const float inv_std = 1.0f / std::sqrt(var[c] + eps);
        const float scale   = gamma[c] * inv_std;
        fb[c]               = beta[c] - mean[c] * scale;
        for (int i = 0; i < cin * ksize; ++i)
          fw[c * cin * ksize + i] = cw[c * cin * ksize + i] * scale;
      }
    };

    const auto pack_weights_4x = [](const std::vector<float> &w, int cout, int cin, std::vector<float> &w_packed) {
      w_packed.resize(static_cast<std::size_t>(cout * cin * 9));
      for (int co_blk = 0; co_blk < cout / 4; ++co_blk) {
        for (int ci = 0; ci < cin; ++ci) {
          for (int t = 0; t < 9; ++t) {
            for (int lane = 0; lane < 4; ++lane) {
              const int co                                      = co_blk * 4 + lane;
              w_packed[(co_blk * cin + ci) * 36 + t * 4 + lane] = w[(co * cin + ci) * 9 + t];
            }
          }
        }
      }
    };

    fuse_conv_bn(src.conv_stem_w, 64, 2, 9, src.bn_stem_gamma, src.bn_stem_beta, src.bn_stem_mean, src.bn_stem_var,
                 stem_w, stem_b);
    pack_weights_4x(stem_w, 64, 2, stem_w_packed);

    for (int b = 0; b < kNumResBlocks; ++b) {
      fuse_conv_bn(src.blocks[b].conv1_w, 64, 64, 9, src.blocks[b].bn1_gamma, src.blocks[b].bn1_beta,
                   src.blocks[b].bn1_mean, src.blocks[b].bn1_var, blocks[b].conv1_w, blocks[b].conv1_b);
      pack_weights_4x(blocks[b].conv1_w, 64, 64, blocks[b].conv1_w_packed);

      fuse_conv_bn(src.blocks[b].conv2_w, 64, 64, 9, src.blocks[b].bn2_gamma, src.blocks[b].bn2_beta,
                   src.blocks[b].bn2_mean, src.blocks[b].bn2_var, blocks[b].conv2_w, blocks[b].conv2_b);
      pack_weights_4x(blocks[b].conv2_w, 64, 64, blocks[b].conv2_w_packed);
    }

    fuse_conv_bn(src.conv_pol_w, 2, 64, 1, src.bn_pol_gamma, src.bn_pol_beta, src.bn_pol_mean, src.bn_pol_var,
                 pol_conv_w, pol_conv_b);
    pol_fc_w = src.fc_pol_w;
    pol_fc_b = src.fc_pol_b;

    fuse_conv_bn(src.conv_val_w, 1, 64, 1, src.bn_val_gamma, src.bn_val_beta, src.bn_val_mean, src.bn_val_var,
                 val_conv_w, val_conv_b);
    val_fc1_w = src.fc_val1_w;
    val_fc1_b = src.fc_val1_b;
    val_fc2_w = src.fc_val2_w;
    val_fc2_b = src.fc_val2_b;
  }

  // ---------------------------------------------------------------------------
  //  NativeNeuralNet Implementation
  // ---------------------------------------------------------------------------

  struct NativeNeuralNet::Impl {
    // Scratch buffer for inference (reused, zero-allocation)
    mutable std::vector<float> input_buf; // 2 * 64
    mutable std::vector<float> trunk_buf1; // 64 * 64
    mutable std::vector<float> trunk_buf2; // 64 * 64
    mutable std::vector<float> pol_buf; // 2 * 64 = 128
    mutable std::vector<float> val_buf; // 1 * 64 = 64
    mutable std::vector<float> val_fc1_buf; // 64

    // Optimizer momentum buffers
    ModelWeights momentum;

    Impl() {
      input_buf.resize(2 * 64);
      trunk_buf1.resize(64 * 64);
      trunk_buf2.resize(64 * 64);
      pol_buf.resize(2 * 64);
      val_buf.resize(1 * 64);
      val_fc1_buf.resize(64);
    }
  };

  NativeNeuralNet::NativeNeuralNet() : impl_(std::make_unique<Impl>()) {
    weights_.init_random(20260909);
    ema_weights_     = weights_;
    ema_initialized_ = true;
    fused_.fuse_from(weights_);
  }

  NativeNeuralNet::NativeNeuralNet(const std::string &checkpoint_path) : impl_(std::make_unique<Impl>()) {
    load(checkpoint_path);
  }

  NativeNeuralNet::~NativeNeuralNet() = default;

  void NativeNeuralNet::update_fused_weights() { fused_.fuse_from(weights_); }

  void NativeNeuralNet::update_ema(float decay) noexcept {
    if (!ema_initialized_) {
      ema_weights_     = weights_;
      ema_initialized_ = true;
    } else {
      ema_weights_.apply_ema(weights_, decay);
    }
  }

  void NativeNeuralNet::fuse_ema() noexcept {
    if (ema_initialized_)
      fused_.fuse_from(ema_weights_);
  }

  void NativeNeuralNet::evaluate_board(const Board &board, Evaluation &output, bool mask_legal) const {
    alignas(64) thread_local float in[2 * 64];
    alignas(64) thread_local float x[64 * 64];
    alignas(64) thread_local float y[64 * 64];
    alignas(64) thread_local float z[64 * 64];
    alignas(64) thread_local float p_conv[128];
    alignas(64) thread_local float v_conv[64];
    alignas(64) thread_local float v_fc1[64];

    // 1. Encode board
    for (int sq = 0; sq < 64; ++sq) {
      in[sq]      = static_cast<float>((board.player >> sq) & 1);
      in[64 + sq] = static_cast<float>((board.opponent >> sq) & 1);
    }

    // 2. Stem: Conv3x3 + Fused Bias -> ReLU
    conv3x3_bias_packed(in, 2, fused_.stem_w_packed.data(), fused_.stem_b.data(), x, 64);
    apply_relu(x, 64 * 64);

    // 3. 8 ResBlocks
    for (int b = 0; b < kNumResBlocks; ++b) {
      // conv1 -> relu
      conv3x3_bias_packed(x, 64, fused_.blocks[b].conv1_w_packed.data(), fused_.blocks[b].conv1_b.data(), y, 64);
      apply_relu(y, 64 * 64);

      // conv2
      conv3x3_bias_packed(y, 64, fused_.blocks[b].conv2_w_packed.data(), fused_.blocks[b].conv2_b.data(), z, 64);

      // skip add -> relu
      add_inplace(x, z, 64 * 64);
      apply_relu(x, 64 * 64);
    }

    // 4. Policy Head: Conv1x1 (64->2) -> ReLU -> FC (128->65)
    conv1x1_bias(x, 64, fused_.pol_conv_w.data(), fused_.pol_conv_b.data(), p_conv, 2);
    apply_relu(p_conv, 128);
    linear(p_conv, 128, fused_.pol_fc_w.data(), fused_.pol_fc_b.data(), output.logits.data(), 65);

    // 5. Value Head: Conv1x1 (64->1) -> ReLU -> FC1 (64->64) -> ReLU -> FC2 (64->1) -> Tanh
    conv1x1_bias(x, 64, fused_.val_conv_w.data(), fused_.val_conv_b.data(), v_conv, 1);
    apply_relu(v_conv, 64);

    linear(v_conv, 64, fused_.val_fc1_w.data(), fused_.val_fc1_b.data(), v_fc1, 64);
    apply_relu(v_fc1, 64);

    float v_raw[2]{};
    linear(v_fc1, 64, fused_.val_fc2_w.data(), fused_.val_fc2_b.data(), v_raw, 2);
    output.value  = std::tanh(v_raw[0]);
    output.margin = std::tanh(v_raw[1]);

    // 6. Mask illegal moves
    if (mask_legal) {
      const auto status = game_status(board);
      for (int sq = 0; sq < 64; ++sq) {
        if (!(status.moves & (Bitboard{1} << sq)))
          output.logits[sq] = -1e9f;
      }
      if (!status.forced_pass)
        output.logits[PASS] = -1e9f;
    }
  }

  void NativeNeuralNet::evaluate_batch(std::span<const Board> boards, std::span<Evaluation> outputs,
                                       bool mask_legal) const {
    if (boards.size() != outputs.size())
      throw std::invalid_argument("batch size mismatch in evaluate_batch");
    for (std::size_t i = 0; i < boards.size(); ++i)
      evaluate_board(boards[i], outputs[i], mask_legal);
  }

  // ---------------------------------------------------------------------------
  //  Training & Optimizer Implementation
  // ---------------------------------------------------------------------------

  NativeNeuralNet::TrainMetrics NativeNeuralNet::train_batch(std::span<const TrainingSample> batch,
                                                             const OptimizerConfig          &opt_cfg) {
    if (batch.empty())
      return {};

    const std::size_t B            = batch.size();
    const auto        zero_weights = [](ModelWeights &weights) {
      const auto zero_vec = [](std::vector<float> &v) { std::fill(v.begin(), v.end(), 0.0f); };
      zero_vec(weights.conv_stem_w);
      zero_vec(weights.bn_stem_gamma);
      zero_vec(weights.bn_stem_beta);
      for (int b = 0; b < kNumResBlocks; ++b) {
        zero_vec(weights.blocks[b].conv1_w);
        zero_vec(weights.blocks[b].bn1_gamma);
        zero_vec(weights.blocks[b].bn1_beta);
        zero_vec(weights.blocks[b].conv2_w);
        zero_vec(weights.blocks[b].bn2_gamma);
        zero_vec(weights.blocks[b].bn2_beta);
      }
      zero_vec(weights.conv_pol_w);
      zero_vec(weights.bn_pol_gamma);
      zero_vec(weights.bn_pol_beta);
      zero_vec(weights.fc_pol_w);
      zero_vec(weights.fc_pol_b);
      zero_vec(weights.conv_val_w);
      zero_vec(weights.bn_val_gamma);
      zero_vec(weights.bn_val_beta);
      zero_vec(weights.fc_val1_w);
      zero_vec(weights.fc_val1_b);
      zero_vec(weights.fc_val2_w);
      zero_vec(weights.fc_val2_b);
    };

    // Keep sample order within each chunk and reduce workers in a fixed order.
    // This avoids synchronization in the hot path while keeping updates reproducible.
    const unsigned worker_count = static_cast<unsigned>(
            std::min<std::size_t>(B, std::max(1u, std::min(8u, std::thread::hardware_concurrency()))));
    std::vector<ModelWeights> worker_grads(worker_count);
    for (auto &worker_grad: worker_grads)
      zero_weights(worker_grad);
    std::vector<float> worker_pol_loss(worker_count, 0.0f);
    std::vector<float> worker_val_loss(worker_count, 0.0f);

    const auto process_samples = [&](std::size_t begin, std::size_t end, ModelWeights &local_grads,
                                     float &local_pol_loss, float &local_val_loss) {
      // Forward and Backward for each sample in this worker's disjoint chunk
      for (std::size_t s = begin; s < end; ++s) {
        const auto       &sample = batch[s];
        alignas(64) float in[2 * 64];
        for (int sq = 0; sq < 64; ++sq) {
          in[sq]      = static_cast<float>((sample.board.player >> sq) & 1);
          in[64 + sq] = static_cast<float>((sample.board.opponent >> sq) & 1);
        }

        // Record forward activations
        alignas(64) float stem_out[64 * 64];
        conv3x3_bias_packed(in, 2, fused_.stem_w_packed.data(), fused_.stem_b.data(), stem_out, 64);
        apply_relu(stem_out, 64 * 64);

        struct ResActivation {
          alignas(64) float in_x[64 * 64];
          alignas(64) float conv1_out[64 * 64];
          alignas(64) float conv2_out[64 * 64];
        };
        std::vector<ResActivation> res_acts(kNumResBlocks);

        float *cur_x = stem_out;
        for (int b = 0; b < kNumResBlocks; ++b) {
          std::memcpy(res_acts[b].in_x, cur_x, 64 * 64 * sizeof(float));
          conv3x3_bias_packed(cur_x, 64, fused_.blocks[b].conv1_w_packed.data(), fused_.blocks[b].conv1_b.data(),
                              res_acts[b].conv1_out, 64);
          apply_relu(res_acts[b].conv1_out, 64 * 64);

          conv3x3_bias_packed(res_acts[b].conv1_out, 64, fused_.blocks[b].conv2_w_packed.data(),
                              fused_.blocks[b].conv2_b.data(), res_acts[b].conv2_out, 64);

          add_inplace(cur_x, res_acts[b].conv2_out, 64 * 64);
          apply_relu(cur_x, 64 * 64);
        }

        // Policy head forward
        alignas(64) float pol_conv[128];
        conv1x1_bias(cur_x, 64, fused_.pol_conv_w.data(), fused_.pol_conv_b.data(), pol_conv, 2);
        apply_relu(pol_conv, 128);

        alignas(64) float logits[65];
        linear(pol_conv, 128, fused_.pol_fc_w.data(), fused_.pol_fc_b.data(), logits, 65);

        // Value head forward
        alignas(64) float val_conv[64];
        conv1x1_bias(cur_x, 64, fused_.val_conv_w.data(), fused_.val_conv_b.data(), val_conv, 1);
        apply_relu(val_conv, 64);

        alignas(64) float val_fc1[64];
        linear(val_conv, 64, fused_.val_fc1_w.data(), fused_.val_fc1_b.data(), val_fc1, 64);
        apply_relu(val_fc1, 64);

        float val_raw[2]{};
        linear(val_fc1, 64, fused_.val_fc2_w.data(), fused_.val_fc2_b.data(), val_raw, 2);
        const float pred_wdl    = std::tanh(val_raw[0]);
        const float pred_margin = std::tanh(val_raw[1]);

        // --- Loss and Output Gradients with Legal Move Masking ---
        // Policy Legal Mask & Softmax
        float max_l = -1e30f;
        for (int a = 0; a < 65; ++a) {
          const bool legal = (a < 64 && (sample.legal_moves & (Bitboard{1} << a))) || (a == 64 && sample.forced_pass);
          if (legal)
            max_l = std::max(max_l, logits[a]);
        }

        double                sum_exp = 0.0;
        std::array<float, 65> probs{};
        for (int a = 0; a < 65; ++a) {
          const bool legal = (a < 64 && (sample.legal_moves & (Bitboard{1} << a))) || (a == 64 && sample.forced_pass);
          if (legal) {
            probs[a] = static_cast<float>(std::exp(logits[a] - max_l));
            sum_exp += probs[a];
          }
        }

        std::array<float, 65> d_logits{};
        float                 cur_pol_loss = 0.0f;
        for (int a = 0; a < 65; ++a) {
          const bool legal = (a < 64 && (sample.legal_moves & (Bitboard{1} << a))) || (a == 64 && sample.forced_pass);
          if (legal && sum_exp > 0.0) {
            probs[a] /= static_cast<float>(sum_exp);
            const float target = sample.target_policy[a];
            if (target > 0.0f)
              cur_pol_loss -= target * std::log(std::max(probs[a], 1e-12f));
            d_logits[a] = probs[a] - target;
          } else {
            d_logits[a] = 0.0f;
          }
        }
        local_pol_loss += cur_pol_loss;

        // WDL drives move selection; margin is a lower-weight calibration target.
        constexpr float kWdlLossWeight    = 1.0f;
        constexpr float kMarginLossWeight = 0.25f;
        const float     wdl_diff          = pred_wdl - sample.target_wdl;
        const float     margin_diff       = pred_margin - sample.target_margin;
        local_val_loss += kWdlLossWeight * wdl_diff * wdl_diff + kMarginLossWeight * margin_diff * margin_diff;
        const float d_val_raw[2] = {2.0f * kWdlLossWeight * wdl_diff * (1.0f - pred_wdl * pred_wdl),
                                    2.0f * kMarginLossWeight * margin_diff * (1.0f - pred_margin * pred_margin)};

        // --- Backward through Value Head ---
        // FC2: val_raw = W * val_fc1 + b
        for (int head = 0; head < 2; ++head) {
          for (int i = 0; i < 64; ++i)
            local_grads.fc_val2_w[head * 64 + i] += d_val_raw[head] * val_fc1[i];
          local_grads.fc_val2_b[head] += d_val_raw[head];
        }

        alignas(64) float d_val_fc1[64];
        for (int i = 0; i < 64; ++i) {
          float sum    = d_val_raw[0] * fused_.val_fc2_w[i] + d_val_raw[1] * fused_.val_fc2_w[64 + i];
          d_val_fc1[i] = (val_fc1[i] > 0.0f ? sum : 0.0f);
        }

        // FC1: val_fc1 = W * val_conv + b
        for (int o = 0; o < 64; ++o) {
          local_grads.fc_val1_b[o] += d_val_fc1[o];
          for (int i = 0; i < 64; ++i)
            local_grads.fc_val1_w[o * 64 + i] += d_val_fc1[o] * val_conv[i];
        }

        alignas(64) float d_val_conv[64];
        for (int i = 0; i < 64; ++i) {
          float sum = 0.0f;
          for (int o = 0; o < 64; ++o)
            sum += d_val_fc1[o] * fused_.val_fc1_w[o * 64 + i];
          d_val_conv[i] = (val_conv[i] > 0.0f ? sum : 0.0f);
        }

        // Conv 1x1: val_conv = W * cur_x + b
        for (int ci = 0; ci < 64; ++ci) {
          float sum = 0.0f;
          for (int p = 0; p < 64; ++p)
            sum += d_val_conv[p] * cur_x[ci * 64 + p];
          local_grads.conv_val_w[ci] += sum;
        }
        for (int p = 0; p < 64; ++p)
          local_grads.bn_val_beta[0] += d_val_conv[p];

        alignas(64) float d_trunk[64 * 64];
        for (int ci = 0; ci < 64; ++ci) {
          for (int p = 0; p < 64; ++p)
            d_trunk[ci * 64 + p] = d_val_conv[p] * fused_.val_conv_w[ci];
        }

        // --- Backward through Policy Head ---
        // FC: logits = W * pol_conv + b
        for (int o = 0; o < 65; ++o) {
          local_grads.fc_pol_b[o] += d_logits[o];
          for (int i = 0; i < 128; ++i)
            local_grads.fc_pol_w[o * 128 + i] += d_logits[o] * pol_conv[i];
        }

        alignas(64) float d_pol_conv[128];
        for (int i = 0; i < 128; ++i) {
          float sum = 0.0f;
          for (int o = 0; o < 65; ++o)
            sum += d_logits[o] * fused_.pol_fc_w[o * 128 + i];
          d_pol_conv[i] = (pol_conv[i] > 0.0f ? sum : 0.0f);
        }

        // Conv 1x1: pol_conv (2x64) = W (2x64) * cur_x (64x64) + b
        for (int co = 0; co < 2; ++co) {
          for (int ci = 0; ci < 64; ++ci) {
            float sum = 0.0f;
            for (int p = 0; p < 64; ++p)
              sum += d_pol_conv[co * 64 + p] * cur_x[ci * 64 + p];
            local_grads.conv_pol_w[co * 64 + ci] += sum;
          }
          for (int p = 0; p < 64; ++p)
            local_grads.bn_pol_beta[co] += d_pol_conv[co * 64 + p];
        }

        // Add to d_trunk
        for (int ci = 0; ci < 64; ++ci) {
          for (int p = 0; p < 64; ++p) {
            d_trunk[ci * 64 + p] += d_pol_conv[0 * 64 + p] * fused_.pol_conv_w[0 * 64 + ci] +
                                    d_pol_conv[1 * 64 + p] * fused_.pol_conv_w[1 * 64 + ci];
          }
        }

        // --- Backward through 8 ResBlocks ---
        for (int b = kNumResBlocks - 1; b >= 0; --b) {
          // d_trunk is gradient at output of block b
          for (int i = 0; i < 64 * 64; ++i)
            if (cur_x[i] <= 0.0f)
              d_trunk[i] = 0.0f;

          // Skip connection branch: d_skip = d_trunk
          alignas(64) float d_conv2_out[64 * 64];
          std::memcpy(d_conv2_out, d_trunk, 64 * 64 * sizeof(float));

          // Conv2 backward
          alignas(64) float d_conv1_out[64 * 64];
          std::memset(d_conv1_out, 0, 64 * 64 * sizeof(float));

          alignas(64) float shifted_in[9 * 64];
          for (int ci = 0; ci < 64; ++ci) {
            const float *in_c = res_acts[b].conv1_out + ci * 64;
            for (int t = 0; t < 9; ++t)
              shift_plane(in_c, shifted_in + t * 64, (t / 3) - 1, (t % 3) - 1);

            for (int co = 0; co < 64; ++co) {
              const float *dy_c = d_conv2_out + co * 64;
              for (int t = 0; t < 9; ++t) {
                const float *s_c = shifted_in + t * 64;
                float        dot = 0.0f;
                for (int p = 0; p < 64; ++p)
                  dot += dy_c[p] * s_c[p];
                local_grads.blocks[b].conv2_w[(co * 64 + ci) * 9 + t] += dot;
              }
            }
          }

          // Conv2 input gradient (d_conv1_out) using inverted shifts
          for (int co = 0; co < 64; ++co) {
            const float *dy_c = d_conv2_out + co * 64;
            for (int t = 0; t < 9; ++t) {
              alignas(64) float shifted_dy[64];
              // Inverted tap: -(dy, dx)
              shift_plane(dy_c, shifted_dy, -((t / 3) - 1), -((t % 3) - 1));
              for (int ci = 0; ci < 64; ++ci) {
                const float w    = fused_.blocks[b].conv2_w[(co * 64 + ci) * 9 + t];
                float      *dx_c = d_conv1_out + ci * 64;
                for (int p = 0; p < 64; ++p)
                  dx_c[p] += shifted_dy[p] * w;
              }
            }
          }

          // ReLU backward on conv1 output
          for (int i = 0; i < 64 * 64; ++i)
            if (res_acts[b].conv1_out[i] <= 0.0f)
              d_conv1_out[i] = 0.0f;

          // Conv1 backward
          alignas(64) float d_in_x[64 * 64];
          std::memset(d_in_x, 0, 64 * 64 * sizeof(float));

          for (int ci = 0; ci < 64; ++ci) {
            const float *in_c = res_acts[b].in_x + ci * 64;
            for (int t = 0; t < 9; ++t)
              shift_plane(in_c, shifted_in + t * 64, (t / 3) - 1, (t % 3) - 1);

            for (int co = 0; co < 64; ++co) {
              const float *dy_c = d_conv1_out + co * 64;
              for (int t = 0; t < 9; ++t) {
                const float *s_c = shifted_in + t * 64;
                float        dot = 0.0f;
                for (int p = 0; p < 64; ++p)
                  dot += dy_c[p] * s_c[p];
                local_grads.blocks[b].conv1_w[(co * 64 + ci) * 9 + t] += dot;
              }
            }
          }

          for (int co = 0; co < 64; ++co) {
            const float *dy_c = d_conv1_out + co * 64;
            for (int t = 0; t < 9; ++t) {
              alignas(64) float shifted_dy[64];
              shift_plane(dy_c, shifted_dy, -((t / 3) - 1), -((t % 3) - 1));
              for (int ci = 0; ci < 64; ++ci) {
                const float w    = fused_.blocks[b].conv1_w[(co * 64 + ci) * 9 + t];
                float      *dx_c = d_in_x + ci * 64;
                for (int p = 0; p < 64; ++p)
                  dx_c[p] += shifted_dy[p] * w;
              }
            }
          }

          // Add residual skip gradient: d_trunk_prev = d_in_x + d_trunk
          add_inplace(d_trunk, d_in_x, 64 * 64);
        }

        // --- Backward through Stem ---
        for (int i = 0; i < 64 * 64; ++i)
          if (stem_out[i] <= 0.0f)
            d_trunk[i] = 0.0f;

        alignas(64) float shifted_stem_in[9 * 64];
        for (int ci = 0; ci < 2; ++ci) {
          const float *in_c = in + ci * 64;
          for (int t = 0; t < 9; ++t)
            shift_plane(in_c, shifted_stem_in + t * 64, (t / 3) - 1, (t % 3) - 1);

          for (int co = 0; co < 64; ++co) {
            const float *dy_c = d_trunk + co * 64;
            for (int t = 0; t < 9; ++t) {
              const float *s_c = shifted_stem_in + t * 64;
              float        dot = 0.0f;
              for (int p = 0; p < 64; ++p)
                dot += dy_c[p] * s_c[p];
              local_grads.conv_stem_w[(co * 2 + ci) * 9 + t] += dot;
            }
          }
        }
      }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (unsigned worker = 0; worker < worker_count; ++worker) {
      const std::size_t begin = B * worker / worker_count;
      const std::size_t end   = B * (worker + 1) / worker_count;
      workers.emplace_back(process_samples, begin, end, std::ref(worker_grads[worker]),
                           std::ref(worker_pol_loss[worker]), std::ref(worker_val_loss[worker]));
    }
    for (auto &worker: workers)
      worker.join();

    ModelWeights grads;
    zero_weights(grads);
    float      total_pol_loss = 0.0f;
    float      total_val_loss = 0.0f;
    const auto add_vec        = [](std::vector<float> &dst, const std::vector<float> &src) {
      for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] += src[i];
    };
    const auto add_weights = [&](const ModelWeights &src) {
      add_vec(grads.conv_stem_w, src.conv_stem_w);
      for (int b = 0; b < kNumResBlocks; ++b) {
        add_vec(grads.blocks[b].conv1_w, src.blocks[b].conv1_w);
        add_vec(grads.blocks[b].conv2_w, src.blocks[b].conv2_w);
      }
      add_vec(grads.conv_pol_w, src.conv_pol_w);
      add_vec(grads.fc_pol_w, src.fc_pol_w);
      add_vec(grads.fc_pol_b, src.fc_pol_b);
      add_vec(grads.conv_val_w, src.conv_val_w);
      add_vec(grads.fc_val1_w, src.fc_val1_w);
      add_vec(grads.fc_val1_b, src.fc_val1_b);
      add_vec(grads.fc_val2_w, src.fc_val2_w);
      add_vec(grads.fc_val2_b, src.fc_val2_b);
    };
    for (unsigned worker = 0; worker < worker_count; ++worker) {
      add_weights(worker_grads[worker]);
      total_pol_loss += worker_pol_loss[worker];
      total_val_loss += worker_val_loss[worker];
    }

    // Average gradients over batch
    const float inv_B      = 1.0f / static_cast<float>(B);
    const auto  scale_grad = [inv_B](std::vector<float> &g) {
      for (auto &x: g)
        x *= inv_B;
    };

    scale_grad(grads.conv_stem_w);
    for (int b = 0; b < kNumResBlocks; ++b) {
      scale_grad(grads.blocks[b].conv1_w);
      scale_grad(grads.blocks[b].conv2_w);
    }
    scale_grad(grads.conv_pol_w);
    scale_grad(grads.fc_pol_w);
    scale_grad(grads.fc_pol_b);
    scale_grad(grads.conv_val_w);
    scale_grad(grads.fc_val1_w);
    scale_grad(grads.fc_val1_b);
    scale_grad(grads.fc_val2_w);
    scale_grad(grads.fc_val2_b);

    // Compute Global Gradient L2 Norm
    double     sum_sq   = 0.0;
    const auto acc_norm = [&sum_sq](const std::vector<float> &g) {
      for (float x: g)
        sum_sq += static_cast<double>(x) * static_cast<double>(x);
    };
    acc_norm(grads.conv_stem_w);
    for (int b = 0; b < kNumResBlocks; ++b) {
      acc_norm(grads.blocks[b].conv1_w);
      acc_norm(grads.blocks[b].conv2_w);
    }
    acc_norm(grads.conv_pol_w);
    acc_norm(grads.fc_pol_w);
    acc_norm(grads.fc_pol_b);
    acc_norm(grads.conv_val_w);
    acc_norm(grads.fc_val1_w);
    acc_norm(grads.fc_val1_b);
    acc_norm(grads.fc_val2_w);
    acc_norm(grads.fc_val2_b);

    const float grad_norm = static_cast<float>(std::sqrt(sum_sq));

    // Gradient clipping
    float clip_scale = 1.0f;
    if (grad_norm > opt_cfg.grad_clip && grad_norm > 0.0f)
      clip_scale = opt_cfg.grad_clip / grad_norm;

    // SGD Update with Momentum and Weight Decay
    const auto sgd_step = [&](std::vector<float> &w, const std::vector<float> &g, std::vector<float> &v) {
      if (v.size() != w.size())
        v.assign(w.size(), 0.0f);
      for (std::size_t i = 0; i < w.size(); ++i) {
        const float grad = g[i] * clip_scale + opt_cfg.weight_decay * w[i];
        v[i]             = opt_cfg.momentum * v[i] + grad;
        w[i] -= opt_cfg.learning_rate * v[i];
      }
    };

    sgd_step(weights_.conv_stem_w, grads.conv_stem_w, impl_->momentum.conv_stem_w);
    for (int b = 0; b < kNumResBlocks; ++b) {
      sgd_step(weights_.blocks[b].conv1_w, grads.blocks[b].conv1_w, impl_->momentum.blocks[b].conv1_w);
      sgd_step(weights_.blocks[b].conv2_w, grads.blocks[b].conv2_w, impl_->momentum.blocks[b].conv2_w);
    }
    sgd_step(weights_.conv_pol_w, grads.conv_pol_w, impl_->momentum.conv_pol_w);
    sgd_step(weights_.fc_pol_w, grads.fc_pol_w, impl_->momentum.fc_pol_w);
    sgd_step(weights_.fc_pol_b, grads.fc_pol_b, impl_->momentum.fc_pol_b);

    sgd_step(weights_.conv_val_w, grads.conv_val_w, impl_->momentum.conv_val_w);
    sgd_step(weights_.fc_val1_w, grads.fc_val1_w, impl_->momentum.fc_val1_w);
    sgd_step(weights_.fc_val1_b, grads.fc_val1_b, impl_->momentum.fc_val1_b);
    sgd_step(weights_.fc_val2_w, grads.fc_val2_w, impl_->momentum.fc_val2_w);
    sgd_step(weights_.fc_val2_b, grads.fc_val2_b, impl_->momentum.fc_val2_b);

    // Update fused weights for subsequent inference
    update_fused_weights();
    ++training_steps_;

    return {
            (total_pol_loss + total_val_loss) * inv_B,
            total_pol_loss * inv_B,
            total_val_loss * inv_B,
            grad_norm,
    };
  }

  // ---------------------------------------------------------------------------
  //  Serialization (.bin format)
  // ---------------------------------------------------------------------------

  void NativeNeuralNet::save(const std::string &path, std::uint64_t training_steps, bool use_ema) const {
    std::ofstream out(path, std::ios::binary);
    if (!out)
      throw std::runtime_error("cannot open file for saving weights: " + path);

    const char magic[8] = {'I', 'S', 'L', 'A', 'Y', 'N', 'E', 'T'};
    out.write(magic, 8);
    const std::uint32_t version = 2;
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    out.write(reinterpret_cast<const char *>(&training_steps), sizeof(training_steps));

    const ModelWeights &w = (use_ema && ema_initialized_) ? ema_weights_ : weights_;

    const auto write_vec = [&out](const std::vector<float> &v) {
      out.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    };

    write_vec(w.conv_stem_w);
    write_vec(w.bn_stem_gamma);
    write_vec(w.bn_stem_beta);
    write_vec(w.bn_stem_mean);
    write_vec(w.bn_stem_var);

    for (int b = 0; b < kNumResBlocks; ++b) {
      write_vec(w.blocks[b].conv1_w);
      write_vec(w.blocks[b].bn1_gamma);
      write_vec(w.blocks[b].bn1_beta);
      write_vec(w.blocks[b].bn1_mean);
      write_vec(w.blocks[b].bn1_var);

      write_vec(w.blocks[b].conv2_w);
      write_vec(w.blocks[b].bn2_gamma);
      write_vec(w.blocks[b].bn2_beta);
      write_vec(w.blocks[b].bn2_mean);
      write_vec(w.blocks[b].bn2_var);
    }

    write_vec(w.conv_pol_w);
    write_vec(w.bn_pol_gamma);
    write_vec(w.bn_pol_beta);
    write_vec(w.bn_pol_mean);
    write_vec(w.bn_pol_var);
    write_vec(w.fc_pol_w);
    write_vec(w.fc_pol_b);

    write_vec(w.conv_val_w);
    write_vec(w.bn_val_gamma);
    write_vec(w.bn_val_beta);
    write_vec(w.bn_val_mean);
    write_vec(w.bn_val_var);
    write_vec(w.fc_val1_w);
    write_vec(w.fc_val1_b);
    write_vec(w.fc_val2_w);
    write_vec(w.fc_val2_b);
  }

  void NativeNeuralNet::load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot open file for loading weights: " + path);

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "ISLAYNET", 8) != 0)
      throw std::runtime_error("invalid magic in weights file: " + path);

    std::uint32_t version = 0;
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 1 && version != 2)
      throw std::runtime_error("unsupported weights version (WDL+margin network required)");

    in.read(reinterpret_cast<char *>(&training_steps_), sizeof(training_steps_));

    const auto read_vec = [&in](std::vector<float> &v) {
      in.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    };

    read_vec(weights_.conv_stem_w);
    read_vec(weights_.bn_stem_gamma);
    read_vec(weights_.bn_stem_beta);
    read_vec(weights_.bn_stem_mean);
    read_vec(weights_.bn_stem_var);

    for (int b = 0; b < kNumResBlocks; ++b) {
      read_vec(weights_.blocks[b].conv1_w);
      read_vec(weights_.blocks[b].bn1_gamma);
      read_vec(weights_.blocks[b].bn1_beta);
      read_vec(weights_.blocks[b].bn1_mean);
      read_vec(weights_.blocks[b].bn1_var);

      read_vec(weights_.blocks[b].conv2_w);
      read_vec(weights_.blocks[b].bn2_gamma);
      read_vec(weights_.blocks[b].bn2_beta);
      read_vec(weights_.blocks[b].bn2_mean);
      read_vec(weights_.blocks[b].bn2_var);
    }

    read_vec(weights_.conv_pol_w);
    read_vec(weights_.bn_pol_gamma);
    read_vec(weights_.bn_pol_beta);
    read_vec(weights_.bn_pol_mean);
    read_vec(weights_.bn_pol_var);
    read_vec(weights_.fc_pol_w);
    read_vec(weights_.fc_pol_b);

    read_vec(weights_.conv_val_w);
    read_vec(weights_.bn_val_gamma);
    read_vec(weights_.bn_val_beta);
    read_vec(weights_.bn_val_mean);
    read_vec(weights_.bn_val_var);
    read_vec(weights_.fc_val1_w);
    read_vec(weights_.fc_val1_b);
    if (version == 1) {
      std::array<float, 64> wdl_weights{};
      float                wdl_bias = 0.0f;
      in.read(reinterpret_cast<char *>(wdl_weights.data()), sizeof(wdl_weights));
      in.read(reinterpret_cast<char *>(&wdl_bias), sizeof(wdl_bias));
      std::copy(wdl_weights.begin(), wdl_weights.end(), weights_.fc_val2_w.begin());
      std::fill(weights_.fc_val2_w.begin() + 64, weights_.fc_val2_w.end(), 0.0f);
      weights_.fc_val2_b[0] = wdl_bias;
      weights_.fc_val2_b[1] = 0.0f;
    } else {
      read_vec(weights_.fc_val2_w);
      read_vec(weights_.fc_val2_b);
    }

    ema_weights_     = weights_;
    ema_initialized_ = true;
    update_fused_weights();
  }

  // ---------------------------------------------------------------------------
  //  NativeNeuralEvaluator Adapter
  // ---------------------------------------------------------------------------

  NativeNeuralEvaluator::NativeNeuralEvaluator(std::shared_ptr<NativeNeuralNet> net, bool mask_legal) :
      net_(std::move(net)), mask_legal_(mask_legal) {}

  void NativeNeuralEvaluator::evaluate(std::span<const Board> boards, std::span<Evaluation> outputs,
                                       std::stop_token stop) {
    if (boards.size() != outputs.size())
      throw std::invalid_argument("mismatched boards/outputs in NativeNeuralEvaluator");
    if (stop.stop_requested())
      return;
    net_->evaluate_batch(boards, outputs, mask_legal_);
  }

  std::string NativeNeuralEvaluator::description() const {
    return "native-simd b8c64-v2 training_steps " + std::to_string(net_->training_steps());
  }

  // ---------------------------------------------------------------------------
  //  Selftest
  // ---------------------------------------------------------------------------

  bool nn_selftest() {
    try {
      NativeNeuralNet net;
      if (net.weights().total_trainable_params() != kNumParams)
        return false;

      // 1. Shift plane check
      alignas(64) float src[64];
      for (int i = 0; i < 64; ++i)
        src[i] = static_cast<float>(i + 1);

      alignas(64) float dst[64];
      shift_plane(src, dst, 0, 0);
      for (int i = 0; i < 64; ++i)
        if (dst[i] != src[i])
          return false;

      shift_plane(src, dst, 0, -1); // shift right by 1
      for (int y = 0; y < 8; ++y) {
        if (dst[y * 8] != 0.0f)
          return false;
        for (int x = 1; x < 8; ++x)
          if (dst[y * 8 + x] != src[y * 8 + x - 1])
            return false;
      }

      shift_plane(src, dst, 0, 1); // shift left by 1
      for (int y = 0; y < 8; ++y) {
        if (dst[y * 8 + 7] != 0.0f)
          return false;
        for (int x = 0; x < 7; ++x)
          if (dst[y * 8 + x] != src[y * 8 + x + 1])
            return false;
      }

      // 2. Forward evaluate check
      const Board start = Board::start();
      Evaluation  out;
      net.evaluate_board(start, out, true);

      if (!std::isfinite(out.value) || std::abs(out.value) > 1.0f || !std::isfinite(out.margin) ||
          std::abs(out.margin) > 1.0f)
        return false;

      const auto status = game_status(start);
      for (int sq = 0; sq < 64; ++sq) {
        if (status.moves & (Bitboard{1} << sq)) {
          if (out.logits[sq] <= -1e8f || !std::isfinite(out.logits[sq]))
            return false;
        } else {
          // Must be masked
          if (out.logits[sq] > -1e8f)
            return false;
        }
      }

      // 3. Training step check
      TrainingSample sample;
      sample.board       = start;
      sample.legal_moves = status.moves;
      sample.forced_pass = false;
      for (int sq = 0; sq < 64; ++sq)
        if (status.moves & (Bitboard{1} << sq))
          sample.target_policy[sq] = 0.25f;
      sample.target_wdl    = 0.5f;
      sample.target_margin = 0.25f;

      OptimizerConfig cfg;
      cfg.learning_rate  = 0.01f;
      const auto metrics = net.train_batch(std::span(&sample, 1), cfg);
      if (!std::isfinite(metrics.total_loss) || metrics.total_loss <= 0.0f || !std::isfinite(metrics.grad_norm) ||
          metrics.grad_norm <= 0.0f)
        return false;

      Evaluation out_trained;
      net.evaluate_board(start, out_trained, true);

      // 4. Save and reload check
      const std::string tmp_path = "_test_weights.bin";
      net.save(tmp_path, 42);
      NativeNeuralNet reloaded(tmp_path);
      if (reloaded.training_steps() != 42)
        return false;
      std::remove(tmp_path.c_str());

      Evaluation out2;
      reloaded.evaluate_board(start, out2, true);
      if (std::abs(out2.value - out_trained.value) > 1e-4f || std::abs(out2.margin - out_trained.margin) > 1e-4f)
        return false;

      return true;
    } catch (...) {
      return false;
    }
  }

} // namespace islay
