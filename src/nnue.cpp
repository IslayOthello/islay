#include "nnue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace islay {
  namespace {
    constexpr char kMagic[8]  = {'I', 'S', 'L', 'A', 'Y', 'N', 'N', '2'}; // float emb (legacy)
    constexpr char kMagic3[8] = {'I', 'S', 'L', 'A', 'Y', 'N', 'N', '3'}; // int16 emb (shipped)
    constexpr char kMagic4[8] = {'I', 'S', 'L', 'A', 'Y', 'N', 'N', '4'}; // grouped + interactions

    NnueNet g_net;
    bool    g_nnue_active = false;

    bool rw_all(std::FILE *f, void *p, std::size_t bytes, bool write) {
      return (write ? std::fwrite(p, 1, bytes, f) : std::fread(p, 1, bytes, f)) == bytes;
    }
  } // namespace

  NnueNet &nnue_net() noexcept { return g_net; }
  bool     nnue_enabled() noexcept { return g_nnue_active && g_net.loaded(); }
  void     nnue_set_active(bool on) noexcept { g_nnue_active = on; }

  void NnueNet::reset() {
    feat_ = pattern_weights_per_stage() + kNnueRFeat;
    emb_.assign(static_cast<std::size_t>(kStageCount) * feat_ * kNnueHidden, 0.0f);
    emb16_.clear(); // stale inference table must not outlive the weights it mirrored
    w2a_.assign(static_cast<std::size_t>(kStageCount) * kNnueHidden, 0.0f);
    w2h_.assign(static_cast<std::size_t>(kStageCount) * kNnueHidden, 0.0f);
    wgroup_.clear();
    winter_.clear();
    b2_.assign(kStageCount, 0.0f);
    grouped_ = false;
    loaded_  = true;
  }

  void NnueNet::set_grouped(bool on) {
    if (on) {
      const std::size_t gn = static_cast<std::size_t>(kStageCount) * kNnueGroups * kNnueHidden;
      const std::size_t in = static_cast<std::size_t>(kStageCount) * kNnuePairs * kNnueHidden;
      if (wgroup_.size() != gn)
        wgroup_.assign(gn, 0.0f);
      if (winter_.size() != in)
        winter_.assign(in, 0.0f);
    }
    grouped_ = on;
  }

  void NnueNet::prepare_training() {
    if (!emb_.empty() || emb16_.empty())
      return;
    emb_.resize(emb16_.size());
    constexpr float kInv = 1.0f / kNnueQuant;
    for (std::size_t i = 0; i < emb16_.size(); ++i)
      emb_[i] = static_cast<float>(emb16_[i]) * kInv;
  }

  void NnueNet::choose_chunk_rows() noexcept {
    int peak = 0;
    for (const std::int16_t value: emb16_)
      peak = std::max(peak, std::abs(static_cast<int>(value)));
    // Keep each lane sum inside int16.
    chunk_rows_ = peak == 0 ? kPatternInstances + 9 : std::max(1, 32767 / peak);
    chunk_rows_ = std::min(chunk_rows_, kPatternInstances + 9);
  }

  void NnueNet::quantize() {
    emb16_.resize(emb_.size());
    for (std::size_t i = 0; i < emb_.size(); ++i)
      emb16_[i] = static_cast<std::int16_t>(std::clamp(std::lround(emb_[i] * kNnueQuant), -32767L, 32767L));
    choose_chunk_rows();
    emb_.clear();
    emb_.shrink_to_fit(); // inference never reads the float table; 80MB back to the OS
  }

  bool NnueNet::selftest() noexcept {
    static_assert(nnue_pair_index(0, 1) == 0 && nnue_pair_index(0, 3) == 2 && nnue_pair_index(1, 2) == 3 &&
                  nnue_pair_index(2, 3) == 5);
    NnueNet       net;
    constexpr int n = kPatternInstances + 9;
    net.feat_       = n;
    net.emb_.assign(static_cast<std::size_t>(kStageCount) * n * kNnueHidden, 1.0f / kNnueQuant);
    net.w2a_.assign(static_cast<std::size_t>(kStageCount) * kNnueHidden, 0.0f);
    net.w2h_.assign(static_cast<std::size_t>(kStageCount) * kNnueHidden, 0.0f);
    net.b2_.assign(kStageCount, 0.0f);
    net.w2a_[0] = 1.0f;
    net.loaded_ = true;
    net.quantize();

    std::uint32_t idx[n];
    for (int i = 0; i < n; ++i)
      idx[i] = static_cast<std::uint32_t>(i);
    const int legacy = net.score(idx, n, 0);
    net.set_grouped(true); // zero residual heads must be an exact NN3 warm start
    const int zero_grouped   = net.score(idx, n, 0);
    net.wgroup_[0]           = 1.0f;
    const int active_grouped = net.score(idx, n, 0);
    net.set_grouped(false);
    return legacy == zero_grouped && active_grouped > zero_grouped && net.score(idx, n, 0) == legacy;
  }

  int NnueNet::score(const std::uint32_t *idx, int n, int stage) const noexcept {
    if (grouped_)
      return score_grouped(idx, n, stage);
    const std::int16_t *base = emb16_.data() + static_cast<std::size_t>(stage) * feat_ * kNnueHidden;
    // M3: bulk PRFM regresses both NN3 and NN4 by about 10%.
#if !defined(__ARM_NEON)
    for (int i = 0; i < n; ++i)
      __builtin_prefetch(base + static_cast<std::size_t>(idx[i]) * kNnueHidden, 0, 0);
#endif
    const float    *a = w2a_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    const float    *h = w2h_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    float           s;
    constexpr float kInv = 1.0f / kNnueQuant;
#if defined(__ARM_NEON)
    static_assert(kNnueHidden == 8, "the NEON path is written for H = 8");
    int32x4_t     s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
    constexpr int kInferenceRows = kPatternInstances + 9;
    if (n == kInferenceRows && chunk_rows_ >= 8) {
      // Fixed chunks let Clang unroll loads without overflowing int16.
      constexpr int kRowsPerChunk = 8;
      constexpr int kFullRows     = kInferenceRows / kRowsPerChunk * kRowsPerChunk;
      for (int first = 0; first < kFullRows; first += kRowsPerChunk) {
        int16x8_t chunk = vdupq_n_s16(0);
        for (int i = 0; i < kRowsPerChunk; ++i) {
          const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[first + i]) * kNnueHidden);
          chunk               = vaddq_s16(chunk, row);
        }
        s0 = vaddw_s16(s0, vget_low_s16(chunk));
        s1 = vaddw_s16(s1, vget_high_s16(chunk));
      }
      for (int i = kFullRows; i < kInferenceRows; ++i) {
        const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[i]) * kNnueHidden);
        s0                  = vaddw_s16(s0, vget_low_s16(row));
        s1                  = vaddw_s16(s1, vget_high_s16(row));
      }
    } else {
      for (int first = 0; first < n; first += chunk_rows_) {
        const int end   = std::min(first + chunk_rows_, n);
        int16x8_t chunk = vdupq_n_s16(0);
        for (int i = first; i < end; ++i) {
          const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[i]) * kNnueHidden);
          chunk               = vaddq_s16(chunk, row);
        }
        s0 = vaddw_s16(s0, vget_low_s16(chunk));
        s1 = vaddw_s16(s1, vget_high_s16(chunk));
      }
    }
    const float32x4_t acc0 = vmulq_n_f32(vcvtq_f32_s32(s0), kInv);
    const float32x4_t acc1 = vmulq_n_f32(vcvtq_f32_s32(s1), kInv);
    const float32x4_t r0   = vmaxq_f32(acc0, vdupq_n_f32(0.0f)); // relu
    const float32x4_t r1   = vmaxq_f32(acc1, vdupq_n_f32(0.0f));
    float32x4_t       sv   = vmulq_f32(vld1q_f32(a), acc0);
    sv                     = vmlaq_f32(sv, vld1q_f32(a + 4), acc1);
    sv                     = vmlaq_f32(sv, vld1q_f32(h), r0);
    sv                     = vmlaq_f32(sv, vld1q_f32(h + 4), r1);
    s                      = vaddvq_f32(sv) + b2_[static_cast<std::size_t>(stage)];
#else
    std::int32_t acci[kNnueHidden] = {};
    for (int i = 0; i < n; ++i) {
      const std::int16_t *row = base + static_cast<std::size_t>(idx[i]) * kNnueHidden;
      for (int j = 0; j < kNnueHidden; ++j)
        acci[j] += row[j];
    }
    s = b2_[static_cast<std::size_t>(stage)];
    for (int j = 0; j < kNnueHidden; ++j) {
      const float aj = static_cast<float>(acci[j]) * kInv;
      s += a[j] * aj + (aj > 0.0f ? h[j] * aj : 0.0f);
    }
#endif
    // Keep static scores below the terminal range.
    const long cd = std::lround(static_cast<double>(s) * 100.0);
    return static_cast<int>(std::clamp(cd, -6399L, 6399L));
  }

  int NnueNet::score_grouped(const std::uint32_t *idx, int n, int stage) const noexcept {
    const std::int16_t *base = emb16_.data() + static_cast<std::size_t>(stage) * feat_ * kNnueHidden;
#if !defined(__ARM_NEON)
    for (int i = 0; i < n; ++i)
      __builtin_prefetch(base + static_cast<std::size_t>(idx[i]) * kNnueHidden, 0, 0);
#endif
    const float    *a  = w2a_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    const float    *h  = w2h_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    const float    *wg = wgroup_.data() + static_cast<std::size_t>(stage) * kNnueGroups * kNnueHidden;
    const float    *wi = winter_.data() + static_cast<std::size_t>(stage) * kNnuePairs * kNnueHidden;
    float           s;
    constexpr float kInv = 1.0f / kNnueQuant;
#if defined(__ARM_NEON)
    static_assert(kNnueHidden == 8, "the NEON path is written for H = 8");
    int32x4_t gs0[kNnueGroups], gs1[kNnueGroups];
    for (int g = 0; g < kNnueGroups; ++g) {
      gs0[g] = vdupq_n_s32(0);
      gs1[g] = vdupq_n_s32(0);
    }
    const auto add_range = [&](int group, int first, int end) {
      for (; first < end; first += chunk_rows_) {
        const int stop  = std::min(first + chunk_rows_, end);
        int16x8_t chunk = vdupq_n_s16(0);
        for (int i = first; i < stop; ++i) {
          const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[i]) * kNnueHidden);
          chunk               = vaddq_s16(chunk, row);
        }
        gs0[group] = vaddw_s16(gs0[group], vget_low_s16(chunk));
        gs1[group] = vaddw_s16(gs1[group], vget_high_s16(chunk));
      }
    };
    constexpr int kInferenceRows = kPatternInstances + 9;
    if (n == kInferenceRows && chunk_rows_ >= 8) {
      // Fixed chunks preserve the int16 overflow bound.
      const auto add8 = [&](int group, int first) {
        int16x8_t chunk = vdupq_n_s16(0);
        for (int i = 0; i < 8; ++i) {
          const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[first + i]) * kNnueHidden);
          chunk               = vaddq_s16(chunk, row);
        }
        gs0[group] = vaddw_s16(gs0[group], vget_low_s16(chunk));
        gs1[group] = vaddw_s16(gs1[group], vget_high_s16(chunk));
      };
      const auto add_tail = [&](int group, int first, int count) {
        for (int i = 0; i < count; ++i) {
          const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[first + i]) * kNnueHidden);
          gs0[group]          = vaddw_s16(gs0[group], vget_low_s16(row));
          gs1[group]          = vaddw_s16(gs1[group], vget_high_s16(row));
        }
      };
      add8(0, 0);
      add8(0, 38);
      add8(1, 8);
      add_tail(1, 16, 4);
      add8(2, 20);
      add8(2, 28);
      add_tail(2, 36, 2);
      add8(3, 46);
      add_tail(3, 54, 1);
    } else if (n == kInferenceRows) {
      add_range(0, 0, 8);
      add_range(1, 8, 20);
      add_range(2, 20, 38);
      add_range(0, 38, 46);
      add_range(3, 46, kInferenceRows);
    } else {
      for (int i = 0; i < n; ++i) {
        const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[i]) * kNnueHidden);
        const int       g   = nnue_feature_group(i);
        gs0[g]              = vaddw_s16(gs0[g], vget_low_s16(row));
        gs1[g]              = vaddw_s16(gs1[g], vget_high_s16(row));
      }
    }
    float32x4_t gr0[kNnueGroups], gr1[kNnueGroups];
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (int g = 0; g < kNnueGroups; ++g) {
      const float32x4_t ga0 = vmulq_n_f32(vcvtq_f32_s32(gs0[g]), kInv);
      const float32x4_t ga1 = vmulq_n_f32(vcvtq_f32_s32(gs1[g]), kInv);
      acc0                  = vaddq_f32(acc0, ga0);
      acc1                  = vaddq_f32(acc1, ga1);
      gr0[g]                = vmaxq_f32(ga0, vdupq_n_f32(0.0f));
      gr1[g]                = vmaxq_f32(ga1, vdupq_n_f32(0.0f));
    }
    const float32x4_t r0 = vmaxq_f32(acc0, vdupq_n_f32(0.0f));
    const float32x4_t r1 = vmaxq_f32(acc1, vdupq_n_f32(0.0f));
    float32x4_t       sv = vmulq_f32(vld1q_f32(a), acc0);
    sv                   = vmlaq_f32(sv, vld1q_f32(a + 4), acc1);
    sv                   = vmlaq_f32(sv, vld1q_f32(h), r0);
    sv                   = vmlaq_f32(sv, vld1q_f32(h + 4), r1);
    for (int g = 0; g < kNnueGroups; ++g) {
      const float *head = wg + static_cast<std::size_t>(g) * kNnueHidden;
      sv                = vmlaq_f32(sv, vld1q_f32(head), gr0[g]);
      sv                = vmlaq_f32(sv, vld1q_f32(head + 4), gr1[g]);
    }
    int pair = 0;
    for (int g = 0; g < kNnueGroups; ++g)
      for (int q = g + 1; q < kNnueGroups; ++q) {
        const float      *head = wi + static_cast<std::size_t>(pair++) * kNnueHidden;
        const float32x4_t x0   = vmulq_n_f32(vmulq_f32(gr0[g], gr0[q]), 1.0f / kNnueInteractionScale);
        const float32x4_t x1   = vmulq_n_f32(vmulq_f32(gr1[g], gr1[q]), 1.0f / kNnueInteractionScale);
        sv                     = vmlaq_f32(sv, vld1q_f32(head), x0);
        sv                     = vmlaq_f32(sv, vld1q_f32(head + 4), x1);
      }
    s = vaddvq_f32(sv) + b2_[static_cast<std::size_t>(stage)];
#else
    std::int32_t gacci[kNnueGroups][kNnueHidden] = {};
    for (int i = 0; i < n; ++i) {
      const std::int16_t *row = base + static_cast<std::size_t>(idx[i]) * kNnueHidden;
      const int           g   = nnue_feature_group(i);
      for (int j = 0; j < kNnueHidden; ++j)
        gacci[g][j] += row[j];
    }
    s = b2_[static_cast<std::size_t>(stage)];
    for (int j = 0; j < kNnueHidden; ++j) {
      float acc = 0.0f;
      float gr[kNnueGroups];
      for (int g = 0; g < kNnueGroups; ++g) {
        const float ga = static_cast<float>(gacci[g][j]) * kInv;
        acc += ga;
        gr[g] = std::max(ga, 0.0f);
        s += wg[(static_cast<std::size_t>(g) * kNnueHidden) + j] * gr[g];
      }
      s += a[j] * acc + (acc > 0.0f ? h[j] * acc : 0.0f);
      int pair = 0;
      for (int g = 0; g < kNnueGroups; ++g)
        for (int q = g + 1; q < kNnueGroups; ++q)
          s += wi[(static_cast<std::size_t>(pair++) * kNnueHidden) + j] * gr[g] * gr[q] / kNnueInteractionScale;
    }
#endif
    const long cd = std::lround(static_cast<double>(s) * 100.0);
    return static_cast<int>(std::clamp(cd, -6399L, 6399L));
  }

  // NN2 float embeddings remain loadable; NN3 and NN4 store int16.
  bool NnueNet::save(const std::string &path, std::ostream &log) const {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
      log << "info error: cannot write " << path << '\n';
      return false;
    }
    std::vector<std::int16_t> q(emb_.size());
    for (std::size_t i = 0; i < emb_.size(); ++i)
      q[i] = static_cast<std::int16_t>(std::clamp(std::lround(emb_[i] * kNnueQuant), -32767L, 32767L));
    std::uint32_t hidden = kNnueHidden, phases = 0, stages = kStageCount;
    std::uint64_t feat  = feat_;
    const char   *magic = grouped_ ? kMagic4 : kMagic3;
    bool ok = std::fwrite(magic, 1, 8, f) == 8 && rw_all(f, &hidden, 4, true) && rw_all(f, &phases, 4, true) &&
              rw_all(f, &stages, 4, true) && rw_all(f, &feat, 8, true) && rw_all(f, q.data(), q.size() * 2, true) &&
              rw_all(f, const_cast<float *>(w2a_.data()), w2a_.size() * 4, true) &&
              rw_all(f, const_cast<float *>(w2h_.data()), w2h_.size() * 4, true) &&
              rw_all(f, const_cast<float *>(b2_.data()), b2_.size() * 4, true);
    if (grouped_)
      ok = ok && rw_all(f, const_cast<float *>(wgroup_.data()), wgroup_.size() * 4, true) &&
           rw_all(f, const_cast<float *>(winter_.data()), winter_.size() * 4, true);
    std::fclose(f);
    if (!ok)
      log << "info error: short write to " << path << '\n';
    else
      log << "info string nnue: saved " << path << '\n';
    return ok;
  }

  bool NnueNet::load(const std::string &path, std::ostream &log) {
    loaded_      = false;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
      log << "info error: cannot open " << path << '\n';
      return false;
    }
    char          magic[8];
    std::uint32_t hidden = 0, phases = 0, stages = 0;
    std::uint64_t feat = 0;
    bool       hdr = std::fread(magic, 1, 8, f) == 8 && rw_all(f, &hidden, 4, false) && rw_all(f, &phases, 4, false) &&
                     rw_all(f, &stages, 4, false) && rw_all(f, &feat, 8, false);
    const bool is3 = hdr && std::memcmp(magic, kMagic3, 8) == 0;
    const bool is2 = hdr && std::memcmp(magic, kMagic, 8) == 0;
    const bool is4 = hdr && std::memcmp(magic, kMagic4, 8) == 0;
    if (!hdr || (!is2 && !is3 && !is4) || hidden != kNnueHidden || phases != 0 || stages != kStageCount ||
        feat != pattern_weights_per_stage() + kNnueRFeat) {
      std::fclose(f);
      log << "info error: " << path << " is not a compatible ISLAY NNUE net\n";
      return false;
    }
    feat_                  = feat;
    const std::size_t embn = static_cast<std::size_t>(kStageCount) * feat_ * kNnueHidden;
    w2a_.resize(static_cast<std::size_t>(kStageCount) * kNnueHidden);
    w2h_.resize(static_cast<std::size_t>(kStageCount) * kNnueHidden);
    b2_.resize(kStageCount);
    bool ok;
    if (is3 || is4) {
      emb16_.resize(embn);
      emb_.clear();
      emb_.shrink_to_fit();
      ok = rw_all(f, emb16_.data(), embn * 2, false);
    } else {
      emb_.resize(embn);
      ok = rw_all(f, emb_.data(), embn * 4, false);
    }
    ok = ok && rw_all(f, w2a_.data(), w2a_.size() * 4, false) && rw_all(f, w2h_.data(), w2h_.size() * 4, false) &&
         rw_all(f, b2_.data(), b2_.size() * 4, false);
    if (is4) {
      wgroup_.resize(static_cast<std::size_t>(kStageCount) * kNnueGroups * kNnueHidden);
      winter_.resize(static_cast<std::size_t>(kStageCount) * kNnuePairs * kNnueHidden);
      ok = ok && rw_all(f, wgroup_.data(), wgroup_.size() * 4, false) &&
           rw_all(f, winter_.data(), winter_.size() * 4, false);
    } else {
      wgroup_.clear();
      winter_.clear();
    }
    std::fclose(f);
    if (!ok) {
      log << "info error: short read from " << path << '\n';
      return false;
    }
    loaded_  = true;
    grouped_ = is4;
    if (is2)
      quantize(); // float NN2 -> the int16 inference table
    else
      choose_chunk_rows();
    log << "info string nnue: loaded " << path << " (" << (emb16_.size() * 2 / (1024 * 1024)) << " MiB)\n";
    return true;
  }

} // namespace islay
