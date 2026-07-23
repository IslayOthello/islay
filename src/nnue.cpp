/**
 * @file nnue.cpp
 * @brief NNUE-lite net: storage, file IO, and the leaf-time forward pass (nnue.hpp).
 */
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
    constexpr char kMagic[8] = {'I', 'S', 'L', 'A', 'Y', 'N', 'N', '2'};

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
    w2a_.assign(static_cast<std::size_t>(kStageCount) * kNnueHidden, 0.0f);
    w2h_.assign(static_cast<std::size_t>(kStageCount) * kNnueHidden, 0.0f);
    b2_.assign(kStageCount, 0.0f);
    loaded_ = true;
  }

  int NnueNet::score(const std::uint32_t *idx, int n, int stage) const noexcept {
    // Sum the active features' embedding rows. A row is kNnueHidden floats = 32
    // bytes, two rows per cache line, so a leaf touches about as many lines as the
    // linear eval did -- but this table is 80MB where the linear one is 5MB, so the
    // lines are usually COLD, and the real cost is DRAM. All row addresses are known
    // up front, so they are prefetched in one burst before any is consumed: the
    // misses overlap each other instead of the (short) add chain.
    const float *base = emb_.data() + static_cast<std::size_t>(stage) * feat_ * kNnueHidden;
    for (int i = 0; i < n; ++i)
      __builtin_prefetch(base + static_cast<std::size_t>(idx[i]) * kNnueHidden, 0, 0);
    const float *a = w2a_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    const float *h = w2h_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    float        s;
#if defined(__ARM_NEON)
    static_assert(kNnueHidden == 8, "the NEON path is written for H = 8");
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; ++i) {
      const float *row = base + static_cast<std::size_t>(idx[i]) * kNnueHidden;
      acc0             = vaddq_f32(acc0, vld1q_f32(row));
      acc1             = vaddq_f32(acc1, vld1q_f32(row + 4));
    }
    // Per-stage head: linear skip term + relu term (see nnue.hpp). relu(x) * h ==
    // max(x,0) * h vectorises directly.
    const float32x4_t r0 = vmaxq_f32(acc0, vdupq_n_f32(0.0f));
    const float32x4_t r1 = vmaxq_f32(acc1, vdupq_n_f32(0.0f));
    float32x4_t       sv = vmulq_f32(vld1q_f32(a), acc0);
    sv                   = vmlaq_f32(sv, vld1q_f32(a + 4), acc1);
    sv                   = vmlaq_f32(sv, vld1q_f32(h), r0);
    sv                   = vmlaq_f32(sv, vld1q_f32(h + 4), r1);
    s                    = vaddvq_f32(sv) + b2_[static_cast<std::size_t>(stage)];
#else
    float acc[kNnueHidden] = {};
    for (int i = 0; i < n; ++i) {
      const float *row = base + static_cast<std::size_t>(idx[i]) * kNnueHidden;
      for (int j = 0; j < kNnueHidden; ++j)
        acc[j] += row[j];
    }
    // Per-stage head: linear skip term + relu term. The skip is what lets the net be
    // initialised AS the linear eval; the relu dimensions carry everything a linear
    // model cannot (see nnue.hpp).
    s = b2_[static_cast<std::size_t>(stage)];
    for (int j = 0; j < kNnueHidden; ++j)
      s += a[j] * acc[j] + (acc[j] > 0.0f ? h[j] * acc[j] : 0.0f);
#endif
    // The net works in DISCS; the engine speaks centi-discs, clamped inside the
    // terminal range exactly like the linear eval (eval.cpp's kEvalMax contract).
    const long cd = std::lround(static_cast<double>(s) * 100.0);
    return static_cast<int>(std::clamp(cd, -6399L, 6399L));
  }

  bool NnueNet::save(const std::string &path, std::ostream &log) const {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
      log << "info error: cannot write " << path << '\n';
      return false;
    }
    std::uint32_t hidden = kNnueHidden, phases = 0 /* v2: per-stage */, stages = kStageCount;
    std::uint64_t feat = feat_;
    bool ok = std::fwrite(kMagic, 1, 8, f) == 8 && rw_all(f, &hidden, 4, true) && rw_all(f, &phases, 4, true) &&
              rw_all(f, &stages, 4, true) && rw_all(f, &feat, 8, true) &&
              rw_all(f, const_cast<float *>(emb_.data()), emb_.size() * 4, true) &&
              rw_all(f, const_cast<float *>(w2a_.data()), w2a_.size() * 4, true) &&
              rw_all(f, const_cast<float *>(w2h_.data()), w2h_.size() * 4, true) &&
              rw_all(f, const_cast<float *>(b2_.data()), b2_.size() * 4, true);
    std::fclose(f);
    if (!ok)
      log << "info error: short write to " << path << '\n';
    else
      log << "info string nnue: saved " << path << '\n';
    return ok;
  }

  bool NnueNet::load(const std::string &path, std::ostream &log) {
    loaded_ = false;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
      log << "info error: cannot open " << path << '\n';
      return false;
    }
    char          magic[8];
    std::uint32_t hidden = 0, phases = 0, stages = 0;
    std::uint64_t feat = 0;
    bool hdr = std::fread(magic, 1, 8, f) == 8 && rw_all(f, &hidden, 4, false) && rw_all(f, &phases, 4, false) &&
               rw_all(f, &stages, 4, false) && rw_all(f, &feat, 8, false);
    if (!hdr || std::memcmp(magic, kMagic, 8) != 0 || hidden != kNnueHidden || phases != 0 ||
        stages != kStageCount || feat != pattern_weights_per_stage() + kNnueRFeat) {
      std::fclose(f);
      log << "info error: " << path << " is not a compatible ISLAYNN2 net\n";
      return false;
    }
    feat_ = feat;
    emb_.resize(static_cast<std::size_t>(kStageCount) * feat_ * kNnueHidden);
    w2a_.resize(static_cast<std::size_t>(kStageCount) * kNnueHidden);
    w2h_.resize(static_cast<std::size_t>(kStageCount) * kNnueHidden);
    b2_.resize(kStageCount);
    bool ok = rw_all(f, emb_.data(), emb_.size() * 4, false) && rw_all(f, w2a_.data(), w2a_.size() * 4, false) &&
              rw_all(f, w2h_.data(), w2h_.size() * 4, false) && rw_all(f, b2_.data(), b2_.size() * 4, false);
    std::fclose(f);
    if (!ok) {
      log << "info error: short read from " << path << '\n';
      return false;
    }
    loaded_ = true;
    log << "info string nnue: loaded " << path << " (" << (emb_.size() * 4 / (1024 * 1024)) << " MiB)\n";
    return true;
  }

} // namespace islay
