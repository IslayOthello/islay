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
    constexpr char kMagic[8]  = {'I', 'S', 'L', 'A', 'Y', 'N', 'N', '2'}; // float emb (legacy)
    constexpr char kMagic3[8] = {'I', 'S', 'L', 'A', 'Y', 'N', 'N', '3'}; // int16 emb (shipped)

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
    b2_.assign(kStageCount, 0.0f);
    loaded_ = true;
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
    // A chunk's worst possible lane sum stays inside int16; v19 therefore adds
    // ten rows per vector before it needs the wider int32 accumulator.
    chunk_rows_ = peak == 0 ? kPatternInstances + 9 : std::max(1, 32767 / peak);
    chunk_rows_ = std::min(chunk_rows_, kPatternInstances + 9);
  }

  void NnueNet::quantize() {
    emb16_.resize(emb_.size());
    for (std::size_t i = 0; i < emb_.size(); ++i)
      emb16_[i] = static_cast<std::int16_t>(
          std::clamp(std::lround(emb_[i] * kNnueQuant), -32767L, 32767L));
    choose_chunk_rows();
    emb_.clear();
    emb_.shrink_to_fit(); // inference never reads the float table; 80MB back to the OS
  }

  int NnueNet::score(const std::uint32_t *idx, int n, int stage) const noexcept {
    // The rows are scattered across a cold 40MB table, so their addresses are all
    // prefetched up front to overlap the misses before any row is summed.
    const std::int16_t *base = emb16_.data() + static_cast<std::size_t>(stage) * feat_ * kNnueHidden;
    for (int i = 0; i < n; ++i)
      __builtin_prefetch(base + static_cast<std::size_t>(idx[i]) * kNnueHidden, 0, 0);
    const float *a = w2a_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    const float *h = w2h_.data() + static_cast<std::size_t>(stage) * kNnueHidden;
    float        s;
    constexpr float kInv = 1.0f / kNnueQuant;
#if defined(__ARM_NEON)
    static_assert(kNnueHidden == 8, "the NEON path is written for H = 8");
    int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
    constexpr int kInferenceRows = kPatternInstances + 9;
    if (n == kInferenceRows && chunk_rows_ >= 8) {
      // v20 admits eight rows per narrow accumulator. Keeping that trip count
      // compile-time constant lets Clang issue eight independent scattered
      // loads without the min/compare/branch pair of the generic inner loop.
      // Six full chunks plus the final row preserve the exact int16 grouping.
      constexpr int kRowsPerChunk = 8;
      constexpr int kFullRows = kInferenceRows / kRowsPerChunk * kRowsPerChunk;
      for (int first = 0; first < kFullRows; first += kRowsPerChunk) {
        int16x8_t chunk = vdupq_n_s16(0);
        for (int i = 0; i < kRowsPerChunk; ++i) {
          const int16x8_t row =
              vld1q_s16(base + static_cast<std::size_t>(idx[first + i]) * kNnueHidden);
          chunk = vaddq_s16(chunk, row);
        }
        s0 = vaddw_s16(s0, vget_low_s16(chunk));
        s1 = vaddw_s16(s1, vget_high_s16(chunk));
      }
      for (int i = kFullRows; i < kInferenceRows; ++i) {
        const int16x8_t row = vld1q_s16(base + static_cast<std::size_t>(idx[i]) * kNnueHidden);
        s0                    = vaddw_s16(s0, vget_low_s16(row));
        s1                    = vaddw_s16(s1, vget_high_s16(row));
      }
    } else {
      for (int first = 0; first < n; first += chunk_rows_) {
        const int end = std::min(first + chunk_rows_, n);
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
    const float32x4_t r0 = vmaxq_f32(acc0, vdupq_n_f32(0.0f)); // relu
    const float32x4_t r1 = vmaxq_f32(acc1, vdupq_n_f32(0.0f));
    float32x4_t       sv = vmulq_f32(vld1q_f32(a), acc0);
    sv                   = vmlaq_f32(sv, vld1q_f32(a + 4), acc1);
    sv                   = vmlaq_f32(sv, vld1q_f32(h), r0);
    sv                   = vmlaq_f32(sv, vld1q_f32(h + 4), r1);
    s                    = vaddvq_f32(sv) + b2_[static_cast<std::size_t>(stage)];
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
    // discs -> centi-discs, clamped inside the terminal range like the linear eval.
    const long cd = std::lround(static_cast<double>(s) * 100.0);
    return static_cast<int>(std::clamp(cd, -6399L, 6399L));
  }

  // The shipped file (NN3) stores the embedding table as int16 -- exactly what inference
  // reads, half the size, no float precision wasted on disk. NN2 (float emb) is still
  // accepted so trained nets from before the format change load; a float NN2 is
  // quantized on load, an int16 NN3 goes straight to emb16_.
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
    std::uint64_t feat = feat_;
    bool ok = std::fwrite(kMagic3, 1, 8, f) == 8 && rw_all(f, &hidden, 4, true) && rw_all(f, &phases, 4, true) &&
              rw_all(f, &stages, 4, true) && rw_all(f, &feat, 8, true) && rw_all(f, q.data(), q.size() * 2, true) &&
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
    const bool is3 = hdr && std::memcmp(magic, kMagic3, 8) == 0;
    const bool is2 = hdr && std::memcmp(magic, kMagic, 8) == 0;
    if (!hdr || (!is2 && !is3) || hidden != kNnueHidden || phases != 0 || stages != kStageCount ||
        feat != pattern_weights_per_stage() + kNnueRFeat) {
      std::fclose(f);
      log << "info error: " << path << " is not a compatible ISLAY NNUE net\n";
      return false;
    }
    feat_ = feat;
    const std::size_t embn = static_cast<std::size_t>(kStageCount) * feat_ * kNnueHidden;
    w2a_.resize(static_cast<std::size_t>(kStageCount) * kNnueHidden);
    w2h_.resize(static_cast<std::size_t>(kStageCount) * kNnueHidden);
    b2_.resize(kStageCount);
    bool ok;
    if (is3) {
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
    std::fclose(f);
    if (!ok) {
      log << "info error: short read from " << path << '\n';
      return false;
    }
    loaded_ = true;
    if (is2)
      quantize(); // float NN2 -> the int16 inference table
    else
      choose_chunk_rows();
    log << "info string nnue: loaded " << path << " (" << (emb16_.size() * 2 / (1024 * 1024)) << " MiB)\n";
    return true;
  }

} // namespace islay
