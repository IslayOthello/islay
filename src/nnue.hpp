// Per-stage NNUE over pattern features; NN4 adds grouped and interaction residual heads.
#ifndef ISLAY_NNUE_HPP
#define ISLAY_NNUE_HPP

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "pattern.hpp"

namespace islay {

  inline constexpr int   kNnueHidden           = 8;
  inline constexpr int   kNnueRFeat            = 4; // the appended r = (discs-4)%4 one-hot rows
  inline constexpr int   kNnueGroups           = 4;
  inline constexpr int   kNnuePairs            = kNnueGroups * (kNnueGroups - 1) / 2;
  inline constexpr float kNnueQuant            = 64.0f; // int16 grid: 1/64 disc ~ 1.6 cd per step
  inline constexpr float kNnueInteractionScale = 8.0f;

  // Ordinal mapping avoids a leaf-time lookup.
  [[nodiscard]] inline constexpr int nnue_feature_group(int active) noexcept {
    if (active < 8 || (active >= 38 && active < 46))
      return 0; // Corner3x3, Edge2X, Corner2x5
    if (active < 20)
      return 1; // Row2..Row4
    if (active < 38)
      return 2; // Diag8..Diag4
    return 3; // bias, mobility, stability, parity, frontier, interpolation r
  }

  [[nodiscard]] inline constexpr int nnue_pair_index(int a, int b) noexcept {
    if (a > b) {
      const int t = a;
      a           = b;
      b           = t;
    }
    return a * (2 * kNnueGroups - a - 1) / 2 + (b - a - 1);
  }

  class NnueNet {
  public:
    [[nodiscard]] bool        loaded() const noexcept { return loaded_; }
    [[nodiscard]] bool        grouped() const noexcept { return grouped_; }
    [[nodiscard]] static bool selftest() noexcept;
    bool                      load(const std::string &path, std::ostream &log);
    bool                      save(const std::string &path, std::ostream &log) const;

    void reset();

    // Restore float embeddings for warm-start training.
    void prepare_training();

    // A legacy net gets zeroed residual heads.
    void set_grouped(bool on);

    // Black-relative centi-disc score.
    [[nodiscard]] int score(const std::uint32_t *idx, int n, int stage) const noexcept;

    // Trainer access to flat float tables.
    [[nodiscard]] float      *emb() noexcept { return emb_.data(); }
    [[nodiscard]] float      *w2a() noexcept { return w2a_.data(); }
    [[nodiscard]] float      *w2h() noexcept { return w2h_.data(); }
    [[nodiscard]] float      *wgroup() noexcept { return wgroup_.data(); }
    [[nodiscard]] float      *winter() noexcept { return winter_.data(); }
    [[nodiscard]] float      *b2() noexcept { return b2_.data(); }
    [[nodiscard]] std::size_t features() const noexcept { return feat_; } // per_stage + kNnueRFeat

    // Build the int16 inference table and drop float embeddings.
    void quantize();

  private:
    void              choose_chunk_rows() noexcept;
    [[nodiscard]] int score_grouped(const std::uint32_t *idx, int n, int stage) const noexcept;

    std::vector<float>        emb_; // [kStageCount][feat_][kNnueHidden], training + file IO
    std::vector<std::int16_t> emb16_; // same layout, x kNnueQuant -- what score() reads
    std::vector<float>        w2a_; // [kStageCount][kNnueHidden]
    std::vector<float>        w2h_; // [kStageCount][kNnueHidden]
    std::vector<float>        wgroup_; // [kStageCount][kNnueGroups][kNnueHidden]
    std::vector<float>        winter_; // [kStageCount][kNnuePairs][kNnueHidden]
    std::vector<float>        b2_; // [kStageCount]
    std::size_t               feat_       = 0;
    int                       chunk_rows_ = 1;
    bool                      loaded_     = false;
    bool                      grouped_    = false;
  };

  [[nodiscard]] NnueNet &nnue_net() noexcept;
  [[nodiscard]] bool     nnue_enabled() noexcept;
  void                   nnue_set_active(bool on) noexcept;

} // namespace islay

#endif // ISLAY_NNUE_HPP
