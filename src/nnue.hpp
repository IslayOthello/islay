/**
 * @file nnue.hpp
 * @brief NNUE-lite: a two-layer non-linear eval over the existing pattern features.
 *
 * Each feature index selects an 8-float embedding row instead of a scalar; the rows
 * of the active features are summed, and a per-stage head adds a relu term the linear
 * eval cannot express:
 *     acc[j] = sum over active features f of E[stage][f][j]
 *     score  = W2a[stage] . acc  +  W2h[stage] . relu(acc)  +  b2[stage]     (discs)
 * from BLACK's point of view, like the linear eval. The skip term (W2a) lets a warm
 * start reproduce v18 exactly (dim 0 = the v18 weight, W2a = (1,0,..)).
 *
 * Per-stage tables, not phase-bucketed: v1 bucketed by phase and lost 57 Elo, having
 * dropped the two most-valuable measured properties (4-disc stage resolution + stage
 * interpolation). Interpolation returns as an INPUT here: r = (discs-4)%4 appended as
 * a feature (kNnueRFeat rows/stage), so the net learns its own within-bucket blend.
 */
#ifndef ISLAY_NNUE_HPP
#define ISLAY_NNUE_HPP

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "pattern.hpp"

namespace islay {

  inline constexpr int kNnueHidden = 8;
  inline constexpr int kNnueRFeat  = 4; // the appended r = (discs-4)%4 one-hot rows
  inline constexpr float kNnueQuant = 64.0f; // int16 grid: 1/64 disc ~ 1.6 cd per step

  class NnueNet {
  public:
    [[nodiscard]] bool loaded() const noexcept { return loaded_; }
    bool load(const std::string &path, std::ostream &log);
    bool save(const std::string &path, std::ostream &log) const;

    /** Allocate and zero; the trainer then fills it (see identity init there). */
    void reset();

    /** Restore the float embedding table from the quantized inference table.
     *  Heads are already stored as floats, so this makes a loaded NN3 net an exact
     *  warm start for another training round. The inference table stays live until
     *  the updated float table is saved and quantized. */
    void prepare_training();

    /**
     * BLACK's score in centi-discs. `idx` are the flat per-stage feature indices
     * the pattern layer already produces (pattern_indices with stage 0), plus the
     * caller-appended r index (features() - kNnueRFeat + r).
     */
    [[nodiscard]] int score(const std::uint32_t *idx, int n, int stage) const noexcept;

    // Trainer access: training runs on the float table, laid out flat.
    [[nodiscard]] float *emb() noexcept { return emb_.data(); }
    [[nodiscard]] float *w2a() noexcept { return w2a_.data(); }
    [[nodiscard]] float *w2h() noexcept { return w2h_.data(); }
    [[nodiscard]] float *b2() noexcept { return b2_.data(); }
    [[nodiscard]] std::size_t features() const noexcept { return feat_; } // per_stage + kNnueRFeat

    /** Build the int16 inference table from the float one and drop the float table.
     *  Called by load(); a trainer that keeps the net live calls it after save(). */
    void quantize();

  private:
    void choose_chunk_rows() noexcept;

    std::vector<float>        emb_;   // [kStageCount][feat_][kNnueHidden], training + file IO
    std::vector<std::int16_t> emb16_; // same layout, x kNnueQuant -- what score() reads
    std::vector<float>        w2a_;   // [kStageCount][kNnueHidden]
    std::vector<float>        w2h_;   // [kStageCount][kNnueHidden]
    std::vector<float>        b2_;    // [kStageCount]
    std::size_t               feat_ = 0;
    int                       chunk_rows_ = 1;
    bool                      loaded_ = false;
  };

  /** The active net, or null. Loaded via `setoption EvalFile <file>.nnue`. */
  [[nodiscard]] NnueNet &nnue_net() noexcept;
  [[nodiscard]] bool     nnue_enabled() noexcept;
  void                   nnue_set_active(bool on) noexcept;

} // namespace islay

#endif // ISLAY_NNUE_HPP
