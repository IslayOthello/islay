/**
 * @file nnue.hpp
 * @brief A small non-linear evaluation ("NNUE-lite") over the existing pattern features.
 *
 * WHY THIS EXISTS. The linear pattern eval is provably blind to any function that is
 * not a weighted sum of its windows -- and every eval feature that ever won here
 * (mobility +52, stability +12.7, parity +12.1, frontier +11.1) won precisely by
 * hand-delivering one such function as a new input. This is the general form of that
 * move: instead of a scalar weight per feature, each feature index selects a small
 * VECTOR (an embedding), the vectors of all active features are summed, and one
 * hidden non-linearity lets the network represent interactions between features that
 * no linear model can, however many windows it is given.
 *
 * SHAPE (v2).
 *     acc[j]  = sum over active features f of E[stage][f][j]         (j < kHidden)
 *     score   = W2a[stage] . acc  +  W2h[stage] . relu(acc)  +  b2[stage]
 * in DISCS (x100 for centi-discs), from BLACK's point of view like the linear eval.
 *
 * The FIRST version of this file bucketed embeddings by phase (3 stages each) to
 * save memory, and lost 57 Elo [-73, -42] to the linear eval at equal nodes. The
 * post-mortem was clean: that design threw away exactly the two properties this
 * project MEASURED to be the most valuable -- fine 4-disc stage resolution, and
 * stage interpolation (worth ~+58 on its own) which the net had no equivalent of.
 * Hence v2's deliberate choices:
 *   * Embeddings are PER STAGE (15 tables, H=8): dim 0 is initialised to the v18
 *     weight for THAT stage, so with W2a = (1,0,..) the warm start reproduces v18
 *     exactly -- not a phase-averaged approximation of it.
 *   * Interpolation becomes an INPUT: one extra feature, r = (discs-4)%4, appended
 *     as index per_stage + r. Four embedding rows per stage let the net learn its
 *     own within-bucket blend, per stage and non-linearly, for one extra gather.
 *   * A row is 8 floats = 32 bytes, two rows per cache line: a leaf still touches
 *     ~the same number of lines as the linear eval -- the added cost is arithmetic,
 *     not memory traffic, and arithmetic is what the machine has spare.
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

  class NnueNet {
  public:
    [[nodiscard]] bool loaded() const noexcept { return loaded_; }
    bool load(const std::string &path, std::ostream &log);
    bool save(const std::string &path, std::ostream &log) const;

    /** Allocate and zero; the trainer then fills it (see identity init there). */
    void reset();

    /**
     * BLACK's score in centi-discs. `idx` are the flat per-stage feature indices
     * the pattern layer already produces (pattern_indices with stage 0), plus the
     * caller-appended r index (features() - kNnueRFeat + r).
     */
    [[nodiscard]] int score(const std::uint32_t *idx, int n, int stage) const noexcept;

    // Trainer access: everything is plain float, laid out flat.
    [[nodiscard]] float *emb() noexcept { return emb_.data(); }
    [[nodiscard]] float *w2a() noexcept { return w2a_.data(); }
    [[nodiscard]] float *w2h() noexcept { return w2h_.data(); }
    [[nodiscard]] float *b2() noexcept { return b2_.data(); }
    [[nodiscard]] std::size_t features() const noexcept { return feat_; } // per_stage + kNnueRFeat

  private:
    std::vector<float> emb_;  // [kStageCount][feat_][kNnueHidden]
    std::vector<float> w2a_;  // [kStageCount][kNnueHidden]
    std::vector<float> w2h_;  // [kStageCount][kNnueHidden]
    std::vector<float> b2_;   // [kStageCount]
    std::size_t        feat_ = 0;
    bool               loaded_ = false;
  };

  /** The active net, or null. Loaded via `setoption EvalFile <file>.nnue`. */
  [[nodiscard]] NnueNet &nnue_net() noexcept;
  [[nodiscard]] bool     nnue_enabled() noexcept;
  void                   nnue_set_active(bool on) noexcept;

} // namespace islay

#endif // ISLAY_NNUE_HPP
