/**
 * @file match.hpp
 * @brief Engine-vs-engine match: the only honest way to answer "is this stronger?".
 *
 * Every eval or search change in this repo is a guess until a match measures it.
 * Node counts and nps say how FAST the engine is, never how WELL it plays -- an
 * eval change usually costs nodes and is still worth it, or is free and still
 * loses. So this harness exists to settle those questions.
 *
 * Three things it does that a naive loop gets wrong, and without which the
 * numbers are worthless:
 *
 *  1. **Opening diversity.** Both engines are deterministic, so from the start
 *     position every game would be a replay of the same one. Each pairing starts
 *     from a short random opening instead.
 *  2. **Colour balance.** Black has an edge, so every opening is played TWICE
 *     with the engines swapped. A result is only counted as a pair.
 *  3. **Error bars, over PAIRS.** "A won 52%" over 100 games is noise. The report
 *     carries the standard error, a z-score against "no difference", and an Elo
 *     estimate with a confidence interval. The observation unit is the PAIR, not the
 *     game: the two games of a pair share an opening and swap colours, so they are
 *     not independent draws, and pooling them as if they were measures the wrong
 *     variance. The pairing that removes colour bias from the estimate has to be
 *     respected in the error bar too. The measured within-pair correlation is
 *     reported so this stays a fact and not an assumption.
 */
#ifndef ISLAY_MATCH_HPP
#define ISLAY_MATCH_HPP

#include <cstdint>
#include <ostream>
#include <string>

#include "options.hpp"
#include "pattern.hpp"
#include "search.hpp"

namespace islay {

  struct MatchConfig {
    int           pairs      = 50;  // games = 2 * pairs (each opening played both ways)
    int           depth      = 6;   // per-move search depth (0 -> use movetime)
    double        movetime_ms = 0.0;
    int           opening_plies = 6; // random plies before the engines take over
    std::uint64_t seed       = 0;
    Rule          rule       = Rule::Othello;
    std::string   eval_a;           // ISLAYPAT path for side A; empty = hand-written eval
    std::string   eval_b;           // ditto for side B
    bool          pc_a = true;      // search config per side; see Searcher::set_probcut_enabled
    bool          pc_b = true;
    bool          lmr_a = true;     // ditto for late move reduction
    bool          lmr_b = true;
    bool          mpc_a = true;     // ditto for per-stage (Multi) ProbCut vs pooled per-depth
    bool          mpc_b = true;
    float         pct_a = 1.5f;     // ProbCut t (sigmas) per side, for a t-sweep
    float         pct_b = 1.5f;
    bool          lmp_a = false;    // late move pruning per side (re-test on trained eval)
    bool          lmp_b = false;
    bool          eg_a = true;      // endgame stack per side (exact speedup)
    bool          eg_b = true;
    double        tc_base_ms = 0.0; // clock time control: base + increment per move (0 = off)
    double        tc_inc_ms  = 0.0;
    bool          si_a = false;     // stage interpolation per side (A/B of the eval feature)
    bool          si_b = false;
    bool          pcg_a = false;    // ProbCut probe gate per side (skip hopeless probes)
    bool          pcg_b = false;
    bool          lmrc_a = false;   // calibrated LMR reduction table per side
    bool          lmrc_b = false;
  };

  struct MatchResult {
    int    wins = 0, draws = 0, losses = 0; // from A's point of view
    double score = 0.0;                     // (wins + draws/2) / games
    double stderr_ = 0.0; // SE of the mean, computed over PAIRS -- see match.cpp
    double z     = 0.0; // (score - 0.5) / stderr; |z| > 1.96 is ~95% confidence
    // Diagnostics for the pairing itself. `pair_rho` is the measured correlation
    // between the two games of a pair; `stderr_naive` is what pooling all games as
    // independent would have given. They differ by exactly sqrt(1 + rho), and
    // reporting both is what keeps the pairing honest rather than assumed.
    double pair_rho     = 0.0;
    double stderr_naive = 0.0;
    double elo    = 0.0;
    double elo_lo = 0.0; // ~95% interval. NOT symmetric: the Elo transform is
    double elo_hi = 0.0; // non-linear, so a "+/-" would misstate it (and could
                         // even straddle 0 while the z-test says significant).
    [[nodiscard]] bool significant() const noexcept { return z > 1.96 || z < -1.96; }
  };

  /** Play the match, streaming progress to `log`. */
  MatchResult run_match(const MatchConfig &cfg, std::ostream &log);

} // namespace islay

#endif // ISLAY_MATCH_HPP
