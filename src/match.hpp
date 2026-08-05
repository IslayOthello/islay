// Paired colour-reversed matches with pair-level error bars.
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
    bool          pcg4_a = false;   // wider (4-ply) ProbCut probe gap at deep nodes
    bool          pcg4_b = false;
    bool          etm_a = false;    // ENGINE time management: hand the side its raw clock and
    bool          etm_b = false;    // let search.cpp allocate, instead of the harness's even split
    bool          tma_a = false;    // adaptive soft budget (see Searcher::set_tm_adaptive)
    bool          tma_b = false;
    bool          wld_a = false;    // Win/Loss/Draw solve at the solving iteration (timed only)
    bool          wld_b = false;
    bool          nmp_a = false;    // null-move pruning per side
    bool          nmp_b = false;
  };

  struct MatchResult {
    int    wins = 0, draws = 0, losses = 0; // from A's point of view
    double score = 0.0;                     // (wins + draws/2) / games
    double stderr_ = 0.0; // SE of the mean, computed over PAIRS -- see match.cpp
    double z     = 0.0; // (score - 0.5) / stderr; |z| > 1.96 is ~95% confidence
    // Pair correlation and the naive independent-game SE.
    double pair_rho     = 0.0;
    double stderr_naive = 0.0;
    double elo    = 0.0;
    double elo_lo = 0.0;
    double elo_hi = 0.0;
    [[nodiscard]] bool significant() const noexcept { return z > 1.96 || z < -1.96; }
  };

  MatchResult run_match(const MatchConfig &cfg, std::ostream &log);

} // namespace islay

#endif // ISLAY_MATCH_HPP
