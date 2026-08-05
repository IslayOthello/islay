#include "match.hpp"

#include <chrono>
#include <cmath>
#include <sstream>

#include "eval.hpp"
#include "movegen.hpp"

namespace islay {
  namespace {
    using Clock = std::chrono::steady_clock;

    struct Rng {
      std::uint64_t s;
      std::uint64_t next() noexcept {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
      }
    };

    struct Opening {
      Board b;
      Color stm;
      bool  ok;
    };

    [[nodiscard]] Opening random_opening(Rng &rng, int plies, Rule rule) noexcept {
      Board b   = Board::start();
      Color stm = Color::Black;
      for (int i = 0; i < plies; ++i) {
        Bitboard m = b.moves();
        if (m == 0) {
          const Board p = b.passed();
          if (rule != Rule::Othello || !p.has_moves())
            return {b, stm, false};
          b   = p;
          stm = ~stm;
          continue;
        }
        unsigned k = static_cast<unsigned>(rng.next() % static_cast<unsigned>(popcount(m)));
        while (k-- > 0)
          m &= m - 1;
        b   = b.play(lsb(m));
        stm = ~stm;
      }
      return {b, stm, b.moves() != 0 || b.passed().has_moves()};
    }

    [[nodiscard]] double play_game(const Opening &op, bool a_is_black, PatternWeights *wa, PatternWeights *wb,
                                   Searcher &sa, Searcher &sb, const MatchConfig &cfg) {
      Board              b   = op.b;
      Color              stm = op.stm;
      std::ostringstream sink; // engines' info lines are noise here
      const bool         tc  = cfg.tc_base_ms > 0.0; // clock time control vs fixed depth/movetime

      double clk_a = cfg.tc_base_ms, clk_b = cfg.tc_base_ms;
      int    flagged = 0; // 0 none, 1 = A lost on time, 2 = B lost on time

      for (int ply = 0; ply < 80; ++ply) {
        if (b.moves() == 0) {
          const Board p = b.passed();
          if (cfg.rule != Rule::Othello || !p.has_moves())
            break; // game over
          b   = p;
          stm = ~stm;
          continue;
        }
        const bool a_to_move = (stm == Color::Black) == a_is_black; // A plays Black iff a_is_black
        pattern_set_active(a_to_move ? wa : wb);
        pattern_set_stage_interp(a_to_move ? cfg.si_a : cfg.si_b);
        Searcher &sr = a_to_move ? sa : sb;

        SearchLimits lim{cfg.depth, 0, cfg.movetime_ms};
        double      *clk = a_to_move ? &clk_a : &clk_b;
        if (tc) {
          if (*clk <= 0.0) { // out of time before even moving
            flagged = a_to_move ? 1 : 2;
            break;
          }
          if (a_to_move ? cfg.etm_a : cfg.etm_b) {
            lim = SearchLimits{0, 0, 0.0, *clk, cfg.tc_inc_ms};
          } else {
            const int    ml     = std::max(1, (64 - b.count()) / 2);
            const double budget = std::min(*clk, *clk / ml + cfg.tc_inc_ms);
            lim                 = SearchLimits{0, 0, budget};
          }
        }

        const auto t0 = Clock::now();
        const SearchResult r = sr.search(b, lim, cfg.rule, stm, sink);
        if (tc) {
          const double used = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
          *clk -= used;
          *clk += cfg.tc_inc_ms;
          if (*clk <= 0.0) { // overspent -> flag falls
            flagged = a_to_move ? 1 : 2;
            break;
          }
        }

        if (r.best == NOMOVE)
          break;
        if (r.best == PASS) {
          b   = b.passed();
          stm = ~stm;
          continue;
        }
        b   = b.play(r.best);
        stm = ~stm;
      }
      pattern_set_active(nullptr);
      pattern_set_stage_interp(false);

      if (flagged == 1)
        return 0.0; // A lost on time
      if (flagged == 2)
        return 1.0; // B lost on time

      const int mine   = popcount(b.player); // `b` is mover-relative to `stm`
      const int theirs = popcount(b.opponent);
      const int stm_diff = mine - theirs;
      const int black_diff = (stm == Color::Black) ? stm_diff : -stm_diff;
      const int a_diff     = a_is_black ? black_diff : -black_diff;
      if (a_diff > 0)
        return 1.0;
      if (a_diff < 0)
        return 0.0;
      return 0.5;
    }

  } // namespace

  MatchResult run_match(const MatchConfig &cfg, std::ostream &log) {
    MatchResult res;

    PatternWeights wa, wb; // unloaded == hand-written eval
    if (!cfg.eval_a.empty() && !wa.load(cfg.eval_a, log))
      return res;
    if (!cfg.eval_b.empty() && !wb.load(cfg.eval_b, log))
      return res;

    log << "info string match: A=" << (cfg.eval_a.empty() ? "handcrafted" : cfg.eval_a)
        << "  B=" << (cfg.eval_b.empty() ? "handcrafted" : cfg.eval_b) << "  " << (2 * cfg.pairs) << " games, "
        << (cfg.depth > 0 ? "depth " + std::to_string(cfg.depth)
                          : "movetime " + std::to_string(static_cast<int>(cfg.movetime_ms)) + "ms")
        << ", rule " << rule_name(cfg.rule) << ", seed " << (cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL) << '\n';
    if (cfg.eval_a == cfg.eval_b && cfg.pc_a == cfg.pc_b && cfg.lmr_a == cfg.lmr_b && cfg.mpc_a == cfg.mpc_b)
      log << "info string note: both sides are identical -- expect ~50% (this is the harness's own sanity check)\n";
    log.flush();

    Rng      rng{cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL};
    Searcher sa(32), sb(32);
    sa.set_probcut_enabled(cfg.pc_a);
    sb.set_probcut_enabled(cfg.pc_b);
    sa.set_lmr_enabled(cfg.lmr_a);
    sb.set_lmr_enabled(cfg.lmr_b);
    sa.set_mpc_perstage(cfg.mpc_a);
    sb.set_mpc_perstage(cfg.mpc_b);
    sa.set_probcut_t(cfg.pct_a);
    sb.set_probcut_t(cfg.pct_b);
    sa.set_lmp_enabled(cfg.lmp_a);
    sb.set_lmp_enabled(cfg.lmp_b);
    sa.set_endgame_enabled(cfg.eg_a);
    sb.set_endgame_enabled(cfg.eg_b);
    sa.set_probcut_gate_enabled(cfg.pcg_a);
    sb.set_probcut_gate_enabled(cfg.pcg_b);
    sa.set_lmr_calibrated(cfg.lmrc_a);
    sb.set_lmr_calibrated(cfg.lmrc_b);
    sa.set_probcut_gap4(cfg.pcg4_a);
    sb.set_probcut_gap4(cfg.pcg4_b);
    sa.set_tm_adaptive(cfg.tma_a);
    sb.set_tm_adaptive(cfg.tma_b);
    sa.set_wld(cfg.wld_a);
    sb.set_wld(cfg.wld_b);
    sa.set_nmp(cfg.nmp_a);
    sb.set_nmp(cfg.nmp_b);

    double sum = 0.0, sumsq = 0.0;         // over individual games -- diagnostic only
    double psum = 0.0, psumsq = 0.0;       // over PAIR means -- this is what the stats use
    int    games = 0, done_pairs = 0;
    for (int p = 0; p < cfg.pairs; ++p) {
      const Opening op = random_opening(rng, cfg.opening_plies, cfg.rule);
      if (!op.ok) {
        --p; // opening died early; draw another
        continue;
      }
      double pair_sum = 0.0;
      for (int side = 0; side < 2; ++side) {
        sa.clear();
        sb.clear(); // no table carries between games
        const double r = play_game(op, side == 0, &wa, &wb, sa, sb, cfg);
        sum += r;
        sumsq += r * r;
        pair_sum += r;
        ++games;
        if (r > 0.75)
          ++res.wins;
        else if (r < 0.25)
          ++res.losses;
        else
          ++res.draws;
      }
      const double pm = pair_sum / 2.0;
      psum += pm;
      psumsq += pm * pm;
      ++done_pairs;
      if ((p + 1) % 10 == 0 || p + 1 == cfg.pairs) {
        log << "info string match: " << games << " games, A +" << res.wins << " =" << res.draws << " -" << res.losses
            << "  (" << static_cast<int>(100.0 * sum / games) << "%)\n";
        log.flush();
      }
    }
    if (games == 0)
      return res;

    // Treat each colour-reversed pair as one correlated observation.
    res.score = psum / done_pairs;
    const double pvar = done_pairs > 1 ? (psumsq - psum * psum / done_pairs) / (done_pairs - 1) : 0.0;
    res.stderr_       = done_pairs > 1 ? std::sqrt(pvar / done_pairs) : 0.0;
    res.z             = res.stderr_ > 0.0 ? (res.score - 0.5) / res.stderr_ : 0.0;

    // Var(pair mean) = sigma^2 (1 + rho) / 2, so rho = 2*Var(pair mean)/sigma^2 - 1,
    const double gvar = games > 1 ? (sumsq - sum * sum / games) / (games - 1) : 0.0;
    res.pair_rho      = gvar > 0.0 ? 2.0 * pvar / gvar - 1.0 : 0.0;
    res.stderr_naive  = games > 1 ? std::sqrt(gvar / games) : 0.0;

    const auto to_elo = [](double s) {
      const double e = std::min(std::max(s, 1e-4), 1.0 - 1e-4);
      return -400.0 * std::log10(1.0 / e - 1.0);
    };
    res.elo = to_elo(res.score);
    // Transform score bounds separately because the Elo map is nonlinear.
    res.elo_lo = to_elo(std::max(res.score - 1.96 * res.stderr_, 1e-4));
    res.elo_hi = to_elo(std::min(res.score + 1.96 * res.stderr_, 1.0 - 1e-4));

    log << "matchagg " << done_pairs << ' ' << psum << ' ' << psumsq << ' ' << res.wins << ' ' << res.draws
        << ' ' << res.losses << '\n';
    log << "match done: " << games << " games  A +" << res.wins << " =" << res.draws << " -" << res.losses << '\n'
        << "  score " << res.score << " +/- " << res.stderr_ << "   z " << res.z << '\n'
        << "  pairing: within-pair rho " << res.pair_rho << "  (naive per-game SE would be " << res.stderr_naive
        << ", off by sqrt(1+rho))\n"
        << "  elo   " << static_cast<int>(res.elo) << "  95% CI [" << static_cast<int>(res.elo_lo) << ", "
        << static_cast<int>(res.elo_hi) << "]\n"
        << "  verdict: "
        << (res.significant() ? (res.z > 0 ? "A is stronger (95% confidence)" : "B is stronger (95% confidence)")
                              : "NO significant difference -- more games needed to tell them apart")
        << '\n';
    log.flush();
    return res;
  }

} // namespace islay
