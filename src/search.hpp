/**
 * @file search.hpp
 * @brief Negamax / PVS search: picks the move the engine actually plays.
 *
 * Scores are centi-discs from the MOVER's point of view (see eval.hpp).
 *
 * Two properties this search leans on, both specific to Othello:
 *
 *  1. **A pass costs no depth.** Every real move fills exactly one empty square,
 *     so with passes free, `depth >= empties` guarantees every leaf is terminal
 *     and the returned score is the exact game-theoretic result. That is the
 *     endgame solver, for free, and it is what `SearchResult::exact` reports.
 *     (A pass cannot chain: it only recurses into a board that has moves.)
 *  2. **No repetition is possible.** Real moves only ever add discs, so the tree
 *     is acyclic. There is no repetition/50-move rule to implement, and a PV
 *     walked out of the transposition table cannot loop.
 */
#ifndef ISLAY_SEARCH_HPP
#define ISLAY_SEARCH_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

#include "board.hpp"
#include "common.hpp"
#include "options.hpp"
#include "pattern.hpp"
#include "searchstats.hpp"

namespace islay {

  /** Stop conditions. Zero means "not limited by this"; at least one must be set. */
  struct SearchLimits {
    int           depth      = 0;
    std::uint64_t nodes      = 0;
    double        movetime_ms = 0.0;
  };

  struct SearchResult {
    Square        best     = NOMOVE; // PASS if the mover must pass; NOMOVE if the game is over
    int           score    = 0;
    int           depth    = 0; // last COMPLETED iteration (nominal depth)
    int           seldepth = 0; // deepest ply actually reached, incl. extensions and passes
    std::uint64_t nodes    = 0;
    bool          exact    = false; // depth reached the empty count -> game-theoretic result
  };

  class Searcher {
  public:
    explicit Searcher(std::size_t mib = 256) { resize(mib); }

    /** Reallocate the transposition table to about `mib` MiB (also wipes it). */
    void resize(std::size_t mib);

    /** Forget everything: TT, killers, history, age. Use on `ucinewgame`. */
    void clear() noexcept;

    /** Search `root` under `rule`; `info` receives one UCI info line per iteration.
     *  `stm` is only used to case the PV (White uppercase / Black lowercase) --
     *  Board is mover-relative and carries no colour of its own. */
    SearchResult search(const Board &root, const SearchLimits &limits, Rule rule, Color stm, std::ostream &info);

    /**
     * TEST HOOK -- leave enabled in play.
     *
     * A transposition table legitimately answers a depth-N query with a value
     * proved by a DEEPER search (`e.depth >= depth`), because deeper is strictly
     * better information. The consequence is that an iterative-deepening + TT
     * search does NOT compute the fixed-depth minimax value -- this is classic
     * search instability, not a defect. So a fixed-depth oracle can only be
     * compared against the search with the table switched off; the table's own
     * soundness is checked separately, on exact (depth >= empties) solves where
     * every stored score is the true game result and depth mixing is harmless.
     */
    void set_tt_enabled(bool on) noexcept { tt_enabled_ = on; }

    /**
     * TEST HOOK -- leave enabled in play. Extensions and reductions/futility
     * deliberately search deeper or shallower than `depth` says, so the result is
     * no longer the depth-d minimax value. "Equals a fixed-depth minimax" is not
     * a property a selective searcher has (or wants), so the oracle comparison
     * switches them off; what they MUST still preserve is the exact solve, which
     * is checked separately.
     */
    void set_selective_enabled(bool on) noexcept { selective_enabled_ = on; }

    /**
     * ProbCut, at runtime. Not a test hook: it lets the match harness put two
     * SEARCH configurations against each other in one process, the way swapping the
     * weight pointer pits two evals. This is mandatory equipment, not a nicety --
     * late move pruning cut nodes 55% and the engine 154 Elo, so a node count is
     * not evidence about a pruning technique and only games at equal TIME are.
     */
    void set_probcut_enabled(bool on) noexcept { probcut_enabled_ = on; }

    /** ProbCut confidence in sigmas (default 1.5). Lower = more, riskier cuts. */
    void set_probcut_t(float t) noexcept { probcut_t_ = t; }

    /** ProbCut probe GATE: skip a d-2 probe the static eval says will almost surely
     *  miss (telemetry showed ~62% of probe attempts cut nothing while the probes were
     *  68% of all searched nodes). MEASURED +25 Elo, 95% CI [9, 40], z=3.18 over 600
     *  pairs, and it is the first node-saving change to also raise completed depth at
     *  equal time. Default ON; `match pcg` is the A/B. Exact-safe: skipping a
     *  probabilistic cut only ever makes a node more accurate. */
    void set_probcut_gate_enabled(bool on) noexcept { probcut_gate_enabled_ = on; }

    /** Probe across 4 plies instead of 2 at deep nodes: same parity, ~5x cheaper probe.
     *  MEASURED +19 Elo, 95% CI [4, 34], z=2.47 over 600 pairs, all six seeds positive,
     *  and it raises completed depth at equal time at both 50ms and 500ms. The node
     *  saving GROWS with depth (-13% at d14, -41% at d16) because the wider gap only
     *  fires from depth 9 up, so it should be worth more at longer time controls than
     *  the 50ms this was measured at. Default ON; `match pcg4` is the A/B. */
    void set_probcut_gap4(bool on) noexcept { probcut_gap4_ = on; }

    /** Late move reduction, at runtime -- for an equal-time A/B of LMR itself. */
    void set_lmr_enabled(bool on) noexcept { lmr_enabled_ = on; }

    /** CALIBRATED reduction table (fitted from the engine's own re-search profile)
     *  instead of the old fixed rule. `match lmrc` is the A/B. */
    void set_lmr_calibrated(bool on) noexcept { lmr_calibrated_ = on; }

    /** Late move PRUNING (drop tail of list). Default off; being re-tested on v12. */
    void set_lmp_enabled(bool on) noexcept { lmp_enabled_ = on; }

    /** The endgame stack (tight solver + stability cutoff + parity ordering), for an A/B. */
    void set_endgame_enabled(bool on) noexcept { endgame_enabled_ = on; }

    /** Parity-aware aspiration windows, at runtime -- for an equal-time A/B. */
    void set_aspiration_enabled(bool on) noexcept { aspiration_enabled_ = on; }

    /** Per-stage (Multi-ProbCut) fit vs the pooled per-depth one. Default OFF: measured
     *  neutral at 50ms (-9 Elo, CI [-36,18]) because v12's shallow->deep predictability
     *  is phase-uniform, so the pooled fit is already calibrated. Kept for longer TC. */
    void set_mpc_perstage(bool on) noexcept { mpc_perstage_ = on; }

    /**
     * Search telemetry (opt-in, compiled out unless `kStats` in search.cpp is on).
     * `stats()` is null in a normal build -- callers key the "enabled?" question off
     * that, so no separate flag is exported. It is reset at the start of every
     * search(), so a dump reflects exactly the last `go`.
     */
    [[nodiscard]] const SearchStats *stats() const noexcept { return stats_.get(); }

    /**
     * The mover-relative static eval of a position, built from scratch (no search).
     * Byte-identical to what the search sees at a depth-0 leaf. Exposed for `pcdata`,
     * which fits the static eval against the deep search score to calibrate the
     * ProbCut PROBE GATE (skip a probe the static eval says almost surely won't cut).
     */
    [[nodiscard]] int static_eval(const Board &b, Color stm) const noexcept;

  private:
    enum : std::uint8_t { kNone = 0, kExact = 1, kLower = 2, kUpper = 3 };

    // 16 bytes: two per cache line. `flag == kNone` (0) marks an unused slot, so
    // a zeroed table can never be misread as a real EXACT score.
    struct Entry {
      std::uint64_t key;
      std::int16_t  score;
      std::uint8_t  depth;
      std::uint8_t  flag;
      std::uint8_t  best; // 0..63, or >= 64 for "none" -- ALWAYS validated before use
      std::uint8_t  age;
      std::uint8_t  pad[2];
    };
    static_assert(sizeof(Entry) == 16, "TT entry must stay 16 bytes");

    // ply != depth once passes are free, so this is sized by the real worst case
    // (~60 moves interleaved with ~60 passes), not by depth.
    static constexpr int kMaxPly = 128;

    std::vector<Entry> tt_;
    std::size_t        mask_       = 0;
    std::uint8_t       age_        = 0;
    bool               tt_enabled_        = true;
    bool               selective_enabled_ = true;
    bool               probcut_enabled_   = true;
    float              probcut_t_         = 1.5f;
    bool               probcut_gap4_         = true;  // MEASURED +19 Elo; see kProbCutFit4
    bool               probcut_gate_enabled_ = true;  // MEASURED +25 Elo; see kProbCutGateFit
    bool               lmr_enabled_       = true;
    bool               lmr_calibrated_    = false; // MEASURED WORSE -- see lmr_reduction
    bool               lmp_enabled_       = false;
    bool               endgame_enabled_   = true;
    bool               aspiration_enabled_ = true;
    bool               mpc_perstage_      = false; // pooled per-depth by default; see kMpcFit note

    // Copy-and-update per ply: `ps_[ply+1] = ps_[ply]; update(...)`. 38 int32 =
    // 152B/ply, which is what makes an incremental pattern eval possible at all
    // against a value-semantic Board (there is no make/unmake to hang state on).
    //
    // Heap, and only allocated when weights are loaded: inline, this array is
    // 19KB sitting INSIDE Searcher, next to the hot fields (nodes_, mask_,
    // killers_). Cheap insurance, not a measured fix.
    std::vector<PatternState> ps_;
    bool                      pat_on_ = false; // cached per search; see new_search()

    Square killers_[kMaxPly][2]{};
    int    history_[64]{};

    std::uint64_t nodes_    = 0;
    std::uint64_t node_cap_ = 0;
    double        deadline_ms_ = 0.0;
    double        start_ms_    = 0.0;
    bool          stopped_     = false;
    int           seldepth_    = 0; // max ply reached this iteration
    std::size_t   tt_used_     = 0; // distinct slots written since the last clear (for hashfull)

    // Opt-in telemetry (see searchstats.hpp). Allocated lazily in new_search only
    // when kStats is on, so a shipping build never pays the ~200KB or the counters.
    std::unique_ptr<SearchStats> stats_;

    // `stm` is threaded because the pattern eval's features are colour-ABSOLUTE
    // (Board is mover-relative and carries no colour). It costs one register and
    // flips every ply, passes included.
    // `Pat` is a template parameter so `if constexpr (Pat)` compiles the pattern
    // hooks out entirely when no weights are loaded (the default) -- the
    // off-by-default eval then provably cannot tax the default path.
    //
    // Honesty note: an earlier A/B seemed to show a runtime check costing ~5-7%,
    // and that number was WRONG -- it compared a CMake/LTO build against a
    // hand-rolled `-flto` one. Re-run through the SAME build system the cost is
    // nil (61ms vs 62ms at depth 13). Templating is kept because it makes the
    // cost provably zero rather than merely measured-equal, but do not cite the
    // old figure. Lesson: an A/B across two build systems measures the builds.
    //
    // `stm` is threaded because pattern features are colour-ABSOLUTE and Board
    // carries no colour of its own.
    template<Rule R, bool Pat>
    ISLAY_HOT ISLAY_FLATTEN int pvs(Board b, int depth, int alpha, int beta, int ply, Color stm) noexcept;
    template<Rule R, bool Pat>
    void root_search(const Board &root, int depth, Color stm, int alpha, int beta, SearchResult &res) noexcept;
    template<bool Pat>
    [[nodiscard]] int leaf_eval(const Board &b, Bitboard moves, int ply, Color stm) const noexcept;

    void check_stop() noexcept;
    void new_search(const SearchLimits &limits) noexcept;
    [[nodiscard]] std::string pv_string(Board b, Rule rule, Square first, Color stm, int max_len);
  };

  /** Cross-check the optimized search against a plain negamax oracle. */
  [[nodiscard]] bool search_selftest() noexcept;

} // namespace islay

#endif // ISLAY_SEARCH_HPP
