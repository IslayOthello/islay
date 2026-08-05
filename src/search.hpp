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

#include <atomic>
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
    // CLOCK mode (real time control): the MOVER's remaining clock and Fischer
    // increment. When time_ms > 0 the engine allocates its own budget for this move
    // (see the policy in search.cpp) instead of being handed a fixed movetime.
    double        time_ms = 0.0;
    double        inc_ms  = 0.0;
  };

  /**
   * The hand-guessed search constants, gathered so a tuner can move them. Every value
   * here shipped as a constexpr somebody typed in; none of them was ever fitted. They
   * are PER-SEARCHER so the match harness (and the SPSA loop) can give the two sides
   * different values -- which is the entire point. Defaults reproduce the shipped
   * constants exactly, and a Threads=1 search with defaults must stay byte-identical.
   *
   * TUNED, AND THE DEFAULTS WON. Two `tune spsa` runs on v18 at movetime 15ms:
   * a conservative one (2000 iters) converged to within ~3% of the defaults on every
   * parameter and then oscillated there; an aggressive one (800 iters, c x2, a x3,
   * different seed) drifted the futility margins ~10-15% up and left the rest alone.
   * That candidate was validated independently -- first 400 pairs read +11.7 Elo at
   * LOS 0.97, which the boundary-z rule says to distrust, and pooling a second 400
   * pairs on a fresh book collapsed it to +3.7, CI [-5, 13]. So the hand-guessed
   * values below are, at measurement resolution, already the optimum -- consistent
   * with the flat ProbCut t-sweep, the insensitive LMR schedule, and ordering that
   * cuts on the first move 82.7% of the time. The knobs are flat because the engine
   * barely consults them.
   */
  struct SearchParams {
    int fut1 = 250, fut2 = 450, fut3 = 700; // futility margins at depth 1/2/3, centi-discs
    int killer0 = 65536, killer1 = 32768;   // move-ordering bonus for the two killers
    int mob_w   = 32;                       // ordering weight per opponent reply
    int hist_div = 64;                      // history score divisor
    int parity_bonus = 8192;                // endgame odd-quadrant ordering nudge
    int sqv_mult = 8;                       // static square-value multiplier
  };

  struct SearchResult {
    Square        best     = NOMOVE; // PASS if the mover must pass; NOMOVE if the game is over
    int           score    = 0;
    int           depth    = 0; // last COMPLETED iteration (nominal depth)
    int           seldepth = 0; // deepest ply actually reached, incl. extensions and passes
    std::uint64_t nodes    = 0;
    bool          exact    = false; // depth reached the empty count -> game-theoretic result
  };

  /**
   * The transposition table, deliberately SEPARATE from Searcher so that several
   * searchers can share ONE of them. That sharing is the whole mechanism of lazy SMP:
   * the helper threads are not dividing the tree between them, they are filling this
   * table with results the main thread then finds already answered.
   *
   * LOCKLESS, and it has to be. A 16-byte entry cannot be written atomically, so two
   * threads storing to the same slot can leave a mixture of both -- a score from one
   * and a depth from the other. Validating the move (which the search already does)
   * catches an illegal `best`, but nothing catches a plausible score carrying someone
   * else's depth, and this engine PROVES exact endgame results out of this table. So
   * each slot is two 64-bit atomics holding Hyatt's XOR check: the key is stored
   * XOR-ed with the payload, and a reader that recomputes `key_xor ^ data` only gets
   * its key back if the pair it read belongs together. A torn pair simply misses.
   *
   * Relaxed ordering throughout: the table is a hint, every value read from it is
   * re-validated or bounded by the search, and no other memory is published through it.
   */
  class TranspositionTable {
  public:
    enum : std::uint8_t { kNone = 0, kExact = 1, kLower = 2, kUpper = 3 };

    struct Hit {
      int          score = 0;
      int          depth = 0;
      std::uint8_t flag  = kNone;
      std::uint8_t best  = 255;
    };

    void resize(std::size_t mib);
    void clear() noexcept;

    /** One generation per search; older entries lose to newer ones on replacement. */
    void new_generation() noexcept { age_.fetch_add(1, std::memory_order_relaxed); }

    [[nodiscard]] bool probe(std::uint64_t key, Hit &out) const noexcept;
    void store(std::uint64_t key, int score, int depth, std::uint8_t flag, std::uint8_t best) noexcept;

    void prefetch(std::uint64_t key) const noexcept {
      if (mask_) ISLAY_PREFETCH(&slots_[key & mask_]);
    }

    /** Per-mille occupancy, for the UCI `hashfull` field. Approximate under SMP. */
    [[nodiscard]] int hashfull() const noexcept;

    /**
     * ABDADA busy table: which positions are being searched RIGHT NOW, so that other
     * threads defer them instead of duplicating the work -- the failure mode all three
     * divergence schemes measured (helpers on a narrow tree redo the main thread's
     * subtrees). A hint, deliberately imprecise: a stale mark defers a search that
     * did not need deferring and a lost mark permits a duplicate, both of which cost
     * only time, never correctness -- deferral REORDERS moves within a node and every
     * move is still searched unless a real cutoff ends the node.
     */
    void busy_enter(std::uint64_t key) noexcept {
      busy_[key & (kBusySlots - 1)].store(key, std::memory_order_relaxed);
    }
    void busy_leave(std::uint64_t key) noexcept {
      std::uint64_t e = key; // clear only our own mark; losing the race is benign
      busy_[key & (kBusySlots - 1)].compare_exchange_strong(e, 0, std::memory_order_relaxed,
                                                            std::memory_order_relaxed);
    }
    [[nodiscard]] bool busy(std::uint64_t key) const noexcept {
      return busy_[key & (kBusySlots - 1)].load(std::memory_order_relaxed) == key;
    }

  private:
    struct Slot {
      std::atomic<std::uint64_t> key_xor;
      std::atomic<std::uint64_t> data;
    };
    static_assert(sizeof(Slot) == 16, "TT slot must stay 16 bytes");

    static constexpr std::size_t kBusySlots = 1u << 15;

    std::vector<Slot> slots_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> busy_ =
            std::make_unique<std::atomic<std::uint64_t>[]>(kBusySlots); // zero-initialised
    std::size_t                mask_ = 0;
    std::atomic<std::uint8_t>  age_{0};
    std::atomic<std::size_t>   used_{0};
  };

  /** Small online residual model used only by the ProbCut probe gate. */
  class CorrectionHistory {
  public:
    void clear() noexcept;
    [[nodiscard]] int predict(const Board &b, int stage, Square prev_move, Square prev2_move) const noexcept;
    void update(const Board &b, int stage, Square prev_move, Square prev2_move,
                int deep_score, int static_score, int depth) noexcept;

  private:
    [[nodiscard]] static unsigned edge_bucket(const Board &b) noexcept;
    static void update_entry(std::int16_t &entry, int target, int rate) noexcept;

    std::int16_t stage_[kStageCount]{};
    std::int16_t prev_[kStageCount][64]{};
    std::int16_t prev2_[kStageCount][64]{};
    std::int16_t edge_[kStageCount][256]{};
  };

  class Searcher {
  public:
    explicit Searcher(std::size_t mib = 256)
        : tt_(std::make_shared<TranspositionTable>()) {
      resize(mib);
    }

    /** Share another searcher's table. THIS is what makes lazy SMP work -- helpers must
     *  write into the same table the main thread reads. Killers, history, the pattern
     *  stack and the node counter stay per-searcher, which is why each thread needs its
     *  own Searcher rather than just its own stack. */
    void share_table_with(const Searcher &other) noexcept { tt_ = other.tt_; }
    [[nodiscard]] bool shares_table_with(const Searcher &other) const noexcept { return tt_ == other.tt_; }

    /** Helper threads must not bump the shared generation; only the main search does. */
    void set_bump_age(bool on) noexcept { bump_age_ = on; }

    /** Search-constant overrides for tuning; defaults are the shipped values. */
    void set_params(const SearchParams &p) noexcept { params_ = p; }
    [[nodiscard]] const SearchParams &params() const noexcept { return params_; }

    /** ABDADA work deferral for lazy SMP; on only when several threads search. */
    void set_abdada(bool on) noexcept { abdada_ = on; }

    /** Win/Loss/Draw solve at the solving iteration of TIMED searches (see the note in
     *  search.cpp): ~6x cheaper than the exact solve, so it lands ~2 empties earlier.
     *  `match wldtc` is the A/B. */
    void set_wld(bool on) noexcept { wld_enabled_ = on; }

    /** Adaptive time management: scale the soft budget by what iterative deepening
     *  LEARNS -- extend on a changed best move or a score drop, shrink when the move is
     *  obvious. `match tma` is the A/B against the fixed allocation. */
    void set_tm_adaptive(bool on) noexcept { tm_adaptive_ = on; }

    /** Start the iterative deepening at 1 + offset, so helper threads do not all
     *  duplicate the same iteration at the same moment. */
    void set_depth_offset(int d) noexcept { depth_offset_ = d < 0 ? 0 : d; }

    /** Reallocate the transposition table to about `mib` MiB (also wipes it). */
    void resize(std::size_t mib);

    /** Forget everything: TT, killers, history, age. Use on `ucinewgame`. */
    void clear() noexcept;

    /**
     * Ask a running search to stop, FROM ANOTHER THREAD. The search now runs on its
     * own thread, so `stop`, `quit` and any state-changing command arrive while it is
     * still working.
     *
     * The hot path deliberately keeps reading the plain `stopped_` bool: an atomic
     * load at every node would be a real cost for a flag that changes at most once per
     * search. This one is folded into it inside check_stop(), which already runs once
     * every 1024 nodes -- so noticing a stop costs nothing and is at worst a thousand
     * nodes late, which no clock cares about. Relaxed ordering suffices: the flag is
     * the only thing published, and the searching thread's own results are handed over
     * by joining, which is itself a synchronisation point.
     */
    void request_stop() noexcept { stop_flag_.store(true, std::memory_order_relaxed); }

    /** Clear a previous stop. MUST be called by the owner BEFORE starting a search, on
     *  the thread that will later request the stop -- doing it inside the search would
     *  race with a request arriving just after launch and silently swallow it. */
    void arm() noexcept { stop_flag_.store(false, std::memory_order_relaxed); }

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

    /** Online correction for the ProbCut gate. Zero disables it; 100/200 are
     *  conservative A/B candidates in centi-discs. */
    void set_correction_history_cap(int cap) noexcept;

    /** Probe across 4 plies instead of 2 at deep nodes: same parity, ~5x cheaper probe.
     *  MEASURED +19 Elo, 95% CI [4, 34], z=2.47 over 600 pairs, all six seeds positive,
     *  and it raises completed depth at equal time at both 50ms and 500ms. The node
     *  saving GROWS with depth (-13% at d14, -41% at d16) because the wider gap only
     *  fires from depth 9 up, so it should be worth more at longer time controls than
     *  the 50ms this was measured at. Default ON; `match pcg4` is the A/B. */
    void set_probcut_gap4(bool on) noexcept { probcut_gap4_ = on; }

    /** Null-move pruning. Default off; `match nmp` is the A/B. */
    void set_nmp(bool on) noexcept { nmp_enabled_ = on; }

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
    using Entry = TranspositionTable::Hit;
    static constexpr std::uint8_t kNone  = TranspositionTable::kNone;
    static constexpr std::uint8_t kExact = TranspositionTable::kExact;
    static constexpr std::uint8_t kLower = TranspositionTable::kLower;
    static constexpr std::uint8_t kUpper = TranspositionTable::kUpper;

    // ply != depth once passes are free, so this is sized by the real worst case
    // (~60 moves interleaved with ~60 passes), not by depth.
    static constexpr int kMaxPly = 128;
    bool               tt_enabled_        = true;
    bool               selective_enabled_ = true;
    bool               probcut_enabled_   = true;
    float              probcut_t_         = 1.5f;
    bool               probcut_gap4_         = true;  // MEASURED +19 Elo; see kProbCutFit4
    bool               probcut_gate_enabled_ = true;  // MEASURED +25 Elo; see kProbCutGateFit
    int                correction_history_cap_ = 200; // +4.54 Elo at 50ms; zero is the A/B baseline
    bool               lmr_enabled_       = true;
    bool               lmr_calibrated_    = false; // MEASURED WORSE -- see lmr_reduction
    bool               lmp_enabled_       = false;
    bool               nmp_enabled_       = false; // null-move pruning; ships off until `match nmp` wins
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
    std::shared_ptr<TranspositionTable> tt_;
    bool                                bump_age_     = true;
    bool                                tm_adaptive_  = false; // fixed allocation ships until `match tma` wins
    bool                                abdada_       = false; // meaningless single-threaded
    bool                                wld_enabled_  = false; // exact solve ships until `match wldtc` wins
    SearchParams                        params_;
    int                                 depth_offset_ = 0;

    std::vector<PatternState> ps_;
    Bitboard                  move_stack_[kMaxPly]{};
    bool                      pat_on_ = false; // cached per search; see new_search()
    bool                      nnue_on_ = false; // ditto: NNUE leaf instead of the linear sum

    Square killers_[kMaxPly][2]{};
    int    history_[64]{};
    int    continuation_history_[64][64]{};
    int    continuation_history_2_[64][64]{};
    CorrectionHistory correction_history_{};

    std::uint64_t nodes_    = 0;
    std::uint64_t node_cap_ = 0;
    double        deadline_ms_ = 0.0; // HARD: check_stop aborts past this
    double        soft_ms_     = 0.0; // SOFT: the ID loop will not START an iteration past a fraction of it
    double        start_ms_    = 0.0;
    bool          stopped_     = false; // thread-local to the search; see request_stop
    std::atomic<bool> stop_flag_{false}; // set from the command thread
    int           seldepth_    = 0; // max ply reached this iteration


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
    ISLAY_HOT ISLAY_FLATTEN int pvs(Board b, int depth, int alpha, int beta, int ply, Color stm,
                                    Square prev_move = NOMOVE, Square prev2_move = NOMOVE,
                                    bool can_null = true) noexcept;
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
