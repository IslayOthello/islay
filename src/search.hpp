// Mover-relative search; passes cost no depth and positions cannot repeat.
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

  struct SearchLimits {
    int           depth       = 0;
    std::uint64_t nodes       = 0;
    double        movetime_ms = 0.0;
    // Raw clock and increment for engine-side allocation.
    double time_ms = 0.0;
    double inc_ms  = 0.0;
  };

  // Per-searcher parameters for A/B and SPSA; defaults are the shipped values.
  struct SearchParams {
    int fut1 = 250, fut2 = 450, fut3 = 700; // futility margins at depth 1/2/3, centi-discs
    int killer0 = 65536, killer1 = 32768; // move-ordering bonus for the two killers
    int mob_w        = 32; // ordering weight per opponent reply
    int hist_div     = 64; // history score divisor
    int parity_bonus = 8192; // endgame odd-quadrant ordering nudge
    int sqv_mult     = 8; // static square-value multiplier
  };

  struct SearchResult {
    Square        best     = NOMOVE; // PASS if the mover must pass; NOMOVE if the game is over
    int           score    = 0;
    int           depth    = 0; // last COMPLETED iteration (nominal depth)
    int           seldepth = 0; // deepest ply actually reached, incl. extensions and passes
    std::uint64_t nodes    = 0;
    bool          exact    = false; // depth reached the empty count -> game-theoretic result
  };

  // Shared lockless TT. Hyatt XOR validation turns torn 16-byte writes into misses.
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

    void new_generation() noexcept { age_.fetch_add(1, std::memory_order_relaxed); }

    [[nodiscard]] bool probe(std::uint64_t key, Hit &out) const noexcept;
    void               store(std::uint64_t key, int score, int depth, std::uint8_t flag, std::uint8_t best) noexcept;

    void prefetch(std::uint64_t key) const noexcept {
      if (mask_)
        ISLAY_PREFETCH(&slots_[key & mask_]);
    }

    [[nodiscard]] int hashfull() const noexcept;

    // ABDADA marks are hints; stale or lost marks only change ordering.
    void busy_enter(std::uint64_t key) noexcept { busy_[key & (kBusySlots - 1)].store(key, std::memory_order_relaxed); }
    void busy_leave(std::uint64_t key) noexcept {
      std::uint64_t e = key; // clear only our own mark; losing the race is benign
      busy_[key & (kBusySlots - 1)].compare_exchange_strong(e, 0, std::memory_order_relaxed, std::memory_order_relaxed);
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

    std::vector<Slot>                             slots_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> busy_ =
            std::make_unique<std::atomic<std::uint64_t>[]>(kBusySlots); // zero-initialised
    std::size_t               mask_ = 0;
    std::atomic<std::uint8_t> age_{0};
    std::atomic<std::size_t>  used_{0};
  };

  // Online residual model used only by the ProbCut gate.
  class CorrectionHistory {
  public:
    void              clear() noexcept;
    [[nodiscard]] int predict(const Board &b, int stage, Square prev_move, Square prev2_move) const noexcept;
    void update(const Board &b, int stage, Square prev_move, Square prev2_move, int deep_score, int static_score,
                int depth) noexcept;

  private:
    [[nodiscard]] static unsigned edge_bucket(const Board &b) noexcept;
    static void                   update_entry(std::int16_t &entry, int target, int rate) noexcept;

    std::int16_t stage_[kStageCount]{};
    std::int16_t prev_[kStageCount][64]{};
    std::int16_t prev2_[kStageCount][64]{};
    std::int16_t edge_[kStageCount][256]{};
  };

  class Searcher {
  public:
    explicit Searcher(std::size_t mib = 256) : tt_(std::make_shared<TranspositionTable>()) { resize(mib); }

    // Lazy SMP shares only the TT; other state stays thread-local.
    void               share_table_with(const Searcher &other) noexcept { tt_ = other.tt_; }
    [[nodiscard]] bool shares_table_with(const Searcher &other) const noexcept { return tt_ == other.tt_; }

    void set_bump_age(bool on) noexcept { bump_age_ = on; }

    void                              set_params(const SearchParams &p) noexcept { params_ = p; }
    [[nodiscard]] const SearchParams &params() const noexcept { return params_; }

    void set_abdada(bool on) noexcept { abdada_ = on; }

    void set_wld(bool on) noexcept { wld_enabled_ = on; }

    void set_tm_adaptive(bool on) noexcept { tm_adaptive_ = on; }

    void set_depth_offset(int d) noexcept { depth_offset_ = d < 0 ? 0 : d; }

    void resize(std::size_t mib);

    void clear() noexcept;

    // check_stop polls this relaxed flag every 1024 nodes.
    void request_stop() noexcept { stop_flag_.store(true, std::memory_order_relaxed); }

    // Arm before launching the search thread.
    void arm() noexcept { stop_flag_.store(false, std::memory_order_relaxed); }

    SearchResult search(const Board &root, const SearchLimits &limits, Rule rule, Color stm, std::ostream &info);

    // Oracle hook; fixed-depth checks disable TT depth mixing.
    void set_tt_enabled(bool on) noexcept { tt_enabled_ = on; }

    // Oracle hook; fixed-depth checks disable selectivity.
    void set_selective_enabled(bool on) noexcept { selective_enabled_ = on; }

    void set_probcut_enabled(bool on) noexcept { probcut_enabled_ = on; }

    void set_probcut_t(float t) noexcept { probcut_t_ = t; }

    void set_probcut_gate_enabled(bool on) noexcept { probcut_gate_enabled_ = on; }

    void set_correction_history_cap(int cap) noexcept;

    void set_probcut_gap4(bool on) noexcept { probcut_gap4_ = on; }

    void set_nmp(bool on) noexcept { nmp_enabled_ = on; }

    void set_lmr_enabled(bool on) noexcept { lmr_enabled_ = on; }

    void set_lmr_calibrated(bool on) noexcept { lmr_calibrated_ = on; }

    void set_lmp_enabled(bool on) noexcept { lmp_enabled_ = on; }

    void set_endgame_enabled(bool on) noexcept { endgame_enabled_ = on; }

    void set_aspiration_enabled(bool on) noexcept { aspiration_enabled_ = on; }

    void set_mpc_perstage(bool on) noexcept { mpc_perstage_ = on; }

    // Null in normal builds; reset per search when telemetry is compiled in.
    [[nodiscard]] const SearchStats *stats() const noexcept { return stats_.get(); }

    // Scratch leaf eval used by pcdata.
    [[nodiscard]] int static_eval(const Board &b, Color stm) const noexcept;

  private:
    using Entry                          = TranspositionTable::Hit;
    static constexpr std::uint8_t kNone  = TranspositionTable::kNone;
    static constexpr std::uint8_t kExact = TranspositionTable::kExact;
    static constexpr std::uint8_t kLower = TranspositionTable::kLower;
    static constexpr std::uint8_t kUpper = TranspositionTable::kUpper;

    // Covers moves interleaved with passes.
    static constexpr int kMaxPly                 = 128;
    bool                 tt_enabled_             = true;
    bool                 selective_enabled_      = true;
    bool                 probcut_enabled_        = true;
    float                probcut_t_              = 1.5f;
    bool                 probcut_gap4_           = true;
    bool                 probcut_gate_enabled_   = true;
    int                  correction_history_cap_ = 200;
    bool                 lmr_enabled_            = true;
    bool                 lmr_calibrated_         = false;
    bool                 lmp_enabled_            = false;
    bool                 nmp_enabled_            = false;
    bool                 endgame_enabled_        = true;
    bool                 aspiration_enabled_     = true;
    bool                 mpc_perstage_           = false;

    // PatternState is heap-backed and allocated only when weights are loaded.
    std::shared_ptr<TranspositionTable> tt_;
    bool                                bump_age_    = true;
    bool                                tm_adaptive_ = false;
    bool                                abdada_      = false;
    bool                                wld_enabled_ = false;
    SearchParams                        params_;
    int                                 depth_offset_ = 0;

    std::vector<PatternState> ps_;
    Bitboard                  move_stack_[kMaxPly]{};
    bool                      pat_on_  = false;
    bool                      nnue_on_ = false;

    Square            killers_[kMaxPly][2]{};
    int               history_[64]{};
    int               continuation_history_[64][64]{};
    int               continuation_history_2_[64][64]{};
    CorrectionHistory correction_history_{};

    std::uint64_t     nodes_       = 0;
    std::uint64_t     node_cap_    = 0;
    double            deadline_ms_ = 0.0;
    double            soft_ms_     = 0.0;
    double            start_ms_    = 0.0;
    bool              stopped_     = false;
    std::atomic<bool> stop_flag_{false};
    int               seldepth_ = 0;


    // Allocated lazily only in telemetry builds.
    std::unique_ptr<SearchStats> stats_;

    // Pat compiles pattern hooks out; stm tracks colour-absolute features.
    template<Rule R, bool Pat>
    ISLAY_HOT ISLAY_FLATTEN int pvs(Board b, int depth, int alpha, int beta, int ply, Color stm,
                                    Square prev_move = NOMOVE, Square prev2_move = NOMOVE,
                                    bool can_null = true) noexcept;
    template<Rule R, bool Pat>
    void root_search(const Board &root, int depth, Color stm, int alpha, int beta, SearchResult &res) noexcept;
    template<bool Pat>
    [[nodiscard]] int leaf_eval(const Board &b, Bitboard moves, int ply, Color stm) const noexcept;

    void                      check_stop() noexcept;
    void                      new_search(const SearchLimits &limits) noexcept;
    [[nodiscard]] std::string pv_string(Board b, Rule rule, Square first, Color stm, int max_len);
  };

  [[nodiscard]] bool search_selftest() noexcept;

} // namespace islay

#endif // ISLAY_SEARCH_HPP
