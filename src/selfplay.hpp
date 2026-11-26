#ifndef ISLAY_SELFPLAY_HPP
#define ISLAY_SELFPLAY_HPP

#include <atomic>
#include <iosfwd>
#include <mutex>
#include <vector>

#include "mcts.hpp"

namespace islay {
  inline constexpr std::size_t kReplayRecordBytes = 560;

  struct SelfplayConfig {
    std::uint64_t seed              = 20260909;
    std::uint32_t simulations       = 128;
    std::uint32_t batch_size        = 1;
    std::size_t   tree_bytes        = 8 * 1024 * 1024;
    double        epsilon           = 0.25;
    double        alpha             = 0.3;
    std::uint16_t temperature_plies = 16; // tau=1 before this ply, then tau=0
    std::uint16_t opening_plies     = 8; // plies with enhanced exploration & soft action sampling
    std::uint32_t fast_simulations  = 0; // 0 disables playout cap randomization; >0 uses fast cap
    double        full_search_prob  = 1.0;
  };

  struct ReplayRecord {
    std::uint64_t                          game_id{};
    Board                                  board;
    Bitboard                               legal{};
    std::array<std::uint32_t, kPolicySize> visits{};
    std::array<float, kPolicySize>         priors{};
    std::uint16_t                          ply{};
    std::uint8_t                           action{}, stm{}, temperature{}, pass{};
    std::int8_t                            outcome{}; // WLD or quantized tanh margin, selected by value_format
    std::uint8_t                           value_format{}; // 0=WLD, 1=tanh(disc margin / 15) scaled by 127
  };

  class CachedEvaluator final : public Evaluator {
  public:
    CachedEvaluator(Evaluator &backend, std::size_t bytes);
    void                      evaluate(std::span<const Board>, std::span<Evaluation>, std::stop_token) override;
    std::uint64_t             hits{}, misses{};
    [[nodiscard]] std::size_t bytes() const noexcept;

  private:
    struct Entry {
      Board      board;
      Evaluation output;
      bool       valid = false;
    };
    Evaluator         &backend_;
    std::vector<Entry> entries_;
  };

  class SharedEvalCache {
  public:
    explicit SharedEvalCache(std::size_t bytes);
    [[nodiscard]] std::size_t  bytes() const noexcept;
    bool                       lookup(const Board &canon, Evaluation &out) const;
    void                       insert(const Board &canon, const Evaluation &out);
    std::atomic<std::uint64_t> hits{0}, misses{0};

  private:
    static constexpr std::size_t kShards = 64;
    struct Entry {
      Board      board;
      Evaluation output;
      bool       valid = false;
    };
    std::vector<Entry>                      entries_;
    mutable std::array<std::mutex, kShards> mutexes_;
  };

  class SharedCachedEvaluator final : public Evaluator {
  public:
    SharedCachedEvaluator(Evaluator &backend, SharedEvalCache &cache) : backend_(backend), cache_(cache) {}
    void evaluate(std::span<const Board>, std::span<Evaluation>, std::stop_token) override;

  private:
    Evaluator       &backend_;
    SharedEvalCache &cache_;
  };

  [[nodiscard]] std::uint64_t             selfplay_seed(std::uint64_t master, std::uint64_t game_id) noexcept;
  [[nodiscard]] std::vector<ReplayRecord> selfplay_game(std::uint64_t id, Evaluator &, const SelfplayConfig &);
  void                                    validate_replay_game(std::span<const ReplayRecord>, const SelfplayConfig &);
  [[nodiscard]] float                     replay_wdl_target(const ReplayRecord &) noexcept;
  [[nodiscard]] float                     replay_margin_target(const ReplayRecord &) noexcept;
  [[nodiscard]] inline float replay_value_target(const ReplayRecord &r) noexcept { return replay_wdl_target(r); }
  void                       write_replay_record(std::ostream &, const ReplayRecord &);
  // Clean EOF returns false; partial/truncated records throw.
  bool               read_replay_record(std::istream &, ReplayRecord &);
  [[nodiscard]] bool selfplay_selftest();
} // namespace islay
#endif
