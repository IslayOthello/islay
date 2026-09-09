#ifndef ISLAY_SELFPLAY_HPP
#define ISLAY_SELFPLAY_HPP

#include <iosfwd>
#include <vector>

#include "mcts.hpp"

namespace islay {
  inline constexpr std::size_t kReplayRecordBytes = 560;

  struct SelfplayConfig {
    std::uint64_t seed              = 20260909;
    std::uint32_t simulations       = 128;
    std::size_t   tree_bytes        = 8 * 1024 * 1024;
    double        epsilon           = 0.25;
    double        alpha             = 0.3;
    std::uint16_t temperature_plies = 16; // tau=1 before this ply, then tau=0
  };

  struct ReplayRecord {
    std::uint64_t                          game_id{};
    Board                                  board;
    Bitboard                               legal{};
    std::array<std::uint32_t, kPolicySize> visits{};
    std::array<float, kPolicySize>         priors{};
    std::uint16_t                          ply{};
    std::uint8_t                           action{}, stm{}, temperature{}, pass{};
    std::int8_t                            outcome{};
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

  [[nodiscard]] std::uint64_t             selfplay_seed(std::uint64_t master, std::uint64_t game_id) noexcept;
  [[nodiscard]] std::vector<ReplayRecord> selfplay_game(std::uint64_t id, Evaluator &, const SelfplayConfig &);
  void                                    validate_replay_game(std::span<const ReplayRecord>, const SelfplayConfig &);
  void                                    write_replay_record(std::ostream &, const ReplayRecord &);
  // Clean EOF returns false; partial/truncated records throw.
  bool               read_replay_record(std::istream &, ReplayRecord &);
  [[nodiscard]] bool selfplay_selftest();
} // namespace islay
#endif
