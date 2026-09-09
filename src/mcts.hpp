#ifndef ISLAY_MCTS_HPP
#define ISLAY_MCTS_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "evaluator.hpp"
#include "options.hpp"

namespace islay {

  using MctsClock                  = std::chrono::steady_clock;
  inline constexpr int kMctsMaxPly = 128;

  struct MctsLimits {
    std::uint64_t simulations      = 800;
    std::size_t   tree_bytes       = 64 * 1024 * 1024; // arena only; excludes evaluator and fixed stack/result storage
    MctsClock::time_point deadline = MctsClock::time_point::max();
    double                c_puct   = 1.5;
  };

  enum class MctsStop { SimulationLimit, Deadline, Cancelled, MemoryLimit, Terminal };

  struct RootAction {
    Square        action = NOMOVE;
    float         prior{};
    std::uint64_t visits{};
    double        value{}; // mean value from the root mover's perspective
  };

  struct MctsResult {
    Square                              best_move = NOMOVE;
    MctsStop                            reason    = MctsStop::SimulationLimit;
    std::uint64_t                       simulations{};
    std::uint64_t                       evaluations{};
    std::size_t                         nodes{};
    std::size_t                         tree_bytes{}; // allocated arena capacity, not just occupied nodes
    double                              value{};
    double                              elapsed_us{};
    std::array<RootAction, kPolicySize> actions{};
    int                                 action_count{};
    std::array<Square, kMctsMaxPly>     pv{};
    int                                 pv_length{};
  };

  // Synchronous, single-owner PUCT; does not change UCI or use PerftTT.
  // Rejects Reversi/overlapping boards/invalid c_puct with invalid_argument.
  // Invalid evaluator outputs throw runtime_error; allocation/evaluator exceptions propagate.
  // Deadline/stop are checked between evaluations, not inside a blocking evaluator.
  [[nodiscard]] MctsResult mcts_search(const Board &board, Rule rule, Evaluator &evaluator,
                                       const MctsLimits &limits = {}, std::stop_token stop = {});
  [[nodiscard]] bool       mcts_selftest();

} // namespace islay

#endif // ISLAY_MCTS_HPP
