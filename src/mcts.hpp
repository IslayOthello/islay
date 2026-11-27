#ifndef ISLAY_MCTS_HPP
#define ISLAY_MCTS_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "evaluator.hpp"
#include "options.hpp"

namespace islay {

  using MctsClock                  = std::chrono::steady_clock;
  inline constexpr int kMctsMaxPly = 128;

  struct RootExploration {
    std::array<float, kPolicySize> noise{}; // normalized over legal actions only
    double                         fraction = 0.25;
  };

  struct MctsResult;

  struct MctsLimits {
    std::uint64_t simulations = 800;
    std::size_t   tree_bytes  = 64 * 1024 * 1024; // arena only; excludes evaluator and fixed stack/result storage
    MctsClock::time_point                   deadline    = MctsClock::time_point::max();
    double                                  c_puct      = 1.5;
    const RootExploration                  *exploration = nullptr; // self-play only; valid for this synchronous call
    int                                     endgame_solver_empties = 14;
    int                                     batch_size             = 1;
    bool                                    q_aware_early_stop     = true;
    bool                                    tactical_bias          = true;
    double                                  fpu_reduction          = 0.2;
    std::function<void(const MctsResult &)> progress;
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
    std::array<float, kPolicySize>      network_priors{}; // legal softmax before root noise
    int                                 action_count{};
    std::array<Square, kMctsMaxPly>     pv{};
    int                                 pv_length{};
  };

  class MctsTree {
  public:
    MctsTree();
    ~MctsTree();
    MctsTree(MctsTree &&) noexcept;
    MctsTree &operator=(MctsTree &&) noexcept;

    void                        clear() noexcept;
    [[nodiscard]] bool          empty() const noexcept;
    [[nodiscard]] std::size_t   node_count() const noexcept;
    [[nodiscard]] std::uint64_t root_visits() const noexcept;
    [[nodiscard]] const Board  &root_board() const noexcept;
    [[nodiscard]] bool          can_reuse(const Board &target) const noexcept;

    struct Impl;
    [[nodiscard]] Impl       *impl() noexcept { return impl_.get(); }
    [[nodiscard]] const Impl *impl() const noexcept { return impl_.get(); }

  private:
    std::unique_ptr<Impl> impl_;
  };

  // Synchronous, single-owner PUCT; does not change UCI or use PerftTT.
  // Rejects Reversi/overlapping boards/invalid c_puct with invalid_argument.
  // Invalid evaluator outputs throw runtime_error; allocation/evaluator exceptions propagate.
  // Deadline/stop are checked between evaluations, not inside a blocking evaluator.
  // When tree is non-null and matches a subtree of board, re-roots and accumulates visits.
  [[nodiscard]] MctsResult mcts_search(const Board &board, Rule rule, Evaluator &evaluator,
                                       const MctsLimits &limits = {}, std::stop_token stop = {},
                                       MctsTree *tree = nullptr);
  [[nodiscard]] bool       mcts_selftest();

} // namespace islay

#endif // ISLAY_MCTS_HPP
