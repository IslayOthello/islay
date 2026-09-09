#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace islay {
  namespace {

    class TestEvaluator final : public Evaluator {
    public:
      Evaluation       output{};
      std::uint64_t    calls{};
      std::uint64_t    cancel_on{};
      std::stop_source cancel;

      void evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token) override {
        if (boards.size() != outputs.size())
          throw std::logic_error("mismatched evaluation batch");
        ++calls;
        if (calls == cancel_on)
          cancel.request_stop();
        for (std::size_t i = 0; i < boards.size(); ++i) {
          if (game_status(boards[i]).terminal())
            throw std::logic_error("terminal position sent to evaluator");
          outputs[i] = output;
        }
      }
    };

    bool close(double a, double b) { return std::abs(a - b) < 1e-6; }

    // Test oracle only: never used to train or evaluate production searches.
    int exact_outcome(const Board &board) {
      auto moves = board.moves();
      if (!moves) {
        if (board.passed().has_moves())
          return -exact_outcome(board.passed());
        const int margin = popcount(board.player) - popcount(board.opponent);
        return (margin > 0) - (margin < 0);
      }
      int best = -1;
      while (moves)
        best = std::max(best, -exact_outcome(board.play(pop_lsb(moves))));
      return best;
    }

    class ExactTestEvaluator final : public Evaluator {
    public:
      void evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token) override {
        for (std::size_t i = 0; i < boards.size(); ++i) {
          if (boards[i].count() < 61)
            throw std::logic_error("exact test oracle called outside small endgame");
          outputs[i]       = {};
          outputs[i].value = static_cast<float>(exact_outcome(boards[i]));
        }
      }
    };

    bool valid_result(const Board &board, const MctsResult &result, const MctsLimits &limits) {
      const auto status = game_status(board);
      if (result.tree_bytes > limits.tree_bytes || result.simulations > limits.simulations ||
          result.action_count != status.action_count() || !std::isfinite(result.value) || std::abs(result.value) > 1 ||
          result.elapsed_us < 0)
        return false;
      if (status.terminal())
        return result.reason == MctsStop::Terminal && result.best_move == NOMOVE && result.value == status.value &&
               !result.simulations && !result.evaluations;
      if (!status.legal(result.best_move))
        return false;
      std::uint64_t visits      = 0;
      double        probability = 0, sum = 0;
      for (int i = 0; i < result.action_count; ++i) {
        const auto &action = result.actions[i];
        if (!status.legal(action.action) || action.prior < 0 || !std::isfinite(action.prior) ||
            !std::isfinite(action.value) || std::abs(action.value) > 1 ||
            (i && action.action <= result.actions[i - 1].action))
          return false;
        visits += action.visits;
        probability += action.prior;
        sum += action.visits * action.value;
      }
      if (visits != result.simulations || !close(probability, 1) || (visits && !close(sum / visits, result.value)))
        return false;
      Board pv = board;
      for (int i = 0; i < result.pv_length; ++i) {
        if (!try_play(pv, result.pv[i], pv))
          return false;
      }
      return !result.pv_length || result.pv[0] == result.best_move;
    }

    bool same_tree(const MctsResult &a, const MctsResult &b) {
      if (a.best_move != b.best_move || a.value != b.value || a.simulations != b.simulations ||
          a.evaluations != b.evaluations || a.nodes != b.nodes || a.action_count != b.action_count ||
          a.pv_length != b.pv_length)
        return false;
      for (int i = 0; i < a.action_count; ++i) {
        if (a.actions[i].action != b.actions[i].action || a.actions[i].visits != b.actions[i].visits ||
            a.actions[i].prior != b.actions[i].prior || a.actions[i].value != b.actions[i].value)
          return false;
      }
      for (int i = 0; i < a.pv_length; ++i) {
        if (a.pv[i] != b.pv[i])
          return false;
      }
      return true;
    }
  } // namespace

  bool mcts_selftest() {
    try {
      MctsLimits limits;
      limits.simulations = 128;
      limits.tree_bytes  = 1024 * 1024;
      TestEvaluator evaluator;
      const Board   start    = Board::start();
      const auto    baseline = mcts_search(start, Rule::Othello, evaluator, limits);
      if (!valid_result(start, baseline, limits) || baseline.simulations != 128 ||
          baseline.reason != MctsStop::SimulationLimit || baseline.evaluations != 129 ||
          !same_tree(baseline, mcts_search(start, Rule::Othello, evaluator, limits)))
        return false;

      // With zero values/uniform priors, root exploration balances visits exactly.
      for (int i = 0; i < baseline.action_count; ++i) {
        if (baseline.actions[i].visits != 32 || baseline.actions[i].prior != 0.25f)
          return false;
      }

      RootExploration exploration;
      exploration.noise[19]    = 1;
      exploration.fraction     = 1;
      auto noisy_limits        = limits;
      noisy_limits.simulations = 0;
      noisy_limits.exploration = &exploration;
      const auto noisy         = mcts_search(start, Rule::Othello, evaluator, noisy_limits);
      for (int i = 0; i < noisy.action_count; ++i) {
        const auto &a = noisy.actions[i];
        if (noisy.network_priors[a.action] != 0.25f || a.prior != (a.action == 19 ? 1 : 0))
          return false;
      }
      exploration.fraction     = 0;
      noisy_limits.simulations = limits.simulations;
      if (!same_tree(baseline, mcts_search(start, Rule::Othello, evaluator, noisy_limits)))
        return false;
      for (int test = 0; test < 4; ++test) {
        exploration           = {};
        exploration.noise[19] = 1;
        if (test == 0)
          exploration.fraction = -1;
        if (test == 1)
          exploration.noise[0] = 1;
        if (test == 2)
          exploration.noise[19] = 0.5f;
        if (test == 3)
          exploration.noise[19] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try {
          (void) mcts_search(start, Rule::Othello, evaluator, noisy_limits);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        if (!rejected)
          return false;
      }

      for (const Board board: {Board{~Bitboard{0}, 0}, Board{0, ~Bitboard{0}}, Board{1, 0}, Board{0, 1},
                               Board{0xffffffffULL, 0xffffffff00000000ULL}, Board{0, 0}}) {
        const auto calls = evaluator.calls;
        if (!valid_result(board, mcts_search(board, Rule::Othello, evaluator, limits), limits) ||
            calls != evaluator.calls)
          return false;
      }

      // Exactly one legal move, followed by a terminal win/loss.
      for (const Board board: {Board{1, 2}, Board{1, ~Bitboard{5}}}) {
        const auto result = mcts_search(board, Rule::Othello, evaluator, limits);
        if (!valid_result(board, result, limits) || result.best_move != 2 || result.evaluations != 1 ||
            result.simulations != limits.simulations || result.value != (board.opponent == 2 ? 1 : -1))
          return false;
      }

      // PASS -> c1 -> terminal. Both traversed edges must invert the value.
      const Board forced{2, 1};
      evaluator.output.value = 0.5f;
      limits.simulations     = 1;
      auto result            = mcts_search(forced, Rule::Othello, evaluator, limits);
      if (!valid_result(forced, result, limits) || result.best_move != PASS || result.value != -0.5 ||
          result.actions[0].visits != 1 || result.actions[0].prior != 1)
        return false;
      limits.simulations = 2;
      result             = mcts_search(forced, Rule::Othello, evaluator, limits);
      if (!valid_result(forced, result, limits) || result.value != -0.75 || result.evaluations != 2 ||
          result.pv_length != 2 || result.pv[0] != PASS || result.pv[1] != 2)
        return false;
      limits.simulations = 1;
      result             = mcts_search(start, Rule::Othello, evaluator, limits);
      if (result.value != -0.5 || result.actions[0].visits != 1)
        return false;

      evaluator.output = {};
      evaluator.output.logits.fill(10000); // occupied/illegal actions must not influence softmax
      auto moves = start.moves();
      while (moves)
        evaluator.output.logits[pop_lsb(moves)] = -10000;
      evaluator.output.logits[37] = -9990;
      limits.simulations          = 0;
      result                      = mcts_search(start, Rule::Othello, evaluator, limits);
      if (!valid_result(start, result, limits) || result.best_move != 37 || result.simulations ||
          result.evaluations != 1 || result.actions[2].prior < 0.999f)
        return false;
      limits.simulations = 1;
      result             = mcts_search(start, Rule::Othello, evaluator, limits);
      if (result.actions[2].visits != 1)
        return false;

      evaluator.output   = {};
      limits.simulations = 1000;
      for (const std::size_t bytes: {std::size_t{0}, std::size_t{1}, std::size_t{4096}}) {
        limits.tree_bytes = bytes;
        result            = mcts_search(start, Rule::Othello, evaluator, limits);
        if (!valid_result(start, result, limits) || result.reason != MctsStop::MemoryLimit ||
            result.simulations == limits.simulations || (bytes < 2 && result.evaluations))
          return false;
      }
      limits.tree_bytes = 1024 * 1024;
      limits.deadline   = MctsClock::now();
      result            = mcts_search(start, Rule::Othello, evaluator, limits);
      if (!valid_result(start, result, limits) || result.reason != MctsStop::Deadline || result.evaluations)
        return false;
      limits.deadline = MctsClock::time_point::max();
      for (std::uint64_t cancel_on: {0, 1, 2, 5}) {
        TestEvaluator cancelled;
        cancelled.cancel_on = cancel_on;
        if (!cancel_on)
          cancelled.cancel.request_stop();
        result               = mcts_search(start, Rule::Othello, cancelled, limits, cancelled.cancel.get_token());
        const auto completed = cancel_on > 1 ? cancel_on - 2 : 0;
        if (!valid_result(start, result, limits) || result.reason != MctsStop::Cancelled ||
            result.simulations != completed || result.evaluations != cancel_on)
          return false;
      }

      for (int test = 0; test < 5; ++test) {
        evaluator.output = {};
        if (test == 0)
          evaluator.output.value = std::numeric_limits<float>::quiet_NaN();
        else if (test == 1)
          evaluator.output.value = 1.01f;
        else if (test == 2)
          evaluator.output.value = -1.01f;
        else
          evaluator.output.logits[test == 3 ? 0 : 19] = std::numeric_limits<float>::infinity();
        bool rejected = false;
        try {
          (void) mcts_search(start, Rule::Othello, evaluator, limits);
        } catch (const std::runtime_error &) {
          rejected = true;
        }
        if (!rejected)
          return false;
      }
      evaluator.output = {};
      for (int test = 0; test < 5; ++test) {
        auto invalid = limits;
        if (test == 2)
          invalid.c_puct = -1;
        if (test == 3)
          invalid.c_puct = std::numeric_limits<double>::quiet_NaN();
        if (test == 4)
          invalid.c_puct = std::numeric_limits<double>::infinity();
        bool rejected = false;
        try {
          (void) mcts_search(test == 1 ? Board{1, 1} : start, test == 0 ? Rule::Reversi : Rule::Othello, evaluator,
                             invalid);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        if (!rejected)
          return false;
      }

      // All phases and orientations, including late-game repeated terminal traversals.
      constexpr std::array<Board, 5> positions{{{240652910592ULL, 370411524ULL},
                                                {141013910561792ULL, 2314885659704166404ULL},
                                                {8024582863072280ULL, 2314986746359087110ULL},
                                                {80817503139037048ULL, 3486033126843646080ULL},
                                                {2455651727219126272ULL, 576743339727456007ULL}}};
      limits.simulations = 256;
      for (const auto &position: positions) {
        for (int s = 0; s < 8; ++s) {
          const auto board = position.symmetry(s);
          result           = mcts_search(board, Rule::Othello, evaluator, limits);
          if (!valid_result(board, result, limits) || result.simulations != limits.simulations ||
              !same_tree(result, mcts_search(board, Rule::Othello, evaluator, limits)))
            return false;
        }
      }
      ExactTestEvaluator exact;
      std::uint64_t      rng     = 0x9e3779b97f4a7c15ULL;
      int                checked = 0, draws = 0, mixed = 0;
      limits.simulations = 2048;
      for (int game = 0; game < 64; ++game) {
        Board board = start;
        while (board.count() < 61 && !game_status(board).terminal()) {
          auto moves = board.moves();
          if (!moves) {
            board = board.passed();
            continue;
          }
          rng ^= rng << 13;
          rng ^= rng >> 7;
          rng ^= rng << 17;
          auto skip = rng % popcount(moves);
          while (skip--)
            moves &= moves - 1;
          board = board.play(lsb(moves));
        }
        if (game_status(board).terminal())
          continue;
        const int expected = exact_outcome(board);
        result             = mcts_search(board, Rule::Othello, exact, limits);
        Board child;
        if (!valid_result(board, result, limits) || !try_play(board, result.best_move, child) ||
            -exact_outcome(child) != expected)
          return false;
        ++checked;
        draws += expected == 0;
        auto moves = board.moves();
        while (moves) {
          if (-exact_outcome(board.play(pop_lsb(moves))) != expected) {
            ++mixed;
            break;
          }
        }
      }
      if (checked < 32 || !draws || !mixed)
        return false;
      return true;
    } catch (...) {
      return false;
    }
  }

} // namespace islay
