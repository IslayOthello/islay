#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace islay {
  namespace {

    using Index = std::uint32_t;
    enum class State : std::uint8_t { Fresh, Expanded, Terminal };

    // Each child is also its incoming edge; siblings occupy one contiguous range.
    struct Node {
      std::uint64_t visits{};
      double        value_sum{}; // parent-relative except the root, which is root-relative
      float         prior{};
      Index         first_child{};
      std::uint8_t  action = NOMOVE;
      std::uint8_t  child_count{};
      std::int8_t   terminal_value{};
      State         state = State::Fresh;
    };

    class Search {
    public:
      Search(const Board &board, Evaluator &evaluator, const MctsLimits &limits, std::stop_token stop) :
          board_(board), evaluator_(evaluator), limits_(limits), stop_(stop) {}

      MctsResult run() {
        const auto start  = MctsClock::now();
        const auto status = game_status(board_);
        auto       moves  = status.moves;
        while (moves)
          result_.actions[result_.action_count++].action = pop_lsb(moves);
        if (status.forced_pass)
          result_.actions[result_.action_count++].action = PASS;
        if (status.terminal()) {
          result_.value  = status.value;
          result_.reason = MctsStop::Terminal;
        } else {
          for (int i = 0; i < result_.action_count; ++i)
            result_.actions[i].prior = 1.0f / result_.action_count;
          result_.best_move = result_.actions[0].action;
          if (!interrupted()) {
            const auto capacity = std::min(limits_.tree_bytes / sizeof(Node),
                                           static_cast<std::size_t>(std::numeric_limits<Index>::max()));
            if (capacity < 1 + static_cast<std::size_t>(status.action_count())) {
              result_.reason = MctsStop::MemoryLimit;
            } else {
              nodes_.reserve(capacity);
              result_.tree_bytes = nodes_.capacity() * sizeof(Node);
              if (result_.tree_bytes > limits_.tree_bytes)
                throw std::runtime_error("MCTS allocator exceeded arena budget");
              nodes_.emplace_back();
              float root_value = 0;
              if (expand(0, board_, status, root_value)) {
                result_.value = root_value;
                for (std::uint64_t i = 0; i < limits_.simulations; ++i) {
                  if (interrupted() || !simulate())
                    break;
                }
              }
              collect();
            }
          }
        }
        result_.elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - start).count();
        return result_;
      }

    private:
      Board             board_;
      Evaluator        &evaluator_;
      const MctsLimits &limits_;
      std::stop_token   stop_;
      std::vector<Node> nodes_;
      MctsResult        result_;

      bool interrupted() {
        if (stop_.stop_requested()) {
          result_.reason = MctsStop::Cancelled;
          return true;
        }
        if (limits_.deadline != MctsClock::time_point::max() && MctsClock::now() >= limits_.deadline) {
          result_.reason = MctsStop::Deadline;
          return true;
        }
        return false;
      }

      bool expand(Index index, const Board &board, const GameStatus &status, float &value) {
        if (status.terminal()) {
          nodes_[index].state          = State::Terminal;
          nodes_[index].terminal_value = static_cast<std::int8_t>(status.value);
          value                        = static_cast<float>(status.value);
          return true;
        }
        const auto count = static_cast<std::size_t>(status.action_count());
        if (count > nodes_.capacity() - nodes_.size()) {
          result_.reason = MctsStop::MemoryLimit;
          return false;
        }
        Evaluation output;
        evaluator_.evaluate(std::span(&board, 1), std::span(&output, 1), stop_);
        ++result_.evaluations;
        if (interrupted())
          return false;
        if (!std::isfinite(output.value) || output.value < -1 || output.value > 1 ||
            !std::all_of(output.logits.begin(), output.logits.end(), [](float x) { return std::isfinite(x); }))
          throw std::runtime_error("MCTS evaluator returned invalid logits/value");

        std::array<Square, kPolicySize> actions;
        int                             n     = 0;
        auto                            moves = status.moves;
        while (moves)
          actions[n++] = pop_lsb(moves);
        if (status.forced_pass)
          actions[n++] = PASS;
        float maximum = output.logits[actions[0]];
        for (int i = 1; i < n; ++i)
          maximum = std::max(maximum, output.logits[actions[i]]);
        std::array<double, kPolicySize> probabilities;
        double                          total = 0;
        for (int i = 0; i < n; ++i) {
          probabilities[i] = std::exp(static_cast<double>(output.logits[actions[i]]) - maximum);
          total += probabilities[i];
        }
        const auto first = static_cast<Index>(nodes_.size());
        for (int i = 0; i < n; ++i) {
          Node child;
          child.action = static_cast<std::uint8_t>(actions[i]);
          child.prior  = static_cast<float>(probabilities[i] / total);
          if (index == 0) {
            result_.network_priors[actions[i]] = child.prior;
            if (limits_.exploration) {
              const auto &exploration = *limits_.exploration;
              child.prior             = static_cast<float>((1 - exploration.fraction) * child.prior +
                                                           exploration.fraction * exploration.noise[actions[i]]);
            }
          }
          nodes_.push_back(child);
        }
        nodes_[index].first_child = first;
        nodes_[index].child_count = static_cast<std::uint8_t>(n);
        nodes_[index].state       = State::Expanded;
        value                     = output.value;
        return true;
      }

      Index select(Index parent) const {
        const auto  &node       = nodes_[parent];
        const double scale      = limits_.c_puct * std::sqrt(1.0 + static_cast<double>(node.visits));
        Index        best       = node.first_child;
        double       best_score = -std::numeric_limits<double>::infinity();
        for (Index i = node.first_child; i < node.first_child + node.child_count; ++i) {
          const auto  &child = nodes_[i];
          const double q     = child.visits ? child.value_sum / static_cast<double>(child.visits) : 0;
          const double score = q + scale * child.prior / (1.0 + static_cast<double>(child.visits));
          if (score > best_score) {
            best_score = score;
            best       = i;
          }
        }
        return best;
      }

      bool simulate() {
        std::array<Index, kMctsMaxPly + 1> path;
        int                                depth = 0;
        path[0]                                  = 0;
        Board board                              = board_;
        Index index                              = 0;
        while (nodes_[index].state == State::Expanded) {
          if (depth == kMctsMaxPly)
            throw std::logic_error("MCTS exceeded Othello path bound");
          index             = select(index);
          path[++depth]     = index;
          const auto action = nodes_[index].action;
          board             = action == PASS ? board.passed() : board.play(action);
        }
        float value;
        if (nodes_[index].state == State::Terminal) {
          value = nodes_[index].terminal_value;
        } else if (!expand(index, board, game_status(board), value)) {
          return false;
        }
        for (int i = depth; i > 0; --i) {
          value      = -value;
          auto &node = nodes_[path[i]];
          ++node.visits;
          node.value_sum += value;
        }
        ++nodes_[0].visits;
        nodes_[0].value_sum += value;
        return true;
      }

      Index most_visited(Index parent) const {
        const auto &node = nodes_[parent];
        Index       best = node.first_child;
        for (Index i = node.first_child + 1; i < node.first_child + node.child_count; ++i) {
          if (nodes_[i].visits > nodes_[best].visits ||
              (nodes_[i].visits == nodes_[best].visits && nodes_[i].prior > nodes_[best].prior))
            best = i;
        }
        return best;
      }

      void collect() {
        result_.nodes       = nodes_.size();
        result_.simulations = nodes_[0].visits;
        if (nodes_[0].visits)
          result_.value = nodes_[0].value_sum / static_cast<double>(nodes_[0].visits);
        if (nodes_[0].state != State::Expanded)
          return;
        for (int i = 0; i < result_.action_count; ++i) {
          const auto &child  = nodes_[nodes_[0].first_child + i];
          result_.actions[i] = {child.action, child.prior, child.visits,
                                child.visits ? child.value_sum / static_cast<double>(child.visits) : 0};
        }
        result_.best_move = nodes_[most_visited(0)].action;
        Index index       = 0;
        while (nodes_[index].state == State::Expanded && result_.pv_length < kMctsMaxPly) {
          index                           = most_visited(index);
          result_.pv[result_.pv_length++] = nodes_[index].action;
          if (!nodes_[index].visits)
            break;
        }
      }
    };
  } // namespace

  MctsResult mcts_search(const Board &board, Rule rule, Evaluator &evaluator, const MctsLimits &limits,
                         std::stop_token stop) {
    if (rule != Rule::Othello || (board.player & board.opponent) || !std::isfinite(limits.c_puct) ||
        limits.c_puct < 0 || limits.c_puct > 1e6)
      throw std::invalid_argument("MCTS requires Othello, disjoint discs and c_puct in [0, 1000000]");
    if (limits.exploration) {
      const auto &e      = *limits.exploration;
      const auto  status = game_status(board);
      double      total  = 0;
      if (!std::isfinite(e.fraction) || e.fraction < 0 || e.fraction > 1)
        throw std::invalid_argument("root noise fraction must be in [0,1]");
      for (int a = 0; a < kPolicySize; ++a) {
        if (!std::isfinite(e.noise[a]) || e.noise[a] < 0 || (!status.legal(a) && e.noise[a] != 0))
          throw std::invalid_argument("root noise must be finite and legal");
        total += e.noise[a];
      }
      if (!status.terminal() && std::abs(total - 1) > 1e-5)
        throw std::invalid_argument("root noise must sum to one");
    }
    return Search(board, evaluator, limits, stop).run();
  }

} // namespace islay
