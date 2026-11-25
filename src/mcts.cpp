#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "endgame.hpp"
#include "hash.hpp"

namespace islay {

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
    float         terminal_value{};
    State         state = State::Fresh;
  };

  // Transposition entry for sharing evaluations & search guidance across branches
  struct alignas(16) MctsTTEntry {
    std::uint64_t key         = 0; // hash_board(player, opponent)
    float         q_value     = 0; // state-mover's perspective Q value
    std::uint32_t visits      = 0; // visits seen on this board state
    std::uint8_t  best_action = NOMOVE;
    std::uint8_t  valid       = 0;
    std::uint16_t pad         = 0;
  };

  struct MctsTree::Impl {
    std::vector<Node>              nodes;
    std::vector<MctsTTEntry>       tt;
    Board                          root_board{};
    bool                           valid = false;
    std::array<float, kPolicySize> root_network_priors{};
  };

  namespace {

    Index find_subtree(const std::vector<Node> &nodes, Index cur_idx, const Board &cur_board, const Board &target,
                       int depth_limit, bool is_root = true) {
      if (!is_root && cur_board == target)
        return cur_idx;
      if (depth_limit <= 0 || cur_idx >= nodes.size() || nodes[cur_idx].state != State::Expanded)
        return std::numeric_limits<Index>::max();
      const auto &cur_node = nodes[cur_idx];
      for (Index i = cur_node.first_child; i < cur_node.first_child + cur_node.child_count; ++i) {
        if (i >= nodes.size())
          break;
        const auto  action     = nodes[i].action;
        const Board next_board = action == PASS ? cur_board.passed() : cur_board.play(action);
        if (next_board == target)
          return i;
        if (depth_limit > 1) {
          const Index sub = find_subtree(nodes, i, next_board, target, depth_limit - 1, false);
          if (sub != std::numeric_limits<Index>::max())
            return sub;
        }
      }
      return std::numeric_limits<Index>::max();
    }

    void re_root_tree(std::vector<Node> &nodes, Index root_idx) {
      if (root_idx == 0 || root_idx >= nodes.size())
        return;

      std::vector<Node> new_nodes;
      new_nodes.reserve(nodes.size());

      Node new_root      = nodes[root_idx];
      new_root.value_sum = -new_root.value_sum;
      new_root.action    = NOMOVE;
      new_nodes.push_back(new_root);

      std::vector<std::pair<Index, Index>> queue;
      queue.push_back({root_idx, 0});

      std::size_t head = 0;
      while (head < queue.size()) {
        const auto [old_idx, new_idx] = queue[head++];
        const auto &old_node          = nodes[old_idx];
        if (old_node.state == State::Expanded && old_node.child_count > 0 &&
            old_node.first_child + old_node.child_count <= nodes.size()) {
          const Index new_first          = static_cast<Index>(new_nodes.size());
          new_nodes[new_idx].first_child = new_first;
          for (Index c = 0; c < old_node.child_count; ++c) {
            const Index old_child_idx = old_node.first_child + c;
            const Index new_child_idx = static_cast<Index>(new_nodes.size());
            new_nodes.push_back(nodes[old_child_idx]);
            queue.push_back({old_child_idx, new_child_idx});
          }
        } else {
          new_nodes[new_idx].first_child = 0;
          new_nodes[new_idx].child_count = 0;
        }
      }
      nodes = std::move(new_nodes);
    }

  } // namespace

  MctsTree::MctsTree() : impl_(std::make_unique<Impl>()) {}
  MctsTree::~MctsTree()                               = default;
  MctsTree::MctsTree(MctsTree &&) noexcept            = default;
  MctsTree &MctsTree::operator=(MctsTree &&) noexcept = default;

  void MctsTree::clear() noexcept {
    if (impl_) {
      impl_->nodes.clear();
      impl_->tt.clear();
      impl_->valid = false;
    }
  }

  bool MctsTree::empty() const noexcept { return !impl_ || !impl_->valid || impl_->nodes.empty(); }

  std::size_t MctsTree::node_count() const noexcept { return (impl_ && impl_->valid) ? impl_->nodes.size() : 0; }

  std::uint64_t MctsTree::root_visits() const noexcept {
    return (impl_ && impl_->valid && !impl_->nodes.empty()) ? impl_->nodes[0].visits : 0;
  }

  const Board &MctsTree::root_board() const noexcept {
    static const Board kEmptyBoard{};
    return (impl_ && impl_->valid) ? impl_->root_board : kEmptyBoard;
  }

  bool MctsTree::can_reuse(const Board &target) const noexcept {
    if (empty())
      return false;
    constexpr Index kNone = std::numeric_limits<Index>::max();
    return find_subtree(impl_->nodes, 0, impl_->root_board, target, 4) != kNone;
  }

  namespace {

    constexpr double kCpuctBase = 19652.0;

    float margin_value(int margin) noexcept {
      return static_cast<float>(std::tanh(static_cast<double>(margin) / 15.0));
    }

    float network_value(const Evaluation &output) noexcept {
      return std::clamp(output.value + 0.1f * output.margin, -1.0f, 1.0f);
    }

    class Search {
    public:
      Search(const Board &board, Evaluator &evaluator, const MctsLimits &limits, std::stop_token stop,
             MctsTree *tree = nullptr) :
          board_(board), evaluator_(evaluator), limits_(limits), stop_(stop), tree_(tree) {}

      MctsResult run() {
        const auto start         = MctsClock::now();
        auto       last_progress = start;
        const auto status        = game_status(board_);
        auto       moves         = status.moves;
        while (moves)
          result_.actions[result_.action_count++].action = pop_lsb(moves);
        if (status.forced_pass)
          result_.actions[result_.action_count++].action = PASS;
        if (status.terminal()) {
          result_.value  = status.value;
          result_.reason = MctsStop::Terminal;
          if (tree_)
            tree_->clear();
        } else {
          for (int i = 0; i < result_.action_count; ++i)
            result_.actions[i].prior = 1.0f / result_.action_count;
          result_.best_move = result_.actions[0].action;
          if (!interrupted()) {
            const auto capacity = std::min(limits_.tree_bytes / sizeof(Node),
                                           static_cast<std::size_t>(std::numeric_limits<Index>::max()));
            if (capacity < 1 + static_cast<std::size_t>(status.action_count())) {
              result_.reason = MctsStop::MemoryLimit;
              if (tree_)
                tree_->clear();
            } else {
              init_tt();
              bool reused = false;
              if (tree_ && tree_->can_reuse(board_)) {
                constexpr Index kNone  = std::numeric_limits<Index>::max();
                const Index target_idx = find_subtree(tree_->impl()->nodes, 0, tree_->impl()->root_board, board_, 4);
                if (target_idx != kNone) {
                  re_root_tree(tree_->impl()->nodes, target_idx);
                  nodes_ = std::move(tree_->impl()->nodes);
                  if (nodes_.capacity() < capacity)
                    nodes_.reserve(capacity);
                  result_.tree_bytes = nodes_.capacity() * sizeof(Node);
                  reused             = true;
                }
              }

              if (!reused) {
                thread_local std::vector<Node> tls_nodes;
                nodes_ = std::move(tls_nodes);
                nodes_.clear();
                if (nodes_.capacity() < capacity)
                  nodes_.reserve(capacity);
                result_.tree_bytes = nodes_.capacity() * sizeof(Node);
                if (result_.tree_bytes > limits_.tree_bytes)
                  throw std::runtime_error("MCTS allocator exceeded arena budget");
                nodes_.emplace_back();
                float root_value = 0;
                if (!expand(0, board_, status, root_value)) {
                  collect();
                  if (tree_)
                    tree_->clear();
                  else {
                    thread_local std::vector<Node> tls_back;
                    tls_back = std::move(nodes_);
                  }
                  result_.elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - start).count();
                  return result_;
                }
                result_.value = root_value;
              } else {
                if (nodes_[0].state == State::Expanded && nodes_[0].child_count > 0) {
                  for (Index c = 0; c < nodes_[0].child_count; ++c) {
                    const auto &child                    = nodes_[nodes_[0].first_child + c];
                    result_.network_priors[child.action] = child.prior;
                  }
                  if (limits_.exploration) {
                    const auto &exploration = *limits_.exploration;
                    for (Index c = 0; c < nodes_[0].child_count; ++c) {
                      auto &child = nodes_[nodes_[0].first_child + c];
                      child.prior = static_cast<float>((1 - exploration.fraction) * child.prior +
                                                       exploration.fraction * exploration.noise[child.action]);
                    }
                  }
                  if (nodes_[0].visits)
                    result_.value = nodes_[0].value_sum / static_cast<double>(nodes_[0].visits);
                }
              }

              const bool is_timed_search    = (limits_.deadline != MctsClock::time_point::max() &&
                                               limits_.simulations == std::numeric_limits<std::uint64_t>::max());
              auto       check_early_cutoff = [&](std::uint64_t sim_count) -> bool {
                if (!is_timed_search || nodes_[0].child_count == 0)
                  return false;
                if (nodes_[0].child_count == 1 && sim_count >= 1) {
                  result_.reason = MctsStop::Deadline;
                  return true;
                }
                bool all_terminal = true;
                for (Index c = 0; c < nodes_[0].child_count; ++c) {
                  const auto &child = nodes_[nodes_[0].first_child + c];
                  if (child.state != State::Terminal || child.visits == 0) {
                    all_terminal = false;
                    break;
                  }
                }
                if (all_terminal) {
                  result_.reason = MctsStop::Deadline;
                  return true;
                }
                if (sim_count < 256 || (sim_count & 63) != 0)
                  return false;
                std::uint64_t n_best   = 0;
                std::uint64_t n_second = 0;
                double        q_best   = 0.0;
                double        q_second = 0.0;
                for (Index c = 0; c < nodes_[0].child_count; ++c) {
                  const auto  &child = nodes_[nodes_[0].first_child + c];
                  const auto   v     = child.visits;
                  const double q     = v ? child.value_sum / static_cast<double>(v) : 0.0;
                  if (v > n_best) {
                    n_second = n_best;
                    q_second = q_best;
                    n_best   = v;
                    q_best   = q;
                  } else if (v > n_second) {
                    n_second = v;
                    q_second = q;
                  }
                }
                const auto now          = MctsClock::now();
                const auto elapsed      = now - start;
                const auto total_budget = limits_.deadline - start;
                return n_best >= 4 * n_second && elapsed * 3 >= total_budget &&
                       (!limits_.q_aware_early_stop || q_best - q_second >= 0.05);
              };
              const auto report_progress = [&] {
                if (!limits_.progress || MctsClock::now() - last_progress < std::chrono::milliseconds(500))
                  return;
                collect();
                result_.elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - start).count();
                try {
                  limits_.progress(result_);
                } catch (...) {
                }
                last_progress = MctsClock::now();
              };

              if (limits_.batch_size > 1) {
                for (std::uint64_t i = 0; i < limits_.simulations; i += limits_.batch_size) {
                  const int cur_batch = static_cast<int>(
                          std::min(static_cast<std::uint64_t>(limits_.batch_size), limits_.simulations - i));
                  if (interrupted() || !simulate_batch(cur_batch))
                    break;
                  report_progress();
                  if (check_early_cutoff(i + cur_batch))
                    break;
                }
              } else {
                for (std::uint64_t i = 0; i < limits_.simulations; ++i) {
                  if (interrupted() || !simulate())
                    break;
                  if ((i & 127) == 127)
                    report_progress();
                  if (check_early_cutoff(i + 1))
                    break;
                }
              }
              collect();

              if (tree_) {
                tree_->impl()->nodes               = std::move(nodes_);
                tree_->impl()->root_board          = board_;
                tree_->impl()->valid               = true;
                tree_->impl()->root_network_priors = result_.network_priors;
              } else {
                thread_local std::vector<Node> tls_back;
                tls_back = std::move(nodes_);
              }
            }
          }
        }
        result_.elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - start).count();
        return result_;
      }

    private:
      Board                     board_;
      Evaluator                &evaluator_;
      const MctsLimits         &limits_;
      std::stop_token           stop_;
      MctsTree                 *tree_ = nullptr;
      std::vector<Node>         nodes_;
      MctsResult                result_;
      std::vector<MctsTTEntry> *tt_      = nullptr;
      std::size_t               tt_mask_ = 0;

      void init_tt() {
        if (tree_) {
          tt_ = &tree_->impl()->tt;
        } else {
          thread_local std::vector<MctsTTEntry> tls_tt;
          tt_ = &tls_tt;
        }
        const std::size_t mib   = std::max<std::size_t>(1, limits_.tree_bytes / (1024 * 1024));
        const std::size_t slots = tt_slots_for(mib, sizeof(MctsTTEntry));
        if (tt_->size() != slots) {
          tt_->assign(slots, MctsTTEntry{});
        } else if (!tree_) {
          std::fill(tt_->begin(), tt_->end(), MctsTTEntry{});
        }
        tt_mask_ = slots - 1;
      }

      const MctsTTEntry *probe_tt(const Board &b) const noexcept {
        if (!tt_ || tt_->empty())
          return nullptr;
        const std::uint64_t key = hash_board(b.player, b.opponent);
        const auto         &e   = (*tt_)[key & tt_mask_];
        if (e.valid && e.key == key)
          return &e;
        return nullptr;
      }

      void store_tt(const Board &b, float q_val, std::uint32_t visits, std::uint8_t best_act) noexcept {
        if (!tt_ || tt_->empty())
          return;
        const std::uint64_t key = hash_board(b.player, b.opponent);
        auto               &e   = (*tt_)[key & tt_mask_];
        if (!e.valid || e.key == key || visits >= e.visits) {
          e.key         = key;
          e.q_value     = q_val;
          e.visits      = visits;
          e.best_action = best_act;
          e.valid       = 1;
        }
      }

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
        const int empties = 64 - board.count();
        if (index != 0 && empties <= limits_.endgame_solver_empties) {
          const int margin             = solve_exact_margin(board, empties);
          nodes_[index].state          = State::Terminal;
          nodes_[index].terminal_value = margin_value(margin);
          value                        = nodes_[index].terminal_value;
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
        if (!std::isfinite(output.value) || output.value < -1 || output.value > 1 || !std::isfinite(output.margin) ||
            output.margin < -1 || output.margin > 1 ||
            !std::all_of(output.logits.begin(), output.logits.end(), [](float x) { return std::isfinite(x); }))
          throw std::runtime_error("MCTS evaluator returned invalid logits/value");

        std::array<Square, kPolicySize> actions;
        int                             n     = 0;
        auto                            moves = status.moves;
        while (moves)
          actions[n++] = pop_lsb(moves);
        if (status.forced_pass)
          actions[n++] = PASS;

        if (limits_.tactical_bias && n > 1) {
          const Bitboard empty_bb = ~(board.player | board.opponent);
          for (int i = 0; i < n; ++i) {
            const Square   sq = actions[i];
            const Bitboard bb = square_bb(sq);
            if (bb & kCorners) {
              output.logits[sq] += 1.5f;
            } else if (bb & kXSquares) {
              const Square corner = (sq == 9 ? 0 : sq == 14 ? 7 : sq == 49 ? 56 : 63);
              if (square_bb(corner) & empty_bb)
                output.logits[sq] -= 2.0f;
            }
          }
        }

        // TT-guided prior initialization: if board state has a proven best action in TT, boost its logit
        const auto *tt_entry = probe_tt(board);
        if (tt_entry && tt_entry->best_action != NOMOVE && n > 1) {
          for (int i = 0; i < n; ++i) {
            if (actions[i] == tt_entry->best_action) {
              output.logits[actions[i]] += 0.8f;
              break;
            }
          }
        }

        float maximum = output.logits[actions[0]];
        for (int i = 1; i < n; ++i)
          maximum = std::max(maximum, output.logits[actions[i]]);
        std::array<double, kPolicySize> probabilities;
        double                          total = 0;
        for (int i = 0; i < n; ++i) {
          probabilities[i] = std::exp(static_cast<double>(output.logits[actions[i]]) - maximum);
          total += probabilities[i];
        }
        const double inv_total = 1.0 / total;
        const auto   first     = static_cast<Index>(nodes_.size());
        for (int i = 0; i < n; ++i) {
          Node child;
          child.action = static_cast<std::uint8_t>(actions[i]);
          child.prior  = static_cast<float>(probabilities[i] * inv_total);
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
        // Keep WDL as the primary search value; margin is a small calibration signal.
        value = network_value(output);
        return true;
      }

      Index select(Index parent) const {
        const auto  &node = nodes_[parent];
        const double cpuct =
                limits_.c_puct + std::log((static_cast<double>(node.visits) + kCpuctBase + 1.0) / kCpuctBase);
        const double scale      = cpuct * std::sqrt(1.0 + static_cast<double>(node.visits));
        const double parent_q   = node.visits > 0 ? (node.value_sum / static_cast<double>(node.visits)) : 0.0;
        const double fpu_parent = parent == 0 ? parent_q : -parent_q;
        Index        best       = node.first_child;
        double       best_score = -std::numeric_limits<double>::infinity();
        for (Index i = node.first_child; i < node.first_child + node.child_count; ++i) {
          const auto &child = nodes_[i];
          double      score;
          if (child.visits == 0) {
            const double fpu_q =
                    fpu_parent - limits_.fpu_reduction * std::sqrt(std::max(0.0, static_cast<double>(child.prior)));
            score = fpu_q + scale * child.prior;
          } else {
            const double inv_v1 = 1.0 / (1.0 + static_cast<double>(child.visits));
            const double q      = child.value_sum / static_cast<double>(child.visits);
            score               = q + scale * child.prior * inv_v1;
          }
          if (score > best_score) {
            best_score = score;
            best       = i;
          }
        }
        return best;
      }

      bool simulate() {
        std::array<Index, kMctsMaxPly + 1> path;
        std::array<Board, kMctsMaxPly + 1> boards;
        int                                depth = 0;
        path[0]                                  = 0;
        boards[0]                                = board_;
        Board board                              = board_;
        Index index                              = 0;
        while (nodes_[index].state == State::Expanded) {
          if (depth == kMctsMaxPly)
            throw std::logic_error("MCTS exceeded Othello path bound");
          index             = select(index);
          path[++depth]     = index;
          const auto action = nodes_[index].action;
          board             = action == PASS ? board.passed() : board.play(action);
          boards[depth]     = board;
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
          // When an expanded node has accumulated reliable visits, record into MCTS TT
          if (node.state == State::Expanded && node.visits >= 4 && node.child_count > 0) {
            const Index best_child = most_visited(path[i]);
            const float q_val      = static_cast<float>(node.value_sum / static_cast<double>(node.visits));
            store_tt(boards[i], q_val, static_cast<std::uint32_t>(node.visits), nodes_[best_child].action);
          }
        }
        ++nodes_[0].visits;
        nodes_[0].value_sum += value;
        if (nodes_[0].state == State::Expanded && nodes_[0].visits >= 4 && nodes_[0].child_count > 0) {
          const Index best_child = most_visited(0);
          const float q_val      = static_cast<float>(nodes_[0].value_sum / static_cast<double>(nodes_[0].visits));
          store_tt(board_, q_val, static_cast<std::uint32_t>(nodes_[0].visits), nodes_[best_child].action);
        }
        return true;
      }

      struct PendingSim {
        std::array<Index, kMctsMaxPly + 1> path;
        std::array<Board, kMctsMaxPly + 1> boards;
        int                                depth = 0;
        Board                              board;
        Index                              leaf_index  = 0;
        float                              value       = 0;
        bool                               is_terminal = false;
        bool                               needs_eval  = false;
        int                                eval_idx    = -1;
      };

      bool simulate_batch(int b_size) {
        std::array<PendingSim, 64> sims;
        int                        active_sims = 0;

        std::vector<Board> eval_boards;
        eval_boards.reserve(static_cast<std::size_t>(b_size));

        // 1. Selection with virtual loss
        for (int s = 0; s < b_size; ++s) {
          if (interrupted())
            break;
          auto &sim     = sims[active_sims++];
          sim.depth     = 0;
          sim.path[0]   = 0;
          sim.boards[0] = board_;
          Board board   = board_;
          Index index   = 0;

          while (nodes_[index].state == State::Expanded) {
            if (sim.depth == kMctsMaxPly)
              throw std::logic_error("MCTS exceeded Othello path bound");
            index                 = select(index);
            sim.path[++sim.depth] = index;
            const auto action     = nodes_[index].action;
            board                 = action == PASS ? board.passed() : board.play(action);
            sim.boards[sim.depth] = board;

            auto &node = nodes_[index];
            ++node.visits;
            node.value_sum -= 1.0;
          }

          sim.board      = board;
          sim.leaf_index = index;

          if (nodes_[index].state == State::Terminal) {
            sim.is_terminal = true;
            sim.value       = static_cast<float>(nodes_[index].terminal_value);
          } else {
            const int empties = 64 - board.count();
            if (index != 0 && empties <= limits_.endgame_solver_empties) {
              const int margin             = solve_exact_margin(board, empties);
              nodes_[index].state          = State::Terminal;
              nodes_[index].terminal_value = margin_value(margin);
              sim.is_terminal              = true;
              sim.value                    = nodes_[index].terminal_value;
            } else {
              const auto status = game_status(board);
              if (status.terminal()) {
                nodes_[index].state          = State::Terminal;
                nodes_[index].terminal_value = static_cast<std::int8_t>(status.value);
                sim.is_terminal              = true;
                sim.value                    = static_cast<float>(status.value);
              } else {
                int existing_eval = -1;
                for (int prev = 0; prev < active_sims - 1; ++prev) {
                  if (sims[prev].leaf_index == index && sims[prev].eval_idx >= 0) {
                    existing_eval = sims[prev].eval_idx;
                    break;
                  }
                }
                if (existing_eval >= 0) {
                  sim.eval_idx = existing_eval;
                } else {
                  const auto count = static_cast<std::size_t>(status.action_count());
                  if (count > nodes_.capacity() - nodes_.size()) {
                    result_.reason = MctsStop::MemoryLimit;
                    break;
                  }
                  sim.needs_eval = true;
                  sim.eval_idx   = static_cast<int>(eval_boards.size());
                  eval_boards.push_back(board);
                }
              }
            }
          }
        }

        if (active_sims == 0)
          return false;

        // 2. Batch Evaluation
        std::vector<Evaluation> eval_outputs(eval_boards.size());
        if (!eval_boards.empty()) {
          evaluator_.evaluate(eval_boards, eval_outputs, stop_);
          result_.evaluations += eval_boards.size();
          if (interrupted()) {
            for (int s = 0; s < active_sims; ++s) {
              for (int i = sims[s].depth; i > 0; --i) {
                auto &node = nodes_[sims[s].path[i]];
                --node.visits;
                node.value_sum += 1.0;
              }
            }
            return false;
          }
          for (std::size_t i = 0; i < eval_outputs.size(); ++i) {
            if (!std::isfinite(eval_outputs[i].value) || eval_outputs[i].value < -1 || eval_outputs[i].value > 1 ||
                !std::isfinite(eval_outputs[i].margin) || eval_outputs[i].margin < -1 || eval_outputs[i].margin > 1 ||
                !std::all_of(eval_outputs[i].logits.begin(), eval_outputs[i].logits.end(),
                             [](float x) { return std::isfinite(x); }))
              throw std::runtime_error("MCTS evaluator returned invalid logits/value in batch");
          }
        }

        // 3. Revert virtual loss, expand nodes and backpropagate true values
        for (int s = 0; s < active_sims; ++s) {
          auto &sim = sims[s];
          for (int i = sim.depth; i > 0; --i) {
            auto &node = nodes_[sim.path[i]];
            --node.visits;
            node.value_sum += 1.0;
          }

          if (sim.eval_idx >= 0) {
            const auto &out = eval_outputs[static_cast<std::size_t>(sim.eval_idx)];
            sim.value       = network_value(out);

            if (sim.needs_eval && nodes_[sim.leaf_index].state != State::Expanded) {
              const auto status = game_status(sim.board);
              const auto count  = static_cast<std::size_t>(status.action_count());
              if (count <= nodes_.capacity() - nodes_.size()) {
                std::array<Square, kPolicySize> actions;
                int                             n     = 0;
                auto                            moves = status.moves;
                while (moves)
                  actions[n++] = pop_lsb(moves);
                if (status.forced_pass)
                  actions[n++] = PASS;

                auto logits = out.logits;
                if (limits_.tactical_bias && n > 1) {
                  const Bitboard empty_bb = ~(sim.board.player | sim.board.opponent);
                  for (int i = 0; i < n; ++i) {
                    const Square   sq = actions[i];
                    const Bitboard bb = square_bb(sq);
                    if (bb & kCorners) {
                      logits[sq] += 1.5f;
                    } else if (bb & kXSquares) {
                      const Square corner = (sq == 9 ? 0 : sq == 14 ? 7 : sq == 49 ? 56 : 63);
                      if (square_bb(corner) & empty_bb)
                        logits[sq] -= 2.0f;
                    }
                  }
                }

                // TT-guided prior initialization: if board state has a proven best action in TT, boost its logit
                const auto *tt_entry = probe_tt(sim.board);
                if (tt_entry && tt_entry->best_action != NOMOVE && n > 1) {
                  for (int i = 0; i < n; ++i) {
                    if (actions[i] == tt_entry->best_action) {
                      logits[actions[i]] += 0.8f;
                      break;
                    }
                  }
                }

                float maximum = logits[actions[0]];
                for (int i = 1; i < n; ++i)
                  maximum = std::max(maximum, logits[actions[i]]);
                std::array<double, kPolicySize> probabilities;
                double                          total = 0;
                for (int i = 0; i < n; ++i) {
                  probabilities[i] = std::exp(static_cast<double>(logits[actions[i]]) - maximum);
                  total += probabilities[i];
                }
                const double inv_total = 1.0 / total;
                const auto   first     = static_cast<Index>(nodes_.size());
                for (int i = 0; i < n; ++i) {
                  Node child;
                  child.action = static_cast<std::uint8_t>(actions[i]);
                  child.prior  = static_cast<float>(probabilities[i] * inv_total);
                  nodes_.push_back(child);
                }
                nodes_[sim.leaf_index].first_child = first;
                nodes_[sim.leaf_index].child_count = static_cast<std::uint8_t>(n);
                nodes_[sim.leaf_index].state       = State::Expanded;
              }
            }
          }

          float val = sim.value;
          for (int i = sim.depth; i > 0; --i) {
            val        = -val;
            auto &node = nodes_[sim.path[i]];
            ++node.visits;
            node.value_sum += val;
            if (node.state == State::Expanded && node.visits >= 4 && node.child_count > 0) {
              const Index best_child = most_visited(sim.path[i]);
              const float q_val      = static_cast<float>(node.value_sum / static_cast<double>(node.visits));
              store_tt(sim.boards[i], q_val, static_cast<std::uint32_t>(node.visits), nodes_[best_child].action);
            }
          }
          ++nodes_[0].visits;
          nodes_[0].value_sum += val;
          if (nodes_[0].state == State::Expanded && nodes_[0].visits >= 4 && nodes_[0].child_count > 0) {
            const Index best_child = most_visited(0);
            const float q_val      = static_cast<float>(nodes_[0].value_sum / static_cast<double>(nodes_[0].visits));
            store_tt(board_, q_val, static_cast<std::uint32_t>(nodes_[0].visits), nodes_[best_child].action);
          }
        }

        if (result_.reason == MctsStop::MemoryLimit)
          return false;
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
        result_.pv_length   = 0;
        if (nodes_[0].visits)
          result_.value = nodes_[0].value_sum / static_cast<double>(nodes_[0].visits);
        if (nodes_[0].state != State::Expanded)
          return;
        for (int i = 0; i < result_.action_count; ++i) {
          const auto &child  = nodes_[nodes_[0].first_child + i];
          result_.actions[i] = {child.action, child.prior, child.visits,
                                child.visits ? child.value_sum / static_cast<double>(child.visits) : 0};
        }
        bool all_root_terminal = nodes_[0].child_count > 0;
        for (Index c = 0; c < nodes_[0].child_count; ++c) {
          const auto &child = nodes_[nodes_[0].first_child + c];
          if (child.state != State::Terminal || child.visits == 0) {
            all_root_terminal = false;
            break;
          }
        }
        auto best_root_child = most_visited(0);
        if (all_root_terminal) {
          best_root_child = nodes_[0].first_child;
          double best_q   = nodes_[best_root_child].value_sum / static_cast<double>(nodes_[best_root_child].visits);
          for (Index i = nodes_[0].first_child + 1; i < nodes_[0].first_child + nodes_[0].child_count; ++i) {
            const auto  &c = nodes_[i];
            const double q = c.value_sum / static_cast<double>(c.visits);
            if (q > best_q + 1e-9 || (std::abs(q - best_q) <= 1e-9 && (c.visits > nodes_[best_root_child].visits ||
                                                                       (c.visits == nodes_[best_root_child].visits &&
                                                                        c.prior > nodes_[best_root_child].prior)))) {
              best_root_child = i;
              best_q          = q;
            }
          }
        }
        result_.best_move               = nodes_[best_root_child].action;
        Index index                     = best_root_child;
        result_.pv[result_.pv_length++] = nodes_[index].action;
        while (nodes_[index].state == State::Expanded && result_.pv_length < kMctsMaxPly && nodes_[index].visits) {
          index                           = most_visited(index);
          result_.pv[result_.pv_length++] = nodes_[index].action;
          if (!nodes_[index].visits)
            break;
        }
      }
    };
  } // namespace

  MctsResult mcts_search(const Board &board, Rule rule, Evaluator &evaluator, const MctsLimits &limits,
                         std::stop_token stop, MctsTree *tree) {
    if (rule != Rule::Othello || (board.player & board.opponent) || !std::isfinite(limits.c_puct) ||
        limits.c_puct < 0 || limits.c_puct > 1e6 || limits.endgame_solver_empties < 0 ||
        limits.endgame_solver_empties > 64 || limits.batch_size < 1 || limits.batch_size > 64)
      throw std::invalid_argument("MCTS requires Othello, disjoint discs, valid endgame empties, batch_size in [1, 64] "
                                  "and c_puct in [0, 1000000]");
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
    return Search(board, evaluator, limits, stop, tree).run();
  }

} // namespace islay
