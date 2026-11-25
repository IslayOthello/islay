#include "search_controller.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "selfplay.hpp"

namespace islay {
  namespace {
    // Explicit P2 scaffold; not a trained network or a strength evaluator.
    class UniformEvaluator final : public Evaluator {
    public:
      void evaluate(std::span<const Board>, std::span<Evaluation> outputs, std::stop_token) override {
        for (auto &output: outputs)
          output = {};
      }
    };

    const char *stop_name(MctsStop reason) {
      switch (reason) {
        case MctsStop::SimulationLimit:
          return "nodes";
        case MctsStop::Deadline:
          return "time";
        case MctsStop::Cancelled:
          return "stop";
        case MctsStop::MemoryLimit:
          return "memory";
        case MctsStop::Terminal:
          return "terminal";
      }
      return "unknown";
    }
  } // namespace

  SearchController::~SearchController() { cancel(); }

  void SearchController::clear_tree() noexcept {
    tree_.clear();
    const std::lock_guard lock(output_mutex_);
    has_last_result_ = false;
  }

  bool SearchController::last_result(const Board &board, MctsResult &result) const {
    const std::lock_guard lock(output_mutex_);
    if (!has_last_result_ || result_board_ != board)
      return false;
    result = last_result_;
    return true;
  }

  void SearchController::stop() {
    if (worker_.joinable()) {
      worker_.request_stop();
      worker_.join();
    }
  }

  void SearchController::cancel() {
    if (!worker_.joinable())
      return;
    {
      const std::lock_guard lock(output_mutex_);
      publish_ = false;
    }
    stop();
  }

  void SearchController::start(Board board, Rule rule, MctsLimits limits, bool infinite,
                               std::shared_ptr<Evaluator> evaluator) {
    cancel();
    if (rule != Rule::Othello)
      throw std::invalid_argument("MCTS supports Rule=Othello only");
    // Offline exploration is borrowed by synchronous self-play, never by the worker.
    if (limits.exploration)
      throw std::invalid_argument("root exploration is only supported by offline self-play");
    {
      const std::lock_guard lock(output_mutex_);
      publish_         = true;
      has_last_result_ = false;
    }
    if (evaluator) {
      if (cached_backend_ != evaluator || !eval_cache_) {
        eval_cache_     = std::make_unique<CachedEvaluator>(*evaluator, 32 * 1024 * 1024);
        cached_backend_ = evaluator;
      }
    } else {
      eval_cache_.reset();
      cached_backend_.reset();
    }
    CachedEvaluator *cache   = eval_cache_.get();
    const auto       started = MctsClock::now();
    limits.progress          = [this, board, started](const MctsResult &progress) {
      std::ostringstream message;
      const double       elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - started).count();
      message << "info nodes " << progress.simulations << " time " << static_cast<std::uint64_t>(elapsed_us / 1000)
              << "\ninfo string search progress Q " << std::showpos << std::fixed << std::setprecision(4)
              << progress.value << std::noshowpos << " (draw rate unavailable: scalar WDL head)";
      if (progress.action_count > 0) {
        std::vector<RootAction> actions(progress.actions.begin(), progress.actions.begin() + progress.action_count);
        std::sort(actions.begin(), actions.end(),
                  [](const RootAction &a, const RootAction &b) { return a.visits > b.visits; });
        std::uint64_t total_visits = 0;
        for (const auto &action: actions)
          total_visits += action.visits;
        message << "\ninfo string search policy:";
        for (std::size_t i = 0; i < std::min<std::size_t>(actions.size(), 6); ++i) {
          const auto  &action    = actions[i];
          const double visit_pct = total_visits ? 100.0 * action.visits / total_visits : 0.0;
          message << " " << square_to_string(action.action) << "=" << std::fixed << std::setprecision(1) << visit_pct
                  << "%/N" << action.visits << "/Q" << std::showpos << std::setprecision(3) << action.value
                  << std::noshowpos;
        }
      }
      message << '\n';
      const std::lock_guard lock(output_mutex_);
      if (publish_) {
        last_result_     = progress;
        result_board_    = board;
        has_last_result_ = true;
        output_ << message.str() << std::flush;
      }
    };
    worker_ = std::jthread([this, board, rule, limits, infinite, started, cache,
                            evaluator = std::move(evaluator)](std::stop_token stop) {
      MctsResult  result;
      std::string error;
      try {
        UniformEvaluator uniform;
        const bool       use_tree = (limits.simulations == std::numeric_limits<std::uint64_t>::max());
        if (cache) {
          result = mcts_search(board, rule, *cache, limits, stop, use_tree ? &tree_ : nullptr);
        } else {
          result = mcts_search(board, rule, uniform, limits, stop, use_tree ? &tree_ : nullptr);
        }
      } catch (const std::exception &failure) {
        error = failure.what();
      } catch (...) {
        error = "unknown search failure";
      }
      if (!error.empty()) {
        std::replace(error.begin(), error.end(), '\n', ' ');
        std::replace(error.begin(), error.end(), '\r', ' ');
        const auto status = game_status(board);
        result.best_move  = status.moves ? lsb(status.moves) : status.forced_pass ? PASS : NOMOVE;
      }
      // An infinite search may hit the arena cap or terminal before stop arrives.
      // Park without spinning and retain the result until stop (or cancellation).
      if (infinite) {
        std::unique_lock lock(wait_mutex_);
        stopped_.wait(lock, stop, [] { return false; });
      }
      const double       elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - started).count();
      std::ostringstream message;
      if (!error.empty())
        message << "info string search error: " << error << '\n';
      message << "info nodes " << result.simulations << " nps " << std::fixed << std::setprecision(0)
              << (elapsed_us > 0 ? result.simulations * 1e6 / elapsed_us : 0) << " time "
              << static_cast<std::uint64_t>(elapsed_us / 1000);
      if (result.pv_length) {
        message << " pv";
        for (int i = 0; i < result.pv_length; ++i)
          message << ' ' << square_to_string(result.pv[i]);
      }
      message << "\ninfo string search " << (error.empty() ? stop_name(result.reason) : "error") << " Q "
              << std::setprecision(6) << result.value << " (draw rate unavailable: scalar WDL head) evaluations "
              << result.evaluations;
      if (result.reason == MctsStop::Terminal)
        message << "\ninfo string exact WDL win/draw/loss "
                << (result.value > 0   ? "100/0/0"
                    : result.value < 0 ? "0/0/100"
                                       : "0/100/0")
                << "%";

      if (result.action_count > 1) {
        std::vector<RootAction> sorted_actions(result.actions.begin(), result.actions.begin() + result.action_count);
        std::sort(sorted_actions.begin(), sorted_actions.end(),
                  [](const RootAction &a, const RootAction &b) { return a.visits > b.visits; });
        std::uint64_t total_visits = 0;
        for (const auto &act: sorted_actions)
          total_visits += act.visits;
        message << "\ninfo string candidate moves:";
        const int show_count = std::min<int>(result.action_count, 6);
        for (int i = 0; i < show_count; ++i) {
          const auto  &act     = sorted_actions[i];
          const double pct     = total_visits > 0 ? (100.0 * act.visits / total_visits) : 0.0;
          const double prior_p = act.prior * 100.0;
          message << "\ninfo string   #" << (i + 1) << " " << std::setw(2) << square_to_string(act.action)
                  << ": N=" << std::setw(6) << act.visits << " (" << std::fixed << std::setprecision(1) << std::setw(4)
                  << pct << "%) Q=" << std::showpos << std::setprecision(3) << act.value << std::noshowpos
                  << " P=" << std::fixed << std::setprecision(1) << std::setw(4) << prior_p << "%";
        }
      }
      message << "\nbestmove " << (result.best_move == NOMOVE ? "0000" : square_to_string(result.best_move)) << '\n';
      const std::lock_guard lock(output_mutex_);
      if (error.empty()) {
        last_result_     = result;
        result_board_    = board;
        has_last_result_ = true;
      }
      if (publish_)
        output_ << message.str() << std::flush;
    });
  }

  bool search_controller_selftest() {
    std::ostringstream output;
    std::mutex         mutex;
    SearchController   controller(output, mutex);
    MctsLimits         limits;
    limits.simulations = 16;
    limits.tree_bytes  = 1024 * 1024;
    for (bool infinite: {false, true}) {
      controller.start(Board{1, 0}, Rule::Othello, limits, infinite);
      controller.stop();
      const auto first = output.str();
      if (first.find("bestmove 0000\n") == std::string::npos)
        return false;
      controller.stop();
      if (output.str() != first)
        return false;
      output.str("");
      controller.start(Board::start(), Rule::Othello, limits, true);
      controller.cancel();
      if (!output.str().empty())
        return false;
    }
    controller.start(Board{2, 1}, Rule::Othello, limits, true);
    controller.stop();
    if (output.str().find("bestmove pass\n") == std::string::npos)
      return false;
    RootExploration exploration;
    limits.exploration = &exploration;
    bool rejected      = false;
    try {
      controller.start(Board::start(), Rule::Othello, limits, false);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      return false;
    limits.exploration = nullptr;
    try {
      controller.start(Board::start(), Rule::Reversi, limits, false);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  }

} // namespace islay
