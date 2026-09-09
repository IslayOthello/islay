#include "search_controller.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
    publish_           = true;
    const auto started = MctsClock::now();
    worker_            = std::jthread(
            [this, board, rule, limits, infinite, started, evaluator = std::move(evaluator)](std::stop_token stop) {
              MctsResult  result;
              std::string error;
              try {
                UniformEvaluator uniform;
                result = mcts_search(board, rule, evaluator ? *evaluator : uniform, limits, stop);
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
              const double elapsed_us = std::chrono::duration<double, std::micro>(MctsClock::now() - started).count();
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
              message << "\ninfo string search " << (error.empty() ? stop_name(result.reason) : "error") << " value "
                      << std::setprecision(6) << result.value << " evaluations " << result.evaluations << "\nbestmove "
                      << (result.best_move == NOMOVE ? "0000" : square_to_string(result.best_move)) << '\n';
              const std::lock_guard lock(output_mutex_);
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
    try {
      controller.start(Board::start(), Rule::Reversi, limits, false);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  }

} // namespace islay
