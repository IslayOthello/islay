#ifndef ISLAY_SEARCH_CONTROLLER_HPP
#define ISLAY_SEARCH_CONTROLLER_HPP

#include <condition_variable>
#include <mutex>
#include <ostream>
#include <thread>

#include "mcts.hpp"

namespace islay {

  // Commands have one owner; the worker owns its board, limits and evaluator.
  // All protocol writers share output_mutex. Never join while holding that mutex.
  class SearchController {
  public:
    SearchController(std::ostream &output, std::mutex &output_mutex) : output_(output), output_mutex_(output_mutex) {}
    ~SearchController();
    void start(Board board, Rule rule, MctsLimits limits, bool infinite);
    void stop(); // join, publishing exactly one result unless already published
    void cancel(); // join, suppressing any result not yet published

  private:
    std::ostream               &output_;
    std::mutex                 &output_mutex_;
    std::mutex                  wait_mutex_;
    std::condition_variable_any stopped_;
    bool                        publish_ = false; // output_mutex protects concurrent access
    std::jthread                worker_;
  };

  [[nodiscard]] bool search_controller_selftest();

} // namespace islay

#endif // ISLAY_SEARCH_CONTROLLER_HPP
