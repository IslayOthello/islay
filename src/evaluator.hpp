#ifndef ISLAY_EVALUATOR_HPP
#define ISLAY_EVALUATOR_HPP

#include <array>
#include <span>
#include <stop_token>

#include "game.hpp"

namespace islay {

  struct Evaluation {
    std::array<float, kPolicySize> logits{};
    float                          value{}; // [-1, 1], from the input board's mover perspective
  };

  class Evaluator {
  public:
    virtual ~Evaluator() = default;
    // Equal-sized, non-overlapping spans. Fill every output or throw on failure.
    // Cancellation may leave outputs unspecified; the caller discards them.
    // The caller owns this evaluator exclusively for the duration of a search.
    virtual void evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token stop) = 0;
  };

} // namespace islay

#endif // ISLAY_EVALUATOR_HPP
