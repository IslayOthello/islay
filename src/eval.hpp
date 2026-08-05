// Mover-relative centi-discs. Heuristic scores stay inside proven terminal scores.
#ifndef ISLAY_EVAL_HPP
#define ISLAY_EVAL_HPP

#include "bitboard.hpp"
#include "board.hpp"
#include "common.hpp"

namespace islay {

  inline constexpr int kInf     = 30000; // beyond any real score
  inline constexpr int kScoreMax = 6400; // |terminal_score| upper bound (64 discs)
  inline constexpr int kEvalMax  = 6399; // eval clamp: strictly inside kScoreMax

  [[nodiscard]] int terminal_score(const Board &b) noexcept;

  // `moves` must be nonzero and belong to `b`.
  [[nodiscard]] int eval(const Board &b, Bitboard moves) noexcept;

  [[nodiscard]] bool eval_selftest() noexcept;

} // namespace islay

#endif // ISLAY_EVAL_HPP
