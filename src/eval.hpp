/**
 * @file eval.hpp
 * @brief Static evaluation: final scores and a hand-written positional heuristic.
 *
 * Units are centi-discs, always from the MOVER's point of view (Board is
 * mover-relative, so no colour conversion is ever needed -- `score cp` can be
 * printed straight out).
 *
 * One scale, deliberately: `terminal_score` is the real final disc differential
 * (+/-6400) and `eval` is an estimate of that same quantity, clamped strictly
 * inside it (+/-6399). The clamp is load-bearing -- without it a heuristic guess
 * could outrank a proven win, which matters constantly under Rule::Reversi where
 * games end early and terminal scores mix with heuristic ones at shallow depth.
 *
 * THE WEIGHTS BELOW ARE HAND-GUESSED. There is no tuning data and no fitting
 * pipeline; they are a starting point, not measured values. Do not cite them.
 */
#ifndef ISLAY_EVAL_HPP
#define ISLAY_EVAL_HPP

#include "bitboard.hpp"
#include "board.hpp"
#include "common.hpp"

namespace islay {

  inline constexpr int kInf     = 30000; // beyond any real score
  inline constexpr int kScoreMax = 6400; // |terminal_score| upper bound (64 discs)
  inline constexpr int kEvalMax  = 6399; // eval clamp: strictly inside kScoreMax

  /**
   * Final score of a finished game, mover-relative, in centi-discs.
   * Empty squares are awarded to the winner (the tournament rule). That only
   * changes the margin, never the sign, so it cannot flip a win into a loss.
   */
  [[nodiscard]] int terminal_score(const Board &b) noexcept;

  /**
   * Positional estimate, mover-relative, centi-discs, clamped to +/-kEvalMax.
   *
   * `moves` must be `b.moves()` already computed by the caller (the search has
   * it in hand), and must be NON-ZERO: eval is never valid on a finished or
   * must-pass position -- those are terminal_score's job.
   */
  [[nodiscard]] int eval(const Board &b, Bitboard moves) noexcept;

  /** Cross-check eval's invariants (symmetry, antisymmetry, clamp) on a sweep. */
  [[nodiscard]] bool eval_selftest() noexcept;

} // namespace islay

#endif // ISLAY_EVAL_HPP
