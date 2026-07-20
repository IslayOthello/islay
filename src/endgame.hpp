/**
 * @file endgame.hpp
 * @brief Specialised exact solver for the last few empties. Self-designed (not ported
 *        from Edax); the general search hands off to it once the game is nearly full.
 *
 * WHY A SEPARATE SOLVER. Down in the endgame the general node machinery -- transposition
 * table, incremental pattern state, the ordering arrays, ProbCut -- costs more than the
 * tiny subtree it wraps. The last handful of plies want a TIGHT alpha-beta with almost
 * no per-node overhead: enumerate the empties, try each as a move, recurse, and settle
 * the very last square by counting flips instead of playing it out. This is where an
 * endgame's node count is decided.
 *
 * EXACT. Every path returns the true game value (fail-soft within the window). It is
 * pinned by `search_selftest`, which compares exact solves against a plain negamax
 * oracle -- so a bug here surfaces as a score mismatch, not silent bad play.
 *
 * WHAT IT ADDS over a plain recursion, all self-made and all exact:
 *   - a one-empty base case that counts flips directly (no board built, no recursion);
 *   - endgame PARITY move ordering (an odd-parity empty region hands its last move to
 *     whoever plays into it, so those moves go first -> more cutoffs);
 *   - the stability cutoff (if the opponent already holds enough unflippable discs the
 *     node's ceiling cannot reach alpha), cheaply gated so it only runs where it bites.
 */
#ifndef ISLAY_ENDGAME_HPP
#define ISLAY_ENDGAME_HPP

#include <cstdint>

#include "board.hpp"
#include "options.hpp"

namespace islay {

  /**
   * Empty count at or below which the general search hands off to this tight solver.
   * MEASURED at 7 (22-empty solve, 3 runs): max=6 ~9s (noisy), max=7 6.4s (stable, the
   * lowest node count of any setting), max=8 ~9.7s. The crossover is real, not noise:
   * the tight solver drops the transposition table, and at 8 empties transpositions are
   * dense enough that TT DEDUP (which the general search keeps) beats the tight solver's
   * cheaper-per-node work. So 7 is the deepest empty count where "no hash" still wins --
   * already deeper than Edax's ~4-empty tight solvers, which likewise hand off to a hash
   * above that. Pushing to 8 measurably SLOWS the solve; do not.
   */
  inline constexpr int kEndgameSolverMax = 7;

  /**
   * Exact score of `b` (mover-relative, centi-discs) with `empties` in 1..kEndgameSolverMax
   * empty squares, alpha-beta within [alpha, beta], fail-soft. Adds every node it visits
   * to `nodes` so the reported count stays honest.
   */
  template<Rule R>
  [[nodiscard]] int endgame_solve(const Board &b, int alpha, int beta, int empties, std::uint64_t &nodes) noexcept;

} // namespace islay

#endif // ISLAY_ENDGAME_HPP
