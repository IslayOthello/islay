#ifndef ISLAY_ENDGAME_HPP
#define ISLAY_ENDGAME_HPP

#include "board.hpp"

namespace islay {

  inline constexpr int kEndgameMaxEmpties = 14;

  // Exact Win/Loss/Draw solver for small endgames (empties <= kEndgameMaxEmpties).
  // Returns +1 for Win, 0 for Draw, -1 for Loss from the perspective of board's mover.
  // Assumes Othello rules (pass if mover has no legal moves; terminal if both pass or board full).
  [[nodiscard]] int solve_exact_wld(const Board &board, int empties) noexcept;

  // Exact final disc margin from the perspective of board's mover.
  [[nodiscard]] int solve_exact_margin(const Board &board, int empties) noexcept;

  // Clear the calling thread's endgame transposition table.
  void clear_endgame_tt() noexcept;

  // Standalone self-test comparing solve_exact_wld against an unpruned minimax reference.
  [[nodiscard]] bool endgame_selftest();

} // namespace islay

#endif // ISLAY_ENDGAME_HPP
