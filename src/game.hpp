#ifndef ISLAY_GAME_HPP
#define ISLAY_GAME_HPP

#include "board.hpp"

namespace islay {

  inline constexpr int kPolicySize = 65;

  // Othello only. Inputs must have disjoint player/opponent bitboards.
  struct GameStatus {
    Bitboard moves{};
    bool     forced_pass{};
    int      value{}; // meaningful only at terminal, from the mover's perspective

    [[nodiscard]] bool terminal() const noexcept { return !moves && !forced_pass; }
    [[nodiscard]] int  action_count() const noexcept { return popcount(moves) + forced_pass; }
    [[nodiscard]] bool legal(Square action) const noexcept {
      return action == PASS ? forced_pass : action >= 0 && action < PASS && (moves & square_bb(action));
    }
  };

  [[nodiscard]] GameStatus game_status(const Board &board) noexcept;
  // Checked boundary; leaves next unchanged on failure, including overlapping discs.
  [[nodiscard]] bool try_play(const Board &board, Square action, Board &next) noexcept;
  // PASS is invariant; invalid action/symmetry returns NOMOVE.
  [[nodiscard]] Square symmetry_action(Square action, int symmetry) noexcept;
  [[nodiscard]] bool   game_selftest();

} // namespace islay

#endif // ISLAY_GAME_HPP
