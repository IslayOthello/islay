// Tight no-TT exact solver for the last few empties.
#ifndef ISLAY_ENDGAME_HPP
#define ISLAY_ENDGAME_HPP

#include <cstdint>

#include "board.hpp"
#include "options.hpp"

namespace islay {

  // Eight empties regressed against the TT-backed general search.
  inline constexpr int kEndgameSolverMax = 7;

  // Mover-relative fail-soft score; adds visited nodes to `nodes`.
  template<Rule R>
  [[nodiscard]] int endgame_solve(const Board &b, int alpha, int beta, int empties, std::uint64_t &nodes) noexcept;

} // namespace islay

#endif // ISLAY_ENDGAME_HPP
