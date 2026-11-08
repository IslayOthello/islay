#include "endgame.hpp"

#include <algorithm>
#include <cstdint>

#include "bitboard.hpp"
#include "game.hpp"
#include "movegen.hpp"

namespace islay {
  namespace {

    ISLAY_FORCEINLINE int solve1(const Board &b, Bitboard moves) noexcept {
      const int    p = popcount(b.player), o = popcount(b.opponent);
      const Square e = static_cast<Square>(lsb(~(b.player | b.opponent)));
      if (moves != 0) {
        const int f      = popcount(flip(e, b.player, b.opponent));
        const int margin = p - o + 1 + 2 * f;
        return (margin > 0) - (margin < 0);
      }
      const Bitboard of = flip(e, b.opponent, b.player);
      if (of != 0) {
        const int margin = p - o - 1 - 2 * popcount(of);
        return (margin > 0) - (margin < 0);
      }
      const int margin = p - o;
      return (margin > 0) - (margin < 0);
    }

    int solve_wld(const Board &b, int alpha, int beta, int empties) noexcept;

    ISLAY_FORCEINLINE int solve2(const Board &b, Bitboard moves, int alpha, int beta) noexcept {
      if (!moves) {
        const Board passed = b.passed();
        if (passed.has_moves())
          return -solve_wld(passed, -beta, -alpha, 2);
        const int margin = popcount(b.player) - popcount(b.opponent);
        return (margin > 0) - (margin < 0);
      }
      Bitboard m = moves;
      while (m) {
        const Square sq    = pop_lsb(m);
        const Board  nb    = b.play(sq);
        const int    score = -solve1(nb, nb.moves());
        if (score >= beta)
          return score;
        if (score > alpha)
          alpha = score;
      }
      return alpha;
    }

    int solve_wld(const Board &b, int alpha, int beta, int empties) noexcept {
      if (empties == 1)
        return solve1(b, b.moves());
      if (empties == 2)
        return solve2(b, b.moves(), alpha, beta);

      const Bitboard moves = b.moves();
      if (!moves) {
        const Board passed = b.passed();
        if (passed.has_moves())
          return -solve_wld(passed, -beta, -alpha, empties);
        const int margin = popcount(b.player) - popcount(b.opponent);
        return (margin > 0) - (margin < 0);
      }

      // Corners first for quick cutoffs
      Bitboard m       = moves;
      Bitboard corners = m & kCorners;
      if (corners) {
        Bitboard c = corners;
        while (c) {
          const Square sq    = pop_lsb(c);
          const int    score = -solve_wld(b.play(sq), -beta, -alpha, empties - 1);
          if (score >= beta)
            return score;
          if (score > alpha)
            alpha = score;
        }
        m &= ~kCorners;
      }

      if (!m)
        return alpha;

      // Fast path for small empties or 1 remaining move:
      if (empties <= 4 || !(m & (m - 1))) {
        while (m) {
          const Square sq    = pop_lsb(m);
          const int    score = -solve_wld(b.play(sq), -beta, -alpha, empties - 1);
          if (score >= beta)
            return score;
          if (score > alpha)
            alpha = score;
        }
        return alpha;
      }

      // Min-mobility heuristic: sort moves by opponent mobility ascending
      struct MoveScore {
        Square sq;
        int    opp_mobility;
      };
      std::array<MoveScore, 32> candidates;
      int                       count = 0;
      Bitboard                  rem   = m;
      while (rem) {
        const Square sq      = pop_lsb(rem);
        const Board  nb      = b.play(sq);
        const int    opp_mob = popcount(nb.moves());
        if (opp_mob == 0) {
          const int score = -solve_wld(nb, -beta, -alpha, empties - 1);
          if (score >= beta)
            return score;
          if (score > alpha)
            alpha = score;
        } else {
          candidates[count++] = {sq, opp_mob};
        }
      }

      for (int i = 1; i < count; ++i) {
        MoveScore key = candidates[i];
        int       j   = i - 1;
        while (j >= 0 && candidates[j].opp_mobility > key.opp_mobility) {
          candidates[j + 1] = candidates[j];
          --j;
        }
        candidates[j + 1] = key;
      }

      for (int i = 0; i < count; ++i) {
        const int score = -solve_wld(b.play(candidates[i].sq), -beta, -alpha, empties - 1);
        if (score >= beta)
          return score;
        if (score > alpha)
          alpha = score;
      }
      return alpha;
    }

    // Unpruned minimax reference oracle used solely for self-testing.
    int reference_exact_outcome(const Board &board) {
      auto moves = board.moves();
      if (!moves) {
        if (board.passed().has_moves())
          return -reference_exact_outcome(board.passed());
        const int margin = popcount(board.player) - popcount(board.opponent);
        return (margin > 0) - (margin < 0);
      }
      int best = -1;
      while (moves)
        best = std::max(best, -reference_exact_outcome(board.play(pop_lsb(moves))));
      return best;
    }

  } // namespace

  int solve_exact_wld(const Board &board, int empties) noexcept {
    if (empties <= 0) {
      const int margin = popcount(board.player) - popcount(board.opponent);
      return (margin > 0) - (margin < 0);
    }
    return solve_wld(board, -1, 1, empties);
  }

  bool endgame_selftest() {
    // 1. Edge case: full board
    const Board full{0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL};
    if (solve_exact_wld(full, 0) != 0)
      return false;

    // 2. Play out deterministic pseudo-random games to 1..7 empties and verify against reference
    std::uint64_t rng     = 20260910;
    int           checked = 0, wins = 0, draws = 0, losses = 0;

    for (int trial = 0; trial < 200; ++trial) {
      Board b = Board::start();
      rng ^= rng << 13;
      rng ^= rng >> 7;
      rng ^= rng << 17;
      const int target_empties = 1 + static_cast<int>(rng % 7); // 1..7

      while (64 - b.count() > target_empties) {
        Bitboard moves = b.moves();
        if (!moves) {
          if (!b.passed().has_moves())
            break;
          b = b.passed();
          continue;
        }
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        int    count = popcount(moves);
        int    pick  = static_cast<int>(rng % static_cast<std::uint64_t>(count));
        Square sq    = 0;
        for (int i = 0; i <= pick; ++i)
          sq = pop_lsb(moves);
        b = b.play(sq);
      }

      const int empties = 64 - b.count();
      if (empties <= kEndgameMaxEmpties) {
        const int actual   = solve_exact_wld(b, empties);
        const int expected = reference_exact_outcome(b);
        if (actual != expected)
          return false;
        ++checked;
        wins += (actual == 1);
        draws += (actual == 0);
        losses += (actual == -1);
      }
    }

    // Ensure we covered all 3 outcomes across diverse positions
    if (checked < 100 || !wins || !draws || !losses)
      return false;

    return true;
  }

} // namespace islay
