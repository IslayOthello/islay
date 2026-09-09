#include "game.hpp"

#include <array>
#include <limits>

namespace islay {

  GameStatus game_status(const Board &board) noexcept {
    const Bitboard moves = board.moves();
    if (moves)
      return {moves, false, 0};
    if (board.passed().has_moves())
      return {0, true, 0};
    const int margin = popcount(board.player) - popcount(board.opponent);
    return {0, false, (margin > 0) - (margin < 0)};
  }

  bool try_play(const Board &board, Square action, Board &next) noexcept {
    if ((board.player & board.opponent) || !game_status(board).legal(action))
      return false;
    next = action == PASS ? board.passed() : board.play(action);
    return true;
  }

  Square symmetry_action(Square action, int symmetry) noexcept {
    if (symmetry < 0 || symmetry > 7 || action < 0 || action > PASS)
      return NOMOVE;
    return action == PASS ? PASS : lsb(Board::symmetry_bb(square_bb(action), symmetry));
  }

  namespace {
    // Independent square-scanning oracle, used only by the self-test.
    Bitboard reference_flips(const Board &board, Square action) {
      if ((board.player | board.opponent) & square_bb(action))
        return 0;
      Bitboard result = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy)
            continue;
          int      x = action % 8 + dx, y = action / 8 + dy;
          Bitboard ray = 0;
          while (x >= 0 && x < 8 && y >= 0 && y < 8) {
            const Bitboard bit = square_bb(y * 8 + x);
            if (!(board.opponent & bit)) {
              if (board.player & bit)
                result |= ray;
              break;
            }
            ray |= bit;
            x += dx;
            y += dy;
          }
        }
      }
      return result;
    }
  } // namespace

  bool game_selftest() {
    const Bitboard full = ~Bitboard{0};
    for (const Board board: {Board{full, 0}, Board{0, full}, Board{0xffffffffULL, 0xffffffff00000000ULL}, Board{1, 0},
                             Board{0, 1}, Board{0, 0}}) {
      const auto status = game_status(board);
      const int  margin = popcount(board.player) - popcount(board.opponent);
      Board      next   = Board::start();
      if (!status.terminal() || status.action_count() || status.value != (margin > 0) - (margin < 0) ||
          try_play(board, PASS, next) || next != Board::start())
        return false;
    }
    const Board forced{square_bb(1), square_bb(0)};
    Board       next;
    if (!game_status(forced).forced_pass || !try_play(forced, PASS, next) || next != forced.passed() ||
        !try_play(next, 2, next) || !game_status(next).terminal() || game_status(next).value != -1)
      return false;
    for (Square action: {-1, 0, 27, PASS, NOMOVE, std::numeric_limits<int>::max()}) {
      next = forced;
      if (try_play(Board::start(), action, next) || next != forced)
        return false;
    }
    if (try_play(Board{1, 1}, 2, next) || symmetry_action(-1, 0) != NOMOVE || symmetry_action(NOMOVE, 0) != NOMOVE ||
        symmetry_action(0, 8) != NOMOVE)
      return false;
    for (int s = 0; s < 8; ++s) {
      Bitboard seen = 0;
      for (Square action = 0; action < PASS; ++action)
        seen |= square_bb(symmetry_action(action, s));
      if (seen != full || symmetry_action(PASS, s) != PASS)
        return false;
    }

    std::uint64_t rng = 0x9e3779b97f4a7c15ULL;
    for (int game = 0; game < 16; ++game) {
      Board board = Board::start();
      for (int ply = 0; ply < 128; ++ply) {
        const auto status   = game_status(board);
        Bitboard   expected = 0;
        for (Square action = 0; action < PASS; ++action) {
          const auto flips = reference_flips(board, action);
          if (!flips)
            continue;
          expected |= square_bb(action);
          const Board child{board.opponent ^ flips, board.player ^ (flips | square_bb(action))};
          if (!try_play(board, action, next) || next != child)
            return false;
          for (int s = 0; s < 8; ++s) {
            if (!try_play(board.symmetry(s), symmetry_action(action, s), next) || next != child.symmetry(s))
              return false;
          }
        }
        if (status.moves != expected)
          return false;
        if (status.terminal())
          break;
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        Bitboard moves  = status.moves;
        Square   action = PASS;
        if (moves) {
          auto skip = rng % popcount(moves);
          while (skip--)
            moves &= moves - 1;
          action = lsb(moves);
        }
        if (!try_play(board, action, board) || ply == 127)
          return false;
      }
    }
    return true;
  }

} // namespace islay
