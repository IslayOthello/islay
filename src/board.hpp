// Board is always mover-relative.
#ifndef ISLAY_BOARD_HPP
#define ISLAY_BOARD_HPP

#include <ostream>
#include <string>

#include "bitboard.hpp"
#include "common.hpp"
#include "movegen.hpp"

namespace islay {

  class Board {
  public:
    Bitboard player{};
    Bitboard opponent{};

    [[nodiscard]] static Board start() noexcept {
      //     a b c d e f g h
      //   4       O X            d4=27 e5=36 white
      //   5       X O            e4=28 d5=35 black
      return Board{square_bb(28) | square_bb(35), square_bb(27) | square_bb(36)};
    }

    [[nodiscard]] ISLAY_FORCEINLINE Bitboard moves() const noexcept { return get_moves(player, opponent); }

    [[nodiscard]] ISLAY_FORCEINLINE bool has_moves() const noexcept { return can_move(player, opponent); }

    [[nodiscard]] ISLAY_FORCEINLINE int count() const noexcept { return popcount(player | opponent); }

    [[nodiscard]] ISLAY_FORCEINLINE Board play(Square sq) const noexcept {
      const Bitboard f = flip(sq, player, opponent);
      return Board{opponent ^ f, player ^ (f | square_bb(sq))};
    }

    [[nodiscard]] ISLAY_FORCEINLINE Board passed() const noexcept { return Board{opponent, player}; }

    [[nodiscard]] static ISLAY_FORCEINLINE Bitboard symmetry_bb(Bitboard b, int s) noexcept {
      if (s & 1)
        b = flip_horizontal(b);
      if (s & 2)
        b = flip_vertical(b);
      if (s & 4)
        b = transpose(b);
      return b;
    }

    [[nodiscard]] ISLAY_FORCEINLINE Board symmetry(int s) const noexcept {
      return Board{symmetry_bb(player, s), symmetry_bb(opponent, s)};
    }

    // Cached perft is TT-bound; transform chaining and forced unrolling regressed.
    [[nodiscard]] Board canonical() const noexcept {
      Board best = *this;
      for (int s = 1; s < 8; ++s) {
        const Board c = symmetry(s);
        if (c.player < best.player || (c.player == best.player && c.opponent < best.opponent)) {
          best = c;
        }
      }
      return best;
    }

    [[nodiscard]] bool operator==(const Board &) const noexcept = default;

    // Whitespace is ignored; false means malformed diagram or side-to-move.
    [[nodiscard]] bool set(const std::string &diagram, char stm) noexcept;

    [[nodiscard]] std::string to_string(Color stm) const;

    void print(Color stm, std::ostream &os) const;
  };

  [[nodiscard]] Square parse_square(const std::string &s) noexcept;

  [[nodiscard]] std::string square_to_string(Square sq);

} // namespace islay

#endif // ISLAY_BOARD_HPP
