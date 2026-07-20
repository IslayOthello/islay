/**
 * @file board.hpp
 * @brief The Board value type: state, moves, symmetry and text I/O.
 *
 * Board keeps two bitboards, always from the mover's point of view ("player"
 * to move, "opponent" waiting). It stays an aggregate -- `Board{p, o}` and
 * `Board{}` both work -- while offering member functions for the common
 * operations. The hot ones are header-inlined so the perft core pays no call
 * overhead.
 */
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

    /** Standard cross opening, Black (as `player`) to move. */
    [[nodiscard]] static Board start() noexcept {
      //     a b c d e f g h
      //   4       O X            d4=27 e5=36 white
      //   5       X O            e4=28 d5=35 black
      return Board{square_bb(28) | square_bb(35), square_bb(27) | square_bb(36)};
    }

    /** Bitboard of the mover's legal moves. */
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard moves() const noexcept { return get_moves(player, opponent); }

    /** True iff the mover has a legal move. */
    [[nodiscard]] ISLAY_FORCEINLINE bool has_moves() const noexcept { return can_move(player, opponent); }

    /** Discs on the board. */
    [[nodiscard]] ISLAY_FORCEINLINE int count() const noexcept { return popcount(player | opponent); }

    /** Result of playing legal move `sq`; sides swap so it stays mover-relative. */
    [[nodiscard]] ISLAY_FORCEINLINE Board play(Square sq) const noexcept {
      const Bitboard f = flip(sq, player, opponent);
      return Board{opponent ^ f, player ^ (f | square_bb(sq))};
    }

    /** Pass: swap sides without placing a disc. */
    [[nodiscard]] ISLAY_FORCEINLINE Board passed() const noexcept { return Board{opponent, player}; }

    /** Apply symmetry `s` (0..7) of the dihedral group D4 to one bitboard. */
    [[nodiscard]] static ISLAY_FORCEINLINE Bitboard symmetry_bb(Bitboard b, int s) noexcept {
      if (s & 1)
        b = flip_horizontal(b);
      if (s & 2)
        b = flip_vertical(b);
      if (s & 4)
        b = transpose(b);
      return b;
    }

    /** Apply symmetry `s` (0..7) of the dihedral group D4 to both bitboards. */
    [[nodiscard]] ISLAY_FORCEINLINE Board symmetry(int s) const noexcept {
      return Board{symmetry_bb(player, s), symmetry_bb(opponent, s)};
    }

    /**
     * Lexicographically smallest of the 8 symmetries -- a canonical key.
     * perft is symmetry-invariant, so all rotations/mirrors share this key.
     *
     * Two "optimizations" were tried here and BOTH measured as no better, on
     * interleaved A/B of cached perft(13):
     *  - building the 8 images as a chain (7 transforms instead of 12) and only
     *    transforming `opponent` for images tying on `player` (24 -> ~10
     *    transforms): 3298ms vs 3251ms -- a wash.
     *  - ISLAY_UNROLL(7) on the loop below: median 3576ms vs 3469ms -- worse.
     * This path is bound by TT memory traffic, not by work done here, and -O3
     * already unrolls a 7-iteration constant loop. Kept simple on purpose;
     * don't "optimize" it without measuring end-to-end first.
     */
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

    /**
     * Set from a 64-char diagram (square order a1,b1,..,h1,a2,..,h8) and a
     * side-to-move flag. Glyphs: 'X' 'x' '*' black, 'O' 'o' '0' white, '-' '.' '_' empty;
     * `stm`: X/B black, O/W white. Whitespace ignored. False on bad input.
     */
    [[nodiscard]] bool set(const std::string &diagram, char stm) noexcept;

    /** 64-char diagram of the board given the side to move. */
    [[nodiscard]] std::string to_string(Color stm) const;

    /** Pretty 8x8 board with coordinates; the mover's legal moves marked '*'. */
    void print(Color stm, std::ostream &os) const;
  };

  /** Parse "e6"/"E6" to a square (0..63); "pass"/"--" -> PASS; else NOMOVE. */
  [[nodiscard]] Square parse_square(const std::string &s) noexcept;

  /** Algebraic name of a square: 0 -> "a1", ..., 63 -> "h8"; PASS -> "pass". */
  [[nodiscard]] std::string square_to_string(Square sq);

} // namespace islay

#endif // ISLAY_BOARD_HPP
