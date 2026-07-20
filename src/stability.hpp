/**
 * @file stability.hpp
 * @brief Provably-unflippable discs, computed by a fixpoint. Self-designed from the
 *        rules of Othello (NOT ported from Edax), header-only.
 *
 * WHY: the endgame solver's biggest missing pruner. If the opponent holds S discs
 * that can never be flipped, then whatever happens the opponent finishes with at
 * least S, so the mover's final margin is at most 64 - 2S (the empties cancel out of
 * the bound however they break). A solving node whose ceiling 100*(64 - 2S) cannot
 * reach alpha is dead. The earlier corner-anchored-edge version undercounted so badly
 * (it needed S >= ~31 to bite) that the cut never fired; this computes the real set.
 *
 * THE ALGORITHM. A disc can only be flipped by a move that brackets it along one of
 * the four axes (horizontal, vertical, and the two diagonals). So a disc of colour p
 * is unflippable iff it is safe along ALL FOUR axes, and it is safe along an axis if
 * ANY of these holds -- each is genuinely sufficient, so the result never OVER-counts:
 *
 *   1. the whole line through it on that axis is full (no empty square remains on the
 *      line, so no move can ever be played on it -> nothing to flip with);
 *   2. it sits on the board edge for that axis (one bracket end would be off the
 *      board, so it can never be bracketed along that axis);
 *   3. one of its two neighbours along that axis is a same-colour disc that is itself
 *      already known stable (that side is sealed by a permanent wall of p, so no
 *      opponent bracket can form on that side, and a bracket needs BOTH sides).
 *
 * Rule 3 is recursive, so the set is grown to a fixpoint: start from what rules 1-2
 * give (edges and full lines), then keep adding discs that rule 3 now admits until
 * nothing new appears. Monotone, so it converges in a handful of passes.
 *
 * SOUNDNESS is the one thing that matters (an over-count makes the cutoff prune a real
 * line and corrupts an exact solve). Every rule above is a sufficient condition for
 * true unflippability, so the count is a lower bound on the real stable set -- exactly
 * what a sound upper-bound cut needs. `search_selftest` is the backstop: it compares
 * exact solves against the plain oracle, so any over-count would surface as a score
 * mismatch there.
 */
#ifndef ISLAY_STABILITY_HPP
#define ISLAY_STABILITY_HPP

#include <array>

#include "bitboard.hpp"

namespace islay {
  namespace stability_detail {

    // 8-neighbour shifts, no wrap across a file edge. `sE(bb)` has a bit at x iff bb
    // had one at x's WEST neighbour -- i.e. "x has a stable west neighbour".
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sE(Bitboard b) noexcept { return (b << 1) & ~kFileA; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sW(Bitboard b) noexcept { return (b >> 1) & ~kFileH; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sN(Bitboard b) noexcept { return b >> 8; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sS(Bitboard b) noexcept { return b << 8; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sNE(Bitboard b) noexcept { return (b >> 7) & ~kFileA; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sSW(Bitboard b) noexcept { return (b << 7) & ~kFileH; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sNW(Bitboard b) noexcept { return (b >> 9) & ~kFileH; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sSE(Bitboard b) noexcept { return (b << 9) & ~kFileA; }

    // The 15 diagonals of each direction, generated from geometry (repo convention:
    // masks come from the board's shape, not a transcribed table).
    struct DiagMasks {
      std::array<Bitboard, 15> d1{}; // a1-h8 direction: constant (rank - file)
      std::array<Bitboard, 15> d2{}; // a8-h1 direction: constant (rank + file)
    };
    [[nodiscard]] constexpr DiagMasks make_diags() noexcept {
      DiagMasks m{};
      for (int r = 0; r < 8; ++r)
        for (int f = 0; f < 8; ++f) {
          const Bitboard bit = Bitboard{1} << (r * 8 + f);
          m.d1[static_cast<std::size_t>((r - f) + 7)] |= bit;
          m.d2[static_cast<std::size_t>(r + f)] |= bit;
        }
      return m;
    }
    inline constexpr DiagMasks kDiags = make_diags();

  } // namespace stability_detail

  /**
   * Count `p`'s provably-unflippable discs, given the opponent `o`. A sound LOWER
   * bound on the true stable set (never over-counts) -- see the header.
   */
  [[nodiscard]] inline int stable_count(Bitboard p, Bitboard o) noexcept {
    using namespace stability_detail;
    const Bitboard occ = p | o;

    // Rule 1: squares lying on a FULL line, per axis.
    Bitboard fh = 0, fv = 0, fd1 = 0, fd2 = 0;
    for (int r = 0; r < 8; ++r) {
      const Bitboard m = kRank1 << (r * 8);
      if ((occ & m) == m)
        fh |= m;
    }
    for (int f = 0; f < 8; ++f) {
      const Bitboard m = kFileA << f;
      if ((occ & m) == m)
        fv |= m;
    }
    for (const Bitboard m: kDiags.d1)
      if (m && (occ & m) == m)
        fd1 |= m;
    for (const Bitboard m: kDiags.d2)
      if (m && (occ & m) == m)
        fd2 |= m;

    // Rules 1+2 give the per-axis "already safe" sets that do not depend on the
    // fixpoint (full line, or on the edge for that axis). Diagonal endpoints all lie
    // on a board edge, so kEdges seals a diagonal from its open end.
    const Bitboard base_h  = fh | kFileA | kFileH;
    const Bitboard base_v  = fv | kRank1 | kRank8;
    const Bitboard base_d1 = fd1 | kEdges;
    const Bitboard base_d2 = fd2 | kEdges;

    // Rule 3: grow to a fixpoint. A disc is stable when it is p and safe on all four
    // axes, where "safe" also credits a stable same-colour neighbour toward either
    // end of the axis.
    Bitboard st = 0, prev;
    do {
      prev              = st;
      const Bitboard h  = base_h | sE(st) | sW(st);
      const Bitboard v  = base_v | sN(st) | sS(st);
      const Bitboard d1 = base_d1 | sNW(st) | sSE(st); // d1 = r-f const, the NW-SE axis
      const Bitboard d2 = base_d2 | sNE(st) | sSW(st); // d2 = r+f const, the NE-SW axis
      st                = p & h & v & d1 & d2;
    } while (st != prev);

    return popcount(st);
  }

} // namespace islay

#endif // ISLAY_STABILITY_HPP
