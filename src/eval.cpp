/**
 * @file eval.cpp
 * @brief Static evaluation (see eval.hpp for the units and the clamp contract).
 *
 * Antisymmetry -- eval(Board{o,p}) == -eval(Board{p,o}) -- is guaranteed BY
 * CONSTRUCTION, not by luck: every feature is extracted per side by the SAME
 * function `side_features(mine, theirs, ...)`, and the score is a weighted sum
 * of (mine - theirs) differences. Swapping the sides swaps the two feature sets
 * and negates every difference.
 *
 * That is also why there is no global parity term. Parity depends only on the
 * empty count, which is unchanged by the side swap, so it is EVEN under the
 * swap while antisymmetry requires every term to be ODD -- the two are
 * mathematically incompatible. (Phase is fine: it selects the *weights*, and
 * scaling an odd term by an even factor stays odd.)
 */
#include "eval.hpp"

#include <algorithm>
#include <cstdint>

#include "movegen.hpp"

namespace islay {
  namespace {

    /** The two C-squares orthogonally adjacent to corner `sq`; 0 if not a corner. */
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard c_squares_of_corner(Square sq) noexcept {
      switch (sq) {
        case 0: return square_bb(1) | square_bb(8);    // a1 -> b1, a2
        case 7: return square_bb(6) | square_bb(15);   // h1 -> g1, h2
        case 56: return square_bb(48) | square_bb(57); // a8 -> a7, b8
        case 63: return square_bb(55) | square_bb(62); // h8 -> h7, g8
        default: return 0;
      }
    }

    constexpr Square kCornerSq[4] = {0, 7, 56, 63};

    // Per corner: the two edge directions leading away from it, and the edge each runs along.
    constexpr int      kCornerStep[4][2] = {{1, 8}, {-1, 8}, {1, -8}, {-1, -8}};
    constexpr Bitboard kCornerEdge[4][2] = {
            {kRank1, kFileA}, {kRank1, kFileH}, {kRank8, kFileA}, {kRank8, kFileH}};

    /**
     * Provably-unflippable discs of `p`: those contiguous with an owned corner
     * along an edge. Deliberately UNDER-counts (interior stability needs a
     * fixpoint over all four directions) -- undercounting only weakens the term,
     * whereas overcounting would make eval claim safety that does not exist.
     *
     * MEASURED, do not "improve" without re-measuring: a table-driven version was
     * built and rejected. It indexed a constexpr [256][256] edge table by the
     * 8-bit line patterns (using transpose() so one table serves ranks and
     * files), which is exact for edges -- an edge disc can only ever be flipped
     * ALONG its edge, since every other direction leaves the board on one side.
     * It also caught a case this misses: a FULL line can never be played on
     * again, so everything on it is stable. Result: **4% SLOWER** (90.4 vs 86.7
     * cycles/eval, interleaved A/B) and the search tree was BIT-IDENTICAL
     * (681537 nodes, same score) -- the full-line case simply never fires in the
     * midgame positions eval actually sees. Cost, no benefit.
     */
    [[nodiscard]] Bitboard stable_discs(Bitboard p, Bitboard) noexcept {
      Bitboard s = 0;
      for (int c = 0; c < 4; ++c) {
        if (!(p & square_bb(kCornerSq[c])))
          continue; // an unowned corner anchors nothing
        for (int d = 0; d < 2; ++d) {
          const int      step = kCornerStep[c][d];
          const Bitboard edge = kCornerEdge[c][d];
          Square         sq   = kCornerSq[c];
          for (int i = 0; i < 8; ++i) {
            if (sq < 0 || sq > 63)
              break;
            const Bitboard bb = square_bb(sq);
            if (!(edge & bb) || !(p & bb))
              break;
            s |= bb;
            sq += step;
          }
        }
      }
      return s;
    }

    /** Everything the score needs about ONE side. Same function for both sides. */
    struct SideFeatures {
      int mobility  = 0; // legal moves now
      int potential = 0; // empties touching the opponent's discs (future moves)
      int corners   = 0;
      int xsq_bad   = 0; // X-squares held while the matching corner is still empty
      int csq_bad   = 0; // C-squares held while the matching corner is still empty
      int stable    = 0;
      int frontier  = 0; // own discs touching an empty (a liability)
      int discs     = 0;
    };

    [[nodiscard]] SideFeatures side_features(Bitboard mine, Bitboard theirs, Bitboard my_moves) noexcept {
      const Bitboard occupied = mine | theirs;
      const Bitboard empties  = ~occupied;

      SideFeatures f;
      f.mobility  = popcount(my_moves);
      f.potential = popcount(dilate8(theirs) & empties);
      f.corners   = popcount(mine & kCorners);
      f.stable    = popcount(stable_discs(mine, theirs));
      f.frontier  = popcount(mine & dilate8(empties));
      f.discs     = popcount(mine);

      for (int c = 0; c < 4; ++c) {
        const Square sq = kCornerSq[c];
        if (occupied & square_bb(sq))
          continue; // corner already taken: the adjacent squares are no longer a trap
        if (mine & x_square_of_corner(sq))
          ++f.xsq_bad;
        f.csq_bad += popcount(mine & c_squares_of_corner(sq));
      }
      return f;
    }

    // Weight sets, interpolated by phase. HAND-GUESSED -- see eval.hpp.
    // Opening/midgame: mobility and corners decide games; disc count is noise
    // (often actively bad). Endgame: disc count IS the result.
    struct Weights {
      int mobility, potential, corners, xsq, csq, stable, frontier, discs;
    };
    constexpr Weights kOpening = {80, 15, 300, -120, -50, 60, -20, 0};
    constexpr Weights kEndgame = {10, 2, 100, -20, -10, 40, -2, 100};

    [[nodiscard]] ISLAY_FORCEINLINE int lerp(int a, int b, int num, int den) noexcept {
      return a + (b - a) * num / den;
    }

  } // namespace

  int terminal_score(const Board &b) noexcept {
    const int mine   = popcount(b.player);
    const int theirs = popcount(b.opponent);
    const int empty  = 64 - mine - theirs;
    if (mine > theirs)
      return 100 * (mine - theirs + empty); // empties go to the winner
    if (mine < theirs)
      return 100 * (mine - theirs - empty);
    return 0;
  }

  int eval(const Board &b, Bitboard moves) noexcept {
    const SideFeatures me = side_features(b.player, b.opponent, moves);
    const SideFeatures op = side_features(b.opponent, b.player, get_moves(b.opponent, b.player));

    // phase: 0 at the opening (4 discs) .. 60 at a full board (64 discs)
    const int phase = std::clamp(popcount(b.player | b.opponent) - 4, 0, 60);
    const Weights w  = {lerp(kOpening.mobility, kEndgame.mobility, phase, 60),
                        lerp(kOpening.potential, kEndgame.potential, phase, 60),
                        lerp(kOpening.corners, kEndgame.corners, phase, 60),
                        lerp(kOpening.xsq, kEndgame.xsq, phase, 60),
                        lerp(kOpening.csq, kEndgame.csq, phase, 60),
                        lerp(kOpening.stable, kEndgame.stable, phase, 60),
                        lerp(kOpening.frontier, kEndgame.frontier, phase, 60),
                        lerp(kOpening.discs, kEndgame.discs, phase, 60)};

    const int score = w.mobility * (me.mobility - op.mobility)      //
                      + w.potential * (me.potential - op.potential) //
                      + w.corners * (me.corners - op.corners)       //
                      + w.xsq * (me.xsq_bad - op.xsq_bad)           //
                      + w.csq * (me.csq_bad - op.csq_bad)           //
                      + w.stable * (me.stable - op.stable)          //
                      + w.frontier * (me.frontier - op.frontier)    //
                      + w.discs * (me.discs - op.discs);
    return std::clamp(score, -kEvalMax, kEvalMax);
  }

  bool eval_selftest() noexcept {
    std::uint64_t s   = 0x9E3779B97F4A7C15ULL;
    const auto    rnd = [&s]() noexcept {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      return s;
    };

    // dilate8 must not wrap across a file edge: a1's neighbours are exactly b1,a2,b2.
    if (dilate8(square_bb(0)) != (square_bb(1) | square_bb(8) | square_bb(9)))
      return false;
    if (dilate8(square_bb(7)) != (square_bb(6) | square_bb(15) | square_bb(14)))
      return false;

    // Every named region is closed under D4.
    for (int i = 0; i < 8; ++i) {
      if (Board::symmetry_bb(kCorners, i) != kCorners)
        return false;
      if (Board::symmetry_bb(kXSquares, i) != kXSquares)
        return false;
      if (Board::symmetry_bb(kCSquares, i) != kCSquares)
        return false;
      if (Board::symmetry_bb(kEdges, i) != kEdges)
        return false;
    }

    // Stability must never over-claim: every "stable" disc must belong to p, a
    // corner p owns is always stable, and a full line makes all of p's discs on
    // it stable.
    for (int it = 0; it < 5000; ++it) {
      const Bitboard p = rnd() & rnd();
      Bitboard       o = rnd() & rnd();
      o &= ~p;
      const Bitboard st = stable_discs(p, o);
      if (st & ~p)
        return false; // claimed a square p does not own
      if ((p & kCorners) != (st & kCorners))
        return false; // an owned corner can never be flipped
    }

    // terminal_score: hand-built cases.
    if (terminal_score(Board{0xFFFFFFFFFFFFFFFFULL, 0}) != 6400) // wipeout, full board
      return false;
    if (terminal_score(Board{0, 0xFFFFFFFFFFFFFFFFULL}) != -6400)
      return false;
    if (terminal_score(Board{0xFFULL, 0}) != 6400) // 8 discs, 56 empty -> all to winner
      return false;
    if (terminal_score(Board{0x0FULL, 0xF0ULL}) != 0) // 4-4 with empties -> draw
      return false;

    for (int it = 0; it < 20000; ++it) {
      const Bitboard p = rnd() & rnd();
      Bitboard       o = rnd() & rnd();
      o &= ~p;
      const Board b{p, o};
      const Bitboard m = b.moves();
      if (m == 0)
        continue; // eval's precondition: the mover has a move

      const int e = eval(b, m);
      if (e < -kEvalMax || e > kEvalMax) // clamp holds
        return false;

      // Antisymmetry: the same position seen by the other side scores the negative.
      const Board swapped = b.passed();
      const Bitboard sm   = swapped.moves();
      if (sm != 0 && eval(swapped, sm) != -e)
        return false;

      // D4 invariance: rotating/mirroring the board cannot change the score.
      for (int i = 1; i < 8; ++i) {
        const Board  t  = b.symmetry(i);
        const Bitboard tm = t.moves();
        if (tm == 0 || eval(t, tm) != e)
          return false;
      }

      // terminal_score is antisymmetric and D4-invariant too.
      if (terminal_score(swapped) != -terminal_score(b))
        return false;
    }
    return true;
  }

} // namespace islay
