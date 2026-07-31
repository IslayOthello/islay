/**
 * @file endgame.cpp
 * @brief The last-few-empties solver (contract and rationale: endgame.hpp).
 */
#include "endgame.hpp"

#include <utility>

#include "eval.hpp"      // terminal_score, kInf
#include "movegen.hpp"   // flip
#include "stability.hpp" // stable_count

namespace islay {
  namespace {

    /**
     * Exact score of a ONE-EMPTY position, without recursing or building a board.
     * At one empty the game is decided by how many discs the single move flips:
     * mover ends with P+1+f, opponent with O-f, a difference of (P - O + 1 + 2f), and
     * there are no empties left to award. Rule-aware, mirroring the search's own
     * terminal/pass handling so the oracle can pin them together.
     */
    template<Rule R>
    [[nodiscard]] ISLAY_FORCEINLINE int solve1(const Board &b, Bitboard moves) noexcept {
      const int    P = popcount(b.player), O = popcount(b.opponent);
      const Square e = static_cast<Square>(lsb(~(b.player | b.opponent)));
      if (moves != 0) { // at one empty the only possible move is e itself
        const int f = popcount(flip(e, b.player, b.opponent));
        return 100 * (P - O + 1 + 2 * f);
      }
      if constexpr (R == Rule::Othello) {
        const Bitboard of = flip(e, b.opponent, b.player); // mover passed; can the opponent play?
        if (of != 0)
          return 100 * (P - O - 1 - 2 * popcount(of));
      }
      // Reversi never passes; Othello here means neither side can move. Empty to the winner.
      if (P > O)
        return 100 * (P - O + 1);
      if (P < O)
        return 100 * (P - O - 1);
      return 0;
    }

  } // namespace

  template<Rule R>
  int endgame_solve(const Board &b, int alpha, int beta, int empties, std::uint64_t &nodes) noexcept {
    ++nodes;
    if (empties == 1)
      return solve1<R>(b, b.moves());

    // Stability cutoff. If the opponent already holds S provably-unflippable discs it
    // finishes with at least S, so the mover's final margin is at most 64 - 2S. Cheap
    // gate first: stable_count <= popcount(opponent), so the (pricey) fixpoint runs only
    // when even all-opponent-stable could drop the ceiling to alpha.
    {
      const int opp = popcount(b.opponent);
      if (100 * (64 - 2 * opp) <= alpha) {
        const int ub = 100 * (64 - 2 * stable_count(b.opponent, b.player));
        if (ub <= alpha)
          return ub;
      }
    }

    const Bitboard moves = b.moves();
    if (moves == 0) {
      if constexpr (R == Rule::Othello) {
        const Board passed = b.passed();
        if (passed.has_moves())
          return -endgame_solve<R>(passed, -beta, -alpha, empties, nodes); // pass costs no empty
      }
      return terminal_score(b);
    }

    // Move ordering. PARITY is primary -- an empty region with an ODD count hands its
    // last move to whoever plays into it, so true odd connected regions go first.
    // MOBILITY is the secondary key, but only from `empties`
    // >= kOrderMobility: it costs a get_moves per move, which is worth it up top (few
    // nodes, big subtrees, ordering drives the alpha-beta) but not down among the many
    // near-leaf nodes. Reorders only -- never a score change.
    constexpr int  kOrderMobility = 5;
    const Bitboard empt = ~(b.player | b.opponent);
    const Bitboard odd = odd_empty_regions(empt);
    const bool with_mob = empties >= kOrderMobility;

    Board child[36];
    Square sq[36];
    int    key[36];
    int    n = 0;
    for (Bitboard m = moves; m;) {
      const Square s = pop_lsb(m);
      int          k = (odd & square_bb(s)) ? (1 << 16) : 0;
      if (with_mob) {
        child[n] = b.play(s);                 // cache: the loop below would play it anyway
        k -= popcount(child[n].moves());      // fewer opponent replies first
      }
      sq[n]  = s;
      key[n] = k;
      ++n;
    }

    int best = -kInf;
    for (int i = 0; i < n; ++i) {
      int pick = i; // selection sort: a cutoff must not pay to order the rest
      for (int j = i + 1; j < n; ++j)
        if (key[j] > key[pick])
          pick = j;
      if (pick != i) {
        std::swap(sq[i], sq[pick]);
        std::swap(key[i], key[pick]);
        if (with_mob)
          std::swap(child[i], child[pick]);
      }
      const Board c = with_mob ? child[i] : b.play(sq[i]);
      const int   s = -endgame_solve<R>(c, -beta, -alpha, empties - 1, nodes);
      if (s > best) {
        best = s;
        if (s > alpha)
          alpha = s;
        if (alpha >= beta)
          break; // fail-high: the rest cannot matter
      }
    }
    return best;
  }

  template int endgame_solve<Rule::Othello>(const Board &, int, int, int, std::uint64_t &) noexcept;
  template int endgame_solve<Rule::Reversi>(const Board &, int, int, int, std::uint64_t &) noexcept;

} // namespace islay
