#include "endgame.hpp"

#include <utility>

#include "eval.hpp"      // terminal_score, kInf
#include "movegen.hpp"   // flip
#include "stability.hpp" // stable_count

namespace islay {
  namespace {

    // Direct one-empty score.
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
      // Award the last empty at game over.
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

    // Opponent-stability upper bound; popcount gates the fixpoint.
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

    // Odd regions first, then low child mobility from five empties.
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
        child[n] = b.play(s);            // reused below
        k -= popcount(child[n].moves()); // fewer replies first
      }
      sq[n]  = s;
      key[n] = k;
      ++n;
    }

    int best = -kInf;
    for (int i = 0; i < n; ++i) {
      int pick = i; // stop ordering after a cutoff
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
