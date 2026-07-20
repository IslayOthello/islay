/**
 * @file perft.cpp
 * @brief perft() / perft_cached() and the symmetry-aware perft transposition table.
 */
#include "perft.hpp"

#include "common.hpp"
#include "hash.hpp" // hash_board, tt_slots_for (shared with the search TT)

namespace islay {
  namespace {

    // Leaf count for a depth-1 node that has NO legal move. Mirrors the depth-1
    // semantics of perft_impl<R> exactly -- the ONLY place the two rules differ.
    template<Rule R>
    [[nodiscard]] ISLAY_FORCEINLINE std::uint64_t stuck_leaf_count(const Board &b) noexcept {
      if constexpr (R == Rule::Othello)
        return b.passed().has_moves() ? 1u : 0u; // a pass is one leaf
      else
        return 0u; // Reversi: no move ends the game
    }

    /**
     * depth-2 node, batched: its children are all depth-1 leaves, so their
     * get_moves calls are independent and can run lane-parallel. Rule handling
     * stays per-lane and scalar (a stuck child is rare), via stuck_leaf_count<R>,
     * so Othello/Reversi still diverge at exactly the same positions.
     * Compiled away entirely when movegen_batch_width() == 1.
     */
    template<Rule R>
    [[nodiscard]] std::uint64_t perft2_batched(const Board &b, Bitboard moves) noexcept {
      constexpr int W = movegen_batch_width();
      Board         kb[W];
      Bitboard      bp[W], bo[W], bm[W];
      std::uint64_t total = 0;
      int           fill  = 0;
      Bitboard      m     = moves;
      while (m) {
        const Board c = b.play(pop_lsb(m));
        kb[fill]      = c;
        bp[fill]      = c.player;
        bo[fill]      = c.opponent;
        if (++fill == W) {
          get_moves_x(bp, bo, bm, W);
          for (int k = 0; k < W; ++k)
            total += bm[k] ? static_cast<std::uint64_t>(popcount(bm[k])) : stuck_leaf_count<R>(kb[k]);
          fill = 0;
        }
      }
      for (int k = 0; k < fill; ++k) { // tail: fewer than W children left
        const Bitboard cm = kb[k].moves();
        total += cm ? static_cast<std::uint64_t>(popcount(cm)) : stuck_leaf_count<R>(kb[k]);
      }
      return total;
    }

    // Rule is a compile-time template parameter so the per-node game-over test
    // (`if constexpr`) folds away -- the hot recursion carries no rule branch.
    template<Rule R>
    ISLAY_HOT ISLAY_FLATTEN std::uint64_t perft_impl(const Board &b, int depth) noexcept {
      const Bitboard moves = b.moves();

      if (depth == 1) [[likely]] {
        // Bulk-counting: the number of legal moves *is* the leaf count.
        if (moves) [[likely]] {
          return static_cast<std::uint64_t>(popcount(moves));
        }
        return stuck_leaf_count<R>(b);
      }

      if (moves == 0) [[unlikely]] {
        if constexpr (R == Rule::Othello) {
          const Board passed = b.passed();
          return passed.has_moves() ? perft_impl<R>(passed, depth - 1) : 0u;
        } else {
          return 0u; // Reversi: game over
        }
      }

      if constexpr (movegen_batch_width() > 1) {
        if (depth == 2)
          return perft2_batched<R>(b, moves);
      }

      std::uint64_t n = 0;
      Bitboard      m = moves;
      while (m) {
        n += perft_impl<R>(b.play(pop_lsb(m)), depth - 1);
      }
      return n;
    }

    /**
     * Same tree walk as perft_impl, but with the depth as a template parameter.
     * That removes the per-node depth compare and lets the compiler specialise
     * (and flatten) each ply -- measured +12% on bench 11 (755M -> 846M nps,
     * interleaved A/B). Only the shallow depths are instantiated; deeper runs
     * fall back to the runtime-depth walk, whose overhead is amortised by then.
     */
    template<Rule R, int D>
    ISLAY_HOT ISLAY_FLATTEN std::uint64_t perft_td(const Board &b) noexcept {
      const Bitboard moves = b.moves();

      if constexpr (D == 1) {
        if (moves) [[likely]] {
          return static_cast<std::uint64_t>(popcount(moves)); // bulk-count
        }
        return stuck_leaf_count<R>(b);
      } else {
        if (moves == 0) [[unlikely]] {
          if constexpr (R == Rule::Othello) {
            const Board passed = b.passed();
            return passed.has_moves() ? perft_td<R, D - 1>(passed) : 0u;
          } else {
            return 0u; // Reversi: game over
          }
        }
        if constexpr (D == 2 && movegen_batch_width() > 1)
          return perft2_batched<R>(b, moves);

        std::uint64_t n = 0;
        Bitboard      m = moves;
        while (m) {
          n += perft_td<R, D - 1>(b.play(pop_lsb(m)));
        }
        return n;
      }
    }

    template<Rule R>
    [[nodiscard]] std::uint64_t perft_dispatch(const Board &b, int depth) noexcept {
      switch (depth) {
        case 1:
          return perft_td<R, 1>(b);
        case 2:
          return perft_td<R, 2>(b);
        case 3:
          return perft_td<R, 3>(b);
        case 4:
          return perft_td<R, 4>(b);
        case 5:
          return perft_td<R, 5>(b);
        case 6:
          return perft_td<R, 6>(b);
        case 7:
          return perft_td<R, 7>(b);
        case 8:
          return perft_td<R, 8>(b);
        case 9:
          return perft_td<R, 9>(b);
        case 10:
          return perft_td<R, 10>(b);
        case 11:
          return perft_td<R, 11>(b);
        case 12:
          return perft_td<R, 12>(b);
        default:
          return perft_impl<R>(b, depth);
      }
    }

  } // namespace

  std::uint64_t perft(const Board &b, int depth, Rule rule) noexcept {
    return rule == Rule::Othello ? perft_dispatch<Rule::Othello>(b, depth) : perft_dispatch<Rule::Reversi>(b, depth);
  }

  // ---------------------------------------------------------------------------
  //  Transposition table
  // ---------------------------------------------------------------------------

  void PerftTT::resize(std::size_t mib) {
    const std::size_t pow2 = tt_slots_for(mib, sizeof(Entry));
    table_.assign(pow2, Entry{});
    mask_ = pow2 - 1;
  }

  void PerftTT::clear() noexcept {
    for (Entry &e: table_)
      e = Entry{};
  }

  bool PerftTT::probe(const Board &key, int depth, std::uint64_t &nodes) const noexcept {
    const Entry &e = table_[hash_board(key.player, key.opponent) & mask_];
    if (e.depth == depth && e.board == key) [[unlikely]] {
      nodes = e.nodes;
      return true;
    }
    return false;
  }

  void PerftTT::store(const Board &key, int depth, std::uint64_t nodes) noexcept {
    Entry &e = table_[hash_board(key.player, key.opponent) & mask_];
    e.board  = key;
    e.nodes  = nodes;
    e.depth  = depth;
  }

  namespace {

    template<Rule R>
    ISLAY_HOT ISLAY_FLATTEN std::uint64_t perft_cached_impl(const Board &b, int depth, PerftTT &tt) noexcept {
      const Bitboard moves = b.moves();

      if (depth == 1) [[likely]] {
        if (moves) [[likely]] {
          return static_cast<std::uint64_t>(popcount(moves));
        }
        return stuck_leaf_count<R>(b);
      }

      if (moves == 0) [[unlikely]] {
        if constexpr (R == Rule::Othello) {
          const Board passed = b.passed();
          return passed.has_moves() ? perft_cached_impl<R>(passed, depth - 1, tt) : 0u;
        } else {
          return 0u;
        }
      }

      // depth 2 is below the TT gate (use_tt is depth >= 3), so batching here
      // cannot interact with the cache.
      if constexpr (movegen_batch_width() > 1) {
        if (depth == 2)
          return perft2_batched<R>(b, moves);
      }

      // Caching tiny subtrees costs more than it saves; gate on depth (matches
      // Edax, which only hashes perft nodes above depth 2). Key on the canonical
      // form so all 8 rotations/mirrors of a position share one entry.
      const bool use_tt = depth >= 3;
      Board      key{};
      if (use_tt) {
        key = b.canonical();
        std::uint64_t cached;
        if (tt.probe(key, depth, cached))
          return cached;
      }

      std::uint64_t n = 0;
      Bitboard      m = moves;
      while (m) {
        n += perft_cached_impl<R>(b.play(pop_lsb(m)), depth - 1, tt);
      }

      if (use_tt)
        tt.store(key, depth, n);
      return n;
    }

    /** Compile-time-depth twin of perft_cached_impl (see perft_td). The TT gate
     *  (depth >= 3) folds away too, so shallow plies carry no cache branch. */
    template<Rule R, int D>
    ISLAY_HOT ISLAY_FLATTEN std::uint64_t perft_cached_td(const Board &b, PerftTT &tt) noexcept {
      const Bitboard moves = b.moves();

      if constexpr (D == 1) {
        if (moves) [[likely]] {
          return static_cast<std::uint64_t>(popcount(moves));
        }
        return stuck_leaf_count<R>(b);
      } else {
        if (moves == 0) [[unlikely]] {
          if constexpr (R == Rule::Othello) {
            const Board passed = b.passed();
            return passed.has_moves() ? perft_cached_td<R, D - 1>(passed, tt) : 0u;
          } else {
            return 0u;
          }
        }
        if constexpr (D == 2 && movegen_batch_width() > 1)
          return perft2_batched<R>(b, moves);

        // Caching tiny subtrees costs more than it saves; gate on depth (matches
        // Edax). Key on the canonical form so all 8 rotations/mirrors share one
        // entry. Measured: gate 3 beats 4/5/6 (3529/3738/4494/4695 ms at perft 13).
        if constexpr (D >= 3) {
          const Board   key = b.canonical();
          std::uint64_t cached;
          if (tt.probe(key, D, cached))
            return cached;
          std::uint64_t n = 0;
          Bitboard      m = moves;
          while (m) {
            n += perft_cached_td<R, D - 1>(b.play(pop_lsb(m)), tt);
          }
          tt.store(key, D, n);
          return n;
        } else {
          std::uint64_t n = 0;
          Bitboard      m = moves;
          while (m) {
            n += perft_cached_td<R, D - 1>(b.play(pop_lsb(m)), tt);
          }
          return n;
        }
      }
    }

    template<Rule R>
    [[nodiscard]] std::uint64_t perft_cached_dispatch(const Board &b, int depth, PerftTT &tt) noexcept {
      switch (depth) {
        case 1:
          return perft_cached_td<R, 1>(b, tt);
        case 2:
          return perft_cached_td<R, 2>(b, tt);
        case 3:
          return perft_cached_td<R, 3>(b, tt);
        case 4:
          return perft_cached_td<R, 4>(b, tt);
        case 5:
          return perft_cached_td<R, 5>(b, tt);
        case 6:
          return perft_cached_td<R, 6>(b, tt);
        case 7:
          return perft_cached_td<R, 7>(b, tt);
        case 8:
          return perft_cached_td<R, 8>(b, tt);
        case 9:
          return perft_cached_td<R, 9>(b, tt);
        case 10:
          return perft_cached_td<R, 10>(b, tt);
        case 11:
          return perft_cached_td<R, 11>(b, tt);
        case 12:
          return perft_cached_td<R, 12>(b, tt);
        default:
          return perft_cached_impl<R>(b, depth, tt);
      }
    }

  } // namespace

  std::uint64_t perft_cached(const Board &b, int depth, PerftTT &tt, Rule rule) noexcept {
    return rule == Rule::Othello ? perft_cached_dispatch<Rule::Othello>(b, depth, tt)
                                 : perft_cached_dispatch<Rule::Reversi>(b, depth, tt);
  }

} // namespace islay
