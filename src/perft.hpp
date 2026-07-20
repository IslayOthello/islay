/**
 * @file perft.hpp
 * @brief Move-generator performance/verification counts (perft).
 *
 * `perft(b, d, rule)` counts the leaf nodes of the move tree of depth `d`, using
 * bulk-counting at the final ply (the number of legal moves is added directly
 * via popcount instead of playing each one out). Game-over handling follows the
 * `Rule`: under Othello a stuck side passes (a pass consumes one ply, matching
 * Edax's `count_games` -- 4, 12, 56, 244, 1396, 8200, ...); under Reversi a
 * side with no move ends the game immediately (no passing). The two agree until
 * a position with a forced pass first appears in the tree.
 */
#ifndef ISLAY_PERFT_HPP
#define ISLAY_PERFT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "board.hpp"
#include "common.hpp"
#include "options.hpp"

namespace islay {

  /** Bulk-counting perft without a transposition table, under the given rule. */
  [[nodiscard]] std::uint64_t perft(const Board &b, int depth, Rule rule) noexcept;

  /**
   * Transposition table caching perft(position, depth) node counts.
   *
   * The caller supplies the canonical (symmetry-reduced) board as the key, so a
   * hit is always exact -- the full (player, opponent, depth) triple is verified.
   */
  class PerftTT {
  public:
    explicit PerftTT(std::size_t mib = 256) { resize(mib); }

    /** Reallocate to about `mib` mebibytes (rounded to a power-of-two slot count). */
    void resize(std::size_t mib);

    /** Wipe all entries. */
    void clear() noexcept;

    /** Number of slots. */
    [[nodiscard]] std::size_t slots() const noexcept { return mask_ + 1; }

    [[nodiscard]] bool probe(const Board &key, int depth, std::uint64_t &nodes) const noexcept;
    void               store(const Board &key, int depth, std::uint64_t nodes) noexcept;

  private:
    struct Entry {
      Board         board;
      std::uint64_t nodes;
      std::int32_t  depth; // 0 == empty slot (perft never probes depth < 3)
    };
    std::vector<Entry> table_;
    std::size_t        mask_ = 0;
  };

  /** Bulk-counting perft using the symmetry-aware transposition cache `tt`.
   *  The cache holds counts for one rule at a time; clear it when the rule changes. */
  [[nodiscard]] std::uint64_t perft_cached(const Board &b, int depth, PerftTT &tt, Rule rule) noexcept;

} // namespace islay

#endif // ISLAY_PERFT_HPP
