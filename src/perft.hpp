// Bulk-counting perft; Othello passes consume one ply, Reversi stops.
#ifndef ISLAY_PERFT_HPP
#define ISLAY_PERFT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "board.hpp"
#include "common.hpp"
#include "options.hpp"

namespace islay {

  [[nodiscard]] std::uint64_t perft(const Board &b, int depth, Rule rule) noexcept;

  // Full canonical board and depth are verified on every hit.
  class PerftTT {
  public:
    explicit PerftTT(std::size_t mib = 256) { resize(mib); }

    void resize(std::size_t mib);

    void clear() noexcept;

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

  // Clear `tt` before changing rules.
  [[nodiscard]] std::uint64_t perft_cached(const Board &b, int depth, PerftTT &tt, Rule rule) noexcept;

} // namespace islay

#endif // ISLAY_PERFT_HPP
