#ifndef ISLAY_HASH_HPP
#define ISLAY_HASH_HPP

#include <cstddef>
#include <cstdint>

#include "bitboard.hpp"
#include "common.hpp"

namespace islay {

  [[nodiscard]] ISLAY_FORCEINLINE std::uint64_t hash_board(Bitboard p, Bitboard o) noexcept {
    std::uint64_t h = p * 0x9E3779B97F4A7C15ULL + (o ^ 0x0123456789ABCDEFULL);
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
  }

  // Round down to a power of two for masked indexing.
  [[nodiscard]] inline std::size_t tt_slots_for(std::size_t mib, std::size_t entry_bytes) noexcept {
    const std::size_t bytes = (mib ? mib : 1) << 20;
    const std::size_t n     = bytes / entry_bytes;
    std::size_t       pow2  = 1;
    while ((pow2 << 1) <= n)
      pow2 <<= 1;
    if (pow2 < (std::size_t{1} << 10))
      pow2 = std::size_t{1} << 10; // >= 1024 slots
    return pow2;
  }

} // namespace islay

#endif // ISLAY_HASH_HPP
