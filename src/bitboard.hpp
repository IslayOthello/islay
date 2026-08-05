// Rank-major squares: a1=0, b1=1, ..., h8=63.
#ifndef ISLAY_BITBOARD_HPP
#define ISLAY_BITBOARD_HPP

#include <bit>
#include <cstdint>

#include "common.hpp"

namespace islay {

  using Bitboard = std::uint64_t;

  // int keeps square and sentinel arithmetic cheap.
  using Square = int;

  inline constexpr Square PASS   = 64;
  inline constexpr Square NOMOVE = 65;

  enum class Color : int { Black = 0, White = 1 };

  [[nodiscard]] constexpr Color operator~(Color c) noexcept { return static_cast<Color>(static_cast<int>(c) ^ 1); }

  [[nodiscard]] ISLAY_FORCEINLINE int popcount(Bitboard b) noexcept { return std::popcount(b); }

  // lsb/msb require a nonzero mask.
  [[nodiscard]] ISLAY_FORCEINLINE Square lsb(Bitboard b) noexcept { return std::countr_zero(b); }

  [[nodiscard]] ISLAY_FORCEINLINE Square msb(Bitboard b) noexcept { return 63 - std::countl_zero(b); }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard square_bb(Square sq) noexcept { return Bitboard{1} << sq; }

  ISLAY_FORCEINLINE Square pop_lsb(Bitboard &b) noexcept {
    const Square i = std::countr_zero(b);
    b &= b - 1;
    return i;
  }

  // D4-closed masks; eval selftests catch file wrapping.

  inline constexpr Bitboard kFileA = 0x0101010101010101ULL;
  inline constexpr Bitboard kFileH = 0x8080808080808080ULL;
  inline constexpr Bitboard kRank1 = 0x00000000000000FFULL;
  inline constexpr Bitboard kRank8 = 0xFF00000000000000ULL;
  inline constexpr Bitboard kEdges = kFileA | kFileH | kRank1 | kRank8;

  inline constexpr Bitboard kCorners = 0x8100000000000081ULL;
  inline constexpr Bitboard kXSquares = 0x0042000000004200ULL;
  inline constexpr Bitboard kCSquares = 0x4281000000008142ULL;

  // TL, TR, BL, BR.
  inline constexpr Bitboard kQuadrant[4] = {
          0x000000000F0F0F0FULL, 0x00000000F0F0F0F0ULL, 0x0F0F0F0F00000000ULL, 0xF0F0F0F000000000ULL};
  [[nodiscard]] ISLAY_FORCEINLINE unsigned quadrant_of(Square sq) noexcept {
    return (static_cast<unsigned>(sq) >> 5 << 1) | ((static_cast<unsigned>(sq) >> 2) & 1u);
  }

  // Odd orthogonally connected empty regions.
  [[nodiscard]] ISLAY_FORCEINLINE Bitboard odd_empty_regions(Bitboard empties) noexcept {
    Bitboard remaining = empties;
    Bitboard odd       = 0;
    while (remaining) {
      Bitboard region = remaining & (0 - remaining);
      Bitboard prev;
      do {
        prev = region;
        const Bitboard east_west = ((region << 1) & ~kFileA) | ((region >> 1) & ~kFileH);
        region |= (east_west | (region << 8) | (region >> 8)) & empties;
      } while (region != prev);
      if (popcount(region) & 1)
        odd |= region;
      remaining &= ~region;
    }
    return odd;
  }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard x_square_of_corner(Square sq) noexcept {
    switch (sq) {
      case 0: return square_bb(9);   // a1 -> b2
      case 7: return square_bb(14);  // h1 -> g2
      case 56: return square_bb(49); // a8 -> b7
      case 63: return square_bb(54); // h8 -> g7
      default: return 0;
    }
  }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard dilate8(Bitboard b) noexcept {
    const Bitboard h = ((b << 1) & ~kFileA) | ((b >> 1) & ~kFileH); // no file wrap
    const Bitboard t = b | h;
    return h | (t << 8) | (t >> 8);
  }

  // D4 generators.
  [[nodiscard]] ISLAY_FORCEINLINE Bitboard byteswap(Bitboard b) noexcept {
#if defined(_MSC_VER)
    return _byteswap_uint64(b);
#else
    return __builtin_bswap64(b);
#endif
  }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard flip_vertical(Bitboard b) noexcept { return byteswap(b); }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard flip_horizontal(Bitboard b) noexcept {
    constexpr Bitboard k1 = 0x5555555555555555ULL;
    constexpr Bitboard k2 = 0x3333333333333333ULL;
    constexpr Bitboard k4 = 0x0F0F0F0F0F0F0F0FULL;
    b                     = ((b >> 1) & k1) | ((b & k1) << 1);
    b                     = ((b >> 2) & k2) | ((b & k2) << 2);
    b                     = ((b >> 4) & k4) | ((b & k4) << 4);
    return b;
  }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard transpose(Bitboard b) noexcept {
    constexpr Bitboard k1 = 0x5500550055005500ULL;
    constexpr Bitboard k2 = 0x3333000033330000ULL;
    constexpr Bitboard k4 = 0x0F0F0F0F00000000ULL;
    Bitboard           t  = k4 & (b ^ (b << 28));
    b ^= t ^ (t >> 28);
    t = k2 & (b ^ (b << 14));
    b ^= t ^ (t >> 14);
    t = k1 & (b ^ (b << 7));
    b ^= t ^ (t >> 7);
    return b;
  }

} // namespace islay

#endif // ISLAY_BITBOARD_HPP
