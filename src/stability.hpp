// Sound lower bound on stable discs. Full lines, edges, and stable neighbours
// grow a four-axis fixpoint; overcounting would corrupt the endgame cutoff.
#ifndef ISLAY_STABILITY_HPP
#define ISLAY_STABILITY_HPP

#include <array>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "bitboard.hpp"

namespace islay {
  namespace stability_detail {

    // Neighbour shifts with file wrapping masked out.
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sE(Bitboard b) noexcept { return (b << 1) & ~kFileA; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sW(Bitboard b) noexcept { return (b >> 1) & ~kFileH; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sN(Bitboard b) noexcept { return b >> 8; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sS(Bitboard b) noexcept { return b << 8; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sNE(Bitboard b) noexcept { return (b >> 7) & ~kFileA; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sSW(Bitboard b) noexcept { return (b << 7) & ~kFileH; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sNW(Bitboard b) noexcept { return (b >> 9) & ~kFileH; }
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard sSE(Bitboard b) noexcept { return (b << 9) & ~kFileA; }

    struct StableBases {
      Bitboard h, v, d1, d2;
    };

    struct DiagMasks {
      std::array<Bitboard, 15> d1{};
      std::array<Bitboard, 15> d2{};
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

    [[nodiscard]] inline StableBases stable_bases(Bitboard occ) noexcept {
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

      return {fh | kFileA | kFileH, fv | kRank1 | kRank8, fd1 | kEdges, fd2 | kEdges};
    }

    [[nodiscard]] ISLAY_FORCEINLINE Bitboard spread_d1(Bitboard b) noexcept {
      constexpr Bitboard kNoA  = 0xFEFEFEFEFEFEFEFEULL;
      constexpr Bitboard kNoAB = 0xFCFCFCFCFCFCFCFCULL;
      constexpr Bitboard kNoAD = 0xF0F0F0F0F0F0F0F0ULL;
      constexpr Bitboard kNoH  = 0x7F7F7F7F7F7F7F7FULL;
      constexpr Bitboard kNoGH = 0x3F3F3F3F3F3F3F3FULL;
      constexpr Bitboard kNoEH = 0x0F0F0F0F0F0F0F0FULL;
      b |= ((b << 9) & kNoA) | ((b >> 9) & kNoH);
      b |= ((b << 18) & kNoAB) | ((b >> 18) & kNoGH);
      b |= ((b << 36) & kNoAD) | ((b >> 36) & kNoEH);
      return b;
    }

    [[nodiscard]] ISLAY_FORCEINLINE Bitboard spread_d2(Bitboard b) noexcept {
      constexpr Bitboard kNoA  = 0xFEFEFEFEFEFEFEFEULL;
      constexpr Bitboard kNoAB = 0xFCFCFCFCFCFCFCFCULL;
      constexpr Bitboard kNoAD = 0xF0F0F0F0F0F0F0F0ULL;
      constexpr Bitboard kNoH  = 0x7F7F7F7F7F7F7F7FULL;
      constexpr Bitboard kNoGH = 0x3F3F3F3F3F3F3F3FULL;
      constexpr Bitboard kNoEH = 0x0F0F0F0F0F0F0F0FULL;
      b |= ((b << 7) & kNoH) | ((b >> 7) & kNoA);
      b |= ((b << 14) & kNoGH) | ((b >> 14) & kNoAB);
      b |= ((b << 28) & kNoEH) | ((b >> 28) & kNoAD);
      return b;
    }

    // Mark empty bytes, then expand each high-bit marker to a full rank.
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard zero_byte_mask(Bitboard b) noexcept {
      constexpr Bitboard kLow7 = 0x7F7F7F7F7F7F7F7FULL;
      constexpr Bitboard kHigh = 0x8080808080808080ULL;
      const Bitboard     high  = ~(((b & kLow7) + kLow7) | b | kLow7) & kHigh;
      return (high >> 7) * 0xFFULL;
    }

#if defined(__ARM_NEON)
    // Lanes carry the shift-9 and shift-7 diagonal families together.
    [[nodiscard]] ISLAY_FORCEINLINE uint64x2_t spread_diags(Bitboard b) noexcept {
      constexpr Bitboard kNoA   = 0xFEFEFEFEFEFEFEFEULL;
      constexpr Bitboard kNoAB  = 0xFCFCFCFCFCFCFCFCULL;
      constexpr Bitboard kNoAD  = 0xF0F0F0F0F0F0F0F0ULL;
      constexpr Bitboard kNoH   = 0x7F7F7F7F7F7F7F7FULL;
      constexpr Bitboard kNoGH  = 0x3F3F3F3F3F3F3F3FULL;
      constexpr Bitboard kNoEH  = 0x0F0F0F0F0F0F0F0FULL;
      uint64x2_t         d      = vdupq_n_u64(b);
      const auto         spread = [&d](int64x2_t shifts, uint64x2_t left_mask, uint64x2_t right_mask) noexcept {
        const uint64x2_t left  = vandq_u64(vshlq_u64(d, shifts), left_mask);
        const uint64x2_t right = vandq_u64(vshlq_u64(d, vnegq_s64(shifts)), right_mask);
        d                      = vorrq_u64(d, vorrq_u64(left, right));
      };
      spread(vcombine_s64(vcreate_s64(9), vcreate_s64(7)), vcombine_u64(vcreate_u64(kNoA), vcreate_u64(kNoH)),
             vcombine_u64(vcreate_u64(kNoH), vcreate_u64(kNoA)));
      spread(vcombine_s64(vcreate_s64(18), vcreate_s64(14)), vcombine_u64(vcreate_u64(kNoAB), vcreate_u64(kNoGH)),
             vcombine_u64(vcreate_u64(kNoGH), vcreate_u64(kNoAB)));
      spread(vcombine_s64(vcreate_s64(36), vcreate_s64(28)), vcombine_u64(vcreate_u64(kNoAD), vcreate_u64(kNoEH)),
             vcombine_u64(vcreate_u64(kNoEH), vcreate_u64(kNoAD)));
      return d;
    }
#endif

    [[nodiscard]] ISLAY_FORCEINLINE StableBases stable_bases_spread(Bitboard occ) noexcept {
      constexpr Bitboard kBytes = 0x0101010101010101ULL;
      const Bitboard     empty  = ~occ;
      Bitboard           cols   = empty | (empty >> 8);
      cols |= cols >> 16;
      cols |= cols >> 32;
      const Bitboard h = zero_byte_mask(empty) | kFileA | kFileH;
      const Bitboard v = ~((cols & 0xFFULL) * kBytes) | kRank1 | kRank8;
#if defined(__ARM_NEON)
      const uint64x2_t d = spread_diags(empty);
      return {h, v, ~vgetq_lane_u64(d, 0) | kEdges, ~vgetq_lane_u64(d, 1) | kEdges};
#else
      return {h, v, ~spread_d1(empty) | kEdges, ~spread_d2(empty) | kEdges};
#endif
    }

    [[nodiscard]] ISLAY_FORCEINLINE Bitboard stable_bits(Bitboard p, const StableBases &base) noexcept {
      Bitboard st = 0, prev;
      do {
        prev              = st;
        const Bitboard h  = base.h | sE(st) | sW(st);
        const Bitboard v  = base.v | sN(st) | sS(st);
        const Bitboard d1 = base.d1 | sNW(st) | sSE(st);
        const Bitboard d2 = base.d2 | sNE(st) | sSW(st);
        st                = p & h & v & d1 & d2;
      } while (st != prev);
      return st;
    }

  } // namespace stability_detail

  struct StableCounts {
    int player;
    int opponent;
  };

  [[nodiscard]] inline int stable_count(Bitboard p, Bitboard o) noexcept {
    using namespace stability_detail;
    const StableBases base = stable_bases(p | o);
    return popcount(stable_bits(p, base));
  }

  // Share the occupancy-dependent line scan.
  [[nodiscard]] inline StableCounts stable_counts(Bitboard p, Bitboard o) noexcept {
    using namespace stability_detail;
    const StableBases base = stable_bases_spread(p | o);
    return {popcount(stable_bits(p, base)), popcount(stable_bits(o, base))};
  }

} // namespace islay

#endif // ISLAY_STABILITY_HPP
