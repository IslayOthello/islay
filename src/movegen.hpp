// Ported from Edax-reversi (Richard Delorme, Toshihiko Okuhara; GPLv3).
#ifndef ISLAY_MOVEGEN_HPP
#define ISLAY_MOVEGEN_HPP

#include "bitboard.hpp"
#include "common.hpp"

namespace islay {

  [[nodiscard]] Bitboard get_moves(Bitboard player, Bitboard opponent) noexcept;

  [[nodiscard]] bool can_move(Bitboard player, Bitboard opponent) noexcept;

  [[nodiscard]] ISLAY_FORCEINLINE int mobility(Bitboard player, Bitboard opponent) noexcept {
    return popcount(get_moves(player, opponent));
  }

  // Excludes the placed disc.
  [[nodiscard]] Bitboard flip(Square sq, Bitboard player, Bitboard opponent) noexcept;

  [[nodiscard]] const char *movegen_backend() noexcept;

  // Optional x86 SIMD across perft siblings; ARM batching regressed.

  [[nodiscard]] constexpr int movegen_batch_width() noexcept {
#if defined(ISLAY_BATCH_PERFT) && defined(__AVX512F__)
    return 8;
#elif defined(ISLAY_BATCH_PERFT) && defined(__AVX2__)
    return 4;
#else
    return 1;
#endif
  }

  // Arrays must not overlap.
  void get_moves_x(const Bitboard *__restrict p, const Bitboard *__restrict o, Bitboard *__restrict moves,
                   int n) noexcept;

  [[nodiscard]] bool movegen_selftest() noexcept;

} // namespace islay

#endif // ISLAY_MOVEGEN_HPP
