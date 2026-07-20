/**
 * @file movegen.hpp
 * @brief Othello move generation: legal moves, flips, mobility.
 *
 * Ported and streamlined from Edax-reversi (Richard Delorme, Toshihiko
 * Okuhara; GPLv3). Three interchangeable back-ends are selected at compile
 * time -- AVX2, NEON, or a portable scalar path -- all producing identical
 * results (checked by movegen_selftest()).
 */
#ifndef ISLAY_MOVEGEN_HPP
#define ISLAY_MOVEGEN_HPP

#include "bitboard.hpp"
#include "common.hpp"

namespace islay {

  /** Bitboard of every legal move for `player` against `opponent`. */
  [[nodiscard]] Bitboard get_moves(Bitboard player, Bitboard opponent) noexcept;

  /** True iff `player` has at least one legal move. */
  [[nodiscard]] bool can_move(Bitboard player, Bitboard opponent) noexcept;

  /** Number of legal moves (mobility). */
  [[nodiscard]] ISLAY_FORCEINLINE int mobility(Bitboard player, Bitboard opponent) noexcept {
    return popcount(get_moves(player, opponent));
  }

  /**
   * Discs flipped by playing `sq` (0..63); excludes `sq` itself. Assumes `sq` is
   * a legal move for `player`; returns 0 for a non-flipping square.
   */
  [[nodiscard]] Bitboard flip(Square sq, Bitboard player, Bitboard opponent) noexcept;

  /** Name of the active back-end ("AVX2" / "NEON" / "hybrid (int+NEON)" / "scalar"). */
  [[nodiscard]] const char *movegen_backend() noexcept;

  // --- batched move generation (SIMD across independent boards) --------------
  // Sibling positions in a perft tree are independent, so N of them can occupy
  // the N lanes of one vector. Every lane then shifts by the SAME direction, so
  // the fills use *immediate* shifts and cost ~112 instructions for ALL N boards
  // (~14/board at N=8) versus 83 for a single board.
  //
  // Only profitable when the vector unit's issue throughput is comparable to the
  // integer ALUs' AND the vector is wide. MEASURED: on Apple M3 (NEON, 2 lanes,
  // ~1.3-2 vector instr/cycle vs ~6 integer) batching is -30% -- so arm64 keeps
  // width 1. x86 (4 lanes AVX2 / 8 lanes AVX-512) is expected to win but is
  // UNVALIDATED for speed here (no x86 hardware; Rosetta can only prove
  // correctness). Hence opt-in: -DISLAY_BATCH_PERFT.

  /** Lanes per batched get_moves_x call; 1 = batching unavailable/disabled. */
  [[nodiscard]] constexpr int movegen_batch_width() noexcept {
#if defined(ISLAY_BATCH_PERFT) && defined(__AVX512F__)
    return 8;
#elif defined(ISLAY_BATCH_PERFT) && defined(__AVX2__)
    return 4;
#else
    return 1;
#endif
  }

  /**
   * Moves for `n` independent boards at once (n <= movegen_batch_width()).
   * `moves[i] = get_moves(p[i], o[i])` for every i -- identical results, just
   * computed lane-parallel. Rule-agnostic: game rules live in the caller.
   *
   * The three arrays must not overlap (`__restrict`): without it the compiler
   * must assume a store to `moves` could alias `p`/`o` and reload them.
   */
  void get_moves_x(const Bitboard *__restrict p, const Bitboard *__restrict o, Bitboard *__restrict moves,
                   int n) noexcept;

  /** Cross-check every back-end against the portable reference over a random sweep. */
  [[nodiscard]] bool movegen_selftest() noexcept;

} // namespace islay

#endif // ISLAY_MOVEGEN_HPP
