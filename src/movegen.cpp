/**
 * @file movegen.cpp
 * @brief Move-generation back-ends (AVX2 / NEON / scalar) and their dispatch.
 *
 * Adapted from Edax-reversi (GPLv3):
 *   - get_moves        : Kogge-Stone style directional fill (board.c).
 *   - flip / mm_flip   : "parallel-prefix fill" outflank (flip_*_ppfill.c).
 *
 * The per-square directional masks that drive the vector flip (Edax's
 * MASK_LR[]) are regenerated here from board geometry at compile time, so no
 * hand-transcribed constant table is needed. The single flat layout
 * MASKS.m[sq][0..7] feeds both the NEON and AVX2 kernels unchanged.
 *
 * Direction index -> (drank, dfile) and equivalent bit shift:
 *   0:E(+1)  1:N(+8)  2:NE(+9)  3:NW(+7)   <- toward higher indices (LS1B)
 *   4:W(-1)  5:S(-8)  6:SW(-9)  7:SE(-7)   <- toward lower  indices (MS1B)
 */
#include "movegen.hpp"

#include <array>
#include <cstdint>

#include "common.hpp"

// --- back-end selection -----------------------------------------------------
// AVX2 is the fastest path on x86-64: one 256-bit vector covers all 4 directions.
//
// On arm64 NEON is only 2 lanes wide, so an all-NEON get_moves is actually
// *slower* than scalar on wide out-of-order cores (measured on Apple M3:
// all-NEON -13% vs scalar). But the integer ALUs and the NEON units are separate
// issue ports, so running *both* concurrently beats either alone: the HYBRID
// path puts 2 directions on the integer ALUs and 2 on NEON (measured +48%
// get_moves throughput vs scalar on M3). That is the arm64 default.
//   -DISLAY_USE_NEON forces the all-NEON kernels (narrow / in-order arm cores,
//   where the integer pipe is not wide enough to co-issue).
#if defined(__AVX2__)
#define ISLAY_MOVEGEN_AVX2 1
#elif defined(__ARM_NEON) && defined(ISLAY_USE_NEON)
#define ISLAY_MOVEGEN_NEON 1
#elif defined(__ARM_NEON)
#define ISLAY_MOVEGEN_HYBRID 1
#endif

#if defined(ISLAY_MOVEGEN_AVX2)
#include <immintrin.h>
#elif defined(ISLAY_MOVEGEN_NEON) || defined(ISLAY_MOVEGEN_HYBRID)
#include <arm_neon.h>
#endif

// Batched (SIMD-across-boards) kernels need the x86 intrinsics even when the
// per-board back-end above is something else.
#if defined(ISLAY_BATCH_PERFT) && (defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace islay {
  namespace {

    // Board-edge mask: files b..g. Stops horizontal/diagonal fills wrapping rows.
    constexpr Bitboard kInner = 0x7E7E7E7E7E7E7E7EULL;

    // Per-direction (drank, dfile) for the 8 half-directions, LS1B group first.
    constexpr std::array<int, 8> kDr{0, 1, 1, 1, 0, -1, -1, -1};
    constexpr std::array<int, 8> kDf{1, 0, 1, -1, -1, 0, -1, 1};

    // --- MASK_LR: for each square, the ray of squares in each half-direction -----
    // --- (excluding the square, stopping at the board edge). ---------------------
    struct alignas(64) MaskTable {
      std::array<std::array<Bitboard, 8>, 66> m; // 64 squares + PASS + NOMOVE (zero)
    };

    constexpr MaskTable make_masks() noexcept {
      MaskTable t{};
      for (Square sq = 0; sq < 64; ++sq) {
        const int r = sq >> 3;
        const int f = sq & 7;
        for (int d = 0; d < 8; ++d) {
          Bitboard ray = 0;
          int      rr  = r + kDr[d];
          int      ff  = f + kDf[d];
          while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
            ray |= Bitboard{1} << (rr * 8 + ff);
            rr += kDr[d];
            ff += kDf[d];
          }
          t.m[sq][d] = ray;
        }
      }
      return t;
    }

    constexpr MaskTable MASKS = make_masks();

    // ============================================================================
    //  Portable reference implementations (ground truth for the self-test).
    // ============================================================================

    [[nodiscard]] Bitboard flip_reference(Square sq, Bitboard p, Bitboard o) noexcept {
      Bitboard  flipped = 0;
      const int r0      = sq >> 3;
      const int f0      = sq & 7;
      for (int d = 0; d < 8; ++d) {
        Bitboard run     = 0;
        int      r       = r0 + kDr[d];
        int      f       = f0 + kDf[d];
        bool     bounded = false;
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
          const Bitboard bb = square_bb(r * 8 + f);
          if (o & bb) { // opponent disc: part of a candidate run
            run |= bb;
            r += kDr[d];
            f += kDf[d];
          } else if (p & bb) { // our disc: the run (if any) is flipped
            bounded = true;
            break;
          } else { // empty: dead end
            break;
          }
        }
        if (bounded)
          flipped |= run;
      }
      return flipped;
    }

    [[nodiscard]] Bitboard get_moves_reference(Bitboard p, Bitboard o) noexcept {
      Bitboard moves = 0;
      Bitboard e     = ~(p | o);
      while (e) {
        const Square sq = pop_lsb(e);
        if (flip_reference(sq, p, o))
          moves |= square_bb(sq);
      }
      return moves;
    }

    // ============================================================================
    //  Scalar fast path (portable fallback; also cross-checked by the self-test).
    // ============================================================================

    // One directional component of the move generator (Edax's 1-stage parallel
    // prefix): contiguous opponent runs bounded by a player disc, both directions.
    [[nodiscard]] ISLAY_FORCEINLINE Bitboard some_moves_scalar(Bitboard p, Bitboard mask, int dir) noexcept {
      const int d2 = dir + dir;
      Bitboard  fl = mask & (p << dir);
      Bitboard  fr = mask & (p >> dir);
      fl |= mask & (fl << dir);
      fr |= mask & (fr >> dir);
      const Bitboard ml = mask & (mask << dir);
      // mr := ml >> dir saves an AND versus mask & (mask >> dir). The two differ
      // only in the top `dir` bits, and bit i of (fr >> d2) is sourced from fr
      // bit i+d2 >= 64 for exactly those bits -- always zero -- so the difference
      // can never survive `mr & (fr >> d2)`. (Edax's AVX2 path does the same.)
      const Bitboard mr = ml >> dir;
      fl |= ml & (fl << d2);
      fr |= mr & (fr >> d2);
      fl |= ml & (fl << d2);
      fr |= mr & (fr >> d2);
      return (fl << dir) | (fr >> dir);
    }

    [[nodiscard]] Bitboard get_moves_scalar(Bitboard p, Bitboard o) noexcept {
      const Bitboard mask    = o & kInner;
      const Bitboard empties = ~(p | o);
      return (some_moves_scalar(p, mask, 1) // horizontal
              | some_moves_scalar(p, o, 8) // vertical
              | some_moves_scalar(p, mask, 7) // diagonal
              | some_moves_scalar(p, mask, 9)) // anti-diagonal
             & empties;
    }

    // Branchless "carry" outflank (Okuhara). On wide out-of-order cores (e.g.
    // Apple Silicon) this scalar kernel beats the 2-lane NEON vector flip: the
    // four LS1B directions resolve their bounding disc via a single add whose
    // carry sweeps the contiguous opponent run; the four MS1B directions use a
    // count-leading-zeros to find the same boundary from the top.
    [[nodiscard]] Bitboard flip_scalar(Square sq, Bitboard p, Bitboard o) noexcept {
      Bitboard flipped = 0;
      // Directions 0..3 (toward higher bit indices). (o | ~mask) + 1 carries up
      // through the on-ray opponent run and lands on the first non-opponent.
      for (int d = 0; d < 4; ++d) {
        const Bitboard mask     = MASKS.m[sq][d];
        Bitboard       outflank = (o | ~mask) + 1;
        outflank &= p & mask; // bounding player disc on the ray (one bit, or 0)
        const Bitboard nz = -static_cast<Bitboard>(outflank != 0); // guards overflow
        flipped |= (outflank - 1) & mask & nz;
      }
      // Directions 4..7 (toward lower bit indices): the boundary is the
      // most-significant non-opponent square on the ray.
      for (int d = 4; d < 8; ++d) {
        const Bitboard mask     = MASKS.m[sq][d];
        const Bitboard non_o    = mask & ~o;
        const Bitboard boundary = non_o ? square_bb(msb(non_o)) : 0; // MS1B
        flipped |= (boundary & p) ? (mask & ~((boundary << 1) - 1)) : 0;
      }
      return flipped;
    }

// ============================================================================
//  NEON kernels -- used by both the all-NEON path and the hybrid path.
// ============================================================================
#if defined(ISLAY_MOVEGEN_NEON) || defined(ISLAY_MOVEGEN_HYBRID)

    [[nodiscard]] ISLAY_FORCEINLINE uint64x2_t some_moves_neon(uint64x2_t pp, uint64x2_t mm, int dir) noexcept {
      const int64x2_t dd =
              vcombine_s64(vcreate_s64(static_cast<uint64_t>(dir)), vcreate_s64(static_cast<uint64_t>(-dir)));
      const int64x2_t ddx2 = vaddq_s64(dd, dd);
      uint64x2_t      ff   = vandq_u64(mm, vshlq_u64(pp, dd));
      ff                   = vorrq_u64(ff, vandq_u64(mm, vshlq_u64(ff, dd)));
      mm                   = vandq_u64(mm, vshlq_u64(mm, dd));
      ff                   = vorrq_u64(ff, vandq_u64(mm, vshlq_u64(ff, ddx2)));
      ff                   = vorrq_u64(ff, vandq_u64(mm, vshlq_u64(ff, ddx2)));
      return vshlq_u64(ff, dd);
    }

#endif // ISLAY_MOVEGEN_NEON || ISLAY_MOVEGEN_HYBRID

// ============================================================================
//  Hybrid path (arm64 default): 2 directions on the integer ALUs, 2 on NEON.
//  The two pipelines are independent issue ports, so the halves overlap; this
//  beats both all-scalar (+48% on M3) and all-NEON (which is itself -13% vs
//  scalar because 2 lanes cannot amortise the vector-unit round trip).
// ============================================================================
#if defined(ISLAY_MOVEGEN_HYBRID)

    [[nodiscard]] Bitboard get_moves_hybrid(Bitboard p, Bitboard o) noexcept {
      const Bitboard mask    = o & kInner;
      const Bitboard empties = ~(p | o);
      // NEON half: the two diagonals.
      const uint64x2_t pp = vdupq_n_u64(p);
      const uint64x2_t mm = vdupq_n_u64(mask);
      uint64x2_t       v  = some_moves_neon(pp, mm, 7);
      v                   = vorrq_u64(v, some_moves_neon(pp, mm, 9));
      // Integer half: horizontal + vertical.
      const Bitboard s = some_moves_scalar(p, mask, 1) | some_moves_scalar(p, o, 8);
      return (s | vgetq_lane_u64(v, 0) | vgetq_lane_u64(v, 1)) & empties;
    }

#endif // ISLAY_MOVEGEN_HYBRID

#if defined(ISLAY_MOVEGEN_NEON)

    [[nodiscard]] Bitboard get_moves_neon(Bitboard p, Bitboard o) noexcept {
      const Bitboard   mask    = o & kInner;
      const Bitboard   empties = ~(p | o);
      const uint64x2_t pp      = vdupq_n_u64(p);
      const uint64x2_t oo      = vdupq_n_u64(o);
      const uint64x2_t mm      = vdupq_n_u64(mask);
      uint64x2_t       moves   = some_moves_neon(pp, mm, 1);
      moves                    = vorrq_u64(moves, some_moves_neon(pp, oo, 8));
      moves                    = vorrq_u64(moves, some_moves_neon(pp, mm, 7));
      moves                    = vorrq_u64(moves, some_moves_neon(pp, mm, 9));
      return (vgetq_lane_u64(moves, 0) | vgetq_lane_u64(moves, 1)) & empties;
    }

    [[nodiscard]] Bitboard flip_neon(Square pos, Bitboard p, Bitboard o) noexcept {
      const int64x2_t  lshift18 = {1, 8};
      const int64x2_t  lshift79 = {9, 7};
      int64x2_t        rshift18 = {-1, -8};
      int64x2_t        rshift79 = {-9, -7};
      const uint64x2_t one      = vdupq_n_u64(1);
      const uint64x2_t pp       = vdupq_n_u64(p);
      const uint64x2_t oo       = vdupq_n_u64(o);

      // --- MS1B half (directions W,S / SW,SE): parallel-prefix "eraser" fill. ---
      uint64x2_t mask0   = vld1q_u64(&MASKS.m[pos][4]); // {W, S}
      uint64x2_t mask1   = vld1q_u64(&MASKS.m[pos][6]); // {SW, SE}
      uint64x2_t eraser0 = vbicq_u64(mask0, oo); // mask & ~o : non-opponent squares
      uint64x2_t eraser1 = vbicq_u64(mask1, oo);
      uint64x2_t oflank0 = vshlq_u64(vandq_u64(pp, mask0), lshift18);
      uint64x2_t oflank1 = vshlq_u64(vandq_u64(pp, mask1), lshift79);
      eraser0            = vorrq_u64(eraser0, vshlq_u64(eraser0, rshift18));
      eraser1            = vorrq_u64(eraser1, vshlq_u64(eraser1, rshift79));
      rshift18           = vaddq_s64(rshift18, rshift18);
      rshift79           = vaddq_s64(rshift79, rshift79);
      eraser0            = vorrq_u64(eraser0, vshlq_u64(eraser0, rshift18));
      eraser1            = vorrq_u64(eraser1, vshlq_u64(eraser1, rshift79));
      eraser0            = vorrq_u64(eraser0, vshlq_u64(eraser0, rshift18));
      eraser1            = vorrq_u64(eraser1, vshlq_u64(eraser1, rshift79));
      oflank0            = vbicq_u64(oflank0, eraser0);
      oflank1            = vbicq_u64(oflank1, eraser1);
      uint64x2_t flip    = vbicq_u64(mask0, vsubq_u64(oflank0, one));
      flip               = vorrq_u64(flip, vbicq_u64(mask1, vsubq_u64(oflank1, one)));

      // --- LS1B half (directions E,N / NE,NW): carry-propagation outflank. ------
      mask0   = vld1q_u64(&MASKS.m[pos][0]); // {E, N}
      mask1   = vld1q_u64(&MASKS.m[pos][2]); // {NE, NW}
      oflank0 = vaddq_u64(vornq_u64(oo, mask0), one); // (o | ~mask) + 1
      oflank1 = vaddq_u64(vornq_u64(oo, mask1), one);
      oflank0 = vandq_u64(vandq_u64(pp, mask0), oflank0);
      oflank1 = vandq_u64(vandq_u64(pp, mask1), oflank1);
      oflank0 = vqsubq_u64(oflank0, one); // saturating: 0 when no outflank
      oflank1 = vqsubq_u64(oflank1, one);
      flip    = vbslq_u64(mask1, oflank1, vbslq_u64(mask0, oflank0, flip));

      const uint64x2_t r = vorrq_u64(flip, vextq_u64(flip, flip, 1));
      return vgetq_lane_u64(r, 0);
    }

#endif // ISLAY_MOVEGEN_NEON

// ============================================================================
//  AVX2 path (x86-64).
// ============================================================================
#if defined(ISLAY_MOVEGEN_AVX2)

    [[nodiscard]] Bitboard get_moves_avx2(Bitboard p, Bitboard o) noexcept {
      const __m256i pp       = _mm256_set1_epi64x(static_cast<long long>(p));
      const __m256i oo       = _mm256_set1_epi64x(static_cast<long long>(o));
      const __m256i dir1     = _mm256_set_epi64x(7, 9, 8, 1);
      const __m256i dir2     = _mm256_add_epi64(dir1, dir1);
      const __m256i mask     = _mm256_and_si256(oo, _mm256_set_epi64x(0x007E7E7E7E7E7E00LL, 0x007E7E7E7E7E7E00LL,
                                                                      0x00FFFFFFFFFFFF00LL, 0x7E7E7E7E7E7E7E7ELL));
      const __m128i occupied = _mm_or_si128(_mm256_castsi256_si128(pp), _mm256_castsi256_si128(oo));

      __m256i flip_l      = _mm256_and_si256(mask, _mm256_sllv_epi64(pp, dir1));
      __m256i flip_r      = _mm256_and_si256(mask, _mm256_srlv_epi64(pp, dir1));
      flip_l              = _mm256_or_si256(flip_l, _mm256_and_si256(mask, _mm256_sllv_epi64(flip_l, dir1)));
      flip_r              = _mm256_or_si256(flip_r, _mm256_and_si256(mask, _mm256_srlv_epi64(flip_r, dir1)));
      const __m256i pre_l = _mm256_and_si256(mask, _mm256_sllv_epi64(mask, dir1));
      const __m256i pre_r = _mm256_srlv_epi64(pre_l, dir1);
      flip_l              = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, dir2)));
      flip_r              = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, dir2)));
      flip_l              = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, dir2)));
      flip_r              = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, dir2)));
      const __m256i mm    = _mm256_or_si256(_mm256_sllv_epi64(flip_l, dir1), _mm256_srlv_epi64(flip_r, dir1));

      const __m128i reduced = _mm_or_si128(_mm256_castsi256_si128(mm), _mm256_extracti128_si256(mm, 1));
      return static_cast<Bitboard>(_mm_cvtsi128_si64(
              _mm_andnot_si128(occupied, _mm_or_si128(reduced, _mm_unpackhi_epi64(reduced, reduced)))));
    }

    [[nodiscard]] Bitboard flip_avx2(Square pos, Bitboard p, Bitboard o) noexcept {
      const __m256i pp     = _mm256_set1_epi64x(static_cast<long long>(p));
      const __m256i oo     = _mm256_set1_epi64x(static_cast<long long>(o));
      const __m256i shift1 = _mm256_set_epi64x(7, 9, 8, 1);
      const __m256i shift2 = _mm256_set_epi64x(14, 18, 16, 2);

      // --- MS1B half (MASKS.m[pos][4..7] = {W,S,SW,SE}). ---
      __m256i mask     = _mm256_load_si256(reinterpret_cast<const __m256i *>(&MASKS.m[pos][4]));
      __m256i eraser   = _mm256_andnot_si256(oo, mask);
      __m256i outflank = _mm256_sllv_epi64(_mm256_and_si256(pp, mask), shift1);
      eraser           = _mm256_or_si256(eraser, _mm256_srlv_epi64(eraser, shift1));
      outflank         = _mm256_andnot_si256(eraser, outflank);
      eraser           = _mm256_srlv_epi64(eraser, shift2);
      outflank         = _mm256_andnot_si256(eraser, outflank);
      outflank         = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, shift2), outflank);
      __m256i flip     = _mm256_and_si256(mask, _mm256_sub_epi64(_mm256_setzero_si256(), outflank));

      // --- LS1B half (MASKS.m[pos][0..3] = {E,N,NE,NW}). ---
      mask     = _mm256_load_si256(reinterpret_cast<const __m256i *>(&MASKS.m[pos][0]));
      outflank = _mm256_andnot_si256(oo, mask);
      outflank = _mm256_and_si256(outflank, _mm256_sub_epi64(_mm256_setzero_si256(), outflank));
      outflank = _mm256_and_si256(outflank, pp);
      eraser   = _mm256_sub_epi64(_mm256_cmpeq_epi64(outflank, _mm256_setzero_si256()), outflank);
      flip     = _mm256_or_si256(flip, _mm256_andnot_si256(eraser, mask));

      __m128i r = _mm_or_si128(_mm256_castsi256_si128(flip), _mm256_extracti128_si256(flip, 1));
      r         = _mm_or_si128(r, _mm_shuffle_epi32(r, 0x4e));
      return static_cast<Bitboard>(_mm_cvtsi128_si64(r));
    }

#endif // ISLAY_MOVEGEN_AVX2

// ============================================================================
//  Batched path: SIMD across independent BOARDS (one board per lane).
//
//  Because every lane runs the same direction, the fills use *immediate* shifts
//  -- the cheapest form -- and one instruction advances all N boards at once.
//  The kernel is a lane-parallel transcription of some_moves_scalar (same
//  algorithm, same `mr = ml >> dir` shortcut), so results are bit-identical to
//  the scalar path; movegen_selftest() checks that lane-for-lane.
// ============================================================================
#if defined(ISLAY_BATCH_PERFT) && defined(__AVX512F__)

    template<int dir>
    [[nodiscard]] ISLAY_FORCEINLINE __m512i some_moves_x8(__m512i pp, __m512i mm) noexcept {
      __m512i fl       = _mm512_and_si512(mm, _mm512_slli_epi64(pp, dir));
      __m512i fr       = _mm512_and_si512(mm, _mm512_srli_epi64(pp, dir));
      fl               = _mm512_or_si512(fl, _mm512_and_si512(mm, _mm512_slli_epi64(fl, dir)));
      fr               = _mm512_or_si512(fr, _mm512_and_si512(mm, _mm512_srli_epi64(fr, dir)));
      const __m512i ml = _mm512_and_si512(mm, _mm512_slli_epi64(mm, dir));
      const __m512i mr = _mm512_srli_epi64(ml, dir);
      fl               = _mm512_or_si512(fl, _mm512_and_si512(ml, _mm512_slli_epi64(fl, dir * 2)));
      fr               = _mm512_or_si512(fr, _mm512_and_si512(mr, _mm512_srli_epi64(fr, dir * 2)));
      fl               = _mm512_or_si512(fl, _mm512_and_si512(ml, _mm512_slli_epi64(fl, dir * 2)));
      fr               = _mm512_or_si512(fr, _mm512_and_si512(mr, _mm512_srli_epi64(fr, dir * 2)));
      return _mm512_or_si512(_mm512_slli_epi64(fl, dir), _mm512_srli_epi64(fr, dir));
    }

    [[nodiscard]] ISLAY_FORCEINLINE __m512i get_moves_x8(__m512i pp, __m512i oo) noexcept {
      const __m512i mm = _mm512_and_si512(oo, _mm512_set1_epi64(static_cast<long long>(kInner)));
      __m512i       v  = some_moves_x8<1>(pp, mm);
      v                = _mm512_or_si512(v, some_moves_x8<8>(pp, oo));
      v                = _mm512_or_si512(v, some_moves_x8<7>(pp, mm));
      v                = _mm512_or_si512(v, some_moves_x8<9>(pp, mm));
      return _mm512_andnot_si512(_mm512_or_si512(pp, oo), v);
    }

#endif

#if defined(ISLAY_BATCH_PERFT) && defined(__AVX2__)

    template<int dir>
    [[nodiscard]] ISLAY_FORCEINLINE __m256i some_moves_x4(__m256i pp, __m256i mm) noexcept {
      __m256i fl       = _mm256_and_si256(mm, _mm256_slli_epi64(pp, dir));
      __m256i fr       = _mm256_and_si256(mm, _mm256_srli_epi64(pp, dir));
      fl               = _mm256_or_si256(fl, _mm256_and_si256(mm, _mm256_slli_epi64(fl, dir)));
      fr               = _mm256_or_si256(fr, _mm256_and_si256(mm, _mm256_srli_epi64(fr, dir)));
      const __m256i ml = _mm256_and_si256(mm, _mm256_slli_epi64(mm, dir));
      const __m256i mr = _mm256_srli_epi64(ml, dir);
      fl               = _mm256_or_si256(fl, _mm256_and_si256(ml, _mm256_slli_epi64(fl, dir * 2)));
      fr               = _mm256_or_si256(fr, _mm256_and_si256(mr, _mm256_srli_epi64(fr, dir * 2)));
      fl               = _mm256_or_si256(fl, _mm256_and_si256(ml, _mm256_slli_epi64(fl, dir * 2)));
      fr               = _mm256_or_si256(fr, _mm256_and_si256(mr, _mm256_srli_epi64(fr, dir * 2)));
      return _mm256_or_si256(_mm256_slli_epi64(fl, dir), _mm256_srli_epi64(fr, dir));
    }

    [[nodiscard]] ISLAY_FORCEINLINE __m256i get_moves_x4(__m256i pp, __m256i oo) noexcept {
      const __m256i mm = _mm256_and_si256(oo, _mm256_set1_epi64x(static_cast<long long>(kInner)));
      __m256i       v  = some_moves_x4<1>(pp, mm);
      v                = _mm256_or_si256(v, some_moves_x4<8>(pp, oo));
      v                = _mm256_or_si256(v, some_moves_x4<7>(pp, mm));
      v                = _mm256_or_si256(v, some_moves_x4<9>(pp, mm));
      return _mm256_andnot_si256(_mm256_or_si256(pp, oo), v);
    }

#endif

  } // namespace

  void get_moves_x(const Bitboard *__restrict p, const Bitboard *__restrict o, Bitboard *__restrict moves,
                   int n) noexcept {
#if defined(ISLAY_BATCH_PERFT) && defined(__AVX512F__)
    if (n == 8) {
      const __m512i pp = _mm512_loadu_si512(p);
      const __m512i oo = _mm512_loadu_si512(o);
      _mm512_storeu_si512(moves, get_moves_x8(pp, oo));
      return;
    }
#endif
#if defined(ISLAY_BATCH_PERFT) && defined(__AVX2__)
    if (n == 4) {
      const __m256i pp = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
      const __m256i oo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(o));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(moves), get_moves_x4(pp, oo));
      return;
    }
#endif
    for (int i = 0; i < n; ++i)
      moves[i] = get_moves(p[i], o[i]);
  }

  // ============================================================================
  //  Public dispatch: pick the widest ISA the build was compiled for.
  // ============================================================================

  ISLAY_HOT Bitboard get_moves(Bitboard p, Bitboard o) noexcept {
#if defined(ISLAY_MOVEGEN_AVX2)
    return get_moves_avx2(p, o);
#elif defined(ISLAY_MOVEGEN_NEON)
    return get_moves_neon(p, o);
#elif defined(ISLAY_MOVEGEN_HYBRID)
    return get_moves_hybrid(p, o);
#else
    return get_moves_scalar(p, o);
#endif
  }

  ISLAY_HOT Bitboard flip(Square sq, Bitboard p, Bitboard o) noexcept {
    ISLAY_ASSUME(static_cast<unsigned>(sq) < 64u); // sq always comes from pop_lsb
#if defined(ISLAY_MOVEGEN_AVX2)
    return flip_avx2(sq, p, o);
#elif defined(ISLAY_MOVEGEN_NEON)
    return flip_neon(sq, p, o);
#else
    return flip_scalar(sq, p, o);
#endif
  }

  bool can_move(Bitboard p, Bitboard o) noexcept { return get_moves(p, o) != 0; }

  const char *movegen_backend() noexcept {
#if defined(ISLAY_MOVEGEN_AVX2)
    return "AVX2";
#elif defined(ISLAY_MOVEGEN_NEON)
    return "NEON";
#elif defined(ISLAY_MOVEGEN_HYBRID)
    return "hybrid (int+NEON)";
#else
    return "scalar";
#endif
  }

  bool movegen_selftest() noexcept {
    std::uint64_t s   = 0x9E3779B97F4A7C15ULL;
    const auto    rnd = [&s]() noexcept {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      return s;
    };
    constexpr int W = movegen_batch_width();
    Bitboard      bp[8]{}, bo[8]{}, bm[8]{}, bref[8]{};
    int           filled = 0;

    for (int it = 0; it < 5000; ++it) {
      const Bitboard p = rnd() & rnd(); // ~25% density, board-like
      Bitboard       o = rnd() & rnd();
      o &= ~p; // players are disjoint
      const Bitboard ref_moves = get_moves_reference(p, o);
      if (get_moves(p, o) != ref_moves)
        return false;
      if (get_moves_scalar(p, o) != ref_moves)
        return false;

      // Batched kernel: fill a full vector, then check every lane against the
      // reference. (No-op when batching is disabled: W == 1.)
      if constexpr (W > 1) {
        bp[filled]   = p;
        bo[filled]   = o;
        bref[filled] = ref_moves;
        if (++filled == W) {
          get_moves_x(bp, bo, bm, W);
          for (int i = 0; i < W; ++i)
            if (bm[i] != bref[i])
              return false;
          filled = 0;
        }
      }

      Bitboard e = ~(p | o);
      while (e) {
        const Square   sq  = pop_lsb(e);
        const Bitboard ref = flip_reference(sq, p, o);
        if (flip(sq, p, o) != ref)
          return false;
        if (flip_scalar(sq, p, o) != ref)
          return false;
      }
    }
    return true;
  }

} // namespace islay
