// `ref_negamax` is the fixed-depth oracle used by search_selftest.
#include "search.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

#include "endgame.hpp"
#include "eval.hpp"
#include "hash.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "pattern.hpp"
#include "stability.hpp"

namespace islay {
  namespace {

    constexpr bool kUseTT       = true;
    constexpr bool kUseOrdering = true;
    constexpr bool kUsePVS      = true;
    constexpr bool kUseKillers  = true;
    constexpr bool kUsePrefetch = true;

    // Rebuild with this on for `searchstats` telemetry.
    constexpr bool kStats = false;

    constexpr bool kUseExtensions = true;
    constexpr bool kUsePruning    = true;
    // Runtime-disabled until a positive match.
    constexpr bool kUseNmp        = true;
    constexpr int  kNmpMinDepth   = 3;
    constexpr int  kNmpMinEmpties = 10;
    constexpr int  kNmpBaseR      = 2;
    constexpr int  kNmpDepthDiv   = 6;
    constexpr bool kUseProbCut    = true;
    constexpr bool kUseLastEmpty  = true;
    // Aspiration added 2.6% nodes on top of ProbCut.
    constexpr bool kUseAspiration = false;
    constexpr int  kAspWindow     = 500; // centi-discs; ~2 sigma of the d-2 -> d fit
    constexpr int  kAspMinDepth   = 4;
    // Calibrated LMR added 1.2% nodes; the fixed rule stays shipped.
    constexpr int kLmrMax = 4;

    [[nodiscard]] ISLAY_FORCEINLINE int lmr_reduction(int depth, int i) noexcept {
      int r = 1;
      if (i >= 6)
        ++r;
      if (i >= 9)
        ++r;
      if (i >= 12)
        ++r;
      if (r > kLmrMax)
        r = kLmrMax;
      if (r > depth - 2)
        r = depth - 2;
      return r < 0 ? 0 : r;
    }

    constexpr int kHistoryLimit = 1 << 20;

    ISLAY_FORCEINLINE void update_history(int (&row)[64], Square sq, int bonus) noexcept {
      row[sq] += bonus;
      if (row[sq] > kHistoryLimit || row[sq] < -kHistoryLimit)
        for (int &value: row)
          value /= 2;
    }

    constexpr bool kUseStabilityCut = true;
    constexpr bool kUseParity       = true;

    // Parity-preserving shallow-to-deep regression for ProbCut.
    struct ProbCutFit {
      float a, b, sigma;
    };
    constexpr int kProbCutMaxFitDepth = 12;
    // Fitted on weights/v22.nnue with `pcdata 1600 12`.
    constexpr ProbCutFit kProbCutFit[kProbCutMaxFitDepth + 1] = {
            {1, 0, 999},
            {1, 0, 999},
            {1, 0, 999}, // 0..2: unused, never consulted
            {1.010f, -114.1f, 397.8f}, // d=3, unused
            {1.001f, 103.7f, 317.4f}, // d=4, unused
            {1.010f, -50.9f, 273.4f}, // d=5
            {1.008f, 45.3f, 242.4f}, // d=6
            {1.005f, -34.1f, 225.8f}, // d=7
            {1.005f, 21.1f, 226.1f}, // d=8
            {1.006f, -2.5f, 204.9f}, // d=9
            {1.004f, 4.2f, 189.6f}, // d=10
            {1.007f, -10.4f, 179.8f}, // d=11
            {1.016f, 6.9f, 176.1f}, // d=12
    };
    constexpr int kProbCutMinDepth = 5;
    constexpr int kAbdadaMinDepth  = 4;

    // d-4 preserves parity and makes deep probes about five times cheaper.
    constexpr int        kProbCutGap4MinDepth                  = 9;
    constexpr ProbCutFit kProbCutFit4[kProbCutMaxFitDepth + 1] = {
            {1, 0, 999},
            {1, 0, 999},
            {1, 0, 999},
            {1, 0, 999}, // 0..3: unused
            {1, 0, 999},
            {1, 0, 999},
            {1, 0, 999}, // 4..6: gap 2 is used there
            {1.017f, -86.9f, 328.4f}, // d=7
            {1.017f, 65.2f, 294.8f}, // d=8
            {1.013f, -38.2f, 277.0f}, // d=9
            {1.011f, 24.4f, 268.3f}, // d=10
            {1.014f, -13.8f, 252.1f}, // d=11
            {1.021f, 10.4f, 228.0f}, // d=12
    };

    // Static-eval gate skips shallow probes unlikely to reach the threshold.
    constexpr float      kProbCutGateT = 0.5f; // gate confidence in sigmas; larger = skip fewer
    constexpr ProbCutFit kProbCutGateFit[kProbCutMaxFitDepth + 1] = {
            {0.983f, 279.5f, 549.4f}, {0.980f, 501.4f, 532.4f}, {0.986f, 148.2f, 562.1f}, {0.980f, 438.3f, 635.9f},
            {0.988f, 251.9f, 642.8f}, {0.994f, 390.6f, 674.0f}, {0.996f, 299.4f, 692.9f}, {1.001f, 358.2f, 707.5f},
            {1.004f, 321.7f, 723.4f}, {1.008f, 357.4f, 733.0f}, {1.009f, 326.9f, 743.8f}, {1.015f, 349.3f, 759.7f},
            {1.026f, 332.8f, 772.8f},
    };

    // LMP lost 24 Elo on v12 and stays runtime-disabled.
    constexpr bool kUseLMP      = true;
    constexpr int  kLMPMaxDepth = 3;
    constexpr int  kLMPCount[4] = {0, 4, 6, 9};

    // Per-stage ProbCut was neutral; sparse cells fall back to the pooled fit.
    constexpr ProbCutFit kMpcFit[kStageCount][kProbCutMaxFitDepth + 1] = {
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.999f, 87.1f, 418.5f},
             {0.992f, -21.4f, 356.5f},
             {0.990f, 44.5f, 313.2f},
             {0.991f, -21.4f, 285.3f},
             {1.000f, 26.2f, 271.6f},
             {0.999f, 5.6f, 260.0f},
             {1.001f, 10.1f, 242.5f},
             {1.002f, -1.6f, 224.1f},
             {1.006f, 0.6f, 217.1f}}, // stage 0
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.894f, 138.4f, 271.3f},
             {0.886f, 59.4f, 206.9f},
             {0.880f, 85.0f, 215.4f},
             {0.932f, -13.5f, 158.9f},
             {0.979f, -6.8f, 159.8f},
             {0.987f, -26.3f, 154.8f},
             {0.946f, 27.9f, 135.8f},
             {0.953f, 12.2f, 138.7f},
             {0.990f, -16.9f, 134.5f}}, // stage 1
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1.012f, 88.7f, 298.8f},
             {0.975f, -21.9f, 253.8f},
             {0.970f, 18.6f, 213.1f},
             {0.985f, 1.2f, 218.8f},
             {0.989f, 3.6f, 180.3f},
             {0.971f, 38.6f, 197.7f},
             {0.978f, 11.5f, 202.9f},
             {1.002f, -23.2f, 179.4f},
             {1.005f, -7.1f, 171.3f}}, // stage 2
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.986f, 33.2f, 345.2f},
             {0.971f, 27.9f, 294.3f},
             {0.947f, 31.6f, 282.6f},
             {0.966f, -2.3f, 269.8f},
             {0.993f, 5.2f, 231.3f},
             {0.983f, -22.9f, 269.2f},
             {0.972f, -18.3f, 208.6f},
             {0.983f, -5.0f, 192.6f},
             {0.986f, -4.9f, 176.9f}}, // stage 3
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1.016f, -33.4f, 439.5f},
             {0.986f, -68.2f, 374.4f},
             {0.984f, 20.1f, 293.6f},
             {0.969f, -28.0f, 259.6f},
             {0.980f, -10.8f, 266.9f},
             {0.990f, 20.1f, 224.7f},
             {0.984f, 49.4f, 219.4f},
             {0.991f, -3.5f, 176.3f},
             {0.991f, -10.7f, 197.7f}}, // stage 4
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.973f, 71.8f, 379.5f},
             {0.974f, -28.8f, 313.3f},
             {0.961f, -0.6f, 304.8f},
             {0.971f, -49.8f, 262.2f},
             {0.979f, 18.7f, 230.2f},
             {0.981f, 43.7f, 240.5f},
             {0.978f, 33.7f, 189.9f},
             {0.984f, 9.2f, 184.5f},
             {0.993f, -24.2f, 173.8f}}, // stage 5
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.988f, 96.7f, 403.1f},
             {0.995f, 4.3f, 328.8f},
             {0.987f, 7.5f, 328.0f},
             {0.975f, -18.6f, 318.6f},
             {0.997f, 29.7f, 248.4f},
             {0.997f, -15.0f, 256.9f},
             {1.004f, -30.5f, 203.2f},
             {0.995f, -23.8f, 200.4f},
             {0.994f, 12.2f, 196.6f}}, // stage 6
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1.002f, 94.5f, 411.1f},
             {0.980f, -54.0f, 390.5f},
             {0.996f, 52.5f, 291.7f},
             {0.977f, -21.7f, 255.9f},
             {0.971f, 35.6f, 275.8f},
             {0.994f, -6.8f, 258.0f},
             {1.011f, -22.4f, 241.9f},
             {0.998f, 14.1f, 233.4f},
             {1.013f, 12.8f, 187.0f}}, // stage 7
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1.024f, 137.6f, 491.9f},
             {1.011f, -31.2f, 387.5f},
             {1.010f, 84.0f, 340.3f},
             {1.017f, -35.0f, 300.7f},
             {1.011f, 26.2f, 305.4f},
             {1.029f, 9.2f, 272.0f},
             {1.034f, 15.2f, 271.0f},
             {1.005f, -14.4f, 244.2f},
             {1.011f, 34.9f, 229.0f}}, // stage 8
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.998f, 111.3f, 495.7f},
             {1.029f, -11.9f, 395.8f},
             {1.026f, 54.9f, 319.9f},
             {0.985f, -26.7f, 298.0f},
             {1.036f, 74.5f, 327.9f},
             {1.026f, -38.0f, 356.7f},
             {1.006f, 11.1f, 270.2f},
             {1.025f, 45.2f, 247.5f},
             {1.025f, 6.9f, 279.4f}}, // stage 9
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1.017f, 101.7f, 457.8f},
             {0.987f, -22.2f, 378.9f},
             {0.983f, 84.9f, 350.8f},
             {1.015f, 13.7f, 343.7f},
             {1.010f, 69.0f, 278.6f},
             {0.993f, 49.0f, 279.8f},
             {1.025f, 61.2f, 297.6f},
             {1.022f, 34.0f, 279.4f},
             {1.001f, 30.9f, 239.8f}}, // stage 10
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.999f, 87.1f, 418.5f},
             {0.992f, -21.4f, 356.5f},
             {0.990f, 44.5f, 313.2f},
             {0.991f, -21.4f, 285.3f},
             {1.000f, 26.2f, 271.6f},
             {0.999f, 5.6f, 260.0f},
             {1.001f, 10.1f, 242.5f},
             {1.002f, -1.6f, 224.1f},
             {1.006f, 0.6f, 217.1f}}, // stage 11
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.999f, 87.1f, 418.5f},
             {0.992f, -21.4f, 356.5f},
             {0.990f, 44.5f, 313.2f},
             {0.991f, -21.4f, 285.3f},
             {1.000f, 26.2f, 271.6f},
             {0.999f, 5.6f, 260.0f},
             {1.001f, 10.1f, 242.5f},
             {1.002f, -1.6f, 224.1f},
             {1.006f, 0.6f, 217.1f}}, // stage 12
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.999f, 87.1f, 418.5f},
             {0.992f, -21.4f, 356.5f},
             {0.990f, 44.5f, 313.2f},
             {0.991f, -21.4f, 285.3f},
             {1.000f, 26.2f, 271.6f},
             {0.999f, 5.6f, 260.0f},
             {1.001f, 10.1f, 242.5f},
             {1.002f, -1.6f, 224.1f},
             {1.006f, 0.6f, 217.1f}}, // stage 13
            {{1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {1, 0, 999},
             {0.999f, 87.1f, 418.5f},
             {0.992f, -21.4f, 356.5f},
             {0.990f, 44.5f, 313.2f},
             {0.991f, -21.4f, 285.3f},
             {1.000f, 26.2f, 271.6f},
             {0.999f, 5.6f, 260.0f},
             {1.001f, 10.1f, 242.5f},
             {1.002f, -1.6f, 224.1f},
             {1.006f, 0.6f, 217.1f}}, // stage 14
    };

    constexpr int kOrderMobilityMinDepth = 3;

    using Clock = std::chrono::steady_clock;
    [[nodiscard]] double now_ms() noexcept {
      return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
    }

    constexpr int kSquareValue[64] = {18, -4, 4, 2, 2, 4, -4, 18, //
                                      -4, -8, 1, 1, 1, 1, -8, -4, //
                                      4,  1,  2, 1, 1, 2, 1,  4, //
                                      2,  1,  1, 1, 1, 1, 1,  2, //
                                      2,  1,  1, 1, 1, 1, 1,  2, //
                                      4,  1,  2, 1, 1, 2, 1,  4, //
                                      -4, -8, 1, 1, 1, 1, -8, -4, //
                                      18, -4, 4, 2, 2, 4, -4, 18};

    // 240-byte D4-invariant no-TT policy distilled from v22 depth-10 searches.
    constexpr int          kOrderSquareOrbit[64] = {0, 1, 2, 3, 3, 2, 1, 0, //
                                                    1, 4, 5, 6, 6, 5, 4, 1, //
                                                    2, 5, 7, 8, 8, 7, 5, 2, //
                                                    3, 6, 8, 9, 9, 8, 6, 3, //
                                                    3, 6, 8, 9, 9, 8, 6, 3, //
                                                    2, 5, 7, 8, 8, 7, 5, 2, //
                                                    1, 4, 5, 6, 6, 5, 4, 1, //
                                                    0, 1, 2, 3, 3, 2, 1, 0};
    constexpr std::int16_t kOrderSquare[4][10]   = {
            {55, -39, 12, -7, -139, 20, 43, 85, 106, 8},
            {88, -50, 1, -3, -79, 6, 20, 68, 85, 8},
            {88, -6, 9, 5, -46, 0, 5, 36, 45, 8},
            {83, 4, 18, 9, -24, -26, 4, 41, 27, 8},
    };
    constexpr std::int16_t kOrderFlip[4]     = {-40, -26, -2, -4};
    constexpr std::int16_t kOrderMobility[4] = {-31, -30, -35, -58};
    constexpr std::int16_t kOrderPrev[36]    = {
            0,  4,   0,  14, -2, 7, 1,  4,  10, 20, -5, -4,  -2, 6,  -1, 13, -10, 7,
            -9, -11, -7, -8, -5, 3, -6, 16, -5, 8,  2,  -13, 1,  -7, -2, 3,  -9,  -12,
    };
    constexpr std::int16_t kOrderPrev2[36] = {
            0, 18, 2, -2,  12, -21, 12,  9, 0,  2,  -7, 10, 1, 6,   -12, 0,  3,   10,
            2, -9, 0, -12, 5,  -3,  -11, 1, -6, -1, -7, 5,  6, -10, -2,  -3, -10, 11,
    };

    [[nodiscard]] ISLAY_FORCEINLINE int order_relation(Square previous, Square square) noexcept {
      if (previous < 0 || previous >= 64)
        return 0;
      int dx = (previous & 7) - (square & 7);
      int dy = (previous >> 3) - (square >> 3);
      dx     = dx < 0 ? -dx : dx;
      dy     = dy < 0 ? -dy : dy;
      if (dx > dy)
        std::swap(dx, dy);
      return dy * (dy + 1) / 2 + dx;
    }

    [[nodiscard]] ISLAY_FORCEINLINE int learned_order_score(const Board &b, const Board &child, Square square,
                                                            int replies, Square prev, Square prev2) noexcept {
      int phase = pattern_stage(b.count()) / 4;
      if (phase > 3)
        phase = 3;
      const int flips = popcount(b.player ^ child.opponent ^ square_bb(square));
      return kOrderSquare[phase][kOrderSquareOrbit[square]] + kOrderFlip[phase] * flips +
             kOrderMobility[phase] * replies + kOrderPrev[order_relation(prev, square)] +
             kOrderPrev2[order_relation(prev2, square)];
    }

    [[nodiscard]] int leaf_value_scratch(const Board &b, Bitboard moves, Color stm) noexcept {
      if (!pattern_enabled())
        return eval(b, moves);
      PatternState s;
      s.set(b, stm);
      int black;
      if (nnue_enabled()) {
        std::uint32_t idx[kPatternInstances + 9];
        const int     discs = b.count();
        int           n     = pattern_indices(s, 0, mob_counts(b, stm, moves), idx);
        idx[n++] = static_cast<std::uint32_t>(nnue_net().features() - kNnueRFeat + (discs >= 4 ? (discs - 4) % 4 : 0));
        black    = nnue_net().score(idx, n, pattern_stage(discs));
      } else {
        black = pattern_weights().score_phase(s, b.count(), mob_counts(b, stm, moves), pattern_stage_interp());
      }
      const int mover = (stm == Color::Black) ? black : -black;
      return std::clamp(mover, -kEvalMax, kEvalMax);
    }

    // Full-window oracle: no TT, pruning, ordering, or pass-depth cost.
    template<Rule R>
    int ref_negamax(const Board &b, int depth, Color stm) noexcept {
      const Bitboard moves = b.moves();

      if (moves == 0) {
        if constexpr (R == Rule::Othello) {
          const Board passed = b.passed();
          if (passed.has_moves())
            return -ref_negamax<R>(passed, depth, ~stm);
        }
        return terminal_score(b);
      }
      if (depth <= 0)
        return leaf_value_scratch(b, moves, stm);

      int      best = -kInf;
      Bitboard m    = moves;
      while (m) {
        const int s = -ref_negamax<R>(b.play(pop_lsb(m)), depth - 1, ~stm);
        if (s > best)
          best = s;
      }
      return best;
    }

    struct ScoredMove {
      Board    child;
      Bitboard child_moves;
      int      score;
      Square   sq;
    };

  } // namespace

  namespace {
    // Six-byte TT payload packed into one atomic word.
    [[nodiscard]] ISLAY_FORCEINLINE std::uint64_t tt_pack(int score, int depth, std::uint8_t flag, std::uint8_t best,
                                                          std::uint8_t age) noexcept {
      return static_cast<std::uint64_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(score))) |
             (static_cast<std::uint64_t>(static_cast<std::uint8_t>(depth)) << 16) |
             (static_cast<std::uint64_t>(flag) << 24) | (static_cast<std::uint64_t>(best) << 32) |
             (static_cast<std::uint64_t>(age) << 40);
    }
    [[nodiscard]] ISLAY_FORCEINLINE int tt_score(std::uint64_t d) noexcept {
      return static_cast<std::int16_t>(d & 0xFFFF);
    }
    [[nodiscard]] ISLAY_FORCEINLINE int tt_depth(std::uint64_t d) noexcept {
      return static_cast<int>((d >> 16) & 0xFF);
    }
    [[nodiscard]] ISLAY_FORCEINLINE std::uint8_t tt_flag(std::uint64_t d) noexcept {
      return static_cast<std::uint8_t>((d >> 24) & 0xFF);
    }
    [[nodiscard]] ISLAY_FORCEINLINE std::uint8_t tt_best(std::uint64_t d) noexcept {
      return static_cast<std::uint8_t>((d >> 32) & 0xFF);
    }
    [[nodiscard]] ISLAY_FORCEINLINE std::uint8_t tt_age(std::uint64_t d) noexcept {
      return static_cast<std::uint8_t>((d >> 40) & 0xFF);
    }

  } // namespace

  void TranspositionTable::resize(std::size_t mib) {
    const std::size_t n = tt_slots_for(mib, sizeof(Slot));
    slots_              = std::vector<Slot>(n); // atomics are not copyable, so build fresh
    mask_               = n - 1;
    clear();
  }

  void TranspositionTable::clear() noexcept {
    for (Slot &s: slots_) {
      s.key_xor.store(0, std::memory_order_relaxed);
      s.data.store(0, std::memory_order_relaxed);
    }
    age_.store(0, std::memory_order_relaxed);
    used_.store(0, std::memory_order_relaxed);
  }

  bool TranspositionTable::probe(std::uint64_t key, Hit &out) const noexcept {
    if (!mask_)
      return false;
    const Slot         &s  = slots_[key & mask_];
    const std::uint64_t kx = s.key_xor.load(std::memory_order_relaxed);
    const std::uint64_t d  = s.data.load(std::memory_order_relaxed);
    // XOR validation turns torn entries into misses.
    if ((kx ^ d) != key || tt_flag(d) == kNone)
      return false;
    out.score = tt_score(d);
    out.depth = tt_depth(d);
    out.flag  = tt_flag(d);
    out.best  = tt_best(d);
    return true;
  }

  void TranspositionTable::store(std::uint64_t key, int score, int depth, std::uint8_t flag,
                                 std::uint8_t best) noexcept {
    if (!mask_)
      return;
    Slot               &s       = slots_[key & mask_];
    const std::uint64_t kx      = s.key_xor.load(std::memory_order_relaxed);
    const std::uint64_t old     = s.data.load(std::memory_order_relaxed);
    const bool          same    = ((kx ^ old) == key);
    const std::uint8_t  age     = age_.load(std::memory_order_relaxed);
    const bool replace = same || tt_flag(old) == kNone || tt_age(old) != age || tt_depth(old) <= depth;
    if (!replace)
      return;
    if (tt_flag(old) == kNone)
      used_.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t d = tt_pack(score, std::max(0, depth), flag, best, age);
    s.data.store(d, std::memory_order_relaxed);
    s.key_xor.store(key ^ d, std::memory_order_relaxed);
  }

  int TranspositionTable::hashfull() const noexcept {
    return mask_ ? static_cast<int>(used_.load(std::memory_order_relaxed) * 1000 / (mask_ + 1)) : 0;
  }

  unsigned CorrectionHistory::edge_bucket(const Board &b) noexcept {
    std::uint64_t x = (b.player & kEdges) * 0x9E3779B185EBCA87ULL;
    x ^= (b.opponent & kEdges) * 0xC2B2AE3D27D4EB4FULL;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    return static_cast<unsigned>((x ^ (x >> 32)) & 255U);
  }

  void CorrectionHistory::clear() noexcept {
    std::memset(stage_, 0, sizeof stage_);
    std::memset(prev_, 0, sizeof prev_);
    std::memset(prev2_, 0, sizeof prev2_);
    std::memset(edge_, 0, sizeof edge_);
  }

  int CorrectionHistory::predict(const Board &b, int stage, Square prev_move, Square prev2_move) const noexcept {
    stage   = std::clamp(stage, 0, kStageCount - 1);
    int sum = stage_[stage] + edge_[stage][edge_bucket(b)];
    int n   = 2;
    if (prev_move < 64) {
      sum += prev_[stage][prev_move];
      ++n;
    }
    if (prev2_move < 64) {
      sum += prev2_[stage][prev2_move];
      ++n;
    }
    return sum / n;
  }

  void CorrectionHistory::update_entry(std::int16_t &entry, int target, int rate) noexcept {
    const int value = entry;
    int       step  = (target - value) * rate / 256;
    if (step == 0 && value != target)
      step = target > value ? 1 : -1;
    entry = static_cast<std::int16_t>(std::clamp(value + step, -400, 400));
  }

  void CorrectionHistory::update(const Board &b, int stage, Square prev_move, Square prev2_move, int deep_score,
                                 int static_score, int depth) noexcept {
    stage            = std::clamp(stage, 0, kStageCount - 1);
    const int target = std::clamp(deep_score - static_score, -400, 400);
    const int rate   = std::clamp(4 + 2 * depth, 4, 32);
    update_entry(stage_[stage], target, rate);
    update_entry(edge_[stage][edge_bucket(b)], target, rate);
    if (prev_move < 64)
      update_entry(prev_[stage][prev_move], target, rate);
    if (prev2_move < 64)
      update_entry(prev2_[stage][prev2_move], target, rate);
  }

  void Searcher::resize(std::size_t mib) {
    tt_->resize(mib);
    clear();
  }

  void Searcher::clear() noexcept {
    stop_flag_.store(false, std::memory_order_relaxed);
    tt_->clear();
    std::memset(killers_, 0, sizeof killers_);
    std::memset(history_, 0, sizeof history_);
    std::memset(continuation_history_, 0, sizeof continuation_history_);
    std::memset(continuation_history_2_, 0, sizeof continuation_history_2_);
    correction_history_.clear();
  }

  void Searcher::set_correction_history_cap(int cap) noexcept {
    cap = std::clamp(cap, 0, 200);
    if (cap != correction_history_cap_) {
      correction_history_cap_ = cap;
      correction_history_.clear();
    }
  }

  void Searcher::new_search(const SearchLimits &limits) noexcept {
    pat_on_  = pattern_enabled();
    nnue_on_ = nnue_enabled();
    if (pat_on_ && ps_.size() < static_cast<std::size_t>(kMaxPly))
      ps_.resize(kMaxPly);
    nodes_   = 0;
    stopped_ = false;
    // The owner arms stop_flag_ before launch; clearing it here would race `stop`.
    node_cap_    = limits.nodes;
    start_ms_    = now_ms();
    deadline_ms_ = limits.movetime_ms > 0.0 ? start_ms_ + limits.movetime_ms : 0.0;
    soft_ms_     = 0.0;
    if (bump_age_)
      tt_->new_generation();
    if constexpr (kStats) {
      if (!stats_)
        stats_ = std::make_unique<SearchStats>();
      stats_->reset();
    }
  }

  void Searcher::check_stop() noexcept {
    if (stop_flag_.load(std::memory_order_relaxed))
      stopped_ = true;
    else if (node_cap_ && nodes_ >= node_cap_)
      stopped_ = true;
    else if (deadline_ms_ > 0.0 && now_ms() >= deadline_ms_)
      stopped_ = true;
  }

  void SearchStats::dump(std::ostream &o, bool full) const {
    auto pct = [](std::uint64_t a, std::uint64_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; };
    auto avg = [](std::uint64_t a, std::uint64_t b) { return b ? double(a) / double(b) : 0.0; };

    const char *hdr = "  key       nodes   pv%  cut%  all% | tthit ttcut | fh1%   cIdx    bf |    lmr lmrRe% |    fut  "
                      "cut% |     pc  cut% | pvsRe%\n";

    auto row = [&](const std::string &label, const Cell &a, std::uint64_t npv, std::uint64_t ncut, std::uint64_t nall) {
      o << std::fixed << std::setprecision(1) << "  " << std::setw(4) << std::left << label << std::right
        << std::setw(11) << a.nodes << std::setw(6) << pct(npv, a.nodes) << std::setw(6) << pct(ncut, a.nodes)
        << std::setw(6) << pct(nall, a.nodes) << "  |" << std::setw(6) << pct(a.tt_hit, a.tt_probe) << std::setw(6)
        << pct(a.tt_cut, a.tt_probe) << "  |" << std::setw(6) << pct(a.fh_first, a.fh) << std::setw(7)
        << std::setprecision(2) << avg(a.cut_idx_sum, a.fh) << std::setw(6) << avg(a.moves_searched, a.nodes)
        << std::setprecision(1) << "  |" << std::setw(7) << a.lmr_try << std::setw(7) << pct(a.lmr_re, a.lmr_try)
        << "  |" << std::setw(7) << a.fut_try << std::setw(6) << pct(a.fut_cut, a.fut_try) << "  |" << std::setw(7)
        << a.pc_try << std::setw(6) << pct(a.pc_cut, a.pc_try) << "  |" << std::setw(6) << pct(a.pvs_re, a.pvs_scout)
        << "\n";
    };

    Cell          gtot{};
    std::uint64_t gpv = 0, gcut = 0, gall = 0;
    o << "=== search telemetry (interior pvs nodes; leaves/terminals/endgame-solver excluded) ===\n";

    o << "by remaining depth:\n" << hdr;
    for (int d = 0; d < kD; ++d) {
      Cell          a{};
      std::uint64_t npv = 0, ncut = 0, nall = 0;
      for (int s = 0; s < kS; ++s) {
        a.add(cell[d][s][PV]);
        npv += cell[d][s][PV].nodes;
        a.add(cell[d][s][Cut]);
        ncut += cell[d][s][Cut].nodes;
        a.add(cell[d][s][All]);
        nall += cell[d][s][All].nodes;
      }
      if (!a.nodes)
        continue;
      row("d" + std::to_string(d), a, npv, ncut, nall);
      gtot.add(a);
      gpv += npv;
      gcut += ncut;
      gall += nall;
    }

    if (!gtot.nodes) {
      o << "  (nothing recorded -- run `go` in a kStats build first)\n";
      return;
    }

    o << "by game stage (stage s -> discs " << 4 << "+4s .. +3):\n" << hdr;
    for (int s = 0; s < kS; ++s) {
      Cell          a{};
      std::uint64_t npv = 0, ncut = 0, nall = 0;
      for (int d = 0; d < kD; ++d) {
        a.add(cell[d][s][PV]);
        npv += cell[d][s][PV].nodes;
        a.add(cell[d][s][Cut]);
        ncut += cell[d][s][Cut].nodes;
        a.add(cell[d][s][All]);
        nall += cell[d][s][All].nodes;
      }
      if (!a.nodes)
        continue;
      row("s" + std::to_string(s), a, npv, ncut, nall);
    }

    row("TOT", gtot, gpv, gcut, gall);
    o << "probe cost: ProbCut spent " << gtot.pc_probe_nodes << " probe nodes over " << gtot.pc_try << " attempts, "
      << gtot.pc_cut << " cut (" << std::fixed << std::setprecision(1) << pct(gtot.pc_probe_nodes, total_nodes)
      << "% of all " << total_nodes << " search nodes).\n";
    o << "  hi-probe: " << (gtot.pc_cut - gtot.pc_lo_cut) << " cuts of " << gtot.pc_try << " attempts ("
      << pct(gtot.pc_cut - gtot.pc_lo_cut, gtot.pc_try) << "%), " << (gtot.pc_probe_nodes - gtot.pc_lo_nodes)
      << " nodes\n"
      << "  lo-probe: " << gtot.pc_lo_cut << " cuts of " << gtot.pc_lo_try << " attempts ("
      << pct(gtot.pc_lo_cut, gtot.pc_lo_try) << "%), " << gtot.pc_lo_nodes
      << " nodes = " << pct(gtot.pc_lo_nodes, total_nodes) << "% of the whole search\n";
    o << "no-TT ordering: " << gtot.order_no_tt << " of " << gtot.order_nodes << " ordered nodes ("
      << pct(gtot.order_no_tt, gtot.order_nodes) << "%), " << gtot.order_no_tt_fh << " fail-highs, "
      << gtot.order_no_tt_fh_first << " on the first move (" << pct(gtot.order_no_tt_fh_first, gtot.order_no_tt_fh)
      << "%).\n";

    o << "LMR re-search rate by depth x move ordinal (pct; '.' = no samples, n = tries):\n";
    o << "   d \\ i ";
    for (int i = 0; i < 14; ++i)
      o << std::setw(6) << i;
    o << "        n\n";
    for (int d = 0; d < kD; ++d) {
      std::uint64_t dn = 0;
      for (int i = 0; i < kIdx; ++i)
        dn += lmr_try[d][i];
      if (!dn)
        continue;
      o << "   d" << std::setw(2) << std::left << d << std::right << "   ";
      for (int i = 0; i < 14; ++i) {
        if (!lmr_try[d][i]) {
          o << std::setw(6) << ".";
          continue;
        }
        o << std::setw(6) << std::fixed << std::setprecision(1) << pct(lmr_re[d][i], lmr_try[d][i]);
      }
      o << std::setw(9) << dn << "\n";
    }
    {
      std::uint64_t col[kIdx] = {}, all = 0;
      for (int d = 0; d < kD; ++d)
        for (int i = 0; i < kIdx; ++i) {
          col[i] += lmr_try[d][i];
          all += lmr_try[d][i];
        }
      o << "   share ";
      for (int i = 0; i < 14; ++i)
        o << std::setw(6) << std::fixed << std::setprecision(1) << pct(col[i], all);
      o << std::setw(9) << all << "\n";
    }

    if (full) {
      o << "full (depth x stage) node counts, nonzero cells only:\n";
      for (int d = 0; d < kD; ++d)
        for (int s = 0; s < kS; ++s) {
          const std::uint64_t nn = cell[d][s][PV].nodes + cell[d][s][Cut].nodes + cell[d][s][All].nodes;
          if (nn)
            o << "  d" << d << " s" << s << " nodes=" << nn << " pv=" << cell[d][s][PV].nodes
              << " cut=" << cell[d][s][Cut].nodes << " all=" << cell[d][s][All].nodes << "\n";
        }
    }
  }

  int Searcher::static_eval(const Board &b, Color stm) const noexcept {
    const Bitboard moves = b.moves();
    if (!pattern_enabled())
      return eval(b, moves); // hand-written eval (eval.cpp)
    PatternState ps;
    ps.set(b, stm);
    int black;
    if (nnue_enabled()) {
      std::uint32_t idx[kPatternInstances + 9];
      const int     discs = b.count();
      int           n     = pattern_indices(ps, 0, mob_counts(b, stm, moves), idx);
      idx[n++] = static_cast<std::uint32_t>(nnue_net().features() - kNnueRFeat + (discs >= 4 ? (discs - 4) % 4 : 0));
      black    = nnue_net().score(idx, n, pattern_stage(discs));
    } else {
      black = pattern_weights().score_phase(ps, b.count(), mob_counts(b, stm, moves), pattern_stage_interp());
    }
    const int mover = (stm == Color::Black) ? black : -black;
    return std::clamp(mover, -kEvalMax, kEvalMax);
  }

  template<bool Pat>
  int Searcher::leaf_eval(const Board &b, Bitboard moves, int ply, Color stm) const noexcept {
    if constexpr (!Pat) {
      (void) ply;
      (void) stm;
      return eval(b, moves); // hand-written eval (eval.cpp) is the default
    } else {
      (void) moves;
      int black;
      if (nnue_on_) {
        const int     discs = b.count();
        std::uint32_t idx[kPatternInstances + 9];
        int           n = pattern_indices(ps_[ply], 0, mob_counts(b, stm, moves), idx);
        idx[n++] = static_cast<std::uint32_t>(nnue_net().features() - kNnueRFeat + (discs >= 4 ? (discs - 4) % 4 : 0));
        black    = nnue_net().score(idx, n, pattern_stage(discs));
      } else {
        black = pattern_weights().score_phase(ps_[ply], b.count(), mob_counts(b, stm, moves), pattern_stage_interp());
      }
      const int mover = (stm == Color::Black) ? black : -black; // zero-sum
      return std::clamp(mover, -kEvalMax, kEvalMax);
    }
  }

  template<Rule R, bool Pat>
  int Searcher::pvs(Board b, int depth, int alpha, int beta, int ply, Color stm, Square prev_move, Square prev2_move,
                    bool can_null) noexcept {
    if (stopped_)
      return 0;
    ++nodes_;
    if (ply > seldepth_)
      seldepth_ = ply; // deepest ply reached: extensions and passes push past `depth`
    if ((nodes_ & 1023) == 0)
      check_stop();

    const Bitboard moves = move_stack_[ply];

    const int eg_empties = 64 - b.count();
    if (kUseLastEmpty && endgame_enabled_ && depth >= eg_empties && eg_empties <= kEndgameSolverMax) [[unlikely]]
      return endgame_solve<R>(b, alpha, beta, eg_empties, nodes_);

    // Resolve terminal/pass before the depth cutoff; passes cost no depth.
    if (moves == 0) [[unlikely]] {
      if constexpr (R == Rule::Othello) {
        const Board    passed       = b.passed();
        const Bitboard passed_moves = passed.moves();
        if (passed_moves != 0) {
          // Pass leaves features unchanged.
          if constexpr (Pat)
            if (ply + 1 < kMaxPly)
              ps_[ply + 1] = ps_[ply];
          move_stack_[ply + 1] = passed_moves;
          return -pvs<R, Pat>(passed, depth, -beta, -alpha, ply + 1, ~stm, prev_move, prev2_move);
        }
      }
      return terminal_score(b);
    }
    if (depth <= 0)
      return leaf_eval<Pat>(b, moves, ply, stm);
    if (ply >= kMaxPly - 1)
      return leaf_eval<Pat>(b, moves, ply, stm); // paranoia; unreachable on a legal 8x8 game

    const int           alpha_orig = alpha; // MUST be captured before the loop moves alpha
    const std::uint64_t key        = hash_board(b.player, b.opponent);
    Square              tt_move    = NOMOVE;

    SearchStats::Acc sacc;
    const bool       is_pv = beta - alpha > 1;

    if (kUseTT && tt_enabled_) {
      if constexpr (kStats)
        ++sacc.tt_probe;
      Entry e;
      if (tt_->probe(key, e)) {
        if constexpr (kStats)
          ++sacc.tt_hit;
        // Validate TT moves before any square arithmetic.
        if (e.best < 64 && (moves & square_bb(e.best)))
          tt_move = static_cast<Square>(e.best);
        if (e.depth >= depth) { // never at the root: root_search doesn't call this
          const int s = e.score;
          if (e.flag == kExact) {
            if constexpr (kStats) {
              ++sacc.tt_cut;
              stats_->flush(depth, pattern_stage(b.count()), SearchStats::PV, sacc);
            }
            return s;
          }
          if (e.flag == kLower && s >= beta) {
            if constexpr (kStats) {
              ++sacc.tt_cut;
              stats_->flush(depth, pattern_stage(b.count()), SearchStats::Cut, sacc);
            }
            return s;
          }
          if (e.flag == kUpper && s <= alpha) {
            if constexpr (kStats) {
              ++sacc.tt_cut;
              stats_->flush(depth, pattern_stage(b.count()), SearchStats::All, sacc);
            }
            return s;
          }
        }
      }
    }

    // Heuristic pruning stays off once depth covers every empty.
    const bool solving = depth >= 64 - b.count();

    // Exact upper bound from the opponent's stable discs.
    if (kUseStabilityCut && endgame_enabled_ && solving) {
      // Popcount gate avoids a fixpoint that cannot cut.
      const int opp = popcount(b.opponent);
      if (100 * (64 - 2 * opp) <= alpha) {
        const int ub = 100 * (64 - 2 * stable_count(b.opponent, b.player));
        if (ub <= alpha) {
          if constexpr (kStats)
            stats_->flush(depth, pattern_stage(b.count()), SearchStats::All, sacc);
          return ub; // fail-soft: a true upper bound on this node's value
        }
      }
    }

    // ProbCut is trained-eval only, non-PV, and never used in exact solves.
    if (Pat && kUseProbCut && probcut_enabled_ && selective_enabled_ && !solving && depth >= kProbCutMinDepth &&
        beta == alpha + 1 && beta < kScoreMax && alpha > -kScoreMax) {
      const int di = depth < kProbCutMaxFitDepth ? depth : kProbCutMaxFitDepth;
      const int sg = pattern_stage(b.count());
      // Deep nodes use the cheaper same-parity d-4 probe.
      const bool          g4  = probcut_gap4_ && depth >= kProbCutGap4MinDepth;
      const ProbCutFit   &f   = g4 ? kProbCutFit4[di] : (mpc_perstage_ ? kMpcFit[sg][di] : kProbCutFit[di]);
      const int           d2  = depth - (g4 ? 4 : 2);
      const std::uint64_t pc0 = kStats ? nodes_ : 0; // probe cost = nodes the probes spend
      if constexpr (kStats)
        ++sacc.pc_try;

      // Gate probes with optimistic/pessimistic static-eval bounds.
      const ProbCutFit &g     = kProbCutGateFit[d2 < 0 ? 0 : (d2 > kProbCutMaxFitDepth ? kProbCutMaxFitDepth : d2)];
      const float       gate  = kProbCutGateT * g.sigma;
      float             predf = 0.0f;
      if (probcut_gate_enabled_) {
        const int raw        = leaf_eval<Pat>(b, moves, ply, stm);
        const int correction = correction_history_cap_ > 0
                                       ? std::clamp(correction_history_.predict(b, sg, prev_move, prev2_move),
                                                    -correction_history_cap_, correction_history_cap_)
                                       : 0;
        predf                = g.a * static_cast<float>(raw + correction) + g.b;
      }

      // v_d >= beta  <=  a*v_{d-2} + b - t*sigma >= beta
      const int  hi     = static_cast<int>((static_cast<float>(beta) + probcut_t_ * f.sigma - f.b) / f.a);
      const bool try_hi = !probcut_gate_enabled_ || (predf + gate >= static_cast<float>(hi));
      if (try_hi && hi < kScoreMax && pvs<R, Pat>(b, d2, hi - 1, hi, ply, stm, prev_move, prev2_move) >= hi) {
        if constexpr (kStats) {
          ++sacc.pc_cut;
          sacc.pc_probe_nodes += nodes_ - pc0;
          stats_->flush(depth, sg, SearchStats::Cut, sacc);
        }
        return beta;
      }

      // v_d <= alpha  <=  a*v_{d-2} + b + t*sigma <= alpha
      const int           lo     = static_cast<int>((static_cast<float>(alpha) - probcut_t_ * f.sigma - f.b) / f.a);
      const bool          try_lo = !probcut_gate_enabled_ || (predf - gate <= static_cast<float>(lo));
      const std::uint64_t lo0    = kStats ? nodes_ : 0; // the lo-probe's own cost
      if constexpr (kStats)
        if (try_lo && lo > -kScoreMax)
          ++sacc.pc_lo_try;
      if (try_lo && lo > -kScoreMax && pvs<R, Pat>(b, d2, lo, lo + 1, ply, stm, prev_move, prev2_move) <= lo) {
        if constexpr (kStats) {
          ++sacc.pc_cut;
          ++sacc.pc_lo_cut;
          sacc.pc_lo_nodes += nodes_ - lo0;
          sacc.pc_probe_nodes += nodes_ - pc0;
          stats_->flush(depth, sg, SearchStats::All, sacc);
        }
        return alpha;
      }
      if constexpr (kStats)
        sacc.pc_lo_nodes += nodes_ - lo0;

      if constexpr (kStats)
        sacc.pc_probe_nodes += nodes_ - pc0;

      if (stopped_)
        return 0;

      // Reusing a failed probe's TT move added 8% nodes at depth 13.
    }

    // NMP lost 264 Elo; Othello cannot assume a tempo is beneficial.
    if (kUseNmp && nmp_enabled_ && selective_enabled_ && can_null && !solving && beta == alpha + 1 &&
        depth >= kNmpMinDepth && 64 - b.count() > kNmpMinEmpties) {
      const int stand_pat = leaf_eval<Pat>(b, moves, ply, stm);
      if (stand_pat >= beta) {
        const int      red          = kNmpBaseR + depth / kNmpDepthDiv;
        const int      nd           = depth - red - 1 < 1 ? 1 : depth - red - 1;
        const Board    passed       = b.passed(); // the heuristic pass: mover changes, no square does
        const Bitboard passed_moves = passed.moves();
        if constexpr (Pat)
          if (ply + 1 < kMaxPly)
            ps_[ply + 1] = ps_[ply]; // a pass leaves every feature unchanged
        move_stack_[ply + 1] = passed_moves;
        const int score      = -pvs<R, Pat>(passed, nd, -beta, -beta + 1, ply + 1, ~stm, prev_move, prev2_move,
                                            /*can_null=*/false);
        if (stopped_)
          return 0;
        if (score >= beta) {
          if constexpr (kStats) {
            ++sacc.fut_cut;
            stats_->flush(depth, pattern_stage(b.count()), SearchStats::Cut, sacc);
          }
          return score; // fail-soft lower bound
        }
      }
    }

    // Scout-only shallow futility, disabled while solving.
    if (kUsePruning && selective_enabled_ && !solving && depth <= 3 && beta == alpha + 1) {
      if constexpr (kStats)
        ++sacc.fut_try;
      const int stand_pat  = leaf_eval<Pat>(b, moves, ply, stm);
      const int fut_margin = depth <= 1 ? params_.fut1 : (depth == 2 ? params_.fut2 : params_.fut3);
      if (stand_pat + fut_margin <= alpha) {
        if constexpr (kStats) {
          ++sacc.fut_cut;
          stats_->flush(depth, pattern_stage(b.count()), SearchStats::All, sacc);
        }
        return stand_pat;
      }
    }

    struct BusyGuard {
      TranspositionTable *t = nullptr;
      std::uint64_t       k = 0;
      ~BusyGuard() {
        if (t)
          t->busy_leave(k);
      }
    } bguard;
    if (abdada_ && depth >= kAbdadaMinDepth) {
      tt_->busy_enter(key);
      bguard.t = tt_.get();
      bguard.k = key;
    }

    // Odd empty regions first during exact search.
    Bitboard odd_regions = 0;
    if (kUseParity && endgame_enabled_ && solving) {
      const Bitboard empties = ~(b.player | b.opponent);
      // Use exact regions only after the board fragments.
      if (eg_empties <= 12) {
        odd_regions = odd_empty_regions(empties);
      } else {
        for (unsigned q = 0; q < 4; ++q)
          if (popcount(empties & kQuadrant[q]) & 1)
            odd_regions |= empties & kQuadrant[q];
      }
    }

    const int  move_count = popcount(moves);
    ScoredMove list[36];
    int        n = 0;
    if (tt_move != NOMOVE) {
      int      tt_index = 0;
      Bitboard staged   = moves;
      while (staged) {
        const Square sq = pop_lsb(staged);
        ScoredMove  &sm = list[n];
        sm.sq           = sq;
        sm.child        = {};
        sm.child_moves  = 0;
        if (sq == tt_move) {
          tt_index = n;
          sm.child = b.play(sq);
          sm.score = 1 << 24;
          if (kUsePrefetch && kUseTT && tt_enabled_)
            tt_->prefetch(hash_board(sm.child.player, sm.child.opponent));
        } else if constexpr (kUseOrdering) {
          int s = kSquareValue[sq] * params_.sqv_mult;
          if constexpr (kUseKillers) {
            if (sq == killers_[ply][0])
              s += params_.killer0;
            else if (sq == killers_[ply][1])
              s += params_.killer1;
          }
          s += history_[sq] / params_.hist_div;
          if (prev_move < 64)
            s += continuation_history_[prev_move][sq] / params_.hist_div;
          if (prev2_move < 64)
            s += continuation_history_2_[prev2_move][sq] / params_.hist_div;
          if (odd_regions & square_bb(sq))
            s += params_.parity_bonus;
          sm.score = s;
        } else {
          sm.score = 0;
        }
        ++n;
      }
      if (tt_index != 0)
        std::swap(list[0], list[tt_index]);
    }

    if constexpr (kStats) {
      ++sacc.order_nodes;
      if (tt_move == NOMOVE)
        ++sacc.order_no_tt;
    }

    int    best      = -kInf;
    Square best_move = NOMOVE;
    bool   aborted   = false;
    Square searched_moves[36];
    int    searched_count = 0;

    const auto search_move = [&](int idx, int ord) -> int {
      if (kUseLMP && lmp_enabled_ && kUsePruning && selective_enabled_ && kUseOrdering && !solving &&
          beta == alpha + 1 && depth <= kLMPMaxDepth && ord >= kLMPCount[depth] && best > -kInf)
        return 2;

      const ScoredMove &sm = list[idx];
      if constexpr (kStats)
        ++sacc.moves_searched;

      const Bitboard child_moves =
              (depth >= kOrderMobilityMinDepth && sm.sq != tt_move) ? sm.child_moves : sm.child.moves();
      move_stack_[ply + 1] = child_moves;

      int ext = 0;
      if (kUseExtensions && selective_enabled_ && move_count == 1)
        ext = 1;

      int red = 0;
      if (kUsePruning && lmr_enabled_ && selective_enabled_ && !solving && !ext && kUseOrdering && depth >= 3 &&
          ord >= 3 && sm.sq != tt_move)
        red = lmr_calibrated_ ? lmr_reduction(depth, ord) : 1 + (ord >= 6 && depth >= 5 ? 1 : 0);
      if constexpr (kStats)
        if (red)
          ++sacc.lmr_try;

      if constexpr (Pat) {
        const Bitboard flipped = b.player ^ sm.child.opponent ^ square_bb(sm.sq);
        ps_[ply + 1]           = ps_[ply];
        ps_[ply + 1].update(sm.sq, flipped, stm);
      }

      searched_moves[searched_count++] = sm.sq;

      int s;
      if (ord == 0 || !kUsePVS) {
        s = -pvs<R, Pat>(sm.child, depth - 1 + ext, -beta, -alpha, ply + 1, ~stm, sm.sq, prev_move);
      } else {
        if constexpr (kStats)
          ++sacc.pvs_scout;
        s = -pvs<R, Pat>(sm.child, depth - 1 + ext - red, -alpha - 1, -alpha, ply + 1, ~stm, sm.sq, prev_move);
        if constexpr (kStats)
          if (red)
            stats_->lmr_sample(depth, ord, s > alpha);
        if (red && s > alpha) {
          if constexpr (kStats)
            ++sacc.lmr_re;
          s = -pvs<R, Pat>(sm.child, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, ~stm, sm.sq, prev_move);
        }
        if (s > alpha && s < beta) {
          if constexpr (kStats)
            ++sacc.pvs_re;
          s = -pvs<R, Pat>(sm.child, depth - 1 + ext, -beta, -alpha, ply + 1, ~stm, sm.sq, prev_move);
        }
      }
      if (stopped_) {
        aborted = true; // partial score: the node must not use or store it
        return 2;
      }

      if (s > best) {
        best      = s;
        best_move = sm.sq;
      }
      if (s > alpha)
        alpha = s;
      if (alpha >= beta) { // fail high
        if constexpr (kStats)
          sacc.cut_idx = ord; // ordinal of the cutoff move
        if constexpr (kStats) {
          if (tt_move == NOMOVE) {
            ++sacc.order_no_tt_fh;
            if (ord == 0)
              ++sacc.order_no_tt_fh_first;
          }
        }
        if constexpr (kUseKillers) {
          if (killers_[ply][0] != sm.sq) {
            killers_[ply][1] = killers_[ply][0];
            killers_[ply][0] = sm.sq;
          }
          const int bonus = depth * depth;
          for (int i = 0; i + 1 < searched_count; ++i) {
            update_history(history_, searched_moves[i], -bonus);
            if (prev_move < 64)
              update_history(continuation_history_[prev_move], searched_moves[i], -bonus);
            if (prev2_move < 64)
              update_history(continuation_history_2_[prev2_move], searched_moves[i], -bonus);
          }
          update_history(history_, sm.sq, bonus);
          if (prev_move < 64)
            update_history(continuation_history_[prev_move], sm.sq, bonus);
          if (prev2_move < 64)
            update_history(continuation_history_2_[prev2_move], sm.sq, bonus);
        }
        return 1;
      }
      return 0;
    };

    int ended = 0;
    if (tt_move != NOMOVE)
      ended = search_move(0, 0);

    const int first_unsearched = tt_move != NOMOVE ? 1 : 0;
    if (!ended) {
      if (tt_move != NOMOVE) {
        for (int i = 1; i < n; ++i) {
          ScoredMove &sm = list[i];
          sm.child       = b.play(sm.sq);
          if (kUsePrefetch && kUseTT && tt_enabled_)
            tt_->prefetch(hash_board(sm.child.player, sm.child.opponent));
          if constexpr (kUseOrdering) {
            if (depth >= kOrderMobilityMinDepth) {
              sm.child_moves = sm.child.moves();
              sm.score -= params_.mob_w * popcount(sm.child_moves);
            }
          }
        }
      } else {
        Bitboard m = moves;
        while (m) {
          const Square sq = pop_lsb(m);
          ScoredMove  &sm = list[n++];
          sm.sq           = sq;
          sm.child        = b.play(sq);
          sm.child_moves  = 0;
          if (kUsePrefetch && kUseTT && tt_enabled_)
            tt_->prefetch(hash_board(sm.child.player, sm.child.opponent));
          if constexpr (kUseOrdering) {
            const bool learned = depth >= kOrderMobilityMinDepth;
            if (depth >= kOrderMobilityMinDepth)
              sm.child_moves = sm.child.moves();
            const int replies = depth >= kOrderMobilityMinDepth ? popcount(sm.child_moves) : 0;
            int       s       = learned ? learned_order_score(b, sm.child, sq, replies, prev_move, prev2_move)
                                        : kSquareValue[sq] * params_.sqv_mult;
            if constexpr (kUseKillers) {
              if (sq == killers_[ply][0])
                s += params_.killer0;
              else if (sq == killers_[ply][1])
                s += params_.killer1;
            }
            s += history_[sq] / params_.hist_div;
            if (prev_move < 64)
              s += continuation_history_[prev_move][sq] / params_.hist_div;
            if (prev2_move < 64)
              s += continuation_history_2_[prev2_move][sq] / params_.hist_div;
            if (odd_regions & square_bb(sq))
              s += params_.parity_bonus;
            sm.score = s;
          } else {
            sm.score = 0;
          }
        }
      }
    }

    if (!ended) {
      // Defer only deep multi-move subtrees.
      const bool abdada_here = abdada_ && depth > kAbdadaMinDepth && n > 1;
      if (!abdada_here) {
        for (int i = first_unsearched; i < n; ++i) {
          if constexpr (kUseOrdering) {
            int pick = i;
            for (int j = i + 1; j < n; ++j)
              if (list[j].score > list[pick].score)
                pick = j;
            if (pick != i)
              std::swap(list[i], list[pick]);
          }
          if (search_move(i, i))
            break;
        }
      } else {
        // ABDADA revisits the list, so sort once.
        for (int i = first_unsearched + 1; i < n; ++i) {
          ScoredMove key_ = list[i];
          int        j    = i - 1;
          for (; j >= first_unsearched && list[j].score < key_.score; --j)
            list[j + 1] = list[j];
          list[j + 1] = key_;
        }
        bool deferred[36] = {};
        int  ord          = first_unsearched;
        for (int i = first_unsearched; i < n && !ended; ++i) {
          if (i > 0 && tt_->busy(hash_board(list[i].child.player, list[i].child.opponent))) {
            deferred[i] = true;
            continue;
          }
          ended = search_move(i, ord++);
        }
        for (int i = first_unsearched; i < n && !ended; ++i)
          if (deferred[i])
            ended = search_move(i, ord++);
      }
    }
    if (aborted)
      return 0;

    const std::uint8_t bound = best <= alpha_orig ? kUpper : (best >= beta ? kLower : kExact);
    if constexpr (Pat) {
      if (correction_history_cap_ > 0 && !solving && depth >= kProbCutMinDepth && is_pv && bound == kExact) {
        const int raw = leaf_eval<Pat>(b, moves, ply, stm);
        correction_history_.update(b, pattern_stage(b.count()), prev_move, prev2_move, best, raw, depth);
      }
    }

    if (kUseTT && tt_enabled_) {
      if (!stopped_) { // never poison the table with an aborted node's score
        tt_->store(key, std::clamp(best, -kScoreMax, kScoreMax), depth, bound,
                   best_move >= 0 && best_move < 64 ? static_cast<std::uint8_t>(best_move) : 255);
      }
    }

    if constexpr (kStats) {
      const SearchStats::NodeType t = is_pv ? SearchStats::PV : (best >= beta ? SearchStats::Cut : SearchStats::All);
      stats_->flush(depth, pattern_stage(b.count()), t, sacc);
    }
    return best;
  }

  template<Rule R, bool Pat>
  void Searcher::root_search(const Board &root, int depth, Color stm, int alpha, int beta, SearchResult &res) noexcept {
    const Bitboard moves = root.moves(); // caller guarantees non-zero
    ScoredMove     list[36];
    int            n = 0;
    Bitboard       m = moves;
    while (m) {
      const Square sq     = pop_lsb(m);
      list[n].sq          = sq;
      list[n].child       = root.play(sq);
      list[n].child_moves = list[n].child.moves();
      if (sq == res.best)
        list[n].score = 1 << 24;
      else
        list[n].score = kSquareValue[sq] * params_.sqv_mult - params_.mob_w * popcount(list[n].child_moves);
      ++n;
    }

    Square best_move  = NOMOVE;
    int    best_score = -kInf;
    for (int i = 0; i < n; ++i) {
      int pick = i;
      for (int j = i + 1; j < n; ++j)
        if (list[j].score > list[pick].score)
          pick = j;
      if (pick != i)
        std::swap(list[i], list[pick]);
      const ScoredMove &sm = list[i];
      move_stack_[1]       = sm.child_moves;

      if constexpr (Pat) {
        const Bitboard flipped = root.player ^ sm.child.opponent ^ square_bb(sm.sq);
        ps_[1]                 = ps_[0];
        ps_[1].update(sm.sq, flipped, stm);
      }

      int s;
      if (i == 0 || !kUsePVS) {
        s = -pvs<R, Pat>(sm.child, depth - 1, -beta, -alpha, 1, ~stm, sm.sq, NOMOVE);
      } else {
        s = -pvs<R, Pat>(sm.child, depth - 1, -alpha - 1, -alpha, 1, ~stm, sm.sq, NOMOVE);
        if (s > alpha && s < beta)
          s = -pvs<R, Pat>(sm.child, depth - 1, -beta, -alpha, 1, ~stm, sm.sq, NOMOVE);
      }
      if (stopped_)
        return; // leave res holding the last COMPLETED iteration

      if (s > best_score) {
        best_score = s;
        best_move  = sm.sq;
      }
      if (s > alpha)
        alpha = s;
      if (alpha >= beta)
        break; // fail-high: only reachable under a narrow aspiration window
    }
    res.best  = best_move;
    res.score = best_score;
  }

  std::string Searcher::pv_string(Board b, Rule rule, Square first, Color stm, int max_len) {
    // White PV moves are uppercase; Black moves are lowercase.
    const auto emit = [&stm](std::string &out, Square sq) {
      std::string s = square_to_string(sq);
      if (stm == Color::White)
        for (char &c: s)
          c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      out += ' ';
      out += s;
      stm = ~stm;
    };

    std::string out;
    // The root is not stored in TT, so its move is supplied by the caller.
    if (first == PASS) {
      emit(out, PASS);
      b = b.passed();
    } else if (first >= 0 && first < 64 && (b.moves() & square_bb(first))) {
      emit(out, first);
      b = b.play(first);
    } else {
      return out;
    }
    for (int i = 1; i < max_len; ++i) {
      const Bitboard moves = b.moves();
      if (moves == 0) {
        const Board passed = b.passed();
        if (rule != Rule::Othello || !passed.has_moves())
          break;
        b = passed;
        emit(out, PASS);
        continue;
      }
      const std::uint64_t k = hash_board(b.player, b.opponent);
      Entry               e;
      if (!tt_->probe(k, e))
        break; // probe already rejects a miss, a torn pair and an empty slot
      if (e.best >= 64 || !(moves & square_bb(e.best)))
        break; // stale/colliding entry: stop rather than emit an illegal move
      emit(out, static_cast<Square>(e.best));
      b = b.play(static_cast<Square>(e.best));
    }
    return out;
  }

  namespace {
    // Soft steers iteration starts; hard aborts an overrun.
    struct TimeBudget {
      double soft_ms, hard_ms;
    };
    constexpr double kMoveOverheadMs = 15.0; // protocol + scheduling latency, charged to us

    constexpr int    kTmAdjustMinDepth = 6; // below this, iterations are noise
    constexpr int    kTmDropCd         = 120; // a fall this size counts as instability
    constexpr double kTmBump           = 1.5; // extension per unstable iteration
    constexpr double kTmDecay          = 0.85; // shrink per calm iteration
    constexpr double kTmMaxFactor      = 2.5;
    constexpr double kTmMinFactor      = 0.6;

    [[nodiscard]] TimeBudget allocate_time(double clk, double inc, int empties) noexcept {
      clk = clk - kMoveOverheadMs;
      if (clk < 1.0)
        return {1.0, 1.0};
      const int my_moves = empties > 1 ? (empties + 1) / 2 : 1;
      double    soft     = clk / my_moves + 0.9 * inc;
      if (empties >= 44)
        soft *= 0.6; // opening: cheap plies, bank the increment
      else if (empties >= 20 && empties <= 30)
        soft *= 1.8; // solve window: one deep think can end the game exactly
      double hard = std::min(clk * 0.30, soft * 4.0);
      soft        = std::min(soft, hard);
      return {std::max(soft, 1.0), std::max(hard, 1.0)};
    }
  } // namespace

  SearchResult Searcher::search(const Board &root, const SearchLimits &limits, Rule rule, Color stm,
                                std::ostream &info) {
    new_search(limits);
    double tm_soft_base = 0.0, tm_factor = 1.0; // adaptive-TM state; see kTmBump
    if (limits.time_ms > 0.0) { // clock mode: the engine allocates its own budget
      const TimeBudget tb = allocate_time(limits.time_ms, limits.inc_ms, 64 - root.count());
      deadline_ms_        = start_ms_ + tb.hard_ms;
      soft_ms_            = start_ms_ + tb.soft_ms;
      tm_soft_base        = tb.soft_ms;
    }
    if (pat_on_)
      ps_[0].set(root, stm); // the only from-scratch build; everything below is incremental
    SearchResult res;

    const Bitboard moves = root.moves();
    move_stack_[0]       = moves;
    if (moves == 0) {
      const Board    passed       = root.passed();
      const Bitboard passed_moves = passed.moves();
      if (rule == Rule::Othello && passed_moves != 0) {
        res.best = PASS; // forced: no choice to search for
        if (pat_on_)
          ps_[1] = ps_[0]; // a pass changes the mover, not the squares
        move_stack_[1] = passed_moves;
        const int sc   = rule == Rule::Othello ? (pat_on_ ? pvs<Rule::Othello, true>(passed, 4, -kInf, kInf, 1, ~stm)
                                                          : pvs<Rule::Othello, false>(passed, 4, -kInf, kInf, 1, ~stm))
                                               : (pat_on_ ? pvs<Rule::Reversi, true>(passed, 4, -kInf, kInf, 1, ~stm)
                                                          : pvs<Rule::Reversi, false>(passed, 4, -kInf, kInf, 1, ~stm));
        res.score      = -sc;
        return res;
      }
      res.best  = NOMOVE; // game over
      res.score = terminal_score(root);
      return res;
    }

    const int empties   = 64 - root.count();
    const int max_depth = limits.depth > 0 ? std::min(limits.depth, empties) : empties;

    int  score_hist[2] = {0, 0}; // score two and one iterations ago (index by parity of d)
    bool have_hist[2]  = {false, false};
    // Compare adaptive-TM signals at the same depth parity.
    Square    tm_prev_best[2]  = {NOMOVE, NOMOVE};
    int       tm_prev_score[2] = {0, 0};
    bool      tm_have_prev[2]  = {false, false};
    const int first_depth      = depth_offset_ < max_depth ? 1 + depth_offset_ : 1;
    for (int d = first_depth; d <= max_depth; ++d) {
      seldepth_       = 0; // per-iteration, like the UCI convention
      SearchResult it = res; // carry the previous best in for root ordering

      const bool aspire = kUseAspiration && aspiration_enabled_ && d >= kAspMinDepth && have_hist[d & 1];
      const int  centre = score_hist[d & 1];
      int        window = kAspWindow;
      int        alpha  = aspire ? centre - window : -kInf;
      int        beta   = aspire ? centre + window : kInf;
      // Timed exact solves use a narrow WLD window.
      bool wld = wld_enabled_ && (limits.movetime_ms > 0.0 || limits.time_ms > 0.0) && d >= empties;
      if (wld) {
        alpha = -100;
        beta  = 100;
      }
      for (;;) {
        if (rule == Rule::Othello) {
          if (pat_on_)
            root_search<Rule::Othello, true>(root, d, stm, alpha, beta, it);
          else
            root_search<Rule::Othello, false>(root, d, stm, alpha, beta, it);
        } else {
          if (pat_on_)
            root_search<Rule::Reversi, true>(root, d, stm, alpha, beta, it);
          else
            root_search<Rule::Reversi, false>(root, d, stm, alpha, beta, it);
        }
        if (stopped_)
          break;
        if (wld) {
          // Re-search WLD losses for a usable root margin.
          if (it.score <= -100 && !stopped_) {
            wld   = false;
            alpha = -kInf;
            beta  = kInf;
            continue;
          }
          break;
        }
        if (it.score <= alpha && alpha > -kInf) {
          window *= 3;
          alpha = it.score - window <= -kScoreMax ? -kInf : it.score - window;
          continue; // failed low: relax the lower bound and retry
        }
        if (it.score >= beta && beta < kInf) {
          window *= 3;
          beta = it.score + window >= kScoreMax ? kInf : it.score + window;
          continue; // failed high
        }
        break; // strictly inside -> exact
      }

      if (stopped_ && d > 1)
        break;
      score_hist[d & 1] = it.score; // remember this parity's score for d+2
      have_hist[d & 1]  = true;

      res          = it;
      res.depth    = d;
      res.seldepth = seldepth_;
      res.nodes    = nodes_;
      res.exact    = (d >= empties);

      const double        dt       = now_ms() - start_ms_;
      const std::uint64_t nps      = dt > 0.0 ? static_cast<std::uint64_t>(nodes_ / (dt / 1000.0)) : 0;
      const int           hashfull = tt_->hashfull();
      info << "info depth " << d << " seldepth " << res.seldepth << " score cd " << res.score << " nodes " << nodes_
           << " nps " << nps << " hashfull " << hashfull << " time " << static_cast<std::uint64_t>(dt) << " pv"
           << pv_string(root, rule, res.best, stm, d) << '\n';
      info.flush();

      if (res.exact || stopped_)
        break;
      if (tm_adaptive_ && tm_soft_base > 0.0 && d >= kTmAdjustMinDepth && tm_have_prev[d & 1]) {
        const bool unstable = (res.best != tm_prev_best[d & 1]) || (res.score <= tm_prev_score[d & 1] - kTmDropCd);
        tm_factor =
                unstable ? std::min(tm_factor * kTmBump, kTmMaxFactor) : std::max(tm_factor * kTmDecay, kTmMinFactor);
        soft_ms_ = start_ms_ + std::min(deadline_ms_ - start_ms_, tm_soft_base * tm_factor);
      }
      tm_prev_best[d & 1]  = res.best;
      tm_prev_score[d & 1] = res.score;
      tm_have_prev[d & 1]  = true;
      const double steer   = soft_ms_ > 0.0 ? soft_ms_ : deadline_ms_;
      if (steer > 0.0 && dt > 0.5 * (steer - start_ms_))
        break;
    }
    if constexpr (kStats)
      stats_->total_nodes = nodes_; // for the probe-cost ratio (same unit: incl. leaves)
    return res;
  }

  bool search_selftest() noexcept {
    {
      CorrectionHistory h;
      const Board       b     = Board::start();
      const int         stage = pattern_stage(b.count());
      if (h.predict(b, stage, 19, 26) != 0)
        return false;
      h.update(b, stage, 19, 26, 600, 100, 8);
      const int learned = h.predict(b, stage, 19, 26);
      if (learned <= 0 || h.predict(b, stage, 20, 26) >= learned)
        return false;
      for (int i = 0; i < 256; ++i)
        h.update(b, stage, 19, 26, 5000, 0, 16);
      if (h.predict(b, stage, 19, 26) > 400)
        return false;
      h.clear();
      if (h.predict(b, stage, 19, 26) != 0)
        return false;
    }

    {
      const Bitboard odd_three = square_bb(0) | square_bb(1) | square_bb(2);
      const Bitboard even_two  = square_bb(62) | square_bb(63);
      if (odd_empty_regions(odd_three | even_two) != odd_three)
        return false;
      const Bitboard no_wrap = square_bb(7) | square_bb(8);
      if (odd_empty_regions(no_wrap) != no_wrap)
        return false;
    }

    std::uint64_t s   = 0x9E3779B97F4A7C15ULL;
    const auto    rnd = [&s]() noexcept {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      return s;
    };

    Searcher           sr(16);
    std::ostringstream sink;

    // Fixed-depth oracle with TT and selectivity disabled.
    sr.set_tt_enabled(false);
    sr.set_selective_enabled(false); // group B changes the fixed-depth value by design
    for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
      Board b = Board::start();
      for (int step = 0; step < 220; ++step) {
        Bitboard m = b.moves(); // sample real positions via random playout
        if (m == 0) {
          const Board p = b.passed();
          b             = p.has_moves() ? p : Board::start();
          continue;
        }
        for (int d = 1; d <= 4; ++d) {
          const int          want = rule == Rule::Othello ? ref_negamax<Rule::Othello>(b, d, Color::Black)
                                                          : ref_negamax<Rule::Reversi>(b, d, Color::Black);
          const SearchResult got  = sr.search(b, SearchLimits{d, 0, 0.0}, rule, Color::Black, sink);
          if (got.score != want)
            return false; // an "optimization" changed the score: that is a bug
          if (got.best == NOMOVE || (got.best != PASS && !(m & square_bb(got.best))))
            return false; // the move played must always be legal
        }
        unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
        while (k-- > 0)
          m &= m - 1;
        b = b.play(lsb(m));
      }
    }

    // Compare incremental pattern state against scratch evaluation.
    {
      PatternWeights saved = pattern_weights();
      pattern_weights().reset_zero();
      for (std::size_t i = 0; i < pattern_weights().size(); ++i)
        pattern_weights().data()[i] = static_cast<std::int16_t>(static_cast<int>(rnd() % 41) - 20);

      bool ok = true;
      for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
        Board b = Board::start();
        for (int step = 0; step < 60 && ok; ++step) {
          Bitboard m = b.moves();
          if (m == 0) {
            const Board p = b.passed();
            b             = p.has_moves() ? p : Board::start();
            continue;
          }
          const Color stm = (step & 1) ? Color::White : Color::Black;
          for (int d = 1; d <= 3; ++d) {
            const int          want = rule == Rule::Othello ? ref_negamax<Rule::Othello>(b, d, stm)
                                                            : ref_negamax<Rule::Reversi>(b, d, stm);
            const SearchResult got  = sr.search(b, SearchLimits{d, 0, 0.0}, rule, stm, sink);
            if (got.score != want) {
              ok = false;
              break;
            }
          }
          unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
          while (k-- > 0)
            m &= m - 1;
          b = b.play(lsb(m));
        }
      }
      pattern_weights() = saved; // never leave test weights behind
      if (!ok)
        return false;
    }

    // Exact solves must remain stable with a warm TT.
    sr.set_tt_enabled(true);
    sr.set_selective_enabled(true); // extensions/pruning must NOT corrupt an exact solve
    for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
      sr.clear(); // the two rules value stuck positions differently: never share
      for (int trial = 0; trial < 12; ++trial) {
        Board b = Board::start();
        for (int i = 0; i < 52 && 64 - b.count() > 9; ++i) { // play down to ~9 empties
          Bitboard m = b.moves();
          if (m == 0) {
            const Board p = b.passed();
            if (!p.has_moves())
              break;
            b = p;
            continue;
          }
          unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
          while (k-- > 0)
            m &= m - 1;
          b = b.play(lsb(m));
        }
        const int empties = 64 - b.count();
        if (b.moves() == 0 || empties > 12)
          continue;
        const int          want = rule == Rule::Othello ? ref_negamax<Rule::Othello>(b, empties, Color::Black)
                                                        : ref_negamax<Rule::Reversi>(b, empties, Color::Black);
        const SearchResult got  = sr.search(b, SearchLimits{empties, 0, 0.0}, rule, Color::Black, sink);
        if (got.score != want || !got.exact)
          return false;
        const SearchResult again = sr.search(b, SearchLimits{empties, 0, 0.0}, rule, Color::Black, sink);
        if (again.score != want)
          return false;
      }
    }
    return true;
  }

} // namespace islay
