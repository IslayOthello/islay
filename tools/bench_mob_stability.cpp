#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "board.hpp"
#include "stability.hpp"

namespace {
  using namespace islay;
  using stability_detail::StableBases;

  struct Sample {
    Board    board;
    Color    stm;
    Bitboard moves;
  };

  struct Counts {
    int black_mob, white_mob;
    int black_stab, white_stab;
    int parity;
    int black_front, white_front;

    bool operator==(const Counts &) const noexcept = default;
  };

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard reference_spread_h(Bitboard b) noexcept {
    constexpr Bitboard kNoA  = 0xFEFEFEFEFEFEFEFEULL;
    constexpr Bitboard kNoAB = 0xFCFCFCFCFCFCFCFCULL;
    constexpr Bitboard kNoAD = 0xF0F0F0F0F0F0F0F0ULL;
    constexpr Bitboard kNoH  = 0x7F7F7F7F7F7F7F7FULL;
    constexpr Bitboard kNoGH = 0x3F3F3F3F3F3F3F3FULL;
    constexpr Bitboard kNoEH = 0x0F0F0F0F0F0F0F0FULL;
    b |= ((b << 1) & kNoA) | ((b >> 1) & kNoH);
    b |= ((b << 2) & kNoAB) | ((b >> 2) & kNoGH);
    b |= ((b << 4) & kNoAD) | ((b >> 4) & kNoEH);
    return b;
  }

  [[nodiscard]] ISLAY_FORCEINLINE Bitboard reference_spread_v(Bitboard b) noexcept {
    b |= (b << 8) | (b >> 8);
    b |= (b << 16) | (b >> 16);
    b |= (b << 32) | (b >> 32);
    return b;
  }

  __attribute__((noinline)) StableCounts reference_stable_counts(Bitboard p, Bitboard o) noexcept {
    const Bitboard    empty = ~(p | o);
    const StableBases base{~reference_spread_h(empty) | kFileA | kFileH, ~reference_spread_v(empty) | kRank1 | kRank8,
                           ~stability_detail::spread_d1(empty) | kEdges, ~stability_detail::spread_d2(empty) | kEdges};
    return {popcount(stability_detail::stable_bits(p, base)), popcount(stability_detail::stable_bits(o, base))};
  }

  __attribute__((noinline)) StableCounts optimized_stable_counts(Bitboard p, Bitboard o) noexcept {
    return stable_counts(p, o);
  }

  [[nodiscard]] ISLAY_FORCEINLINE int reference_parity(Bitboard empties) noexcept {
    int odd = 0;
    for (Bitboard quadrant: kQuadrant)
      odd += popcount(empties & quadrant) & 1;
    return (popcount(empties) & 1) + 2 * odd;
  }

  [[nodiscard]] ISLAY_FORCEINLINE int optimized_parity(Bitboard empties) noexcept {
    Bitboard folded = empties ^ (empties >> 8) ^ (empties >> 16) ^ (empties >> 24);
    folded ^= folded >> 2;
    folded ^= folded >> 1;
    const int odd = popcount(folded & 0x0000001100000011ULL);
    return (odd & 1) + 2 * odd;
  }

  template<StableCounts (*StableFn)(Bitboard, Bitboard), int (*ParityFn)(Bitboard)>
  __attribute__((noinline)) Counts make_counts(const Sample &s) noexcept {
    const Board       &b       = s.board;
    const int          my_mob  = popcount(s.moves);
    const int          opp_mob = popcount(get_moves(b.opponent, b.player));
    const StableCounts stable  = StableFn(b.player, b.opponent);
    const bool         black   = s.stm == Color::Black;
    Counts             out;
    out.black_mob            = black ? my_mob : opp_mob;
    out.white_mob            = black ? opp_mob : my_mob;
    out.black_stab           = black ? stable.player : stable.opponent;
    out.white_stab           = black ? stable.opponent : stable.player;
    const Bitboard empty     = ~(b.player | b.opponent);
    out.parity               = (black ? 0 : 10) + ParityFn(empty);
    const Bitboard near      = dilate8(empty);
    const int      my_front  = popcount(b.player & near);
    const int      opp_front = popcount(b.opponent & near);
    out.black_front          = black ? my_front : opp_front;
    out.white_front          = black ? opp_front : my_front;
    return out;
  }

  __attribute__((noinline)) Counts reference_mob_counts(const Sample &s) noexcept {
    return make_counts<reference_stable_counts, reference_parity>(s);
  }

  __attribute__((noinline)) Counts optimized_mob_counts(const Sample &s) noexcept {
    return make_counts<optimized_stable_counts, optimized_parity>(s);
  }

  std::uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

  std::uint64_t random64() noexcept {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
  }

  std::vector<Sample> make_samples(std::size_t count) {
    std::vector<Sample> out;
    out.reserve(count);
    while (out.size() < count) {
      Board b      = Board::start();
      Color stm    = Color::Black;
      int   passes = 0;
      while (passes < 2 && out.size() < count) {
        const Bitboard moves = b.moves();
        if (moves == 0) {
          b   = b.passed();
          stm = ~stm;
          ++passes;
          continue;
        }
        passes = 0;
        out.push_back({b, stm, moves});
        Bitboard pick = moves;
        int      skip = static_cast<int>(random64() % static_cast<unsigned>(popcount(moves)));
        while (skip--)
          pop_lsb(pick);
        b   = b.play(pop_lsb(pick));
        stm = ~stm;
      }
    }
    return out;
  }

  std::uint64_t hash_counts(const Counts &c) noexcept {
    return static_cast<unsigned>(c.black_mob) * 3 + static_cast<unsigned>(c.white_mob) * 5 +
           static_cast<unsigned>(c.black_stab) * 7 + static_cast<unsigned>(c.white_stab) * 11 +
           static_cast<unsigned>(c.parity) * 13 + static_cast<unsigned>(c.black_front) * 17 +
           static_cast<unsigned>(c.white_front) * 19;
  }

  template<class Fn>
  double bench(const std::vector<Sample> &samples, Fn fn, std::uint64_t &checksum) {
    constexpr int repeats = 256;
    const auto    begin   = std::chrono::steady_clock::now();
    std::uint64_t local   = 0;
    for (int repeat = 0; repeat < repeats; ++repeat)
      for (const Sample &sample: samples)
        local += fn(sample);
    const auto end = std::chrono::steady_clock::now();
    checksum += local;
    return std::chrono::duration<double, std::nano>(end - begin).count() / (repeats * samples.size());
  }

  template<std::size_t N>
  double median(std::array<double, N> values) {
    std::sort(values.begin(), values.end());
    return values[N / 2];
  }
} // namespace

int main() {
  const std::vector<Sample> samples = make_samples(1U << 15);
  for (const Sample &sample: samples)
    if (!(reference_mob_counts(sample) == optimized_mob_counts(sample))) {
      std::puts("legal-position mismatch");
      return EXIT_FAILURE;
    }

  for (int i = 0; i < 1000000; ++i) {
    const Bitboard     p         = random64();
    const Bitboard     o         = random64() & ~p;
    const StableCounts reference = reference_stable_counts(p, o);
    const StableCounts optimized = optimized_stable_counts(p, o);
    if (reference.player != optimized.player || reference.opponent != optimized.opponent ||
        reference_parity(~(p | o)) != optimized_parity(~(p | o))) {
      std::puts("random-position mismatch");
      return EXIT_FAILURE;
    }
  }

  std::array<double, 9> stable_reference{}, stable_optimized{}, mob_reference{}, mob_optimized{};
  std::uint64_t         checksum = 0;
  for (std::size_t round = 0; round < stable_reference.size(); ++round) {
    const auto stable_ref = [](const Sample &s) {
      const StableCounts c = reference_stable_counts(s.board.player, s.board.opponent);
      return static_cast<std::uint64_t>(c.player + 67 * c.opponent);
    };
    const auto stable_opt = [](const Sample &s) {
      const StableCounts c = optimized_stable_counts(s.board.player, s.board.opponent);
      return static_cast<std::uint64_t>(c.player + 67 * c.opponent);
    };
    const auto mob_ref = [](const Sample &s) { return hash_counts(reference_mob_counts(s)); };
    const auto mob_opt = [](const Sample &s) { return hash_counts(optimized_mob_counts(s)); };
    if ((round & 1) == 0) {
      stable_reference[round] = bench(samples, stable_ref, checksum);
      stable_optimized[round] = bench(samples, stable_opt, checksum);
      mob_reference[round]    = bench(samples, mob_ref, checksum);
      mob_optimized[round]    = bench(samples, mob_opt, checksum);
    } else {
      mob_optimized[round]    = bench(samples, mob_opt, checksum);
      mob_reference[round]    = bench(samples, mob_ref, checksum);
      stable_optimized[round] = bench(samples, stable_opt, checksum);
      stable_reference[round] = bench(samples, stable_ref, checksum);
    }
  }

  const double stable_ref = median(stable_reference);
  const double stable_opt = median(stable_optimized);
  const double mob_ref    = median(mob_reference);
  const double mob_opt    = median(mob_optimized);
  std::printf("stable_counts  %8.3f -> %8.3f ns  %+6.2f%%\n", stable_ref, stable_opt,
              100.0 * (stable_opt / stable_ref - 1.0));
  std::printf("mob_counts     %8.3f -> %8.3f ns  %+6.2f%%\n", mob_ref, mob_opt, 100.0 * (mob_opt / mob_ref - 1.0));
  std::printf("checksum %llu\n", static_cast<unsigned long long>(checksum));
}
