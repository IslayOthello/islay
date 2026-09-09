// Single-thread perft measurement driver. Build instructions: tools/perft_study.py --help.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "board.hpp"
#include "perft.hpp"

using namespace islay;
using Clock = std::chrono::steady_clock;

// Match UCI's root divide, including pass and depth-zero semantics.
std::uint64_t divide(const Board &board, int depth, Rule rule, PerftTT &tt, bool cached) {
  if (depth == 0)
    return 1;
  const auto child_count = [&](const Board &child) {
    if (depth == 1)
      return std::uint64_t{1};
    return cached ? perft_cached(child, depth - 1, tt, rule) : perft(child, depth - 1, rule);
  };
  Bitboard moves = board.moves();
  if (!moves)
    return rule == Rule::Othello && board.passed().has_moves() ? child_count(board.passed()) : 0;
  std::uint64_t nodes = 0;
  while (moves)
    nodes += child_count(board.play(pop_lsb(moves)));
  return nodes;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--verify") {
    if (!movegen_selftest()) {
      std::cerr << "movegen self-test failed\n";
      return 1;
    }
    std::cout << "movegen self-test passed\n";
    PerftTT       tt(1);
    std::uint64_t player, opponent, expected;
    int           depth;
    std::string   rule_name;
    std::size_t   checks = 0;
    while (std::cin >> player >> opponent >> depth >> rule_name >> expected) {
      const Board board{player, opponent};
      const Rule  rule = rule_name == "Othello" ? Rule::Othello : Rule::Reversi;
      tt.clear();
      for (int symmetry = 0; symmetry < 8; ++symmetry) {
        for (bool cached: {false, true}) {
          const auto got = divide(board.symmetry(symmetry), depth, rule, tt, cached);
          if (got != expected) {
            std::cerr << "count mismatch: " << player << ' ' << opponent << ' ' << depth << ' ' << rule_name
                      << " expected " << expected << " got " << got << '\n';
            return 1;
          }
          ++checks;
        }
      }
    }
    std::cout << "verified " << checks << " counts\n";
    return std::cin.eof() ? 0 : 1;
  }
  if (argc != 9) {
    std::cerr << "usage: perft_bench player opponent depth rule mode hash_mib target_ms batch\n";
    return 2;
  }
  const Board       board{std::stoull(argv[1]), std::stoull(argv[2])};
  const int         depth     = std::stoi(argv[3]);
  const Rule        rule      = std::string(argv[4]) == "Othello" ? Rule::Othello : Rule::Reversi;
  const std::string mode      = argv[5];
  const int         hash_mib  = std::stoi(argv[6]);
  const double      target_ms = std::stod(argv[7]);
  const int         batch     = std::stoi(argv[8]);
  if (depth < 0 || depth > 64 || hash_mib < 1 || batch < 1 || (mode != "cold" && mode != "warm" && mode != "nocache")) {
    std::cerr << "invalid benchmark arguments\n";
    return 2;
  }
  PerftTT             tt(hash_mib);
  const bool          cached     = mode != "nocache";
  const std::uint64_t expected   = divide(board, depth, rule, tt, cached);
  std::uint64_t       iterations = 0, checksum = 0;
  double              elapsed_ms = 0;
  do {
    // Exclude allocation/clear from traversal time; cold means an empty TT each traversal.
    if (mode == "cold")
      tt.clear();
    const auto start = Clock::now();
    const int  count = mode == "warm" ? batch : 1;
    for (int i = 0; i < count; ++i) {
      const auto nodes = divide(board, depth, rule, tt, cached);
      if (nodes != expected) {
        std::cerr << "unstable node count\n";
        return 1;
      }
      checksum += nodes;
      ++iterations;
    }
    elapsed_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  } while (elapsed_ms < target_ms);
  std::cout.precision(12);
  std::cout << "{\"nodes\":" << expected << ",\"iterations\":" << iterations << ",\"elapsed_ms\":" << elapsed_ms
            << ",\"nps\":" << static_cast<double>(expected) * iterations * 1000.0 / elapsed_ms
            << ",\"ns_per_call\":" << elapsed_ms * 1e6 / iterations << ",\"checksum\":" << checksum << "}\n";
}
