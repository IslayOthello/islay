// Single-thread PUCT microbenchmark; the synthetic evaluator is not a trained network.
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mcts.hpp"

using namespace islay;

class BenchEvaluator final : public Evaluator {
public:
  bool synthetic = false;

  void evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token) override {
    for (std::size_t i = 0; i < boards.size(); ++i) {
      outputs[i] = {};
      if (!synthetic)
        continue;
      auto hash = boards[i].player ^ std::rotl(boards[i].opponent, 23);
      for (auto &logit: outputs[i].logits) {
        hash ^= hash << 13;
        hash ^= hash >> 7;
        hash ^= hash << 17;
        logit = static_cast<float>(hash & 255) / 128.0f - 1.0f;
      }
      outputs[i].value = static_cast<float>((hash >> 8) & 255) / 128.0f - 1.0f;
    }
  }
};

std::uint64_t signature(const MctsResult &result) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto    mix  = [&](std::uint64_t value) { hash = (hash ^ value) * 1099511628211ULL; };
  mix(result.simulations);
  mix(result.evaluations);
  mix(result.nodes);
  mix(result.best_move);
  mix(std::bit_cast<std::uint64_t>(result.value));
  for (int i = 0; i < result.action_count; ++i) {
    mix(result.actions[i].action);
    mix(result.actions[i].visits);
    mix(std::bit_cast<std::uint32_t>(result.actions[i].prior));
    mix(std::bit_cast<std::uint64_t>(result.actions[i].value));
  }
  for (int i = 0; i < result.pv_length; ++i)
    mix(result.pv[i]);
  return hash;
}

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--verify") {
      const bool ok = game_selftest() && mcts_selftest();
      std::cout << (ok ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
      return ok ? 0 : 1;
    }
    if (argc != 6) {
      std::cerr << "usage: mcts_bench player opponent simulations milliseconds uniform|synthetic\n";
      return 2;
    }
    const Board board{std::stoull(argv[1]), std::stoull(argv[2])};
    MctsLimits  limits;
    limits.simulations             = std::stoull(argv[3]);
    const double      milliseconds = std::stod(argv[4]);
    const std::string mode         = argv[5];
    if (!limits.simulations || !std::isfinite(milliseconds) || milliseconds < 0 ||
        (mode != "uniform" && mode != "synthetic"))
      throw std::invalid_argument("invalid benchmark arguments");
    BenchEvaluator evaluator;
    evaluator.synthetic = mode == "synthetic";
    const auto expected = mcts_search(board, Rule::Othello, evaluator, limits);
    if (expected.reason != MctsStop::SimulationLimit || expected.simulations != limits.simulations)
      throw std::runtime_error("workload did not complete its simulation budget");
    const auto    checksum   = signature(expected);
    double        elapsed_us = 0;
    std::uint64_t iterations = 0;
    do {
      const auto start  = MctsClock::now();
      const auto result = mcts_search(board, Rule::Othello, evaluator, limits);
      elapsed_us += std::chrono::duration<double, std::micro>(MctsClock::now() - start).count();
      if (signature(result) != checksum || result.reason != MctsStop::SimulationLimit)
        throw std::runtime_error("unstable tree statistics");
      ++iterations;
    } while (elapsed_us < milliseconds * 1000);
    std::cout.precision(12);
    std::cout << "{\"simulations\":" << expected.simulations << ",\"evaluations\":" << expected.evaluations
              << ",\"nodes\":" << expected.nodes << ",\"tree_bytes\":" << expected.tree_bytes
              << ",\"signature\":" << checksum << ",\"iterations\":" << iterations << ",\"elapsed_us\":" << elapsed_us
              << ",\"nps\":" << static_cast<double>(expected.simulations) * iterations * 1e6 / elapsed_us << "}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
