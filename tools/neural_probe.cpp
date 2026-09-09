// Batch parity/latency driver. Reads up to 128 mover-relative bitboard pairs.
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "neural.hpp"

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 4)
      throw std::invalid_argument("usage: islay_nn_probe model.onnx [repeats=1] [all|off]");
    std::size_t       parsed       = 0;
    const int         repeats      = argc > 2 ? std::stoi(argv[2], &parsed) : 1;
    const std::string optimization = argc > 3 ? argv[3] : "all";
    if ((argc > 2 && parsed != std::string(argv[2]).size()) || repeats < 1 || repeats > 10000 ||
        (optimization != "all" && optimization != "off"))
      throw std::invalid_argument("invalid repeats/optimization");
    islay::NeuralEvaluator    evaluator(argv[1], optimization == "all");
    std::vector<islay::Board> boards;
    islay::Board              board;
    while (std::cin >> board.player) {
      if (!(std::cin >> board.opponent))
        throw std::invalid_argument("incomplete bitboard pair");
      boards.push_back(board);
      if (boards.size() > islay::kNeuralMaxBatch)
        throw std::invalid_argument("batch exceeds 128");
    }
    if (!std::cin.eof() || boards.empty())
      throw std::invalid_argument("expected bitboard pairs");
    std::vector<islay::Evaluation> outputs(boards.size());
    std::stop_source               cancelled;
    cancelled.request_stop();
    outputs[0].value = 123;
    evaluator.evaluate(boards, outputs, cancelled.get_token());
    if (outputs[0].value != 123)
      throw std::runtime_error("cancelled evaluation modified output");
    evaluator.evaluate(boards, outputs, {});
    std::vector<double> timings;
    for (int i = 0; i < repeats; ++i) {
      const auto start = std::chrono::steady_clock::now();
      evaluator.evaluate(boards, outputs, {});
      timings.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double median  = (timings[(repeats - 1) / 2] + timings[repeats / 2]) / 2;
    double       rss_mib = 0;
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
      rss_mib = usage.ru_maxrss / (1024.0 * 1024.0);
#else
      rss_mib = usage.ru_maxrss / 1024.0;
#endif
    }
#endif
    std::cout << std::setprecision(9) << "{\"peak_rss_mib\":" << rss_mib << ",\"median_us\":" << median
              << ",\"p95_us\":" << timings[(repeats - 1) * 95 / 100] << ",\"outputs\":[";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (i)
        std::cout << ',';
      std::cout << '[';
      for (float logit: outputs[i].logits)
        std::cout << logit << ',';
      std::cout << outputs[i].value << ']';
    }
    std::cout << "]}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
