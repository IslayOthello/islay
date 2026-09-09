#include <charconv>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "neural.hpp"
#include "selfplay.hpp"

namespace {
  std::uint64_t integer(const std::string &s, std::uint64_t minimum, std::uint64_t maximum) {
    std::uint64_t value;
    const auto [p, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error != std::errc{} || p != s.data() + s.size() || value < minimum || value > maximum)
      throw std::invalid_argument("invalid integer: " + s);
    return value;
  }
  double real(const std::string &s) {
    std::size_t end;
    const auto  value = std::stod(s, &end);
    if (end != s.size())
      throw std::invalid_argument("invalid real: " + s);
    return value;
  }
  template<class T>
  void put(std::ostream &out, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i)
      out.put(static_cast<char>((value >> (8 * i)) & 255));
  }
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--identity") {
      std::cout << "{\"backend\":\"" << islay::neural_backend() << "\",\"runtime\":\""
                << islay::neural_runtime_version() << "\"}\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--selftest") {
      if (!islay::selfplay_selftest())
        throw std::runtime_error("self-play selftest failed");
      std::cout << "SELFPLAY TESTS PASSED\n";
      return 0;
    }
    std::map<std::string, std::string> options{
            {"--games", "1"},         {"--start", "0"},    {"--seed", "20260909"},
            {"--simulations", "128"}, {"--tree-mib", "8"}, {"--cache-mib", "16"},
            {"--epsilon", "0.25"},    {"--alpha", "0.3"},  {"--temperature-plies", "16"},
            {"--model", ""},          {"--verify", ""}};
    std::map<std::string, bool> seen;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc || !options.contains(argv[i]) || seen[argv[i]])
        throw std::invalid_argument("expected unique --option value pairs; use training/selfplay.py for generation");
      seen[argv[i]]    = true;
      options[argv[i]] = argv[i + 1];
    }
    islay::SelfplayConfig config;
    config.seed              = integer(options["--seed"], 0, UINT64_MAX);
    config.simulations       = integer(options["--simulations"], 1, 1000000);
    config.tree_bytes        = integer(options["--tree-mib"], 1, 4096) * 1024 * 1024;
    config.epsilon           = real(options["--epsilon"]);
    config.alpha             = real(options["--alpha"]);
    config.temperature_plies = integer(options["--temperature-plies"], 0, 128);
    const auto start         = integer(options["--start"], 0, UINT64_MAX);
    const auto games         = integer(options["--games"], 1, 1000000000);
    if (games - 1 > UINT64_MAX - start)
      throw std::invalid_argument("game ID overflow");
    if (!options["--verify"].empty()) {
      std::ifstream file(options["--verify"], std::ios::binary);
      if (!file)
        throw std::runtime_error("cannot open replay");
      std::vector<islay::ReplayRecord> records;
      islay::ReplayRecord              record;
      std::uint64_t                    count = 0, samples = 0;
      auto                             finish = [&] {
        if (records.empty())
          return;
        if (records[0].game_id != start + count)
          throw std::runtime_error("noncontiguous game IDs");
        islay::validate_replay_game(records, config);
        ++count;
        records.clear();
      };
      while (islay::read_replay_record(file, record)) {
        if (!records.empty() && record.game_id != records[0].game_id)
          finish();
        records.push_back(record);
        if (records.size() > islay::kMctsMaxPly)
          throw std::runtime_error("oversized replay game");
        ++samples;
      }
      finish();
      if (count != games)
        throw std::runtime_error("replay game count mismatch");
      std::cout << "{\"games\":" << count << ",\"samples\":" << samples << "}\n";
      return 0;
    }
    if (options["--model"].empty())
      throw std::invalid_argument("--model is required; no fake self-play evaluator");
    const auto             begun = islay::MctsClock::now();
    islay::NeuralEvaluator neural(options["--model"]);
    islay::CachedEvaluator cached(neural, integer(options["--cache-mib"], 0, 4096) * 1024 * 1024);
    std::uint64_t          samples = 0;
    for (std::uint64_t i = 0; i < games; ++i) {
      const auto         id      = start + i;
      const auto         records = islay::selfplay_game(id, cached, config);
      std::ostringstream frame(std::ios::out | std::ios::binary);
      frame.write("ISLAGM01", 8);
      put(frame, id);
      put(frame, islay::selfplay_seed(config.seed, id));
      put(frame, static_cast<std::uint32_t>(records.size()));
      put(frame, std::uint32_t{0});
      for (const auto &record: records)
        islay::write_replay_record(frame, record);
      const auto bytes = frame.str();
      std::cout.write(bytes.data(), bytes.size()).flush();
      if (!std::cout)
        throw std::runtime_error("self-play output pipe closed");
      samples += records.size();
    }
    const double seconds = std::chrono::duration<double>(islay::MctsClock::now() - begun).count();
    std::cerr << "{\"games\":" << games << ",\"samples\":" << samples << ",\"seconds\":" << seconds
              << ",\"games_per_hour\":" << games * 3600 / seconds << ",\"cache_hits\":" << cached.hits
              << ",\"neural_positions\":" << cached.misses << ",\"cache_bytes\":" << cached.bytes() << "}\n";
  } catch (const std::exception &error) {
    std::cerr << "selfplay error: " << error.what() << '\n';
    return 1;
  }
}
