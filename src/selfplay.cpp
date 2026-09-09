#include "selfplay.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "hash.hpp"

namespace islay {
  namespace {
    std::uint64_t mix(std::uint64_t x) {
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      return x ^ (x >> 31);
    }

    class Random {
    public:
      explicit Random(std::uint64_t seed) : state_(seed) {}
      double uniform() { return ((next() >> 12) + 0.5) * 0x1.0p-52; }
      double gamma(double shape) {
        if (shape < 1)
          return gamma(shape + 1) * std::pow(uniform(), 1 / shape);
        const double d = shape - 1.0 / 3, c = 1 / std::sqrt(9 * d);
        for (;;) {
          const double x = std::sqrt(-2 * std::log(uniform())) * std::cos(6.2831853071795864769 * uniform());
          double       v = 1 + c * x;
          if (v <= 0)
            continue;
          v              = v * v * v;
          const double u = uniform();
          if (u < 1 - 0.0331 * x * x * x * x || std::log(u) < 0.5 * x * x + d * (1 - v + std::log(v)))
            return d * v;
        }
      }

    private:
      std::uint64_t state_;
      std::uint64_t next() { return mix(state_ += 0x9e3779b97f4a7c15ULL); }
    };

    void check_config(const SelfplayConfig &c) {
      if (!c.simulations || c.simulations > 1000000 || c.tree_bytes < 4096 || !std::isfinite(c.epsilon) ||
          c.epsilon < 0 || c.epsilon > 1 || !std::isfinite(c.alpha) || c.alpha < 0.01 || c.alpha > 100 ||
          c.temperature_plies > 128)
        throw std::invalid_argument("invalid self-play config");
    }

    template<class T>
    void put(std::ostream &out, T value) {
      for (std::size_t i = 0; i < sizeof(T); ++i)
        out.put(static_cast<char>((value >> (8 * i)) & 255));
    }
    template<class T>
    T get(std::istream &in) {
      T result = 0;
      for (std::size_t i = 0; i < sizeof(T); ++i) {
        const int ch = in.get();
        if (ch == std::char_traits<char>::eof())
          throw std::runtime_error("truncated replay record");
        result |= static_cast<T>(ch) << (8 * i);
      }
      return result;
    }
  } // namespace

  CachedEvaluator::CachedEvaluator(Evaluator &backend, std::size_t bytes) : backend_(backend) {
    const auto count = bytes / sizeof(Entry);
    if (count)
      entries_.resize(std::bit_floor(count));
  }
  std::size_t CachedEvaluator::bytes() const noexcept { return entries_.size() * sizeof(Entry); }
  void CachedEvaluator::evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token stop) {
    if (boards.size() != outputs.size())
      throw std::invalid_argument("cache batch mismatch");
    for (std::size_t i = 0; i < boards.size() && !stop.stop_requested(); ++i) {
      Entry *entry = entries_.empty()
                             ? nullptr
                             : &entries_[hash_board(boards[i].player, boards[i].opponent) & (entries_.size() - 1)];
      if (entry && entry->valid && entry->board == boards[i]) {
        outputs[i] = entry->output;
        ++hits;
      } else {
        backend_.evaluate(boards.subspan(i, 1), outputs.subspan(i, 1), stop);
        if (stop.stop_requested())
          return;
        if (!std::isfinite(outputs[i].value) || std::abs(outputs[i].value) > 1 ||
            !std::all_of(outputs[i].logits.begin(), outputs[i].logits.end(), [](float x) { return std::isfinite(x); }))
          throw std::runtime_error("invalid cached neural output");
        ++misses;
        if (entry)
          *entry = {boards[i], outputs[i], true};
      }
    }
  }

  std::uint64_t selfplay_seed(std::uint64_t master, std::uint64_t game_id) noexcept {
    return mix(master ^ mix(game_id + 0x9e3779b97f4a7c15ULL));
  }

  std::vector<ReplayRecord> selfplay_game(std::uint64_t id, Evaluator &evaluator, const SelfplayConfig &config) {
    check_config(config);
    Random                    rng(selfplay_seed(config.seed, id));
    Board                     board = Board::start();
    std::vector<ReplayRecord> records;
    records.reserve(64);
    while (!game_status(board).terminal()) {
      if (records.size() >= kMctsMaxPly)
        throw std::runtime_error("self-play exceeded Othello ply bound");
      const auto      status = game_status(board);
      RootExploration exploration;
      exploration.fraction                  = config.epsilon;
      double                          total = 0;
      std::array<double, kPolicySize> noise{};
      for (int a = 0; a < kPolicySize; ++a) {
        if (status.legal(a)) {
          noise[a] = rng.gamma(config.alpha);
          total += noise[a];
        }
      }
      if (!std::isfinite(total) || total <= 0)
        throw std::runtime_error("invalid Dirichlet draw");
      for (int a = 0; a < kPolicySize; ++a)
        exploration.noise[a] = static_cast<float>(noise[a] / total);
      MctsLimits limits;
      limits.simulations = config.simulations;
      limits.tree_bytes  = config.tree_bytes;
      limits.exploration = &exploration;
      const auto result  = mcts_search(board, Rule::Othello, evaluator, limits);
      if (result.reason != MctsStop::SimulationLimit || result.simulations != config.simulations)
        throw std::runtime_error("incomplete self-play search; increase tree budget");
      ReplayRecord r;
      r.game_id     = id;
      r.board       = board;
      r.legal       = status.moves;
      r.pass        = status.forced_pass;
      r.ply         = static_cast<std::uint16_t>(records.size());
      r.stm         = r.ply % 2;
      r.temperature = r.ply < config.temperature_plies;
      r.priors      = result.network_priors;
      for (int i = 0; i < result.action_count; ++i)
        r.visits[result.actions[i].action] = static_cast<std::uint32_t>(result.actions[i].visits);
      if (r.temperature) {
        const auto    ticket     = static_cast<std::uint64_t>(rng.uniform() * config.simulations);
        std::uint64_t cumulative = 0;
        for (int a = 0; a < kPolicySize; ++a) {
          cumulative += r.visits[a];
          if (ticket < cumulative) {
            r.action = a;
            break;
          }
        }
      } else {
        r.action = std::max_element(r.visits.begin(), r.visits.end()) - r.visits.begin();
      }
      if (!try_play(board, r.action, board))
        throw std::logic_error("self-play selected illegal action");
      records.push_back(r);
    }
    const int black_outcome = game_status(board).value * (records.size() % 2 ? -1 : 1);
    for (auto &r: records)
      r.outcome = r.stm ? -black_outcome : black_outcome;
    validate_replay_game(records, config);
    return records;
  }

  void validate_replay_game(std::span<const ReplayRecord> records, const SelfplayConfig &config) {
    check_config(config);
    if (records.empty() || records.size() > kMctsMaxPly)
      throw std::runtime_error("invalid replay game length");
    Board board = Board::start();
    for (std::size_t i = 0; i < records.size(); ++i) {
      const auto &r      = records[i];
      const auto  status = game_status(board);
      if (r.game_id != records[0].game_id || r.ply != i || r.stm != i % 2 || r.board != board ||
          r.legal != status.moves || r.pass != status.forced_pass || r.temperature != (i < config.temperature_plies) ||
          status.terminal() || !status.legal(r.action))
        throw std::runtime_error("replay board/action/header mismatch");
      std::uint64_t visits = 0;
      double        priors = 0;
      for (int a = 0; a < kPolicySize; ++a) {
        if (!std::isfinite(r.priors[a]) || r.priors[a] < 0 || r.priors[a] > 1 ||
            (!status.legal(a) && (r.visits[a] || r.priors[a] != 0)))
          throw std::runtime_error("replay illegal visits/prior");
        visits += r.visits[a];
        priors += r.priors[a];
      }
      if (visits != config.simulations || std::abs(priors - 1) > 1e-5 || !r.visits[r.action] ||
          (!r.temperature && r.action != std::max_element(r.visits.begin(), r.visits.end()) - r.visits.begin()))
        throw std::runtime_error("replay visits/target mismatch");
      if (!try_play(board, r.action, board))
        throw std::runtime_error("replay illegal transition");
    }
    const auto status = game_status(board);
    if (!status.terminal())
      throw std::runtime_error("truncated game is not training data");
    const int black_outcome = status.value * (records.size() % 2 ? -1 : 1);
    for (const auto &r: records) {
      if (r.outcome != (r.stm ? -black_outcome : black_outcome))
        throw std::runtime_error("replay outcome perspective mismatch");
    }
  }

  void write_replay_record(std::ostream &out, const ReplayRecord &r) {
    put(out, r.game_id);
    put(out, r.board.player);
    put(out, r.board.opponent);
    put(out, r.legal);
    for (auto v: r.visits)
      put(out, v);
    for (auto p: r.priors)
      put(out, std::bit_cast<std::uint32_t>(p));
    put(out, r.ply);
    out.put(r.action);
    out.put(r.stm);
    out.put(static_cast<char>(r.outcome));
    out.put(r.temperature);
    out.put(r.pass);
    out.put(0);
    if (!out)
      throw std::runtime_error("replay write failed");
  }

  bool read_replay_record(std::istream &in, ReplayRecord &r) {
    if (in.peek() == std::char_traits<char>::eof()) {
      if (in.bad())
        throw std::runtime_error("replay read failed");
      return false;
    }
    r.game_id        = get<std::uint64_t>(in);
    r.board.player   = get<std::uint64_t>(in);
    r.board.opponent = get<std::uint64_t>(in);
    r.legal          = get<std::uint64_t>(in);
    for (auto &v: r.visits)
      v = get<std::uint32_t>(in);
    for (auto &p: r.priors)
      p = std::bit_cast<float>(get<std::uint32_t>(in));
    r.ply         = get<std::uint16_t>(in);
    r.action      = get<std::uint8_t>(in);
    r.stm         = get<std::uint8_t>(in);
    r.outcome     = std::bit_cast<std::int8_t>(get<std::uint8_t>(in));
    r.temperature = get<std::uint8_t>(in);
    r.pass        = get<std::uint8_t>(in);
    if (get<std::uint8_t>(in) != 0)
      throw std::runtime_error("unknown replay flags");
    return true;
  }

  bool selfplay_selftest() {
    try {
      class Fake final : public Evaluator {
      public:
        int  calls = 0;
        void evaluate(std::span<const Board> boards, std::span<Evaluation> out, std::stop_token) override {
          ++calls;
          for (std::size_t i = 0; i < boards.size(); ++i)
            out[i] = {};
        }
      } fake;
      CachedEvaluator cache(fake, 1024);
      const Board     board = Board::start();
      Evaluation      output;
      cache.evaluate(std::span(&board, 1), std::span(&output, 1), {});
      cache.evaluate(std::span(&board, 1), std::span(&output, 1), {});
      if (fake.calls != 1 || cache.hits != 1 || cache.misses != 1 || cache.bytes() > 1024)
        return false;
      CachedEvaluator            collision(fake, 300); // exactly one slot; full keys must disambiguate
      const std::array<Board, 3> sequence{Board{1, 2}, Board{2, 1}, Board{1, 2}};
      for (const auto &b: sequence)
        collision.evaluate(std::span(&b, 1), std::span(&output, 1), {});
      if (collision.hits || collision.misses != 3)
        return false;
      std::stop_source cancelled;
      cancelled.request_stop();
      output.value = 7;
      cache.evaluate(std::span(&board, 1), std::span(&output, 1), cancelled.get_token());
      if (output.value != 7)
        return false;
      SelfplayConfig config;
      config.simulations = 8;
      config.tree_bytes  = 65536;
      for (std::uint64_t id = 0; id < 4; ++id) {
        const auto         a = selfplay_game(id, fake, config);
        const auto         b = selfplay_game(id, cache, config);
        std::ostringstream x, y;
        for (const auto &r: a)
          write_replay_record(x, r);
        for (const auto &r: b)
          write_replay_record(y, r);
        if (x.str() != y.str() || x.str().size() != a.size() * kReplayRecordBytes)
          return false;
        for (int action: {19, 26, 37, 44})
          if (a[0].priors[action] != 0.25f)
            return false;
        std::istringstream truncated(x.str().substr(0, kReplayRecordBytes - 1));
        ReplayRecord       partial;
        bool               partial_rejected = false;
        try {
          read_replay_record(truncated, partial);
        } catch (const std::runtime_error &) {
          partial_rejected = true;
        }
        if (!partial_rejected)
          return false;
        std::istringstream        input(x.str());
        std::vector<ReplayRecord> read;
        ReplayRecord              r;
        while (read_replay_record(input, r))
          read.push_back(r);
        validate_replay_game(read, config);
        read[0].outcome = 9;
        bool rejected   = false;
        try {
          validate_replay_game(read, config);
        } catch (const std::runtime_error &) {
          rejected = true;
        }
        if (!rejected)
          return false;
      }
      if (selfplay_seed(42, 0) == selfplay_seed(42, 1))
        return false;
      return true;
    } catch (...) {
      return false;
    }
  }
} // namespace islay
