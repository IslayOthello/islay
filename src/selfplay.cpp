#include "selfplay.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "game.hpp"
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
      if (!c.simulations || c.simulations > 1000000 || c.batch_size < 1 || c.batch_size > 64 || c.tree_bytes < 4096 ||
          !std::isfinite(c.epsilon) || c.epsilon < 0 || c.epsilon > 1 || !std::isfinite(c.alpha) || c.alpha < 0.01 ||
          c.alpha > 100 || c.temperature_plies > 128 || c.fast_simulations > c.simulations ||
          !std::isfinite(c.full_search_prob) || c.full_search_prob <= 0.0 || c.full_search_prob > 1.0)
        throw std::invalid_argument("invalid self-play config");
    }

    std::int8_t encode_margin_target(int margin) {
      return static_cast<std::int8_t>(std::lround(std::tanh(static_cast<double>(margin) / 15.0) * 127.0));
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
    struct Miss {
      std::size_t output_index;
      std::size_t cache_index;
      int         symmetry;
    };
    thread_local std::vector<Board>      miss_boards;
    thread_local std::vector<Miss>       misses;
    thread_local std::vector<Evaluation> miss_outputs;
    miss_boards.clear();
    misses.clear();
    miss_outputs.clear();
    miss_boards.reserve(boards.size());
    misses.reserve(boards.size());
    for (std::size_t i = 0; i < boards.size() && !stop.stop_requested(); ++i) {
      const auto [canon, sym] = boards[i].canonical_symmetry();
      const std::size_t cache_index =
              entries_.empty() ? 0 : hash_board(canon.player, canon.opponent) & (entries_.size() - 1);
      Entry *entry = entries_.empty() ? nullptr : &entries_[cache_index];
      if (entry && entry->valid && entry->board == canon) {
        if (sym == 0) {
          outputs[i] = entry->output;
        } else {
          outputs[i] = entry->output;
          for (int a = 0; a < kPolicySize; ++a) {
            outputs[i].logits[a] = entry->output.logits[symmetry_action(a, sym)];
          }
        }
        ++hits;
      } else {
        miss_boards.push_back(canon);
        misses.push_back({i, cache_index, sym});
      }
    }
    if (miss_boards.empty() || stop.stop_requested())
      return;
    miss_outputs.resize(miss_boards.size());
    backend_.evaluate(miss_boards, miss_outputs, stop);
    if (stop.stop_requested())
      return;
    for (std::size_t i = 0; i < misses.size(); ++i) {
      const auto &miss  = misses[i];
      auto       &value = miss_outputs[i];
      if (!std::isfinite(value.value) || std::abs(value.value) > 1 || !std::isfinite(value.margin) ||
          std::abs(value.margin) > 1 ||
          !std::all_of(value.logits.begin(), value.logits.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("invalid cached neural output");
      ++this->misses;
      if (!entries_.empty())
        entries_[miss.cache_index] = {miss_boards[i], value, true};
      outputs[miss.output_index] = value;
      if (miss.symmetry != 0) {
        for (int a = 0; a < kPolicySize; ++a)
          outputs[miss.output_index].logits[a] = value.logits[symmetry_action(a, miss.symmetry)];
      }
    }
  }

  SharedEvalCache::SharedEvalCache(std::size_t bytes) {
    const auto count = bytes / sizeof(Entry);
    if (count)
      entries_.resize(std::bit_floor(count));
  }

  std::size_t SharedEvalCache::bytes() const noexcept { return entries_.size() * sizeof(Entry); }

  bool SharedEvalCache::lookup(const Board &canon, Evaluation &out) const {
    if (entries_.empty())
      return false;
    const std::size_t     idx   = hash_board(canon.player, canon.opponent) & (entries_.size() - 1);
    const std::size_t     shard = idx & (kShards - 1);
    const std::lock_guard lock(mutexes_[shard]);
    const Entry          &entry = entries_[idx];
    if (entry.valid && entry.board == canon) {
      out = entry.output;
      return true;
    }
    return false;
  }

  void SharedEvalCache::insert(const Board &canon, const Evaluation &out) {
    if (entries_.empty())
      return;
    const std::size_t     idx   = hash_board(canon.player, canon.opponent) & (entries_.size() - 1);
    const std::size_t     shard = idx & (kShards - 1);
    const std::lock_guard lock(mutexes_[shard]);
    entries_[idx] = {canon, out, true};
  }

  void SharedCachedEvaluator::evaluate(std::span<const Board> boards, std::span<Evaluation> outputs,
                                       std::stop_token stop) {
    if (boards.size() != outputs.size())
      throw std::invalid_argument("cache batch mismatch");
    struct Miss {
      std::size_t output_index;
      int         symmetry;
    };
    thread_local std::vector<Board>      miss_boards;
    thread_local std::vector<Miss>       misses;
    thread_local std::vector<Evaluation> miss_outputs;
    miss_boards.clear();
    misses.clear();
    miss_outputs.clear();
    miss_boards.reserve(boards.size());
    misses.reserve(boards.size());
    std::uint64_t local_hits = 0;
    for (std::size_t i = 0; i < boards.size() && !stop.stop_requested(); ++i) {
      const auto [canon, sym] = boards[i].canonical_symmetry();
      Evaluation cached_out;
      if (cache_.lookup(canon, cached_out)) {
        if (sym == 0) {
          outputs[i] = cached_out;
        } else {
          outputs[i] = cached_out;
          for (int a = 0; a < kPolicySize; ++a)
            outputs[i].logits[a] = cached_out.logits[symmetry_action(a, sym)];
        }
        ++local_hits;
      } else {
        miss_boards.push_back(canon);
        misses.push_back({i, sym});
      }
    }
    if (local_hits)
      cache_.hits.fetch_add(local_hits, std::memory_order_relaxed);
    if (miss_boards.empty() || stop.stop_requested())
      return;
    miss_outputs.resize(miss_boards.size());
    backend_.evaluate(miss_boards, miss_outputs, stop);
    if (stop.stop_requested())
      return;
    for (std::size_t i = 0; i < misses.size(); ++i) {
      const auto &miss  = misses[i];
      auto       &value = miss_outputs[i];
      if (!std::isfinite(value.value) || std::abs(value.value) > 1 || !std::isfinite(value.margin) ||
          std::abs(value.margin) > 1 ||
          !std::all_of(value.logits.begin(), value.logits.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("invalid cached neural output");
      cache_.insert(miss_boards[i], value);
      outputs[miss.output_index] = value;
      if (miss.symmetry != 0) {
        for (int a = 0; a < kPolicySize; ++a)
          outputs[miss.output_index].logits[a] = value.logits[symmetry_action(a, miss.symmetry)];
      }
    }
    cache_.misses.fetch_add(misses.size(), std::memory_order_relaxed);
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
      const auto status      = game_status(board);
      const auto current_ply = static_cast<std::uint16_t>(records.size());
      const bool is_opening  = (current_ply < config.opening_plies);

      RootExploration exploration;
      exploration.fraction                  = is_opening ? std::max(config.epsilon, 0.5) : config.epsilon;
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
      const bool use_fast = (config.fast_simulations > 0 && !is_opening && rng.uniform() >= config.full_search_prob);
      const std::uint32_t move_sims = use_fast ? config.fast_simulations : config.simulations;

      MctsLimits limits;
      limits.simulations = move_sims;
      limits.batch_size  = config.batch_size;
      limits.tree_bytes  = config.tree_bytes;
      limits.exploration = &exploration;
      limits.c_puct      = is_opening ? 2.5 : 1.5;
      const auto result  = mcts_search(board, Rule::Othello, evaluator, limits);
      if (result.reason != MctsStop::SimulationLimit || result.simulations != move_sims)
        throw std::runtime_error("incomplete self-play search; increase tree budget");
      ReplayRecord r;
      r.game_id     = id;
      r.board       = board;
      r.legal       = status.moves;
      r.pass        = status.forced_pass;
      r.ply         = current_ply;
      r.stm         = r.ply % 2;
      r.temperature = r.ply < config.temperature_plies;
      r.priors      = result.network_priors;
      for (int i = 0; i < result.action_count; ++i)
        r.visits[result.actions[i].action] = static_cast<std::uint32_t>(result.actions[i].visits);
      if (r.temperature) {
        if (is_opening && current_ply == 0) {
          // At ply 0, all 4 legal moves (d3, c4, f5, e6) are D4-symmetric official openings.
          // Sample uniformly among legal visited moves to ensure balanced opening exploration.
          std::array<int, 4> visited_moves{};
          int                cnt = 0;
          for (int a = 0; a < kPolicySize; ++a) {
            if (r.visits[a] > 0)
              visited_moves[cnt++] = a;
          }
          if (cnt > 0) {
            const int idx = static_cast<int>(rng.uniform() * cnt);
            r.action      = visited_moves[std::min(idx, cnt - 1)];
          } else {
            r.action = std::max_element(r.visits.begin(), r.visits.end()) - r.visits.begin();
          }
        } else if (is_opening) {
          // Soft exploration for early plies (official opening branches): sample with tau ~ 1.25
          // (v^0.8) among visited actions to explore parallel/diagonal/perpendicular lines
          std::array<double, kPolicySize> weights{};
          double                          weight_sum = 0.0;
          for (int a = 0; a < kPolicySize; ++a) {
            if (r.visits[a] > 0) {
              weights[a] = std::pow(static_cast<double>(r.visits[a]), 0.8);
              weight_sum += weights[a];
            }
          }
          if (weight_sum > 0.0) {
            const double ticket     = rng.uniform() * weight_sum;
            double       cumulative = 0.0;
            for (int a = 0; a < kPolicySize; ++a) {
              if (weights[a] > 0.0) {
                cumulative += weights[a];
                if (ticket <= cumulative) {
                  r.action = a;
                  break;
                }
              }
            }
          } else {
            r.action = std::max_element(r.visits.begin(), r.visits.end()) - r.visits.begin();
          }
        } else {
          const auto    ticket     = static_cast<std::uint64_t>(rng.uniform() * move_sims);
          std::uint64_t cumulative = 0;
          for (int a = 0; a < kPolicySize; ++a) {
            cumulative += r.visits[a];
            if (ticket < cumulative) {
              r.action = a;
              break;
            }
          }
        }
      } else {
        r.action = std::max_element(r.visits.begin(), r.visits.end()) - r.visits.begin();
      }
      if (!try_play(board, r.action, board))
        throw std::logic_error("self-play selected illegal action");
      records.push_back(r);
    }
    const int final_margin = popcount(board.player) - popcount(board.opponent);
    const int black_margin = records.size() % 2 ? -final_margin : final_margin;
    for (auto &r: records) {
      const int mover_margin = r.stm ? -black_margin : black_margin;
      r.outcome              = encode_margin_target(mover_margin);
      r.value_format         = 1;
    }
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
      const bool valid_visits =
              (visits == config.simulations) || (config.fast_simulations > 0 && visits == config.fast_simulations);
      if (!valid_visits || std::abs(priors - 1) > 1e-5 || !r.visits[r.action] ||
          (!r.temperature && r.action != std::max_element(r.visits.begin(), r.visits.end()) - r.visits.begin()))
        throw std::runtime_error("replay visits/target mismatch");
      if (!try_play(board, r.action, board))
        throw std::runtime_error("replay illegal transition");
    }
    const auto status = game_status(board);
    if (!status.terminal())
      throw std::runtime_error("truncated game is not training data");
    const int final_margin  = popcount(board.player) - popcount(board.opponent);
    const int black_margin  = records.size() % 2 ? -final_margin : final_margin;
    const int black_outcome = status.value * (records.size() % 2 ? -1 : 1);
    for (const auto &r: records) {
      if (r.value_format > 1)
        throw std::runtime_error("replay has unknown value target format");
      const int  mover_outcome = r.stm ? -black_outcome : black_outcome;
      const int  mover_margin  = r.stm ? -black_margin : black_margin;
      const auto expected      = r.value_format == 0 ? mover_outcome : encode_margin_target(mover_margin);
      if (r.outcome != expected)
        throw std::runtime_error("replay outcome perspective mismatch");
    }
  }

  float replay_margin_target(const ReplayRecord &record) noexcept {
    if (record.value_format == 1)
      return static_cast<float>(record.outcome) / 127.0f;
    if (record.value_format == 0)
      return static_cast<float>(record.outcome);
    return std::numeric_limits<float>::quiet_NaN();
  }

  float replay_wdl_target(const ReplayRecord &record) noexcept {
    const float margin = replay_margin_target(record);
    if (!std::isfinite(margin))
      return margin;
    return (margin > 0) - (margin < 0);
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
    out.put(static_cast<char>(r.value_format));
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
    r.ply          = get<std::uint16_t>(in);
    r.action       = get<std::uint8_t>(in);
    r.stm          = get<std::uint8_t>(in);
    r.outcome      = std::bit_cast<std::int8_t>(get<std::uint8_t>(in));
    r.temperature  = get<std::uint8_t>(in);
    r.pass         = get<std::uint8_t>(in);
    r.value_format = get<std::uint8_t>(in);
    if (r.value_format > 1 || (r.value_format == 0 && std::abs(static_cast<int>(r.outcome)) > 1) ||
        (r.value_format == 1 && r.outcome == std::numeric_limits<std::int8_t>::min()))
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
      const Board board_sym = board.symmetry(1);
      cache.evaluate(std::span(&board_sym, 1), std::span(&output, 1), {});
      if (fake.calls != 1 || cache.hits != 2 || cache.misses != 1 || cache.bytes() > 1024)
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
      SelfplayConfig pcr_config;
      pcr_config.simulations      = 8;
      pcr_config.fast_simulations = 2;
      pcr_config.full_search_prob = 0.25;
      pcr_config.tree_bytes       = 65536;
      const auto pcr_game         = selfplay_game(7, fake, pcr_config);
      if (pcr_game.empty())
        return false;
      validate_replay_game(pcr_game, pcr_config);

      SelfplayConfig config;
      config.simulations = 8;
      config.tree_bytes  = 65536;
      for (std::uint64_t id = 0; id < 4; ++id) {
        const auto a      = selfplay_game(id, fake, config);
        const auto b      = selfplay_game(id, cache, config);
        auto       legacy = a;
        for (auto &record: legacy) {
          record.outcome      = (record.outcome > 0) - (record.outcome < 0);
          record.value_format = 0;
        }
        validate_replay_game(legacy, config);
        std::ostringstream x, y;
        for (const auto &r: a)
          write_replay_record(x, r);
        for (const auto &r: b)
          write_replay_record(y, r);
        if (x.str() != y.str() || x.str().size() != a.size() * kReplayRecordBytes)
          return false;
        std::ostringstream old_bytes;
        for (const auto &r: legacy)
          write_replay_record(old_bytes, r);
        std::istringstream        old_input(old_bytes.str());
        std::vector<ReplayRecord> old_roundtrip;
        ReplayRecord              old_record;
        while (read_replay_record(old_input, old_record))
          old_roundtrip.push_back(old_record);
        if (old_roundtrip.size() != legacy.size())
          return false;
        validate_replay_game(old_roundtrip, config);
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
      ReplayRecord compatibility{};
      compatibility.outcome = 1;
      if (replay_value_target(compatibility) != 1.0f)
        return false;
      compatibility.outcome      = 127;
      compatibility.value_format = 1;
      if (replay_wdl_target(compatibility) != 1.0f || replay_margin_target(compatibility) != 1.0f)
        return false;
      compatibility.outcome = -64;
      if (replay_wdl_target(compatibility) >= 0 || replay_margin_target(compatibility) >= 0)
        return false;
      if (selfplay_seed(42, 0) == selfplay_seed(42, 1))
        return false;
      return true;
    } catch (...) {
      return false;
    }
  }
} // namespace islay
