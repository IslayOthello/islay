// Public and debug-only commands are documented in UCI.md.
#include "uci.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "board.hpp"
#include "endgame.hpp"
#include "game.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "neural.hpp"
#include "nn.hpp"
#include "options.hpp"
#include "perft.hpp"
#include "search_controller.hpp"
#include "selfplay.hpp"
#include "train.hpp"

namespace islay {
  namespace {

    constexpr const char *kName   = "islay 0.1.0";
    constexpr const char *kAuthor = "islay";

    using Clock = std::chrono::steady_clock;

    [[nodiscard]] double us_since(Clock::time_point t0) {
      return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }

    [[nodiscard]] std::string time_string(double elapsed_us) {
      std::ostringstream os;
      os.setf(std::ios::fixed);
      os.precision(9);
      os << elapsed_us / 1'000'000.0;
      return os.str();
    }

    [[nodiscard]] std::string grouped_count(std::string digits) {
      for (std::size_t pos = digits.size(); pos > 3;) {
        pos -= 3;
        digits.insert(pos, 1, ',');
      }
      return digits;
    }

    [[nodiscard]] std::string nps_string(std::uint64_t nodes, double elapsed_us) {
      if (elapsed_us <= 0.0)
        return "inf";
      std::ostringstream os;
      os.setf(std::ios::fixed);
      os.precision(0);
      os << nodes / (elapsed_us / 1'000'000.0);
      return os.str();
    }

    [[nodiscard]] Color color_of(char stm) noexcept {
      return (stm == 'O' || stm == 'o' || stm == 'W' || stm == 'w') ? Color::White : Color::Black;
    }

    [[nodiscard]] std::string format_duration(double seconds) {
      if (seconds < 60.0) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << seconds << "s";
        return ss.str();
      }
      const int mins = static_cast<int>(seconds) / 60;
      const int secs = static_cast<int>(seconds) % 60;
      return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    }

    class Engine {
    public:
      void run() {
        std::ios::sync_with_stdio(false);
        // getline must not flush cout outside the shared output mutex.
        std::cin.tie(nullptr);
        std::string line;
        while (std::getline(std::cin, line)) {
          std::istringstream is(line);
          std::string        cmd;
          if (!(is >> cmd))
            continue;
          if (cmd == "quit" || cmd == "exit") {
            break;
          }
          // Quiesce outside the output lock; rejected mutations also cancel.
          if (cmd == "position" || cmd == "ucinewgame" || cmd == "setoption" || cmd == "go" || cmd == "datagen" ||
              cmd == "train" || cmd == "value" || cmd == "eval" || cmd == "policy" ||
              (debug_ && (cmd == "test" || cmd == "selftest" || cmd == "bench")))
            search_.cancel();
          dispatch(cmd, is);
          flush_output();
        }
        search_.cancel();
      }

    private:
      Board                      board_ = Board::start();
      Color                      stm_   = Color::Black;
      Options                    options_{};
      int                        perft_tt_mib_ = 512; // tracks Options::perft_hash_mib so setoption can resize
      PerftTT                    tt_{512};
      bool                       debug_ = false; // `debug on` unlocks the development commands
      std::ostringstream         out_;
      std::mutex                 output_mutex_;
      SearchController           search_{std::cout, output_mutex_};
      std::shared_ptr<Evaluator> neural_;

      void flush_output() {
        const std::lock_guard lock(output_mutex_);
        std::cout << out_.str() << std::flush;
        out_.str("");
        out_.clear();
      }

      void dispatch(const std::string &cmd, std::istringstream &is) {
        if (cmd == "stop") {
          search_.stop();
          return;
        }
        if (cmd == "isready") {
          out_ << "readyok\n";
          return;
        }
        if (cmd == "debug") {
          cmd_debug(is);
          return;
        }

        if (cmd == "uci") {
          out_ << "id name " << kName << '\n' << "id author " << kAuthor << '\n';
          print_option_specs(out_);
          out_ << "uciok\n";
        } else if (cmd == "ucinewgame") {
          board_ = Board::start();
          stm_   = Color::Black;
          tt_.clear();
          search_.clear_tree();
        } else if (cmd == "position") {
          cmd_position(is);
        } else if (cmd == "setoption") {
          cmd_setoption(is);
        } else if (cmd == "go") {
          cmd_go(is);
        } else if (cmd == "datagen") {
          cmd_datagen(is);
        } else if (cmd == "train") {
          cmd_train(is);
        } else if (cmd == "value" || cmd == "eval") {
          cmd_value();
        } else if (cmd == "policy") {
          cmd_policy();
        } else if (is_debug_command(cmd)) {
          if (!debug_)
            unknown(cmd);
          else
            dispatch_debug(cmd, is);
        } else {
          unknown(cmd);
        }
      }

      void unknown(const std::string &cmd) { out_ << "info error: unknown command '" << cmd << "'\n"; }

      [[nodiscard]] static bool is_debug_command(const std::string &c) noexcept {
        return c == "d" || c == "display" || c == "board" || c == "bench" || c == "test" || c == "selftest" ||
               c == "backend";
      }

      void dispatch_debug(const std::string &cmd, std::istringstream &is) {
        if (cmd == "d" || cmd == "display" || cmd == "board") {
          board_.print(stm_, out_);
        } else if (cmd == "bench") {
          cmd_bench(is);
        } else if (cmd == "test" || cmd == "selftest") {
          cmd_test();
        } else if (cmd == "backend") {
          out_ << "movegen backend: " << movegen_backend() << '\n';
          out_ << "neural backend: " << neural_backend() << '\n';
        }
      }

      void cmd_debug(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok)) {
          out_ << "info string debug " << (debug_ ? "on" : "off") << '\n';
          return;
        }
        if (tok == "on") {
          debug_ = true;
        } else if (tok == "off") {
          debug_ = false;
        } else {
          out_ << "info error: expected 'debug on' or 'debug off'\n";
          return;
        }
        out_ << "info string debug " << (debug_ ? "on" : "off") << '\n';
      }

      void cmd_setoption(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok) || tok != "name") {
          out_ << "info error: expected 'setoption name <Name> value <Value>'\n";
          return;
        }
        std::vector<std::string> name_toks, value_toks;
        bool                     in_value = false;
        while (is >> tok) {
          if (!in_value && tok == "value") {
            in_value = true;
          } else {
            (in_value ? value_toks : name_toks).push_back(tok);
          }
        }
        const auto join = [](const std::vector<std::string> &v) {
          std::string s;
          for (std::size_t i = 0; i < v.size(); ++i) {
            if (i)
              s += ' ';
            s += v[i];
          }
          return s;
        };
        const std::string name      = join(name_toks);
        const std::string value     = join(value_toks);
        Options           candidate = options_;
        if (apply_option(candidate, name, value)) {
          std::string lower_name = name;
          for (auto &c: lower_name)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
          if (lower_name == "evalfile") {
            try {
              std::shared_ptr<Evaluator> loaded;
              if (!candidate.eval_file.empty()) {
                bool          is_native = false;
                std::ifstream probe(candidate.eval_file, std::ios::binary);
                if (probe) {
                  char magic[8] = {};
                  probe.read(magic, 8);
                  if (std::memcmp(magic, "ISLAYNET", 8) == 0)
                    is_native = true;
                }
                if (is_native) {
                  auto net = std::make_shared<NativeNeuralNet>(candidate.eval_file);
                  loaded   = std::make_shared<NativeNeuralEvaluator>(net, true);
                } else {
                  loaded = std::make_shared<NeuralEvaluator>(candidate.eval_file);
                }
              }
              neural_ = std::move(loaded);
            } catch (const std::exception &failure) {
              std::string error = failure.what();
              std::replace(error.begin(), error.end(), '\n', ' ');
              std::replace(error.begin(), error.end(), '\r', ' ');
              out_ << "info error: EvalFile rejected; previous evaluator retained: " << error << '\n';
              return;
            }
          }
          options_ = std::move(candidate);
          // Option changes invalidate rule-dependent caches.
          if (perft_tt_mib_ != options_.perft_hash_mib) {
            perft_tt_mib_ = options_.perft_hash_mib;
            tt_.resize(static_cast<std::size_t>(perft_tt_mib_));
          } else {
            tt_.clear();
          }
          search_.clear_tree();
          out_ << "info string option " << name << " = " << value << '\n';
        } else {
          out_ << "info error: unknown option or invalid value: '" << name << "' = '" << value << "'\n";
        }
      }

      // Commit the new position only after every move validates.
      void cmd_position(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok))
          return;

        Board tb;
        Color ts;
        if (tok == "startpos") {
          tb = Board::start();
          ts = Color::Black;
        } else if (tok == "fen") {
          std::string diagram, stmtok;
          if (!(is >> diagram >> stmtok)) {
            out_ << "info error: 'position fen' needs <diagram> <stm>\n";
            return;
          }
          if (!tb.set(diagram, stmtok.empty() ? '?' : stmtok[0])) {
            out_ << "info error: invalid diagram/side-to-move\n";
            return;
          }
          ts = color_of(stmtok[0]);
        } else {
          out_ << "info error: expected 'startpos' or 'fen'\n";
          return;
        }

        bool has_moves = false;
        if ((is >> tok) && tok == "moves") {
          std::string mv;
          while (is >> mv) {
            has_moves       = true;
            const Square sq = parse_square(mv);
            if (sq == PASS) {
              if (options_.rule != Rule::Othello || tb.has_moves() || !tb.passed().has_moves()) {
                out_ << "info error: illegal pass\n";
                return;
              }
              tb = tb.passed();
              ts = ~ts;
            } else if (sq != NOMOVE && (tb.moves() & square_bb(sq))) {
              tb = tb.play(sq);
              ts = ~ts;
            } else {
              out_ << "info error: illegal move '" << mv << "'\n";
              return;
            }
          }
        }
        if (!has_moves)
          search_.clear_tree();
        board_ = tb; // commit only after the whole command validated
        stm_   = ts;
      }

      void cmd_go(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok)) {
          out_ << "info error: expected go perft, nodes, movetime or infinite\n";
          return;
        }
        if (tok != "perft") {
          cmd_search(tok, is);
          return;
        }
        int depth = 0;
        if (!(is >> depth) || depth < 0) {
          out_ << "info error: 'go perft' needs a non-negative integer depth\n";
          return;
        }
        bool use_cache = true;
        if (is >> tok) {
          if (tok != "nocache") {
            out_ << "info error: expected 'go perft <depth> [nocache]'\n";
            return;
          }
          use_cache = false;
          if (is >> tok) {
            out_ << "info error: unexpected perft argument '" << tok << "'\n";
            return;
          }
        }
        run_perft(depth, use_cache);
      }

      void cmd_search(std::string tok, std::istringstream &is) {
        MctsLimits limits;
        limits.simulations = std::numeric_limits<std::uint64_t>::max();
        limits.tree_bytes  = static_cast<std::size_t>(options_.mcts_hash_mib) * 1024 * 1024;

        bool          nodes = false, timed = false, infinite = false;
        std::uint64_t milliseconds = 0;

        // Fischer-clock fields (all in ms; -1 means not provided)
        std::int64_t wtime = -1, btime = -1, winc = 0, binc = 0, movestogo = 0;

        // Helper: read the next token as int64; return false on failure
        auto parse_i64 = [&](std::int64_t &out) -> bool {
          std::string v;
          if (!(is >> v))
            return false;
          auto res = std::from_chars(v.data(), v.data() + v.size(), out);
          return res.ec == std::errc{} && res.ptr == v.data() + v.size();
        };
        // Helper: read the next token as uint64; return false on failure
        auto parse_u64 = [&](std::uint64_t &out) -> bool {
          std::string v;
          if (!(is >> v))
            return false;
          auto res = std::from_chars(v.data(), v.data() + v.size(), out);
          return res.ec == std::errc{} && res.ptr == v.data() + v.size();
        };

        // Track whether we saw any error that should abort
        std::string parse_error;

        do {
          if (tok == "infinite") {
            if (nodes || timed) {
              parse_error = "'infinite' cannot be combined with another limit";
              break;
            }
            if (infinite) {
              parse_error = "duplicate 'infinite'";
              break;
            }
            infinite = true;
          } else if (tok == "nodes") {
            if (infinite) {
              parse_error = "'infinite' cannot be combined with another limit";
              break;
            }
            if (nodes) {
              parse_error = "duplicate 'nodes'";
              break;
            }
            std::uint64_t n = 0;
            if (!parse_u64(n)) {
              parse_error = "invalid or missing value for 'nodes'";
              break;
            }
            limits.simulations = n;
            nodes              = true;
          } else if (tok == "movetime") {
            if (infinite) {
              parse_error = "'infinite' cannot be combined with another limit";
              break;
            }
            if (timed) {
              parse_error = "duplicate 'movetime'";
              break;
            }
            std::uint64_t n = 0;
            if (!parse_u64(n) || n > 86'400'000) {
              parse_error = "invalid search limit (movetime maximum 86400000 ms)";
              break;
            }
            milliseconds = n;
            timed        = true;
          } else if (tok == "wtime") {
            parse_i64(wtime);
          } else if (tok == "btime") {
            parse_i64(btime);
          } else if (tok == "winc") {
            parse_i64(winc);
          } else if (tok == "binc") {
            parse_i64(binc);
          } else if (tok == "movestogo") {
            parse_i64(movestogo);
          }
          // unknown tokens (depth, ponder, searchmoves…) are silently skipped
        } while (is >> tok);

        if (!parse_error.empty()) {
          out_ << "info error: " << parse_error << '\n';
          return;
        }

        // Fischer TC requires both wtime and btime
        const bool has_tc = (wtime >= 0 && btime >= 0);

        // Must have at least one recognised limit
        if (!infinite && !nodes && !timed && !has_tc) {
          out_ << "info error: expected go nodes <N> [movetime <MS>], "
                  "movetime <MS>, wtime <MS> btime <MS> [...], or infinite\n";
          return;
        }

        if (options_.rule != Rule::Othello) {
          out_ << "info error: MCTS supports Rule=Othello only\n";
          return;
        }
        out_ << "info string evaluator "
             << (neural_ ? neural_->description() : "uniform (P2 scaffold; no trained network)") << '\n';
        flush_output();

        // --- Time allocation (priority: movetime > Fischer TC > nodes/infinite) ---
        if (timed) {
          const std::uint64_t margin = (milliseconds >= 10) ? 3 : (milliseconds > 2 ? 1 : 0);
          limits.deadline            = Clock::now() + std::chrono::milliseconds(milliseconds - margin);
        } else if (!nodes && has_tc) {
          // UCI standard: wtime = White's clock, btime = Black's clock.
          // Othello starts with Black to move.
          const bool         we_are_black = (stm_ == Color::Black);
          const std::int64_t our_time     = we_are_black ? btime : wtime;
          const std::int64_t our_inc      = we_are_black ? binc : winc;
          if (our_time > 0) {
            // --- Othello-aware time management ---
            //
            // Othello on 8×8 lasts ~54 half-moves on average (range 30-60).
            // Discs on board = board_.count(), starts at 4.
            // Ply played ≈ disc_count - 4  (each move adds ≥1 disc on average).
            //
            // We allocate time with three components:
            //   1. base  = our_time / estimated_remaining_moves
            //   2. bonus = importance_factor * base   (peaks at midgame)
            //   3. inc   = 0.85 * our_inc
            //
            // Safety: always leave at least 150ms + 1% of remaining time on clock.

            const int disc_count = board_.count(); // 4..64
            const int ply_played = disc_count - 4; // 0..60

            std::int64_t remaining;
            if (movestogo > 0) {
              remaining = movestogo;
            } else {
              // Estimate remaining moves for our side.
              // Total game ≈ 54 half-moves; we play half of them.
              // At ply p, expected remaining for us = max(1, (54 - p + 1) / 2).
              const int total_ply = 54;
              const int ply_left  = std::max(1, total_ply - ply_played);
              remaining           = std::max(std::int64_t{1}, static_cast<std::int64_t>((ply_left + 1) / 2));
            }

            // Importance curve: flat 1.0 in opening (ply 0-10),
            // rises to 1.4 at midgame (ply ~25), falls back to 0.7 near endgame (ply 48+).
            // Piecewise linear over ply_played.
            double importance;
            if (ply_played <= 10)
              importance = 1.00;
            else if (ply_played <= 25)
              importance = 1.00 + 0.40 * (ply_played - 10) / 15.0;
            else if (ply_played <= 40)
              importance = 1.40 - 0.40 * (ply_played - 25) / 15.0;
            else
              importance = 1.00 - 0.30 * std::min((ply_played - 40) / 20.0, 1.0);

            // Safety reserve: keep at least 150ms + 1% of remaining time
            const std::int64_t reserve = std::max(std::int64_t{150}, our_time / 100);
            const std::int64_t usable  = std::max(std::int64_t{10}, our_time - reserve);

            // Base allocation
            const double base = static_cast<double>(usable) / static_cast<double>(remaining);
            const double inc  = static_cast<double>(our_inc) * 0.85;

            std::int64_t alloc = static_cast<std::int64_t>(base * importance + inc);

            // Hard cap: never use more than 30% of remaining clock in one move
            alloc = std::min(alloc, usable * 3 / 10);
            // Floor: always think at least 10ms
            alloc = std::max(alloc, std::int64_t{10});

            const std::int64_t ipc_margin = 8; // IPC pipe delivery safety (ms)
            limits.deadline               = Clock::now() + std::chrono::milliseconds(alloc - ipc_margin);
          }
        }

        try {
          search_.start(board_, options_.rule, limits, infinite, neural_);
        } catch (const std::exception &error) {
          out_ << "info error: could not start search: " << error.what() << '\n';
        }
      }

      void run_perft(int depth, bool use_cache) {
        if (depth < 1) {
          out_ << "Nodes searched: 1\nTime: 0.000000000 s\n";
          return;
        }

        const Rule         rule  = options_.rule;
        const auto         t0    = Clock::now();
        std::uint64_t      total = 0;
        std::ostringstream lines;

        const auto count_child = [&](const Board &child) -> std::uint64_t {
          if (depth == 1)
            return 1;
          return use_cache ? perft_cached(child, depth - 1, tt_, rule) : perft(child, depth - 1, rule);
        };

        Bitboard moves = board_.moves();
        if (moves) {
          while (moves) {
            const Square        sq  = pop_lsb(moves);
            const std::uint64_t cnt = count_child(board_.play(sq));
            lines << square_to_string(sq) << ": " << cnt << '\n';
            total += cnt;
          }
        } else if (rule == Rule::Othello && board_.passed().has_moves()) {
          const std::uint64_t cnt = count_child(board_.passed());
          lines << "pass: " << cnt << '\n';
          total = cnt;
        } else {
          lines << "(game over)\n";
        }

        const double elapsed_us = us_since(t0);
        out_ << lines.str() << '\n'
             << "Nodes searched: " << grouped_count(std::to_string(total)) << '\n'
             << "Time: " << time_string(elapsed_us) << " s\n"
             << "Speed: " << grouped_count(nps_string(total, elapsed_us)) << " N/s\n";
      }

      void cmd_datagen(std::istringstream &is) {
        if (options_.rule != Rule::Othello) {
          out_ << "info error: datagen supports Rule=Othello only\n";
          return;
        }

        std::uint64_t games       = 10;
        std::uint32_t sims        = 64;
        std::uint32_t threads     = 1;
        std::uint64_t seed        = 20260909;
        std::string   output_file = "data.bin";

        std::string tok;
        while (is >> tok) {
          if (tok == "games" || tok == "-g") {
            if (!(is >> games) || games == 0) {
              out_ << "info error: invalid games count\n";
              return;
            }
          } else if (tok == "sims" || tok == "-s") {
            if (!(is >> sims) || sims == 0) {
              out_ << "info error: invalid sims count\n";
              return;
            }
          } else if (tok == "threads" || tok == "-t") {
            if (!(is >> threads) || threads == 0) {
              out_ << "info error: invalid threads count\n";
              return;
            }
          } else if (tok == "output" || tok == "file" || tok == "-o") {
            if (!(is >> output_file) || output_file.empty()) {
              out_ << "info error: invalid output file\n";
              return;
            }
          } else if (tok == "seed") {
            if (!(is >> seed)) {
              out_ << "info error: invalid seed\n";
              return;
            }
          } else {
            try {
              std::size_t pos = 0;
              const auto  val = std::stoull(tok, &pos);
              if (pos == tok.size() && val > 0) {
                games = val;
              } else {
                out_ << "info error: unexpected datagen argument '" << tok << "'\n";
                return;
              }
            } catch (...) {
              out_ << "info error: unexpected datagen argument '" << tok << "'\n";
              return;
            }
          }
        }

        SelfplayConfig config;
        config.simulations = sims;
        config.seed        = seed;
        config.tree_bytes  = static_cast<std::size_t>(options_.mcts_hash_mib) * 1024 * 1024;

        std::shared_ptr<Evaluator> eval_backend;
        if (neural_) {
          eval_backend = neural_;
          out_ << "info string datagen using trained evaluator: " << neural_->description() << "\n";
        } else {
          auto gen0_net = std::make_shared<NativeNeuralNet>();
          eval_backend  = std::make_shared<NativeNeuralEvaluator>(gen0_net, true);
          out_ << "info string datagen using gen-0 evaluator (random initialization)\n";
        }

        out_ << "info string datagen starting: games=" << games << " sims=" << sims << " threads=" << threads
             << " output=" << output_file << "\n";
        flush_output();

        std::ofstream out_file(output_file, std::ios::binary | std::ios::trunc);
        if (!out_file) {
          out_ << "info error: cannot open output file '" << output_file << "'\n";
          return;
        }

        std::mutex                                         file_mutex;
        std::atomic<std::uint64_t>                         game_counter{0};
        std::atomic<std::uint64_t>                         total_samples{0};
        std::uint64_t                                      next_game_to_write = 0;
        std::map<std::uint64_t, std::vector<ReplayRecord>> pending;
        const auto                                         start_time      = Clock::now();
        auto                                               last_print_time = start_time;

        const std::uint64_t print_interval = (games <= 20)    ? 1
                                             : (games <= 100) ? 5
                                                              : std::max<std::uint64_t>(10, games / 50);

        std::vector<std::thread> workers;
        workers.reserve(threads);

        for (std::uint32_t t = 0; t < threads; ++t) {
          workers.emplace_back([&]() {
            CachedEvaluator cached(*eval_backend, 16 * 1024 * 1024);
            while (true) {
              const std::uint64_t g = game_counter.fetch_add(1);
              if (g >= games)
                break;

              std::vector<ReplayRecord> records;
              try {
                records = selfplay_game(g, cached, config);
              } catch (const std::exception &e) {
                const std::lock_guard lock(output_mutex_);
                std::cout << "info string datagen game " << g << " error: " << e.what() << "\n" << std::flush;
                continue;
              }

              {
                std::lock_guard lock(file_mutex);
                pending.emplace(g, std::move(records));
                while (!pending.empty() && pending.begin()->first == next_game_to_write) {
                  auto it = pending.begin();
                  for (const auto &r: it->second)
                    write_replay_record(out_file, r);
                  total_samples.fetch_add(it->second.size());
                  pending.erase(it);
                  ++next_game_to_write;

                  const bool should_print = (next_game_to_write == 1) || (next_game_to_write == games) ||
                                            (next_game_to_write % print_interval == 0) ||
                                            (us_since(last_print_time) >= 1'000'000);

                  if (should_print) {
                    last_print_time         = Clock::now();
                    const double elapsed_us = us_since(start_time);
                    const double elapsed_s  = elapsed_us / 1e6;
                    const double gps        = elapsed_us > 0 ? next_game_to_write * 1e6 / elapsed_us : 0.0;
                    const double sps        = elapsed_us > 0 ? total_samples.load() * 1e6 / elapsed_us : 0.0;
                    const double pct   = 100.0 * static_cast<double>(next_game_to_write) / static_cast<double>(games);
                    const double eta_s = gps > 0 ? (games - next_game_to_write) / gps : 0.0;

                    const std::lock_guard out_lock(output_mutex_);
                    std::cout << "info string datagen progress: " << next_game_to_write << "/" << games << " games ("
                              << std::fixed << std::setprecision(1) << pct << "%) | "
                              << grouped_count(std::to_string(total_samples.load())) << " samples | " << std::fixed
                              << std::setprecision(1) << gps << " games/s | " << std::setprecision(0) << sps
                              << " pos/s | "
                              << "elapsed: " << format_duration(elapsed_s) << " | "
                              << "ETA: " << format_duration(eta_s) << "\n"
                              << std::flush;
                  }
                }
                out_file.flush();
              }
            }
          });
        }

        for (auto &w: workers)
          w.join();

        const double elapsed_us = us_since(start_time);
        const double elapsed_s  = elapsed_us / 1e6;
        const double gps        = elapsed_us > 0 ? next_game_to_write * 1e6 / elapsed_us : 0.0;
        const double avg_plies =
                next_game_to_write > 0 ? static_cast<double>(total_samples.load()) / next_game_to_write : 0.0;

        out_ << "info string datagen complete: " << next_game_to_write << "/" << games << " games (100.0%) | "
             << grouped_count(std::to_string(total_samples.load())) << " samples (avg " << std::fixed
             << std::setprecision(1) << avg_plies << " moves/game) | " << std::setprecision(1) << gps << " games/s | "
             << "time: " << format_duration(elapsed_s) << " -> " << output_file << "\n";
      }

      void cmd_train(std::istringstream &is) {
        if (options_.rule != Rule::Othello) {
          out_ << "info error: train supports Rule=Othello only\n";
          return;
        }

        PipelineConfig config;
        config.num_workers             = 4;
        config.target_steps            = 0;
        config.batch_size              = 64;
        config.checkpoint_every        = 0;
        config.warmup_games            = 10;
        config.output_file             = "model.bin";
        config.optimizer.learning_rate = 0.01f;
        config.selfplay.simulations    = 64;
        config.selfplay.seed           = 20260909;

        std::string tok;
        while (is >> tok) {
          if (tok == "data" || tok == "file" || tok == "dataset" || tok == "-d") {
            if (!(is >> config.dataset_file) || config.dataset_file.empty()) {
              out_ << "info error: invalid data file path\n";
              return;
            }
          } else if (tok == "steps" || tok == "-n") {
            if (!(is >> config.target_steps) || config.target_steps == 0) {
              out_ << "info error: invalid steps count\n";
              return;
            }
          } else if (tok == "batch" || tok == "-b") {
            if (!(is >> config.batch_size) || config.batch_size == 0) {
              out_ << "info error: invalid batch size\n";
              return;
            }
          } else if (tok == "lr") {
            if (!(is >> config.optimizer.learning_rate) || config.optimizer.learning_rate <= 0.0f) {
              out_ << "info error: invalid learning rate\n";
              return;
            }
          } else if (tok == "output" || tok == "-o") {
            if (!(is >> config.output_file) || config.output_file.empty()) {
              out_ << "info error: invalid output path\n";
              return;
            }
          } else if (tok == "init" || tok == "-i") {
            if (!(is >> config.init_checkpoint) || config.init_checkpoint.empty()) {
              out_ << "info error: invalid init checkpoint\n";
              return;
            }
          } else if (tok == "workers" || tok == "threads" || tok == "-t") {
            if (!(is >> config.num_workers) || config.num_workers == 0) {
              out_ << "info error: invalid workers count\n";
              return;
            }
          } else if (tok == "sims" || tok == "-s") {
            if (!(is >> config.selfplay.simulations) || config.selfplay.simulations == 0) {
              out_ << "info error: invalid sims count\n";
              return;
            }
          } else if (tok == "warmup") {
            if (!(is >> config.warmup_games)) {
              out_ << "info error: invalid warmup games count\n";
              return;
            }
          } else if (tok == "checkpoint") {
            if (!(is >> config.checkpoint_every)) {
              out_ << "info error: invalid checkpoint interval\n";
              return;
            }
          } else {
            try {
              std::size_t pos = 0;
              const auto  val = std::stoull(tok, &pos);
              if (pos == tok.size() && val > 0) {
                config.target_steps = val;
              } else {
                out_ << "info error: unexpected train argument '" << tok << "'\n";
                return;
              }
            } catch (...) {
              out_ << "info error: unexpected train argument '" << tok << "'\n";
              return;
            }
          }
        }

        const bool is_offline = !config.dataset_file.empty();
        if (is_offline) {
          if (!std::filesystem::exists(config.dataset_file)) {
            out_ << "info error: dataset file not found: '" << config.dataset_file << "'\n";
            out_ << "info string hint: generate data first using 'datagen output " << config.dataset_file
                 << "' or specify an existing .bin dataset file.\n";
            return;
          }
        }
        if (!config.init_checkpoint.empty() && !std::filesystem::exists(config.init_checkpoint)) {
          out_ << "info error: init checkpoint file not found: '" << config.init_checkpoint << "'\n";
          return;
        }

        if (config.target_steps == 0)
          config.target_steps = is_offline ? 500 : 1000;

        if (is_offline) {
          out_ << "info string train mode: offline dataset (" << config.dataset_file << ")\n";
        } else {
          out_ << "info string train mode: online AlphaZero self-play (workers=" << config.num_workers
               << " sims=" << config.selfplay.simulations << ")\n";
        }
        out_ << "info string train starting: steps=" << config.target_steps << " batch=" << config.batch_size
             << " lr=" << config.optimizer.learning_rate << " output=" << config.output_file << "\n";
        flush_output();

        std::chrono::steady_clock::time_point last_print = Clock::now();
        StepMetrics                           last_metrics;

        auto progress_callback = [&](const StepMetrics &m) {
          last_metrics                     = m;
          const std::uint64_t step_idx     = m.step;
          const bool          should_print = (step_idx == 1) || (step_idx == m.target_steps) || (step_idx % 10 == 0) ||
                                             (us_since(last_print) >= 1'000'000);
          if (!should_print)
            return;
          last_print       = Clock::now();
          const double pct = m.target_steps > 0
                                     ? 100.0 * static_cast<double>(step_idx) / static_cast<double>(m.target_steps)
                                     : 0.0;

          const std::lock_guard lock(output_mutex_);
          std::cout << "info string train progress: step " << step_idx << "/" << m.target_steps << " (" << std::fixed
                    << std::setprecision(1) << pct << "%) | "
                    << "loss: " << std::setprecision(4) << m.total_loss << " (pol: " << m.policy_loss
                    << ", val: " << m.value_loss << ") | "
                    << "grad_norm: " << m.grad_norm << " | ";
          if (m.games_completed > 0) {
            std::cout << "games: " << m.games_completed << " | "
                      << "buf: " << m.buffer_size << " | ";
          }
          std::cout << std::fixed << std::setprecision(1) << m.steps_per_sec << " step/s | "
                    << "elapsed: " << format_duration(m.elapsed_sec) << " | "
                    << "ETA: " << format_duration(m.eta_sec) << "\n"
                    << std::flush;
        };

        try {
          AlphaZeroTrainer trainer(config);
          trainer.run(progress_callback);

          const std::lock_guard lock(output_mutex_);
          std::cout << "info string train complete: " << last_metrics.step << "/" << last_metrics.target_steps
                    << " steps (100.0%) | "
                    << "final loss: " << std::fixed << std::setprecision(4) << last_metrics.total_loss
                    << " (pol: " << last_metrics.policy_loss << ", val: " << last_metrics.value_loss << ") | "
                    << std::setprecision(1) << last_metrics.steps_per_sec << " step/s | "
                    << "time: " << format_duration(last_metrics.elapsed_sec) << " -> " << config.output_file << "\n"
                    << std::flush;

          // Update active neural evaluator in current UCI session
          neural_ = std::make_shared<NativeNeuralEvaluator>(trainer.net(), true);
          std::cout << "info string active evaluator updated to trained model (" << trainer.net()->training_steps()
                    << " steps)\n"
                    << std::flush;
        } catch (const std::exception &e) {
          const std::lock_guard lock(output_mutex_);
          std::cout << "info error: train failed: " << e.what() << "\n" << std::flush;
        }
      }

      void cmd_value() {
        if (options_.rule != Rule::Othello) {
          out_ << "info error: value supports Rule=Othello only\n";
          return;
        }

        board_.print(stm_, out_);

        Evaluation eval{};
        if (neural_) {
          try {
            neural_->evaluate(std::span(&board_, 1), std::span(&eval, 1), std::stop_token{});
          } catch (const std::exception &e) {
            out_ << "info error: evaluator failed: " << e.what() << '\n';
            return;
          }
        }

        const double mover_val = eval.value;
        const double mover_wr  = (mover_val + 1.0) * 50.0;
        const double black_wr  = (stm_ == Color::Black) ? mover_wr : (100.0 - mover_wr);
        const double white_wr  = 100.0 - black_wr;
        const double black_val = (stm_ == Color::Black) ? mover_val : -mover_val;

        const GameStatus status = game_status(board_);
        out_ << "Evaluator: " << (neural_ ? neural_->description() : "uniform (no network loaded)") << '\n';
        out_ << "Value (mover): " << std::showpos << std::fixed << std::setprecision(4) << mover_val << std::noshowpos
             << " | Winrate: " << std::fixed << std::setprecision(2) << mover_wr << "%\n";
        out_ << "Perspective: Black: " << std::showpos << std::fixed << std::setprecision(4) << black_val
             << std::noshowpos << " (" << std::fixed << std::setprecision(2) << black_wr
             << "%) | White: " << std::showpos << std::fixed << std::setprecision(4) << -black_val << std::noshowpos
             << " (" << std::fixed << std::setprecision(2) << white_wr << "%)\n";

        if (status.terminal()) {
          const Bitboard black       = (stm_ == Color::Black) ? board_.player : board_.opponent;
          const Bitboard white       = (stm_ == Color::Black) ? board_.opponent : board_.player;
          const int      black_count = popcount(black);
          const int      white_count = popcount(white);
          out_ << "Status: Terminal (Game Over) | Outcome: ";
          if (black_count > white_count)
            out_ << "Black wins (" << black_count << " - " << white_count << ")\n";
          else if (white_count > black_count)
            out_ << "White wins (" << white_count << " - " << black_count << ")\n";
          else
            out_ << "Draw (" << black_count << " - " << white_count << ")\n";
        } else {
          const int empties = 64 - board_.count();
          if (empties <= 10) {
            const int outcome = solve_exact_wld(board_, empties);
            out_ << "Endgame solver (depth " << empties << "): "
                 << (outcome > 0   ? "Mover Win (+1)"
                     : outcome < 0 ? "Mover Loss (-1)"
                                   : "Draw (0)")
                 << '\n';
          }
        }
      }

      void cmd_policy() {
        if (options_.rule != Rule::Othello) {
          out_ << "info error: policy supports Rule=Othello only\n";
          return;
        }

        Evaluation eval{};
        if (neural_) {
          try {
            neural_->evaluate(std::span(&board_, 1), std::span(&eval, 1), std::stop_token{});
          } catch (const std::exception &e) {
            out_ << "info error: evaluator failed: " << e.what() << '\n';
            return;
          }
        }

        const GameStatus                status = game_status(board_);
        std::array<Square, kPolicySize> actions;
        int                             n     = 0;
        auto                            moves = status.moves;
        while (moves)
          actions[n++] = pop_lsb(moves);
        if (status.forced_pass)
          actions[n++] = PASS;

        std::array<double, kPolicySize> probs{};
        if (n > 0) {
          float max_logit = eval.logits[actions[0]];
          for (int i = 1; i < n; ++i)
            max_logit = std::max(max_logit, eval.logits[actions[i]]);
          double sum_exp = 0.0;
          for (int i = 0; i < n; ++i) {
            probs[actions[i]] = std::exp(static_cast<double>(eval.logits[actions[i]]) - max_logit);
            sum_exp += probs[actions[i]];
          }
          const double inv_sum = 1.0 / sum_exp;
          for (int i = 0; i < n; ++i)
            probs[actions[i]] *= inv_sum;
        }

        const Bitboard black = (stm_ == Color::Black) ? board_.player : board_.opponent;
        const Bitboard white = (stm_ == Color::Black) ? board_.opponent : board_.player;

        out_ << "         a       b       c       d       e       f       g       h\n";
        out_ << "     +-------+-------+-------+-------+-------+-------+-------+-------+\n";

        constexpr const char *kEmptyRow = "     |       |       |       |       |       |       |       |       |\n";

        for (int rank = 0; rank < 8; ++rank) {
          out_ << kEmptyRow;

          out_ << "  " << (rank + 1) << "  |";
          for (int file = 0; file < 8; ++file) {
            const Square   sq = rank * 8 + file;
            const Bitboard bb = square_bb(sq);
            if (black & bb) {
              out_ << "   X   |";
            } else if (white & bb) {
              out_ << "   O   |";
            } else if (status.moves & bb) {
              const double       pct = probs[sq] * 100.0;
              std::ostringstream cell;
              if (pct >= 99.95) {
                cell << " 100%  ";
              } else if (pct >= 10.0) {
                cell << ' ' << std::fixed << std::setprecision(1) << pct << "% ";
              } else if (pct >= 0.05) {
                cell << "  " << std::fixed << std::setprecision(1) << pct << "% ";
              } else if (pct > 0.0) {
                cell << " <0.1% ";
              } else {
                cell << "  0.0% ";
              }
              out_ << cell.str() << '|';
            } else {
              out_ << "   .   |";
            }
          }
          out_ << "  " << (rank + 1) << '\n';
          out_ << kEmptyRow;
          out_ << "     +-------+-------+-------+-------+-------+-------+-------+-------+\n";
        }
        out_ << "         a       b       c       d       e       f       g       h\n";

        const double mover_val = eval.value;
        const double mover_wr  = (mover_val + 1.0) * 50.0;
        const double black_wr  = (stm_ == Color::Black) ? mover_wr : (100.0 - mover_wr);
        const double white_wr  = 100.0 - black_wr;

        out_ << "\nEvaluator: " << (neural_ ? neural_->description() : "uniform (no network loaded)") << '\n';
        out_ << "Side to move: " << (stm_ == Color::Black ? "Black (X)" : "White (O)")
             << " | Discs: X=" << popcount(black) << " O=" << popcount(white)
             << " | Legal moves: " << popcount(status.moves) << '\n';
        out_ << "Value (mover): " << std::showpos << std::fixed << std::setprecision(4) << mover_val << std::noshowpos
             << " | Winrate: " << std::fixed << std::setprecision(2) << mover_wr << "% (Black: " << black_wr
             << "%, White: " << white_wr << "%)\n";

        if (status.terminal()) {
          const int black_count = popcount(black);
          const int white_count = popcount(white);
          out_ << "Status: Terminal (Game Over) | Outcome: ";
          if (black_count > white_count)
            out_ << "Black wins (" << black_count << " - " << white_count << ")\n";
          else if (white_count > black_count)
            out_ << "White wins (" << white_count << " - " << black_count << ")\n";
          else
            out_ << "Draw (" << black_count << " - " << white_count << ")\n";
        } else if (status.forced_pass) {
          out_ << "Legal moves policy ranking:\n";
          out_ << "  #1  pass: 100.0% (forced pass, logit: " << std::showpos << std::fixed << std::setprecision(3)
               << eval.logits[PASS] << std::noshowpos << ")\n";
        } else if (n > 0) {
          struct ActionPolicy {
            Square action;
            double prob;
            float  logit;
          };
          std::vector<ActionPolicy> sorted;
          sorted.reserve(n);
          for (int i = 0; i < n; ++i) {
            sorted.push_back({actions[i], probs[actions[i]], eval.logits[actions[i]]});
          }
          std::sort(sorted.begin(), sorted.end(), [](const ActionPolicy &a, const ActionPolicy &b) {
            if (std::abs(a.prob - b.prob) > 1e-7)
              return a.prob > b.prob;
            return a.action < b.action;
          });

          out_ << "Legal moves policy ranking:\n";
          for (std::size_t i = 0; i < sorted.size(); ++i) {
            const auto  &ap  = sorted[i];
            const double pct = ap.prob * 100.0;
            out_ << "  #" << std::left << std::setw(2) << (i + 1) << " " << std::setw(4) << square_to_string(ap.action)
                 << ": " << std::right << std::fixed << std::setprecision(1) << std::setw(5) << pct
                 << "%  (logit: " << std::showpos << std::fixed << std::setprecision(3) << ap.logit << std::noshowpos
                 << ")\n";
          }
        }
      }

      void cmd_bench(std::istringstream &is) {
        int maxd = 11;
        is >> maxd;
        const Board start = Board::start();
        out_ << "depth            nodes        time(s)             nps\n"
             << "---------------------------------------------------------\n";
        for (int d = 1; d <= maxd; ++d) {
          const auto          t0         = Clock::now();
          const std::uint64_t n          = perft(start, d, options_.rule);
          const double        elapsed_us = us_since(t0);
          out_.width(5);
          out_ << d << ' ';
          out_.width(16);
          out_ << n << ' ';
          out_.width(14);
          out_ << time_string(elapsed_us) << ' ';
          out_.width(15);
          out_ << nps_string(n, elapsed_us) << '\n';
        }
      }

      void cmd_test() {
        out_ << "output formatting self-test ... " << std::flush;
        if (grouped_count("0") != "0" || grouped_count("1") != "1" || grouped_count("999") != "999" ||
            grouped_count("1000") != "1,000" || grouped_count("999999") != "999,999" ||
            grouped_count("1000000") != "1,000,000" ||
            grouped_count("18446744073709551615") != "18,446,744,073,709,551,615" ||
            grouped_count(nps_string(1, 0)) != "inf" || grouped_count(nps_string(1234567, 1'000'000)) != "1,234,567") {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";

        out_ << "microsecond timing self-test ... " << std::flush;
        if (time_string(0) != "0.000000000" || time_string(0.125) != "0.000000125" ||
            time_string(1234.5) != "0.001234500" || time_string(1'000'000) != "1.000000000" ||
            nps_string(1, 0.5) != "2000000" || nps_string(1000, 250) != "4000000" || nps_string(1, 3) != "333333" ||
            nps_string(0, 10) != "0" || nps_string(1, -1) != "inf") {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";

        out_ << "movegen self-test (" << movegen_backend() << ") ... " << std::flush;
        if (!movegen_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";

        out_ << "Othello game self-test ... " << std::flush;
        if (!game_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";
        out_ << "endgame solver self-test ... " << std::flush;
        if (!endgame_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";
        out_ << "PUCT core self-test ... " << std::flush;
        if (!mcts_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";

        out_ << "neural encoding self-test ... ";
        if (!neural_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";
        out_ << "native neural network self-test ... ";
        if (!nn_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";
        out_ << "alpha-zero training self-test ... ";
        if (!train_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";
        out_ << "self-play/replay self-test ... ";
        if (!selfplay_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";
        out_ << "search controller self-test ... ";
        if (!search_controller_selftest()) {
          out_ << "FAILED\n";
          return;
        }
        out_ << "ok\n";

        constexpr std::array<std::uint64_t, 9> known{0, 4, 12, 56, 244, 1396, 8200, 55092, 390216};
        const Board                            start  = Board::start();
        bool                                   all_ok = true;
        for (int d = 1; d <= 8; ++d) {
          const std::uint64_t got = perft(start, d, Rule::Othello);
          const bool          ok  = (got == known[static_cast<std::size_t>(d)]);
          all_ok                  = all_ok && ok;
          out_ << "perft(" << d << ") = " << got << (ok ? "  ok\n" : "  MISMATCH\n");
        }

        PerftTT    tt(64);
        const bool cache_ok = (perft(start, 8, Rule::Othello) == perft_cached(start, 8, tt, Rule::Othello));
        all_ok              = all_ok && cache_ok;
        out_ << "cache consistency perft(8): " << (cache_ok ? "ok" : "MISMATCH") << '\n';

        const Board         asym   = Board::start().play(parse_square("d3")).play(parse_square("c3"));
        const std::uint64_t base   = perft(asym, 6, Rule::Othello);
        bool                sym_ok = true;
        for (int s = 0; s < 8; ++s)
          sym_ok = sym_ok && (perft(asym.symmetry(s), 6, Rule::Othello) == base);
        PerftTT    tt2(64);
        const bool symcache_ok = (perft_cached(asym, 7, tt2, Rule::Othello) == perft(asym, 7, Rule::Othello));
        all_ok                 = all_ok && sym_ok && symcache_ok;
        out_ << "symmetry invariance perft(6): " << (sym_ok ? "ok" : "MISMATCH") << '\n'
             << "symmetry cache perft(7): " << (symcache_ok ? "ok" : "MISMATCH") << '\n';

        // Fixed legal playouts cover opening, middlegame, endgame and forced pass.
        // Counts are frozen from the pre-optimization uncached implementation.
        struct PerftCase {
          Board         board;
          std::uint64_t othello;
          std::uint64_t reversi;
        };
        constexpr std::array<PerftCase, 9> positions{{
                {{240652910592ULL, 370411524ULL}, 54185, 54185},
                {{141013910561792ULL, 2314885659704166404ULL}, 431432, 431432},
                {{8024582863072280ULL, 2314986746359087110ULL}, 142630, 142630},
                {{643436922635553823ULL, 16289534172264902688ULL}, 13878, 13878},
                {{4449989296128ULL, 69362253824ULL}, 75504, 75504},
                {{71057083469824ULL, 9594459700723712ULL}, 526743, 526743},
                {{76082365011218696ULL, 1166769015944912944ULL}, 346517, 346517},
                {{80817503139037048ULL, 3486033126843646080ULL}, 9043, 9028},
                {{2455651727219126272ULL, 576743339727456007ULL}, 1825, 0},
        }};
        PerftTT                            regression_tt(1);
        bool                               positions_ok = true;
        for (const auto &position: positions) {
          for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
            regression_tt.clear();
            const auto expected = rule == Rule::Othello ? position.othello : position.reversi;
            for (int symmetry = 0; symmetry < 8; ++symmetry) {
              const Board board  = position.board.symmetry(symmetry);
              const auto  plain  = perft(board, 5, rule);
              const auto  cached = perft_cached(board, 5, regression_tt, rule);
              positions_ok       = (plain == expected && cached == expected) && positions_ok;
            }
          }
        }
        all_ok = all_ok && positions_ok;
        out_ << "multi-position perft(5), both rules, symmetry and 1 MiB cache: " << (positions_ok ? "ok" : "MISMATCH")
             << '\n';

        std::uint64_t s   = 0x9E3779B97F4A7C15ULL;
        const auto    rnd = [&s]() noexcept {
          s ^= s << 13;
          s ^= s >> 7;
          s ^= s << 17;
          return s;
        };
        bool rule_tested = false;
        for (int game = 0; game < 6000 && !rule_tested; ++game) {
          Board b = Board::start();
          for (int ply = 0; ply < 70; ++ply) {
            Bitboard m = b.moves();
            if (!m) {
              if (b.passed().has_moves()) {
                const std::uint64_t oth = perft(b, 2, Rule::Othello);
                const std::uint64_t rev = perft(b, 2, Rule::Reversi);
                const bool          ok  = (oth > 0 && rev == 0);
                all_ok                  = all_ok && ok;
                out_ << "rule at stuck position: Othello perft(2)=" << oth << " Reversi perft(2)=" << rev
                     << (ok ? "  ok\n" : "  MISMATCH\n");
                rule_tested = true;
                break;
              }
              b = b.passed();
              if (!b.has_moves())
                break; // both stuck: game over
              continue;
            }
            unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
            while (k-- > 0)
              m &= m - 1;
            b = b.play(lsb(m));
          }
        }
        if (!rule_tested)
          out_ << "rule check: no stuck position sampled (skipped)\n";

        out_ << (all_ok ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
      }
    };

  } // namespace

  int uci_loop() {
    Engine().run();
    return 0;
  }

} // namespace islay
