// Public and debug-only commands are documented in UCI.md.
#include "uci.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "board.hpp"
#include "game.hpp"
#include "mcts.hpp"
#include "movegen.hpp"
#include "options.hpp"
#include "perft.hpp"
#include "search_controller.hpp"

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
          if (cmd == "position" || cmd == "ucinewgame" || cmd == "setoption" || cmd == "go" ||
              (debug_ && (cmd == "test" || cmd == "selftest" || cmd == "bench")))
            search_.cancel();
          dispatch(cmd, is);
          flush_output();
        }
        search_.cancel();
      }

    private:
      Board              board_ = Board::start();
      Color              stm_   = Color::Black;
      Options            options_{};
      int                perft_tt_mib_ = 256; // tracks Options::perft_hash_mib so setoption can resize
      PerftTT            tt_{256};
      bool               debug_ = false; // `debug on` unlocks the development commands
      std::ostringstream out_;
      std::mutex         output_mutex_;
      SearchController   search_{std::cout, output_mutex_};

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
        } else if (cmd == "position") {
          cmd_position(is);
        } else if (cmd == "setoption") {
          cmd_setoption(is);
        } else if (cmd == "go") {
          cmd_go(is);
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
        const std::string name  = join(name_toks);
        const std::string value = join(value_toks);
        if (apply_option(options_, name, value)) {
          // Option changes invalidate rule-dependent caches.
          if (perft_tt_mib_ != options_.perft_hash_mib) {
            perft_tt_mib_ = options_.perft_hash_mib;
            tt_.resize(static_cast<std::size_t>(perft_tt_mib_));
          } else {
            tt_.clear();
          }
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

        if ((is >> tok) && tok == "moves") {
          std::string mv;
          while (is >> mv) {
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
        limits.simulations  = std::numeric_limits<std::uint64_t>::max();
        limits.tree_bytes   = static_cast<std::size_t>(options_.mcts_hash_mib) * 1024 * 1024;
        bool          nodes = false, timed = false, infinite = false;
        std::uint64_t milliseconds = 0;
        do {
          if (tok == "infinite" && !nodes && !timed && !infinite) {
            infinite = true;
            continue;
          }
          if (infinite || (tok != "nodes" && tok != "movetime") || (tok == "nodes" ? nodes : timed)) {
            out_ << "info error: expected go nodes <N> [movetime <MS>], movetime <MS>, or infinite\n";
            return;
          }
          std::string   value;
          std::uint64_t number = 0;
          if (!(is >> value)) {
            out_ << "info error: missing search limit\n";
            return;
          }
          const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
          if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
              (tok == "movetime" && number > 86'400'000)) {
            out_ << "info error: invalid search limit (movetime maximum 86400000 ms)\n";
            return;
          }
          if (tok == "nodes") {
            nodes              = true;
            limits.simulations = number;
          } else {
            timed        = true;
            milliseconds = number;
          }
        } while (is >> tok);
        if (options_.rule != Rule::Othello) {
          out_ << "info error: MCTS supports Rule=Othello only\n";
          return;
        }
        out_ << "info string evaluator uniform (P2 scaffold; no trained network)\n";
        flush_output();
        if (timed)
          limits.deadline = Clock::now() + std::chrono::milliseconds(milliseconds);
        try {
          search_.start(board_, options_.rule, limits, infinite);
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
        out_ << "PUCT core self-test ... " << std::flush;
        if (!mcts_selftest()) {
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
