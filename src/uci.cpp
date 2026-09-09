// Public and debug-only commands are documented in UCI.md.
#include "uci.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board.hpp"
#include "movegen.hpp"
#include "options.hpp"
#include "perft.hpp"

namespace islay {
  namespace {

    constexpr const char *kName   = "islay 0.1.0";
    constexpr const char *kAuthor = "islay";

    using Clock = std::chrono::steady_clock;

    [[nodiscard]] double ms_since(Clock::time_point t0) {
      return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    [[nodiscard]] std::string grouped_count(std::string digits) {
      for (std::size_t pos = digits.size(); pos > 3;) {
        pos -= 3;
        digits.insert(pos, 1, ',');
      }
      return digits;
    }

    [[nodiscard]] std::string nps_string(std::uint64_t nodes, double ms) {
      if (ms <= 0.0)
        return "inf";
      std::ostringstream os;
      os.setf(std::ios::fixed);
      os.precision(0);
      os << nodes / (ms / 1000.0);
      return os.str();
    }

    [[nodiscard]] Color color_of(char stm) noexcept {
      return (stm == 'O' || stm == 'o' || stm == 'W' || stm == 'w') ? Color::White : Color::Black;
    }

    class Engine {
    public:
      void run() {
        std::ios::sync_with_stdio(false);
        std::string line;
        while (std::getline(std::cin, line)) {
          std::istringstream is(line);
          std::string        cmd;
          if (!(is >> cmd))
            continue;
          if (cmd == "quit" || cmd == "exit") {
            break;
          }
          dispatch(cmd, is);
          std::cout.flush();
        }
      }

    private:
      Board   board_ = Board::start();
      Color   stm_   = Color::Black;
      Options options_{};
      int     perft_tt_mib_ = 256; // tracks Options::perft_hash_mib so setoption can resize
      PerftTT tt_{256};
      bool    debug_ = false; // `debug on` unlocks the development commands

      void dispatch(const std::string &cmd, std::istringstream &is) {
        if (cmd == "stop") {
          // Perft is synchronous; stop is accepted as a no-op.
          return;
        }
        if (cmd == "isready") {
          std::cout << "readyok\n";
          return;
        }
        if (cmd == "debug") {
          cmd_debug(is);
          return;
        }

        if (cmd == "uci") {
          std::cout << "id name " << kName << '\n' << "id author " << kAuthor << '\n';
          print_option_specs(std::cout);
          std::cout << "uciok\n";
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

      void unknown(const std::string &cmd) { std::cout << "info error: unknown command '" << cmd << "'\n"; }

      [[nodiscard]] static bool is_debug_command(const std::string &c) noexcept {
        return c == "d" || c == "display" || c == "board" || c == "bench" || c == "test" || c == "selftest" ||
               c == "backend";
      }

      void dispatch_debug(const std::string &cmd, std::istringstream &is) {
        if (cmd == "d" || cmd == "display" || cmd == "board") {
          board_.print(stm_, std::cout);
        } else if (cmd == "bench") {
          cmd_bench(is);
        } else if (cmd == "test" || cmd == "selftest") {
          cmd_test();
        } else if (cmd == "backend") {
          std::cout << "movegen backend: " << movegen_backend() << '\n';
        }
      }

      void cmd_debug(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok)) {
          std::cout << "info string debug " << (debug_ ? "on" : "off") << '\n';
          return;
        }
        if (tok == "on") {
          debug_ = true;
        } else if (tok == "off") {
          debug_ = false;
        } else {
          std::cout << "info error: expected 'debug on' or 'debug off'\n";
          return;
        }
        std::cout << "info string debug " << (debug_ ? "on" : "off") << '\n';
      }

      void cmd_setoption(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok) || tok != "name") {
          std::cout << "info error: expected 'setoption name <Name> value <Value>'\n";
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
          std::cout << "info string option " << name << " = " << value << '\n';
        } else {
          std::cout << "info error: unknown option or invalid value: '" << name << "' = '" << value << "'\n";
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
            std::cout << "info error: 'position fen' needs <diagram> <stm>\n";
            return;
          }
          if (!tb.set(diagram, stmtok.empty() ? '?' : stmtok[0])) {
            std::cout << "info error: invalid diagram/side-to-move\n";
            return;
          }
          ts = color_of(stmtok[0]);
        } else {
          std::cout << "info error: expected 'startpos' or 'fen'\n";
          return;
        }

        if ((is >> tok) && tok == "moves") {
          std::string mv;
          while (is >> mv) {
            const Square sq = parse_square(mv);
            if (sq == PASS) {
              if (options_.rule != Rule::Othello || tb.has_moves() || !tb.passed().has_moves()) {
                std::cout << "info error: illegal pass\n";
                return;
              }
              tb = tb.passed();
              ts = ~ts;
            } else if (sq != NOMOVE && (tb.moves() & square_bb(sq))) {
              tb = tb.play(sq);
              ts = ~ts;
            } else {
              std::cout << "info error: illegal move '" << mv << "'\n";
              return;
            }
          }
        }
        board_ = tb; // commit only after the whole command validated
        stm_   = ts;
      }

      void cmd_go(std::istringstream &is) {
        std::string tok;
        if (!(is >> tok) || tok != "perft") {
          std::cout << "info error: only 'go perft <depth> [nocache]' is supported\n";
          return;
        }
        int depth = 0;
        if (!(is >> depth) || depth < 0) {
          std::cout << "info error: 'go perft' needs a non-negative integer depth\n";
          return;
        }
        bool use_cache = true;
        if (is >> tok) {
          if (tok != "nocache") {
            std::cout << "info error: expected 'go perft <depth> [nocache]'\n";
            return;
          }
          use_cache = false;
          if (is >> tok) {
            std::cout << "info error: unexpected perft argument '" << tok << "'\n";
            return;
          }
        }
        run_perft(depth, use_cache);
      }

      void run_perft(int depth, bool use_cache) {
        if (depth < 1) {
          std::cout << "Nodes searched: 1\nTime: 0 ms\n";
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

        const double dt = ms_since(t0);
        std::cout << lines.str() << '\n'
                  << "Nodes searched: " << grouped_count(std::to_string(total)) << '\n'
                  << "Time: " << static_cast<std::uint64_t>(dt) << " ms\n"
                  << "Speed: " << grouped_count(nps_string(total, dt)) << " N/s\n";
      }

      void cmd_bench(std::istringstream &is) {
        int maxd = 11;
        is >> maxd;
        const Board start = Board::start();
        std::cout << "depth            nodes       time(ms)             nps\n"
                  << "---------------------------------------------------------\n";
        for (int d = 1; d <= maxd; ++d) {
          const auto          t0 = Clock::now();
          const std::uint64_t n  = perft(start, d, options_.rule);
          const double        dt = ms_since(t0);
          std::cout.width(5);
          std::cout << d << ' ';
          std::cout.width(16);
          std::cout << n << ' ';
          std::cout.width(14);
          std::cout << static_cast<std::uint64_t>(dt) << ' ';
          std::cout.width(15);
          std::cout << nps_string(n, dt) << '\n';
        }
      }

      void cmd_test() {
        std::cout << "output formatting self-test ... " << std::flush;
        if (grouped_count("0") != "0" || grouped_count("1") != "1" || grouped_count("999") != "999" ||
            grouped_count("1000") != "1,000" || grouped_count("999999") != "999,999" ||
            grouped_count("1000000") != "1,000,000" ||
            grouped_count("18446744073709551615") != "18,446,744,073,709,551,615" ||
            grouped_count(nps_string(1, 0)) != "inf" || grouped_count(nps_string(1234567, 1000)) != "1,234,567") {
          std::cout << "FAILED\n";
          return;
        }
        std::cout << "ok\n";

        std::cout << "movegen self-test (" << movegen_backend() << ") ... " << std::flush;
        if (!movegen_selftest()) {
          std::cout << "FAILED\n";
          return;
        }
        std::cout << "ok\n";

        constexpr std::array<std::uint64_t, 9> known{0, 4, 12, 56, 244, 1396, 8200, 55092, 390216};
        const Board                            start  = Board::start();
        bool                                   all_ok = true;
        for (int d = 1; d <= 8; ++d) {
          const std::uint64_t got = perft(start, d, Rule::Othello);
          const bool          ok  = (got == known[static_cast<std::size_t>(d)]);
          all_ok                  = all_ok && ok;
          std::cout << "perft(" << d << ") = " << got << (ok ? "  ok\n" : "  MISMATCH\n");
        }

        PerftTT    tt(64);
        const bool cache_ok = (perft(start, 8, Rule::Othello) == perft_cached(start, 8, tt, Rule::Othello));
        all_ok              = all_ok && cache_ok;
        std::cout << "cache consistency perft(8): " << (cache_ok ? "ok" : "MISMATCH") << '\n';

        const Board         asym   = Board::start().play(parse_square("d3")).play(parse_square("c3"));
        const std::uint64_t base   = perft(asym, 6, Rule::Othello);
        bool                sym_ok = true;
        for (int s = 0; s < 8; ++s)
          sym_ok = sym_ok && (perft(asym.symmetry(s), 6, Rule::Othello) == base);
        PerftTT    tt2(64);
        const bool symcache_ok = (perft_cached(asym, 7, tt2, Rule::Othello) == perft(asym, 7, Rule::Othello));
        all_ok                 = all_ok && sym_ok && symcache_ok;
        std::cout << "symmetry invariance perft(6): " << (sym_ok ? "ok" : "MISMATCH") << '\n'
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
        std::cout << "multi-position perft(5), both rules, symmetry and 1 MiB cache: "
                  << (positions_ok ? "ok" : "MISMATCH") << '\n';

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
                std::cout << "rule at stuck position: Othello perft(2)=" << oth << " Reversi perft(2)=" << rev
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
          std::cout << "rule check: no stuck position sampled (skipped)\n";

        std::cout << (all_ok ? "ALL TESTS PASSED\n" : "TESTS FAILED\n");
      }
    };

  } // namespace

  int uci_loop() {
    Engine().run();
    return 0;
  }

} // namespace islay
