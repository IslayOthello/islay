/**
 * @file uci.cpp
 * @brief A small chess-UCI-flavoured protocol for Othello.
 *
 * The surface is deliberately split in two. RELEASE commands are what UCI.md
 * documents and what a GUI may rely on; DEVELOPMENT commands exist only while
 * `debug on` is in effect and are absent from the documentation. While debug is
 * off they answer with the ordinary "unknown command" error, so a client cannot
 * tell a hidden command from a typo and the shipped protocol is exactly the
 * documented one.
 *
 * Release:
 *   uci                         -> id + options + "uciok"
 *   isready                     -> "readyok"
 *   ucinewgame                  -> reset to the opening, clear all tables
 *   position startpos [moves ..]        set the opening (optionally play moves)
 *   position fen <64> <stm> [moves ..]  set an arbitrary diagram
 *   setoption name <N> value <V>-> set an engine option (e.g. name Rule value Reversi)
 *   go [depth N] [movetime MS] [nodes N] -> search, ends with "bestmove"
 *   go perft <depth> [nocache]  -> perft with bulk-counting (cache on by default)
 *   debug [ on | off ]          -> toggle the development surface below
 *   quit | exit                 -> leave
 *
 * Development (requires `debug on`; see is_debug_command / dispatch_debug):
 *   d | display | board         -> pretty-print the board
 *   bench [depth]               -> timed perft sweep from the opening
 *   test | selftest             -> run every self-check (movegen, eval, search oracle)
 *   match ...                   -> engine-vs-engine A/B harness
 *   features                    -> dump the eval's feature indices for this position
 *   pcdata [n] [maxdepth]       -> CSV (stage,depth,score,standpat) to fit ProbCut against
 *   train [games] [epochs] ...  -> fit pattern weights from self-play
 *   backend                     -> report the active SIMD back-end
 *   searchstats [full]          -> search telemetry for the last `go`
 */
#include "uci.hpp"

#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "board.hpp"
#include "eval.hpp"
#include "match.hpp"
#include "movegen.hpp"
#include "options.hpp"
#include "pattern.hpp"
#include "perft.hpp"
#include "search.hpp"
#include "train.hpp"

namespace islay {
  namespace {

    constexpr const char *kName   = "islay 0.1.0";
    constexpr const char *kAuthor = "islay";

    using Clock = std::chrono::steady_clock;

    [[nodiscard]] double ms_since(Clock::time_point t0) {
      return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
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
          if (cmd == "quit" || cmd == "exit")
            break;
          dispatch(cmd, is);
          std::cout.flush();
        }
      }

    private:
      Board    board_ = Board::start();
      Color    stm_   = Color::Black;
      Options  options_{};
      int      perft_tt_mib_  = 256; // tracks Options::perft_hash_mib so setoption can resize
      int      search_tt_mib_ = 256; // tracks Options::hash_mib
      PerftTT  tt_{256};
      Searcher searcher_{256};
      bool     debug_ = false; // `debug on` unlocks the development commands

      void dispatch(const std::string &cmd, std::istringstream &is) {
        if (cmd == "uci") {
          std::cout << "id name " << kName << '\n' << "id author " << kAuthor << '\n';
          print_option_specs(std::cout);
          std::cout << "uciok\n";
        } else if (cmd == "isready") {
          std::cout << "readyok\n";
        } else if (cmd == "ucinewgame") {
          board_ = Board::start();
          stm_   = Color::Black;
          tt_.clear();
          searcher_.clear(); // TT + killers + history: nothing carries into a new game
        } else if (cmd == "position") {
          cmd_position(is);
        } else if (cmd == "setoption") {
          cmd_setoption(is);
        } else if (cmd == "debug") {
          cmd_debug(is);
        } else if (cmd == "go") {
          cmd_go(is);
        } else if (is_debug_command(cmd)) {
          // Development surface. Hidden unless `debug on`, and when hidden it must be
          // INDISTINGUISHABLE from a typo -- same error as any unknown word -- so the
          // release protocol is exactly what UCI.md documents and nothing more.
          if (!debug_)
            unknown(cmd);
          else
            dispatch_debug(cmd, is);
        } else {
          unknown(cmd);
        }
      }

      void unknown(const std::string &cmd) {
        std::cout << "info error: unknown command '" << cmd << "'\n";
      }

      /** The commands that exist only while `debug on` is in effect. */
      [[nodiscard]] static bool is_debug_command(const std::string &c) noexcept {
        return c == "d" || c == "display" || c == "board" || c == "bench" || c == "test" ||
               c == "selftest" || c == "match" || c == "features" || c == "pcdata" ||
               c == "train" || c == "backend" || c == "searchstats";
      }

      void dispatch_debug(const std::string &cmd, std::istringstream &is) {
        if (cmd == "d" || cmd == "display" || cmd == "board") {
          board_.print(stm_, std::cout);
        } else if (cmd == "bench") {
          cmd_bench(is);
        } else if (cmd == "test" || cmd == "selftest") {
          cmd_test();
        } else if (cmd == "match") {
          cmd_match(is);
        } else if (cmd == "features") {
          cmd_features();
        } else if (cmd == "pcdata") {
          cmd_pcdata(is);
        } else if (cmd == "train") {
          cmd_train(is);
        } else if (cmd == "backend") {
          std::cout << "movegen backend: " << movegen_backend() << '\n';
        } else if (cmd == "searchstats") {
          std::string sub;
          const bool  full = (is >> sub) && sub == "full";
          if (const SearchStats *s = searcher_.stats())
            s->dump(std::cout, full);
          else
            std::cout << "info string search telemetry is compiled out (set kStats=true in search.cpp, rebuild, then `go`)\n";
        }
      }

      // debug [ on | off ] -- the UCI-standard switch. Here it also gates the whole
      // development command surface (see is_debug_command). A bare `debug` reports the
      // current state rather than guessing at one.
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

      // setoption name <Name...> value <Value...>  (both may contain spaces)
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
          // A size change resizes (which also wipes); any other option change
          // still invalidates both tables, because their contents are
          // rule-specific -- the two rules disagree on what a stuck position is
          // worth, so an Othello score is simply wrong under Reversi.
          if (perft_tt_mib_ != options_.perft_hash_mib) {
            perft_tt_mib_ = options_.perft_hash_mib;
            tt_.resize(static_cast<std::size_t>(perft_tt_mib_));
          } else {
            tt_.clear();
          }
          if (search_tt_mib_ != options_.hash_mib) {
            search_tt_mib_ = options_.hash_mib;
            searcher_.resize(static_cast<std::size_t>(search_tt_mib_));
          } else {
            searcher_.clear();
          }
          std::string lname = name;
          for (char &ch: lname)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
          pattern_set_stage_interp(options_.stage_interp); // keep the eval flag in sync
          if (lname == "evalfile") {
            // Empty value UNLOADS to the hand-written eval; a failed load must not leave
            // stale weights active, so it also unloads (a defined state, not the old one).
            if (options_.eval_file.empty())
              pattern_weights().unload();
            else if (!pattern_weights().load(options_.eval_file, std::cout))
              pattern_weights().unload();
          }
          std::cout << "info string option " << name << " = " << value << '\n';
        } else {
          std::cout << "info error: unknown option or invalid value: '" << name << "' = '" << value << "'\n";
        }
      }

      // position startpos [moves ...] | position fen <diagram> <stm> [moves ...]
      // Fully TRANSACTIONAL: the base position AND every move are validated on a
      // temporary board; `board_`/`stm_` are assigned only when the whole command is
      // legal, so any error leaves the previous position untouched.
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
              // A pass is legal only when the mover is stuck AND (Othello) the opponent
              // can still move; if neither can move the game is over -- no pass.
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

      // go [depth N] [movetime MS] [nodes N] | go perft <depth> [nocache]
      // `perft` is an argument of `go`, not a separate command, so it stays on the
      // release surface with the rest of `go`.
      void cmd_go(std::istringstream &is) {
        std::string tok;
        is >> tok;
        if (tok == "perft") {
          int depth = 0;
          if (!(is >> depth)) {
            std::cout << "info error: 'go perft' needs a depth\n";
            return;
          }
          bool use_cache = true;
          if (is >> tok)
            use_cache = (tok != "nocache");
          run_perft(depth, use_cache);
          return;
        }

        // Validate each limit; a malformed or out-of-range value REJECTS the command
        // rather than silently starting an unlimited/exact search. `tok` already holds
        // the first keyword (read above), so process it before pulling the next one.
        SearchLimits lim;
        for (; !tok.empty(); tok.clear()) {
          if (tok == "depth") {
            long v;
            if (!(is >> v) || v < 1) {
              std::cout << "info error: depth must be positive\n";
              return;
            }
            lim.depth = static_cast<int>(v);
          } else if (tok == "movetime") {
            double v;
            if (!(is >> v) || v < 0.0) {
              std::cout << "info error: movetime must be non-negative\n";
              return;
            }
            lim.movetime_ms = v;
          } else if (tok == "nodes") {
            long long v;
            if (!(is >> v) || v < 0) {
              std::cout << "info error: nodes must be non-negative\n";
              return;
            }
            lim.nodes = static_cast<std::uint64_t>(v);
          } else if (tok == "infinite") {
            // The command loop is a blocking getline: with no search thread there is no
            // way to read `stop`, so this would hang for good.
            std::cout << "info string 'go infinite' is not supported; use depth/movetime/nodes\n";
          }
          if (!(is >> tok))
            break;
        }
        if (lim.depth == 0 && lim.nodes == 0 && lim.movetime_ms == 0.0)
          lim.depth = 8; // a bare `go` still has to return a move

        run_search(lim);
      }

      void run_search(const SearchLimits &lim) {
        const SearchResult r = searcher_.search(board_, lim, options_.rule, stm_, std::cout);
        if (r.best == NOMOVE) {
          // Reversi with no move, or both sides stuck: there is nothing to play.
          std::cout << "info string game over (final score " << r.score / 100 << ")\n"
                    << "bestmove --\n";
          return;
        }
        std::cout << "info string " << (r.exact ? "exact" : "heuristic") << " score, depth " << r.depth << '\n'
                  << "bestmove " << square_to_string(r.best) << '\n';
        if (const SearchStats *s = searcher_.stats()) // only non-null in a kStats build
          s->dump(std::cout, false);
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
          // Only Othello passes; under Reversi no move means the game is over.
          const std::uint64_t cnt = count_child(board_.passed());
          lines << "pass: " << cnt << '\n';
          total = cnt;
        } else {
          lines << "(game over)\n";
        }

        const double dt = ms_since(t0);
        std::cout << lines.str() << '\n'
                  << "Nodes searched: " << total << '\n'
                  << "Time: " << static_cast<std::uint64_t>(dt) << " ms\n"
                  << "Speed: " << nps_string(total, dt) << " N/s\n";
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

      // match [pairs] [depth] [evalA] [evalB]  -- "" or "-" means the hand-written eval
      // match pc [pairs] [movetime_ms]         -- ProbCut on vs off, same eval
      //
      // The `pc` form takes a movetime, not a depth, on purpose: a pruning technique
      // only ever removes work, so at equal depth it can only lose, and its entire
      // case is the depth that the saved time buys back. Equal depth would answer a
      // question nobody asked. LMP is why this rule is written down: it cut nodes
      // 55% and cost 154 Elo.
      void cmd_match(std::istringstream &is) {
        MatchConfig cfg;
        cfg.rule = options_.rule;

        std::string first;
        is >> first;
        if (first == "pc") {
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1)
            cfg.pairs = 50;
          if (!(is >> ms) || ms < 1)
            ms = 50;
          cfg.depth       = 0;
          cfg.movetime_ms = ms;
          cfg.pc_a        = true;
          cfg.pc_b        = false;
          // Both sides get the SAME eval -- otherwise this measures the eval, not
          // ProbCut. It takes an argument because ProbCut's whole viability hangs on
          // eval quality, so testing it against the hand-written eval answers a
          // question about the wrong engine.
          std::string ev;
          if (is >> ev && ev != "-")
            cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "et") { // EVAL vs EVAL at equal TIME -- the honest test for a feature
          // that costs search speed. The fixed-depth form measures only what the eval
          // KNOWS; this one also charges it for what it costs to know it.
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          std::string ea, eb;
          if (is >> ea && ea != "-") cfg.eval_a = ea;
          if (is >> eb && eb != "-") cfg.eval_b = eb;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "pcg4") { // 4-ply ProbCut probe gap at deep nodes vs the usual 2, equal time
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.pcg4_a = true; cfg.pcg4_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "lmrc") { // CALIBRATED LMR table vs the old fixed rule, equal time
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.lmrc_a = true; cfg.lmrc_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "pcg") { // ProbCut probe GATE on vs off, equal time (both keep ProbCut on)
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.pc_a = cfg.pc_b = true;   // both prune; only the gate differs
          cfg.pcg_a = true; cfg.pcg_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "si") { // stage interpolation ON vs OFF, FIXED DEPTH (it changes the eval)
          int d = 6;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> d) || d < 1) d = 6;
          cfg.depth = d; cfg.movetime_ms = 0.0;
          cfg.si_a = true; cfg.si_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "egtc") { // endgame on vs off under a CLOCK time control (base_ms + inc_ms)
          double base = 8000.0, inc = 80.0;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          is >> base; is >> inc;
          cfg.depth = 0; cfg.movetime_ms = 0.0;
          cfg.tc_base_ms = base; cfg.tc_inc_ms = inc;
          cfg.eg_a = true; cfg.eg_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "egd") { // endgame on vs off, FIXED DEPTH -- must be ~50%: the stack is EXACT
          int d = 12;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> d) || d < 1) d = 12;
          cfg.depth = d; cfg.movetime_ms = 0.0;
          cfg.eg_a = true; cfg.eg_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "eg") { // endgame stack on vs off, equal time (measures its exact-speedup Elo)
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.eg_a = true; cfg.eg_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "lmp") { // late move pruning on vs off, equal time (re-test on trained eval)
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.lmp_a = true; cfg.lmp_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "pct") { // ProbCut t-sweep: A uses t_a, B uses t_b, both on, equal time
          double ta = 1.0, tb = 1.5; int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          is >> ta; is >> tb;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.pct_a = static_cast<float>(ta); cfg.pct_b = static_cast<float>(tb);
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "mpc") { // per-stage (Multi) ProbCut vs pooled per-depth, equal time
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1)
            cfg.pairs = 50;
          if (!(is >> ms) || ms < 1)
            ms = 50;
          cfg.depth       = 0;
          cfg.movetime_ms = ms;
          cfg.mpc_a       = true;  // A = per-stage
          cfg.mpc_b       = false; // B = pooled per-depth
          std::string ev;
          if (is >> ev && ev != "-")
            cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "lmr") { // LMR on vs off, equal time, same eval -- same rationale as pc
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1)
            cfg.pairs = 50;
          if (!(is >> ms) || ms < 1)
            ms = 50;
          cfg.depth       = 0;
          cfg.movetime_ms = ms;
          cfg.lmr_a       = true;
          cfg.lmr_b       = false;
          std::string ev;
          if (is >> ev && ev != "-")
            cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        cfg.pairs = first.empty() ? 0 : std::atoi(first.c_str());
        cfg.si_a = cfg.si_b = options_.stage_interp; // both sides use the engine's interp setting
        is >> cfg.depth;
        std::string a, b;
        if (is >> a && a != "-")
          cfg.eval_a = a;
        if (is >> b && b != "-")
          cfg.eval_b = b;
        std::uint64_t sd = 0;
        if (is >> sd)
          cfg.seed = sd; // deterministic openings + result
        if (cfg.pairs < 1)
          cfg.pairs = 50;
        if (cfg.depth < 1)
          cfg.depth = 6;
        run_match(cfg, std::cout);
      }

      // Training hook: the flat weight indices this position contributes to. The
      // eval is their weights summed, so a design row is exactly these indices --
      // a tuner needs nothing else from the engine.
      void cmd_features() {
        std::uint32_t idx[64];
        const int     n = pattern_features(board_, stm_, idx);
        std::cout << "stage " << pattern_stage(board_.count()) << " weights/stage " << pattern_weights_per_stage()
                  << " active " << (pattern_enabled() ? "yes" : "no (hand-written eval in use)") << '\n'
                  << "features";
        for (int i = 0; i < n; ++i)
          std::cout << ' ' << idx[i];
        std::cout << '\n';
      }

      /**
       * ProbCut data hook: `pcdata [positions] [maxdepth]` prints one CSV row per
       * (position, depth) as `stage,depth,score`.
       *
       * This exists because ProbCut is the one selective technique here whose
       * parameters are FITTED rather than guessed: it predicts the depth-d score
       * from a depth-(d-r) one, and the prediction is only worth making if the
       * relationship is actually measured on THIS engine's eval. Four hand-tuned
       * techniques were already rejected on measurement; this one gets its numbers
       * from data or it does not go in.
       *
       * Iterative deepening already computes every intermediate depth on the way to
       * `maxdepth`, so one search yields a whole column of (d', d) pairs for free.
       * The scores are read back out of the engine's own info lines -- the same
       * surface an external tuner would use, which keeps this a hook and not a
       * second implementation that could drift from the real search.
       */
      void cmd_pcdata(std::istringstream &is) {
        int           n = 200, maxd = 12;
        std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
        if (!(is >> n) || n < 1)
          n = 200;
        if (!(is >> maxd) || maxd < 2)
          maxd = 12;

        const auto next = [&seed]() noexcept {
          seed ^= seed << 13;
          seed ^= seed >> 7;
          seed ^= seed << 17;
          return seed;
        };

        std::cout << "stage,depth,score,standpat\n";
        Searcher s(64);
        int      emitted = 0;
        for (int i = 0; i < n; ++i) {
          // A random playout to a random ply: the fit has to cover every stage,
          // because the shallow-to-deep relationship is not the same in the opening
          // as it is at move 50.
          Board b     = Board::start();
          Color stm   = Color::Black;
          const int plies = 4 + static_cast<int>(next() % 46);
          bool  dead  = false;
          for (int k = 0; k < plies; ++k) {
            Bitboard m = b.moves();
            if (m == 0) {
              const Board p = b.passed();
              if (!p.has_moves()) {
                dead = true;
                break;
              }
              b   = p;
              stm = ~stm;
              continue;
            }
            unsigned pick = static_cast<unsigned>(next() % static_cast<unsigned>(popcount(m)));
            while (pick-- > 0)
              m &= m - 1;
            b   = b.play(lsb(m));
            stm = ~stm;
          }
          if (dead || b.moves() == 0)
            continue;

          s.clear(); // each position independent: no table carries between samples
          std::ostringstream sink;
          s.search(b, SearchLimits{maxd, 0, 0.0}, options_.rule, stm, sink);

          const int   stage    = pattern_stage(b.count());
          const int   standpat = s.static_eval(b, stm); // depth-0 eval, mover-relative
          std::istringstream lines(sink.str());
          std::string        line;
          while (std::getline(lines, line)) {
            std::istringstream ls(line);
            std::string        tok;
            int                d = 0, sc = 0;
            bool               have_d = false, have_s = false;
            while (ls >> tok) {
              if (tok == "depth" && (ls >> d))
                have_d = true;
              else if (tok == "cd" && (ls >> sc))
                have_s = true;
            }
            if (have_d && have_s) {
              std::cout << stage << ',' << d << ',' << sc << ',' << standpat << '\n';
              ++emitted;
            }
          }
        }
        std::cout << "info string pcdata: " << emitted << " rows\n";
      }

      // train [games] [epochs] [depth] [solve_empties] [lr] [l2] [mob0/1] [c2x50/1] [out]
      void cmd_train(std::istringstream &is) {
        TrainConfig cfg;
        cfg.rule = options_.rule;
        if (!(is >> cfg.games) || cfg.games < 1)
          cfg.games = 50000;
        if (!(is >> cfg.epochs) || cfg.epochs < 1)
          cfg.epochs = 8;
        if (!(is >> cfg.depth) || cfg.depth < 1)
          cfg.depth = 4; // measured strength default (d2->d4 = +81 Elo)
        if (!(is >> cfg.solve_empties) || cfg.solve_empties < 0)
          cfg.solve_empties = 12;
        if (!(is >> cfg.lr) || cfg.lr <= 0.0)
          cfg.lr = 0.0005;
        if (!(is >> cfg.l2) || cfg.l2 < 0.0)
          cfg.l2 = 1e-6;
        int mob = 1, c2x5 = 1, stab = 1; // gate each feature so its value can be A/B'd cleanly
        if (is >> mob)
          cfg.use_mobility = (mob != 0);
        if (is >> c2x5)
          cfg.use_c2x5 = (c2x5 != 0);
        if (is >> stab)
          cfg.use_stab = (stab != 0);
        std::string out;
        if (is >> out && !out.empty())
          cfg.out = out;
        std::uint64_t sd = 0;
        if (is >> sd)
          cfg.seed = sd; // deterministic self-play + shuffles
        int itp = 0;
        if (is >> itp)
          cfg.interp = (itp != 0); // fit for StageInterpolation
        // The teacher must be the eval as it stands, not a half-loaded set: an
        // EvalFile left over from an earlier setoption would quietly change what is
        // being bootstrapped from.
        run_train(cfg, std::cout);
      }

      void cmd_test() {
        std::cout << "movegen self-test (" << movegen_backend() << ") ... " << std::flush;
        if (!movegen_selftest()) {
          std::cout << "FAILED\n";
          return;
        }
        std::cout << "ok\n";

        std::cout << "eval self-test ... " << std::flush;
        const bool eval_ok = eval_selftest();
        std::cout << (eval_ok ? "ok\n" : "FAILED\n");

        std::cout << "search self-test (vs plain negamax oracle) ... " << std::flush;
        const bool search_ok = search_selftest();
        std::cout << (search_ok ? "ok\n" : "FAILED\n");

        std::cout << "pattern self-test (incremental vs scratch) ... " << std::flush;
        const bool pattern_ok = pattern_selftest();
        std::cout << (pattern_ok ? "ok\n" : "FAILED\n");

        constexpr std::array<std::uint64_t, 9> known{0, 4, 12, 56, 244, 1396, 8200, 55092, 390216};
        const Board                            start  = Board::start();
        bool                                   all_ok = eval_ok && search_ok && pattern_ok;
        for (int d = 1; d <= 8; ++d) {
          const std::uint64_t got = perft(start, d, Rule::Othello);
          const bool          ok  = (got == known[static_cast<std::size_t>(d)]);
          all_ok                  = all_ok && ok;
          std::cout << "perft(" << d << ") = " << got << (ok ? "  ok\n" : "  MISMATCH\n");
        }

        // Cached and uncached perft must agree.
        PerftTT    tt(64);
        const bool cache_ok = (perft(start, 8, Rule::Othello) == perft_cached(start, 8, tt, Rule::Othello));
        all_ok              = all_ok && cache_ok;
        std::cout << "cache consistency perft(8): " << (cache_ok ? "ok" : "MISMATCH") << '\n';

        // Symmetry: perft is invariant under all 8 rotations/mirrors, and the
        // symmetry-aware cache must still match the uncached count.
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

        // Rule: at a position where the mover is stuck but the opponent can move,
        // Othello passes (perft > 0) while Reversi ends the game (perft == 0).
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
