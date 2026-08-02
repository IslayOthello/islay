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
 *   stop                        -> end the search in progress (it still emits bestmove)
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
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "board.hpp"
#include "book.hpp"
#include "eval.hpp"
#include "match.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "options.hpp"
#include "pattern.hpp"
#include "perft.hpp"
#include "search.hpp"
#include "train.hpp"

namespace islay {
  namespace {

    // ABDADA work deferral under lazy SMP: mark subtrees being searched in a shared
    // busy table; other threads defer a busy child to the end of the node instead of
    // duplicating it. MEASURED AND OFF: -6.5 Elo, 95% CI [-20, 7], 700 games at equal
    // time against plain 4-thread SMP -- the FOURTH parallel-shaping idea to return
    // nothing (after three divergence schemes). Same root cause every time: this tree
    // cuts on the first move 82.7% of the time at branching 2.27, so the non-first
    // children whose duplication ABDADA prevents are tiny scouts not worth deferring,
    // while its bookkeeping (and a full sort where lazy selection used to stop at the
    // first cutoff) is paid at every deep node. Parallel search here is capped by the
    // narrowness of the tree, not by how cleverly the threads are arranged.
    constexpr bool kUseAbdada = false;

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

    /**
     * The search runs on its own thread while the command loop keeps reading stdin, so
     * two threads can reach stdout at once. Whole lines may interleave harmlessly; half
     * an `info` line spliced into `readyok` breaks a GUI. So the search writes through
     * this buffer, which accumulates until a newline and then emits the COMPLETE line
     * under a shared mutex, and the command loop takes the same mutex for the handful
     * of replies it can still emit while a search is running.
     */
    std::mutex g_out_mu;

    class LineSyncBuf final : public std::streambuf {
    public:
      ~LineSyncBuf() override { if (!buf_.empty()) flush_line(); }

    protected:
      int overflow(int c) override {
        if (c == traits_type::eof())
          return traits_type::not_eof(c);
        buf_.push_back(static_cast<char>(c));
        if (c == '\n')
          flush_line();
        return c;
      }
      std::streamsize xsputn(const char *sp, std::streamsize n) override {
        for (std::streamsize i = 0; i < n; ++i)
          overflow(traits_type::to_int_type(sp[i]));
        return n;
      }
      int sync() override { return 0; } // a line is the unit; partial flushes would defeat it

    private:
      void flush_line() {
        const std::lock_guard<std::mutex> lk(g_out_mu);
        std::cout << buf_;
        std::cout.flush();
        buf_.clear();
      }
      std::string buf_;
    };

    /** One whole line to stdout, atomically against the search thread. */
    void say(const std::string &line) {
      const std::lock_guard<std::mutex> lk(g_out_mu);
      std::cout << line;
      std::cout.flush();
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
            stop_search_and_join(/*abandon=*/false); // let a bounded search finish; see above
            break;
          }
          dispatch(cmd, is);
        }
        // Leaving with a thread still running would terminate(); end-of-input is a
        // legitimate way to quit, so it has to unwind exactly like `quit`.
        stop_search_and_join(/*abandon=*/false);
      }

      ~Engine() { stop_search_and_join(); }

    private:
      Board    board_ = Board::start();
      Color    stm_   = Color::Black;
      Options  options_{};
      int      perft_tt_mib_  = 256; // tracks Options::perft_hash_mib so setoption can resize
      int      search_tt_mib_ = 256; // tracks Options::hash_mib
      PerftTT  tt_{256};
      Searcher searcher_{256};
      Book     book_{};
      bool     debug_ = false; // `debug on` unlocks the development commands

      // At most one search at a time. `joinable()` IS the "is a search running" flag --
      // a separate bool could disagree with reality, this cannot.
      std::thread search_thread_;
      bool        search_infinite_ = false; // only ever touched by the command thread

      /**
       * LAZY SMP. The helpers do NOT divide the tree -- they run the same iterative
       * deepening on the same root, and everything they discover lands in the SHARED
       * transposition table, where the main thread finds it already answered. That
       * sharing is the entire mechanism; helpers with private tables would be N
       * independent searches and worth nothing.
       *
       * Each helper needs its own Searcher because killers, history, the pattern stack
       * and the node counter are per-thread state -- only the table is shared. They are
       * built once and kept, because constructing one allocates that per-thread state.
       *
       * A small depth offset stops them all grinding the identical iteration at the same
       * instant; from there the table itself keeps them diverged.
       *
       * THREE DIVERGENCE SCHEMES WERE TRIED. Two neutral, one negative -- do not reach
       * for a fourth without new evidence:
       *   per-node move-order jitter  +2.8 Elo, 95% CI [-12, 17], 500 games
       *   root-only move-order jitter +2.3 Elo, 95% CI [-11, 16], 600 games
       *   WIDER helpers (t=2.5 / gate off / full-width PVS)  -26.7 Elo, CI [-42,-12], 600 games
       * The jitter results carry a warning about the ordering here: perturbing it at every
       * node cost 39% MORE nodes at fixed depth for a jitter of only +-3, and up to 73%
       * for larger ones, on an erratic non-monotone profile. Near-ties are common and
       * breaking them makes each helper markedly worse at its own job. The root-only form
       * avoids that cost -- one node, a handful of moves -- and still gained nothing.
       *
       * The WIDE-helper attempt was the sharpest lesson. The idea was to have helpers
       * prune less so they explore the parts of the tree the main search cut, which
       * targets the real reason SMP under-delivers: the tree is so narrow (branching
       * 2.27, a first-move cutoff 82.7% of the time) that identically-configured helpers
       * mostly re-do the main thread's work. But wider = more nodes per ply, so the
       * helpers, sharing the same cores, STARVE the main thread -- it reached depth 18
       * where plain SMP reached 19 at the same time -- and the played move comes from the
       * main thread. Helpers must be at least as CHEAP as the main search, never more
       * expensive. Divergence is not the lever here; the shared table's +23.5 is the
       * ceiling the tree's narrowness allows.
       */
      std::vector<std::unique_ptr<Searcher>> helpers_;

      void size_helpers() {
        const std::size_t want = options_.threads > 1 ? static_cast<std::size_t>(options_.threads - 1) : 0;
        if (helpers_.size() == want && (want == 0 || helpers_[0]->shares_table_with(searcher_)))
          return;
        helpers_.clear();
        for (std::size_t i = 0; i < want; ++i) {
          auto h = std::make_unique<Searcher>(1); // tiny private table, dropped on the next line
          h->share_table_with(searcher_);
          h->set_bump_age(false);          // one generation per search, owned by the main thread
          h->set_depth_offset(static_cast<int>(i % 3));
          helpers_.push_back(std::move(h));
        }
      }

      /**
       * `abandon` distinguishes the two reasons a search ends early.
       *
       * TRUE for `stop` and for any command that takes the engine state over: the
       * result is being thrown away, so cut it off now.
       *
       * FALSE for `quit` and end-of-input, where a BOUNDED search is allowed to finish
       * first. That is not politeness -- every batch script in this project is
       * `printf '... go depth N\nquit\n' | islay`, and cutting the search off there
       * would silently truncate it and hand back a shallower result that still looks
       * well-formed. A bounded search always terminates, so waiting is safe; an
       * unlimited one never would, so it is still stopped.
       */
      void stop_search_and_join(bool abandon = true) {
        if (!search_thread_.joinable())
          return;
        if (abandon || search_infinite_)
          searcher_.request_stop();
        search_thread_.join(); // the join is also what publishes the search's writes
      }

      /**
       * `stop`, `isready` and `debug` are the only commands that may be answered WHILE a
       * search is running -- that is the entire point of `stop`, and `isready` is
       * defined to reply at once. Everything else reads or mutates state the search
       * thread is using (board_, options_, the table), so it first stops and joins.
       * A GUI should not send those mid-search anyway; stopping is the forgiving
       * response, and it is what keeps the shared state race-free by construction.
       */
      void dispatch(const std::string &cmd, std::istringstream &is) {
        if (cmd == "stop") {
          stop_search_and_join();
          return;
        }
        if (cmd == "isready") {
          say("readyok\n"); // must answer during a search, so it does not join
          return;
        }
        if (cmd == "debug") {
          cmd_debug(is);
          return;
        }
        stop_search_and_join(); // every remaining command owns the engine state

        if (cmd == "uci") {
          std::cout << "id name " << kName << '\n' << "id author " << kAuthor << '\n';
          print_option_specs(std::cout);
          std::cout << "uciok\n";
        } else if (cmd == "ucinewgame") {
          board_ = Board::start();
          stm_   = Color::Black;
          tt_.clear();
          searcher_.clear(); // TT + killers + history: nothing carries into a new game
        } else if (cmd == "position") {
          cmd_position(is);
        } else if (cmd == "setoption") {
          cmd_setoption(is);
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
               c == "train" || c == "ntrain" || c == "book" || c == "backend" || c == "searchstats" || c == "tune";
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
        } else if (cmd == "ntrain") {
          cmd_ntrain(is);
        } else if (cmd == "book") {
          cmd_book(is);
        } else if (cmd == "backend") {
          std::cout << "movegen backend: " << movegen_backend() << '\n';
        } else if (cmd == "tune") {
          cmd_tune(is);
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
            // A `.nnue` file loads the NNUE-lite net (nnue.hpp) instead of linear
            // weights; exactly one of the two is ever active.
            const std::string &ev     = options_.eval_file;
            const bool         is_net = ev.size() > 5 && ev.compare(ev.size() - 5, 5, ".nnue") == 0;
            nnue_set_active(false);
            if (ev.empty()) {
              pattern_weights().unload();
            } else if (is_net) {
              pattern_weights().unload();
              if (nnue_net().load(ev, std::cout))
                nnue_set_active(true);
            } else if (!pattern_weights().load(ev, std::cout)) {
              pattern_weights().unload();
            }
          }
          if (lname == "bookfile") {
            if (options_.book_file.empty())
              book_.clear();
            else if (!book_.load(options_.book_file, std::cout))
              book_.clear(); // a failed load leaves no stale book active
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
        bool         infinite = false;
        // NOTE the loop has NO increment. It once had `tok.clear()` there, which runs
        // AFTER the body reads the next keyword and so wiped it -- the loop processed
        // exactly one token, silently, since the day it was written. Single-limit
        // commands hid it; the four-token clock form exposed it.
        for (; !tok.empty();) {
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
            infinite = true; // no limit at all: run until `stop`, or until the position is solved
          } else if (tok == "wtime" || tok == "btime" || tok == "winc" || tok == "binc") {
            // Standard UCI clock. Board is mover-relative, so pick out the MOVER's pair;
            // the opponent's clock is irrelevant to allocating this move.
            double v;
            if (!(is >> v) || v < 0.0) {
              std::cout << "info error: " << tok << " must be non-negative\n";
              return;
            }
            const bool mine = (stm_ == Color::White) == (tok[0] == 'w');
            if (mine && (tok == "wtime" || tok == "btime"))
              lim.time_ms = v;
            else if (mine)
              lim.inc_ms = v;
          } else if (tok == "movestogo") {
            long v; // parsed so a GUI sending it is not rejected; Othello's move count
            is >> v; // is already known exactly from the empties, so it adds nothing
          }
          if (!(is >> tok))
            break;
        }
        // Opening book: if it recommends a move here, play it at once and skip the
        // search entirely. `go infinite` is analysis and keeps searching.
        if (options_.own_book && book_.loaded() && !infinite) {
          const Square bm = book_.probe(board_, stm_);
          if (bm != NOMOVE) {
            say("info string book move\n");
            say("bestmove " + square_to_string(bm) + "\n");
            return;
          }
        }

        // A bare `go` still has to return a move; `go infinite` deliberately must not
        // get that default, which is the only thing distinguishing the two.
        if (!infinite && lim.depth == 0 && lim.nodes == 0 && lim.movetime_ms == 0.0 && lim.time_ms == 0.0)
          lim.depth = 8;

        run_search(lim);
      }

      /**
       * Hands the search to its own thread and returns at once, so the command loop can
       * still read `stop`. The board, side to move and rule are COPIED into the thread:
       * the caller has already joined any previous search, so nothing else can be
       * touching them, and a copy removes the question entirely.
       *
       * `bestmove` is printed by the search thread when it finishes, which is what keeps
       * the "exactly one bestmove per go, after its info lines" ordering true whether the
       * search ended on its own or was stopped.
       */
      void run_search(const SearchLimits &lim) {
        const Board root = board_;
        const Color stm  = stm_;
        const Rule  rule = options_.rule;
        search_infinite_ = (lim.depth == 0 && lim.nodes == 0 && lim.movetime_ms == 0.0 && lim.time_ms == 0.0);
        size_helpers();
        searcher_.set_abdada(kUseAbdada && options_.threads > 1);
        for (auto &h: helpers_)
          h->set_abdada(kUseAbdada && options_.threads > 1);
        searcher_.arm(); // clear any earlier stop HERE, not on the search thread
        for (auto &h: helpers_)
          h->arm();
        search_thread_ = std::thread([this, lim, root, stm, rule] {
          // Helpers run UNLIMITED and are cut off when the main search is done: their
          // job is to keep filling the table for as long as the main thread is thinking,
          // not to finish anything of their own.
          std::vector<std::thread> hthreads;
          hthreads.reserve(helpers_.size());
          for (auto &h: helpers_)
            hthreads.emplace_back([&h, root, rule, stm] {
              std::ostream sink(nullptr); // helper output is noise; only the main thread reports
              h->search(root, SearchLimits{}, rule, stm, sink);
            });

          LineSyncBuf  sb;
          std::ostream out(&sb);
          const SearchResult r = searcher_.search(root, lim, rule, stm, out);

          for (auto &h: helpers_)
            h->request_stop();
          for (auto &t: hthreads)
            t.join();
          if (r.best == NOMOVE) {
            // Reversi with no move, or both sides stuck: there is nothing to play.
            out << "info string game over (final score " << r.score / 100 << ")\n"
                << "bestmove --\n";
            return;
          }
          out << "info string " << (r.exact ? "exact" : "heuristic") << " score, depth " << r.depth << '\n'
              << "bestmove " << square_to_string(r.best) << '\n';
          if (const SearchStats *st = searcher_.stats()) // only non-null in a kStats build
            st->dump(out, false);
        });
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
        if (first == "nmp") { // null-move pruning on vs off, equal time
          int ms = 50;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          if (!(is >> ms) || ms < 1) ms = 50;
          cfg.depth = 0; cfg.movetime_ms = ms;
          cfg.nmp_a = true; cfg.nmp_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
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
        if (first == "wldtc") { // W/L/D solve vs exact solve, real clock, engine TM both sides
          double base = 3000.0, inc = 50.0;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          is >> base; is >> inc;
          cfg.depth = 0; cfg.movetime_ms = 0.0;
          cfg.tc_base_ms = base; cfg.tc_inc_ms = inc;
          cfg.etm_a = cfg.etm_b = true;
          cfg.wld_a = true; cfg.wld_b = false;
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "tma") { // ADAPTIVE soft budget vs the fixed engine allocation, real clock
          double base = 3000.0, inc = 50.0;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          is >> base; is >> inc;
          cfg.depth = 0; cfg.movetime_ms = 0.0;
          cfg.tc_base_ms = base; cfg.tc_inc_ms = inc;
          cfg.etm_a = cfg.etm_b = true; // both allocate for themselves...
          cfg.tma_a = true;             // ...only A re-prices per iteration
          std::string ev;
          if (is >> ev && ev != "-") cfg.eval_a = cfg.eval_b = ev;
          std::uint64_t sd = 0;
          if (is >> sd) cfg.seed = sd;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "tm") { // ENGINE time management vs the harness's even split, real clock
          double base = 3000.0, inc = 50.0;
          if (!(is >> cfg.pairs) || cfg.pairs < 1) cfg.pairs = 50;
          is >> base; is >> inc;
          cfg.depth = 0; cfg.movetime_ms = 0.0;
          cfg.tc_base_ms = base; cfg.tc_inc_ms = inc;
          cfg.etm_a = true; cfg.etm_b = false;
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
        int mob = 1, c2x5 = 1, stab = 1, par = 1, front = 1; // gate each feature so its value can be A/B'd cleanly
        if (is >> mob)
          cfg.use_mobility = (mob != 0);
        if (is >> c2x5)
          cfg.use_c2x5 = (c2x5 != 0);
        if (is >> stab)
          cfg.use_stab = (stab != 0);
        if (is >> par)
          cfg.use_par = (par != 0);
        if (is >> front)
          cfg.use_front = (front != 0);
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

      /** ntrain [games] [epochs] [depth] [lr_emb] [lr_out] [out] [seed] -- NNUE-lite
       *  training (train.hpp). A loaded .pat starts a new net; a loaded .nnue
       *  bootstraps another round from that teacher and warm start. */
      void cmd_ntrain(std::istringstream &is) {
        NTrainConfig cfg;
        cfg.rule = options_.rule;
        if (!(is >> cfg.games) || cfg.games < 1)
          cfg.games = 50000;
        if (!(is >> cfg.epochs) || cfg.epochs < 1)
          cfg.epochs = 10;
        if (!(is >> cfg.depth) || cfg.depth < 1)
          cfg.depth = 4;
        if (!(is >> cfg.lr_emb) || cfg.lr_emb <= 0.0)
          cfg.lr_emb = 1e-3;
        if (!(is >> cfg.lr_out) || cfg.lr_out <= 0.0)
          cfg.lr_out = 1e-5;
        std::string out;
        if (is >> out && !out.empty())
          cfg.out = out;
        std::uint64_t sd = 0;
        if (is >> sd)
          cfg.seed = sd;
        run_ntrain(cfg, std::cout);
      }

      // book gen [plies] [depth] [out]  -- build an opening book (uses the loaded EvalFile)
      // book probe                      -- show the current position's book move, if any
      void cmd_book(std::istringstream &is) {
        std::string sub;
        if (!(is >> sub) || sub == "probe") {
          if (!book_.loaded()) {
            std::cout << "info string book: none loaded (setoption name BookFile value <file>)\n";
            return;
          }
          const Square bm = book_.probe(board_, stm_);
          std::cout << "info string book: " << book_.size() << " positions, this position -> "
                    << (bm == NOMOVE ? "(not in book)" : square_to_string(bm)) << '\n';
          return;
        }
        if (sub != "gen") {
          std::cout << "info error: expected 'book gen [plies] [depth] [out]' or 'book probe'\n";
          return;
        }
        BookBuildConfig cfg;
        cfg.rule = options_.rule;
        if (!(is >> cfg.plies) || cfg.plies < 1)
          cfg.plies = 8;
        if (!(is >> cfg.depth) || cfg.depth < 1)
          cfg.depth = 16;
        std::string out;
        if (is >> out && !out.empty())
          cfg.out = out;
        build_book(cfg, std::cout);
      }

      /**
       * tune spsa [iters] [movetime_ms] [eval] [seed] -- SPSA over SearchParams.
       *
       * Classic sign-SPSA, one colour-paired game pair per iteration: perturb every
       * parameter by +-c_k simultaneously, play theta+ against theta- from a random
       * opening (both colours), and step each parameter towards the side that won.
       * One pair is a hopelessly noisy estimate on its own -- the method works because
       * thousands of noisy steps average into the gradient, which is exactly the
       * regime this engine can afford: in-process games at a small movetime are cheap.
       *
       * The RESULT IS A CANDIDATE, NOT A VERDICT. SPSA's own trajectory is not
       * evidence; whatever it converges to must beat the shipped defaults in an
       * independent match before anything changes.
       */
      void cmd_tune(std::istringstream &is) {
        std::string sub;
        if (!(is >> sub) || sub != "spsa") {
          std::cout << "info error: expected 'tune spsa [iters] [movetime_ms] [eval] [seed]'\n";
          return;
        }
        int    iters = 1000, mt = 20;
        if (is >> iters && iters < 1) iters = 1000;
        if (is >> mt && mt < 1) mt = 20;
        std::string   ev;
        PatternWeights w;
        if (is >> ev && ev != "-" && !w.load(ev, std::cout))
          return;
        std::uint64_t seed = 0x9E3779B97F4A7C15ULL, sd = 0;
        if (is >> sd && sd) seed = sd;
        double cscale = 1.0, ascale = 1.0; // optional aggressiveness multipliers
        is >> cscale; is >> ascale;
        if (cscale <= 0) cscale = 1.0;
        if (ascale <= 0) ascale = 1.0;

        struct P { const char *name; double th, c0, lo, hi; };
        SearchParams d0;
        P th[] = {
          {"fut1", (double)d0.fut1, 40, 50, 2000},   {"fut2", (double)d0.fut2, 70, 50, 2000},
          {"fut3", (double)d0.fut3, 100, 50, 2500},  {"killer0", (double)d0.killer0, 9000, 1000, 1048576},
          {"killer1", (double)d0.killer1, 4500, 500, 1048576}, {"mob_w", (double)d0.mob_w, 6, 1, 512},
          {"hist_div", (double)d0.hist_div, 12, 1, 4096},      {"parity", (double)d0.parity_bonus, 1200, 0, 1048576},
          {"sqv", (double)d0.sqv_mult, 2, 0, 128},
        };
        constexpr int kN = 9;
        auto to_params = [&](const double *v) {
          SearchParams p;
          p.fut1 = (int)v[0]; p.fut2 = (int)v[1]; p.fut3 = (int)v[2];
          p.killer0 = (int)v[3]; p.killer1 = (int)v[4]; p.mob_w = (int)v[5];
          p.hist_div = std::max(1, (int)v[6]); p.parity_bonus = (int)v[7]; p.sqv_mult = (int)v[8];
          return p;
        };
        auto rng = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };

        Searcher sa(32), sb(32);
        pattern_set_active(ev.empty() || ev == "-" ? nullptr : &w);
        pattern_set_stage_interp(options_.stage_interp);

        // One game: `first` plays Black. Returns Black's points {0, .5, 1}.
        auto play = [&](Board b, Color stm, Searcher &black, Searcher &white) -> double {
          std::ostringstream sink;
          black.clear(); white.clear();
          for (int ply = 0; ply < 80; ++ply) {
            if (b.moves() == 0) {
              const Board p = b.passed();
              if (options_.rule != Rule::Othello || !p.has_moves()) break;
              b = p; stm = ~stm; continue;
            }
            Searcher &sr = (stm == Color::Black) ? black : white;
            const SearchResult r = sr.search(b, SearchLimits{0, 0, (double)mt}, options_.rule, stm, sink);
            if (r.best == NOMOVE) break;
            if (r.best == PASS) { b = b.passed(); stm = ~stm; continue; }
            b = b.play(r.best); stm = ~stm;
          }
          const int diff = popcount(b.player) - popcount(b.opponent); // mover-relative
          const int bd   = (stm == Color::Black) ? diff : -diff;
          return bd > 0 ? 1.0 : (bd < 0 ? 0.0 : 0.5);
        };

        std::cout << "info string spsa: " << iters << " iters, movetime " << mt << "ms, "
                  << (ev.empty() ? "handcrafted" : ev) << "\n";
        for (int k = 1; k <= iters; ++k) {
          const double ckd = std::pow((double)k, 0.101);
          const double akd = std::pow((double)k + 50.0, 0.602);
          double vp[kN], vm[kN]; int delta[kN];
          for (int i = 0; i < kN; ++i) {
            delta[i]        = (rng() & 1) ? 1 : -1;
            const double ck = cscale * th[i].c0 / ckd;
            vp[i] = std::clamp(th[i].th + ck * delta[i], th[i].lo, th[i].hi);
            vm[i] = std::clamp(th[i].th - ck * delta[i], th[i].lo, th[i].hi);
          }
          sa.set_params(to_params(vp));
          sb.set_params(to_params(vm));
          // random legal opening, then BOTH colours (the same pairing the match uses)
          Board ob = Board::start(); Color os = Color::Black; bool ok = true;
          for (int i = 0; i < 6; ++i) {
            Bitboard m = ob.moves();
            if (!m) { ok = false; break; }
            unsigned pick = (unsigned)(rng() % (unsigned)popcount(m));
            while (pick-- > 0) m &= m - 1;
            ob = ob.play(lsb(m)); os = ~os;
          }
          if (!ok || !ob.has_moves()) { --k; continue; }
          const double s1 = play(ob, os, sa, sb);       // theta+ as Black
          const double s2 = 1.0 - play(ob, os, sb, sa); // theta+ as White
          const double r  = (s1 + s2) - 1.0;            // [-1, +1], + favours theta+
          for (int i = 0; i < kN; ++i) {
            const double ak = ascale * th[i].c0 * 0.25 / akd;
            th[i].th        = std::clamp(th[i].th + ak * r * delta[i], th[i].lo, th[i].hi);
          }
          if (k % 100 == 0) {
            std::cout << "info string spsa " << k << "/" << iters;
            for (int i = 0; i < kN; ++i) std::cout << ' ' << th[i].name << '=' << (int)th[i].th;
            std::cout << '\n';
            std::cout.flush();
          }
        }
        pattern_set_active(nullptr);
        std::cout << "spsa done:";
        for (int i = 0; i < kN; ++i) std::cout << ' ' << th[i].name << '=' << (int)th[i].th;
        std::cout << '\n';
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

        std::cout << "book self-test (probe symmetry mapping) ... " << std::flush;
        const bool book_ok = book_selftest();
        std::cout << (book_ok ? "ok\n" : "FAILED\n");

        constexpr std::array<std::uint64_t, 9> known{0, 4, 12, 56, 244, 1396, 8200, 55092, 390216};
        const Board                            start  = Board::start();
        bool                                   all_ok = eval_ok && search_ok && pattern_ok && book_ok;
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
