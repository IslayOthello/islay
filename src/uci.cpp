// Public and debug-only commands are documented in UCI.md.
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

    // ABDADA deferral lost 6.5 Elo against plain four-thread lazy SMP.
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

    // Serialize complete protocol lines across command and search threads.
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

      // joinable() is the search-running state.
      std::thread search_thread_;
      bool        search_infinite_ = false; // only ever touched by the command thread

      // Lazy-SMP helpers keep private search state and share only the TT.
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

      // EOF/quit lets bounded batch searches finish; stop abandons them.
      void stop_search_and_join(bool abandon = true) {
        if (!search_thread_.joinable())
          return;
        if (abandon || search_infinite_)
          searcher_.request_stop();
        search_thread_.join(); // the join is also what publishes the search's writes
      }

      // Commands that own engine state must join the search first.
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
          for (auto &h: helpers_)
            h->clear();
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

      void unknown(const std::string &cmd) {
        std::cout << "info error: unknown command '" << cmd << "'\n";
      }

      [[nodiscard]] static bool is_debug_command(const std::string &c) noexcept {
        return c == "d" || c == "display" || c == "board" || c == "bench" || c == "test" ||
               c == "selftest" || c == "match" || c == "features" || c == "pcdata" || c == "orderdata" ||
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
        } else if (cmd == "orderdata") {
          cmd_orderdata(is);
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
            // Failed loads leave no stale evaluator active.
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

        SearchLimits lim;
        bool         infinite = false;
        // The body reads the next token; keep the loop increment empty.
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
        if (options_.own_book && book_.loaded() && !infinite) {
          const Square bm = book_.probe(board_, stm_);
          if (bm != NOMOVE) {
            say("info string book move\n");
            say("bestmove " + square_to_string(bm) + "\n");
            return;
          }
        }

        if (!infinite && lim.depth == 0 && lim.nodes == 0 && lim.movetime_ms == 0.0 && lim.time_ms == 0.0)
          lim.depth = 8;

        run_search(lim);
      }

      // Copy root state into the asynchronous search.
      void run_search(const SearchLimits &lim) {
        const Board root = board_;
        const Color stm  = stm_;
        const Rule  rule = options_.rule;
        search_infinite_ = (lim.depth == 0 && lim.nodes == 0 && lim.movetime_ms == 0.0 && lim.time_ms == 0.0);
        size_helpers();
        searcher_.set_abdada(kUseAbdada && options_.threads > 1);
        searcher_.set_correction_history_cap(options_.correction_history);
        for (auto &h: helpers_)
          h->set_abdada(kUseAbdada && options_.threads > 1);
        for (auto &h: helpers_)
          h->set_correction_history_cap(options_.correction_history);
        searcher_.arm(); // clear any earlier stop HERE, not on the search thread
        for (auto &h: helpers_)
          h->arm();
        search_thread_ = std::thread([this, lim, root, stm, rule] {
          // Helpers run until the main search stops them.
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
          std::string ev;
          if (is >> ev && ev != "-")
            cfg.eval_a = cfg.eval_b = ev;
          run_match(cfg, std::cout);
          return;
        }
        if (first == "et") {
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

      void cmd_orderdata(std::istringstream &is) {
        int           n = 10000, depth = 10;
        std::uint64_t seed = 0xD1B54A32D192ED03ULL;
        if (!(is >> n) || n < 1)
          n = 10000;
        if (!(is >> depth) || depth < 2)
          depth = 10;
        if (!(is >> seed))
          seed = 0xD1B54A32D192ED03ULL;

        const auto next = [&seed]() noexcept {
          seed ^= seed << 13;
          seed ^= seed >> 7;
          seed ^= seed << 17;
          return seed;
        };

        std::cout << "sample,stage,prev,prev2,square,flips,replies,label\n";
        Searcher teacher(1);
        int      emitted = 0;
        for (int attempt = 0; emitted < n && attempt < n * 4; ++attempt) {
          Board  b     = Board::start();
          Color  stm   = Color::Black;
          Square prev  = NOMOVE;
          Square prev2 = NOMOVE;
          const int plies = 4 + static_cast<int>(next() % 46);
          bool dead = false;
          for (int k = 0; k < plies; ++k) {
            Bitboard moves = b.moves();
            if (moves == 0) {
              const Board passed = b.passed();
              if (!passed.has_moves()) {
                dead = true;
                break;
              }
              b   = passed;
              stm = ~stm;
              continue; // a pass preserves both placement contexts
            }
            unsigned pick = static_cast<unsigned>(next() % static_cast<unsigned>(popcount(moves)));
            while (pick-- > 0)
              moves &= moves - 1;
            const Square sq = lsb(moves);
            b     = b.play(sq);
            stm   = ~stm;
            prev2 = prev;
            prev  = sq;
          }

          const Bitboard moves = dead ? 0 : b.moves();
          if (popcount(moves) < 2)
            continue;

          teacher.clear();
          std::ostringstream sink;
          const SearchResult result = teacher.search(b, SearchLimits{depth, 0, 0.0}, options_.rule, stm, sink);
          if (result.best < 0 || result.best >= 64 || !(moves & square_bb(result.best)))
            continue;

          Bitboard rest = moves;
          while (rest) {
            const Square sq      = pop_lsb(rest);
            const Board  child   = b.play(sq);
            const int    flips   = popcount(b.player ^ child.opponent ^ square_bb(sq));
            const int    replies = popcount(child.moves());
            std::cout << emitted << ',' << pattern_stage(b.count()) << ','
                      << (prev < 64 ? static_cast<int>(prev) : 64) << ','
                      << (prev2 < 64 ? static_cast<int>(prev2) : 64) << ','
                      << static_cast<int>(sq) << ',' << flips << ',' << replies << ','
                      << (sq == result.best ? 1 : 0) << '\n';
          }
          ++emitted;
        }
        std::cout << "info string orderdata: " << emitted << " positions at depth " << depth << "\n";
      }

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
        run_train(cfg, std::cout);
      }

      void cmd_ntrain(std::istringstream &is) {
        NTrainConfig cfg;
        cfg.rule = options_.rule;
        if (!(is >> cfg.games) || cfg.games < 1)
          cfg.games = 50000;
        if (!(is >> cfg.epochs) || cfg.epochs < 1)
          cfg.epochs = 10;
        if (!(is >> cfg.depth) || cfg.depth < 1)
          cfg.depth = 10;
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
        if (!(is >> cfg.workers) || cfg.workers < 1)
          cfg.workers = 4;
        int grouped = 1;
        if (is >> grouped)
          cfg.grouped = grouped != 0;
        run_ntrain(cfg, std::cout);
      }

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

      // Sign-SPSA uses one colour-reversed pair per iteration.
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

        std::cout << "nnue self-test (legacy/grouped equivalence) ... " << std::flush;
        const bool nnue_ok = NnueNet::selftest();
        std::cout << (nnue_ok ? "ok\n" : "FAILED\n");

        std::cout << "book self-test (probe symmetry mapping) ... " << std::flush;
        const bool book_ok = book_selftest();
        std::cout << (book_ok ? "ok\n" : "FAILED\n");

        constexpr std::array<std::uint64_t, 9> known{0, 4, 12, 56, 244, 1396, 8200, 55092, 390216};
        const Board                            start  = Board::start();
        bool                                   all_ok = eval_ok && search_ok && pattern_ok && nnue_ok && book_ok;
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
