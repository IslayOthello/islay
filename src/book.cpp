/**
 * @file book.cpp
 * @brief Opening book: probe, file IO, and the negamax builder (book.hpp).
 */
#include "book.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <unordered_map>

#include "eval.hpp"
#include "hash.hpp"
#include "search.hpp"

namespace islay {
  namespace {
    constexpr char kMagic[8] = {'I', 'S', 'L', 'A', 'Y', 'B', 'K', '1'};

    // The symmetry s with b.symmetry(s) == b.canonical(), and -1 if none matches
    // (cannot happen: canonical() is one of the eight images). The move stored for a
    // position lives in the canonical frame, and this is how a probe gets back to the
    // caller's frame without inverse-transform arithmetic.
    [[nodiscard]] int canon_symmetry(const Board &b) noexcept {
      const Board c = b.canonical();
      for (int s = 0; s < 8; ++s)
        if (b.symmetry(s) == c)
          return s;
      return -1;
    }
  } // namespace

  void Book::adopt(std::vector<Entry> entries) {
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.key < b.key; });
    entries_ = std::move(entries);
  }

  Square Book::probe(const Board &b, Color) const noexcept {
    if (entries_.empty())
      return NOMOVE;
    const std::uint64_t key = hash_board(b.canonical().player, b.canonical().opponent);
    auto it = std::lower_bound(entries_.begin(), entries_.end(), key,
                               [](const Entry &e, std::uint64_t k) { return e.key < k; });
    if (it == entries_.end() || it->key != key)
      return NOMOVE;

    // The move is a square in the canonical frame; return the legal move of `b` that
    // maps onto it under the symmetry that canonicalises `b`. Matching against `b`'s
    // real moves also rejects a hash collision (the bit would not line up with a move).
    const int s = canon_symmetry(b);
    if (s < 0)
      return NOMOVE;
    const Bitboard want = square_bb(static_cast<Square>(it->move));
    Bitboard       ms   = b.moves();
    while (ms) {
      const Square m = pop_lsb(ms);
      if (Board::symmetry_bb(square_bb(m), s) == want)
        return m;
    }
    return NOMOVE;
  }

  bool Book::save(const std::string &path, std::ostream &log) const {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
      log << "info error: cannot write " << path << '\n';
      return false;
    }
    std::uint32_t version = 1, count = static_cast<std::uint32_t>(entries_.size());
    bool          ok = std::fwrite(kMagic, 1, 8, f) == 8 && std::fwrite(&version, 4, 1, f) == 1 &&
              std::fwrite(&count, 4, 1, f) == 1;
    for (const Entry &e : entries_) {
      ok = ok && std::fwrite(&e.key, 8, 1, f) == 1 && std::fwrite(&e.score, 2, 1, f) == 1 &&
           std::fwrite(&e.move, 1, 1, f) == 1 && std::fwrite(&e.depth, 1, 1, f) == 1;
    }
    std::fclose(f);
    if (!ok)
      log << "info error: short write to " << path << '\n';
    else
      log << "info string book: saved " << entries_.size() << " positions -> " << path << '\n';
    return ok;
  }

  bool Book::load(const std::string &path, std::ostream &log) {
    entries_.clear();
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
      log << "info error: cannot open " << path << '\n';
      return false;
    }
    char          magic[8];
    std::uint32_t version = 0, count = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, kMagic, 8) != 0 || std::fread(&version, 4, 1, f) != 1 ||
        version != 1 || std::fread(&count, 4, 1, f) != 1) {
      std::fclose(f);
      log << "info error: " << path << " is not an ISLAYBK1 book\n";
      return false;
    }
    entries_.resize(count);
    bool ok = true;
    for (std::uint32_t i = 0; i < count && ok; ++i) {
      Entry &e = entries_[i];
      ok = std::fread(&e.key, 8, 1, f) == 1 && std::fread(&e.score, 2, 1, f) == 1 &&
           std::fread(&e.move, 1, 1, f) == 1 && std::fread(&e.depth, 1, 1, f) == 1;
    }
    std::fclose(f);
    if (!ok) {
      entries_.clear();
      log << "info error: short read from " << path << '\n';
      return false;
    }
    // The file is written sorted, but a hand-made book might not be; guarantee it.
    if (!std::is_sorted(entries_.begin(), entries_.end(), [](const Entry &a, const Entry &b) { return a.key < b.key; }))
      std::sort(entries_.begin(), entries_.end(), [](const Entry &a, const Entry &b) { return a.key < b.key; });
    log << "info string book: loaded " << entries_.size() << " positions from " << path << '\n';
    return true;
  }

  namespace {
    // The builder walks the opening tree to `plies` full width, searches each frontier
    // leaf at `depth`, and backs the values up by negamax so every internal node stores
    // the move that leads to the best deeply-searched line -- not just the best shallow
    // move at that node. A memo keyed by the canonical hash searches each distinct
    // position once, which matters because opening move orders transpose heavily.
    struct Builder {
      const BookBuildConfig &cfg;
      Searcher              &searcher;
      std::ostream          &log;
      std::unordered_map<std::uint64_t, Book::Entry> memo;
      std::uint64_t          searches = 0;

      // Returns the mover-relative value of `b`. Stores an entry for every internal
      // node (a node with a chosen move); frontier leaves contribute a value but no
      // stored move, so the book only ever answers where it has a real recommendation.
      int visit(const Board &b, Color stm, int ply) {
        const Board         cb  = b.canonical();
        const std::uint64_t key = hash_board(cb.player, cb.opponent);
        if (auto it = memo.find(key); it != memo.end())
          return it->second.score;

        const Bitboard moves = b.moves();
        if (moves == 0) {
          // A pass costs no ply and does not branch the book; a double pass is terminal.
          const Board p = b.passed();
          if (cfg.rule == Rule::Othello && p.has_moves())
            return -visit(p, ~stm, ply);
          const int mine = popcount(b.player), theirs = popcount(b.opponent);
          const int empties = 64 - mine - theirs;
          const int diff = mine - theirs;
          return 100 * (diff > 0 ? diff + empties : diff < 0 ? diff - empties : 0);
        }

        if (ply >= cfg.plies) {
          // Frontier: one search, its score is the leaf value. No entry stored -- the
          // book ends here and the engine searches from this position onward.
          std::ostringstream  sink;
          const SearchLimits  lim{cfg.depth, 0, 0.0};
          const SearchResult  r = searcher.search(b, lim, cfg.rule, stm, sink);
          ++searches;
          if ((searches % 200) == 0) {
            log << "info string book: " << searches << " searches, " << memo.size() << " positions\n";
            log.flush();
          }
          return r.score;
        }

        int      best_score = -kInf;
        Square   best_move  = NOMOVE;
        Bitboard ms         = moves;
        while (ms) {
          const Square m  = pop_lsb(ms);
          const int    cs = -visit(b.play(m), ~stm, ply + 1);
          if (cs > best_score) {
            best_score = cs;
            best_move  = m;
          }
        }

        // Store the best move in the canonical frame.
        const int s = canon_symmetry(b);
        Book::Entry e;
        e.key   = key;
        e.score = static_cast<std::int16_t>(std::clamp(best_score, -32000, 32000));
        e.move  = static_cast<std::uint8_t>(lsb(Board::symmetry_bb(square_bb(best_move), s)));
        e.depth = static_cast<std::uint8_t>(cfg.depth);
        memo.emplace(key, e);
        return best_score;
      }
    };
  } // namespace

  Book build_book_mem(const BookBuildConfig &cfg, std::ostream &log, BookBuildResult &res) {
    Searcher searcher(64);
    Builder  b{cfg, searcher, log, {}, 0};
    b.visit(Board::start(), Color::Black, 0);

    std::vector<Book::Entry> entries;
    entries.reserve(b.memo.size());
    for (auto &kv : b.memo)
      entries.push_back(kv.second);

    Book book;
    book.adopt(std::move(entries));
    res.positions = b.memo.size();
    res.searches  = b.searches;
    res.ok        = true;
    return book;
  }

  BookBuildResult build_book(const BookBuildConfig &cfg, std::ostream &log) {
    log << "info string book: building to " << cfg.plies << " plies, search depth " << cfg.depth << ", rule "
        << rule_name(cfg.rule) << " -> " << cfg.out << '\n';
    log.flush();

    BookBuildResult res;
    Book            book = build_book_mem(cfg, log, res);
    if (!res.ok || !book.save(cfg.out, log)) {
      res.ok = false;
      return res;
    }
    log << "book done: " << res.positions << " positions, " << res.searches << " searches -> " << cfg.out << '\n';
    log.flush();
    return res;
  }

  bool book_selftest() noexcept {
    BookBuildConfig cfg;
    cfg.plies = 3;
    cfg.depth = 4; // small and deterministic; the mapping, not the moves, is under test
    BookBuildResult res;
    std::ostringstream sink;
    const Book book = build_book_mem(cfg, sink, res);
    if (!res.ok || book.size() == 0)
      return false;

    // For every opening line and every D4 image of its position, the book must either
    // recommend nothing on both, or recommend moves that lead to the SAME canonical
    // successor. (A stronger "move is the exact symmetry-image" check is wrong at
    // positions with a non-trivial stabiliser, where several equivalent moves exist and
    // the probe may return a different representative; the successor is the invariant.)
    const char *lines[] = {"", "f5", "f5 f6", "d3 c3", "f5 d6 c3"};
    for (const char *ln : lines) {
      Board b   = Board::start();
      Color stm = Color::Black;
      std::istringstream ms(ln);
      std::string        mv;
      bool               ok = true;
      while (ms >> mv) {
        const Square sq = parse_square(mv);
        if (sq == NOMOVE || sq == PASS || !(b.moves() & square_bb(sq))) { ok = false; break; }
        b   = b.play(sq);
        stm = ~stm;
      }
      if (!ok)
        continue;
      const Square base = book.probe(b, stm);
      const bool   has  = base != NOMOVE;
      const Board  succ = has ? b.play(base).canonical() : Board{};
      for (int s = 0; s < 8; ++s) {
        const Board  bs  = b.symmetry(s);
        const Square got = book.probe(bs, stm);
        if ((got != NOMOVE) != has)
          return false;
        if (has && bs.play(got).canonical() != succ)
          return false;
      }
    }
    return true;
  }

} // namespace islay
