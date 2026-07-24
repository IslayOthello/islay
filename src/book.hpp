/**
 * @file book.hpp
 * @brief Opening book: precomputed best moves for early positions.
 *
 * Positions are keyed by their CANONICAL form (the lexicographically smallest of the
 * eight D4 symmetries, Board::canonical), so every rotation and mirror of a line
 * shares one entry and the book is ~8x smaller. The stored move is a square in that
 * canonical frame; probe() maps it back onto the actual board by finding the symmetry
 * that canonicalises it and returning the matching legal move.
 *
 * File format ("ISLAYBK1", little-endian, same-machine), entries sorted by key so a
 * lookup is a binary search:
 *   char   magic[8] = "ISLAYBK1"
 *   uint32 version  = 1
 *   uint32 count
 *   Entry  entries[count]   // { uint64 key; int16 score; uint8 move; uint8 depth; }
 */
#ifndef ISLAY_BOOK_HPP
#define ISLAY_BOOK_HPP

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "board.hpp"
#include "options.hpp"

namespace islay {

  class Book {
  public:
    struct Entry {
      std::uint64_t key   = 0; // hash of the canonical board
      std::int16_t  score = 0; // mover-relative centi-discs from the build search
      std::uint8_t  move  = 0; // best move as a square in the CANONICAL frame
      std::uint8_t  depth = 0; // search depth the move came from (diagnostics)
    };

    [[nodiscard]] bool loaded() const noexcept { return !entries_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    void clear() noexcept { entries_.clear(); }

    bool load(const std::string &path, std::ostream &log);
    bool save(const std::string &path, std::ostream &log) const;

    /** Best book move for `b` (mover = `stm`), or NOMOVE if the position is not in
     *  the book. The returned move is legal in `b`'s own frame. */
    [[nodiscard]] Square probe(const Board &b, Color stm) const noexcept;

    /** Replace the contents with `entries` (sorted by key for binary search). */
    void adopt(std::vector<Entry> entries);

  private:
    std::vector<Entry> entries_; // sorted by key
  };

  struct BookBuildConfig {
    int    plies = 8;    // full-width expansion depth (plies from the start position)
    int    depth = 16;   // search depth at each expanded node
    Rule   rule  = Rule::Othello;
    std::string out = "islay.book";
  };

  struct BookBuildResult {
    std::uint64_t positions = 0; // distinct canonical positions stored
    std::uint64_t searches  = 0; // leaf searches actually run
    bool          ok        = false;
  };

  /** Build a negamax-consistent book (does not write). Progress streams to `log`. */
  Book build_book_mem(const BookBuildConfig &cfg, std::ostream &log, BookBuildResult &res);

  /** Build a book and write `cfg.out`. Progress streams to `log`. */
  BookBuildResult build_book(const BookBuildConfig &cfg, std::ostream &log);

  /** Self-check: the probe's symmetry mapping is consistent (probe of a rotated board
   *  returns the rotation of the probe of the board). Returns true on success. */
  [[nodiscard]] bool book_selftest() noexcept;

} // namespace islay

#endif // ISLAY_BOOK_HPP
