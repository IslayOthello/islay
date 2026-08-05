// ISLAYBK1 stores sorted canonical-board entries in little-endian form.
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
      std::uint64_t key   = 0; // canonical board hash
      std::int16_t  score = 0; // mover-relative centi-discs
      std::uint8_t  move  = 0; // canonical-frame square
      std::uint8_t  depth = 0;
    };

    [[nodiscard]] bool loaded() const noexcept { return !entries_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    void clear() noexcept { entries_.clear(); }

    bool load(const std::string &path, std::ostream &log);
    bool save(const std::string &path, std::ostream &log) const;

    [[nodiscard]] Square probe(const Board &b, Color stm) const noexcept;

    void adopt(std::vector<Entry> entries);

  private:
    std::vector<Entry> entries_; // sorted by key
  };

  struct BookBuildConfig {
    int    plies = 8;
    int    depth = 16;
    Rule   rule  = Rule::Othello;
    std::string out = "islay.book";
  };

  struct BookBuildResult {
    std::uint64_t positions = 0;
    std::uint64_t searches  = 0;
    bool          ok        = false;
  };

  Book build_book_mem(const BookBuildConfig &cfg, std::ostream &log, BookBuildResult &res);

  BookBuildResult build_book(const BookBuildConfig &cfg, std::ostream &log);

  [[nodiscard]] bool book_selftest() noexcept;

} // namespace islay

#endif // ISLAY_BOOK_HPP
