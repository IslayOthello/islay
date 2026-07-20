/**
 * @file board.cpp
 * @brief Board text (de)serialization and square naming.
 */
#include "board.hpp"

#include <array>
#include <cctype>

namespace islay {

  Square parse_square(const std::string &s) noexcept {
    if (s == "pass" || s == "PASS" || s == "--" || s == "@@")
      return PASS;
    if (s.size() < 2)
      return NOMOVE;
    const char fc = static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
    const char rc = s[1];
    if (fc < 'a' || fc > 'h' || rc < '1' || rc > '8')
      return NOMOVE;
    return (rc - '1') * 8 + (fc - 'a');
  }

  std::string square_to_string(Square sq) {
    if (sq == PASS)
      return "pass";
    if (sq < 0 || sq >= 64)
      return "--";
    return {static_cast<char>('a' + (sq & 7)), static_cast<char>('1' + (sq >> 3))};
  }

  bool Board::set(const std::string &diagram, char stm) noexcept {
    Bitboard black = 0;
    Bitboard white = 0;
    Square   sq    = 0;
    for (const char c: diagram) {
      if (std::isspace(static_cast<unsigned char>(c)))
        continue;
      if (sq >= 64)
        return false; // too many cells
      switch (c) {
        case 'X':
        case 'x':
        case '*':
          black |= square_bb(sq);
          break;
        case 'O':
        case 'o':
        case '0':
          white |= square_bb(sq);
          break;
        case '-':
        case '.':
        case '_':
          break;
        default:
          return false; // unknown glyph
      }
      ++sq;
    }
    if (sq != 64)
      return false; // too few cells

    switch (static_cast<char>(std::toupper(static_cast<unsigned char>(stm)))) {
      case 'X':
      case 'B':
        player   = black;
        opponent = white;
        return true;
      case 'O':
      case 'W':
        player   = white;
        opponent = black;
        return true;
      default:
        return false;
    }
  }

  std::string Board::to_string(Color stm) const {
    const Bitboard black = (stm == Color::Black) ? player : opponent;
    const Bitboard white = (stm == Color::Black) ? opponent : player;
    std::string    s(64, '-');
    for (Square sq = 0; sq < 64; ++sq) {
      if (black & square_bb(sq)) {
        s[sq] = 'X';
      } else if (white & square_bb(sq)) {
        s[sq] = 'O';
      }
    }
    return s;
  }

  void Board::print(Color stm, std::ostream &os) const {
    const Bitboard black = (stm == Color::Black) ? player : opponent;
    const Bitboard white = (stm == Color::Black) ? opponent : player;
    const Bitboard legal = moves();

    os << "  a b c d e f g h\n";
    for (int rank = 0; rank < 8; ++rank) {
      os << static_cast<char>('1' + rank) << ' ';
      for (int file = 0; file < 8; ++file) {
        const Bitboard bb = square_bb(rank * 8 + file);
        char           c  = '-';
        if (black & bb) {
          c = 'X';
        } else if (white & bb) {
          c = 'O';
        } else if (legal & bb) {
          c = '*';
        }
        os << c << ' ';
      }
      os << '\n';
    }
    os << (stm == Color::Black ? "Black" : "White") << " to move  |  X:" << popcount(black) << " O:" << popcount(white)
       << "  |  moves:" << popcount(legal) << '\n';
  }

} // namespace islay
