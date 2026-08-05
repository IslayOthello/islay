#include <iostream>

#include "movegen.hpp"
#include "uci.hpp"

int main() {
  std::cout << "islay 0.1.0 - Othello/Reversi engine (movegen backend: " << islay::movegen_backend() << ")\n";
  // Hide development commands until `debug on`.
  std::cout << "type 'uci', 'position', 'go depth <N>', 'go perft <N>', or 'quit'\n";
  std::cout.flush();
  return islay::uci_loop();
}
