/**
 * @file main.cpp
 * @brief islay entry point: greet, then hand control to the UCI-style loop.
 */
#include <iostream>

#include "movegen.hpp"
#include "uci.hpp"

int main() {
  std::cout << "islay 0.1.0 - Othello/Reversi engine (movegen backend: " << islay::movegen_backend() << ")\n";
  // Advertise only the release protocol. The development commands stay unlisted;
  // `debug on` is the documented door to them (see UCI.md).
  std::cout << "type 'uci', 'position', 'go depth <N>', 'debug on', or 'quit'\n";
  std::cout.flush();
  return islay::uci_loop();
}
