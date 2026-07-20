/**
 * @file main.cpp
 * @brief islay entry point: greet, then hand control to the UCI-style loop.
 */
#include <iostream>

#include "movegen.hpp"
#include "uci.hpp"

int main() {
  std::cout << "islay 0.1.0 - Othello/Reversi engine (movegen backend: " << islay::movegen_backend() << ")\n";
  std::cout << "type 'uci', 'go perft <depth> [nocache]', 'd', 'test', or 'quit'\n";
  std::cout.flush();
  return islay::uci_loop();
}
