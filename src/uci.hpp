/**
 * @file uci.hpp
 * @brief UCI-style text protocol driver for the islay Othello engine.
 */
#ifndef ISLAY_UCI_HPP
#define ISLAY_UCI_HPP

namespace islay {

  /**
   * @brief Run the command loop on stdin/stdout until "quit"/EOF.
   * @return process exit code.
   */
  int uci_loop();

} // namespace islay

#endif // ISLAY_UCI_HPP
