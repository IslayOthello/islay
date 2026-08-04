/**
 * @file options.hpp
 * @brief Engine options and a small, extensible UCI `setoption` registry.
 *
 * Adding a new option is two edits: add a field to `Options`, then register an
 * `OptionSpec` in options.cpp describing its UCI type and how to parse a value.
 * The `uci` command advertises every registered option automatically.
 */
#ifndef ISLAY_OPTIONS_HPP
#define ISLAY_OPTIONS_HPP

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace islay {

  /**
   * Game-over convention.
   *   Othello : a stuck side passes; the game ends only when neither side can move.
   *   Reversi : a side with no legal move ends the game immediately (no passing).
   */
  enum class Rule { Othello, Reversi };

  [[nodiscard]] const char *rule_name(Rule r) noexcept;

  /** All tunable engine options. Add fields here and register them in options.cpp. */
  struct Options {
    Rule        rule      = Rule::Othello;
    std::string eval_file;            // ISLAYPAT pattern weights; empty = hand-written eval
    int  threads        = 1;   // lazy-SMP search threads (1 = the old single-threaded search)
    int  hash_mib       = 256; // SEARCH transposition table size, in MiB
    int  perft_hash_mib = 256; // perft transposition table size, in MiB (separate table)
    bool stage_interp   = true;  // linear stage interpolation of the pattern eval (+~58 Elo, see uci.cpp)
    int  correction_history = 200; // ProbCut-gate residual cap in centi-discs; zero disables it
    bool own_book       = false; // play from the opening book when it has the position
    std::string book_file;       // ISLAYBK1 opening book; empty = none
  };

  /** Describes one UCI option: how to advertise it and how to apply a value. */
  struct OptionSpec {
    std::string                                         name;
    std::string                                         type; // "combo" | "spin" | "check" | "string" | "button"
    std::string                                         def; // default value, string form
    std::vector<std::string>                            vars; // choices, for "combo"
    long                                                min = 0; // for "spin"
    long                                                max = 0;
    std::function<bool(Options &, const std::string &)> apply; // parse+set; false if invalid
  };

  /** The registry of every option (built once, on first use). */
  [[nodiscard]] const std::vector<OptionSpec> &option_specs();

  /** Apply `value` to option `name` (both matched case-insensitively). */
  bool apply_option(Options &opt, const std::string &name, const std::string &value);

  /** Emit an `option name ... type ...` line per option for the `uci` command. */
  void print_option_specs(std::ostream &os);

} // namespace islay

#endif // ISLAY_OPTIONS_HPP
