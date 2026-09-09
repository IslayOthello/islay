#ifndef ISLAY_OPTIONS_HPP
#define ISLAY_OPTIONS_HPP

#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace islay {

  // Othello passes; Reversi ends on the first stuck side.
  enum class Rule { Othello, Reversi };

  [[nodiscard]] const char *rule_name(Rule r) noexcept;

  struct Options {
    Rule        rule           = Rule::Othello;
    int         perft_hash_mib = 256; // perft transposition table size, in MiB
    int         mcts_hash_mib  = 64; // single-worker tree arena, independent of PerftTT
    std::string eval_file; // empty selects the explicit uniform scaffold
  };

  struct OptionSpec {
    std::string                                         name;
    std::string                                         type; // "combo" | "spin" | "check" | "string" | "button"
    std::string                                         def; // default value, string form
    std::vector<std::string>                            vars; // choices, for "combo"
    long                                                min = 0; // for "spin"
    long                                                max = 0;
    std::function<bool(Options &, const std::string &)> apply; // parse+set; false if invalid
  };

  [[nodiscard]] const std::vector<OptionSpec> &option_specs();

  bool apply_option(Options &opt, const std::string &name, const std::string &value);

  void print_option_specs(std::ostream &os);

} // namespace islay

#endif // ISLAY_OPTIONS_HPP
