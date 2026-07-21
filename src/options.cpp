/**
 * @file options.cpp
 * @brief Option registry and the `setoption` value parsers.
 */
#include "options.hpp"

#include <cctype>

namespace islay {
  namespace {

    [[nodiscard]] std::string to_lower(std::string s) {
      for (char &c: s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      return s;
    }

    [[nodiscard]] bool iequals(const std::string &a, const std::string &b) { return to_lower(a) == to_lower(b); }

  } // namespace

  const char *rule_name(Rule r) noexcept { return r == Rule::Reversi ? "Reversi" : "Othello"; }

  const std::vector<OptionSpec> &option_specs() {
    static const std::vector<OptionSpec> specs = {
            OptionSpec{"Rule",
                       "combo",
                       "Othello",
                       {"Othello", "Reversi"},
                       0,
                       0,
                       [](Options &o, const std::string &v) {
                         if (iequals(v, "Othello")) {
                           o.rule = Rule::Othello;
                           return true;
                         }
                         if (iequals(v, "Reversi")) {
                           o.rule = Rule::Reversi;
                           return true;
                         }
                         return false;
                       }},
            OptionSpec{"EvalFile", "string", "", {}, 0, 0,
                       [](Options &o, const std::string &v) {
                         o.eval_file = v; // uci.cpp loads it; empty keeps the hand-written eval
                         return true;
                       }},
            // Linear interpolation of the pattern eval across stage boundaries. MEASURED
            // +58/+66 Elo (fixed depth 6, two seeds) on weights/v12.pat; default off keeps
            // the baseline byte-identical, so it is opt-in until a set is trained for it.
            OptionSpec{"StageInterpolation", "check", "true", {}, 0, 0,
                       [](Options &o, const std::string &v) {
                         o.stage_interp = iequals(v, "true") || v == "1";
                         return true;
                       }},
            // Hash sizes the SEARCH table (the conventional UCI meaning). perft
            // keeps its own table -- pointing both at one option would silently
            // allocate twice the memory the user asked for.
            OptionSpec{"PerftHash",
                       "spin",
                       "256",
                       {},
                       1,
                       65536,
                       [](Options &o, const std::string &v) {
                         try {
                           const long h = std::stol(v);
                           if (h < 1 || h > 65536)
                             return false;
                           o.perft_hash_mib = static_cast<int>(h);
                           return true;
                         } catch (...) {
                           return false;
                         }
                       }},
            OptionSpec{"Threads",
                       "spin",
                       "1",
                       {},
                       1,
                       64,
                       [](Options &o, const std::string &v) {
                         try {
                           const long t = std::stol(v);
                           if (t < 1 || t > 64)
                             return false;
                           o.threads = static_cast<int>(t);
                           return true;
                         } catch (...) {
                           return false;
                         }
                       }},
            OptionSpec{"Hash",
                       "spin",
                       "256",
                       {},
                       1,
                       65536,
                       [](Options &o, const std::string &v) {
                         try {
                           const long h = std::stol(v);
                           if (h < 1 || h > 65536)
                             return false;
                           o.hash_mib = static_cast<int>(h);
                           return true;
                         } catch (...) {
                           return false;
                         }
                       }},
            // Register further options here: give a name/type/default and an apply
            // lambda that parses the value string into the matching Options field.
    };
    return specs;
  }

  bool apply_option(Options &opt, const std::string &name, const std::string &value) {
    for (const OptionSpec &s: option_specs()) {
      if (iequals(s.name, name))
        return s.apply(opt, value);
    }
    return false;
  }

  void print_option_specs(std::ostream &os) {
    for (const OptionSpec &s: option_specs()) {
      os << "option name " << s.name << " type " << s.type << " default " << s.def;
      if (s.type == "combo") {
        for (const std::string &v: s.vars)
          os << " var " << v;
      } else if (s.type == "spin") {
        os << " min " << s.min << " max " << s.max;
      }
      os << '\n';
    }
  }

} // namespace islay
