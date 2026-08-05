#ifndef ISLAY_TRAIN_HPP
#define ISLAY_TRAIN_HPP

#include <cstdint>
#include <ostream>
#include <string>

#include "options.hpp"

namespace islay {

  struct TrainConfig {
    int           games         = 50000;
    int           epochs        = 8;
    int           depth         = 4;
    int           opening_plies = 10;
    int           solve_empties = 12;
    double        lr            = 0.0005;
    double        l2            = 1e-6;
    double        val_frac      = 0.1;
    bool          use_mobility  = true;
    bool          use_c2x5      = true;
    bool          use_stab      = true;
    bool          use_par       = true;
    bool          use_front     = true;
    bool          interp        = false;
    std::uint64_t seed          = 0;
    Rule          rule          = Rule::Othello;
    std::string   out           = "islay.pat";
  };

  struct TrainResult {
    std::uint64_t games     = 0;
    std::uint64_t positions = 0;
    double        rmse      = 0.0;
    double        val_rmse  = 0.0;
    bool          ok        = false;
  };

  TrainResult run_train(const TrainConfig &cfg, std::ostream &log);

  struct NTrainConfig {
    int           games         = 50000;
    int           epochs        = 10;
    int           depth         = 10;
    int           workers       = 4;
    int           opening_plies = 10;
    int           solve_empties = 12;
    double        lr_emb        = 1e-3;
    double        lr_out        = 1e-5;
    double        val_frac      = 0.1;
    bool          grouped       = true;
    std::uint64_t seed          = 0;
    Rule          rule          = Rule::Othello;
    std::string   out           = "islay.nnue";
  };

  TrainResult run_ntrain(const NTrainConfig &cfg, std::ostream &log);

} // namespace islay

#endif // ISLAY_TRAIN_HPP
