/**
 * @file train.hpp
 * @brief Fit pattern weights from self-play. The eval is the bottleneck; this is
 *        the thing that moves it.
 *
 * WHY THIS EXISTS. Five selective-search techniques were built and measured on this
 * engine, and three of them (late move pruning, aspiration windows, ProbCut) lost
 * for the same underlying reason: each bets on a quantity the EVAL supplies -- move
 * ordering, score stability across iterations, shallow-predicts-deep -- and the
 * hand-written eval is too noisy to back any of those bets (measured: sigma of the
 * d-2 -> d prediction is ~250 cd against a score s.d. of ~800-1000 cd). Search is
 * already tight: the effective branching factor is 2.31, below the sqrt(b) ~= 3.16
 * that perfect ordering buys. There is no more search to win. There is eval.
 *
 * THE SHAPE, and why each choice is what it is. Unless stated otherwise, this
 * section describes the linear `train`; the `ntrain` contract is documented below.
 *
 *  * **For linear `train`, the label is the game's final disc difference.** The
 *    engine's search is only as good as the eval it is fitted from, so training on
 *    its scores would fit this eval's own noise and call it progress. The final
 *    result is what the eval is supposed to predict, and it is ground truth.
 *    `ntrain` is deliberately different: a warm-started NNUE distils depth-10
 *    search scores, learning the teacher's lookahead corrections in its static eval.
 *  * **The endgame is played EXACTLY** (`solve_empties`). A pass costs no depth, so
 *    `depth >= empties` makes every leaf terminal and the score the true game value
 *    -- this engine has an exact solver for free (search.hpp). Without it the label
 *    is "what happened when both sides butchered the last 20 plies at depth 2",
 *    which is not the truth the eval is being asked to predict. Costs one solve per
 *    move near the end, and they shrink geometrically.
 *  * **Games are stored, features are not.** A run visits millions of positions;
 *    at 38 uint32 indices each that is hundreds of megabytes of intermediate. A
 *    game is a move list -- 68 bytes -- and replaying it regenerates the features
 *    through PatternState::update, which is incremental and already written. Storing
 *    games is what makes multiple epochs cheap instead of impossible.
 *  * **A validation split, held out BY GAME.** Every position in a game carries that
 *    game's single label, so splitting by POSITION would leak: a held-out position
 *    whose siblings are in the training set has already had its label fitted. The
 *    split is fixed once and only the training part is reshuffled per epoch.
 *    This exists because train rmse cannot tell underfit from overfit, and that gap
 *    is exactly how the lr default below stayed wrong for so long.
 *  * **lr = 0.0005, and the training loss is ANTI-CORRELATED with strength.** This
 *    is the trap in this file. Fitting the training set BETTER makes the engine
 *    WORSE, monotonically, across the whole sweep (40K games, everything else equal,
 *    300-game matches):
 *        lr 0.0005 -> train rmse 1993 (worst fit)  -- the strongest
 *        lr 0.002  -> train rmse 1903              -- -117 Elo (z = -6.3)
 *        lr 0.008  -> train rmse 1761 (best fit)   -- -279 Elo (z = -14.3)
 *    That is overfitting, and it is severe: 108217 weights per stage against ~3.8
 *    samples per weight. Below 0.0005 it flattens (0.00025: -19 Elo, 0.0001: -8 Elo,
 *    neither significant), so 0.0005 is the floor of the plateau, not a knife edge.
 *    **Never tune this, or l2, or epochs, by watching rmse** -- rmse will point the
 *    wrong way. Use `match 150 4 <new> <old>`.
 *  * **Plain SGD, no framework.** pattern.hpp made the score LINEAR in the weights
 *    on purpose: the gradient of the score with respect to w[j] is just "how many
 *    times index j occurred", so a step is `w[idx] -= lr * error` over 39 indices.
 *    No autodiff, no dependency, and it runs at memory speed.
 *  * **l2 is a DEAD lever, left at 1e-6.** The penalty term lr*l2*w is negligible
 *    against the error term for any l2 up to 1e-2 -- swept 0, 1e-6, 1e-4, 1e-3, 1e-2
 *    and val_rmse did not move one digit (2120 cd throughout). The scale is simply
 *    wrong (w ~ hundreds, err ~ 2000), and fixing it is not worth it: at 100K games
 *    overfitting is already mild. The real regularisers here are a small lr and the
 *    early-stop snapshot, not this.
 *  * **Colour-absolute, Black's point of view**, exactly like the weights (see
 *    pattern.hpp). The label is converted to Black's view once, at game end.
 *
 * WHAT IT DOES NOT DO. The first run learns from games played by the hand-written
 * eval, so its ceiling is set by that teacher -- this is bootstrapping, and one
 * pass is not the destination. Feed the trained weights back in (`EvalFile`) to
 * generate better games and re-run; that loop is the point. Nothing here is
 * validated by loss alone: `match` decides, the same way it decided that a 55%
 * node reduction was worth -154 Elo.
 */
#ifndef ISLAY_TRAIN_HPP
#define ISLAY_TRAIN_HPP

#include <cstdint>
#include <ostream>
#include <string>

#include "options.hpp"

namespace islay {

  struct TrainConfig {
    int           games         = 50000; // self-play games to generate
    int           epochs        = 8;     // passes over those games
    int           depth         = 4;     // self-play search depth. MEASURED: d2->d4 = +81 Elo isolated
                                         // (CI [43,121]); d4->d6 = +4 ns. Sweet spot is 4 -- exact
                                         // endgame labels already perfect the last 12 plies, so deeper
                                         // midgame stops changing outcomes. ~1.4x slower than d2.
    int           opening_plies = 10;    // random plies for diversity: engines are deterministic
    int           solve_empties = 12;    // play EXACTLY once this few squares remain; see train.cpp
    double        lr            = 0.0005; // MEASURED, not guessed -- see below. Do not raise it.
    double        l2            = 1e-6;  // pulls unseen-rare weights back to 0
    double        val_frac      = 0.1;   // games held out to measure generalisation; SPLIT BY GAME
    bool          use_mobility  = true;  // false = don't train the mobility tables (A/B control)
    bool          use_c2x5      = true;  // false = don't train the Corner2x5 table (A/B control)
    bool          use_stab      = true;  // ditto for the two stability tables (feature under test)
    bool          use_par       = true;  // ditto for the region-parity table (feature under test)
    bool          use_front     = true;  // ditto for the two frontier tables (feature under test)
    bool          interp        = false; // fit for stage interpolation (self-play + blended gradient)
    std::uint64_t seed          = 0;
    Rule          rule          = Rule::Othello;
    std::string   out           = "islay.pat";
  };

  struct TrainResult {
    std::uint64_t games     = 0;
    std::uint64_t positions = 0;
    double        rmse      = 0.0; // centi-discs, TRAINING set -- known to point the wrong way; see below
    double        val_rmse  = 0.0; // centi-discs, HELD-OUT games -- the honest fit number
    bool          ok        = false;
  };

  /** Generate, fit, and write `cfg.out`. Progress streams to `log`. */
  TrainResult run_train(const TrainConfig &cfg, std::ostream &log);

  /** NNUE-lite search distillation (nnue.hpp). The loaded EvalFile searches every
   *  post-opening position and its Black-perspective score is the target; the final
   *  result remains only a game record. A .pat teacher seeds a new net, while a
   *  .nnue teacher warm-starts the next round from its embeddings and heads. */
  struct NTrainConfig {
    int           games         = 50000;
    int           epochs        = 10;
    int           depth         = 10;
    int           workers       = 4; // generation only; hard-capped at 4 single-threaded Searchers
    int           opening_plies = 10;
    int           solve_empties = 12;
    double        lr_emb        = 1e-3; // embedding rows (per-disc units; see train.cpp)
    double        lr_out        = 1e-5; // heads see acc ~ tens, so their gradient is ~100x larger
    double        val_frac      = 0.1;
    bool          grouped       = true; // NN4 semantic accumulators + pairwise interaction residuals
    std::uint64_t seed          = 0;
    Rule          rule          = Rule::Othello;
    std::string   out           = "islay.nnue";
  };

  TrainResult run_ntrain(const NTrainConfig &cfg, std::ostream &log);

} // namespace islay

#endif // ISLAY_TRAIN_HPP
