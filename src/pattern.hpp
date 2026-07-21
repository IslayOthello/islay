/**
 * @file pattern.hpp
 * @brief Pattern (n-tuple) evaluation: incremental features + pluggable weights.
 *
 * WHY THIS SHAPE — read before extending.
 *
 * The score is deliberately **linear in the weights**:
 *
 *     score(board, stage) = SUM over pattern instances i of  w[stage][base_i + f_i]
 *
 * where `f_i` is instance i's feature index. Everything a tuner or trainer needs
 * follows from that one property:
 *   * `pattern_features()` hands out the very indices the eval sums, so a training
 *     set is just (indices, target) pairs -- no need to re-implement the eval.
 *   * because the sum is linear, fitting is plain regression / SGD on those
 *     indices; the gradient of the score w.r.t. w[j] is simply "how many times j
 *     occurred". No autodiff, no engine changes.
 *   * `PatternWeights` load/save is the whole integration surface: train
 *     out-of-process, write a file, point the engine at it.
 *
 * FEATURE ENCODING is colour-ABSOLUTE (0 = empty, 1 = Black, 2 = White) and the
 * weight table means "value for BLACK". Mover-relative value is then just a
 * negation for White. That buys two things: antisymmetry is exact by
 * construction (a zero-sum game's value for White IS minus the value for Black),
 * and -- unlike a mover-relative encoding -- no second colour-swapped weight
 * table has to be materialised.
 *
 * INCREMENTAL: `PatternState` keeps one base-3 index per instance. Playing a move
 * touches only the squares that changed, and each square carries a precomputed
 * (instance, place-value) delta list, so an update is a handful of adds. Undo is
 * O(1) because the caller keeps the old state on its own stack -- which is what
 * makes this work at all with islay's value-semantic `Board` (`play()` returns a
 * new board; there is no make/unmake to hang state on).
 *
 * STATUS: TRAINED, and it is now the strong path. train.cpp fits these weights from
 * self-play (`train 100000 15 2 out.pat`), and the measured results against the
 * hand-written eval, 80 games at depth 4 each:
 *     v1 (taught by the hand-written eval): +234 Elo, z = 6.5
 *     v2 (taught by v1)                   : +369 Elo, z = 11.5   (+247 over v1)
 *     v3 (taught by v2)                   : +152 Elo over v2, z = 4.1
 * The loop works and is decelerating (+247 -> +152), which is what convergence looks
 * like. Shipped sets live in weights/; nothing auto-loads, so `pattern_enabled()` is
 * still false until `setoption name EvalFile value weights/v18.pat`, and eval.cpp
 * remains the default for anyone who does not ask.
 *
 * Note the training loss is NOT the signal: v1 finished at rmse 2118 cd against a
 * label s.d. of ~2100 -- i.e. "predicting almost nothing" -- and still won by 234
 * Elo. Most positions in a game carry that game's final result as their label, and
 * an opening position genuinely cannot predict it; what matters for play is RANKING
 * moves, which a weak correlation does well. Judge with match, never with rmse.
 */
#ifndef ISLAY_PATTERN_HPP
#define ISLAY_PATTERN_HPP

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "bitboard.hpp"
#include "board.hpp"
#include "common.hpp"

namespace islay {

  // --- pattern layout ---------------------------------------------------------
  // Instances are grouped by TYPE; the instances of one type share a weight table
  // (they are D4 images of each other, so a symmetric evaluation must give them
  // the same weights -- and sharing cuts both the parameter count and the amount
  // of training data needed by 4x).

  enum class PatternType : std::uint8_t {
    Corner3x3, // 9 sq, 4 instances
    Edge2X,    // 10 sq: a full edge + the two X-squares behind it, 4 instances
    Row2,      // 8 sq, 4 instances (ranks 2/7, files b/g)
    Row3,      // 8 sq, 4 instances
    Row4,      // 8 sq, 4 instances
    Diag8,     // 8 sq, 2 instances
    Diag7,     // 7 sq, 4 instances
    Diag6,     // 6 sq, 4 instances
    Diag5,     // 5 sq, 4 instances
    Diag4,     // 4 sq, 4 instances
    Count
  };

  inline constexpr int kPatternTypes     = static_cast<int>(PatternType::Count);
  // 38 type-based instances + 8 appended Corner2x5 (a 2x5 corner block, both
  // orientations x 4 corners). The 2x5 shares ONE weight table sitting AFTER the
  // mobility tables, so files from before it (v1/v2) still load as a prefix and an
  // older set keeps working as a self-play teacher with the 2x5 zeroed.
  inline constexpr int kC2x5Instances   = 8;
  inline constexpr int kPatternInstances = 4 + 4 + 4 + 4 + 4 + 2 + 4 + 4 + 4 + 4 + 8; // = 46
  // TWO capacity experiments were appended here and both MEASURED NEUTRAL. Read this
  // before adding a third -- the pattern set looks saturated, and the rule that
  // survived both is sharper than the one either produced alone.
  //
  //   Corner2x4 (8 sq):  +8 Elo, CI [-26, 43], isolated, 300 games.
  //   Corner2x6 (12 sq): +3.5 Elo, CI [-14, 21], isolated, 600 pairs.
  //
  // Corner2x4 failed for the obvious reason: it is a strict SUBSET of Corner2x5, so it
  // carries nothing 2x5 does not already have. That produced the first rule -- a new
  // pattern must be BIGGER than what exists.
  //
  // Corner2x6 satisfied that rule (a strict SUPERSET of the 2x5 that won +62) and was
  // still neutral, so bigger is NECESSARY BUT NOT SUFFICIENT. The added squares must
  // lie where nothing else already looks. The 2x6 spans rows 1-2, files a-f, and every
  // one of those squares is already covered: Edge2X sees all of row 1, Row2 sees all of
  // row 2, and Corner3x3 + Corner2x5 see the corner end. Only the JOINT configuration
  // was new, and that turned out to be worth nothing measurable.
  //
  // The data explanation was raised and FALSIFIED rather than assumed. 3^12 = 531441
  // cells is 9x the 2x5's table, and at 150K games only 8.5% of cells were ever touched
  // (~97 samples each, against the 2x5's ~326) -- so "too sparse to learn" was the
  // natural suspect. Retraining both arms at 500K games lifted that to ~216 samples per
  // touched cell and the result did not move: +2.3 Elo, CI [-15, 20]. More data does not
  // rescue information that is not there.
  //
  // It also was not free: 8 more instances per leaf, indexing a 531441-entry table,
  // cost about 6% of nps (9.8M -> 9.2M) with the node count byte-identical.
  inline constexpr int kMaxPatternSquares = 10;

  /**
   * MOBILITY, as a trained feature. n-tuple patterns capture static disc SHAPE and
   * are blind to how many moves a side has -- two positions with identical patterns
   * but different mobility score identically, and mobility is the single most
   * important positional signal in Othello. So each stage carries two extra weight
   * tables, indexed by the (capped) legal-move count of Black and of White, trained
   * by the same SGD as the patterns. Kept as two colour-absolute tables rather than
   * one difference table so the +1-coefficient training loop is untouched; training
   * learns w_white ~= -w_black, the same approximate antisymmetry the patterns have.
   */
  inline constexpr int kMobBuckets = 33; // move counts 0..32, capped

  /**
   * STABILITY, as a trained feature -- the same bet mobility won (+52 Elo), for the
   * same reason. An n-tuple pattern sees a fixed WINDOW of squares, so it is blind to
   * any property that depends on the whole board. "How many of my discs are provably
   * unflippable" is exactly such a property: it comes from a fixpoint over full lines
   * and edge anchors across all 64 squares (stability.hpp), and two positions with
   * identical corner/edge windows can have very different stable counts.
   *
   * This is deliberately NOT a cheap corner-anchored approximation. Cheap edge
   * stability is a deterministic function of the edge configuration, and Edge2X plus
   * Corner3x3 plus Corner2x5 already encode that configuration exactly -- so it would
   * be redundant by construction, which is precisely how Corner2x6 failed. The whole
   * value here is the GLOBAL part that no window can see, so the real fixpoint is what
   * gets used. Measured cost: about +9% search time for both colours.
   *
   * Two colour-absolute tables like mobility, so antisymmetry stays exact and the
   * +1-coefficient training loop is untouched.
   */
  inline constexpr int kStabBuckets = 65; // stable disc counts 0..64

  /**
   * REGION PARITY, as a trained feature. Who gets the LAST move of an empty region is
   * a first-order Othello concept, and this eval provably cannot express it: the score
   * is LINEAR in per-instance weights, and parity is an XOR-like function of the board,
   * which no weighted sum of independent windows can represent -- no matter how many
   * windows there are. That is the same structural argument mobility (+52) and
   * stability (+12.7) won on, and it is why this is worth a table rather than a bigger
   * pattern.
   *
   * WHY THIS IS NOT THE "IMPOSSIBLE GLOBAL PARITY TERM". The hand-written eval could
   * not carry one: parity depends only on the EMPTY squares, so it is unchanged by the
   * p<->o swap, i.e. EVEN, while antisymmetry demands ODD. The escape is to index by
   * WHOSE TURN IT IS. Swapping p<->o also swaps the side to move, so the index moves to
   * the other half of the table and antisymmetry is satisfied by w_white ~= -w_black --
   * exactly the approximate antisymmetry mobility and stability already learn.
   *
   * The bucket folds the two cheap parities that matter: the global empty parity (who
   * is on track for the last move overall) and how many QUADRANTS hold an odd number of
   * empties (a cheap stand-in for true connected-region parity, already used for
   * endgame move ordering in search.cpp). Both are a few instructions, so unlike
   * stability this feature is essentially free.
   */
  /**
   * FRONTIER DISCS, as a trained feature: discs of a colour that touch at least one
   * empty square. Few frontier discs is the classic Othello sign of a safe shape --
   * an exposed disc is what gives the opponent something to flip against.
   *
   * WEAKER PRIOR THAN THE OTHER THREE, and that is worth stating up front. Mobility,
   * stability and parity all won because they are things a sum of independent windows
   * provably cannot express. Frontier is NOT of that kind: it is a sum of LOCAL 3x3
   * indicators, exactly the shape a window-based linear model is good at -- and
   * POTENTIAL MOBILITY, a close relative, already measured NEUTRAL here for precisely
   * the reason that the edge/corner patterns imply it. The one argument for trying it
   * anyway is that the pattern set has no general 3x3 window (only the four corners),
   * and rows/diagonals are one-dimensional so they cannot see a vertical neighbour --
   * so frontier is only PARTLY covered.
   *
   * THE PRIOR WAS WRONG, and that is the lesson worth keeping. Measured isolated
   * against a control trained from the same teacher, seed and games with only the gate
   * differing: **+11.1 Elo, 95% CI [4, 18], z = 3.22** over 4046 games at equal nodes.
   * The mistake was stopping at "frontier is a sum of local 3x3 indicators, which a
   * window model can express". The missing step is whether those windows EXIST: this
   * set has no general 3x3 window (only the four corners), and rows and diagonals are
   * one-dimensional, so for the ~60 non-corner squares a disc's frontier status is
   * invisible to every instance. Expressible-in-principle is not the test; expressible
   * BY THE WINDOWS ACTUALLY PRESENT is.
   *
   * val_rmse understated it again: 1564 against the control's 1567, a 3 cd gap, for an
   * 11 Elo feature -- the same direction of error stability showed (4 cd, +12.7 Elo).
   */
  inline constexpr int kFrontBuckets = 41; // frontier disc counts 0..40, capped

  inline constexpr int kParityBuckets = 10; // (empties & 1) + 2 * odd-quadrant count, 0..9

  /**
   * Legal move counts of each colour. (Potential mobility -- empties next to the
   * enemy -- was tried here as a second trained feature and measured NEUTRAL, so it
   * is not carried; the struct stays for the clean two-count API.)
   */
  struct MobCounts {
    int black_mob = 0, white_mob = 0;
    int black_stab = 0, white_stab = 0; // provably-unflippable discs; see kStabBuckets
    int parity = 0;                     // 0..2*kParityBuckets-1, side-to-move folded in
    int black_front = 0, white_front = 0; // discs touching an empty square; see kFrontBuckets
  };

  /** Both move counts from a mover-relative board + whose turn it is. */
  [[nodiscard]] MobCounts mob_counts(const Board &b, Color stm) noexcept;
  /** Same, but the mover's legal moves are already in hand (search has them at the leaf),
   *  so only the opponent's are generated. Byte-identical to the two-argument form. */
  [[nodiscard]] MobCounts mob_counts(const Board &b, Color stm, Bitboard mover_moves) noexcept;

  /**
   * Game stage = disc count bucketed. Weights are per stage because the value of
   * every feature swings hard between opening and endgame.
   */
  inline constexpr int kStageCount = 15; // 4..64 discs, 4 discs per bucket
  [[nodiscard]] ISLAY_FORCEINLINE int pattern_stage(int discs) noexcept {
    const int s = (discs - 4) / 4;
    return s < 0 ? 0 : (s >= kStageCount ? kStageCount - 1 : s);
  }

  /** Total weights in ONE stage (sum of 3^n over the types, plus a bias term). */
  [[nodiscard]] std::size_t pattern_weights_per_stage() noexcept;

  /** Offset of `t`'s table inside a stage; add the feature index to it. */
  [[nodiscard]] std::size_t pattern_type_base(PatternType t) noexcept;

  // --- incremental state ------------------------------------------------------

  /**
   * One base-3 feature index per instance. Colour-absolute (see the file header).
   * Copy it to recurse and drop the copy to undo: it is 38 ints, and this is what
   * keeps the value-semantic Board workable.
   */
  struct PatternState {
    std::int32_t f[kPatternInstances];

    /** Rebuild from scratch. O(64); use at the search root, not per node. */
    void set(const Board &b, Color stm) noexcept;

    /**
     * Apply a move: `sq` was placed by `mover`, `flipped` changed hands.
     * Cheap -- only the touched squares' instances move.
     */
    void update(Square sq, Bitboard flipped, Color mover) noexcept;
  };

  // --- weights ----------------------------------------------------------------

  /**
   * Weights for every stage, flat: `w[stage * per_stage + base + feature]`.
   * Value is in centi-discs, from BLACK's point of view.
   *
   * File format ("ISLAYPAT", little-endian, same-machine) -- deliberately trivial
   * so a trainer in any language can emit it:
   *   char     magic[8]  = "ISLAYPAT"
   *   uint32   version   = 1
   *   uint32   stages    = kStageCount
   *   uint64   per_stage = pattern_weights_per_stage()
   *   int16    w[stages * per_stage]
   */
  class PatternWeights {
  public:
    [[nodiscard]] bool loaded() const noexcept { return loaded_; }
    bool load(const std::string &path, std::ostream &log);
    bool save(const std::string &path, std::ostream &log) const;

    /** Allocate zeroed weights (a starting point for a tuner). */
    void reset_zero();

    /** Drop the weights; pattern_enabled() goes false and the hand-written eval runs. */
    void unload() noexcept;

    [[nodiscard]] std::int16_t *data() noexcept { return w_.data(); }
    [[nodiscard]] const std::int16_t *data() const noexcept { return w_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return w_.size(); }

    /** Score for BLACK, centi-discs. Caller negates for White. Counts are
     *  colour-absolute; a default MobCounts{} simply zeroes those terms. */
    [[nodiscard]] int score(const PatternState &s, int stage, const MobCounts &mc) const noexcept;

    /** Black score for a given DISC COUNT. `interp` linearly blends toward the next
     *  stage inside each 4-disc bucket (A/B feature); off = the plain per-stage score. */
    [[nodiscard]] int score_phase(const PatternState &s, int discs, const MobCounts &mc, bool interp) const noexcept;

  private:
    std::vector<std::int16_t> w_;
    bool                      loaded_ = false;
  };

  // --- tuning / training surface ---------------------------------------------

  /**
   * The exact flat weight indices this position contributes to, one per instance
   * plus the bias. THIS is the training hook: the score is their weights summed,
   * so a design row is just these indices with coefficient 1 (repeats count).
   *
   * `out` must hold kPatternInstances + 8 entries (patterns + bias + 2 mobility
   * + 2 stability + 1 parity + 2 frontier).
   * Returns how many were written.
   */
  int pattern_features(const Board &b, Color stm, std::uint32_t *out) noexcept;

  /**
   * Same indices, but from an already-incremental PatternState. The trainer walks
   * games move by move, so rebuilding features from scratch per position (O(64))
   * would dominate its inner loop; this is what makes replaying a game cheap.
   */
  int pattern_indices(const PatternState &s, int stage, const MobCounts &mc, std::uint32_t *out) noexcept;

  /** Linear stage interpolation (A/B feature): global, off by default. */
  void pattern_set_stage_interp(bool on) noexcept;
  [[nodiscard]] bool pattern_stage_interp() noexcept;

  /** True once weights are loaded; until then eval.cpp's hand-written eval runs. */
  [[nodiscard]] bool pattern_enabled() noexcept;
  [[nodiscard]] PatternWeights &pattern_weights() noexcept;

  /**
   * Point the engine at a different weight set (nullptr = back to the built-in
   * one). Exists so a match harness can pit two evals against each other inside
   * one process: swapping a pointer per move is free, whereas copying ~3MB of
   * weights would not be. An UNLOADED PatternWeights selects the hand-written
   * eval, which is how "pattern vs handcrafted" is expressed.
   */
  void pattern_set_active(PatternWeights *w) noexcept;

  /** Incremental update must equal a from-scratch rebuild, on a random sweep. */
  [[nodiscard]] bool pattern_selftest() noexcept;

} // namespace islay

#endif // ISLAY_PATTERN_HPP
