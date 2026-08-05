// Linear colour-absolute n-tuples with incremental base-3 indices.
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

  // D4 images of a pattern type share one table.

  enum class PatternType : std::uint8_t {
    Corner3x3, // 9 sq, 4 instances
    Edge2X, // 10 sq: a full edge + the two X-squares behind it, 4 instances
    Row2, // 8 sq, 4 instances (ranks 2/7, files b/g)
    Row3, // 8 sq, 4 instances
    Row4, // 8 sq, 4 instances
    Diag8, // 8 sq, 2 instances
    Diag7, // 7 sq, 4 instances
    Diag6, // 6 sq, 4 instances
    Diag5, // 5 sq, 4 instances
    Diag4, // 4 sq, 4 instances
    Count
  };

  inline constexpr int kPatternTypes = static_cast<int>(PatternType::Count);
  // 38 base instances plus 8 Corner2x5; appended tables keep old files prefix-compatible.
  inline constexpr int kC2x5Instances    = 8;
  inline constexpr int kPatternInstances = 4 + 4 + 4 + 4 + 4 + 2 + 4 + 4 + 4 + 4 + 8; // = 46
  // Corner2x4 and Corner2x6 were neutral; keep the trained layout capped at ten squares.
  inline constexpr int kMaxPatternSquares = 10;

  // Colour-absolute mobility tables, capped at 32 moves.
  inline constexpr int kMobBuckets = 33; // move counts 0..32, capped

  // Full-board stable-disc counts; the fixpoint costs about 9% search time.
  inline constexpr int kStabBuckets = 65; // stable disc counts 0..64

  // Parity folds side-to-move, global empty parity, and odd-quadrant count.
  // Frontier counts cover local shape missing from the one-dimensional patterns.
  inline constexpr int kFrontBuckets = 41; // frontier disc counts 0..40, capped

  inline constexpr int kParityBuckets = 10; // (empties & 1) + 2 * odd-quadrant count, 0..9

  struct MobCounts {
    int black_mob = 0, white_mob = 0;
    int black_stab = 0, white_stab = 0; // provably-unflippable discs; see kStabBuckets
    int parity      = 0; // 0..2*kParityBuckets-1, side-to-move folded in
    int black_front = 0, white_front = 0; // discs touching an empty square; see kFrontBuckets
  };

  [[nodiscard]] MobCounts mob_counts(const Board &b, Color stm) noexcept;
  // Reuses an already-generated mover move mask.
  [[nodiscard]] MobCounts mob_counts(const Board &b, Color stm, Bitboard mover_moves) noexcept;

  inline constexpr int                kStageCount = 15; // 4..64 discs, 4 discs per bucket
  [[nodiscard]] ISLAY_FORCEINLINE int pattern_stage(int discs) noexcept {
    const int s = (discs - 4) / 4;
    return s < 0 ? 0 : (s >= kStageCount ? kStageCount - 1 : s);
  }

  [[nodiscard]] std::size_t pattern_weights_per_stage() noexcept;

  [[nodiscard]] std::size_t pattern_type_base(PatternType t) noexcept;

  // One colour-absolute base-3 index per instance.
  struct PatternState {
    std::int32_t f[kPatternInstances];

    // Root-only O(64) rebuild.
    void set(const Board &b, Color stm) noexcept;

    void update(Square sq, Bitboard flipped, Color mover) noexcept;
  };

  // Flat Black-relative centi-disc weights; ISLAYPAT is little-endian.
  class PatternWeights {
  public:
    [[nodiscard]] bool loaded() const noexcept { return loaded_; }
    bool               load(const std::string &path, std::ostream &log);
    bool               save(const std::string &path, std::ostream &log) const;

    void reset_zero();

    void unload() noexcept;

    [[nodiscard]] std::int16_t       *data() noexcept { return w_.data(); }
    [[nodiscard]] const std::int16_t *data() const noexcept { return w_.data(); }
    [[nodiscard]] std::size_t         size() const noexcept { return w_.size(); }

    [[nodiscard]] int score(const PatternState &s, int stage, const MobCounts &mc) const noexcept;

    // Optionally interpolate inside a four-disc stage.
    [[nodiscard]] int score_phase(const PatternState &s, int discs, const MobCounts &mc, bool interp) const noexcept;

  private:
    std::vector<std::int16_t> w_;
    bool                      loaded_ = false;
  };

  // `out` needs kPatternInstances + 8 entries.
  int pattern_features(const Board &b, Color stm, std::uint32_t *out) noexcept;

  int pattern_indices(const PatternState &s, int stage, const MobCounts &mc, std::uint32_t *out) noexcept;

  void               pattern_set_stage_interp(bool on) noexcept;
  [[nodiscard]] bool pattern_stage_interp() noexcept;

  [[nodiscard]] bool            pattern_enabled() noexcept;
  [[nodiscard]] PatternWeights &pattern_weights() noexcept;

  // nullptr selects the built-in weights; an unloaded set selects handcrafted eval.
  void pattern_set_active(PatternWeights *w) noexcept;

  [[nodiscard]] bool pattern_selftest() noexcept;

} // namespace islay

#endif // ISLAY_PATTERN_HPP
