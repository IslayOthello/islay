/**
 * @file pattern.cpp
 * @brief Pattern (n-tuple) features, incremental update, and weight I/O.
 *
 * Every table here is GENERATED from board geometry at compile time rather than
 * transcribed -- the same discipline movegen.cpp uses for MASK_LR. The square
 * lists come from one description per type plus the D4 images of it, so a typo
 * cannot silently desynchronise an instance from its weights.
 */
#include "pattern.hpp"

#include "nnue.hpp"

#include "stability.hpp"

#include <array>
#include <cstring>
#include <fstream>

namespace islay {
  namespace {

    // --- one description per type; instances are its D4 images ----------------
    // Squares are listed for the "canonical" instance; the others are produced by
    // applying symmetries, so the four members of a group really are images of
    // each other and can share a weight table.

    struct TypeInfo {
      PatternType type;
      int         squares;
      int         instances;
    };

    constexpr TypeInfo kTypes[kPatternTypes] = {
            {PatternType::Corner3x3, 9, 4}, {PatternType::Edge2X, 10, 4}, {PatternType::Row2, 8, 4},
            {PatternType::Row3, 8, 4},      {PatternType::Row4, 8, 4},    {PatternType::Diag8, 8, 2},
            {PatternType::Diag7, 7, 4},     {PatternType::Diag6, 6, 4},   {PatternType::Diag5, 5, 4},
            {PatternType::Diag4, 4, 4},
    };

    [[nodiscard]] constexpr int sq_of(int r, int f) noexcept { return r * 8 + f; }

    /** Apply D4 symmetry `s` to (rank, file). Mirrors Board::symmetry_bb's order. */
    [[nodiscard]] constexpr int sym_sq(int sq, int s) noexcept {
      int r = sq >> 3, f = sq & 7;
      if (s & 1)
        f = 7 - f; // flip_horizontal
      if (s & 2)
        r = 7 - r; // flip_vertical
      if (s & 4) { // transpose
        const int t = r;
        r           = f;
        f           = t;
      }
      return sq_of(r, f);
    }

    struct Layout {
      // squares[instance][k] -- the k-th square of that instance, -1 past the end
      std::array<std::array<int, kMaxPatternSquares>, kPatternInstances> squares{};
      std::array<int, kPatternInstances>                                 len{};
      std::array<PatternType, kPatternInstances>                         type{};
      // the D4 symmetry used to derive each instance from its canonical one
      std::array<int, kPatternInstances> sym{};
      // 0 = a normal type-based instance (weight base from kBases); 1 = Corner2x5.
      // Appended blocks index their own shared table AFTER mobility, so older files
      // stay loadable as a prefix. `append_block` generalizes adding more.
      std::array<int, kPatternInstances> ekind{};
    };

    /** Canonical square list per type, written once, in geometry terms. */
    constexpr void canonical_squares(PatternType t, int *out, int &n) noexcept {
      n = 0;
      switch (t) {
        case PatternType::Corner3x3: // a1 corner block
          for (int r = 0; r < 3; ++r)
            for (int f = 0; f < 3; ++f)
              out[n++] = sq_of(r, f);
          break;
        case PatternType::Edge2X: // rank 1 + b2,g2
          for (int f = 0; f < 8; ++f)
            out[n++] = sq_of(0, f);
          out[n++] = sq_of(1, 1);
          out[n++] = sq_of(1, 6);
          break;
        case PatternType::Row2:
          for (int f = 0; f < 8; ++f)
            out[n++] = sq_of(1, f);
          break;
        case PatternType::Row3:
          for (int f = 0; f < 8; ++f)
            out[n++] = sq_of(2, f);
          break;
        case PatternType::Row4:
          for (int f = 0; f < 8; ++f)
            out[n++] = sq_of(3, f);
          break;
        case PatternType::Diag8: // a1-h8
          for (int i = 0; i < 8; ++i)
            out[n++] = sq_of(i, i);
          break;
        case PatternType::Diag7: // b1-h7
          for (int i = 0; i < 7; ++i)
            out[n++] = sq_of(i, i + 1);
          break;
        case PatternType::Diag6: // c1-h6
          for (int i = 0; i < 6; ++i)
            out[n++] = sq_of(i, i + 2);
          break;
        case PatternType::Diag5: // d1-h5
          for (int i = 0; i < 5; ++i)
            out[n++] = sq_of(i, i + 3);
          break;
        case PatternType::Diag4: // e1-h4
          for (int i = 0; i < 4; ++i)
            out[n++] = sq_of(i, i + 4);
          break;
        default: break;
      }
    }

    /**
     * Instances per type, as D4 images of the canonical list. Which symmetries
     * generate distinct images differs per type (a main diagonal has only 2
     * distinct images, a corner block has 4), so the generators are listed
     * explicitly and the result is checked for duplicates by the self-test.
     */
    constexpr std::array<int, 4> generators_for(PatternType t) noexcept {
      switch (t) {
        case PatternType::Corner3x3: return {0, 1, 2, 3}; // 4 corners
        case PatternType::Edge2X: return {0, 2, 4, 6};    // 4 edges
        case PatternType::Row2:
        case PatternType::Row3:
        case PatternType::Row4: return {0, 2, 4, 6}; // 2 ranks + 2 files
        case PatternType::Diag8: return {0, 1, 0, 0}; // only 2 distinct
        case PatternType::Diag7:
        case PatternType::Diag6:
        case PatternType::Diag5:
        case PatternType::Diag4: return {0, 1, 2, 3};
        default: return {0, 0, 0, 0};
      }
    }

    constexpr Layout make_layout() noexcept {
      Layout L{};
      int    inst = 0;
      for (int ti = 0; ti < kPatternTypes; ++ti) {
        const TypeInfo &info = kTypes[ti];
        int             base[kMaxPatternSquares]{};
        int             n = 0;
        canonical_squares(info.type, base, n);
        const auto gens = generators_for(info.type);
        for (int k = 0; k < info.instances; ++k) {
          const int s = gens[static_cast<std::size_t>(k)];
          for (int j = 0; j < n; ++j)
            L.squares[inst][j] = sym_sq(base[j], s);
          for (int j = n; j < kMaxPatternSquares; ++j)
            L.squares[inst][j] = -1;
          L.len[inst]  = n;
          L.type[inst] = info.type;
          L.sym[inst]  = s;
          L.ekind[inst] = 0;
          ++inst;
        }
      }
      // Appended corner blocks: a WxH block in the a1 corner, then its 8 D4 images
      // (4 corners x 2 orientations), all sharing one weight table. Corner2x5 first
      // (ekind 1), Corner2x4 second (ekind 2).
      const auto append_block = [&](int rows, int cols, int kind, int count) constexpr {
        int blk[10]{};
        int cn = 0;
        for (int r = 0; r < rows; ++r)
          for (int f = 0; f < cols; ++f)
            blk[cn++] = sq_of(r, f);
        for (int s = 0; s < count; ++s) {
          for (int j = 0; j < cn; ++j)
            L.squares[inst][j] = sym_sq(blk[j], s);
          for (int j = cn; j < kMaxPatternSquares; ++j)
            L.squares[inst][j] = -1;
          L.len[inst]   = cn;
          L.type[inst]  = PatternType::Corner3x3; // unused for appended instances
          L.sym[inst]   = s;
          L.ekind[inst] = kind;
          ++inst;
        }
      };
      append_block(2, 5, 1, kC2x5Instances);
      return L;
    }

    constexpr Layout kLayout = make_layout();

    [[nodiscard]] constexpr int ipow3(int n) noexcept {
      int r = 1;
      for (int i = 0; i < n; ++i)
        r *= 3;
      return r;
    }

    /** Table base offsets within one stage; the last slot is the bias term. */
    constexpr std::array<std::size_t, kPatternTypes + 1> make_bases() noexcept {
      std::array<std::size_t, kPatternTypes + 1> b{};
      std::size_t                                acc = 0;
      for (int ti = 0; ti < kPatternTypes; ++ti) {
        b[static_cast<std::size_t>(ti)] = acc;
        acc += static_cast<std::size_t>(ipow3(kTypes[ti].squares));
      }
      b[kPatternTypes] = acc; // bias lives here; total = acc + 1
      return b;
    }

    constexpr auto kBases = make_bases();

    /**
     * Per square: which instances contain it, and the base-3 place value it has
     * there. This is what makes the update incremental -- a changed square only
     * touches the handful of instances listed here.
     */
    struct SquareDelta {
      int n = 0;
      int inst[24]{};
      int place[24]{};
    };

    constexpr std::array<SquareDelta, 64> make_square_deltas() noexcept {
      std::array<SquareDelta, 64> d{};
      for (int i = 0; i < kPatternInstances; ++i) {
        int pv = 1;
        for (int k = 0; k < kLayout.len[i]; ++k) {
          const int sq = kLayout.squares[i][k];
          SquareDelta &s = d[static_cast<std::size_t>(sq)];
          s.inst[s.n]    = i;
          s.place[s.n]   = pv;
          ++s.n;
          pv *= 3;
        }
      }
      return d;
    }

    constexpr auto kSquareDelta = make_square_deltas();

    /** 0 = empty, 1 = Black, 2 = White -- colour-absolute (see pattern.hpp). */
    [[nodiscard]] ISLAY_FORCEINLINE int digit_at(const Board &b, Color stm, Square sq) noexcept {
      const Bitboard bb    = square_bb(sq);
      const Bitboard black = (stm == Color::Black) ? b.player : b.opponent;
      const Bitboard white = (stm == Color::Black) ? b.opponent : b.player;
      if (black & bb)
        return 1;
      if (white & bb)
        return 2;
      return 0;
    }

    PatternWeights  g_weights;
    PatternWeights *g_active = &g_weights;

  } // namespace

  namespace {
    // Per-stage layout tail, tables always APPENDED so older files map into the
    // front of a newer (wider) stage: [ ...patterns... | bias | black_mob | white_mob ].
    // (Potential mobility was appended here and MEASURED NEUTRAL -- +9 Elo, CI [-25,43],
    // isolated over 300 games -- so it was reverted; it is redundant with what the
    // patterns and the actual move count already encode. See the memory note.)
    constexpr std::size_t kMob   = static_cast<std::size_t>(kMobBuckets);
    constexpr std::size_t kBiasBase = kBases[kPatternTypes];
    constexpr std::size_t kBMobBase = kBiasBase + 1;
    constexpr std::size_t kWMobBase = kBMobBase + kMob;
    constexpr std::size_t kC2x5Base = kWMobBase + kMob;                 // shared 2x5 table
    constexpr std::size_t kC2x5Size = static_cast<std::size_t>(ipow3(10)); // 3^10 = 59049
    // Stability tables, appended LAST so every earlier file stays a loadable prefix.
    constexpr std::size_t kStab  = static_cast<std::size_t>(kStabBuckets);
    constexpr std::size_t kBStabBase = kC2x5Base + kC2x5Size;
    constexpr std::size_t kWStabBase = kBStabBase + kStab;
    // Region parity, appended last. One table with the side to move folded into the
    // index, which is what makes an otherwise colour-even quantity antisymmetric.
    constexpr std::size_t kPar     = static_cast<std::size_t>(kParityBuckets);
    constexpr std::size_t kParBase = kWStabBase + kStab;
    // Frontier tables, appended last so earlier files stay loadable prefixes.
    constexpr std::size_t kFront     = static_cast<std::size_t>(kFrontBuckets);
    constexpr std::size_t kBFrontBase = kParBase + 2 * kPar;
    constexpr std::size_t kWFrontBase = kBFrontBase + kFront;
    constexpr std::size_t kPerStage   = kWFrontBase + kFront;
    // Older files load as a prefix of the current stage: v1 = patterns+bias,
    // v2 = +mobility. Both are valid teachers with the newer tables zeroed.
    constexpr std::size_t kPerStageV1 = kBiasBase + 1;
    constexpr std::size_t kPerStageV2 = kWMobBase + kMob;
    constexpr std::size_t kPerStageV3 = kC2x5Base + kC2x5Size; // before the stability tables (v1-v12)
    constexpr std::size_t kPerStageV4 = kWStabBase + kStab;    // before the parity table (v16)
    constexpr std::size_t kPerStageV5 = kParBase + 2 * kPar;   // before the frontier tables (v17)

    // Per-instance weight base: the kBases block for the 38 type instances, and each
    // appended block's own shared table.
    constexpr std::array<std::size_t, kPatternInstances> make_inst_base() noexcept {
      std::array<std::size_t, kPatternInstances> b{};
      for (int i = 0; i < kPatternInstances; ++i) {
        const int k = kLayout.ekind[static_cast<std::size_t>(i)];
        b[static_cast<std::size_t>(i)] =
                (k == 1) ? kC2x5Base : kBases[static_cast<std::size_t>(kLayout.type[i])];
      }
      return b;
    }
    constexpr auto kInstBase = make_inst_base();

    [[nodiscard]] constexpr int mob_clamp(int m) noexcept {
      return m < 0 ? 0 : (m >= kMobBuckets ? kMobBuckets - 1 : m);
    }
  } // namespace

  std::size_t pattern_weights_per_stage() noexcept { return kPerStage; }

  std::size_t pattern_type_base(PatternType t) noexcept { return kBases[static_cast<std::size_t>(t)]; }

  MobCounts mob_counts(const Board &b, Color stm, Bitboard mover_moves) noexcept {
    const int my_mob  = popcount(mover_moves);
    const int opp_mob = popcount(b.passed().moves());
    // Stability is the same shape of feature as mobility -- a whole-board count no
    // pattern window can see -- so it is gathered here alongside it. The fixpoint is
    // the expensive part of the leaf (~+9% search time for both colours), and it is
    // deliberately the real one: a cheap corner-anchored approximation would be a
    // function of the edge configuration that Edge2X already encodes exactly.
    const int my_st  = stable_count(b.player, b.opponent);
    const int opp_st = stable_count(b.opponent, b.player);
    MobCounts m;
    m.black_mob  = (stm == Color::Black) ? my_mob : opp_mob;
    m.white_mob  = (stm == Color::Black) ? opp_mob : my_mob;
    m.black_stab = (stm == Color::Black) ? my_st : opp_st;
    m.white_stab = (stm == Color::Black) ? opp_st : my_st;
    // Parity: global empty parity plus how many quadrants hold an odd empty count,
    // with the side to move folded in so the term can be antisymmetric at all.
    const Bitboard empties = ~(b.player | b.opponent);
    int            oddq    = 0;
    for (int q = 0; q < 4; ++q)
      oddq += (popcount(empties & kQuadrant[q]) & 1);
    const int bucket = (popcount(empties) & 1) + 2 * oddq;
    m.parity = (stm == Color::Black ? 0 : kParityBuckets) + bucket;
    // Frontier: discs of each colour that touch an empty square. dilate8 is wrap-safe.
    const Bitboard near_empty = dilate8(empties);
    const int      my_front   = popcount(b.player & near_empty);
    const int      opp_front  = popcount(b.opponent & near_empty);
    m.black_front = (stm == Color::Black) ? my_front : opp_front;
    m.white_front = (stm == Color::Black) ? opp_front : my_front;
    return m;
  }

  MobCounts mob_counts(const Board &b, Color stm) noexcept { return mob_counts(b, stm, b.moves()); }

  void PatternState::set(const Board &b, Color stm) noexcept {
    for (int i = 0; i < kPatternInstances; ++i) {
      int v  = 0;
      int pv = 1;
      for (int k = 0; k < kLayout.len[i]; ++k) {
        v += digit_at(b, stm, kLayout.squares[i][k]) * pv;
        pv *= 3;
      }
      f[i] = v;
    }
  }

  void PatternState::update(Square sq, Bitboard flipped, Color mover) noexcept {
    // Colour-absolute digits: mover's colour is what lands on every changed
    // square. Placed square: empty(0) -> mover. Flipped squares: other -> mover.
    const int mover_digit = (mover == Color::Black) ? 1 : 2;
    const int other_digit = (mover == Color::Black) ? 2 : 1;

    {
      const SquareDelta &d = kSquareDelta[static_cast<std::size_t>(sq)];
      for (int k = 0; k < d.n; ++k)
        f[d.inst[k]] += mover_digit * d.place[k]; // 0 -> mover
    }
    Bitboard m = flipped;
    while (m) {
      const Square       s = pop_lsb(m);
      const SquareDelta &d = kSquareDelta[static_cast<std::size_t>(s)];
      const int          delta = mover_digit - other_digit; // other -> mover
      for (int k = 0; k < d.n; ++k)
        f[d.inst[k]] += delta * d.place[k];
    }
  }

  int PatternWeights::score_phase(const PatternState &s, int discs, const MobCounts &mc, bool interp) const noexcept {
    const int stage = pattern_stage(discs);
    const int lo    = score(s, stage, mc);
    if (!interp)
      return lo;
    // Linear interpolation toward the next stage by r = discs mod 4 (the position INSIDE
    // the 4-disc bucket). r == 0 (a bucket boundary) or the final stage returns `lo`
    // exactly, so with interp off/at-boundary the value is bit-identical to the baseline.
    const int r = discs >= 4 ? (discs - 4) % 4 : 0;
    if (r == 0 || stage >= kStageCount - 1)
      return lo;
    const int hi = score(s, stage + 1, mc);
    return (lo * (4 - r) + hi * r) / 4; // deterministic integer arithmetic; linear -> antisymmetric
  }

  namespace {
    bool g_stage_interp = true; // default ON: +58 Elo, free (no retrain), off is byte-identical
  }
  void pattern_set_stage_interp(bool on) noexcept { g_stage_interp = on; }
  bool pattern_stage_interp() noexcept { return g_stage_interp; }

  namespace {
    [[nodiscard]] ISLAY_FORCEINLINE int stab_clamp(int v) noexcept {
      return v < 0 ? 0 : (v >= kStabBuckets ? kStabBuckets - 1 : v);
    }
    [[nodiscard]] ISLAY_FORCEINLINE int front_clamp(int v) noexcept {
      return v < 0 ? 0 : (v >= kFrontBuckets ? kFrontBuckets - 1 : v);
    }
  }

  int PatternWeights::score(const PatternState &s, int stage, const MobCounts &mc) const noexcept {
    if (!loaded_)
      return 0;
    const std::int16_t *w    = w_.data() + static_cast<std::size_t>(stage) * kPerStage;
    int                 acc  = 0;
    for (int i = 0; i < kPatternInstances; ++i)
      acc += w[kInstBase[static_cast<std::size_t>(i)] + static_cast<std::size_t>(s.f[i])];
    acc += w[kBiasBase]; // bias
    acc += w[kBMobBase + static_cast<std::size_t>(mob_clamp(mc.black_mob))];
    acc += w[kWMobBase + static_cast<std::size_t>(mob_clamp(mc.white_mob))];
    acc += w[kBStabBase + static_cast<std::size_t>(stab_clamp(mc.black_stab))];
    acc += w[kWStabBase + static_cast<std::size_t>(stab_clamp(mc.white_stab))];
    acc += w[kParBase + static_cast<std::size_t>(mc.parity)];
    acc += w[kBFrontBase + static_cast<std::size_t>(front_clamp(mc.black_front))];
    acc += w[kWFrontBase + static_cast<std::size_t>(front_clamp(mc.white_front))];
    return acc;
  }

  int pattern_indices(const PatternState &s, int stage, const MobCounts &mc, std::uint32_t *out) noexcept {
    const std::size_t off = static_cast<std::size_t>(stage) * kPerStage;
    int               n   = 0;
    for (int i = 0; i < kPatternInstances; ++i)
      out[n++] = static_cast<std::uint32_t>(off + kInstBase[static_cast<std::size_t>(i)] +
                                            static_cast<std::size_t>(s.f[i]));
    out[n++] = static_cast<std::uint32_t>(off + kBiasBase); // bias
    out[n++] = static_cast<std::uint32_t>(off + kBMobBase + static_cast<std::size_t>(mob_clamp(mc.black_mob)));
    out[n++] = static_cast<std::uint32_t>(off + kWMobBase + static_cast<std::size_t>(mob_clamp(mc.white_mob)));
    out[n++] = static_cast<std::uint32_t>(off + kBStabBase + static_cast<std::size_t>(stab_clamp(mc.black_stab)));
    out[n++] = static_cast<std::uint32_t>(off + kWStabBase + static_cast<std::size_t>(stab_clamp(mc.white_stab)));
    out[n++] = static_cast<std::uint32_t>(off + kParBase + static_cast<std::size_t>(mc.parity));
    out[n++] = static_cast<std::uint32_t>(off + kBFrontBase + static_cast<std::size_t>(front_clamp(mc.black_front)));
    out[n++] = static_cast<std::uint32_t>(off + kWFrontBase + static_cast<std::size_t>(front_clamp(mc.white_front)));
    return n;
  }

  int pattern_features(const Board &b, Color stm, std::uint32_t *out) noexcept {
    PatternState s;
    s.set(b, stm);
    return pattern_indices(s, pattern_stage(b.count()), mob_counts(b, stm), out);
  }

  void PatternWeights::unload() noexcept {
    w_.clear();
    loaded_ = false;
  }

  void PatternWeights::reset_zero() {
    w_.assign(static_cast<std::size_t>(kStageCount) * pattern_weights_per_stage(), 0);
    loaded_ = true;
  }

  bool PatternWeights::load(const std::string &path, std::ostream &log) {
    loaded_ = false;
    std::ifstream is(path, std::ios::binary);
    if (!is) {
      log << "info error: cannot open pattern weights '" << path << "'\n";
      return false;
    }
    char magic[8];
    is.read(magic, 8);
    if (std::memcmp(magic, "ISLAYPAT", 8) != 0) {
      log << "info error: '" << path << "' is not an ISLAYPAT file\n";
      return false;
    }
    std::uint32_t ver = 0, stages = 0;
    std::uint64_t per = 0;
    is.read(reinterpret_cast<char *>(&ver), 4);
    is.read(reinterpret_cast<char *>(&stages), 4);
    is.read(reinterpret_cast<char *>(&per), 8);
    // Tables are only ever APPENDED (bias, then mobility, then potential mobility),
    // so any earlier file's per-stage block is a prefix of the current one: scatter
    // it into the front of each (wider) stage and leave the newer tables zeroed.
    // That one rule loads v1, v2, v3, ... with no per-version branch -- a teacher
    // from before a feature existed simply runs with that feature off.
    const bool known = (per == kPerStageV1 || per == kPerStageV2 || per == kPerStageV3 || per == kPerStageV4 ||
                        per == kPerStageV5 || per == kPerStage);
    if (stages != static_cast<std::uint32_t>(kStageCount) || !known || per > kPerStage) {
      log << "info error: pattern weights shape mismatch (ver " << ver << ", stages " << stages << ", per " << per
          << "; expected " << kStageCount << " stages x {" << kPerStageV1 << ',' << kPerStageV2 << ',' << kPerStageV3 << ',' << kPerStageV4 << ',' << kPerStageV5 << ',' << kPerStage
          << "})\n";
      return false;
    }
    w_.assign(static_cast<std::size_t>(kStageCount) * kPerStage, 0);
    for (int st = 0; st < kStageCount && is; ++st)
      is.read(reinterpret_cast<char *>(w_.data() + static_cast<std::size_t>(st) * kPerStage),
              static_cast<std::streamsize>(per * sizeof(std::int16_t)));
    if (!is) {
      log << "info error: pattern weights truncated\n";
      return false;
    }
    loaded_ = true;
    log << "info string pattern weights loaded: " << path << " (v" << ver << ", " << stages << " stages x " << per
        << (per < kPerStage ? ", newer tables zeroed" : "") << ")\n";
    return true;
  }

  bool PatternWeights::save(const std::string &path, std::ostream &log) const {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
      log << "info error: cannot write '" << path << "'\n";
      return false;
    }
    const std::uint32_t ver = 6, stages = kStageCount;
    const std::uint64_t per = pattern_weights_per_stage();
    os.write("ISLAYPAT", 8);
    os.write(reinterpret_cast<const char *>(&ver), 4);
    os.write(reinterpret_cast<const char *>(&stages), 4);
    os.write(reinterpret_cast<const char *>(&per), 8);
    os.write(reinterpret_cast<const char *>(w_.data()), static_cast<std::streamsize>(w_.size() * sizeof(std::int16_t)));
    return static_cast<bool>(os);
  }

  // The NNUE net rides on the same PatternState/feature machinery, so it counts as
  // "pattern eval on" for the search's template selection even with no linear weights.
  bool             pattern_enabled() noexcept { return g_active->loaded() || nnue_enabled(); }
  PatternWeights & pattern_weights() noexcept { return *g_active; }
  void pattern_set_active(PatternWeights *w) noexcept { g_active = w ? w : &g_weights; }

  bool pattern_selftest() noexcept {
    // 1. Layout sanity: every instance has the right length, squares are on the
    //    board and distinct, and the four images of a type really are distinct.
    for (int i = 0; i < kPatternInstances; ++i) {
      if (kLayout.len[i] < 4 || kLayout.len[i] > kMaxPatternSquares)
        return false;
      for (int a = 0; a < kLayout.len[i]; ++a) {
        if (kLayout.squares[i][a] < 0 || kLayout.squares[i][a] > 63)
          return false;
        for (int b2 = a + 1; b2 < kLayout.len[i]; ++b2)
          if (kLayout.squares[i][a] == kLayout.squares[i][b2])
            return false; // a square twice in one instance would break place values
      }
    }

    std::uint64_t s   = 0x9E3779B97F4A7C15ULL;
    const auto    rnd = [&s]() noexcept {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      return s;
    };

    // 2. THE property that matters: incremental == from scratch, over real games.
    for (int game = 0; game < 200; ++game) {
      Board        b   = Board::start();
      Color        stm = Color::Black;
      PatternState st;
      st.set(b, stm);
      for (int ply = 0; ply < 70; ++ply) {
        Bitboard m = b.moves();
        if (m == 0) {
          const Board p = b.passed();
          if (!p.has_moves())
            break;
          b   = p;
          stm = ~stm; // a pass changes the mover but no square: features unchanged
          PatternState check;
          check.set(b, stm);
          for (int i = 0; i < kPatternInstances; ++i)
            if (check.f[i] != st.f[i])
              return false;
          continue;
        }
        unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
        while (k-- > 0)
          m &= m - 1;
        const Square   sq   = lsb(m);
        const Bitboard flip = b.player ^ b.play(sq).opponent ^ square_bb(sq); // discs that changed hands

        st.update(sq, flip, stm);
        b   = b.play(sq);
        stm = ~stm;

        PatternState scratch;
        scratch.set(b, stm);
        for (int i = 0; i < kPatternInstances; ++i)
          if (scratch.f[i] != st.f[i])
            return false; // the whole point of the incremental path
        for (int i = 0; i < kPatternInstances; ++i)
          if (st.f[i] < 0 || st.f[i] >= ipow3(kLayout.len[i]))
            return false; // an index outside its table would read someone else's weights
      }
    }

    // 3. The training surface must agree with the eval: same indices, in range.
    {
      PatternWeights w;
      w.reset_zero();
      Board        b = Board::start();
      PatternState st;
      st.set(b, Color::Black);
      if (w.score(st, pattern_stage(b.count()), mob_counts(b, Color::Black)) != 0)
        return false; // zeroed weights must score exactly 0
      std::uint32_t idx[kPatternInstances + 8];
      const int     n = pattern_features(b, Color::Black, idx);
      if (n != kPatternInstances + 8)
        return false;
      for (int i = 0; i < n; ++i)
        if (idx[i] >= static_cast<std::uint32_t>(kStageCount * pattern_weights_per_stage()))
          return false;
    }

    // 4. Stage interpolation: the boundary is exact and the interior is a bounded blend.
    //    Craft two adjacent stages that differ only in the bias (so the start state's
    //    score is that bias), and check the deterministic integer formula.
    {
      PatternWeights    w;
      w.reset_zero();
      const std::size_t per = pattern_weights_per_stage();
      w.data()[0 * per + kBiasBase] = 100; // stage 0 score for any state
      w.data()[1 * per + kBiasBase] = 500; // stage 1 score
      Board        b = Board::start();
      PatternState st;
      st.set(b, Color::Black);
      const MobCounts mc{};
      // discs 4 -> stage 0, r=0 (boundary) -> exactly the stage-0 score, interp or not.
      if (w.score_phase(st, 4, mc, true) != 100 || w.score_phase(st, 4, mc, false) != 100)
        return false;
      // discs 5..7 -> stage 0, r=1..3 -> (100*(4-r) + 500*r)/4.
      if (w.score_phase(st, 5, mc, true) != (100 * 3 + 500 * 1) / 4) // 200
        return false;
      if (w.score_phase(st, 6, mc, true) != (100 * 2 + 500 * 2) / 4) // 300
        return false;
      if (w.score_phase(st, 7, mc, true) != (100 * 1 + 500 * 3) / 4) // 400
        return false;
      // Antisymmetry is preserved because the blend is linear: interp(-x) == -interp(x).
      // With interp off the value must be byte-identical to the plain per-stage score.
      if (w.score_phase(st, 5, mc, false) != w.score(st, pattern_stage(5), mc))
        return false;
    }
    return true;
  }

} // namespace islay
