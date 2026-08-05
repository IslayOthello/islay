#include "train.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "pattern.hpp"
#include "search.hpp"

namespace islay {
  namespace {

    struct Rng {
      std::uint64_t s;
      std::uint64_t next() noexcept {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
      }
    };

    constexpr std::uint8_t kPassByte = 64;
    struct Game {
      std::uint8_t moves[64];
      std::uint8_t n      = 0;
      std::int16_t result = 0; // final disc difference, centi-discs, BLACK's point of view
    };

    constexpr std::int16_t kNoTeacherScore = std::numeric_limits<std::int16_t>::min();
    struct DistillGame {
      Game         game;
      std::int16_t teacher[64]; // deep-search score before moves[i], BLACK's point of view
    };

    [[nodiscard]] bool play_one(Game &g, Searcher &s, Rng &rng, const TrainConfig &cfg,
                                std::int16_t *teacher = nullptr) {
      Board              b   = Board::start();
      Color              stm = Color::Black;
      std::ostringstream sink; // the engine's info lines are noise here
      g.n = 0;
      if (teacher)
        std::fill(teacher, teacher + 64, kNoTeacherScore);

      // Diversify deterministic self-play.
      for (int i = 0; i < cfg.opening_plies; ++i) {
        Bitboard m = b.moves();
        if (m == 0) {
          const Board p = b.passed();
          if (cfg.rule != Rule::Othello || !p.has_moves())
            return false;
          b = p;
          stm = ~stm;
          if (g.n < 64)
            g.moves[g.n++] = kPassByte;
          continue;
        }
        unsigned k = static_cast<unsigned>(rng.next() % static_cast<unsigned>(popcount(m)));
        while (k-- > 0)
          m &= m - 1;
        const Square sq = lsb(m);
        b   = b.play(sq);
        stm = ~stm;
        if (g.n < 64)
          g.moves[g.n++] = static_cast<std::uint8_t>(sq);
      }

      for (int ply = 0; ply < 80 && g.n < 64; ++ply) {
        if (b.moves() == 0) {
          const Board p = b.passed();
          if (cfg.rule != Rule::Othello || !p.has_moves())
            break; // game over
          b   = p;
          stm = ~stm;
          g.moves[g.n++] = kPassByte;
          continue;
        }
        // depth >= empties gives exact endgame labels because passes cost no depth.
        const int          empties = 64 - b.count();
        const int          d       = empties <= cfg.solve_empties ? empties : cfg.depth;
        const SearchLimits lim{d, 0, 0.0};
        const SearchResult r = s.search(b, lim, cfg.rule, stm, sink);
        if (r.best == NOMOVE)
          break;
        if (r.best == PASS) {
          b   = b.passed();
          stm = ~stm;
          g.moves[g.n++] = kPassByte;
          continue;
        }
        if (teacher)
          teacher[g.n] = static_cast<std::int16_t>((stm == Color::Black) ? r.score : -r.score);
        b   = b.play(r.best);
        stm = ~stm;
        g.moves[g.n++] = static_cast<std::uint8_t>(r.best);
      }

      const int stm_score = terminal_score(b);
      const int black     = (stm == Color::Black) ? stm_score : -stm_score;
      g.result            = static_cast<std::int16_t>(std::clamp(black, -kScoreMax, kScoreMax));
      return g.n > cfg.opening_plies;
    }

  } // namespace

  TrainResult run_train(const TrainConfig &cfg, std::ostream &log) {
    TrainResult res;
    const std::size_t per   = pattern_weights_per_stage();
    const std::size_t total = static_cast<std::size_t>(kStageCount) * per;

    log << "info string train: " << cfg.games << " games @ depth " << cfg.depth << ", " << cfg.epochs << " epochs, lr "
        << cfg.lr << ", l2 " << cfg.l2 << ", seed " << (cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL)
        << ", " << total << " weights -> " << cfg.out << '\n';
    log.flush();

    pattern_set_stage_interp(cfg.interp);

    Rng               rng{cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL};
    Searcher          s(32);
    std::vector<Game> games;
    games.reserve(static_cast<std::size_t>(cfg.games));
    for (int i = 0; i < cfg.games; ++i) {
      Game g;
      s.clear(); // nothing carries between games
      if (play_one(g, s, rng, cfg))
        games.push_back(g);
      if ((i + 1) % 5000 == 0) {
        log << "info string train: generated " << games.size() << "/" << (i + 1) << " games\n";
        log.flush();
      }
    }
    if (games.empty()) {
      log << "info string train: no games generated\n";
      return res;
    }
    res.games = games.size();

    std::vector<float> w(total, 0.0f);
    const float        lr = static_cast<float>(cfg.lr);
    const float        l2 = static_cast<float>(cfg.l2);

    // Split by game to avoid leaking a game's shared label.
    for (std::size_t i = games.size(); i > 1; --i) {
      const std::size_t j = static_cast<std::size_t>(rng.next() % i);
      std::swap(games[i - 1], games[j]);
    }
    const std::size_t nval = static_cast<std::size_t>(static_cast<double>(games.size()) * cfg.val_frac);
    const std::size_t ntrain = games.size() - nval;
    if (ntrain == 0) {
      log << "info string train: validation fraction leaves no training games\n";
      return res;
    }

    const auto pass = [&](std::size_t begin, std::size_t end, bool update, std::uint64_t *out_count) {
      std::uint32_t idx[kPatternInstances + 8];
      std::uint32_t idxhi[kPatternInstances + 8];
      double        sse   = 0.0;
      std::uint64_t count = 0;
      for (std::size_t gi = begin; gi < end; ++gi) {
        const Game  &g   = games[gi];
        Board        b   = Board::start();
        Color        stm = Color::Black;
        PatternState st;
        st.set(b, stm);

        for (int m = 0; m < g.n; ++m) {
          const MobCounts mc    = mob_counts(b, stm);
          const int       discs = b.count();
          const int       stage = pattern_stage(discs);
          const int       n     = pattern_indices(st, stage, mc, idx);
          // Feature gates rely on the fixed prefix layout.
          const int kType    = kPatternInstances - kC2x5Instances;                          // 38
          const int c2x5_lo  = kType, c2x5_hi = kPatternInstances;                          // [38,46)
          const int nfront   = n - 2;                                                        // last 2
          const int npar     = n - 3;                                                        // the 1 before them
          const int nstab    = n - 5;                                                        // the 2 before it
          const int nmob     = n - 7;                                                        // the 2 before those
          const auto gated = [&](int k) {
            if (!cfg.use_front && k >= nfront) return true;
            if (!cfg.use_par && k >= npar && k < nfront) return true;
            if (!cfg.use_stab && k >= nstab && k < npar) return true;
            if (!cfg.use_mobility && k >= nmob && k < nstab) return true;
            if (!cfg.use_c2x5 && k >= c2x5_lo && k < c2x5_hi) return true;
            return false;
          };

          // Split prediction and gradient across adjacent stages.
          float         clo = 1.0f, chi = 0.0f;
          int           nhi = 0;
          if (cfg.interp) {
            const int r = discs >= 4 ? (discs - 4) % 4 : 0;
            if (r > 0 && stage < kStageCount - 1) {
              chi = static_cast<float>(r) / 4.0f;
              clo = 1.0f - chi;
              nhi = pattern_indices(st, stage + 1, mc, idxhi);
            }
          }

          float plo = 0.0f, phi = 0.0f;
          for (int k = 0; k < n; ++k)
            if (!gated(k))
              plo += w[idx[k]];
          for (int k = 0; k < nhi; ++k)
            if (!gated(k))
              phi += w[idxhi[k]];
          const float pred = clo * plo + chi * phi;

          const float err = pred - static_cast<float>(g.result);
          sse += static_cast<double>(err) * err;
          ++count;
          if (update) {
            for (int k = 0; k < n; ++k)
              if (!gated(k))
                w[idx[k]] -= lr * (err * clo + l2 * w[idx[k]]);
            for (int k = 0; k < nhi; ++k)
              if (!gated(k))
                w[idxhi[k]] -= lr * (err * chi + l2 * w[idxhi[k]]);
          }

          const std::uint8_t mv = g.moves[m];
          if (mv == kPassByte) {
            b   = b.passed(); // a pass changes the mover but no square: state unchanged
            stm = ~stm;
            continue;
          }
          const Square   sq      = static_cast<Square>(mv);
          const Board    child   = b.play(sq);
          const Bitboard flipped = b.player ^ child.opponent ^ square_bb(sq);
          st.update(sq, flipped, stm);
          b   = child;
          stm = ~stm;
        }
      }
      if (out_count)
        *out_count = count;
      return count ? std::sqrt(sse / static_cast<double>(count)) : 0.0;
    };

    std::vector<float> w_best;
    double             best_val = 1e30;
    int                best_ep  = 0;

    for (int ep = 0; ep < cfg.epochs; ++ep) {
      for (std::size_t i = games.size(); i > nval + 1; --i) {
        const std::size_t j = nval + static_cast<std::size_t>(rng.next() % (i - nval));
        std::swap(games[i - 1], games[j]);
      }

      std::uint64_t count = 0;
      res.rmse            = pass(nval, games.size(), true, &count);
      res.positions       = count;
      res.val_rmse        = nval ? pass(0, nval, false, nullptr) : 0.0;

      const bool improved = nval && res.val_rmse < best_val;
      if (improved || nval == 0) {
        best_val = res.val_rmse;
        best_ep  = ep + 1;
        w_best   = w; // snapshot the trough
      }
      log << "info string train: epoch " << (ep + 1) << "/" << cfg.epochs << "  positions " << count << "  rmse "
          << static_cast<int>(res.rmse) << "  val_rmse " << static_cast<int>(res.val_rmse)
          << (improved ? " cd *" : " cd") << "\n";
      log.flush();
    }
    if (nval && !w_best.empty()) {
      w = w_best;
      res.val_rmse = best_val;
      log << "info string train: keeping epoch " << best_ep << " (best val_rmse " << static_cast<int>(best_val)
          << " cd)\n";
    }

    PatternWeights out;
    out.reset_zero();
    if (out.size() != total) {
      log << "info string train: weight size mismatch\n";
      return res;
    }
    std::int16_t *dst = out.data();
    for (std::size_t i = 0; i < total; ++i)
      dst[i] = static_cast<std::int16_t>(std::lround(std::clamp(w[i], -32000.0f, 32000.0f)));
    if (!out.save(cfg.out, log))
      return res;

    log << "train done: " << res.games << " games, " << res.positions << " positions/epoch, rmse "
        << static_cast<int>(res.rmse) << " cd -> " << cfg.out << '\n'
        << "  NOTE: rmse is training-set fit, not strength. Settle it with:\n"
        << "        match 100 4 " << cfg.out << " -\n";
    log.flush();
    pattern_set_stage_interp(false);
    res.ok = true;
    return res;
  }

  TrainResult run_ntrain(const NTrainConfig &cfg, std::ostream &log) {
    TrainResult res;
    const bool nnue_teacher = nnue_enabled();
    if (!nnue_teacher && !pattern_weights().loaded()) {
      log << "info error: ntrain needs a .pat or .nnue EvalFile -- it is the teacher AND the warm start\n";
      return res;
    }
    constexpr int     H   = kNnueHidden;
    const std::size_t per = pattern_weights_per_stage();
    const unsigned workers = static_cast<unsigned>(std::clamp(cfg.workers, 1, 4));

    log << "info string ntrain: " << cfg.games << " games @ teacher depth " << cfg.depth << ", " << cfg.epochs
        << " epochs, lr_emb " << cfg.lr_emb << ", lr_out " << cfg.lr_out << ", H " << H << " x " << kStageCount
        << " stages, head " << (cfg.grouped ? "grouped-interaction NN4" : "legacy NN3") << ", " << workers
        << " generation workers -> " << cfg.out << '\n';
    log.flush();

    // Each generator owns its Searcher and TT.
    TrainConfig gen; // reuse play_one's contract for the game loop
    gen.depth         = cfg.depth;
    gen.opening_plies = cfg.opening_plies;
    gen.solve_empties = cfg.solve_empties;
    gen.rule          = cfg.rule;

    const std::uint64_t seed0 = cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL;
    unsigned            nthreads = workers;
    if (static_cast<int>(nthreads) > cfg.games)
      nthreads = static_cast<unsigned>(cfg.games);

    std::vector<std::vector<DistillGame>> parts(nthreads);
    std::atomic<int>               done{0};
    const auto worker = [&](unsigned t) {
      Rng      rng{seed0 + static_cast<std::uint64_t>(t) * 0x9E3779B97F4A7C15ULL};
      Searcher s(32);
      const int lo = static_cast<int>(static_cast<std::uint64_t>(cfg.games) * t / nthreads);
      const int hi = static_cast<int>(static_cast<std::uint64_t>(cfg.games) * (t + 1) / nthreads);
      parts[t].reserve(static_cast<std::size_t>(hi - lo));
      for (int i = lo; i < hi; ++i) {
        DistillGame g;
        s.clear();
        if (play_one(g.game, s, rng, gen, g.teacher))
          parts[t].push_back(g);
        const int n = done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (t == 0 && (n % 5000) == 0) {
          log << "info string ntrain: generated ~" << n << "/" << cfg.games << " games\n";
          log.flush();
        }
      }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < nthreads; ++t)
      pool.emplace_back(worker, t);
    worker(0);
    for (auto &th : pool)
      th.join();

    std::vector<DistillGame> games;
    std::size_t       total = 0;
    for (auto &p : parts)
      total += p.size();
    games.reserve(total);
    for (auto &p : parts) {
      games.insert(games.end(), p.begin(), p.end());
      p = {};
    }
    Rng rng{seed0}; // drives the fit's shuffles (single-threaded, deterministic)
    if (games.empty()) {
      log << "info string ntrain: no games generated\n";
      return res;
    }
    res.games = games.size();

    // Train in disc units to keep layer gradients comparable.
    NnueNet &net = nnue_net();
    if (nnue_teacher)
      net.prepare_training();
    else
      net.reset();
    net.set_grouped(cfg.grouped);
    float *emb = net.emb(), *w2a = net.w2a(), *w2h = net.w2h(), *b2 = net.b2();
    float *wgroup = net.wgroup(), *winter = net.winter();
    const std::size_t   fper = per + kNnueRFeat; // net feature space: per_stage + r rows
    if (nnue_teacher) {
      log << "info string ntrain: warm-starting from loaded NNUE teacher\n";
    } else {
      (void) b2; // starts 0; the bias feature row carries the linear eval's bias
      const auto frand = [&](float a) { // uniform in [-a, a]
        return (static_cast<float>(rng.next() >> 40) / 16777216.0f * 2.0f - 1.0f) * a;
      };
      // Embed the linear teacher in dimension zero.
      const std::int16_t *linear = pattern_weights().data();
      for (int st = 0; st < kStageCount; ++st)
        for (std::size_t f = 0; f < fper; ++f) {
          float *row = emb + (static_cast<std::size_t>(st) * fper + f) * H;
          row[0]     = f < per ? static_cast<float>(linear[static_cast<std::size_t>(st) * per + f]) / 100.0f : 0.0f;
          for (int j = 1; j < H; ++j)
            row[j] = frand(0.05f);
        }
      for (int st = 0; st < kStageCount; ++st) {
        w2a[static_cast<std::size_t>(st) * H + 0] = 1.0f;
        for (int j = 0; j < H; ++j)
          w2h[static_cast<std::size_t>(st) * H + j] = frand(0.05f);
      }
    }

    for (std::size_t i = games.size(); i > 1; --i) {
      const std::size_t j = static_cast<std::size_t>(rng.next() % i);
      std::swap(games[i - 1], games[j]);
    }
    const std::size_t nval   = static_cast<std::size_t>(static_cast<double>(games.size()) * cfg.val_frac);
    const std::size_t ntrain = games.size() - nval;
    if (ntrain == 0) {
      log << "info string ntrain: validation fraction leaves no training games\n";
      return res;
    }
    const float lre = static_cast<float>(cfg.lr_emb), lro = static_cast<float>(cfg.lr_out);

    const auto pass = [&](std::size_t begin, std::size_t end, bool update, std::uint64_t *out_count) {
      std::uint32_t idx[kPatternInstances + 9]; // + the appended r index
      double        sse   = 0.0;
      std::uint64_t count = 0;
      for (std::size_t gi = begin; gi < end; ++gi) {
        const DistillGame &dg  = games[gi];
        const Game        &g   = dg.game;
        Board              b   = Board::start();
        Color              stm = Color::Black;
        PatternState st;
        st.set(b, stm);

        for (int m = 0; m < g.n; ++m) {
          if (dg.teacher[m] != kNoTeacherScore) {
            const MobCounts mc    = mob_counts(b, stm);
            const int       discs = b.count();
            const int       stage = pattern_stage(discs);
            int             n     = pattern_indices(st, 0, mc, idx); // FLAT (stage-0) ids
            const int       r     = discs >= 4 ? (discs - 4) % 4 : 0;
            idx[n++]              = static_cast<std::uint32_t>(per + r); // the interp input
            float          *base  = emb + static_cast<std::size_t>(stage) * fper * H;
            float          *a     = w2a + static_cast<std::size_t>(stage) * H;
            float          *h     = w2h + static_cast<std::size_t>(stage) * H;
            float          *wg    = cfg.grouped ? wgroup + static_cast<std::size_t>(stage) * kNnueGroups * H : nullptr;
            float          *wi    = cfg.grouped ? winter + static_cast<std::size_t>(stage) * kNnuePairs * H : nullptr;

            float gacc[kNnueGroups][H] = {};
            for (int k = 0; k < n; ++k) {
              const float *row = base + static_cast<std::size_t>(idx[k]) * H;
              const int    group = nnue_feature_group(k);
              for (int j = 0; j < H; ++j)
                gacc[group][j] += row[j];
            }
            float acc[H] = {};
            float pred = net.b2()[stage];
            for (int j = 0; j < H; ++j) {
              for (int group = 0; group < kNnueGroups; ++group)
                acc[j] += gacc[group][j];
              pred += a[j] * acc[j] + (acc[j] > 0.0f ? h[j] * acc[j] : 0.0f);
              if (cfg.grouped) {
                for (int group = 0; group < kNnueGroups; ++group)
                  pred += wg[static_cast<std::size_t>(group) * H + j] * std::max(gacc[group][j], 0.0f);
                for (int group = 0; group < kNnueGroups; ++group)
                  for (int other = group + 1; other < kNnueGroups; ++other) {
                    const int pair = nnue_pair_index(group, other);
                    pred += wi[static_cast<std::size_t>(pair) * H + j] * std::max(gacc[group][j], 0.0f) *
                            std::max(gacc[other][j], 0.0f) / kNnueInteractionScale;
                  }
              }
            }

            const float label = static_cast<float>(dg.teacher[m]) / 100.0f; // discs, Black POV
            const float err   = pred - label;
            sse += static_cast<double>(err) * err;
            ++count;

            if (update) {
              float dgacc[kNnueGroups][H]; // d(loss)/d(group acc), before any head steps
              for (int j = 0; j < H; ++j) {
                const float common = a[j] + (acc[j] > 0.0f ? h[j] : 0.0f);
                for (int group = 0; group < kNnueGroups; ++group) {
                  float residual = 0.0f;
                  if (cfg.grouped && gacc[group][j] > 0.0f) {
                    residual += wg[static_cast<std::size_t>(group) * H + j];
                    for (int other = 0; other < kNnueGroups; ++other) {
                      if (other == group)
                        continue;
                      const int pair = nnue_pair_index(group, other);
                      residual += wi[static_cast<std::size_t>(pair) * H + j] * std::max(gacc[other][j], 0.0f) /
                                  kNnueInteractionScale;
                    }
                  }
                  dgacc[group][j] = err * (common + residual);
                }
                a[j] -= lro * err * acc[j];
                h[j] -= lro * err * (acc[j] > 0.0f ? acc[j] : 0.0f);
                if (cfg.grouped) {
                  for (int group = 0; group < kNnueGroups; ++group)
                    wg[static_cast<std::size_t>(group) * H + j] -=
                        lro * err * std::max(gacc[group][j], 0.0f);
                  for (int group = 0; group < kNnueGroups; ++group)
                    for (int other = group + 1; other < kNnueGroups; ++other) {
                      const int pair = nnue_pair_index(group, other);
                      wi[static_cast<std::size_t>(pair) * H + j] -=
                          lro * err * std::max(gacc[group][j], 0.0f) * std::max(gacc[other][j], 0.0f) /
                          kNnueInteractionScale;
                    }
                }
              }
              net.b2()[stage] -= lro * err;
              for (int k = 0; k < n; ++k) {
                float *row = base + static_cast<std::size_t>(idx[k]) * H;
                const int group = nnue_feature_group(k);
                for (int j = 0; j < H; ++j)
                  row[j] -= lre * dgacc[group][j];
              }
            }
          }

          const std::uint8_t mv = g.moves[m];
          if (mv == kPassByte) {
            b   = b.passed();
            stm = ~stm;
            continue;
          }
          const Square   sq      = static_cast<Square>(mv);
          const Board    child   = b.play(sq);
          const Bitboard flipped = b.player ^ child.opponent ^ square_bb(sq);
          st.update(sq, flipped, stm);
          b   = child;
          stm = ~stm;
        }
      }
      if (out_count)
        *out_count = count;
      return count ? std::sqrt(sse / static_cast<double>(count)) * 100.0 : 0.0; // cd
    };

    // Keep epoch zero eligible when bootstrapping an NNUE teacher.
    std::vector<float> best_emb, best_a, best_h, best_group, best_inter, best_b;
    const std::size_t  esz = static_cast<std::size_t>(kStageCount) * fper * H;
    const double       warm_val = nval ? pass(0, nval, false, nullptr) : 0.0;
    double             best_val = nnue_teacher && nval ? warm_val : 1e30;
    int                best_ep  = 0;
    if (nnue_teacher && nval) {
      best_emb.assign(emb, emb + esz);
      best_a.assign(w2a, w2a + static_cast<std::size_t>(kStageCount) * H);
      best_h.assign(w2h, w2h + static_cast<std::size_t>(kStageCount) * H);
      if (cfg.grouped) {
        best_group.assign(wgroup, wgroup + static_cast<std::size_t>(kStageCount) * kNnueGroups * H);
        best_inter.assign(winter, winter + static_cast<std::size_t>(kStageCount) * kNnuePairs * H);
      }
      best_b.assign(net.b2(), net.b2() + kStageCount);
    }
    log << "info string ntrain: warm-start val_rmse " << static_cast<int>(warm_val) << " cd\n";
    log.flush();

    for (int ep = 0; ep < cfg.epochs; ++ep) {
      for (std::size_t i = games.size(); i > nval + 1; --i) {
        const std::size_t j = nval + static_cast<std::size_t>(rng.next() % (i - nval));
        std::swap(games[i - 1], games[j]);
      }
      std::uint64_t count = 0;
      res.rmse            = pass(nval, games.size(), true, &count);
      res.positions       = count;
      res.val_rmse        = nval ? pass(0, nval, false, nullptr) : 0.0;

      const bool improved = nval && res.val_rmse < best_val;
      if (improved || nval == 0) {
        best_val = res.val_rmse;
        best_ep  = ep + 1;
        best_emb.assign(emb, emb + esz);
        best_a.assign(w2a, w2a + static_cast<std::size_t>(kStageCount) * H);
        best_h.assign(w2h, w2h + static_cast<std::size_t>(kStageCount) * H);
        if (cfg.grouped) {
          best_group.assign(wgroup, wgroup + static_cast<std::size_t>(kStageCount) * kNnueGroups * H);
          best_inter.assign(winter, winter + static_cast<std::size_t>(kStageCount) * kNnuePairs * H);
        }
        best_b.assign(net.b2(), net.b2() + kStageCount);
      }
      log << "info string ntrain: epoch " << (ep + 1) << "/" << cfg.epochs << "  positions " << count << "  rmse "
          << static_cast<int>(res.rmse) << "  val_rmse " << static_cast<int>(res.val_rmse)
          << (improved ? " cd *" : " cd") << "\n";
      log.flush();
    }
    if (nval && !best_emb.empty()) {
      std::copy(best_emb.begin(), best_emb.end(), emb);
      std::copy(best_a.begin(), best_a.end(), w2a);
      std::copy(best_h.begin(), best_h.end(), w2h);
      if (cfg.grouped) {
        std::copy(best_group.begin(), best_group.end(), wgroup);
        std::copy(best_inter.begin(), best_inter.end(), winter);
      }
      std::copy(best_b.begin(), best_b.end(), net.b2());
      res.val_rmse = best_val;
      log << "info string ntrain: keeping epoch " << best_ep << " (best val_rmse " << static_cast<int>(best_val)
          << " cd)\n";
    }

    if (!net.save(cfg.out, log))
      return res;
    net.quantize(); // the float table just went to disk; inference wants the int16 one
    log << "ntrain done: " << res.games << " games, " << res.positions << " positions/epoch -> " << cfg.out << '\n'
        << "  NOTE: strength is settled by match, never by rmse. A/B via fastothello:\n"
        << "        EvalFile " << cfg.out << "  vs  the loaded teacher EvalFile\n";
    log.flush();
    res.ok = true;
    return res;
  }

} // namespace islay
