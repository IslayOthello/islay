/**
 * @file train.cpp
 * @brief Self-play weight fitting (contract and rationale: train.hpp).
 */
#include "train.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
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

    /**
     * One finished game: the moves, and what it came to. 68 bytes, which is what
     * lets a whole run's worth of games sit in memory and be replayed per epoch.
     * PASS is stored as 64 so a byte still covers every case.
     */
    constexpr std::uint8_t kPassByte = 64;
    struct Game {
      std::uint8_t moves[64];
      std::uint8_t n      = 0;
      std::int16_t result = 0; // final disc difference, centi-discs, BLACK's point of view
    };

    /** Random legal opening, then the engine plays it out. Returns false if it died. */
    [[nodiscard]] bool play_one(Game &g, Searcher &s, Rng &rng, const TrainConfig &cfg) {
      Board              b   = Board::start();
      Color              stm = Color::Black;
      std::ostringstream sink; // the engine's info lines are noise here
      g.n = 0;

      // Random opening: both engines are deterministic, so without this every game
      // would be the same game and the whole run would fit one line.
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
        // Near the end, ask for depth >= empties: a pass costs no depth, so that
        // makes every leaf terminal and the answer the TRUE game value rather than a
        // guess (search.hpp). This is what makes the label ground truth instead of a
        // record of two weak players fumbling the endgame -- and the solves get
        // geometrically cheaper as the board fills.
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
        b   = b.play(r.best);
        stm = ~stm;
        g.moves[g.n++] = static_cast<std::uint8_t>(r.best);
      }

      // terminal_score already settles the empties-to-the-winner rule; it is
      // mover-relative, so one negation puts it in Black's view -- the same view the
      // weights are written in.
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

    // With interp, the teacher self-plays with it too (it is +58 Elo, so the games --
    // and thus the labels -- are stronger), and the fit below blends two stages.
    pattern_set_stage_interp(cfg.interp);

    // --- phase 1: generate ----------------------------------------------------
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

    // --- phase 2: fit ---------------------------------------------------------
    // float, not int16: int16 cannot hold a gradient step. Quantisation happens once,
    // at the end, when the weights are written.
    std::vector<float> w(total, 0.0f);
    const float        lr = static_cast<float>(cfg.lr);
    const float        l2 = static_cast<float>(cfg.l2);

    // Hold out a validation set, SPLIT BY GAME. Splitting by position would leak:
    // every position in a game carries that game's single label, so a held-out
    // position whose siblings are in the training set is not held out at all -- the
    // model would be scored on a label it had already been fitted to.
    //
    // Shuffle once, fix the split, and only ever shuffle the training part after
    // that, so the validation games stay untouched for the whole run.
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

    /** One pass over [begin,end). `update` off = pure measurement, weights untouched. */
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
          // Score THIS position, then advance. Pattern features come from the
          // incremental state; mobility cannot be incremental, so it is computed from
          // the reconstructed board here -- the same two movegen calls the leaf pays.
          const MobCounts mc    = mob_counts(b, stm);
          const int       discs = b.count();
          const int       stage = pattern_stage(discs);
          const int       n     = pattern_indices(st, stage, mc, idx);
          // Emitted layout: [38 patterns | 8 Corner2x5 | bias | bmob | wmob]. Gating a
          // group skips its updates, leaving those weights 0 -- provably-off for an
          // isolated A/B (eval sums them all; a 0 weight adds nothing). Ranges from the
          // instance layout, not from `n`, so appended patterns never shift the mob gate.
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

          // STAGE INTERPOLATION training: the eval blends this stage with the next by
          // r/4, so fit BOTH -- the prediction is clo*sum(stage) + chi*sum(stage+1), and
          // the gradient splits by the same coefficients so each stage learns its share.
          // With interp off (or r==0 / final stage) chi is 0 and this is the plain fit.
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

    // Keep the weights from the epoch with the best held-out fit, not the last one.
    // val_rmse is U-shaped -- it bottoms out and then climbs as the fit starts
    // memorising -- so the final epoch is usually PAST the best point. This is early
    // stopping without committing to a stopping rule: train the full budget, ship the
    // trough. (It is not monotone, so "stop on first rise" would quit on noise.)
    std::vector<float> w_best;
    double             best_val = 1e30;
    int                best_ep  = 0;

    for (int ep = 0; ep < cfg.epochs; ++ep) {
      // Shuffle the TRAINING games only: consecutive positions share a label, so a
      // fixed order lets each game's tail drag the weights.
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
    // Restore the trough for writing. With no validation set there is nothing to
    // choose, so w is left as the last epoch.
    if (nval && !w_best.empty()) {
      w = w_best;
      res.val_rmse = best_val;
      log << "info string train: keeping epoch " << best_ep << " (best val_rmse " << static_cast<int>(best_val)
          << " cd)\n";
    }

    // --- phase 3: write -------------------------------------------------------
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
    if (!pattern_weights().loaded()) {
      log << "info error: ntrain needs pattern weights loaded (EvalFile) -- they are the teacher AND the warm start\n";
      return res;
    }
    constexpr int     H   = kNnueHidden;
    const std::size_t per = pattern_weights_per_stage();

    log << "info string ntrain: " << cfg.games << " games @ depth " << cfg.depth << ", " << cfg.epochs
        << " epochs, lr_emb " << cfg.lr_emb << ", lr_out " << cfg.lr_out << ", H " << H << " x " << kStageCount
        << " stages -> " << cfg.out << '\n';
    log.flush();

    // --- phase 1: generate (the teacher is the loaded linear eval) --------------
    // Games are independent, so generation fans out across threads: each worker has
    // its own Searcher, TT and Rng, and the shared eval weights are read-only during
    // a search. The fit that follows stays single-threaded (it is not the bottleneck).
    TrainConfig gen; // reuse play_one's contract for the game loop
    gen.depth         = cfg.depth;
    gen.opening_plies = cfg.opening_plies;
    gen.solve_empties = cfg.solve_empties;
    gen.rule          = cfg.rule;

    const std::uint64_t seed0 = cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL;
    unsigned            nthreads = std::thread::hardware_concurrency();
    if (nthreads < 1)
      nthreads = 1;
    if (static_cast<int>(nthreads) > cfg.games)
      nthreads = static_cast<unsigned>(cfg.games);

    std::vector<std::vector<Game>> parts(nthreads);
    std::atomic<int>               done{0};
    const auto worker = [&](unsigned t) {
      Rng      rng{seed0 + static_cast<std::uint64_t>(t) * 0x9E3779B97F4A7C15ULL};
      Searcher s(32);
      const int lo = static_cast<int>(static_cast<std::uint64_t>(cfg.games) * t / nthreads);
      const int hi = static_cast<int>(static_cast<std::uint64_t>(cfg.games) * (t + 1) / nthreads);
      parts[t].reserve(static_cast<std::size_t>(hi - lo));
      for (int i = lo; i < hi; ++i) {
        Game g;
        s.clear();
        if (play_one(g, s, rng, gen))
          parts[t].push_back(g);
        const int n = done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (t == 0 && (n % 20000) == 0) {
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

    std::vector<Game> games;
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

    // --- phase 2: init ----------------------------------------------------------
    // Train in DISCS (labels /100): cd-scale activations would give ~1e5 head
    // gradients and no single lr would fit both layers.
    NnueNet &net = nnue_net();
    net.reset();
    float *emb = net.emb(), *w2a = net.w2a(), *w2h = net.w2h(), *b2 = net.b2();
    (void) b2; // starts 0; the bias feature row already carries the linear eval's bias

    const auto frand = [&](float a) { // uniform in [-a, a]
      return (static_cast<float>(rng.next() >> 40) / 16777216.0f * 2.0f - 1.0f) * a;
    };
    // Warm start = the teacher: emb dim 0 = the v18 weight for that stage, W2a dim 0 = 1,
    // everything else small noise. r rows start at 0 (v18 has no interp term to copy).
    const std::size_t   fper = per + kNnueRFeat; // net feature space: per_stage + r rows
    const std::int16_t *w18  = pattern_weights().data();
    for (int st = 0; st < kStageCount; ++st)
      for (std::size_t f = 0; f < fper; ++f) {
        float *row = emb + (static_cast<std::size_t>(st) * fper + f) * H;
        row[0]     = f < per ? static_cast<float>(w18[static_cast<std::size_t>(st) * per + f]) / 100.0f : 0.0f;
        for (int j = 1; j < H; ++j)
          row[j] = frand(0.05f);
      }
    for (int st = 0; st < kStageCount; ++st) {
      w2a[static_cast<std::size_t>(st) * H + 0] = 1.0f;
      for (int j = 0; j < H; ++j)
        w2h[static_cast<std::size_t>(st) * H + j] = frand(0.05f);
    }

    // --- phase 3: fit -----------------------------------------------------------
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

    /** One pass over [begin,end); backprop when `update`. Same replay skeleton as
     *  the linear pass above, different model. RMSE is reported in centi-discs. */
    const auto pass = [&](std::size_t begin, std::size_t end, bool update, std::uint64_t *out_count) {
      std::uint32_t idx[kPatternInstances + 9]; // + the appended r index
      double        sse   = 0.0;
      std::uint64_t count = 0;
      for (std::size_t gi = begin; gi < end; ++gi) {
        const Game  &g   = games[gi];
        Board        b   = Board::start();
        Color        stm = Color::Black;
        PatternState st;
        st.set(b, stm);
        const float label = static_cast<float>(g.result) / 100.0f; // discs

        for (int m = 0; m < g.n; ++m) {
          const MobCounts mc    = mob_counts(b, stm);
          const int       discs = b.count();
          const int       stage = pattern_stage(discs);
          int             n     = pattern_indices(st, 0, mc, idx); // FLAT (stage-0) ids
          const int       r     = discs >= 4 ? (discs - 4) % 4 : 0;
          idx[n++]              = static_cast<std::uint32_t>(per + r); // the interp input
          float          *base  = emb + static_cast<std::size_t>(stage) * fper * H;
          float          *a     = w2a + static_cast<std::size_t>(stage) * H;
          float          *h     = w2h + static_cast<std::size_t>(stage) * H;

          float acc[H] = {};
          for (int k = 0; k < n; ++k) {
            const float *row = base + static_cast<std::size_t>(idx[k]) * H;
            for (int j = 0; j < H; ++j)
              acc[j] += row[j];
          }
          float pred = net.b2()[stage];
          for (int j = 0; j < H; ++j)
            pred += a[j] * acc[j] + (acc[j] > 0.0f ? h[j] * acc[j] : 0.0f);

          const float err = pred - label;
          sse += static_cast<double>(err) * err;
          ++count;

          if (update) {
            float dacc[H]; // d(loss)/d(acc), using the heads BEFORE they step
            for (int j = 0; j < H; ++j) {
              dacc[j] = err * (a[j] + (acc[j] > 0.0f ? h[j] : 0.0f));
              a[j] -= lro * err * acc[j];
              h[j] -= lro * err * (acc[j] > 0.0f ? acc[j] : 0.0f);
            }
            net.b2()[stage] -= lro * err;
            for (int k = 0; k < n; ++k) {
              float *row = base + static_cast<std::size_t>(idx[k]) * H;
              for (int j = 0; j < H; ++j)
                row[j] -= lre * dacc[j];
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

    // Sanity: the warm start should already fit near the teacher before any step.
    log << "info string ntrain: warm-start val_rmse " << static_cast<int>(nval ? pass(0, nval, false, nullptr) : 0.0)
        << " cd\n";
    log.flush();

    // Early stop by snapshot (like the linear trainer): train the full budget, keep
    // the best-held-out epoch.
    std::vector<float> best_emb, best_a, best_h, best_b;
    double             best_val = 1e30;
    int                best_ep  = 0;
    const std::size_t  esz = static_cast<std::size_t>(kStageCount) * fper * H;

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
        << "        EvalFile " << cfg.out << "  vs  the teacher .pat\n";
    log.flush();
    res.ok = true;
    return res;
  }

} // namespace islay
