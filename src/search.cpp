/**
 * @file search.cpp
 * @brief Negamax / PVS implementation (contract and the two Othello properties
 *        this leans on are documented in search.hpp).
 *
 * Correctness discipline: `ref_negamax` below is a plain full-window negamax with
 * no table, no pruning and no ordering -- the ORACLE. Everything the real search
 * adds (alpha-beta, TT, ordering, PVS, killers) is a pure optimization and must
 * return a byte-identical score. `search_selftest()` enforces exactly that, and
 * the kUse* switches let a mismatch be bisected to one feature in one rebuild.
 */
#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>

#include "eval.hpp"
#include "hash.hpp"
#include "nnue.hpp"
#include "pattern.hpp"
#include "movegen.hpp"
#include "stability.hpp"
#include "endgame.hpp"

namespace islay {
  namespace {

    // Bisection switches: flip one off and re-run search_selftest to localise a
    // mismatch against the oracle. All must be ON in a shipping build.
    constexpr bool kUseTT       = true;
    constexpr bool kUseOrdering = true;
    constexpr bool kUsePVS      = true;
    constexpr bool kUseKillers  = true;
    // Pure hint -- cannot change any score, so it is not part of the oracle
    // bisection above. Kept switchable only to A/B it.
    constexpr bool kUsePrefetch = true;

    // Search telemetry (searchstats.hpp). OFF ships: it is a measurement build only.
    // Pure observation (reads state, bumps counters -- never a score), so it is not
    // part of the oracle. Flip ON, rebuild, `go depth N`, then `searchstats`.
    constexpr bool kStats = false;

    // --- group B: these deliberately CHANGE the fixed-depth value -------------
    // An extension searches deeper than `depth` says; a reduction/futility cut
    // searches less. Either way the result is no longer the depth-d minimax
    // value, so search_selftest turns both OFF for its oracle comparison (that
    // is not a dodge -- "equals a fixed-depth minimax" is simply not a property
    // a selective searcher has, or wants).
    //
    // They must still never corrupt an EXACT solve. `depth - empties` is
    // invariant down a line, so `depth >= 64 - b.count()` identifies a node that
    // is proving the true game result; heuristic pruning is disabled there,
    // while extensions stay safe because they only ever look deeper.
    constexpr bool kUseExtensions = true;
    constexpr bool kUsePruning    = true;
    // Null-move pruning. Default OFF at the flag until a match clears the bar; the
    // constexpr stays true so the code builds and the block is one runtime bool away.
    constexpr bool kUseNmp        = true;
    constexpr int  kNmpMinDepth   = 3; // need room for the R+1 reduction to leave depth >= 1
    constexpr int  kNmpMinEmpties = 10; // above the endgame-parity zone where a pass can beat a move
    constexpr int  kNmpBaseR      = 2;
    constexpr int  kNmpDepthDiv   = 6; // R = base + depth/div: deeper nodes reduce harder
    // OFF: measured -101 Elo. Kept, unlike the four rejects below, because its
    // premise is sound and it is blocked by ONE number that a trained eval fixes.
    // Full story and the exact revival procedure at kProbCutFit.
    constexpr bool kUseProbCut = true;
    // One-empty fast path (solve the last square without recursing). Exact -- the
    // oracle pins it -- so switchable only to A/B its speed, never its result.
    constexpr bool kUseLastEmpty = true;
    // Parity-aware aspiration: centre the window on the SAME-parity score (d-2, which
    // ProbCut showed predicts d at r=0.99), not the opposite-parity d-1 that sank the
    // naive version. Exact-preserving. MEASURED AND REJECTED (flag off) -- but the
    // implementation is CORRECT, not buggy, and the 2x2 that proves it is the useful
    // part. At depth 16 (startpos moves f5 f6 e6 f4 c3), nodes by (ProbCut x aspiration):
    //     PC off ASP off : 4.10M      PC off ASP on : 3.29M   (aspiration alone: -20%)
    //     PC on  ASP off : 3.03M      PC on  ASP on : 3.11M   (aspiration on top: +2.6%)
    // So aspiration genuinely works -- on a plain PVS it cuts a fifth of the tree. It
    // loses here for one reason: it and ProbCut fight over the SAME niche (make cutoffs
    // happen sooner), ProbCut wins it outright (3.03M < 3.29M -- ProbCut alone beats
    // aspiration alone), and stacking them doesn't compound because the cuts overlap --
    // ProbCut's scout-node forward cuts already remove what a narrow root window would,
    // so aspiration only adds its re-search-on-fail cost. More nodes at fixed depth means
    // fewer plies at equal time (a strict loss). Kept switchable -- if ProbCut is ever
    // pulled, this reclaims the -20%; and the root_search window param it added is a clean
    // generalisation, so it stays.
    constexpr bool kUseAspiration   = false;
    constexpr int  kAspWindow       = 500; // centi-discs; ~2 sigma of the d-2 -> d fit
    constexpr int  kAspMinDepth     = 4;
    /**
     * Calibrated LMR reduction. BUILT, MEASURED, AND REJECTED (`lmr_calibrated_` is
     * false; the shipped rule is still the fixed one at the call site). Kept, like
     * aspiration and LMP, so that nobody re-derives it from the same tempting premise.
     *
     * THE TEMPTING PREMISE, and why it was wrong. `searchstats` profiles the re-search
     * rate -- how often a reduced scout comes back above alpha and has to be redone --
     * per (remaining depth x move ordinal). On v12 at depth 14 that rate is 0-3% for
     * every ordinal past 6, against the 5-15% a healthy LMR is usually said to run. The
     * obvious reading is "reductions are far too timid". Three schedules were fitted to
     * the profile and every one measured MORE nodes at fixed depth 14 than the fixed
     * rule it replaced (baseline 1964094 over four positions):
     *     v1  1 + (i-3)/3 + (depth-3)/6                       2130113   +8.5%
     *     v2  ordinal steps + a parity give-back at i < 6      2119912   +7.9%
     *     v3  ordinal steps only (the code below)              1987401   +1.2%
     *
     * TWO HYPOTHESES WERE RAISED AND BOTH FALSIFIED, which is the useful part:
     *   1. "Late ordinals are too rare for a deeper reduction to matter." FALSE -- the
     *      ordinal traffic is nearly flat: i=3..8 carry 16/15/15/14/13/10% of reduced
     *      moves and i>=9 still carries 16.8%.
     *   2. "Reducing deeper buys re-searches that cost more than the saving." FALSE --
     *      under v3 the overall re-search rate moved 3.2% -> 3.1%, i.e. not at all.
     *
     * WHAT IS ACTUALLY GOING ON. The low re-search rate is not slack; it is a symptom
     * of how narrow this tree already is. Ordering gets a first-move cutoff 82.7% of
     * the time and the effective branching factor is 2.27, so a late move is genuinely
     * bad and the null-window scout refutes it at ANY reduction depth -- the subtree
     * being reduced was already nearly empty, so taking another ply off it saves close
     * to nothing. And the budget is not LMR's to win in the first place: ProbCut probes
     * are 68-78% of all searched nodes, so LMR governs roughly a quarter of the tree
     * and even a large relative gain there cannot approach the ~33% total saving the
     * EBF threshold (see kProbCutGateFit) demands before a match can be won.
     *
     * The genuinely new observation, worth keeping even though the schedule failed: the
     * re-search rate splits hard by the PARITY of the remaining depth. At ordinal 3,
     * odd depths re-searched 8.8% (d3, n=47337), 11.0% (d5), 8.0% (d7); even depths
     * 1.9% (d4, n=14543), 1.9% (d6), 4.8% (d8) -- a 4-5x gap on large samples, and the
     * same parity that makes ProbCut predict across d-2 and never d-1. Acting on it
     * (v2's give-back) cost 6.7% of nodes, so it is a real effect that is not
     * profitable to exploit HERE; it may matter to a future technique that is not
     * competing with ProbCut for the same quarter of the tree.
     */
    constexpr int kLmrMax = 4;

    [[nodiscard]] ISLAY_FORCEINLINE int lmr_reduction(int depth, int i) noexcept {
      int r = 1;
      if (i >= 6) ++r;  // re-search rate here is 0.1-3.0%
      if (i >= 9) ++r;  // 0-1.8%
      if (i >= 12) ++r; // ~0
      if (r > kLmrMax)
        r = kLmrMax;
      // Never reduce into the leaf: the child must keep at least one ply, or the
      // "reduced scout" degenerates into a static eval and proves nothing.
      if (r > depth - 2)
        r = depth - 2;
      return r < 0 ? 0 : r;
    }

    constexpr int kHistoryLimit = 1 << 20;

    ISLAY_FORCEINLINE void update_history(int (&row)[64], Square sq, int bonus) noexcept {
      row[sq] += bonus;
      if (row[sq] > kHistoryLimit || row[sq] < -kHistoryLimit)
        for (int &value: row)
          value /= 2;
    }

    // Endgame stability cutoff (exact; oracle-checked). Strong fixpoint in stability.hpp.
    constexpr bool kUseStabilityCut = true;
    // Endgame parity move ordering (reorders only -> exact; oracle-checked).
    constexpr bool kUseParity     = true;

    /**
     * ProbCut. Unlike everything in the rejected list below, its numbers are FITTED,
     * not guessed: a depth-(d-2) search predicts the depth-d score, and the engine
     * measures its own prediction error rather than assuming one.
     *
     * The model, per remaining depth d:   v_d ~= a * v_{d-2} + b,  s.d. sigma
     *
     * So v_d >= beta is claimed at t sigma of confidence when
     *     v_{d-2} >= (beta + t*sigma - b) / a
     * and a null-window search at d-2 answers exactly that question cheaply.
     *
     * WHY d-2 AND NOT d-1 OR d-3 -- this is the whole reason it can work here.
     * Othello scores swing hard with the parity of the depth (a side that gets the
     * last move of a line looks much better), so a prediction across an odd gap is
     * fighting that swing. Measured over 400 random positions to depth 12
     * (`pcdata 400 12`), r and sigma by gap:
     *     d -> d-2 :  r = 0.93-0.97,  sigma = 189-278 cd
     *     d -> d-3 :  r = 0.84-0.95,  sigma = 280-371 cd
     * and the parity shows up directly in the intercepts, which alternate sign and
     * decay with depth (+111, -69, +84, -52, +53, -27, ...) for the d-2 fit but
     * reach +-340 for d-3. Keeping the parity is what buys the correlation.
     *
     * MEASURED, AND CURRENTLY OFF (-101 Elo). Do not switch it on without redoing
     * the fit AND the match.
     *
     * At t = 0.5 it was the only setting that saved anything -- 899871 nodes / 71ms
     * against a 1267094 / 126ms baseline, so -29% nodes and -44% time -- and it lost
     * 60 games at equal 50ms by +21 =1 -38, z = -2.29, **-101 Elo**, 95% CI
     * [-203, -14]. Every safer setting (t >= 0.75) was strictly dominated: MORE nodes
     * than the baseline AND slower, because the two d-2 probes cost more than the
     * cuts they bought. There is no viable t.
     *
     * WHY, precisely -- and this is the useful part. The residual is not small
     * compared to the spread of the thing being predicted:
     *     d= 8: SD(v_d) = 771 cd, sigma = 222 cd  -> a t=1.5 margin is 43% of one SD
     *     d=10: SD(v_d) = 926 cd, sigma = 278 cd  -> 45%
     *     d=12: SD(v_d) = 1046 cd, sigma = 248 cd -> 36%
     * To cut with confidence the shallow score must clear the window by nearly half
     * a standard deviation, which is most of the range where the decision is
     * actually in doubt. sigma is that big because THIS eval is noisy: hand-written,
     * hand-guessed weights, never tuned. ProbCut was invented for Othello and worked
     * (Logistello) on top of a TRAINED pattern eval, i.e. a much smaller sigma.
     *
     * PREDICTION MADE HERE, AND FALSIFIED -- left in, because the correction is the
     * useful part. The claim was "train the eval and sigma drops below ~150 cd,
     * unlocking this". Measured against trained weights, sigma went UP, to 299-458.
     * The criterion was simply wrong: sigma is not scale-invariant, and a better eval
     * uses a WIDER score range (SD(v_d) rose 771-1046 -> 1472-1567), so its absolute
     * residual grows even as the prediction improves. The scale-free quantity is the
     * RATIO, and by that measure training helped only modestly:
     *     handcrafted: r = 0.93-0.97, t=1.5 margin = 36-45% of one SD
     *     trained v1 : r = 0.95-0.98, t=1.5 margin = 29-47% of one SD (best at d=12)
     * A margin still eating ~30% of an SD is still most of the range where the
     * decision is in doubt, so the answer is probably unchanged -- but "probably" is
     * exactly what this file does not accept.
     *
     * RETEST LIKE THIS, and judge on the ratio, never on sigma:
     *   1. `setoption name EvalFile value <weights>` then `pcdata 400 12`; re-fit.
     *   2. Compare 1.5*sigma against SD(v_d), not against a fixed cd figure.
     *   3. Set kUseProbCut = true, sweep t, settle with `match pc 100 50 <weights>`.
     *      A node count is NOT evidence here; see the rejects below for why.
     */
    struct ProbCutFit {
      float a, b, sigma;
    };
    constexpr int        kProbCutMaxFitDepth = 12;
    // RE-FITTED ON weights/v16.pat (`pcdata 800 12`), the eval now shipped. The fit is
    // slightly tighter than v12's (margin 16.0-31.5% of an SD against 16.4-35.6%) and
    // measured NEUTRAL in play: -3.9 Elo, 95% CI [-17, 9], 800 games at equal time.
    // Kept anyway, because a table fitted on an eval the engine no longer ships is a
    // latent inconsistency, and neutral means keeping it costs nothing. The neutrality
    // itself is consistent with the flat t-sweep recorded below: this machinery is
    // insensitive to calibration within a wide band.
    //
    // Also tested and NEUTRAL: lowering kProbCutGap4MinDepth from 9 to 8, which v16's
    // tighter d-4 fit appears to justify (its d=8 margin is 25.9%, inside the band that
    // has won here). Measured +3.9 Elo, 95% CI [-10, 17], 800 games. Left at 9.
    //
    // Historically FITTED ON weights/v12.pat (`pcdata 500 12`). This is the eval ProbCut ships
    // against, and it matters: on v12 the shallow->deep correlation is r=0.99 and a
    // 1.5-sigma margin eats only 17-26% of a score SD, versus 29-47% on the earlier
    // v5 and 36-45% on the hand-written eval. Re-fit here if the shipped eval changes.
    constexpr ProbCutFit kProbCutFit[kProbCutMaxFitDepth + 1] = {
            {1, 0, 999}, {1, 0, 999}, {1, 0, 999},               // 0..2: unused, never consulted
            {1.010f, -114.1f, 397.8f},                           // d=3
            {0.997f, 46.5f, 332.8f},                             // d=4
            {1.000f, -17.1f, 314.9f},                            // d=5
            {0.996f, 28.7f, 250.7f},                             // d=6
            {0.992f, 6.0f, 235.9f},                              // d=7
            {1.007f, 9.9f, 234.0f},                              // d=8
            {1.001f, 1.2f, 242.0f},                              // d=9
            {1.002f, -3.0f, 213.9f},                             // d=10
            {1.004f, 15.3f, 207.9f},                             // d=11
            {1.004f, 18.6f, 209.7f},                             // d=12
    };
    constexpr int   kProbCutMinDepth = 5; // needs room for a d-2 search worth the saving
    // ABDADA: mark subtrees from this remaining depth up; shallower ones finish too
    // fast for deferral to buy anything over the cost of the marks.
    constexpr int kAbdadaMinDepth = 4;

    /**
     * WIDER PROBE GAP at deep nodes. The telemetry says ProbCut probes are 68-78% of
     * every node the engine searches, so the probe itself -- not the cut rate -- is the
     * budget. A d-4 probe costs about 1/EBF^2 = 1/2.27^2 ~ 19% of a d-2 probe, so if the
     * prediction still holds up the saving is large where it is worth most.
     *
     * Gap 4, never 3, for the same reason ProbCut uses d-2 and not d-1: it preserves the
     * PARITY of the remaining depth, and Othello scores swing hard with that parity.
     *
     * Fitted alongside the d-2 table from the same `pcdata 800 12` sample on v12, and
     * judged on the RATIO 1.5*sigma / SD(v_d) -- never on sigma, which is not scale-free
     * (that mistake is recorded at kProbCutFit). The d-4 prediction is weaker but stays
     * inside the band that has actually been shown to win here:
     *     d-2:  r = 0.971-0.994,  margin 16-36% of an SD
     *     d-4:  r = 0.954-0.988,  margin 23-45% of an SD
     * The margin only enters the proven range once the node is deep, so gap 4 is used
     * from kProbCutGap4MinDepth up (d>=9: r>=0.983, margin <=27.8%) and gap 2 below it,
     * where d-4 would be a 37-45% margin -- wider than anything that has worked.
     *
     * MEASURED: +19 Elo, 95% CI [4, 34], z = 2.47 (600 pairs / 1200 games, equal time
     * 50ms, v12, `match pcg4`), all six seeds positive, and completed depth at equal
     * time rises on 4 of 6 positions at BOTH 50ms and 500ms.
     *
     * The saving GROWS with depth -- -13.2% nodes at fixed depth 14 but -41.3% at depth
     * 16 -- because the wider gap only fires from depth 9 up, so a deeper search has a
     * larger share of nodes eligible for it. That puts the 50ms figure above at the
     * PESSIMISTIC end: at a longer time control this should be worth more, not less.
     * It also means the EBF screen has to be applied at the depth the engine will
     * actually reach, not at a convenient one.
     */
    constexpr int kProbCutGap4MinDepth = 9;
    constexpr ProbCutFit kProbCutFit4[kProbCutMaxFitDepth + 1] = {
            {1, 0, 999}, {1, 0, 999}, {1, 0, 999}, {1, 0, 999},  // 0..3: unused
            {1, 0, 999}, {1, 0, 999}, {1, 0, 999},               // 4..6: gap 2 is used there
            {0.992f, -11.3f, 386.0f},                            // d=7
            {1.004f, 38.1f, 332.8f},                             // d=8
            {0.995f, 6.1f, 316.5f},                              // d=9
            {1.010f, 5.8f, 295.5f},                              // d=10
            {1.007f, 15.6f, 301.3f},                             // d=11
            {1.008f, 12.8f, 271.9f},                             // d=12
    };

    /**
     * ProbCut PROBE GATE, motivated by the telemetry (searchstats.hpp): on v12 only
     * ~38% of ProbCut attempts actually cut, so ~62% pay for one or both d-2 probes and
     * buy nothing -- and the probes are 68% of ALL search nodes. This gate skips a probe
     * the STATIC eval says almost surely will not fire, for the price of one leaf eval
     * (cheap next to a d-2 subtree).
     *
     * Fit: the depth-0 static eval predicts the depth-d2 probe result at r = 0.87-0.96
     * (v12, `pcdata 800 12` with the standpat column), v_{d2} ~= a*standpat + b, s.d.
     * sigma. Indexed by PROBE depth d2 (= node depth - 2). The gate SKIPS the hi-probe
     * when even a + kProbCutGateT-sigma optimistic prediction cannot reach the hi
     * threshold, and likewise for the lo-probe -- so it only ever drops probes that were
     * overwhelmingly going to miss. Exact-safe: skipping a probabilistic cut can only
     * make a node MORE accurate (it gets a real search instead), never less, so the
     * oracle is untouched. Re-fit whenever the shipped eval changes.
     *
     * MEASURED: +25 Elo, 95% CI [9, 40], z = 3.18 (600 pairs / 1200 games, equal time
     * 50ms, v12, `match pcg`), with all six seeds >= 0. It also RAISES completed depth
     * at equal time -- +1 ply on 3 of 6 positions at 200ms, +2 on one at 50ms, never
     * lower -- which is what separates it from every earlier node-saving change that
     * measured neutral.
     *
     * WHY t = 0.5, and the general threshold this exposed. Node savings by gate t at
     * fixed depth 14: t=0.0 -31%, t=0.5 -34%, t=1.0 -13%, t=1.5 -13%, t=2.5 -2%,
     * t=4.0 0% (the last is the sanity check -- the gate never fires). t=0.5 is the
     * optimum: below it the gate starts skipping probes that WOULD have cut, above it
     * it stops firing. The first attempt shipped t=1.5 and measured +10 Elo, CI
     * [-17, 36] -- real but unresolvable, because of an arithmetic worth remembering:
     * Othello's EBF is ~2.31, so ONE extra ply costs a factor 2.31 of nodes and a
     * saving must reach 1 - 1/2.31 = 57% to buy it outright. -13% is only
     * log(1/0.87)/log(2.31) = 0.19 ply and rounds to zero completed plies; -34% is
     * 0.50 ply and lands over the line often enough to show up. Below roughly a third,
     * a node-saving technique in this game cannot be expected to win a match at all.
     */
    constexpr float kProbCutGateT = 0.5f; // gate confidence in sigmas; larger = skip fewer
    constexpr ProbCutFit kProbCutGateFit[kProbCutMaxFitDepth + 1] = {
            {0.983f, 279.5f, 549.4f}, {0.983f, 279.5f, 549.4f}, {0.992f, 126.5f, 612.5f},
            {0.986f, 171.4f, 716.4f}, {0.980f, 176.6f, 736.9f}, {0.981f, 155.8f, 798.1f},
            {0.969f, 207.6f, 804.8f}, {0.970f, 162.2f, 840.6f}, {0.974f, 219.7f, 849.5f},
            {0.970f, 164.0f, 879.4f}, {0.975f, 217.4f, 880.4f}, {0.977f, 178.8f, 897.7f},
            {0.973f, 228.2f, 910.9f},
    };

    // Late move PRUNING (drop the tail of the move list unsearched in a scout node).
    // Re-tested on the TRAINED eval (v12): -24 Elo, CI [-53,4] -- a huge lift from the
    // -154 it cost on the hand-written eval, but STILL not a win, so lmp_enabled_
    // defaults off. The lesson: a BLIND cut (drop a move with no search) does not flip
    // positive the way ProbCut's VERIFIED cut (a shallow search confirms it) did, even
    // when a better eval makes the move ordering it bets on much sharper. Runtime-gated
    // so it stays A/B-able. `match lmp [pairs] [ms] [eval]`.
    constexpr bool  kUseLMP       = true;
    constexpr int   kLMPMaxDepth  = 3;
    constexpr int   kLMPCount[4]  = {0, 4, 6, 9};

    // MULTI-ProbCut: the same model, but the fit is per GAME STAGE, not pooled over
    // all of them. MEASURED NEUTRAL vs the pooled per-depth fit at 50ms (per-stage vs
    // pooled = -9 Elo, CI [-36,18], 300 games), so it is OFF by default. WHY it does not
    // help here: it DOES change behaviour -- at depth 14 it cuts more in the opening
    // (519k vs 565k nodes) and less in the midgame (652k vs 556k) -- but the net is a
    // wash, because the quantity that governs cut safety, the sigma/SD ratio, is nearly
    // uniform (~18-22%) across every phase where ProbCut fires. v12 predicts shallow->deep
    // equally well in every phase, so the pooled fit is already well-calibrated and
    // per-stage just redistributes cuts to no effect. It may pay at longer time controls
    // (deeper search exposes more phase variation); kept behind set_mpc_perstage for that. Measured (`pcdata 1500 12` on v12) the shallow->deep relationship
    // differs by phase -- the opening (stage 0-1, ~8 discs) has a looser slope
    // (a ~ 0.88-0.9) and, once corrected, a smaller residual, while the endgame swings
    // wider. A single pooled fit over-cuts one end and under-cuts the other; per-stage
    // gives each phase its own calibrated margin. Sparse high-stage cells fall back to
    // the pooled per-depth fit.
    // kMpcFit[stage][depth] = {a, b, sigma}, fitted per-stage on v12 (pcdata 1500 12).
    // Sparse cells (n<120) fall back to the per-depth pooled fit. Opening stages
    // have the loosest correlation, so their bigger sigma makes ProbCut appropriately
    // cautious there -- the whole point of going per-stage.
    constexpr ProbCutFit kMpcFit[kStageCount][kProbCutMaxFitDepth + 1] = {
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.999f,87.1f,418.5f}, {0.992f,-21.4f,356.5f}, {0.990f,44.5f,313.2f}, {0.991f,-21.4f,285.3f}, {1.000f,26.2f,271.6f}, {0.999f,5.6f,260.0f}, {1.001f,10.1f,242.5f}, {1.002f,-1.6f,224.1f}, {1.006f,0.6f,217.1f}}, // stage 0
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.894f,138.4f,271.3f}, {0.886f,59.4f,206.9f}, {0.880f,85.0f,215.4f}, {0.932f,-13.5f,158.9f}, {0.979f,-6.8f,159.8f}, {0.987f,-26.3f,154.8f}, {0.946f,27.9f,135.8f}, {0.953f,12.2f,138.7f}, {0.990f,-16.9f,134.5f}}, // stage 1
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {1.012f,88.7f,298.8f}, {0.975f,-21.9f,253.8f}, {0.970f,18.6f,213.1f}, {0.985f,1.2f,218.8f}, {0.989f,3.6f,180.3f}, {0.971f,38.6f,197.7f}, {0.978f,11.5f,202.9f}, {1.002f,-23.2f,179.4f}, {1.005f,-7.1f,171.3f}}, // stage 2
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.986f,33.2f,345.2f}, {0.971f,27.9f,294.3f}, {0.947f,31.6f,282.6f}, {0.966f,-2.3f,269.8f}, {0.993f,5.2f,231.3f}, {0.983f,-22.9f,269.2f}, {0.972f,-18.3f,208.6f}, {0.983f,-5.0f,192.6f}, {0.986f,-4.9f,176.9f}}, // stage 3
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {1.016f,-33.4f,439.5f}, {0.986f,-68.2f,374.4f}, {0.984f,20.1f,293.6f}, {0.969f,-28.0f,259.6f}, {0.980f,-10.8f,266.9f}, {0.990f,20.1f,224.7f}, {0.984f,49.4f,219.4f}, {0.991f,-3.5f,176.3f}, {0.991f,-10.7f,197.7f}}, // stage 4
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.973f,71.8f,379.5f}, {0.974f,-28.8f,313.3f}, {0.961f,-0.6f,304.8f}, {0.971f,-49.8f,262.2f}, {0.979f,18.7f,230.2f}, {0.981f,43.7f,240.5f}, {0.978f,33.7f,189.9f}, {0.984f,9.2f,184.5f}, {0.993f,-24.2f,173.8f}}, // stage 5
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.988f,96.7f,403.1f}, {0.995f,4.3f,328.8f}, {0.987f,7.5f,328.0f}, {0.975f,-18.6f,318.6f}, {0.997f,29.7f,248.4f}, {0.997f,-15.0f,256.9f}, {1.004f,-30.5f,203.2f}, {0.995f,-23.8f,200.4f}, {0.994f,12.2f,196.6f}}, // stage 6
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {1.002f,94.5f,411.1f}, {0.980f,-54.0f,390.5f}, {0.996f,52.5f,291.7f}, {0.977f,-21.7f,255.9f}, {0.971f,35.6f,275.8f}, {0.994f,-6.8f,258.0f}, {1.011f,-22.4f,241.9f}, {0.998f,14.1f,233.4f}, {1.013f,12.8f,187.0f}}, // stage 7
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {1.024f,137.6f,491.9f}, {1.011f,-31.2f,387.5f}, {1.010f,84.0f,340.3f}, {1.017f,-35.0f,300.7f}, {1.011f,26.2f,305.4f}, {1.029f,9.2f,272.0f}, {1.034f,15.2f,271.0f}, {1.005f,-14.4f,244.2f}, {1.011f,34.9f,229.0f}}, // stage 8
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.998f,111.3f,495.7f}, {1.029f,-11.9f,395.8f}, {1.026f,54.9f,319.9f}, {0.985f,-26.7f,298.0f}, {1.036f,74.5f,327.9f}, {1.026f,-38.0f,356.7f}, {1.006f,11.1f,270.2f}, {1.025f,45.2f,247.5f}, {1.025f,6.9f,279.4f}}, // stage 9
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {1.017f,101.7f,457.8f}, {0.987f,-22.2f,378.9f}, {0.983f,84.9f,350.8f}, {1.015f,13.7f,343.7f}, {1.010f,69.0f,278.6f}, {0.993f,49.0f,279.8f}, {1.025f,61.2f,297.6f}, {1.022f,34.0f,279.4f}, {1.001f,30.9f,239.8f}}, // stage 10
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.999f,87.1f,418.5f}, {0.992f,-21.4f,356.5f}, {0.990f,44.5f,313.2f}, {0.991f,-21.4f,285.3f}, {1.000f,26.2f,271.6f}, {0.999f,5.6f,260.0f}, {1.001f,10.1f,242.5f}, {1.002f,-1.6f,224.1f}, {1.006f,0.6f,217.1f}}, // stage 11
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.999f,87.1f,418.5f}, {0.992f,-21.4f,356.5f}, {0.990f,44.5f,313.2f}, {0.991f,-21.4f,285.3f}, {1.000f,26.2f,271.6f}, {0.999f,5.6f,260.0f}, {1.001f,10.1f,242.5f}, {1.002f,-1.6f,224.1f}, {1.006f,0.6f,217.1f}}, // stage 12
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.999f,87.1f,418.5f}, {0.992f,-21.4f,356.5f}, {0.990f,44.5f,313.2f}, {0.991f,-21.4f,285.3f}, {1.000f,26.2f,271.6f}, {0.999f,5.6f,260.0f}, {1.001f,10.1f,242.5f}, {1.002f,-1.6f,224.1f}, {1.006f,0.6f,217.1f}}, // stage 13
        {{1,0,999}, {1,0,999}, {1,0,999}, {1,0,999}, {0.999f,87.1f,418.5f}, {0.992f,-21.4f,356.5f}, {0.990f,44.5f,313.2f}, {0.991f,-21.4f,285.3f}, {1.000f,26.2f,271.6f}, {0.999f,5.6f,260.0f}, {1.001f,10.1f,242.5f}, {1.002f,-1.6f,224.1f}, {1.006f,0.6f,217.1f}}, // stage 14
    };

    // --- REJECTED, all four, on measurement. Read before adding any of them back.
    //
    // Baseline for every number below: depth 14 from a 5-ply opening = 1267094
    // nodes / 126ms ("mid"), and a solving 18-empty endgame at depth 18 = 1951023
    // nodes / 123ms ("end"). A leave-one-out A/B reproduced that baseline exactly
    // when all four were off, which is what says the harness was measuring these
    // and not the weather.
    //
    // * LATE MOVE PRUNING (skip the tail of the move list at shallow depth in a
    //   scout node; counts {4,6,9} for depth {1,2,3}). -55% nodes, 2.0x faster
    //   (mid 571695 / 63ms with ETC) -- and **-154 Elo** (60 games at equal 50ms,
    //   +17 =1 -42, z = -3.56, 95% CI [-267, -65]). Node counts lied completely.
    //   The chess assumption behind LMP is that late moves are quiet and rarely
    //   change anything; in Othello EVERY move is a capture, so the move ordered
    //   8th may flip fifteen discs and take a corner. Do not retry without games:
    //   the fixed-depth move choice already disagreed with a full search in 50% of
    //   positions (mean |score| delta 62 cd, max 237 cd).
    // * ENHANCED TRANSPOSITION CUTOFF. Only ever paid WITH LMP (-15% nodes there);
    //   on its own it cost +32% nodes (1671193 vs 1267094). With LMP gone it has no
    //   case left. Storing the cutoff before returning was necessary but not
    //   sufficient -- skipping the store first cost +27.8%, because an unmemoised
    //   node is re-searched in full every time Othello transposes back into it. The
    //   remaining +32% is NOT explained; the suspect is the replacement policy
    //   (`e.key == key` always replaces, so an ETC kLower can evict a deeper kExact)
    //   and that is the lead to pull if anyone retries this.
    // * ASPIRATION WINDOWS (150 cd around the previous score, depth >= 5). +19.6%
    //   nodes (1030083 vs 861407), and slower. Exact-preserving, correctly widened
    //   on every fail -- and still a loss, because the bet it makes is precisely the
    //   one this engine loses: the score OSCILLATES between iterations, so the
    //   window misses and pays for a re-search over and over.
    // * STABILITY CUTOFF. An opponent disc that can never flip is one the mover can
    //   never own, so with S of them the final difference is at most 64 - 2S (the
    //   empties cancel, so it holds however they break) -- a proof, not a heuristic.
    //   It also never fires: alpha near the middle of the range needs S >= ~31,
    //   which only exists in the last plies where the subtree is already trivial.
    //   Measured on a genuinely solving endgame the tree was BIT-IDENTICAL (1430320
    //   nodes both ways) while the check cost 21% of the time. Sound, inert, and not
    //   free. Needs a far stronger stability routine before it could ever bite.
    //
    // NULL MOVE PRUNING is not here and should not be: its premise is that having
    // the move is an advantage, so standing pat and still beating beta proves the
    // real move would too. In Othello a null move is a PASS -- a legal move -- and
    // being forced to move is routinely a liability. Zugzwang is not an exception
    // in this game, it is the whole point of mobility play. The premise is false.

    /** Futility margin by remaining depth, in centi-discs. Hand-guessed. */

    // Ordering only pays where the subtree it saves dwarfs its own cost:
    // play()+get_moves per move is ~120 instructions, so ordering ~10 moves costs
    // ~1200 -- more than a depth-1 subtree (~10^3) is worth. Gate it.
    constexpr int kOrderMobilityMinDepth = 3;

    using Clock = std::chrono::steady_clock;
    [[nodiscard]] double now_ms() noexcept {
      return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
    }

    /** Classic positional bias for cheap ordering. Hand-guessed, D4-symmetric. */
    constexpr int kSquareValue[64] = {
            18, -4, 4, 2, 2, 4, -4, 18, //
            -4, -8, 1, 1, 1, 1, -8, -4, //
            4,  1,  2, 1, 1, 2, 1,  4,  //
            2,  1,  1, 1, 1, 1, 1,  2,  //
            2,  1,  1, 1, 1, 1, 1,  2,  //
            4,  1,  2, 1, 1, 2, 1,  4,  //
            -4, -8, 1, 1, 1, 1, -8, -4, //
            18, -4, 4, 2, 2, 4, -4, 18};

    /**
     * Compact no-TT move policy, distilled from 20k independent v22 searches at
     * depth 10. The 120 int16 weights use D4-invariant square/relative-geometry
     * buckets, so the model is 240 bytes and cannot learn a spurious board
     * orientation. Held-out top-1 teacher-move accuracy: 32.12% -> 37.50%; MRR:
     * 0.5335 -> 0.5794. It replaces only the hand-written square/mobility prior;
     * dynamic killer/history/continuation evidence is still added afterwards.
     */
    constexpr int kOrderSquareOrbit[64] = {
            0, 1, 2, 3, 3, 2, 1, 0, //
            1, 4, 5, 6, 6, 5, 4, 1, //
            2, 5, 7, 8, 8, 7, 5, 2, //
            3, 6, 8, 9, 9, 8, 6, 3, //
            3, 6, 8, 9, 9, 8, 6, 3, //
            2, 5, 7, 8, 8, 7, 5, 2, //
            1, 4, 5, 6, 6, 5, 4, 1, //
            0, 1, 2, 3, 3, 2, 1, 0};
    constexpr std::int16_t kOrderSquare[4][10] = {
            {55, -39, 12, -7, -139, 20, 43, 85, 106, 8},
            {88, -50, 1, -3, -79, 6, 20, 68, 85, 8},
            {88, -6, 9, 5, -46, 0, 5, 36, 45, 8},
            {83, 4, 18, 9, -24, -26, 4, 41, 27, 8},
    };
    constexpr std::int16_t kOrderFlip[4]     = {-40, -26, -2, -4};
    constexpr std::int16_t kOrderMobility[4] = {-31, -30, -35, -58};
    constexpr std::int16_t kOrderPrev[36] = {
            0, 4, 0, 14, -2, 7, 1, 4, 10, 20, -5, -4, -2, 6, -1, 13, -10, 7,
            -9, -11, -7, -8, -5, 3, -6, 16, -5, 8, 2, -13, 1, -7, -2, 3, -9, -12,
    };
    constexpr std::int16_t kOrderPrev2[36] = {
            0, 18, 2, -2, 12, -21, 12, 9, 0, 2, -7, 10, 1, 6, -12, 0, 3, 10,
            2, -9, 0, -12, 5, -3, -11, 1, -6, -1, -7, 5, 6, -10, -2, -3, -10, 11,
    };

    [[nodiscard]] ISLAY_FORCEINLINE int order_relation(Square previous, Square square) noexcept {
      if (previous < 0 || previous >= 64)
        return 0;
      int dx = (previous & 7) - (square & 7);
      int dy = (previous >> 3) - (square >> 3);
      dx = dx < 0 ? -dx : dx;
      dy = dy < 0 ? -dy : dy;
      if (dx > dy)
        std::swap(dx, dy);
      return dy * (dy + 1) / 2 + dx;
    }

    [[nodiscard]] ISLAY_FORCEINLINE int learned_order_score(const Board &b, const Board &child, Square square,
                                                             int replies, Square prev, Square prev2) noexcept {
      int phase = pattern_stage(b.count()) / 4;
      if (phase > 3)
        phase = 3;
      const int flips = popcount(b.player ^ child.opponent ^ square_bb(square));
      return kOrderSquare[phase][kOrderSquareOrbit[square]] + kOrderFlip[phase] * flips +
             kOrderMobility[phase] * replies + kOrderPrev[order_relation(prev, square)] +
             kOrderPrev2[order_relation(prev2, square)];
    }

    /**
     * Leaf value, from scratch. Used by the ORACLE, so it must agree with the
     * search's incremental path -- which is exactly what makes the oracle test
     * also a test of PatternState::update. Rebuilding the features here is O(64)
     * and that is fine: this is the reference, not the hot path.
     */
    [[nodiscard]] int leaf_value_scratch(const Board &b, Bitboard moves, Color stm) noexcept {
      if (!pattern_enabled())
        return eval(b, moves); // hand-written eval (eval.cpp) is the default
      PatternState s;
      s.set(b, stm);
      int black;
      if (nnue_enabled()) {
        std::uint32_t idx[kPatternInstances + 9];
        const int     discs = b.count();
        int           n     = pattern_indices(s, 0, mob_counts(b, stm, moves), idx);
        idx[n++]            = static_cast<std::uint32_t>(nnue_net().features() - kNnueRFeat +
                                                         (discs >= 4 ? (discs - 4) % 4 : 0));
        black               = nnue_net().score(idx, n, pattern_stage(discs));
      } else {
        black = pattern_weights().score_phase(s, b.count(), mob_counts(b, stm, moves), pattern_stage_interp());
      }
      const int mover = (stm == Color::Black) ? black : -black; // zero-sum: White is minus Black
      return std::clamp(mover, -kEvalMax, kEvalMax);
    }

    /**
     * THE ORACLE: plain negamax, full window, no table/pruning/ordering.
     * Deliberately the most obviously-correct thing that can be written.
     * Note the two Othello-specific rules (see search.hpp):
     *   - a pass NEGATES and does NOT consume depth;
     *   - the terminal test comes BEFORE the depth cutoff, so eval() is never
     *     asked to score a finished game.
     */
    template<Rule R>
    int ref_negamax(const Board &b, int depth, Color stm) noexcept {
      const Bitboard moves = b.moves();

      if (moves == 0) {
        if constexpr (R == Rule::Othello) {
          const Board passed = b.passed();
          if (passed.has_moves())
            return -ref_negamax<R>(passed, depth, ~stm); // pass: free, but negated
        }
        return terminal_score(b);
      }
      if (depth <= 0)
        return leaf_value_scratch(b, moves, stm); // b provably has a move here

      int      best = -kInf;
      Bitboard m    = moves;
      while (m) {
        const int s = -ref_negamax<R>(b.play(pop_lsb(m)), depth - 1, ~stm);
        if (s > best)
          best = s;
      }
      return best;
    }

    struct ScoredMove {
      Square sq;
      Board  child; // cached: ordering already paid for play(), the loop reuses it
      int    score;
    };

  } // namespace

  namespace {
    // Payload packing for the lockless table. Six bytes of real content in one 64-bit
    // word, so a reader gets it in a single atomic load and the XOR check below can
    // vouch that it belongs with the key it was read beside.
    [[nodiscard]] ISLAY_FORCEINLINE std::uint64_t tt_pack(int score, int depth, std::uint8_t flag,
                                                          std::uint8_t best, std::uint8_t age) noexcept {
      return static_cast<std::uint64_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(score))) |
             (static_cast<std::uint64_t>(static_cast<std::uint8_t>(depth)) << 16) |
             (static_cast<std::uint64_t>(flag) << 24) | (static_cast<std::uint64_t>(best) << 32) |
             (static_cast<std::uint64_t>(age) << 40);
    }
    [[nodiscard]] ISLAY_FORCEINLINE int          tt_score(std::uint64_t d) noexcept { return static_cast<std::int16_t>(d & 0xFFFF); }
    [[nodiscard]] ISLAY_FORCEINLINE int          tt_depth(std::uint64_t d) noexcept { return static_cast<int>((d >> 16) & 0xFF); }
    [[nodiscard]] ISLAY_FORCEINLINE std::uint8_t tt_flag(std::uint64_t d) noexcept { return static_cast<std::uint8_t>((d >> 24) & 0xFF); }
    [[nodiscard]] ISLAY_FORCEINLINE std::uint8_t tt_best(std::uint64_t d) noexcept { return static_cast<std::uint8_t>((d >> 32) & 0xFF); }
    [[nodiscard]] ISLAY_FORCEINLINE std::uint8_t tt_age(std::uint64_t d) noexcept { return static_cast<std::uint8_t>((d >> 40) & 0xFF); }
  } // namespace

  void TranspositionTable::resize(std::size_t mib) {
    const std::size_t n = tt_slots_for(mib, sizeof(Slot));
    slots_ = std::vector<Slot>(n); // atomics are not copyable, so build fresh
    mask_  = n - 1;
    clear();
  }

  void TranspositionTable::clear() noexcept {
    for (Slot &s: slots_) {
      s.key_xor.store(0, std::memory_order_relaxed);
      s.data.store(0, std::memory_order_relaxed);
    }
    age_.store(0, std::memory_order_relaxed);
    used_.store(0, std::memory_order_relaxed);
  }

  /**
   * DIRECT-MAPPED ON PURPOSE -- a two-way cluster was built and measured NEUTRAL, so
   * do not spend the complexity again. The motive looked sound: the telemetry hit rate
   * collapses to ~30% then ~12% in the late stages, and pairing xor-1 slots puts the
   * second way on the same cache line, so the extra look is nearly free. Measured
   * honestly (same eval both sides -- the first comparison accidentally used a v12-era
   * baseline against v18 and read a fantastical -67%): 552604 -> 552579 nodes at
   * depth 14 (nothing) and 1801033 -> 1728963 at depth 16 (-4%), hit rates unchanged.
   * The late-stage misses are not collisions to associate away; deep positions are
   * simply NEW, and no associativity invents an entry that was never stored. The same
   * re-measurement surfaced a happier fact: v18's tree at fixed depth is ~3x smaller
   * than v12's was -- the eval work compounded into search efficiency.
   */
  bool TranspositionTable::probe(std::uint64_t key, Hit &out) const noexcept {
    if (!mask_)
      return false;
    const Slot         &s  = slots_[key & mask_];
    const std::uint64_t kx = s.key_xor.load(std::memory_order_relaxed);
    const std::uint64_t d  = s.data.load(std::memory_order_relaxed);
    // The XOR is the whole guarantee: a pair written by two different threads will not
    // reproduce this key, so a torn entry reads as a miss rather than as a lie.
    if ((kx ^ d) != key || tt_flag(d) == kNone)
      return false;
    out.score = tt_score(d);
    out.depth = tt_depth(d);
    out.flag  = tt_flag(d);
    out.best  = tt_best(d);
    return true;
  }

  void TranspositionTable::store(std::uint64_t key, int score, int depth, std::uint8_t flag,
                                 std::uint8_t best) noexcept {
    if (!mask_)
      return;
    Slot               &s   = slots_[key & mask_];
    const std::uint64_t kx  = s.key_xor.load(std::memory_order_relaxed);
    const std::uint64_t old = s.data.load(std::memory_order_relaxed);
    const bool          same = ((kx ^ old) == key);
    const std::uint8_t  age = age_.load(std::memory_order_relaxed);
    // Prefer keeping a deeper entry from THIS generation; anything older loses.
    const bool replace = same || tt_flag(old) == kNone || tt_age(old) != age || tt_depth(old) <= depth;
    if (!replace)
      return;
    if (tt_flag(old) == kNone)
      used_.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t d = tt_pack(score, std::max(0, depth), flag, best, age);
    s.data.store(d, std::memory_order_relaxed);
    s.key_xor.store(key ^ d, std::memory_order_relaxed);
  }

  int TranspositionTable::hashfull() const noexcept {
    return mask_ ? static_cast<int>(used_.load(std::memory_order_relaxed) * 1000 / (mask_ + 1)) : 0;
  }

  unsigned CorrectionHistory::edge_bucket(const Board &b) noexcept {
    std::uint64_t x = (b.player & kEdges) * 0x9E3779B185EBCA87ULL;
    x ^= (b.opponent & kEdges) * 0xC2B2AE3D27D4EB4FULL;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    return static_cast<unsigned>((x ^ (x >> 32)) & 255U);
  }

  void CorrectionHistory::clear() noexcept {
    std::memset(stage_, 0, sizeof stage_);
    std::memset(prev_, 0, sizeof prev_);
    std::memset(prev2_, 0, sizeof prev2_);
    std::memset(edge_, 0, sizeof edge_);
  }

  int CorrectionHistory::predict(const Board &b, int stage, Square prev_move, Square prev2_move) const noexcept {
    stage = std::clamp(stage, 0, kStageCount - 1);
    int sum = stage_[stage] + edge_[stage][edge_bucket(b)];
    int n   = 2;
    if (prev_move < 64) {
      sum += prev_[stage][prev_move];
      ++n;
    }
    if (prev2_move < 64) {
      sum += prev2_[stage][prev2_move];
      ++n;
    }
    return sum / n;
  }

  void CorrectionHistory::update_entry(std::int16_t &entry, int target, int rate) noexcept {
    const int value = entry;
    int       step  = (target - value) * rate / 256;
    if (step == 0 && value != target)
      step = target > value ? 1 : -1;
    entry = static_cast<std::int16_t>(std::clamp(value + step, -400, 400));
  }

  void CorrectionHistory::update(const Board &b, int stage, Square prev_move, Square prev2_move,
                                 int deep_score, int static_score, int depth) noexcept {
    stage            = std::clamp(stage, 0, kStageCount - 1);
    const int target = std::clamp(deep_score - static_score, -400, 400);
    const int rate   = std::clamp(4 + 2 * depth, 4, 32);
    update_entry(stage_[stage], target, rate);
    update_entry(edge_[stage][edge_bucket(b)], target, rate);
    if (prev_move < 64)
      update_entry(prev_[stage][prev_move], target, rate);
    if (prev2_move < 64)
      update_entry(prev2_[stage][prev2_move], target, rate);
  }

  void Searcher::resize(std::size_t mib) {
    tt_->resize(mib);
    clear();
  }

  void Searcher::clear() noexcept {
    stop_flag_.store(false, std::memory_order_relaxed); // a stale stop must not kill the next search
    tt_->clear();
    std::memset(killers_, 0, sizeof killers_);
    std::memset(history_, 0, sizeof history_);
    std::memset(continuation_history_, 0, sizeof continuation_history_);
    std::memset(continuation_history_2_, 0, sizeof continuation_history_2_);
    correction_history_.clear();
  }

  void Searcher::set_correction_history_cap(int cap) noexcept {
    cap = std::clamp(cap, 0, 200);
    if (cap != correction_history_cap_) {
      correction_history_cap_ = cap;
      correction_history_.clear();
    }
  }

  void Searcher::new_search(const SearchLimits &limits) noexcept {
    // Cache it once: pattern_enabled() lives in another TU, so leaving the call
    // in the move loop stops the compiler hoisting anything out of it.
    pat_on_  = pattern_enabled();
    nnue_on_ = nnue_enabled(); // ditto (leaf_eval branches on it per leaf)
    if (pat_on_ && ps_.size() < static_cast<std::size_t>(kMaxPly))
      ps_.resize(kMaxPly);
    nodes_       = 0;
    stopped_     = false;
    // NOTE: stop_flag_ is deliberately NOT cleared here. Clearing it on the SEARCH
    // thread races with a `stop` that arrives between launching the thread and this
    // line -- the request would be wiped and the search would run on, which is exactly
    // the bug this comment exists to prevent a re-introduction of. The owner clears it
    // before starting a search instead (see uci.cpp run_search and clear()).
    node_cap_    = limits.nodes;
    start_ms_    = now_ms();
    deadline_ms_ = limits.movetime_ms > 0.0 ? start_ms_ + limits.movetime_ms : 0.0;
    soft_ms_ = 0.0;
    if (bump_age_)
      tt_->new_generation();
    if constexpr (kStats) { // one position's telemetry per go
      if (!stats_) stats_ = std::make_unique<SearchStats>();
      stats_->reset();
    }
  }

  void Searcher::check_stop() noexcept {
    // An asynchronous `stop` outranks the limits: it is the one condition the search
    // cannot discover for itself.
    if (stop_flag_.load(std::memory_order_relaxed))
      stopped_ = true;
    else if (node_cap_ && nodes_ >= node_cap_)
      stopped_ = true;
    else if (deadline_ms_ > 0.0 && now_ms() >= deadline_ms_)
      stopped_ = true;
  }

  // Roll-ups for the opt-in telemetry. Two views of the same buckets -- by remaining
  // depth and by game stage -- because a technique that pays at shallow depth may earn
  // at deep depth, and one that helps in the opening may hurt in the endgame; a single
  // total hides both. Columns, per row:
  //   pv/cut/all%  node-type mix (Knuth-Moore)
  //   tthit/ttcut  TT probe answer rate / cutoff rate
  //   fh1% cIdx    ordering quality: cutoffs landing on move 0, and mean cutoff index
  //   bf           effective branching (moves searched / node)
  //   lmr lmrRe%   reductions applied / of those, the share re-searched (wasted work)
  //   fut cut%     futility attempts / cut rate
  //   pc  cut%     ProbCut attempts / cut rate
  //   pvsRe%       PVS scout searches that needed a full-window re-search
  void SearchStats::dump(std::ostream &o, bool full) const {
    auto pct = [](std::uint64_t a, std::uint64_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; };
    auto avg = [](std::uint64_t a, std::uint64_t b) { return b ? double(a) / double(b) : 0.0; };

    const char *hdr =
      "  key       nodes   pv%  cut%  all% | tthit ttcut | fh1%   cIdx    bf |    lmr lmrRe% |    fut  cut% |     pc  cut% | pvsRe%\n";

    auto row = [&](const std::string &label, const Cell &a,
                   std::uint64_t npv, std::uint64_t ncut, std::uint64_t nall) {
      o << std::fixed << std::setprecision(1)
        << "  " << std::setw(4) << std::left << label << std::right
        << std::setw(11) << a.nodes
        << std::setw(6) << pct(npv, a.nodes)
        << std::setw(6) << pct(ncut, a.nodes)
        << std::setw(6) << pct(nall, a.nodes) << "  |"
        << std::setw(6) << pct(a.tt_hit, a.tt_probe)
        << std::setw(6) << pct(a.tt_cut, a.tt_probe) << "  |"
        << std::setw(6) << pct(a.fh_first, a.fh)
        << std::setw(7) << std::setprecision(2) << avg(a.cut_idx_sum, a.fh)
        << std::setw(6) << avg(a.moves_searched, a.nodes) << std::setprecision(1) << "  |"
        << std::setw(7) << a.lmr_try << std::setw(7) << pct(a.lmr_re, a.lmr_try) << "  |"
        << std::setw(7) << a.fut_try << std::setw(6) << pct(a.fut_cut, a.fut_try) << "  |"
        << std::setw(7) << a.pc_try << std::setw(6) << pct(a.pc_cut, a.pc_try) << "  |"
        << std::setw(6) << pct(a.pvs_re, a.pvs_scout) << "\n";
    };

    Cell          gtot{};
    std::uint64_t gpv = 0, gcut = 0, gall = 0;
    o << "=== search telemetry (interior pvs nodes; leaves/terminals/endgame-solver excluded) ===\n";

    o << "by remaining depth:\n" << hdr;
    for (int d = 0; d < kD; ++d) {
      Cell          a{};
      std::uint64_t npv = 0, ncut = 0, nall = 0;
      for (int s = 0; s < kS; ++s) {
        a.add(cell[d][s][PV]);  npv  += cell[d][s][PV].nodes;
        a.add(cell[d][s][Cut]); ncut += cell[d][s][Cut].nodes;
        a.add(cell[d][s][All]); nall += cell[d][s][All].nodes;
      }
      if (!a.nodes) continue;
      row("d" + std::to_string(d), a, npv, ncut, nall);
      gtot.add(a); gpv += npv; gcut += ncut; gall += nall;
    }

    if (!gtot.nodes) { o << "  (nothing recorded -- run `go` in a kStats build first)\n"; return; }

    o << "by game stage (stage s -> discs " << 4 << "+4s .. +3):\n" << hdr;
    for (int s = 0; s < kS; ++s) {
      Cell          a{};
      std::uint64_t npv = 0, ncut = 0, nall = 0;
      for (int d = 0; d < kD; ++d) {
        a.add(cell[d][s][PV]);  npv  += cell[d][s][PV].nodes;
        a.add(cell[d][s][Cut]); ncut += cell[d][s][Cut].nodes;
        a.add(cell[d][s][All]); nall += cell[d][s][All].nodes;
      }
      if (!a.nodes) continue;
      row("s" + std::to_string(s), a, npv, ncut, nall);
    }

    row("TOT", gtot, gpv, gcut, gall);
    o << "probe cost: ProbCut spent " << gtot.pc_probe_nodes << " probe nodes over "
      << gtot.pc_try << " attempts, " << gtot.pc_cut << " cut ("
      << std::fixed << std::setprecision(1) << pct(gtot.pc_probe_nodes, total_nodes)
      << "% of all " << total_nodes << " search nodes).\n";
    // hi vs lo split. They are not symmetric: hi runs first and returns on success, so
    // the lo-probe is only ever paid at nodes where hi already missed. If lo buys few
    // cuts for many nodes, it is the next thing to cut.
    o << "  hi-probe: " << (gtot.pc_cut - gtot.pc_lo_cut) << " cuts of " << gtot.pc_try
      << " attempts (" << pct(gtot.pc_cut - gtot.pc_lo_cut, gtot.pc_try) << "%), "
      << (gtot.pc_probe_nodes - gtot.pc_lo_nodes) << " nodes\n"
      << "  lo-probe: " << gtot.pc_lo_cut << " cuts of " << gtot.pc_lo_try
      << " attempts (" << pct(gtot.pc_lo_cut, gtot.pc_lo_try) << "%), "
      << gtot.pc_lo_nodes << " nodes = " << pct(gtot.pc_lo_nodes, total_nodes)
      << "% of the whole search\n";
    o << "no-TT ordering: " << gtot.order_no_tt << " of " << gtot.order_nodes
      << " ordered nodes (" << pct(gtot.order_no_tt, gtot.order_nodes) << "%), "
      << gtot.order_no_tt_fh << " fail-highs, " << gtot.order_no_tt_fh_first
      << " on the first move (" << pct(gtot.order_no_tt_fh_first, gtot.order_no_tt_fh) << "%).\n";

    // LMR calibration: re-search rate per (remaining depth x move ordinal). This is the
    // fit input for a reduction table -- a bucket whose reduced scouts almost never come
    // back above alpha is a bucket that is being reduced too little.
    o << "LMR re-search rate by depth x move ordinal (pct; '.' = no samples, n = tries):\n";
    o << "   d \\ i ";
    for (int i = 0; i < 14; ++i) o << std::setw(6) << i;
    o << "        n\n";
    for (int d = 0; d < kD; ++d) {
      std::uint64_t dn = 0;
      for (int i = 0; i < kIdx; ++i) dn += lmr_try[d][i];
      if (!dn) continue;
      o << "   d" << std::setw(2) << std::left << d << std::right << "   ";
      for (int i = 0; i < 14; ++i) {
        if (!lmr_try[d][i]) { o << std::setw(6) << "."; continue; }
        o << std::setw(6) << std::fixed << std::setprecision(1) << pct(lmr_re[d][i], lmr_try[d][i]);
      }
      o << std::setw(9) << dn << "\n";
    }
    // Column totals: how much traffic each ordinal actually carries. A reduction
    // schedule can only save where the moves ARE.
    {
      std::uint64_t col[kIdx] = {}, all = 0;
      for (int d = 0; d < kD; ++d)
        for (int i = 0; i < kIdx; ++i) { col[i] += lmr_try[d][i]; all += lmr_try[d][i]; }
      o << "   share ";
      for (int i = 0; i < 14; ++i) o << std::setw(6) << std::fixed << std::setprecision(1) << pct(col[i], all);
      o << std::setw(9) << all << "\n";
    }

    if (full) {
      o << "full (depth x stage) node counts, nonzero cells only:\n";
      for (int d = 0; d < kD; ++d)
        for (int s = 0; s < kS; ++s) {
          const std::uint64_t nn = cell[d][s][PV].nodes + cell[d][s][Cut].nodes + cell[d][s][All].nodes;
          if (nn)
            o << "  d" << d << " s" << s << " nodes=" << nn
              << " pv=" << cell[d][s][PV].nodes << " cut=" << cell[d][s][Cut].nodes
              << " all=" << cell[d][s][All].nodes << "\n";
        }
    }
  }

  int Searcher::static_eval(const Board &b, Color stm) const noexcept {
    const Bitboard moves = b.moves();
    if (!pattern_enabled())
      return eval(b, moves); // hand-written eval (eval.cpp)
    // Same computation as leaf_eval<true>, but with the pattern state rebuilt from
    // scratch instead of maintained incrementally -- the oracle pins the two equal.
    PatternState ps;
    ps.set(b, stm);
    int black;
    if (nnue_enabled()) {
      std::uint32_t idx[kPatternInstances + 9];
      const int     discs = b.count();
      int           n     = pattern_indices(ps, 0, mob_counts(b, stm, moves), idx);
      idx[n++]            = static_cast<std::uint32_t>(nnue_net().features() - kNnueRFeat +
                                                       (discs >= 4 ? (discs - 4) % 4 : 0));
      black               = nnue_net().score(idx, n, pattern_stage(discs));
    } else {
      black = pattern_weights().score_phase(ps, b.count(), mob_counts(b, stm, moves), pattern_stage_interp());
    }
    const int mover = (stm == Color::Black) ? black : -black;
    return std::clamp(mover, -kEvalMax, kEvalMax);
  }

  template<bool Pat>
  int Searcher::leaf_eval(const Board &b, Bitboard moves, int ply, Color stm) const noexcept {
    if constexpr (!Pat) {
      (void) ply;
      (void) stm;
      return eval(b, moves); // hand-written eval (eval.cpp) is the default
    } else {
    // ps_[ply] was maintained incrementally on the way down; leaf_value_scratch()
    // rebuilds the same thing from nothing, and the oracle test pins the two together.
      // Mobility (actual + potential) is NOT in the incremental state -- it cannot be
      // updated cheaply -- so it is recomputed here, at the one place it is consulted.
      // `moves` is this mover's legal moves, already in hand, but mob_counts recomputes
      // from the board so the eval and the oracle's scratch path stay bit-identical.
      (void) moves;
      int black;
      if (nnue_on_) {
        // NNUE leaf: the SAME flat feature indices the linear eval sums select
        // embedding rows instead (nnue.hpp) -- identical cache-line count, and the
        // per-stage head restores the stage dependence the phase buckets coarsen.
        std::uint32_t idx[kPatternInstances + 9];
        const int     discs = b.count();
        int           n     = pattern_indices(ps_[ply], 0, mob_counts(b, stm, moves), idx);
        // The r index is the net's interpolation input (nnue.hpp): within-bucket
        // disc count, so the eval does not step at stage boundaries.
        idx[n++]            = static_cast<std::uint32_t>(nnue_net().features() - kNnueRFeat +
                                                         (discs >= 4 ? (discs - 4) % 4 : 0));
        black               = nnue_net().score(idx, n, pattern_stage(discs));
      } else {
        black = pattern_weights().score_phase(ps_[ply], b.count(), mob_counts(b, stm, moves), pattern_stage_interp());
      }
      const int mover = (stm == Color::Black) ? black : -black; // zero-sum
      return std::clamp(mover, -kEvalMax, kEvalMax);
    }
  }

  template<Rule R, bool Pat>
  int Searcher::pvs(Board b, int depth, int alpha, int beta, int ply, Color stm, Square prev_move, Square prev2_move,
                    bool can_null) noexcept {
    if (stopped_)
      return 0;
    ++nodes_;
    if (ply > seldepth_)
      seldepth_ = ply; // deepest ply reached: extensions and passes push past `depth`
    if ((nodes_ & 1023) == 0)
      check_stop();

    const Bitboard moves = b.moves();

    // Last-empty fast path. Fires exactly when the normal code would SOLVE this node
    // (depth >= 1 at one empty == `solving` there), never where it would eval, so the
    // score is identical -- the oracle test pins it. It returns the exact game value
    // directly, skipping the child board, its get_moves, and a recursion frame.
    // Hand the last few plies to the specialised endgame solver (endgame.cpp). Fires
    // exactly where the normal code would SOLVE (depth >= empties), so the score is
    // identical and the oracle pins it.
    const int eg_empties = 64 - b.count();
    if (kUseLastEmpty && endgame_enabled_ && depth >= eg_empties && eg_empties <= kEndgameSolverMax) [[unlikely]]
      return endgame_solve<R>(b, alpha, beta, eg_empties, nodes_);

    // Terminal / pass BEFORE the depth cutoff: eval() must never see a finished
    // game, and a pass must not consume depth (that is what makes depth>=empties
    // an exact solve).
    if (moves == 0) [[unlikely]] {
      if constexpr (R == Rule::Othello) {
        const Board passed = b.passed();
        if (passed.has_moves()) {
          // A pass changes the mover but no square, so the features are unchanged.
          if constexpr (Pat)
            if (ply + 1 < kMaxPly)
            ps_[ply + 1] = ps_[ply];
          return -pvs<R, Pat>(passed, depth, -beta, -alpha, ply + 1, ~stm, prev_move, prev2_move);
        }
      }
      return terminal_score(b);
    }
    if (depth <= 0)
      return leaf_eval<Pat>(b, moves, ply, stm);
    if (ply >= kMaxPly - 1)
      return leaf_eval<Pat>(b, moves, ply, stm); // paranoia; unreachable on a legal 8x8 game

    const int      alpha_orig = alpha; // MUST be captured before the loop moves alpha
    const std::uint64_t key   = hash_board(b.player, b.opponent);
    Square         tt_move    = NOMOVE;

    // Telemetry scratch for THIS node (searchstats.hpp). Accumulated across the
    // several early returns and flushed exactly once, at the node's real exit, into
    // the bucket its resolved type selects. Dead and fully elided when kStats is off.
    SearchStats::Acc sacc;
    const bool       is_pv = beta - alpha > 1;

    if (kUseTT && tt_enabled_) {
      if constexpr (kStats) ++sacc.tt_probe;
      Entry e;
      if (tt_->probe(key, e)) {
        if constexpr (kStats) ++sacc.tt_hit;
        // A zeroed slot has best == 0, which aliases a1 and looks legal, and
        // flip() is UB for sq >= 64 -- so ALWAYS validate against `moves`.
        if (e.best < 64 && (moves & square_bb(e.best)))
          tt_move = static_cast<Square>(e.best);
        if (e.depth >= depth) { // never at the root: root_search doesn't call this
          const int s = e.score;
          if (e.flag == kExact) {
            if constexpr (kStats) { ++sacc.tt_cut; stats_->flush(depth, pattern_stage(b.count()), SearchStats::PV, sacc); }
            return s;
          }
          if (e.flag == kLower && s >= beta) {
            if constexpr (kStats) { ++sacc.tt_cut; stats_->flush(depth, pattern_stage(b.count()), SearchStats::Cut, sacc); }
            return s;
          }
          if (e.flag == kUpper && s <= alpha) {
            if constexpr (kStats) { ++sacc.tt_cut; stats_->flush(depth, pattern_stage(b.count()), SearchStats::All, sacc); }
            return s;
          }
        }
      }
    }

    // A node whose remaining depth already covers every empty square is proving
    // the true game result, not guessing. Heuristic pruning must stand down here
    // or the "exact" claim (and the endgame solve) becomes a lie.
    const bool solving = depth >= 64 - b.count();

    // Stability cutoff. If the opponent already holds S provably-unflippable discs,
    // it finishes with at least S, so the mover's final margin is at most 64 - 2S
    // (the empties cancel out of the bound however they break). A solving node whose
    // ceiling cannot reach alpha is therefore dead. Only under `solving`, where the
    // leaves are exact game results the bound actually describes -- below that the
    // leaves are a clamped heuristic the bound says nothing about. EXACT (a proof,
    // not a guess), so not gated on selective_enabled_ and the oracle checks it.
    //
    // Rejected once with a weak corner-anchored stability that needed S ~ 31 to bite;
    // stability.hpp now computes the real fixpoint set, so it fires deep in the
    // endgame. `stable_count` never over-counts, which is what keeps the bound sound.
    if (kUseStabilityCut && endgame_enabled_ && solving) {
      // Cheap gate first: stable_count <= popcount(opponent), so if even ALL of the
      // opponent's discs being stable would not drop the ceiling to alpha, the cut
      // cannot fire and the (much pricier) fixpoint is skipped. This is what makes the
      // cutoff a net win rather than a net cost -- the fixpoint runs only where it can bite.
      const int opp = popcount(b.opponent);
      if (100 * (64 - 2 * opp) <= alpha) {
        const int ub = 100 * (64 - 2 * stable_count(b.opponent, b.player));
        if (ub <= alpha) {
          if constexpr (kStats) stats_->flush(depth, pattern_stage(b.count()), SearchStats::All, sacc);
          return ub; // fail-soft: a true upper bound on this node's value
        }
      }
    }

    // ProbCut: ask a depth-(d-2) search whether the depth-d score is far enough past
    // the window to bet on, using this engine's own measured prediction error (the
    // fit and its derivation are at kProbCutFit). Never while solving -- a
    // probabilistic cut inside a proof would make `exact` a lie -- and never in the
    // PV, where a wrong guess costs the move actually played.
    //
    // Gated on `Pat`: kProbCutFit is measured on the TRAINED eval, where the
    // shallow->deep correlation is r=0.99 (margin ~20% of an SD) and it wins +89 Elo.
    // On the hand-written eval that fit is simply wrong and the correlation is far
    // looser, where the same technique LOST -101; so ProbCut only fires when a trained
    // eval is active. `Pat` is a template bool, so this compiles the block out entirely
    // for the hand-written path.
    if (Pat && kUseProbCut && probcut_enabled_ && selective_enabled_ && !solving && depth >= kProbCutMinDepth &&
        beta == alpha + 1 && beta < kScoreMax && alpha > -kScoreMax) {
      const int         di = depth < kProbCutMaxFitDepth ? depth : kProbCutMaxFitDepth;
      const int         sg = pattern_stage(b.count());
      // A deep node probes across 4 plies instead of 2: same parity, ~5x cheaper, and
      // on v12 the prediction is still inside the band that wins (see kProbCutFit4).
      const bool        g4 = probcut_gap4_ && depth >= kProbCutGap4MinDepth;
      const ProbCutFit &f  = g4 ? kProbCutFit4[di] : (mpc_perstage_ ? kMpcFit[sg][di] : kProbCutFit[di]);
      const int         d2 = depth - (g4 ? 4 : 2);
      const std::uint64_t pc0 = kStats ? nodes_ : 0; // probe cost = nodes the probes spend
      if constexpr (kStats) ++sacc.pc_try;

      // Probe gate: predict the d2 probe result from the static eval and skip a probe
      // that is overwhelmingly going to miss. One leaf eval buys the right to not pay a
      // whole d-2 subtree. `predf +/- gate` is the optimistic/pessimistic bound on
      // v_{d2}; if even the optimistic bound cannot reach hi, the hi-probe is hopeless
      // (symmetric for lo). Off (probcut_gate_enabled_ == false) reproduces the ungated
      // behaviour exactly, for the equal-time A/B. The leaf eval is only paid when on.
      const ProbCutFit &g    = kProbCutGateFit[d2 < 0 ? 0 : (d2 > kProbCutMaxFitDepth ? kProbCutMaxFitDepth : d2)];
      const float       gate = kProbCutGateT * g.sigma;
      float             predf = 0.0f;
      if (probcut_gate_enabled_) {
        const int raw = leaf_eval<Pat>(b, moves, ply, stm);
        const int correction = correction_history_cap_ > 0
                                 ? std::clamp(correction_history_.predict(b, sg, prev_move, prev2_move),
                                              -correction_history_cap_, correction_history_cap_)
                                 : 0;
        predf = g.a * static_cast<float>(raw + correction) + g.b;
      }

      // v_d >= beta  <=  a*v_{d-2} + b - t*sigma >= beta
      const int hi = static_cast<int>((static_cast<float>(beta) + probcut_t_ * f.sigma - f.b) / f.a);
      const bool try_hi = !probcut_gate_enabled_ || (predf + gate >= static_cast<float>(hi));
      if (try_hi && hi < kScoreMax && pvs<R, Pat>(b, d2, hi - 1, hi, ply, stm, prev_move, prev2_move) >= hi) {
        if constexpr (kStats) { ++sacc.pc_cut; sacc.pc_probe_nodes += nodes_ - pc0; stats_->flush(depth, sg, SearchStats::Cut, sacc); }
        return beta;
      }

      // v_d <= alpha  <=  a*v_{d-2} + b + t*sigma <= alpha
      const int lo = static_cast<int>((static_cast<float>(alpha) - probcut_t_ * f.sigma - f.b) / f.a);
      const bool try_lo = !probcut_gate_enabled_ || (predf - gate <= static_cast<float>(lo));
      const std::uint64_t lo0 = kStats ? nodes_ : 0; // the lo-probe's own cost
      if constexpr (kStats)
        if (try_lo && lo > -kScoreMax) ++sacc.pc_lo_try;
      if (try_lo && lo > -kScoreMax && pvs<R, Pat>(b, d2, lo, lo + 1, ply, stm, prev_move, prev2_move) <= lo) {
        if constexpr (kStats) {
          ++sacc.pc_cut; ++sacc.pc_lo_cut;
          sacc.pc_lo_nodes += nodes_ - lo0;
          sacc.pc_probe_nodes += nodes_ - pc0;
          stats_->flush(depth, sg, SearchStats::All, sacc);
        }
        return alpha;
      }
      if constexpr (kStats) sacc.pc_lo_nodes += nodes_ - lo0;

      // No cut: both probes ran and bought nothing -- charge their nodes to this node,
      // which will be flushed later at the move-loop exit.
      if constexpr (kStats) sacc.pc_probe_nodes += nodes_ - pc0;

      if (stopped_)
        return 0;

      // TRIED AND REJECTED: re-reading the TT here to "salvage" the best move the failed
      // probe just stored (the table is read above, BEFORE the probes run, so that move
      // is discarded). It sounds free -- a d-2/d-4 search picked it -- and it measured
      // 8.0% MORE nodes at depth 13 over 18 positions and 0.1% at depth 15. The reason
      // is the window: a probe searches [hi-1, hi], a band ABOVE beta, so when it fails
      // to cut it failed LOW there and its stored move is only the least-bad under a
      // window nobody asked about. Overriding ordering that already gets a first-move
      // cutoff 82.7% of the time with that move makes it worse, not better.
    }

    // Null-move pruning: give the opponent a free pass R+1 plies shallower; if we still
    // hold beta, assume the real move would too and fail high without searching one.
    // MEASURED -264 Elo per node and shipped OFF: the premise (a tempo is an asset) is
    // false in Othello, where mobility strategy routinely makes passing PREFERABLE, so
    // the interior cutoffs are spuriously fired all over the tree with no cheap
    // verification to catch them. Cuts 58% of nodes and loses almost every game. Kept
    // behind the flag as the record; the oracle stays exact because selective_enabled_
    // is off there. `match nmp`.
    if (kUseNmp && nmp_enabled_ && selective_enabled_ && can_null && !solving && beta == alpha + 1 &&
        depth >= kNmpMinDepth && 64 - b.count() > kNmpMinEmpties) {
      const int stand_pat = leaf_eval<Pat>(b, moves, ply, stm);
      if (stand_pat >= beta) {
        const int   red    = kNmpBaseR + depth / kNmpDepthDiv;
        const int   nd     = depth - red - 1 < 1 ? 1 : depth - red - 1;
        const Board passed = b.passed(); // the heuristic pass: mover changes, no square does
        if constexpr (Pat)
          if (ply + 1 < kMaxPly)
            ps_[ply + 1] = ps_[ply]; // a pass leaves every feature unchanged
        const int score =
                -pvs<R, Pat>(passed, nd, -beta, -beta + 1, ply + 1, ~stm, prev_move, prev2_move, /*can_null=*/false);
        if (stopped_)
          return 0;
        if (score >= beta) {
          if constexpr (kStats) { ++sacc.fut_cut; stats_->flush(depth, pattern_stage(b.count()), SearchStats::Cut, sacc); }
          return score; // fail-soft lower bound
        }
      }
    }

    // Futility: at shallow depth, if even a generous swing above the static score
    // cannot reach alpha, searching the moves is unlikely to change the outcome.
    // Never in the PV (a zero-width window means this is a scout node), never
    // while solving, and never when the score is already a proven terminal.
    if (kUsePruning && selective_enabled_ && !solving && depth <= 3 && beta == alpha + 1) {
      if constexpr (kStats) ++sacc.fut_try;
      const int stand_pat = leaf_eval<Pat>(b, moves, ply, stm);
      const int fut_margin = depth <= 1 ? params_.fut1 : (depth == 2 ? params_.fut2 : params_.fut3);
      if (stand_pat + fut_margin <= alpha) {
        if constexpr (kStats) { ++sacc.fut_cut; stats_->flush(depth, pattern_stage(b.count()), SearchStats::All, sacc); }
        return stand_pat;
      }
    }

    // ABDADA: publish that THIS subtree is being searched so other threads defer it,
    // and clear the mark on every exit path via the guard.
    struct BusyGuard {
      TranspositionTable *t = nullptr;
      std::uint64_t       k = 0;
      ~BusyGuard() {
        if (t) t->busy_leave(k);
      }
    } bguard;
    if (abdada_ && depth >= kAbdadaMinDepth) {
      tt_->busy_enter(key);
      bguard.t = tt_.get();
      bguard.k = key;
    }

    // Endgame parity, for move ordering only (never changes a score -- the oracle
    // still holds). Empty squares separated by occupied walls are independent move
    // regions; entering an odd-sized one tends to secure its last move. True
    // connected regions avoid the false merges/splits caused by fixed quadrants.
    Bitboard odd_regions = 0;
    if (kUseParity && endgame_enabled_ && solving) {
      const Bitboard empties = ~(b.player | b.opponent);
      // Region flood-fill pays once occupied walls have fragmented the board.
      // Above 12 empties the old four-quadrant proxy is cheaper and usually
      // equivalent; below it, exact regions materially improve move order.
      if (eg_empties <= 12) {
        odd_regions = odd_empty_regions(empties);
      } else {
        for (unsigned q = 0; q < 4; ++q)
          if (popcount(empties & kQuadrant[q]) & 1)
            odd_regions |= empties & kQuadrant[q];
      }
    }

    // --- build + score the move list -----------------------------------------
    ScoredMove list[36];
    int        n = 0;
    {
      Bitboard m = moves;
      while (m) {
        const Square sq = pop_lsb(m);
        ScoredMove  &sm = list[n++];
        sm.sq           = sq;
        sm.child        = b.play(sq);
        // The child's TT slot is a random DRAM touch (~80-100ns) and the cached
        // perft work showed this table is memory-bound. Start the load now, while
        // ordering still has work to do, so the probe after we recurse is warm.
        if (kUsePrefetch && kUseTT && tt_enabled_)
          tt_->prefetch(hash_board(sm.child.player, sm.child.opponent));
        if constexpr (kUseOrdering) {
          if (sq == tt_move) {
            sm.score = 1 << 24;
          } else {
            const bool learned = tt_move == NOMOVE && depth >= kOrderMobilityMinDepth;
            const int  replies = depth >= kOrderMobilityMinDepth ? popcount(sm.child.moves()) : 0;
            int s = learned ? learned_order_score(b, sm.child, sq, replies, prev_move, prev2_move)
                            : kSquareValue[sq] * params_.sqv_mult;
            if constexpr (kUseKillers) {
              if (sq == killers_[ply][0])
                s += params_.killer0;
              else if (sq == killers_[ply][1])
                s += params_.killer1;
            }
            // Fewer replies for the opponent is the strongest cheap signal, but
            // it costs a get_moves per move -- only worth it deeper up the tree.
            if (!learned && depth >= kOrderMobilityMinDepth)
              s -= params_.mob_w * replies;
            s += history_[sq] / params_.hist_div;
            // The previous placement carries local board context that the global
            // destination-square history cannot distinguish. A pass preserves it.
            if (prev_move < 64)
              s += continuation_history_[prev_move][sq] / params_.hist_div;
            if (prev2_move < 64)
              s += continuation_history_2_[prev2_move][sq] / params_.hist_div;
            if (odd_regions & square_bb(sq))
              s += params_.parity_bonus; // sq's connected region has odd empties
            sm.score = s;
          }
        } else {
          sm.score = 0;
        }
      }
    }

    if constexpr (kStats) {
      ++sacc.order_nodes;
      if (tt_move == NOMOVE)
        ++sacc.order_no_tt;
    }

    int    best       = -kInf;
    Square best_move  = NOMOVE;
    bool   aborted    = false;
    Square searched_moves[36];
    int    searched_count = 0;

    // The move-search body, shared by the two execution orders below. `idx` picks the
    // move from the list; `ord` is its position in the ACTUAL search order, which is
    // what LMR, LMP and the telemetry key on (in the plain path the two are equal).
    // Returns 0 to continue the node, non-zero to end it (cutoff, prune, or abort).
    const auto search_move = [&](int idx, int ord) -> int {
      // Late move pruning: in a scout node at shallow depth, drop the tail of the list
      // unsearched. Ordering is descending, so everything remaining is ordered
      // worse than what already failed to raise alpha. `best > -kInf` keeps at least
      // one move searched so best_move can never come back NOMOVE. Re-tested on v12.
      if (kUseLMP && lmp_enabled_ && kUsePruning && selective_enabled_ && kUseOrdering && !solving &&
          beta == alpha + 1 && depth <= kLMPMaxDepth && ord >= kLMPCount[depth] && best > -kInf)
        return 2;

      const ScoredMove &sm = list[idx];
      if constexpr (kStats) ++sacc.moves_searched;

      // Extension: a forced reply branches nowhere, so spending a ply on it buys
      // nothing -- follow the forced sequence to its end instead. Only ever adds
      // depth, so it cannot corrupt an exact solve.
      int ext = 0;
      if (kUseExtensions && selective_enabled_ && n == 1)
        ext = 1;

      // Late move reduction: with decent ordering, a move this far down the list
      // is probably not best -- scout it shallow and only pay full depth if it
      // surprises us. Heuristic, so: not while solving, not on the first moves,
      // and never below depth 3.
      int red = 0;
      if (kUsePruning && lmr_enabled_ && selective_enabled_ && !solving && !ext && kUseOrdering && depth >= 3 &&
          ord >= 3 && sm.sq != tt_move)
        red = lmr_calibrated_ ? lmr_reduction(depth, ord) : 1 + (ord >= 6 && depth >= 5 ? 1 : 0);
      if constexpr (kStats) if (red) ++sacc.lmr_try;

      // Incremental pattern features for the child: copy the parent's vector and
      // fold in only the squares that changed. `flipped` is recovered from the
      // child rather than recomputed -- play() already did the work.
      if constexpr (Pat) {
        const Bitboard flipped = b.player ^ sm.child.opponent ^ square_bb(sm.sq);
        ps_[ply + 1]           = ps_[ply];
        ps_[ply + 1].update(sm.sq, flipped, stm);
      }

      searched_moves[searched_count++] = sm.sq;

      int s;
      if (ord == 0 || !kUsePVS) {
        s = -pvs<R, Pat>(sm.child, depth - 1 + ext, -beta, -alpha, ply + 1, ~stm, sm.sq, prev_move);
      } else {
        if constexpr (kStats) ++sacc.pvs_scout;
        s = -pvs<R, Pat>(sm.child, depth - 1 + ext - red, -alpha - 1, -alpha, ply + 1, ~stm, sm.sq, prev_move);
        // Calibration sample: was reducing THIS move at THIS (depth, ordinal) a mistake?
        if constexpr (kStats)
          if (red) stats_->lmr_sample(depth, ord, s > alpha);
        // A reduced scout that beats alpha proved nothing -- redo it at full depth
        // before trusting it.
        if (red && s > alpha) {
          if constexpr (kStats) ++sacc.lmr_re;
          s = -pvs<R, Pat>(sm.child, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, ~stm, sm.sq, prev_move);
        }
        // Re-search with the FULL window (-beta,-alpha), not (-beta,-score).
        // `s < beta` is what keeps a zero-width window from ever re-searching.
        if (s > alpha && s < beta) {
          if constexpr (kStats) ++sacc.pvs_re;
          s = -pvs<R, Pat>(sm.child, depth - 1 + ext, -beta, -alpha, ply + 1, ~stm, sm.sq, prev_move);
        }
      }
      if (stopped_) {
        aborted = true; // partial score: the node must not use or store it
        return 2;
      }

      if (s > best) {
        best      = s;
        best_move = sm.sq;
      }
      if (s > alpha)
        alpha = s;
      if (alpha >= beta) { // fail high
        if constexpr (kStats) sacc.cut_idx = ord; // ordinal of the cutoff move
        if constexpr (kStats) {
          if (tt_move == NOMOVE) {
            ++sacc.order_no_tt_fh;
            if (ord == 0)
              ++sacc.order_no_tt_fh_first;
          }
        }
        if constexpr (kUseKillers) {
          if (killers_[ply][0] != sm.sq) {
            killers_[ply][1] = killers_[ply][0];
            killers_[ply][0] = sm.sq;
          }
          const int bonus = depth * depth;
          // Moves that failed before the cutoff are negative evidence for this
          // node's global and continuation contexts. The cutoff move gets the
          // symmetric reward. Passes preserve both previous placement contexts.
          for (int i = 0; i + 1 < searched_count; ++i) {
            update_history(history_, searched_moves[i], -bonus);
            if (prev_move < 64)
              update_history(continuation_history_[prev_move], searched_moves[i], -bonus);
            if (prev2_move < 64)
              update_history(continuation_history_2_[prev2_move], searched_moves[i], -bonus);
          }
          update_history(history_, sm.sq, bonus);
          if (prev_move < 64)
            update_history(continuation_history_[prev_move], sm.sq, bonus);
          if (prev2_move < 64)
            update_history(continuation_history_2_[prev2_move], sm.sq, bonus);
        }
        return 1;
      }
      return 0;
    };

    // ABDADA applies only where deferral can matter: several threads, a subtree deep
    // enough to be worth not duplicating, and more than one move to choose from.
    const bool abdada_here = abdada_ && depth > kAbdadaMinDepth && n > 1;
    if (!abdada_here) {
      for (int i = 0; i < n; ++i) {
        // Lazy selection: a cutoff on the first move must not have paid to sort all.
        if constexpr (kUseOrdering) {
          int pick = i;
          for (int j = i + 1; j < n; ++j)
            if (list[j].score > list[pick].score)
              pick = j;
          if (pick != i)
            std::swap(list[i], list[pick]);
        }
        if (search_move(i, i))
          break;
      }
    } else {
      // Full sort up front: the two-pass order below revisits the list, so lazy
      // selection no longer pays for itself here.
      for (int i = 1; i < n; ++i) {
        ScoredMove key_ = list[i];
        int        j    = i - 1;
        for (; j >= 0 && list[j].score < key_.score; --j)
          list[j + 1] = list[j];
        list[j + 1] = key_;
      }
      bool deferred[36] = {};
      int  ord = 0, ended = 0;
      // Pass 1: search moves whose subtree nobody else is inside. The first move is
      // never deferred -- the node needs a real score to raise alpha with.
      for (int i = 0; i < n && !ended; ++i) {
        if (i > 0 && tt_->busy(hash_board(list[i].child.player, list[i].child.opponent))) {
          deferred[i] = true;
          continue;
        }
        ended = search_move(i, ord++);
      }
      // Pass 2: whatever was deferred. By now the other thread has either finished
      // (its result is in the shared table) or is deep inside (we duplicate, as plain
      // lazy SMP always did) -- either way no move is ever skipped.
      for (int i = 0; i < n && !ended; ++i)
        if (deferred[i])
          ended = search_move(i, ord++);
    }
    if (aborted)
      return 0;

    const std::uint8_t bound = best <= alpha_orig ? kUpper : (best >= beta ? kLower : kExact);
    if constexpr (Pat) {
      if (correction_history_cap_ > 0 && !solving && depth >= kProbCutMinDepth && is_pv && bound == kExact) {
        const int raw = leaf_eval<Pat>(b, moves, ply, stm);
        correction_history_.update(b, pattern_stage(b.count()), prev_move, prev2_move, best, raw, depth);
      }
    }

    if (kUseTT && tt_enabled_) {
      if (!stopped_) { // never poison the table with an aborted node's score
        tt_->store(key, std::clamp(best, -kScoreMax, kScoreMax), depth,
                   bound,
                   best_move >= 0 && best_move < 64 ? static_cast<std::uint8_t>(best_move) : 255);
      }
    }

    // Single node-level flush. Type is the Knuth-Moore trichotomy resolved now that
    // best/beta are known: a full-window node is PV; a scout that got its fail-high
    // is Cut, otherwise it searched everything and is All. (stopped_ nodes returned
    // 0 above and never reach here, so a partial node never pollutes the table.)
    if constexpr (kStats) {
      const SearchStats::NodeType t = is_pv ? SearchStats::PV
                                     : (best >= beta ? SearchStats::Cut : SearchStats::All);
      stats_->flush(depth, pattern_stage(b.count()), t, sacc);
    }
    return best;
  }

  template<Rule R, bool Pat>
  void Searcher::root_search(const Board &root, int depth, Color stm, int alpha, int beta, SearchResult &res) noexcept {
    const Bitboard moves = root.moves(); // caller guarantees non-zero
    // The window is the caller's: a FULL window gives the exact minimax value
    // (search_selftest relies on that), while a narrow ASPIRATION window may fail
    // low/high, and the caller widens and re-searches until the score lands strictly
    // inside -- so the accepted value is still the exact one.

    ScoredMove list[36];
    int        n = 0;
    Bitboard   m = moves;
    while (m) {
      const Square sq = pop_lsb(m);
      list[n]         = {sq, root.play(sq), 0};
      // Previous iteration's best first -- the whole point of iterative deepening.
      if (sq == res.best)
        list[n].score = 1 << 24;
      else
        list[n].score = kSquareValue[sq] * params_.sqv_mult - params_.mob_w * popcount(list[n].child.moves());
      ++n;
    }

    Square best_move  = NOMOVE;
    int    best_score = -kInf;
    for (int i = 0; i < n; ++i) {
      int pick = i;
      for (int j = i + 1; j < n; ++j)
        if (list[j].score > list[pick].score)
          pick = j;
      if (pick != i)
        std::swap(list[i], list[pick]);
      const ScoredMove &sm = list[i];

      if constexpr (Pat) {
        const Bitboard flipped = root.player ^ sm.child.opponent ^ square_bb(sm.sq);
        ps_[1]                 = ps_[0];
        ps_[1].update(sm.sq, flipped, stm);
      }

      int s;
      if (i == 0 || !kUsePVS) {
        s = -pvs<R, Pat>(sm.child, depth - 1, -beta, -alpha, 1, ~stm, sm.sq, NOMOVE);
      } else {
        s = -pvs<R, Pat>(sm.child, depth - 1, -alpha - 1, -alpha, 1, ~stm, sm.sq, NOMOVE);
        if (s > alpha && s < beta)
          s = -pvs<R, Pat>(sm.child, depth - 1, -beta, -alpha, 1, ~stm, sm.sq, NOMOVE);
      }
      if (stopped_)
        return; // leave res holding the last COMPLETED iteration

      if (s > best_score) {
        best_score = s;
        best_move  = sm.sq;
      }
      if (s > alpha)
        alpha = s;
      if (alpha >= beta)
        break; // fail-high: only reachable under a narrow aspiration window
    }
    res.best  = best_move;
    res.score = best_score;
  }

  std::string Searcher::pv_string(Board b, Rule rule, Square first, Color stm, int max_len) {
    // Move case encodes WHO played it: White uppercase (D6), Black lowercase (d6).
    // Board is mover-relative and carries no colour, so `stm` is threaded through
    // and flipped on every ply -- including passes, which also change the mover.
    const auto emit = [&stm](std::string &out, Square sq) {
      std::string s = square_to_string(sq);
      if (stm == Color::White)
        for (char &c: s)
          c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      out += ' ';
      out += s;
      stm = ~stm;
    };

    std::string out;
    // The root is deliberately never stored (root_search takes no TT cutoff, so
    // it writes no entry either) -- so the first move has to be supplied, and the
    // table walk starts from the child.
    if (first == PASS) {
      emit(out, PASS);
      b = b.passed();
    } else if (first >= 0 && first < 64 && (b.moves() & square_bb(first))) {
      emit(out, first);
      b = b.play(first);
    } else {
      return out;
    }
    for (int i = 1; i < max_len; ++i) {
      const Bitboard moves = b.moves();
      if (moves == 0) {
        const Board passed = b.passed();
        if (rule != Rule::Othello || !passed.has_moves())
          break;
        b = passed;
        emit(out, PASS);
        continue;
      }
      const std::uint64_t k = hash_board(b.player, b.opponent);
      Entry e;
      if (!tt_->probe(k, e))
        break; // probe already rejects a miss, a torn pair and an empty slot
      if (e.best >= 64 || !(moves & square_bb(e.best)))
        break; // stale/colliding entry: stop rather than emit an illegal move
      emit(out, static_cast<Square>(e.best));
      b = b.play(static_cast<Square>(e.best));
    }
    return out;
  }

  namespace {
    /**
     * Clock allocation -- how much of MY remaining clock this one move deserves.
     *
     * Othello makes this easier than chess in one important way: the number of moves
     * left is not a guess. Every real move fills exactly one empty square, so the
     * mover has about empties/2 moves left, full stop. The naive reference policy
     * (this engine's own match harness, and fastothello's runner) spends
     * clk/(empties/2) + inc per move -- an even split.
     *
     * Two refinements, both Othello-specific, both cheap:
     *
     *  * SPEND LESS EARLY. An even split prices move 5 like move 35, but the search is
     *    far cheaper early (shallow trees, no solve in sight) and the game is rarely
     *    decided there. Shaving the opening banks Fischer increment for later.
     *
     *  * SPEND MORE AT THE SOLVE WINDOW. Around 20-30 empties one deep think can reach
     *    `depth >= empties` -- the EXACT game result. From that point the engine plays
     *    perfectly at trivial cost (the proof lives in the shared table), so a big
     *    investment there buys the entire remaining game. This is the one place in an
     *    Othello game where time converts directly into truth.
     *
     * SOFT is the target the iterative-deepening loop steers by (it will not start an
     * iteration it cannot roughly half-finish); HARD is the abort line check_stop
     * enforces, a multiple of soft but never a dangerous share of the clock.
     */
    struct TimeBudget {
      double soft_ms, hard_ms;
    };
    constexpr double kMoveOverheadMs = 15.0; // protocol + scheduling latency, charged to us

    /**
     * Adaptive scaling of the SOFT budget, from what iterative deepening just learnt.
     * The allocator above prices a move before searching it; the ID loop knows more.
     * A best move that CHANGED at this depth, or a score that fell hard, is the
     * signature of a position that is still being decided -- the classic case where a
     * little more time changes the move actually played. A best move that has not
     * moved while the score sits still is an obvious move, and its time is better
     * banked (Fischer) for a harder one later. Multiplicative, clamped, and always
     * under the HARD budget, so it can steer but never flag.
     *
     * MEASURED AND REJECTED (tm_adaptive_ stays false). Two versions, both against the
     * fixed allocation at 3000+50ms:
     *   naive (compares to d-1)        -5.4 Elo, 95% CI [-32, 21], 160 pairs
     *   parity-aware (compares to d-2) -8.7 Elo, 95% CI [-31, 13], 240 pairs
     * The parity fix mattered in principle -- an Othello score oscillates across
     * parities by nature, so the naive drop detector fired on noise -- but not in
     * outcome. WHY a chess-proven idea has nothing to collect here: the signals barely
     * fire (the best move is exceptionally stable -- ordering gets a first-move cutoff
     * 82.7% of the time), and Othello's timing-relevant structure is VISIBLE STATICALLY.
     * The move count is exact from the empties and the one genuinely critical moment,
     * the solve window, sits at a known empty count -- and the fixed allocator already
     * prices both. Chess needs dynamic signals because criticality is hidden; here it
     * is on the board.
     */
    constexpr int    kTmAdjustMinDepth = 6;    // below this, iterations are noise
    constexpr int    kTmDropCd         = 120;  // a fall this size counts as instability
    constexpr double kTmBump           = 1.5;  // extension per unstable iteration
    constexpr double kTmDecay          = 0.85; // shrink per calm iteration
    constexpr double kTmMaxFactor      = 2.5;
    constexpr double kTmMinFactor      = 0.6;

    [[nodiscard]] TimeBudget allocate_time(double clk, double inc, int empties) noexcept {
      clk = clk - kMoveOverheadMs;
      if (clk < 1.0)
        return {1.0, 1.0};
      const int    my_moves = empties > 1 ? (empties + 1) / 2 : 1;
      double       soft     = clk / my_moves + 0.9 * inc;
      if (empties >= 44)
        soft *= 0.6; // opening: cheap plies, bank the increment
      else if (empties >= 20 && empties <= 30)
        soft *= 1.8; // solve window: one deep think can end the game exactly
      double hard = std::min(clk * 0.30, soft * 4.0);
      soft        = std::min(soft, hard);
      return {std::max(soft, 1.0), std::max(hard, 1.0)};
    }
  } // namespace

  SearchResult Searcher::search(const Board &root, const SearchLimits &limits, Rule rule, Color stm,
                               std::ostream &info) {
    new_search(limits);
    double tm_soft_base = 0.0, tm_factor = 1.0; // adaptive-TM state; see kTmBump
    if (limits.time_ms > 0.0) { // clock mode: the engine allocates its own budget
      const TimeBudget tb = allocate_time(limits.time_ms, limits.inc_ms, 64 - root.count());
      deadline_ms_        = start_ms_ + tb.hard_ms;
      soft_ms_            = start_ms_ + tb.soft_ms;
      tm_soft_base        = tb.soft_ms;
    }
    if (pat_on_)
      ps_[0].set(root, stm); // the only from-scratch build; everything below is incremental
    SearchResult res;

    const Bitboard moves = root.moves();
    if (moves == 0) {
      const Board passed = root.passed();
      if (rule == Rule::Othello && passed.has_moves()) {
        res.best  = PASS; // forced: no choice to search for
        if (pat_on_)
          ps_[1] = ps_[0]; // a pass changes the mover, not the squares
        const int sc = rule == Rule::Othello
                               ? (pat_on_ ? pvs<Rule::Othello, true>(passed, 4, -kInf, kInf, 1, ~stm)
                                          : pvs<Rule::Othello, false>(passed, 4, -kInf, kInf, 1, ~stm))
                               : (pat_on_ ? pvs<Rule::Reversi, true>(passed, 4, -kInf, kInf, 1, ~stm)
                                          : pvs<Rule::Reversi, false>(passed, 4, -kInf, kInf, 1, ~stm));
        res.score = -sc;
        return res;
      }
      res.best  = NOMOVE; // game over
      res.score = terminal_score(root);
      return res;
    }

    const int empties   = 64 - root.count();
    // Past `empties` every iteration re-searches an identical tree, and reaching
    // it means the score is exact -- so it is both the cap and the stop.
    const int max_depth = limits.depth > 0 ? std::min(limits.depth, empties) : empties;

    int  score_hist[2] = {0, 0}; // score two and one iterations ago (index by parity of d)
    bool have_hist[2]  = {false, false};
    // Adaptive TM keeps its previous best/score PER PARITY of the depth. Comparing to
    // the immediately previous iteration would compare across opposite parities, and in
    // Othello the score oscillates with parity BY NATURE -- the same fact that makes
    // ProbCut predict across d-2 and sank naive aspiration. A drop detector fed with
    // that oscillation fires on noise every other iteration.
    Square tm_prev_best[2]  = {NOMOVE, NOMOVE};
    int    tm_prev_score[2] = {0, 0};
    bool   tm_have_prev[2]  = {false, false};
    // Helper threads start a little deeper so they are not all grinding the identical
    // iteration at the same instant; the shared table keeps them diverged after that.
    const int first_depth = depth_offset_ < max_depth ? 1 + depth_offset_ : 1;
    for (int d = first_depth; d <= max_depth; ++d) {
      seldepth_       = 0; // per-iteration, like the UCI convention
      SearchResult it = res; // carry the previous best in for root ordering

      // Aspiration centred on the SAME-parity score (d-2). Widen x3 on a fail until
      // the score is strictly inside; the final value is the exact full-window one.
      const bool aspire = kUseAspiration && aspiration_enabled_ && d >= kAspMinDepth && have_hist[d & 1];
      const int  centre = score_hist[d & 1];
      int        window = kAspWindow;
      int        alpha  = aspire ? centre - window : -kInf;
      int        beta   = aspire ? centre + window : kInf;
      // WIN/LOSS/DRAW solve: at the solving iteration a +-100 window answers win/loss/draw
      // without the exact margin (diffs are even: win fails high past +100, loss past -100,
      // inside = draw). ~7x fewer nodes, so affordable ~2 empties earlier. Timed only (`go
      // depth` analysis wants the true margin). Off: measured +4.7 [-5,14], the saving banks
      // as idle Fischer time in already-decided positions. `match wldtc`.
      bool wld = wld_enabled_ && (limits.movetime_ms > 0.0 || limits.time_ms > 0.0) && d >= empties;
      if (wld) {
        alpha = -100;
        beta  = 100;
      }
      for (;;) {
        if (rule == Rule::Othello) {
          if (pat_on_)
            root_search<Rule::Othello, true>(root, d, stm, alpha, beta, it);
          else
            root_search<Rule::Othello, false>(root, d, stm, alpha, beta, it);
        } else {
          if (pat_on_)
            root_search<Rule::Reversi, true>(root, d, stm, alpha, beta, it);
          else
            root_search<Rule::Reversi, false>(root, d, stm, alpha, beta, it);
        }
        if (stopped_)
          break;
        if (wld) {
          // Win/draw: play the bound move as-is. Loss: the fail-low bounds barely rank
          // moves and the opponent is not solved either, so re-search full-window for the
          // least-bad margin (measured -15 without this). Only losses pay the re-search.
          if (it.score <= -100 && !stopped_) {
            wld   = false;
            alpha = -kInf;
            beta  = kInf;
            continue;
          }
          break;
        }
        if (it.score <= alpha && alpha > -kInf) {
          window *= 3;
          alpha = it.score - window <= -kScoreMax ? -kInf : it.score - window;
          continue; // failed low: relax the lower bound and retry
        }
        if (it.score >= beta && beta < kInf) {
          window *= 3;
          beta = it.score + window >= kScoreMax ? kInf : it.score + window;
          continue; // failed high
        }
        break; // strictly inside -> exact
      }

      // Iteration 1 is never abandoned: bestmove must never be empty.
      if (stopped_ && d > 1)
        break;
      score_hist[d & 1] = it.score; // remember this parity's score for d+2
      have_hist[d & 1]  = true;

      res          = it;
      res.depth    = d;
      res.seldepth = seldepth_;
      res.nodes    = nodes_;
      res.exact    = (d >= empties);

      const double        dt   = now_ms() - start_ms_;
      const std::uint64_t nps  = dt > 0.0 ? static_cast<std::uint64_t>(nodes_ / (dt / 1000.0)) : 0;
      // hashfull is per-mille of the table occupied (UCI convention).
      const int hashfull = tt_->hashfull();
      // `cd` = centi-discs, not centipawns: this is Othello, the unit is discs.
      info << "info depth " << d << " seldepth " << res.seldepth << " score cd " << res.score << " nodes " << nodes_
           << " nps " << nps << " hashfull " << hashfull << " time " << static_cast<std::uint64_t>(dt) << " pv"
           << pv_string(root, rule, res.best, stm, d) << '\n';
      info.flush();

      if (res.exact || stopped_)
        break;
      // Adaptive TM: re-price the soft budget from what this iteration just revealed.
      // A changed best move or a hard score fall extends it; a still one shrinks it.
      // Clamped under the hard budget, so this can steer but never flag.
      if (tm_adaptive_ && tm_soft_base > 0.0 && d >= kTmAdjustMinDepth && tm_have_prev[d & 1]) {
        const bool unstable = (res.best != tm_prev_best[d & 1]) ||
                              (res.score <= tm_prev_score[d & 1] - kTmDropCd);
        tm_factor = unstable ? std::min(tm_factor * kTmBump, kTmMaxFactor)
                             : std::max(tm_factor * kTmDecay, kTmMinFactor);
        soft_ms_  = start_ms_ + std::min(deadline_ms_ - start_ms_, tm_soft_base * tm_factor);
      }
      tm_prev_best[d & 1]  = res.best;
      tm_prev_score[d & 1] = res.score;
      tm_have_prev[d & 1]  = true;
      // Don't start an iteration there is no hope of finishing (Othello's
      // effective branching is ~3-5, so the next one costs several times this).
      // Under a clock the SOFT budget steers this; the hard deadline stays as the
      // abort line for an iteration that was started and misbehaved.
      const double steer = soft_ms_ > 0.0 ? soft_ms_ : deadline_ms_;
      if (steer > 0.0 && dt > 0.5 * (steer - start_ms_))
        break;
    }
    if constexpr (kStats)
      stats_->total_nodes = nodes_; // for the probe-cost ratio (same unit: incl. leaves)
    return res;
  }

  bool search_selftest() noexcept {
    {
      CorrectionHistory h;
      const Board       b     = Board::start();
      const int         stage = pattern_stage(b.count());
      if (h.predict(b, stage, 19, 26) != 0)
        return false;
      h.update(b, stage, 19, 26, 600, 100, 8);
      const int learned = h.predict(b, stage, 19, 26);
      if (learned <= 0 || h.predict(b, stage, 20, 26) >= learned)
        return false;
      for (int i = 0; i < 256; ++i)
        h.update(b, stage, 19, 26, 5000, 0, 16);
      if (h.predict(b, stage, 19, 26) > 400)
        return false;
      h.clear();
      if (h.predict(b, stage, 19, 26) != 0)
        return false;
    }

    // Connected-region parity: an odd component is marked in full, an even one
    // is omitted, and rank edges must not wrap h-file into the next a-file.
    {
      const Bitboard odd_three = square_bb(0) | square_bb(1) | square_bb(2);
      const Bitboard even_two  = square_bb(62) | square_bb(63);
      if (odd_empty_regions(odd_three | even_two) != odd_three)
        return false;
      const Bitboard no_wrap = square_bb(7) | square_bb(8);
      if (odd_empty_regions(no_wrap) != no_wrap)
        return false;
    }

    std::uint64_t s   = 0x9E3779B97F4A7C15ULL;
    const auto    rnd = [&s]() noexcept {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      return s;
    };

    Searcher           sr(16);
    std::ostringstream sink;

    // ---- Part 1: alpha-beta / ordering / PVS / killers are score-preserving --
    // The table is OFF here, and that is not a dodge: with it on, `e.depth >=
    // depth` lets a node answer with a value proved by a DEEPER search, so the
    // result legitimately differs from a fixed-depth minimax (search
    // instability). Off, the search must reproduce the oracle exactly.
    sr.set_tt_enabled(false);
    sr.set_selective_enabled(false); // group B changes the fixed-depth value by design
    for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
      Board b = Board::start();
      for (int step = 0; step < 220; ++step) {
        Bitboard m = b.moves(); // sample real positions via random playout
        if (m == 0) {
          const Board p = b.passed();
          b = p.has_moves() ? p : Board::start();
          continue;
        }
        for (int d = 1; d <= 4; ++d) {
          const int want = rule == Rule::Othello ? ref_negamax<Rule::Othello>(b, d, Color::Black) : ref_negamax<Rule::Reversi>(b, d, Color::Black);
          const SearchResult got = sr.search(b, SearchLimits{d, 0, 0.0}, rule, Color::Black, sink);
          if (got.score != want)
            return false; // an "optimization" changed the score: that is a bug
          if (got.best == NOMOVE || (got.best != PASS && !(m & square_bb(got.best))))
            return false; // the move played must always be legal
        }
        unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
        while (k-- > 0)
          m &= m - 1;
        b = b.play(lsb(m));
      }
    }

    // ---- Part 1b: the search's INCREMENTAL pattern features are correct ------
    // Without weights the pattern eval is off, so part 1 never exercises the
    // incremental path. Install pseudo-random weights so the eval actually varies,
    // then re-run the oracle comparison: the oracle rebuilds features from
    // scratch at every leaf while the search folds them in move by move, so any
    // disagreement in PatternState::update or in the ply stack shows up as a
    // score mismatch.
    {
      PatternWeights saved = pattern_weights();
      pattern_weights().reset_zero();
      for (std::size_t i = 0; i < pattern_weights().size(); ++i)
        pattern_weights().data()[i] = static_cast<std::int16_t>(static_cast<int>(rnd() % 41) - 20);

      bool ok = true;
      for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
        Board b = Board::start();
        for (int step = 0; step < 60 && ok; ++step) {
          Bitboard m = b.moves();
          if (m == 0) {
            const Board p = b.passed();
            b = p.has_moves() ? p : Board::start();
            continue;
          }
          // stm alternates with the playout so the colour-absolute encoding is
          // exercised from both sides.
          const Color stm = (step & 1) ? Color::White : Color::Black;
          for (int d = 1; d <= 3; ++d) {
            const int want = rule == Rule::Othello ? ref_negamax<Rule::Othello>(b, d, stm)
                                                   : ref_negamax<Rule::Reversi>(b, d, stm);
            const SearchResult got = sr.search(b, SearchLimits{d, 0, 0.0}, rule, stm, sink);
            if (got.score != want) {
              ok = false;
              break;
            }
          }
          unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
          while (k-- > 0)
            m &= m - 1;
          b = b.play(lsb(m));
        }
      }
      pattern_weights() = saved; // never leave test weights behind
      if (!ok)
        return false;
    }

    // ---- Part 2: the table is sound on exact solves -------------------------
    // `depth - empties` is invariant down a line (a move spends one of each), so
    // once the ROOT has depth >= empties every node in the tree is solved
    // exactly. Every stored score is then the true game result, depth mixing is
    // harmless, and the table must not change the answer.
    sr.set_tt_enabled(true);
    sr.set_selective_enabled(true); // extensions/pruning must NOT corrupt an exact solve
    for (const Rule rule: {Rule::Othello, Rule::Reversi}) {
      sr.clear(); // the two rules value stuck positions differently: never share
      for (int trial = 0; trial < 12; ++trial) {
        Board b = Board::start();
        for (int i = 0; i < 52 && 64 - b.count() > 9; ++i) { // play down to ~9 empties
          Bitboard m = b.moves();
          if (m == 0) {
            const Board p = b.passed();
            if (!p.has_moves())
              break;
            b = p;
            continue;
          }
          unsigned k = static_cast<unsigned>(rnd() % static_cast<unsigned>(popcount(m)));
          while (k-- > 0)
            m &= m - 1;
          b = b.play(lsb(m));
        }
        const int empties = 64 - b.count();
        if (b.moves() == 0 || empties > 12)
          continue;
        const int want =
                rule == Rule::Othello ? ref_negamax<Rule::Othello>(b, empties, Color::Black)
                                      : ref_negamax<Rule::Reversi>(b, empties, Color::Black);
        const SearchResult got = sr.search(b, SearchLimits{empties, 0, 0.0}, rule, Color::Black, sink);
        if (got.score != want || !got.exact)
          return false;
        // Warm table, same query: must be identical (and it will be a TT hit).
        const SearchResult again = sr.search(b, SearchLimits{empties, 0, 0.0}, rule, Color::Black, sink);
        if (again.score != want)
          return false;
      }
    }
    return true;
  }

} // namespace islay
