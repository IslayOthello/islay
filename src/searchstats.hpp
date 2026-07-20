/**
 * @file searchstats.hpp
 * @brief Opt-in search telemetry: WHERE selectivity saves and WHERE it pays.
 *
 * Compiled OUT by default (`kStats` in search.cpp is a namespace constexpr, like
 * the kUse* bisection switches). Flip it on, rebuild, `go depth N`, read the dump.
 * Every counter is pure observation -- it reads state and increments, never touches
 * a score -- so the oracle (search_selftest) is unaffected and stays byte-identical.
 *
 * WHAT IT ANSWERS. Before this, the only feedback about a pruning technique was the
 * total node count and the match Elo -- a single number that says "cheaper" or
 * "stronger" but never WHERE. This buckets every INTERIOR decision node by
 * (remaining depth x game stage x node type) and records, per bucket:
 *   * TT: probe / hit / cutoff rate -- is the table actually answering here?
 *   * ordering quality: fail-high-on-first rate and the mean move index at a cutoff
 *     (a low mean = good ordering = PVS/LMR/ProbCut are all safer).
 *   * LMR: reductions applied vs reduced-then-re-searched (the wasted-work rate).
 *   * futility / ProbCut: attempts, cuts, and the NODES spent on ProbCut probes
 *     (the probe is the cost that has to be paid back by the cuts it buys).
 *   * PVS: scout searches vs full-window re-searches (the re-search rate).
 *   * effective branching factor (moves searched per node).
 *
 * NODE TYPE is the Knuth-Moore trichotomy, resolved at the node's single logical
 * exit: a node entered with a full window is PV; a null-window node that ends
 * >= beta is Cut (it got its fail-high), otherwise All (it searched everything and
 * failed low). Leaves, terminals, passes and the endgame-solver handoff are NOT
 * counted -- they exercise no selectivity, so folding them in would only dilute the
 * table. `nodes` here therefore means "interior pvs invocations", not `nodes_`.
 */
#ifndef ISLAY_SEARCHSTATS_HPP
#define ISLAY_SEARCHSTATS_HPP

#include <cstdint>
#include <ostream>

#include "pattern.hpp" // kStageCount

namespace islay {

  struct SearchStats {
    static constexpr int kD = 32;          // remaining depth, clamped
    static constexpr int kS = kStageCount; // game stage (15)
    static constexpr int kT = 3;           // node type
    enum NodeType { PV = 0, Cut = 1, All = 2 };

    // One (depth, stage, type) bucket. All monotone counters, summed at flush.
    struct Cell {
      std::uint64_t nodes = 0;
      std::uint64_t tt_probe = 0, tt_hit = 0, tt_cut = 0;
      std::uint64_t fh = 0, fh_first = 0, cut_idx_sum = 0; // fail-high accounting / ordering
      std::uint64_t moves_searched = 0;                    // for the branching factor
      std::uint64_t lmr_try = 0, lmr_re = 0;
      std::uint64_t fut_try = 0, fut_cut = 0;
      std::uint64_t pc_try = 0, pc_cut = 0, pc_probe_nodes = 0;
      // The lo-probe (prove <= alpha) separately from the hi-probe, because they are
      // NOT symmetric in cost: hi runs first and returns on success, so lo is only ever
      // paid on nodes where hi already missed. hi_cuts = pc_cut - pc_lo_cut.
      std::uint64_t pc_lo_try = 0, pc_lo_cut = 0, pc_lo_nodes = 0;
      std::uint64_t pvs_scout = 0, pvs_re = 0;

      void add(const Cell &o) noexcept {
        nodes += o.nodes;
        tt_probe += o.tt_probe; tt_hit += o.tt_hit; tt_cut += o.tt_cut;
        fh += o.fh; fh_first += o.fh_first; cut_idx_sum += o.cut_idx_sum;
        moves_searched += o.moves_searched;
        lmr_try += o.lmr_try; lmr_re += o.lmr_re;
        fut_try += o.fut_try; fut_cut += o.fut_cut;
        pc_try += o.pc_try; pc_cut += o.pc_cut; pc_probe_nodes += o.pc_probe_nodes;
        pc_lo_try += o.pc_lo_try; pc_lo_cut += o.pc_lo_cut; pc_lo_nodes += o.pc_lo_nodes;
        pvs_scout += o.pvs_scout; pvs_re += o.pvs_re;
      }
    };

    Cell cell[kD][kS][kT];

    // Total pvs nodes of the search (nodes_, incl. leaves) -- the honest denominator
    // for the ProbCut probe-cost ratio, whose numerator also counts probe leaves.
    std::uint64_t total_nodes = 0;

    /**
     * LMR calibration profile, the input to a fitted reduction table.
     *
     * For every reduced move: how often did the reduced scout come back ABOVE alpha,
     * i.e. how often was reducing this move a mistake we had to pay to undo? Bucketed
     * by remaining depth and by the move's ordinal, because that is exactly the pair a
     * reduction formula is a function of. A low re-search rate in a bucket means the
     * reduction there is too timid and can be deepened; a high one means it is already
     * past the point where it pays.
     *
     * Recorded directly rather than through Acc: this is per-MOVE, not per-node.
     */
    static constexpr int kIdx = 36; // max legal moves in Othello
    std::uint64_t        lmr_try[kD][kIdx];
    std::uint64_t        lmr_re[kD][kIdx];

    void lmr_sample(int depth, int idx, bool researched) noexcept {
      const int d = depth < 0 ? 0 : (depth >= kD ? kD - 1 : depth);
      const int i = idx < 0 ? 0 : (idx >= kIdx ? kIdx - 1 : idx);
      ++lmr_try[d][i];
      if (researched) ++lmr_re[d][i];
    }

    /**
     * Per-node accumulator. One lives on the stack of a single pvs() invocation and
     * is flushed once, at the node's exit, into the bucket its resolved type selects.
     * A per-node scratch (not direct global writes) keeps the counters correct across
     * the several early returns and lets the type be decided after the fact.
     */
    struct Acc {
      std::uint32_t tt_probe = 0, tt_hit = 0, tt_cut = 0;
      std::uint32_t lmr_try = 0, lmr_re = 0;
      std::uint32_t fut_try = 0, fut_cut = 0;
      std::uint32_t pc_try = 0, pc_cut = 0;
      std::uint32_t pc_lo_try = 0, pc_lo_cut = 0;
      std::uint64_t pc_probe_nodes = 0, pc_lo_nodes = 0;
      std::uint32_t pvs_scout = 0, pvs_re = 0;
      std::uint32_t moves_searched = 0;
      int           cut_idx = -1; // move index that produced the fail-high, -1 = none
    };

    void reset() noexcept { *this = SearchStats{}; }

    void flush(int depth, int stage, NodeType t, const Acc &a) noexcept {
      const int d = depth < 0 ? 0 : (depth >= kD ? kD - 1 : depth);
      const int s = stage < 0 ? 0 : (stage >= kS ? kS - 1 : stage);
      Cell &c = cell[d][s][t];
      ++c.nodes;
      c.tt_probe += a.tt_probe; c.tt_hit += a.tt_hit; c.tt_cut += a.tt_cut;
      c.moves_searched += a.moves_searched;
      c.lmr_try += a.lmr_try; c.lmr_re += a.lmr_re;
      c.fut_try += a.fut_try; c.fut_cut += a.fut_cut;
      c.pc_try += a.pc_try; c.pc_cut += a.pc_cut; c.pc_probe_nodes += a.pc_probe_nodes;
      c.pc_lo_try += a.pc_lo_try; c.pc_lo_cut += a.pc_lo_cut; c.pc_lo_nodes += a.pc_lo_nodes;
      c.pvs_scout += a.pvs_scout; c.pvs_re += a.pvs_re;
      if (a.cut_idx >= 0) {
        ++c.fh;
        c.cut_idx_sum += static_cast<std::uint64_t>(a.cut_idx);
        if (a.cut_idx == 0) ++c.fh_first;
      }
    }

    // Human-readable roll-ups. Defined in search.cpp (uses <iomanip>); `full` also
    // prints the raw (depth x stage) grid for the busiest stages.
    void dump(std::ostream &o, bool full) const;
  };

} // namespace islay

#endif // ISLAY_SEARCHSTATS_HPP
