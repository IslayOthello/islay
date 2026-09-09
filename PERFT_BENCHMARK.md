# Single-thread perft experiments — 2026-09-09

## Retained change

The retained implementation probes the transposition table before generating
legal moves. It uses raw board keys at remaining depth 3 and D4-canonical keys
at remaining depth 4 and above. The cache threshold remains 3; full board keys
and depth are still checked on every hit. Table size, entry layout, rules and
node counts are unchanged. No perft threading was added.

Measured through the optimized Release engine's actual UCI `go perft` command,
the median paired NPS gain is **+2.38%** across
18 positions / 35 position-rule cases. The geometric mean gain is
**+2.38%**, with a round-bootstrap 95% interval
[+1.46%, +2.90%].
This is a modest speed improvement on this machine and workload.

## Environment and method

- Baseline commit: `4e7b75cda6b197117bf0382d0e357a9cb693b0d8`.
- CPU: Apple M3 Pro; one perft thread and one active benchmark process at a time.
- Compiler: Apple Clang 21.0.0 (clang-2100.1.1.101), C++20.
- Release/native/ThinLTO: `-O3 -DNDEBUG -march=native -flto=thin`.
- Default ARM integer/NEON hybrid backend; only the full-NEON trial changes it.
  AVX2/AVX-512 batching is not applicable on this ARM machine.
- `PerftHash=256` MiB for all performance comparisons. Each candidate has the
  same capacity budget. Correctness checks also use a 1 MiB table.
- Corpus seed: `20260909`. The screening set has 10 legal positions: opening,
  middlegame, endgame, and a forced-pass position with 24 empties.
  Confirmation adds 8 positions from two other seeded games, unseen during screening.
- Both Othello and Reversi are measured. The zero-node Reversi forced-pass root
  is covered by correctness tests but excluded from NPS statistics.
- Four paired A/B rounds per case for screening; eight for confirmation.
  A/B order reverses each round; position order is deterministically shuffled.
  Builds finish before timing begins. Normal macOS background activity remains;
  there is no CPU affinity pinning.
- The A/A control uses byte-identical baseline binaries: median paired gain
  +0.13%, geometric mean +0.19%; individual-case medians range from -5.13% to +4.05%.

### Timing boundaries

The standalone driver matches UCI root-divide behavior, including passes.
Cold-TT runs clear the table before each traversal, outside the timer; this is
an empty logical table, not a flush of CPU caches. Warm-TT runs prime the table
once and repeatedly count the same position. `nocache` bypasses the table.
Allocation, table clears and process startup are excluded from traversal NPS.

Screening samples accumulate at least 80 ms of traversal time. Cold confirmation
samples accumulate at least 120 ms; warm/nocache samples at least 80 ms.
Warm calls are batched in groups of 1024. All three paths use eight confirmation
rounds on the expanded corpus.

The final UCI check uses one discarded warmup and four measured traversals per
sample, with `ucinewgame` clearing the TT before each. It derives elapsed time
from the emitted `Speed`, not the integer-rounded `Time` field. This validates
the actual shipped command path independently of the standalone driver.

NPS means returned perft leaves / elapsed seconds, not physically visited nodes.
In particular, warm-cache NPS reflects memoized result retrieval and must not be
interpreted as fresh-tree traversal throughput.

For each position/rule, the reported percentage is the median of eight paired
`candidate_nps / baseline_nps` ratios (four in screening). The suite median and
geometric mean give each position/rule equal weight. Confidence intervals resample
whole rounds, preserving timing drift shared across positions. They describe
repeatability on this fixed suite, not all possible Othello positions.

## Screening results: cold TT, 19 position-rule cases

| Candidate | Median paired gain | Geometric mean | Per-case range | Decision |
|---|---:|---:|---:|---|
| Probe TT before move generation | -0.27% | -0.36% | -5.52% to +6.13% | Cold-neutral; evaluate in combination |
| Cache only at depth >= 4 | -5.46% | -4.96% | -19.88% to +24.13% | Reject regression |
| Cache only at depth >= 5 | -13.76% | -16.43% | -30.20% to +1.90% | Reject regression |
| Raw board keys at all cached depths | +2.46% | -10.54% | -75.04% to +13.56% | Reject severe position-specific regressions |
| Two entries per bucket; replace shallower | +0.04% | -0.27% | -4.73% to +2.80% | No demonstrated standalone gain |
| One entry; preserve deeper entries | -0.39% | -0.12% | -2.26% to +4.05% | No demonstrated standalone gain |
| Raw keys at depth 3; canonical at depth >= 4 | +1.39% | +2.21% | -7.20% to +10.47% | Confirm on expanded corpus |
| Full NEON move generation | -19.44% | -17.66% | -23.25% to -7.90% | Reject regression |
| PGO on baseline, separate training positions | +0.22% | +0.21% | -4.56% to +5.24% | No demonstrated standalone gain |

The raw-key-only variant illustrates why the suite median is insufficient:
its median improves, but loss of symmetry sharing causes approximately 75%
regressions in the start-position cases. It is not retained.

PGO was trained on 13 separately seeded legal positions (`20260910`), including
a depth-11 opening case so deeper template specializations receive profile data.
It provided no clear gain and is not enabled in the default build. The unused
`board.cpp` parsing/printing functions produced an unprofiled-file warning;
the measured driver does not call them.

## Confirmation of the retained combination

| Path | Cases | Rounds | Median paired gain | Geometric mean | Geometric-mean 95% interval |
|---|---:|---:|---:|---:|---|
| Standalone driver, cold TT | 35 | 8 | +1.43% | +2.34% | +1.71% to +2.66% |
| Standalone driver, warm TT | 35 | 8 | +24.37% | +24.15% | +23.64% to +24.75% |
| Standalone driver, nocache | 35 | 8 | +0.09% | +0.44% | -0.37% to +1.70% |
| Final Release UCI, cold TT | 35 | 8 | +2.38% | +2.38% | +1.46% to +2.90% |

On the eight additional positions alone, the combination's standalone cold-TT
median gain is +2.18%, geometric mean +2.56%. Canonical-4 without the early
probe is also positive (+1.32% median / +2.02% geometric mean on the full suite),
but the combination additionally improves warm-cache retrieval. The unchanged
uncached algorithm shows no clear speed change.

### Final UCI median NPS by position and rule

Values below are millions of returned leaves per second. The two absolute NPS
columns are separate medians. The final column is the median of paired ratios,
so it need not equal the ratio of those two medians when timings drift between
rounds; paired ratios are the comparison used for decisions.

| Position | Depth | Rule | Baseline median MNPS | Candidate median MNPS | Median paired gain |
|---|---:|---|---:|---:|---:|
| start | 11 | Othello | 4324.74 | 4441.61 | +2.26% |
| start | 11 | Reversi | 4490.90 | 4500.55 | +1.00% |
| g0_discs12 | 8 | Othello | 1317.93 | 1346.57 | +0.46% |
| g0_discs12 | 8 | Reversi | 1362.57 | 1378.10 | +1.14% |
| g0_discs24 | 7 | Othello | 2028.51 | 2031.32 | +0.81% |
| g0_discs24 | 7 | Reversi | 2074.47 | 2108.85 | +2.14% |
| g0_discs40 | 7 | Othello | 1295.71 | 1326.68 | +2.39% |
| g0_discs40 | 7 | Reversi | 1362.78 | 1372.32 | +1.79% |
| g0_discs52 | 9 | Othello | 352.81 | 373.97 | +5.72% |
| g0_discs52 | 9 | Reversi | 357.55 | 388.48 | +8.90% |
| g1_discs12 | 8 | Othello | 1422.46 | 1417.63 | -0.03% |
| g1_discs12 | 8 | Reversi | 1434.47 | 1428.13 | -0.92% |
| g1_discs24 | 7 | Othello | 2522.18 | 2555.09 | +3.12% |
| g1_discs24 | 7 | Reversi | 2608.94 | 2677.84 | +1.92% |
| g1_discs40 | 7 | Othello | 1560.25 | 1584.14 | +0.44% |
| g1_discs40 | 7 | Reversi | 1616.08 | 1648.72 | +2.77% |
| g1_discs52 | 9 | Othello | 233.72 | 229.72 | +3.05% |
| g1_discs52 | 9 | Reversi | 245.21 | 253.14 | +3.89% |
| forced_pass | 9 | Othello | 775.58 | 777.21 | +0.68% |
| g2_discs12 | 8 | Othello | 1263.69 | 1293.21 | +3.08% |
| g2_discs12 | 8 | Reversi | 1353.02 | 1326.01 | +1.44% |
| g2_discs24 | 7 | Othello | 1339.14 | 1353.71 | +3.60% |
| g2_discs24 | 7 | Reversi | 1371.53 | 1383.22 | -0.00% |
| g2_discs40 | 7 | Othello | 1124.91 | 1134.78 | +2.72% |
| g2_discs40 | 7 | Reversi | 1169.50 | 1205.43 | +2.73% |
| g2_discs52 | 9 | Othello | 232.43 | 247.35 | +4.37% |
| g2_discs52 | 9 | Reversi | 244.89 | 261.36 | +2.38% |
| g3_discs12 | 8 | Othello | 1514.27 | 1525.47 | +0.35% |
| g3_discs12 | 8 | Reversi | 1494.49 | 1587.78 | +4.67% |
| g3_discs24 | 7 | Othello | 1633.85 | 1628.63 | -0.43% |
| g3_discs24 | 7 | Reversi | 1688.11 | 1673.01 | -0.35% |
| g3_discs40 | 7 | Othello | 1048.31 | 1101.66 | +4.29% |
| g3_discs40 | 7 | Reversi | 1080.12 | 1120.79 | +2.56% |
| g3_discs52 | 9 | Othello | 325.32 | 345.06 | +5.50% |
| g3_discs52 | 9 | Reversi | 323.52 | 335.73 | +5.61% |

Position names `gN_discsD` identify game N and total occupied squares D.
Games 2 and 3 are the additional confirmation positions.

## Correctness and retained regression tests

Every experimental binary passed 4,112 checks:

- 28 positions checked against an independent square-scanning scalar oracle.
- Both rules and depths 0–3; known start-position counts through depth 8.
- Near-terminal depth 4/5 cases and depth 13 to exercise the generic dispatcher.
- Eight symmetries, cached/uncached comparisons, forced pass, empty/full boards.
- A 1 MiB TT, including reuse across symmetry variants.
- All deeper benchmark counts matched the baseline for every A/B sample.

The aggregate engine self-test now additionally contains nine fixed legal
positions at depth 5, both rules, all eight symmetries, and cached/uncached paths
with a 1 MiB TT (288 additional comparisons). Counts were frozen from the
baseline uncached implementation. Both final Release/native/LTO and
Debug/non-native/no-LTO builds ended with `ALL TESTS PASSED`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
printf 'debug on\ntest\nquit\n' | ./build/islay

cmake -S . -B _build/perft-debug -DCMAKE_BUILD_TYPE=Debug \
  -DISLAY_NATIVE=OFF -DISLAY_LTO=OFF
cmake --build _build/perft-debug -j 2
printf 'debug on\ntest\nquit\n' | ./_build/perft-debug/islay
```

## Reproduction and artifacts

`tools/perft_bench.cpp` is the single-thread driver.
`tools/perft_study.py` freezes baseline sources, generates the seeded corpus,
builds and verifies candidates, runs paired timings, and writes JSONL journals.
`tools/perft_variants/*.patch` records every source experiment; these patches
were verified to reproduce the measured source files byte for byte.

Use a fresh artifact directory; snapshot initialization refuses to overwrite it:

```sh
python3 tools/perft_study.py snapshot --directory _build/perft-replay
python3 tools/perft_study.py prepare --directory _build/perft-replay
python3 tools/perft_study.py build baseline baseline_control probe_first \
  threshold4 threshold5 raw_key bucket2 depth_preferred canonical4 combined full_neon \
  --directory _build/perft-replay
python3 tools/perft_study.py verify baseline probe_first threshold4 threshold5 \
  raw_key bucket2 depth_preferred canonical4 combined full_neon \
  --directory _build/perft-replay
python3 tools/perft_study.py run combined --rounds 8 --milliseconds 120 \
  --modes cold --include-holdout --tag confirmation --directory _build/perft-replay
python3 tools/perft_study.py run combined --rounds 8 --milliseconds 80 \
  --modes warm nocache --include-holdout --tag paths --directory _build/perft-replay
python3 tools/perft_study.py report --directory _build/perft-replay
```

PGO reproduction uses `build pgo-instrument`, `train-pgo`, and `build pgo`
in that order. The `--uci` runner expects frozen baseline and candidate engine
executables at `DIRECTORY/baseline/islay` and `DIRECTORY/combined/islay`.

Local raw artifacts for this run remain under `_build/perft-study/` (Git-ignored):

- `corpus.json`, `environment.json`, `summary.json`.
- `results-control.jsonl`, `results-screening.jsonl`,
  `results-confirmation.jsonl`, `results-paths.jsonl`, `results-production.jsonl`.
- Frozen sources/binaries, per-variant build commands and verification logs,
  separate PGO training corpus and profiles.

| Artifact | SHA-256 |
|---|---|
| Corpus | `ce11d5f4251b7d1761d48cde7a38e20dc29fa690c5f41adb90f140881bcd04c8` |
| Baseline driver | `8e1669e246fb253b1b3c8eda083021a846c7c4123a4aa31578b52ea1f265dd27` |
| Candidate driver | `768061f45cb3803586a4b8f7b8398f03a2c5de1c0b11499d94446a42840af55c` |
| Baseline UCI executable | `f0a0ba8c0d7ca38a551ee98bbb9ef812c53312daf7496e82de87ef9de4969e85` |
| Measured candidate UCI executable | `49dd17eb99a247c88ac9be29fccdd2a69c3707856cd7e3cb7f469836c0a6c29a` |

These measurements and executable hashes predate the display-only addition of
thousands separators to `Nodes searched` and `Speed`. Formatting happens after
the traversal timer stops. The UCI benchmark parser accepts both plain and
comma-grouped numbers; the measured binaries remain frozen in the local artifacts.
