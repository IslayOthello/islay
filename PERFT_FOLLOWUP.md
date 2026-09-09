# Perft inlining follow-up — 2026-09-09

## Retained change

Remove `ISLAY_FLATTEN` from the four cached/uncached, generic/depth-specialized
perft functions. Keep `ISLAY_HOT`, template specialization through depth 12,
the raw-key depth-3 cache, canonical keys at depth 4+, and all counting rules.
`src/perft.cpp` differs from the measured `no_flatten` source only by an explanatory
comment. No perft threads were added.

Baseline: `c60821083a9cdd7fbabbb129382b66ea8c8ed5b7`, including the previous
cache optimization and comma-formatted UCI totals. These are incremental gains
against that commit, not against the earlier search engine.

The final CMake-built UCI engine gains **+1.48% median paired NPS**.
The geometric mean gain is +1.39% with a whole-round bootstrap
95% interval [+0.88%, +2.22%].
This is a modest workload-specific improvement, not a guarantee for every position.

These measurements and frozen executable hashes predate the subsequent UCI
microsecond-based timing and seconds-display change. The measured binaries already computed NPS
using fractional milliseconds, not the truncated integer `Time` display.

## Method

- Apple M3 Pro, macOS 26.6.2 arm64, Apple Clang 21.0.0.
- C++20 Release, native tuning, ThinLTO: `-O3 -DNDEBUG -march=native -flto=thin`.
- Final UCI comparison uses the project's CMake build for both binaries.
  A separate direct-Clang UCI comparison is labeled below; it is not the headline result.
- One perft thread and one benchmark process at a time. No builds or self-tests
  run during timing. `caffeinate -i` prevents idle sleep; no CPU affinity pinning.
- `PerftHash=256` MiB throughout performance tests; correctness also uses 1 MiB.
- Same seeded timed corpus as the previous study: 10 screening positions,
  then 8 additional positions held out of screening. Both rules: 35 timed cases.
  The forced-pass Reversi root has zero nodes and is tested for correctness,
  not included in NPS ratios.
- Screening: four A/B rounds, 80 ms accumulated traversal time per driver sample.
  Confirmation: eight rounds on all 18 positions, 120 ms cold samples.
  Warm/nocache confirmation: eight rounds, 80 ms samples.
- UCI: eight rounds per case, five cold traversals per process; discard the first,
  derive combined time from the four remaining `Speed` values (not rounded `Time`).
- A/B order reverses each round; position order is shuffled deterministically.
  Allocation and TT clears are outside traversal timing.
- Per case, take the median of paired candidate/baseline NPS ratios; then take
  the median across cases. Absolute NPS columns below are separate medians:
  their quotient need not equal the median paired ratio.
- Bootstrap intervals resample all positions' A/B rounds together, 2,000 draws,
  and refer to the geometric mean of per-case median ratios.
- Byte-identical A/A control: median +0.35%, geometric mean +0.23%,
  95% interval [-1.16%, +1.96%]. Individual-case medians range [-5.65%, +2.93%].
  Normal macOS background noise remains.

## Results

| Measurement | Median paired gain | Geometric mean | 95% interval, geometric mean |
|---|---:|---:|---|
| CMake UCI, cold TT | +1.48% | +1.39% | [+0.88%, +2.22%] |
| Standalone driver, cold TT | +0.74% | +0.93% | [+0.47%, +1.67%] |
| Standalone driver, nocache | +2.80% | +2.30% | [+1.45%, +2.96%] |
| Standalone driver, warm TT | +7.68% | +7.57% | [+7.42%, +7.87%] |
| Direct-Clang UCI cross-check | +1.94% | +1.62% | [+0.93%, +2.50%] |

Warm-cache NPS represents reuse of previously counted subtrees, not fresh
enumeration of all reported leaves. Cold, warm and nocache results are not interchangeable.
Final CMake UCI case medians range from -3.05% to +4.70%;
the aggregate win does not mean every position became faster.

### Candidate selection

| Candidate | Screening median | Confirmation median | Decision |
|---|---:|---:|---|
| Remove all four perft flatten hints | +1.29% | +0.74% | Retained |
| Specialize only depths 1–4 | -16.37% | — | Rejected |
| Dedicated depth-2 scalar loop | -1.28% | — | Rejected |
| Dedicated depth-2 loop, noinline | +1.11% | -0.19% | Rejected |
| Remove flatten + depth-2 noinline | — | -0.60% | Rejected |

The noinline leaf's apparent screening gain disappears during confirmation.
Combining it with removal of flatten also regresses. Limiting template depth
has a clear slowdown. None of those kernels/dispatcher changes entered production.

### Code size and build observation

| Artifact | Baseline | Retained |
|---|---:|---:|
| CMake UCI executable | 2,167,496 bytes | 138,552 bytes |
| Standalone driver executable | 2,137,800 bytes | 92,312 bytes |
| Driver Mach-O `__TEXT` segment | 2,097,152 bytes | 65,536 bytes |
| One observed driver compile/link | 110.7 s | 2.4 s |

The executable shrinks about 93.6%. Build times are single observations, not
paired compilation benchmarks. Smaller generated code is directly observed;
an instruction-cache explanation for the NPS improvement remains a hypothesis,
not a hardware-counter profiling result.

## Final CMake UCI per-position medians

MNPS means million reported leaves per second. Each row has eight paired samples.

| Position | Rule | Depth | Nodes | Baseline MNPS | Retained MNPS | Paired gain |
|---|---|---:|---:|---:|---:|---:|
| forced_pass | Othello | 9 | 10,232,092 | 799.30 | 818.26 | +2.19% |
| g0_discs12 | Othello | 8 | 51,655,921 | 1363.76 | 1437.88 | +3.66% |
| g0_discs12 | Reversi | 8 | 51,655,914 | 1385.82 | 1405.09 | +0.85% |
| g0_discs24 | Othello | 7 | 67,122,213 | 2056.32 | 2050.97 | +2.79% |
| g0_discs24 | Reversi | 7 | 67,122,213 | 2142.10 | 2151.93 | +1.42% |
| g0_discs40 | Othello | 7 | 15,935,771 | 1335.73 | 1414.46 | +3.40% |
| g0_discs40 | Reversi | 7 | 15,935,771 | 1400.51 | 1390.97 | +0.36% |
| g0_discs52 | Othello | 9 | 3,283,178 | 359.09 | 356.04 | +1.17% |
| g0_discs52 | Reversi | 9 | 3,279,315 | 389.36 | 370.79 | -1.62% |
| g1_discs12 | Othello | 8 | 80,179,786 | 1448.96 | 1470.01 | +1.63% |
| g1_discs12 | Reversi | 8 | 80,179,783 | 1479.16 | 1499.56 | +0.83% |
| g1_discs24 | Othello | 7 | 103,212,939 | 2572.79 | 2702.00 | +3.84% |
| g1_discs24 | Reversi | 7 | 103,212,939 | 2666.25 | 2678.70 | -1.20% |
| g1_discs40 | Othello | 7 | 44,569,404 | 1612.18 | 1656.32 | +3.68% |
| g1_discs40 | Reversi | 7 | 44,569,404 | 1669.40 | 1686.58 | +2.14% |
| g1_discs52 | Othello | 9 | 1,961,943 | 263.53 | 254.34 | -2.91% |
| g1_discs52 | Reversi | 9 | 1,938,170 | 265.08 | 260.19 | -1.46% |
| g2_discs12 | Othello | 8 | 61,280,632 | 1295.27 | 1340.85 | +2.96% |
| g2_discs12 | Reversi | 8 | 61,280,503 | 1369.71 | 1372.23 | -0.67% |
| g2_discs24 | Othello | 7 | 11,182,564 | 1335.28 | 1382.05 | +2.78% |
| g2_discs24 | Reversi | 7 | 11,182,564 | 1353.97 | 1419.41 | +0.97% |
| g2_discs40 | Othello | 7 | 10,050,633 | 1192.24 | 1223.19 | +3.18% |
| g2_discs40 | Reversi | 7 | 10,050,633 | 1197.30 | 1227.35 | +3.21% |
| g2_discs52 | Othello | 9 | 1,009,538 | 263.98 | 279.04 | +4.70% |
| g2_discs52 | Reversi | 9 | 1,004,008 | 286.29 | 266.78 | -3.05% |
| g3_discs12 | Othello | 8 | 78,625,684 | 1533.03 | 1528.66 | +1.48% |
| g3_discs12 | Reversi | 8 | 78,625,684 | 1575.51 | 1593.42 | +1.37% |
| g3_discs24 | Othello | 7 | 31,204,292 | 1671.72 | 1676.07 | -1.54% |
| g3_discs24 | Reversi | 7 | 31,204,292 | 1691.59 | 1686.79 | -0.57% |
| g3_discs40 | Othello | 7 | 5,261,592 | 1068.36 | 1105.51 | +3.77% |
| g3_discs40 | Reversi | 7 | 5,261,587 | 1090.57 | 1080.02 | -0.22% |
| g3_discs52 | Othello | 9 | 2,298,678 | 361.04 | 376.03 | +3.53% |
| g3_discs52 | Reversi | 9 | 2,257,430 | 351.98 | 356.75 | +2.49% |
| start | Othello | 11 | 212,258,216 | 4604.38 | 4552.12 | +3.45% |
| start | Reversi | 11 | 212,257,448 | 4667.40 | 4627.84 | +0.90% |

## Correctness

- Baseline and all five candidate drivers each pass **4,240** checks:
  265 oracle cases × eight symmetries × cached/uncached.
- Independent square-scanning oracle: 28 positions, both rules, shallow depths;
  known start counts through depth 8; endgames and forced passes.
- Extra depth-14 divide checks reach the generic perft dispatcher at remaining
  depth 13. Depth-13 divide alone reaches specialized depth 12; the old suite's
  claim to exercise the generic dispatcher through that case was too broad.
- All deep timed A/B pairs have identical node counts.
- Final CMake Release/native/LTO and Debug/portable/no-LTO both report
  `ALL TESTS PASSED`, including the nine-position regression suite.
- Start-position `go perft 8` and `go perft 8 nocache` both print
  `Nodes searched: 390,216`. Comma formatting is unchanged.
- All six frozen source snapshots (baseline + five candidates) and the corpus
  were reproduced byte for byte from the checked-in patches and seed.

Validation commands:

```sh
cmake --build build -j 4
cmake --build _build/perft-debug -j 4
printf 'debug on\ntest\nposition startpos\ngo perft 8\ngo perft 8 nocache\nquit\n' | ./build/islay
printf 'debug on\ntest\nposition startpos\ngo perft 8\ngo perft 8 nocache\nquit\n' | ./_build/perft-debug/islay
```

## Reproduction

The old snapshot defaults remain unchanged. New optional `--revision`,
`--patches` and `--generic-checks` arguments reproduce this follow-up separately.

```sh
python3 tools/perft_study.py snapshot --directory _build/perft-next-reproduce \
  --revision c60821083a9cdd7fbabbb129382b66ea8c8ed5b7 --patches tools/perft_next_variants
python3 tools/perft_study.py prepare --generic-checks --directory _build/perft-next-reproduce
python3 tools/perft_study.py build baseline baseline_control no_flatten cap4 \
  leaf2 leaf2_noinline no_flatten_leaf2 --directory _build/perft-next-reproduce
python3 tools/perft_study.py verify baseline no_flatten cap4 leaf2 leaf2_noinline \
  no_flatten_leaf2 --directory _build/perft-next-reproduce
python3 tools/perft_study.py run baseline_control no_flatten cap4 leaf2 leaf2_noinline \
  --rounds 4 --milliseconds 80 --tag screening --directory _build/perft-next-reproduce
python3 tools/perft_study.py run no_flatten leaf2_noinline no_flatten_leaf2 \
  --rounds 8 --milliseconds 120 --include-holdout --tag confirmation \
  --directory _build/perft-next-reproduce
python3 tools/perft_study.py run no_flatten --rounds 8 --milliseconds 80 \
  --include-holdout --modes warm nocache --tag paths --directory _build/perft-next-reproduce
python3 tools/perft_study.py report --directory _build/perft-next-reproduce
```

For the final UCI comparison, use a detached baseline checkout and the optimized
working tree, with matching Release/native/LTO options:

```sh
git worktree add --detach _build/perft-next-baseline-src c60821083a9cdd7fbabbb129382b66ea8c8ed5b7
cmake -S _build/perft-next-baseline-src -B _build/perft-next-baseline-build \
  -DCMAKE_BUILD_TYPE=Release -DISLAY_NATIVE=ON -DISLAY_LTO=ON
cmake --build _build/perft-next-baseline-build -j 4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DISLAY_NATIVE=ON -DISLAY_LTO=ON
cmake --build build -j 4
mkdir -p _build/perft-next-reproduce/production
cp _build/perft-next-baseline-build/islay _build/perft-next-reproduce/baseline/islay
cp build/islay _build/perft-next-reproduce/production/islay
python3 tools/perft_study.py run production --rounds 8 --include-holdout --uci \
  --tag production --directory _build/perft-next-reproduce
python3 tools/perft_study.py report --directory _build/perft-next-reproduce
```

Local raw data remains Git-ignored under `_build/perft-next/`:
`corpus.json`, `summary.json`, per-variant build/verification records, frozen
binaries/sources, and journals `results-{screening,confirmation,paths,uci,production}.jsonl`.
The direct-Clang cross-check used the same source tree and optimization flags,
but not CMake's object/link layout:

```sh
clang++ -std=c++20 -O3 -DNDEBUG -march=native -flto=thin -Isrc \
  main.cpp src/movegen.cpp src/board.cpp src/options.cpp src/uci.cpp \
  _build/perft-next/no_flatten/perft.cpp -o _build/perft-next/no_flatten/islay
```

| Artifact | SHA-256 |
|---|---|
| Corpus | `99a459548665f6ab25c354e4c5d12a1fcbea49c9b16895bd4ae9da714ea663ef` |
| Baseline driver | `768061f45cb3803586a4b8f7b8398f03a2c5de1c0b11499d94446a42840af55c` |
| Retained driver | `98e649bbcd3f8f58242caedd96e05b6a4429d964dfa148c5a109eb7a8a857bba` |
| Baseline CMake UCI | `989a149967c3ddb0ccde4a3f6b7af9c5ac565ae3275274072c1baf50cb5e9e5c` |
| Retained CMake UCI | `f0fea7142f88247a98d9baf3b5e176326b9cc78c1987775b50edc071e550c963` |
| Direct-Clang UCI cross-check | `e6528d76ede81ae078942a63fd4713425ed0413b7ee5db542b3aace8439695fa` |

This measures perft performance, not playing strength. Linux/x86 performance,
other compilers and other hash sizes were not measured in this follow-up.
