# Move-generation direction split — 2026-09-09

## Decision: retain the existing engine

No production change is retained. The best screening candidate, NEON directions
`{1, 7}` with integer directions `{8, 9}`, does not demonstrate a repeatable
improvement over the baseline in confirmation:

| Path | Median paired NPS gain | Geometric mean | 95% interval, geometric mean |
|---|---:|---:|---|
| cold | +0.15% | +0.21% | [-0.23%, +0.67%] |
| nocache | +0.01% | +0.18% | [-0.29%, +0.57%] |

Both intervals include zero. The four-round screening median of +0.71%
does not survive confirmation as a convincing gain.

Baseline commit: `3fdc0b91eecac69dd812350a887f532abcb825dc`.
Its default remains integer directions `{1, 8}` plus NEON directions `{7, 9}`.
The flip kernel, perft recursion, cache policy, timing and protocol are unchanged.
Only measurement tooling, experiment patches and this report are added.

## Scope and method

- Exhaust all 16 ways to assign the four bidirectional axes to integer/NEON:
  one baseline plus 15 source variants.
- Step sizes: `1` horizontal, `8` vertical, `7` and `9` diagonal.
  Vertical uses the unmasked opponent board; the other axes use `kInner`.
- Every variant changes only `get_moves_hybrid()`; `flip_hybrid()`, the mask
  table and backend selection are unchanged.
- The all-NEON-directions trial is therefore **not** `ISLAY_USE_NEON=ON`,
  which also selects a different flip implementation.
- Apple M3 Pro, Apple Clang 21.0.0, C++20, native Release/ThinLTO:
  `-O3 -DNDEBUG -march=native -flto=thin`.
- Single-thread perft, one measured process at a time, `PerftHash=256` MiB.
  Builds and correctness runs finish before timing starts; idle sleep is
  inhibited with `caffeinate -i`. No affinity pinning; background noise remains.
- Screening: 10 positions / 19 nonzero position-rule cases, four paired rounds,
  80 ms accumulated traversal time per sample.
- Confirmation: all 18 positions / 35 cases, eight paired rounds, 120 ms per
  sample on both cold-cache and nocache paths. The eight extra positions are
  excluded from this study's screening stage.
- Same corpus seed `20260909` as the preceding studies, with optional depth-14
  generic-dispatch correctness cases. The Reversi forced-pass root is zero
  and belongs in correctness checks, not NPS statistics.
- Each round reverses A/B order and shuffles position order deterministically.
  Cold means a cleared logical TT, not a CPU-cache flush. TT allocation and
  clears are outside measured traversal time.
- Per case, take the median of paired NPS ratios, then the median across cases.
  Geometric-mean intervals bootstrap whole A/B rounds together, 2,000 draws.
  These characterize this suite, not every possible Othello position.
- Byte-identical A/A control: median +0.14%, geometric mean -0.07%,
  95% interval [-0.34%, +1.09%].
- Summary aggregation briefly overlapped the final all-NEON screening trial.
  That trial was repeated for four clean rounds; the table uses the repeat.
  Both raw journals are preserved.
- No warm-cache or production-UCI promotion benchmark was run: the candidate
  already failed to establish a gain in driver confirmation.

## Screening results

Directions listed in the NEON column are handled by explicit NEON kernels;
the complement uses integer kernels. These are source assignments, not a claim
about the compiler's final instruction scheduling.

| NEON axes | Integer axes | Median paired gain | Geometric mean |
|---|---|---:|---:|
| 7, 9 | 1, 8 | baseline | baseline |
| none | 1, 8, 7, 9 | -17.35% | -17.33% |
| 1 | 8, 7, 9 | -7.32% | -7.51% |
| 8 | 1, 7, 9 | -6.10% | -5.83% |
| 1, 8 | 7, 9 | -11.50% | -10.78% |
| 7 | 1, 8, 9 | -5.34% | -5.77% |
| 1, 7 | 8, 9 | +0.71% | +1.26% |
| 8, 7 | 1, 9 | -11.39% | -10.91% |
| 1, 8, 7 | 9 | -6.96% | -6.61% |
| 9 | 1, 8, 7 | -8.85% | -8.06% |
| 1, 9 | 8, 7 | +0.00% | +0.31% |
| 8, 9 | 1, 7 | -11.60% | -11.56% |
| 1, 8, 9 | 7 | -7.35% | -6.55% |
| 1, 7, 9 | 8 | -7.34% | -6.99% |
| 8, 7, 9 | 1 | -8.00% | -7.36% |
| 1, 8, 7, 9 | none | -16.72% | -15.74% |

The other near-neutral split, NEON `{1, 9}`, has approximately zero screening
median gain and is not promoted. Pure integer and all-NEON move generation
regress roughly 17%, even with the same flip kernel as the baseline.

## Confirmation: absolute median throughput

Candidate: NEON `{1, 7}`. MNPS = million reported leaves per second.
Every cell has eight paired samples. Absolute medians and paired-ratio medians
are different statistics; dividing the absolute medians may not reproduce the gain.

| Position / rule | Cold baseline MNPS | Cold candidate MNPS | Cold paired gain | Nocache baseline MNPS | Nocache candidate MNPS | Nocache paired gain |
|---|---:|---:|---:|---:|---:|---:|
| forced_pass / Othello | 814.61 | 822.67 | -1.57% | 795.78 | 769.35 | -0.81% |
| g0_discs12 / Othello | 1412.77 | 1424.56 | -0.24% | 1170.57 | 1170.76 | +0.86% |
| g0_discs12 / Reversi | 1431.88 | 1410.22 | +0.15% | 1136.54 | 1165.82 | +1.86% |
| g0_discs24 / Othello | 2180.12 | 2108.93 | -0.93% | 1638.87 | 1631.50 | +0.21% |
| g0_discs24 / Reversi | 2167.93 | 2206.65 | +0.66% | 1579.15 | 1632.40 | +1.14% |
| g0_discs40 / Othello | 1423.30 | 1438.36 | +0.72% | 1129.71 | 1127.47 | -0.21% |
| g0_discs40 / Reversi | 1444.73 | 1407.11 | -1.28% | 1141.14 | 1140.52 | -0.10% |
| g0_discs52 / Othello | 376.05 | 385.68 | +2.56% | 227.24 | 229.29 | +2.04% |
| g0_discs52 / Reversi | 372.63 | 375.71 | +1.30% | 221.00 | 218.36 | -1.51% |
| g1_discs12 / Othello | 1524.19 | 1479.11 | -0.31% | 1216.40 | 1210.61 | -0.03% |
| g1_discs12 / Reversi | 1503.92 | 1498.26 | +0.19% | 1211.73 | 1203.65 | +0.01% |
| g1_discs24 / Othello | 2585.58 | 2641.29 | +0.95% | 1876.50 | 1860.62 | -0.06% |
| g1_discs24 / Reversi | 2731.98 | 2672.15 | -0.19% | 1895.42 | 1915.36 | +0.09% |
| g1_discs40 / Othello | 1686.72 | 1689.11 | -0.40% | 1190.98 | 1224.23 | +1.77% |
| g1_discs40 / Reversi | 1737.04 | 1719.40 | +0.91% | 1224.65 | 1235.93 | +0.36% |
| g1_discs52 / Othello | 276.50 | 272.75 | -2.21% | 198.71 | 198.55 | -0.18% |
| g1_discs52 / Reversi | 283.74 | 279.66 | -0.66% | 193.92 | 193.34 | +0.35% |
| g2_discs12 / Othello | 1392.34 | 1327.85 | -2.78% | 1192.75 | 1176.52 | +0.00% |
| g2_discs12 / Reversi | 1353.33 | 1343.83 | -1.49% | 1189.43 | 1207.15 | +0.45% |
| g2_discs24 / Othello | 1461.01 | 1421.28 | +0.58% | 1171.01 | 1206.27 | +3.08% |
| g2_discs24 / Reversi | 1430.33 | 1433.13 | +0.69% | 1188.24 | 1198.56 | +0.14% |
| g2_discs40 / Othello | 1268.07 | 1256.29 | -0.89% | 1038.49 | 1024.25 | +0.46% |
| g2_discs40 / Reversi | 1249.04 | 1255.86 | +0.53% | 1020.79 | 1031.76 | +0.60% |
| g2_discs52 / Othello | 257.64 | 271.99 | +2.58% | 218.52 | 217.41 | +0.17% |
| g2_discs52 / Reversi | 258.39 | 263.98 | +2.11% | 211.33 | 208.07 | -1.07% |
| g3_discs12 / Othello | 1569.08 | 1586.93 | +0.12% | 1234.34 | 1227.68 | +0.06% |
| g3_discs12 / Reversi | 1608.69 | 1623.57 | -0.43% | 1231.69 | 1197.52 | -1.10% |
| g3_discs24 / Othello | 1670.71 | 1722.66 | +2.25% | 1325.88 | 1335.46 | +0.00% |
| g3_discs24 / Reversi | 1701.59 | 1686.88 | -0.72% | 1356.52 | 1333.12 | -1.00% |
| g3_discs40 / Othello | 1149.01 | 1174.62 | +1.54% | 994.49 | 968.69 | -1.12% |
| g3_discs40 / Reversi | 1138.14 | 1181.63 | +1.23% | 958.60 | 949.23 | +0.60% |
| g3_discs52 / Othello | 368.74 | 375.71 | +3.24% | 258.24 | 257.54 | -0.13% |
| g3_discs52 / Reversi | 356.53 | 367.55 | +2.15% | 243.07 | 246.48 | -0.41% |
| start / Othello | 4727.00 | 4497.09 | -0.34% | 937.34 | 936.61 | -0.02% |
| start / Reversi | 4697.98 | 4531.96 | -2.33% | 950.84 | 939.00 | -0.06% |

## Correctness and audit

- All 16 binaries pass the built-in movegen self-test: 5,000 deterministic
  disjoint boards, legal-move comparison with a square-scanning reference,
  and flip checks on every empty square.
- Each also passes 4,240 perft checks: 265 expected cases, eight symmetries,
  both cached/uncached paths, and a 1 MiB verification TT.
- All 1,852 timed A/B pairs have identical node counts:
  1,216 initial screening + 76 repeat + 560 confirmation.
- All 16 source snapshots and the complete corpus reproduce byte for byte.
  Every variant's perft source matches production exactly.
- Production sources remain identical to baseline. No experimental movegen
  kernel is installed in the engine.
- `tools/perft_bench.cpp --verify` now runs `movegen_selftest()` before the
  existing perft checks. Timed traversal logic is unchanged.

## Reproduction

Use a fresh directory; snapshots refuse to overwrite an existing base.

```sh
python3 tools/perft_study.py snapshot --directory _build/movegen-split-reproduce \
  --revision 3fdc0b91eecac69dd812350a887f532abcb825dc \
  --patches tools/movegen_variants --variant-movegen
python3 tools/perft_study.py prepare --generic-checks --directory _build/movegen-split-reproduce
python3 tools/perft_study.py build baseline baseline_control neon_none neon_1 neon_8 neon_1_8 neon_7 neon_1_7 neon_8_7 neon_1_8_7 neon_9 neon_1_9 neon_8_9 neon_1_8_9 neon_1_7_9 neon_8_7_9 neon_1_8_7_9 \
  --directory _build/movegen-split-reproduce
python3 tools/perft_study.py verify baseline neon_none neon_1 neon_8 neon_1_8 neon_7 neon_1_7 neon_8_7 neon_1_8_7 neon_9 neon_1_9 neon_8_9 neon_1_8_9 neon_1_7_9 neon_8_7_9 neon_1_8_7_9 \
  --directory _build/movegen-split-reproduce
python3 tools/perft_study.py run baseline_control neon_none neon_1 neon_8 neon_1_8 neon_7 neon_1_7 neon_8_7 neon_1_8_7 neon_9 neon_1_9 neon_8_9 neon_1_8_9 neon_1_7_9 neon_8_7_9 neon_1_8_7_9 \
  --rounds 4 --milliseconds 80 --tag screening --directory _build/movegen-split-reproduce
python3 tools/perft_study.py run neon_1_8_7_9 --rounds 4 --milliseconds 80 \
  --tag screening-repeat --directory _build/movegen-split-reproduce
python3 tools/perft_study.py run neon_1_7 --rounds 8 --milliseconds 120 \
  --include-holdout --modes cold nocache --tag confirmation \
  --directory _build/movegen-split-reproduce
python3 tools/perft_study.py report --directory _build/movegen-split-reproduce
```

`--variant-movegen` additionally copies `movegen.cpp` into each variant snapshot.
The build command uses that copy when present, otherwise the historical shared
baseline source. Existing perft-only snapshot defaults remain unchanged.

Raw artifacts are Git-ignored under `_build/movegen-split/`: corpus,
`results-screening.jsonl`, `results-screening-repeat.jsonl`,
`results-confirmation.jsonl`, `summary.json`, frozen sources/binaries,
and per-variant build commands, hashes and verification logs.

| Artifact | SHA-256 |
|---|---|
| Corpus | `99a459548665f6ab25c354e4c5d12a1fcbea49c9b16895bd4ae9da714ea663ef` |
| Baseline driver | `c2401832efd9c947a4735b464eb86365c552eb04f4e983baada10e39879e7e01` |
| NEON 1,7 driver | `3bb1966f8ce662d2701eabe5188bef37dc9a43753c6c7f6175f81cf5941315a8` |

These are perft throughput experiments, not playing-strength evidence.
No speed claim is made for x86, other compilers or other cache budgets.
