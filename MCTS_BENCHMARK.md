# PUCT foundation: correctness and optimization study

2026-09-09, Apple M3 Pro, AppleClang 21.0.0, arm64, C++20,
`-O3 -mcpu=native -flto=thin -DNDEBUG`. All searches and measurements are single-threaded.

## Delivered scope

This section records the P1 milestone: the Othello game adapter and synchronous PUCT core,
before public UCI search. The later P2 controller/protocol is documented in `UCI.md`;
there is still no trained network.

- `game.hpp`: legal placements/forced pass, terminal outcome, checked action application,
  policy indices 0–64 and D4 action mapping. NOMOVE is not a policy action.
- `evaluator.hpp`: batch-ready policy-logit/value interface with cooperative cancellation.
- `mcts.hpp`: simulation, memory and deadline budgets; root visits/priors/values, PV,
  explicit stop reason and microsecond-resolution elapsed time.
- `mcts.cpp`: contiguous siblings in an index arena; 32-byte node on the measured ABI,
  parent-relative edge values, terminal caching and stable legal-only softmax.
- The core rejects Reversi and invalid boards/configuration. Evaluator failures propagate;
  a zero-budget/cancelled/memory-limited search still returns a legal fallback if nonterminal.
  Terminal roots return NOMOVE without evaluating a network.

The arena budget excludes evaluator memory and fixed stack/result storage. Cancellation and
deadlines are observed between evaluations; they cannot forcibly interrupt a blocking evaluator.
The root is expanded before simulations, and root initialization is not a completed simulation.

## Workload and method

`tools/mcts_study.py` deterministically creates 18 positions: startpos, four seeded games
sampled at ply 8/20/40/52, and a forced-pass position. Position generation uses the independent
square-scanning oracle from `perft_study.py`. Every search uses 4,096 simulations and a
64 MiB arena cap, with no transposition table or tree reuse across searches.

Two evaluators exercise the core:

- `uniform`: zero logits and zero nonterminal value.
- `synthetic`: deterministic board-hash-derived finite logits and values.

Neither is a neural network, a heuristic intended for play, or a strength benchmark.
Timing includes arena allocation, root evaluation, traversal, result collection and destruction;
it excludes process startup, the initial warm-up call and result-signature validation.
Each sample repeats full searches for at least 100 ms. Each experiment has four paired rounds,
shuffled workloads and alternating A/B order: 72 paired observations per evaluator.

The reported change is `median(candidate NPS / reference NPS - 1)` across paired observations.
It is not a ratio of pooled node totals. No benchmark processes run concurrently.
Every pair must match the signature of root visits/priors/values, PV, node count and evaluation count;
each binary also checks repeatability within a timing sample.

## Results

| Candidate versus compact reference | Uniform median NPS change | Synthetic median NPS change | Decision |
|---|---:|---:|---|
| Cache a Board in every node, initialize lazily (48-byte nodes) | -5.70% | -3.57% | Reject |
| Defer board replay until a fresh leaf; skip replay for terminal revisits | -9.86% | -10.87% | Reject |
| Cache double-precision Q during backup (40-byte nodes) | -1.40% | +0.20% | Reject |
| Normalize policy using one reciprocal and per-action multiplication | -0.76% | -0.10% | Reject |

Retain the original compact reference. The small positive result for cached Q is not
consistent across evaluator modes and is not evidence of an improvement. These are screening
measurements, not confidence-bounded performance claims; no candidate warranted promotion.
The alternative patches are retained under `tools/mcts_variants/` for reproducibility.

For scale only, reference medians in the first experiment were 6,376,927 simulations/s
(uniform) and 3,769,806 simulations/s (synthetic) over its position/round observations.
Startpos alone measured 5,984,434 and 3,809,895 simulations/s respectively.
These figures do not predict neural-network throughput and must not be compared with perft leaf NPS.

Raw local artifacts are under ignored `_build/mcts-study/`: `screening`, `deferred-screening`,
`mean-screening`, and `normalize-screening`, each containing corpus/settings, binary hashes,
individual timing records and summary JSON. They are not committed.

## Reproduce

Build the retained reference:

```sh
rtk proxy mkdir -p _build/mcts-repro
rtk proxy c++ -std=c++20 -O3 -mcpu=native -flto=thin -DNDEBUG -Wall -Wextra -Isrc tools/mcts_bench.cpp src/game.cpp src/mcts.cpp src/mcts_selftest.cpp src/board.cpp src/movegen.cpp -o _build/mcts-repro/reference
rtk proxy _build/mcts-repro/reference --verify
```

Example candidate; replace `cached` with `deferred`, `mean` or `normalize` for the other trials.
Use fresh output directories; the study refuses to overwrite a previous run.

```sh
rtk proxy mkdir -p _build/mcts-repro/cached
rtk proxy cp src/mcts.cpp _build/mcts-repro/cached/mcts.cpp
rtk proxy git apply --unidiff-zero --directory=_build/mcts-repro/cached tools/mcts_variants/cached.patch
rtk proxy c++ -std=c++20 -O3 -mcpu=native -flto=thin -DNDEBUG -Wall -Wextra -Isrc tools/mcts_bench.cpp src/game.cpp _build/mcts-repro/cached/mcts.cpp src/mcts_selftest.cpp src/board.cpp src/movegen.cpp -o _build/mcts-repro/cached/bench
rtk proxy python3 tools/mcts_study.py _build/mcts-repro/reference _build/mcts-repro/cached/bench --output _build/mcts-repro/cached-results --rounds 4 --milliseconds 100
```

The compiler flags above are the measured Apple Silicon configuration, not portable flags
for every compiler/architecture.

## Correctness gates

The aggregate `debug on; test; quit` checks both new modules alongside the retained perft suite:

- Checked action application against a square-scanning oracle and all eight board symmetries.
- Terminal wins/losses/draws including boards with empty squares; forced pass and invalid actions.
- Known root visit balance, policy-driven selection, zero-budget fallback and repeatable PV.
- Parent-relative sign changes at one ply and across PASS → placement → terminal.
- Cancellation before/during evaluation, expired deadline, tiny/full memory budgets,
  invalid evaluator outputs and unsupported rules.
- Small endgames from 64 deterministic playout attempts, comparing chosen actions against an
  exact test-only oracle, including draws and positions whose alternatives have different outcomes.
- Multiple middlegame/endgame positions under all D4 transforms.
- Existing perft counts, cache/symmetry consistency, both Othello and Reversi rules.

Validated builds: Release native with LTO, Release `ISLAY_NATIVE=OFF` with LTO, and Debug
`ISLAY_NATIVE=OFF`, `ISLAY_LTO=OFF`, AddressSanitizer + UndefinedBehaviorSanitizer.
Portable configuration was run on the same arm64 Mac, not on x86 or Linux.
Startpos perft(8) remains 390,216; human Time/Speed formatting is unchanged.

No arena matches, Elo, CI or LOS are reported: this milestone has no playable learned evaluator.
The P1 follow-up was interruptible UCI search. P3 now supplies optional policy/value
inference; see [NEURAL.md](NEURAL.md) for its separate parity and inference measurements.

## P2 controller validation

The controller now connects the unchanged compact PUCT core to `go nodes`, `go movetime`,
`go infinite`, `stop` and `bestmove`. Perft remains single-threaded and never overlaps search.
Command output is buffered into complete blocks sharing a mutex with worker output; stdin's
automatic stdout flush is disabled so it cannot bypass that mutex. Infinite jobs park on a
stop-aware condition variable if traversal ends early, rather than polling or expanding past the cap.

On the same M3 Pro, aggregate self-tests and `tools/uci_search_test.py` passed for:

- Native Release + LTO.
- Release `ISLAY_NATIVE=OFF` + LTO, on arm64 (not an x86/Linux validation).
- Debug + ASan/UBSan, native tuning and LTO disabled.
- Debug + ThreadSanitizer, native tuning and LTO disabled; no race diagnostics in these tests.

Protocol coverage includes 18-position PV legality against the scalar oracle, pass/terminal roots,
zero/combined/malformed limits, arena exhaustion, immediate/repeated stop, new go/position/options,
readiness/handshake during search, quit and EOF. The built-in controller self-test checks terminal
and forced-pass results, repeated stop, cancellation and unsupported Reversi. Existing perft
regressions and startpos depth 8 = 390,216 remain unchanged.

In the final isolated native protocol run, 20 stop samples measured median **0.036458 ms** and
p95 **0.038125 ms**. This is host-observed `stop; stop; isready` round-trip time after `go infinite`
and a readiness barrier, alternating startpos and terminal roots, with the uniform evaluator.
It includes pipe/scheduling overhead; it is not a latency guarantee for a future network backend.
No MCTS throughput increase or playing-strength gain is claimed for this protocol milestone.
