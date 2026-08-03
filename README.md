# islay

[Tiếng Việt](README.vi.md) · [UCI Protocol](UCI.md)

`islay` is a competitive Othello/Reversi engine written in C++20. It combines
SIMD-aware bitboard move generation, a selective negamax/PVS search, exact
endgame solving, and a compact NNUE-style evaluator trained through self-play.
The engine exposes a documented UCI-style text protocol for GUIs, match runners,
and command-line analysis.

## Highlights

- Supports both Othello and Reversi rules.
- Uses portable scalar, AVX2, or ARM NEON move-generation backends selected at
  compile time.
- Searches with iterative deepening, PVS, transposition tables, ProbCut, late
  move reductions, futility pruning, killer moves, and continuation history.
- Orders tight endgames by connected-region parity and returns exact
  game-theoretic scores when the requested depth reaches the remaining empties.
- Loads D4-symmetry-aware opening books and maps canonical moves back to the
  original board orientation.
- Ships `weights/v20.nnue`, a per-stage NNUE-lite network with int16 embeddings
  and an eight-value hidden accumulator.
- Includes deterministic self-play training, NNUE-to-NNUE bootstrapping, paired
  match measurement, perft, and built-in correctness tests.

## Technology and Architecture

| Area | Implementation |
|---|---|
| Language and build | C++20, CMake 3.16+, optional native CPU tuning and LTO |
| Board representation | Bitboards with relative player/opponent state |
| Move generation | Scalar fallback, AVX2 on x86-64, hybrid integer/NEON on ARM64 |
| Search | Iterative-deepening negamax/PVS, TT, selective pruning, history-based ordering |
| Endgame | Pass-aware exact search, stability bounds, connected-region parity |
| Evaluation | Built-in heuristic, trained pattern weights, or quantized NNUE-lite |
| Opening book | Canonical D4 keys with rotation/reflection-safe move mapping |
| Interface | Asynchronous UCI-style line protocol over standard input/output |
| Validation | Module self-tests, known perft counts, search oracle, paired Elo matches |

The NNUE-lite evaluator reuses the engine's incremental pattern features. Active
feature rows are accumulated into eight hidden values, then evaluated by
per-stage linear and ReLU heads. Version 21 was warm-started from v20 and trained
on 20,000 games using depth-10 search scores as distillation targets; data
generation was capped at four workers.

## Estimated Elo

`islay` does not yet have a defensible **absolute Elo rating** because it has not
been calibrated against a stable external engine pool. The figures below are
paired, color-reversed estimates relative to earlier `islay` evaluators.

| Comparison | Test | Sample | Elo estimate | 95% CI | LOS |
|---|---:|---:|---:|---:|---:|
| v19 NNUE vs v18 patterns | 200k nodes | repository ledger | +45.9 | [33.7, 58.2] | 100% |
| v19 NNUE vs v18 patterns | 100 ms equal time | repository ledger | +31.4 | [19.8, 42.9] | 100% |
| v20 NNUE vs v19 NNUE | 200k nodes | 4,000 | +8.08 | [2.24, 13.92] | 99.67% |
| v20 NNUE vs v19 NNUE | 100 ms equal time | 2,280 | +3.51 | [-4.25, 11.27] | 81.23% |
| v21 NNUE vs v20 NNUE | 20k nodes, balanced d8 book | 1,506 | +56.32 | [43.22, 69.58] | 100% |

The v20 equal-time run was stopped before its planned 4,000-game cap, so its
confidence interval still includes zero.

The current fixed-node estimate is therefore **v21 ≈ +56 Elo over v20**. The
SPRT accepted H1 at LLR 2.964 before its 10,000-game cap. This is a directional
project estimate, not an absolute playing rating; an equal-time confirmation is
still useful even though v20 and v21 share the same inference architecture.

## Build

Requirements:

- A C++20 compiler such as recent Clang, GCC, or MSVC
- CMake 3.16 or newer

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Release builds enable native CPU tuning and link-time optimization when the
toolchain supports them. Build a portable binary with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DISLAY_NATIVE=OFF -DISLAY_LTO=OFF
cmake --build build -j
```

## Run

Start an interactive engine session:

```sh
./build/islay
```

Load the recommended evaluator and search the initial position:

```sh
printf 'uci\nsetoption name EvalFile value weights/v21.nnue\nisready\nposition startpos\ngo depth 10\nquit\n' \
  | ./build/islay
```

Useful options include:

| Option | Purpose |
|---|---|
| `Rule` | Select `Othello` or `Reversi` |
| `EvalFile` | Load a `.nnue` network or `.pat` pattern weights |
| `Hash` | Set the search transposition-table size in MiB |
| `Threads` | Set the number of search threads |
| `OwnBook` / `BookFile` | Enable and load an opening book |

See [UCI.md](UCI.md) for the complete protocol, position format, search limits,
responses, and error behavior.

## Testing and Development

Run the aggregate correctness suite:

```sh
printf 'debug on\ntest\nquit\n' | ./build/islay
```

The final line must be `ALL TESTS PASSED`. The suite covers move generation,
evaluation, the search oracle, incremental pattern state, opening-book symmetry,
known perft values, cache consistency, symmetry, and rule-specific pass behavior.

Run a reproducible perft check separately:

```sh
printf 'position startpos\ngo perft 8\nquit\n' | ./build/islay
```

For strength changes, use paired openings with reversed colors and report Elo,
the 95% confidence interval, and likelihood of superiority. Node reduction alone
is not evidence of stronger play.

## Repository Layout

```text
main.cpp       Engine entry point
src/           Board, move generation, search, evaluation, training, and UCI code
weights/       Versioned pattern and NNUE evaluation files
UCI.md         Released protocol reference
CMakeLists.txt Build configuration and architecture selection
```
