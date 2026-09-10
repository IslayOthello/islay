# islay

[Tiếng Việt](README.vi.md) · [UCI Protocol](UCI.md)

`islay` is a C++20 Othello/Reversi move-generation and perft engine with a
UCI-style text interface. It counts legal move sequences and supports setting
positions, displaying boards, benchmarking perft, and running built-in tests.
Experimental Othello PUCT search supports `go nodes`, `go movetime`, `go infinite`
and `stop`. Optional ONNX inference connects a shared 8x64 residual policy/value network
to PUCT. The default remains uniform policy/zero value; no trained weights are included.
See [NEURAL.md](NEURAL.md) for architecture B, export, inference and measured throughput.
Offline self-play/replay now includes root exploration, a bounded neural cache, atomic
resumable shards and vectorized training batches. See [SELFPLAY.md](SELFPLAY.md).
The FP32 trainer adds CPU/MPS execution, frozen replay windows and full-state checkpoint/resume.
See [TRAINING.md](TRAINING.md) for the learning workflow and measured performance; no champion
or trained-strength result is provided yet.

## Features

- Othello passes consume one ply; Reversi stops when the side to move has no move.
- Bitboard move generation with scalar, AVX2, and ARM NEON backends selected at compile time.
- Perft with bulk counting and an optional symmetry-aware transposition table.
- Position loading with legal move and pass validation.
- Built-in move-generation, perft, cache, symmetry, and rule checks.

## Build

Requires a C++20 compiler and CMake 3.16 or newer.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Release builds enable native CPU tuning and LTO when supported. Set
`-DISLAY_NATIVE=OFF` for a portable build, and `-DISLAY_LTO=OFF` to disable LTO.

## Run

```sh
./build/islay
printf 'uci\nisready\nposition startpos\ngo perft 8\nquit\n' | ./build/islay
```

Start-position perft at depth 8 returns `390216` nodes. Add `nocache` to
count without the transposition table:

```text
go perft 8 nocache
```

Options are `Rule` (`Othello` or `Reversi`), `PerftHash` (cache MiB, default 256)
and `MctsHash` (PUCT arena MiB, default 64), plus `EvalFile` (ONNX path, empty by default).
Perft runs synchronously and single-threaded;
search runs on one interruptible worker and only supports Othello. `go depth`, full clock
controls and book options are not supported yet. Wait for `bestmove` before `quit`.

See [UCI.md](UCI.md) for commands, position syntax, perft semantics, and errors.

## Validation

```sh
printf 'debug on\ntest\nquit\n' | ./build/islay
printf 'debug on\nbench 8\nquit\n' | ./build/islay
python3 tools/uci_search_test.py build/islay
```

The aggregate test must end with `ALL TESTS PASSED`. The benchmark reports
uncached start-position perft at depths 1 through the requested depth.

The suite also checks nine fixed opening/middlegame/endgame/pass positions under
both rules, all eight symmetries, and a 1 MiB cache.
See [PERFT_BENCHMARK.md](PERFT_BENCHMARK.md) for multi-position, single-thread
median NPS measurements and the reproducible A/B driver.

## Repository Layout

```text
main.cpp       Entry point
src/           Board, bitboards, move generation, perft, options, and UCI
UCI.md         Protocol reference
CMakeLists.txt Build configuration and architecture selection
```
