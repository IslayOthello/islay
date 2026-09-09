# islay

[Tiếng Việt](README.vi.md) · [UCI Protocol](UCI.md)

`islay` is a C++20 Othello/Reversi move-generation and perft engine with a
UCI-style text interface. It counts legal move sequences and supports setting
positions, displaying boards, benchmarking perft, and running built-in tests.
It does not search for a best move or play games.

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

The only options are `Rule` (`Othello` or `Reversi`) and `PerftHash`
(cache size in MiB, default 256). Perft runs synchronously. Search commands such
as `go depth` and evaluator/book options are no longer supported.

See [UCI.md](UCI.md) for commands, position syntax, perft semantics, and errors.

## Validation

```sh
printf 'debug on\ntest\nquit\n' | ./build/islay
printf 'debug on\nbench 8\nquit\n' | ./build/islay
```

The aggregate test must end with `ALL TESTS PASSED`. The benchmark reports
uncached start-position perft at depths 1 through the requested depth.

## Repository Layout

```text
main.cpp       Entry point
src/           Board, bitboards, move generation, perft, options, and UCI
UCI.md         Protocol reference
CMakeLists.txt Build configuration and architecture selection
```
