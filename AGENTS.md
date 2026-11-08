# Repository Guidelines

## Project Structure & Module Organization

`main.cpp` starts the engine and hands control to the UCI-style loop. Production code lives in `src/`, with matching lowercase headers and implementations such as `search.hpp`/`search.cpp`, `movegen.*`, and `eval.*`. Keep protocol behavior documented in `UCI.md`. Versioned evaluation assets are stored in `weights/` (`.pat` pattern weights and `.nnue` networks). Generated CMake output belongs in `build/`, `_build/`, or `cmake-build-debug/`; these paths are ignored and should not be committed.

## Build, Test, and Development Commands

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/islay
printf 'debug on\ntest\nquit\n' | ./build/islay
printf 'position startpos\ngo perft 8\nquit\n' | ./build/islay
```

The first two commands configure and build the C++20 `islay` executable. Release builds enable native CPU tuning and LTO by default; use `-DISLAY_NATIVE=OFF` for a portable binary. The remaining commands start the protocol loop, run all built-in self-tests, and perform a reproducible perft check.

## Coding Style & Naming Conventions

Format C++ with the repository’s `.clang-format` (LLVM-based, two-space indentation, no tabs, 120-column limit), for example:

```sh
clang-format -i main.cpp src/*.cpp src/*.hpp
```

Use lowercase filenames, `PascalCase` for types, `snake_case` for functions and variables, `kCamelCase` for constants, and trailing underscores for private data members. Preserve the existing `islay` namespace and include ordering. Prefer focused changes over unrelated cleanup.

## Testing Guidelines

There is no separate `tests/` tree or CTest target. Add deterministic module checks to the relevant `*_selftest()` function and ensure the aggregate command ends with `ALL TESTS PASSED`. Move-generation, perft, and exact-search changes must preserve known counts and scores. For performance or playing-strength changes, report the build/options, workload, node or time comparison, and paired A/B results (Elo, confidence interval, and LOS); do not infer strength from node reduction alone.

## Commit & Pull Request Guidelines

Recent commits use concise, outcome-oriented subjects, sometimes scoped (`train(nnue): ...`, `docs(UCI): ...`). Keep each commit focused and explain measured tradeoffs in the body. Pull requests should state the rationale, list exact validation commands and results, link relevant issues, and update `UCI.md` when public protocol behavior changes. Include benchmark settings and before/after data for search, evaluation, or move-generation work.
