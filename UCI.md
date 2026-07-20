# islay UCI Protocol

This document describes the text protocol implemented by `islay 0.1.0`.
islay provides a UCI-style interface for Othello and Reversi. In addition to
the core UCI commands, it exposes engine-specific commands for perft,
benchmarking, self-tests, engine matches, feature inspection, and training.

## 1. Protocol Overview

The protocol is line-oriented:

- Commands are read from standard input (`stdin`), one command per line.
- Responses are written to standard output (`stdout`).
- Empty input lines are ignored.
- Output is flushed after every command.
- Commands are case-sensitive and must use the spelling documented below.
- Option names and option values are generally matched case-insensitively.
- The engine exits on `quit`, `exit`, or end-of-file.

When the executable starts, it prints a banner before accepting commands:

```text
islay 0.1.0 - Othello/Reversi engine (movegen backend: <backend>)
type 'uci', 'go perft <depth> [nocache]', 'd', 'test', or 'quit'
```

The protocol loop is synchronous. A search command blocks the command loop
until it completes. The `stop` command is not implemented.

## 2. Quick Start

```text
uci
isready
ucinewgame
position startpos moves d3 C3
go depth 8
quit
```

A typical search response contains one `info` line per completed iteration,
followed by a final status line and `bestmove`:

```text
info depth 1 seldepth 1 score cd -177 nodes 4 nps 5864 hashfull 0 time 0 pv d3
info depth 2 seldepth 2 score cd -63 nodes 14 nps 6455 hashfull 0 time 2 pv d3 C3
info string heuristic score, depth 2
bestmove d3
```

## 3. Core UCI Commands

### `uci`

Requests the engine identification and the list of supported options.

#### Syntax

```text
uci
```

#### Response

```text
id name islay 0.1.0
id author islay
option name Rule type combo default Othello var Othello var Reversi
option name EvalFile type string default
option name PerftHash type spin default 256 min 1 max 65536
option name Hash type spin default 256 min 1 max 65536
uciok
```

### `isready`

Checks whether the engine is ready to accept another command.

#### Syntax

```text
isready
```

#### Response

```text
readyok
```

### `ucinewgame`

Starts a new game. The board is reset to the standard opening position with
Black to move. The perft transposition table and all search state are cleared.

#### Syntax

```text
ucinewgame
```

There is no response on success.

### `position`

Sets the current board position and optionally applies a sequence of moves.

#### Syntax

```text
position startpos [moves <move> ...]
position fen <diagram> <side-to-move> [moves <move> ...]
```

#### Standard opening

`startpos` selects the standard Othello opening with Black to move:

```text
position startpos
position startpos moves d3 C3
```

#### Custom position format

The `fen` form does not use chess FEN. `<diagram>` is a 64-cell board
description in the following order:

```text
a1, b1, ..., h1, a2, ..., h2, ..., a8, ..., h8
```

Whitespace in the diagram is ignored. The accepted cell glyphs are:

| Glyphs | Meaning |
|---|---|
| `X`, `x`, `*` | Black disc |
| `O`, `o`, `0` | White disc |
| `-`, `.`, `_` | Empty square |

The side-to-move token accepts:

| Values | Side to move |
|---|---|
| `X`, `x`, `B`, `b` | Black |
| `O`, `o`, `W`, `w` | White |

Example:

```text
position fen ---------------------------OX------XO--------------------------- X
```

#### Move notation

Moves use square coordinates from `a1` through `h8`. File letters are
accepted in either case. The engine emits normal moves in lowercase. A pass is
accepted as any of the following tokens:

```text
pass
PASS
--
@@
```

When `moves` is present, all following tokens are applied from left to right.
A successful `position` command produces no output.

#### Errors

```text
info error: 'position fen' needs <diagram> <stm>
info error: invalid diagram/side-to-move
info error: expected 'startpos' or 'fen'
```

### `setoption`

Sets an engine option. The option name and value may contain spaces. The first
`value` token separates the option name from its value.

#### Syntax

```text
setoption name <name> value <value>
```

#### Options

| Name | Type | Default | Accepted values / effect |
|---|---|---|---|
| `Rule` | `combo` | `Othello` | `Othello` or `Reversi` |
| `EvalFile` | `string` | Empty | Path to an ISLAYPAT pattern-weight file; empty uses the hand-written evaluation |
| `PerftHash` | `spin` | `256` | Perft transposition-table size in MiB; range `1`–`65536` |
| `Hash` | `spin` | `256` | Search transposition-table size in MiB; range `1`–`65536` |

Examples:

```text
setoption name Rule value Reversi
setoption name Hash value 512
setoption name PerftHash value 1024
setoption name EvalFile value weights/v12.pat
```

After a valid option change, both transposition tables are invalidated and the
engine reports the change:

```text
info string option Rule = Reversi
```

When a non-empty `EvalFile` is set, the engine also reports whether the
pattern weights were loaded:

```text
info string pattern weights loaded: weights/v12.pat (v<version>, <stages> stages x <weights> weights)
info error: cannot open pattern weights 'weights/missing.pat'
```

Invalid input produces one of these errors:

```text
info error: expected 'setoption name <Name> value <Value>'
info error: unknown option or invalid value: '<name>' = '<value>'
```

### `go`

Starts a search from the current position.

#### Syntax

```text
go
go depth <N>
go movetime <MS>
go nodes <N>
go depth <N> movetime <MS> nodes <N>
```

The limits may be combined. A value of `0` means that the corresponding limit
is disabled. A bare `go` defaults to `depth 8`.

`go infinite` is not supported. The engine reports the condition and then
uses the normal default if no other limit is specified:

```text
info string 'go infinite' is not supported; use depth/movetime/nodes
```

#### Iteration information

The engine emits one line for each completed iterative-deepening iteration:

```text
info depth <depth> seldepth <seldepth> score cd <score> nodes <nodes> nps <nps> hashfull <permille> time <ms> pv <move> ...
```

| Field | Description |
|---|---|
| `depth` | Nominal depth of the completed iteration |
| `seldepth` | Maximum search ply reached during the iteration |
| `score cd` | Score in centi-discs from the side-to-move perspective; `100 cd = 1 disc` |
| `nodes` | Number of searched nodes |
| `nps` | Nodes per second |
| `hashfull` | Search-table occupancy in per mille |
| `time` | Elapsed search time in milliseconds |
| `pv` | Principal variation; Black moves are lowercase and White moves uppercase |

#### Final response

For a normal search, the final response is:

```text
info string heuristic score, depth <depth>
bestmove <move>
```

If the search reaches the number of empty squares, the result is exact:

```text
info string exact score, depth <depth>
bestmove <move>
```

If the side to move must pass under Othello:

```text
bestmove pass
```

If the game is over:

```text
info string game over (final score <score-in-discs>)
bestmove --
```

### `quit` and `exit`

Both commands terminate the protocol loop:

```text
quit
exit
```

## 4. Perft and Benchmarking

### `go perft`

Runs bulk-counting perft from the current position. The active `Rule` option
is used. The perft cache is enabled by default.

#### Syntax

```text
go perft <depth> [nocache]
```

Use `nocache` to disable the perft transposition table.

#### Response

For the standard opening at depth 2:

```text
d3: 3
c4: 3
f5: 3
e6: 3

Nodes searched: 12
Time: 0 ms
Speed: 120000 N/s
```

Each line before the summary contains the node count for one root move. If a
pass is required, the root entry is `pass: <count>`. If the game is over, the
root entry is `(game over)`.

If the depth is missing:

```text
info error: 'go perft' needs a depth
```

For a depth below `1`:

```text
Nodes searched: 1
Time: 0 ms
```

### `bench [depth]`

Runs uncached perft from the standard opening for every depth from `1` through
the requested maximum depth. The default maximum depth is `11`.

```text
bench 8
```

Response format:

```text
depth            nodes       time(ms)             nps
---------------------------------------------------------
    1                4              0           <nps> N/s
...
```

## 5. Othello and Reversi Rules

The `Rule` option affects both search and perft:

- **Othello**: if the side to move has no legal move but the opponent can move,
  the side passes. The game ends only when neither side can move.
- **Reversi**: if the side to move has no legal move, the game ends immediately;
  passes are not used.

## 6. Engine-Specific Commands

### `d`, `display`, and `board`

These commands are aliases. Each prints the current board, legal moves, side to
move, and disc counts:

```text
d
display
board
```

Legal moves are marked with `*`:

```text
  a b c d e f g h
1 - - - - - - - -
...
Black to move  |  X:2 O:2  |  moves:4
```

### `test` and `selftest`

These aliases run the move-generation, evaluation, search, pattern, cache,
symmetry, and known-perft checks:

```text
test
selftest
```

The final status is one of:

```text
ALL TESTS PASSED
TESTS FAILED
```

### `backend`

Reports the active move-generation backend:

```text
backend
```

Response:

```text
movegen backend: <backend>
```

### `features`

Reports the pattern stage, the number of weights per stage, whether pattern
evaluation is active, and the feature indices for the current position:

```text
features
```

Response format:

```text
stage <stage> weights/stage <count> active yes
features <index> <index> ...
```

When the hand-written evaluation is active, the first line contains:

```text
active no (hand-written eval in use)
```

### `pcdata [positions] [maxdepth]`

Generates ProbCut fitting data from random positions. Defaults are `200`
positions and maximum depth `12`.

```text
pcdata
pcdata 500 10
```

The output is CSV followed by a summary line:

```text
stage,depth,score
<stage>,<depth>,<score>
...
info string pcdata: <rows> rows
```

The `score` column is measured in centi-discs.

### `searchstats [full]`

Dumps opt-in search telemetry for the most recent `go`: interior search nodes
bucketed by remaining depth and game stage, with per-bucket TT hit/cut rate,
ordering quality (fail-high-on-first-move rate and mean cutoff move index),
effective branching factor, LMR reduction/re-search rate, futility and ProbCut
attempt/cut rates, PVS re-search rate, and the total node cost of ProbCut probes.

```text
searchstats
searchstats full
```

The telemetry is a **measurement build only**: it is compiled out by default
(`kStats` in `src/search.cpp` is `false`), so a shipping engine pays nothing.
To use it, set `kStats = true`, rebuild, run `go depth N`, then read the dump
(it is also printed automatically after each `go` in such a build). With
`kStats = false` the command reports that telemetry is compiled out. Counters are
pure observation and never change a score, so the oracle self-test is unaffected.

### `match`

Runs an engine-vs-engine match. Each randomly selected opening is played twice,
with the sides swapped in the second game. The total number of games is
`2 * pairs`.

#### Evaluation-file comparison

```text
match [pairs] [depth] [evalA] [evalB]
```

Defaults are `50` pairs and depth `6`. An omitted evaluation path or `-`
selects the hand-written evaluation.

#### Search-configuration comparisons

```text
match pc  [pairs] [movetime_ms] [eval]
match pcg [pairs] [movetime_ms] [eval] [seed]
match lmp [pairs] [movetime_ms] [eval]
match lmr [pairs] [movetime_ms] [eval]
match mpc [pairs] [movetime_ms] [eval]
match pct [pairs] [t_a] [t_b] [movetime_ms] [eval]
```

The time-based variants default to `50` pairs and `50 ms`. For `pct`,
`t_a` defaults to `1.0` and `t_b` defaults to `1.5`.

`pcg` compares the **ProbCut probe gate** on (side A) against off (side B) with
ProbCut itself enabled on both sides, so it isolates the gate alone. The gate
uses the static evaluation to skip a depth-`d-2` probe that is overwhelmingly
unlikely to cut; search telemetry showed roughly 62% of probe attempts cut
nothing while the probes accounted for about 68% of all searched nodes.

The gate is **enabled by default**: it measured +25 Elo, 95% CI [9, 40],
z = 3.18 over 600 pairs at equal time, and it raises the completed depth reached
in a fixed time budget. It is exact-preserving, since declining a probabilistic
cut only ever gives a node a real search instead.

#### Response format

Progress is reported periodically, followed by a final statistical summary:

```text
info string match: A=<evalA>  B=<evalB>  <games> games, depth <depth>, rule <Rule>
info string match: <games> games, A +<wins> =<draws> -<losses>  (<percent>%)
match done: <games> games  A +<wins> =<draws> -<losses>
  score <score> +/- <stderr>   z <z>
  pairing: within-pair rho <rho>  (naive per-game SE would be <se>, off by sqrt(1+rho))
  elo   <elo>  95% CI [<elo_lo>, <elo_hi>]
  verdict: <verdict>
```

If both sides are configured identically, the engine may also report:

```text
info string note: both sides are identical -- expect ~50% (this is the harness's own sanity check)
```

### `train`

Generates self-play games, fits pattern weights, and writes an ISLAYPAT file.

#### Syntax

```text
train [games] [epochs] [depth] [solve_empties] [lr] [l2] [use_mobility] [use_c2x5] [out]
```

The command parser uses these defaults:

| Parameter | Default |
|---|---:|
| `games` | `50000` |
| `epochs` | `8` |
| `depth` | `2` |
| `solve_empties` | `12` |
| `lr` | `0.0005` |
| `l2` | `0.000001` |
| `use_mobility` | `1` |
| `use_c2x5` | `1` |
| `out` | `islay.pat` |

The two feature switches are positional integers: `0` disables the feature
group and any non-zero value enables it.

Example:

```text
train 50000 8 2 12 0.0005 0.000001 1 1 weights/new.pat
```

Training progress is reported with `info string train: ...` lines. A
successful run ends with:

```text
train done: <games> games, <positions> positions/epoch, rmse <rmse> cd -> <out>
  NOTE: rmse is training-set fit, not strength. Settle it with:
        match 100 4 <out> -
```

## 7. Error and Informational Output

An unrecognized top-level command produces:

```text
info error: unknown command '<command>'
```

Output beginning with `info error:` indicates a protocol, parsing, or file
processing error. Output beginning with `info string` is engine-specific
informational output. A client that only needs the search result should wait
for `bestmove` after a `go` command.


---

## Revision notes (protocol correctness, reproducibility, and the StageInterpolation option)

### `StageInterpolation` option (new)

```text
option name StageInterpolation type check default false
```

Linearly interpolates the pattern evaluation across stage boundaries. The
weights are stored per 4-disc stage; with this off the eval switches stages
abruptly, and with it on the score blends toward the next stage by
`(discs - 4) mod 4`. Measured **+58 and +66 Elo** (fixed depth 6, two seeds,
300 games each) on `weights/v12.pat`. Default `false` keeps evaluation
byte-for-byte identical to the baseline, so it is opt-in:

```text
setoption name StageInterpolation value true
```

Antisymmetry and the exact per-stage values at bucket boundaries are covered by
the pattern self-test.

### `EvalFile` load / unload (behavior)

- A non-empty value loads that ISLAYPAT file.
- An **empty** value UNLOADS the pattern weights and restores the hand-written
  evaluator; `features` then reports `active no (hand-written eval in use)`.
- A failed load also unloads (a defined state) rather than keeping stale weights.

### `position` and `go` are now validated

- `position ... moves` is transactional: every move (and any `pass`) is checked
  legal on a temporary board; one bad token rejects the whole command with
  `info error: illegal move '<m>'` or `info error: illegal pass`, and the
  previous position is left untouched. A `pass` is legal only when the mover is
  stuck and, under Othello, the opponent can still move.
- `go` rejects malformed or out-of-range limits (`depth must be positive`,
  `movetime must be non-negative`, `nodes must be non-negative`) instead of
  silently starting a search. A bare `go` is still `depth 8`; `go infinite`
  remains unsupported.

### Reproducibility: seeds and the training default

- `train` now defaults to **depth 4** (the measured strength default) when the
  depth argument is omitted; pass an explicit depth to override.
- `train [...] [out] [seed]` and `match [pairs] [depth] [evalA] [evalB] [seed]`
  accept an optional trailing seed. The effective seed is printed in the opening
  status line, and the same command + seed reproduces the openings and result.

### Development A/B match subcommands

`match` has isolation subcommands for measuring one search feature at equal cost
(all take an optional trailing eval and, where shown, a seed): `pc` (ProbCut),
`pct` (ProbCut t-sweep), `mpc` (per-stage ProbCut), `lmr`, `lmp`, `si`
(StageInterpolation, fixed depth), `eg` / `egd` (endgame stack, equal-time /
fixed depth), and `egtc` (endgame under a clock time control `base_ms inc_ms`).
These are engine-development tools, not part of the core UCI surface.
