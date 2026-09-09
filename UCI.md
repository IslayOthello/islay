# islay UCI Protocol

This document describes the UCI-style Othello/Reversi interface implemented by
`islay 0.1.0`. The engine supports position setup, perft and experimental
single-worker Othello PUCT search. Search currently uses an explicitly advertised
uniform policy/zero-value evaluator, not a trained neural network.
Full clock controls and learned inference are not implemented yet.

## Transport

Commands are read one line at a time from standard input; responses are written
to standard output and flushed after each command. Blank lines are ignored.
Command names are case-sensitive. Option names and rule values are
case-insensitive. `quit`, `exit`, and end-of-file end the session.

The startup banner is:

```text
islay 0.1.0 - Othello MCTS / Reversi perft engine (movegen backend: <backend>)
type 'uci', 'position', 'go nodes <N>', 'go perft <N>', or 'quit'
```

Perft and debug commands execute synchronously. Commands received during perft,
including `isready`, `stop`, and `quit`, are processed after it finishes.
PUCT uses one background worker, so `isready` and `stop` remain responsive during search.
Perft never runs concurrently with that worker. Protocol output is serialized in complete blocks.

## Quick Start

```text
uci
isready
position startpos
go perft 8
quit
```

The perft response includes these deterministic counts:

```text
d3: 97554
c4: 97554
f5: 97554
e6: 97554

Nodes searched: 390,216
```

Timing and speed lines follow; their values depend on the machine.

## Session Commands

### uci

Returns engine identification and all supported options:

```text
id name islay 0.1.0
id author islay
option name Rule type combo default Othello var Othello var Reversi
option name MctsHash type spin default 64 min 1 max 4096
option name PerftHash type spin default 256 min 1 max 65536
uciok
```

### isready

Returns `readyok`.

### ucinewgame

Resets the board to the standard opening with Black to move and clears the
perft cache. Options are preserved. Produces no response on success.

### stop

Stops and joins a PUCT search, publishing its final result exactly once. Repeated `stop`
does not publish another `bestmove`. It cannot interrupt synchronous perft.

### quit / exit

Cancels and joins search, suppressing any result not already published, then ends the session.
End-of-file has the same effect. To obtain a result before exiting, send `stop` and wait for
`bestmove`; piping `go ...` immediately followed by `quit` may discard that search.

## Options

```text
setoption name <name> value <value>
```

| Name | Type | Default | Values |
|---|---|---|---|
| Rule | combo | Othello | Othello, Reversi |
| MctsHash | spin | 64 | Tree arena MiB, 1 to 4096; independent of PerftHash |
| PerftHash | spin | 256 | Integer MiB from 1 to 65536 |

Changing an option clears the cache. Changing `PerftHash` resizes it.
Allocation is rounded down to a power-of-two number of entries.

```text
setoption name Rule value Reversi
info string option Rule = Reversi
```

Unknown options and invalid values produce:

```text
info error: unknown option or invalid value: '<name>' = '<value>'
```

Search options (`Hash`, `Threads`, `CorrectionHistory`), evaluator options
(`EvalFile`, `StageInterpolation`), and book options (`OwnBook`, `BookFile`)
have been removed.

## Positions

```text
position startpos [moves <move> ...]
position fen <diagram> <side-to-move> [moves <move> ...]
```

`startpos` selects the standard opening with Black to move.
The `fen` form uses a single 64-character board token, not chess FEN.
Cells are ordered `a1, b1, ..., h1, a2, ..., h8`.

| Glyphs | Meaning |
|---|---|
| X, x, * | Black disc |
| O, o, 0 | White disc |
| -, ., _ | Empty square |

Side to move accepts `X/x/B/b` for Black and `O/o/W/w` for White.

```text
position fen ---------------------------OX------XO--------------------------- X
position startpos moves d3 c3
```

Moves use coordinates `a1` through `h8`; file letters may be uppercase.
Pass tokens are `pass`, `PASS`, `--`, and `@@`.
A pass is legal only under Othello rules when the mover has no legal move and
the opponent does. Reversi does not permit passes.

Positions are applied only after all supplied moves validate. An illegal move,
illegal pass, or invalid diagram leaves the previous position unchanged.
Successful commands produce no output. Errors begin with `info error:`.

## Perft

```text
go perft <depth> [nocache]
```

Depth must be a non-negative integer. The current board and `Rule` option
determine the tree. Cache is enabled by default; `nocache` bypasses it.

- Depth zero returns one node.
- Positive depths count sequences reaching exactly the requested ply.
- Othello passes consume one ply if the opponent can move.
- Reversi ends a branch as soon as the mover has no legal move.
- When neither player can move, the branch contributes zero at positive depth.

For positive depth, the response contains each legal root move and its count,
or a `pass:` line, or `(game over)`. It ends with:

```text
Nodes searched: <nodes>
Time: <seconds> s
Speed: <nodes-per-second> N/s
```

`Nodes searched` and `Speed` use commas as thousands separators, for example
`390,216` and `12,345,678 N/s`. Clients parsing these fields must strip commas.
Root-move counts and `Time` remain ungrouped.

`Time` uses seconds with nine decimal places, for example `0.000589375 s`.
Timing uses `steady_clock` and retains fractional microseconds internally.
`Speed` remains nodes per second, rounded to an integer, and is computed from
the unrounded elapsed time: `nodes / (elapsed_us / 1,000,000)`.
The displayed precision does not guarantee the clock's effective resolution
or eliminate timing noise in very short runs. Clients reading `Time` must
accept the `s` unit and fractional values instead of integer milliseconds.

Depth zero emits only `Nodes searched: 1` and `Time: 0.000000000 s`.

## PUCT Search (experimental)

```text
go nodes <N>
go movetime <MS>
go nodes <N> movetime <MS>
go infinite
```

`nodes` is a nonnegative uint64 simulation budget. `movetime` is an integer from 0 to
86,400,000 milliseconds. Combined limits may appear in either order; the first reached
limit wins. Duplicates, unknown tokens, negative values and overflow are rejected.
`infinite` cannot be combined with another limit. Bare `go`, `depth`, `wtime`, `btime`,
increments, pondering and `searchmoves` are not supported.

Search requires `Rule=Othello`. Each accepted request first emits:

```text
info string evaluator uniform (P2 scaffold; no trained network)
```

On completion, one block contains `info nodes <N> nps <NPS> time <MS> [pv <moves>]`,
`info string search <reason> value <V> evaluations <E>`, and `bestmove <move>`.
Moves use coordinates or `pass`; a terminal root returns `bestmove 0000` with zero simulations.
Reasons are `nodes`, `time`, `stop`, `memory`, `terminal`, or `error`. The reason describes
why traversal ended, even if an infinite search subsequently waited for `stop`.

- `nodes` counts completed simulations, not perft leaves; root evaluation is not a simulation.
- `nps` is simulations/second, rounded to an integer from microsecond-resolution steady-clock
  elapsed time. `time` is integer milliseconds. Both include worker startup/cleanup and, for
  infinite searches, time waiting for `stop`. Machine-readable fields never contain commas.
- `value` is mover-relative expected outcome in [-1,1], not centipawns or win probability.
- Only final search info is emitted in P2, not periodic progress. There is no root noise.
- Finite searches return early if the arena fills. Infinite searches park without spinning when
  memory or terminal ends traversal, and publish only when stopped. They do not allocate past the cap.
- Search failures emit an explicit diagnostic and a legal fallback (or `0000` if terminal).
  Failure to create the worker emits an error without accepting a search.
- `position`, `ucinewgame`, `setoption`, any new `go`, and enabled debug `test`/`bench` commands
  cancel/join old search before processing, even if the new command is invalid. A result already
  published cannot be retracted; no pending old result is published after the new command is processed.
- Deadline/stop latency includes the current evaluator call; future network backends must honor
  cancellation. The current uniform evaluator is only a protocol/testing scaffold.

Human perft `Time` stays in seconds and its `Nodes searched`/`Speed` remain comma-grouped.
Search does not reuse PerftTT or change perft semantics.

## Debug Commands

`debug on` enables the commands below; `debug off` hides them again.
`debug` reports the current state. The default is off.

| Command | Behavior |
|---|---|
| d / display / board | Print the current board, side to move, disc counts, and legal moves |
| backend | Print the compiled move-generation backend |
| bench [depth] | Uncached start-position perft from depth 1 through depth (default 11), under the selected rule; fractional `time(s)` and integer NPS |
| test / selftest | Run movegen, Othello game adapter, PUCT core/controller, known perft, cache, symmetry, and rule checks |

The test suite ends with `ALL TESTS PASSED` on success.
Run `python3 tools/uci_search_test.py build/islay` for black-box search lifecycle,
parsing, PV legality and output tests, including measured stop round-trip latency.
The former search, evaluation, training, tuning, book, and match debug commands
have been removed. Unknown or disabled commands produce:

```text
info error: unknown command '<command>'
```
