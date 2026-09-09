# islay UCI Protocol

This document describes the UCI-style Othello/Reversi interface implemented by
`islay 0.1.0`. The engine supports position setup and perft only.
It does not select moves, emit `bestmove`, or support playing games through
a standard UCI match runner.

## Transport

Commands are read one line at a time from standard input; responses are written
to standard output and flushed after each command. Blank lines are ignored.
Command names are case-sensitive. Option names and rule values are
case-insensitive. `quit`, `exit`, and end-of-file end the session.

The startup banner is:

```text
islay 0.1.0 - Othello/Reversi perft engine (movegen backend: <backend>)
type 'uci', 'position', 'go perft <N>', or 'quit'
```

Perft and debug commands execute synchronously. Commands received during perft,
including `isready`, `stop`, and `quit`, are processed after it finishes.
There is no background search thread.

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

Nodes searched: 390216
```

Timing and speed lines follow; their values depend on the machine.

## Session Commands

### uci

Returns engine identification and all supported options:

```text
id name islay 0.1.0
id author islay
option name Rule type combo default Othello var Othello var Reversi
option name PerftHash type spin default 256 min 1 max 65536
uciok
```

### isready

Returns `readyok`.

### ucinewgame

Resets the board to the standard opening with Black to move and clears the
perft cache. Options are preserved. Produces no response on success.

### stop

Accepted as a no-op. It cannot interrupt synchronous perft.

### quit / exit

Ends the session. End-of-file has the same effect.

## Options

```text
setoption name <name> value <value>
```

| Name | Type | Default | Values |
|---|---|---|---|
| Rule | combo | Othello | Othello, Reversi |
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
Time: <milliseconds> ms
Speed: <nodes-per-second> N/s
```

Depth zero emits only `Nodes searched: 1` and `Time: 0 ms`.

Bare `go` and search forms such as `go depth`, `go nodes`, `go movetime`,
`go infinite`, and clock controls are rejected:

```text
info error: only 'go perft <depth> [nocache]' is supported
```

## Debug Commands

`debug on` enables the commands below; `debug off` hides them again.
`debug` reports the current state. The default is off.

| Command | Behavior |
|---|---|
| d / display / board | Print the current board, side to move, disc counts, and legal moves |
| backend | Print the compiled move-generation backend |
| bench [depth] | Uncached start-position perft from depth 1 through depth (default 11), under the selected rule |
| test / selftest | Run movegen, known perft, cache, symmetry, and rule checks |

The test suite ends with `ALL TESTS PASSED` on success.
The former search, evaluation, training, tuning, book, and match debug commands
have been removed. Unknown or disabled commands produce:

```text
info error: unknown command '<command>'
```
