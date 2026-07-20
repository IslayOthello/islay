# islay UCI Protocol

This document describes the released text protocol implemented by `islay 0.1.0`.
islay provides a UCI-style interface for Othello and Reversi.

Everything a client needs in order to play a game is specified here. Commands
that are not listed in this document are not part of the released protocol and
must not be relied upon.

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
type 'uci', 'position', 'go depth <N>', 'go perft <N>', or 'quit'
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

## 3. Commands

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
option name StageInterpolation type check default true
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
Black to move, and all transposition tables and search state are cleared.

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

The command is transactional: if any token is malformed or any move is illegal,
the entire command is rejected and the previous position is left unchanged.

#### Errors

```text
info error: 'position fen' needs <diagram> <stm>
info error: invalid diagram/side-to-move
info error: expected 'startpos' or 'fen'
info error: illegal move '<move>'
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
| `EvalFile` | `string` | Empty | Path to an ISLAYPAT pattern-weight file; an empty value restores the built-in evaluation |
| `StageInterpolation` | `check` | `true` | Linearly interpolates the pattern evaluation across game-stage boundaries |
| `Hash` | `spin` | `256` | Search transposition-table size in MiB; range `1`–`65536` |
| `PerftHash` | `spin` | `256` | Transposition-table size in MiB for `go perft`; range `1`–`65536` |

Examples:

```text
setoption name Rule value Reversi
setoption name Hash value 512
setoption name EvalFile value weights/v12.pat
setoption name StageInterpolation value false
```

After a valid option change, the transposition tables are invalidated and the
engine reports the change:

```text
info string option Rule = Reversi
```

When a non-empty `EvalFile` is set, the engine also reports whether the
pattern weights were loaded. Setting it to an empty value unloads the weights
and restores the built-in evaluation; a failed load also unloads rather than
leaving stale weights in place:

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

A malformed or out-of-range limit rejects the whole command rather than
starting an unintended search:

```text
info error: depth must be positive
info error: movetime must be non-negative
info error: nodes must be non-negative
```

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

If the search depth reaches the number of empty squares, the score is the exact
game-theoretic result rather than an estimate:

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

#### Node counting

`perft` is an argument of `go` rather than a separate command. It counts the
leaf nodes of the move tree to the given depth from the current position, which
is the standard way to verify move generation.

```text
go perft <depth>
go perft <depth> nocache
```

Results are transposition-cached by default; `nocache` disables the cache, which
is slower but independent of it. The response lists the node count for each root
move, then the totals:

```text
d3: 14
c4: 14
f5: 14
e6: 14

Nodes searched: 56
Time: 0 ms
Speed: 3873288 N/s
```

A missing depth is rejected:

```text
info error: 'go perft' needs a depth
```

The `Rule` option affects the counts, since Othello passes and Reversi does not.
The size of the cache is set by `PerftHash`.

### `debug`

Turns the engine's development mode on or off. It is off when the engine
starts.

Development mode exposes additional internal tooling used to build and validate
the engine. That tooling is intentionally undocumented, is not part of the
released protocol, and may change or disappear between versions without notice.
A client implementing the protocol described in this document never needs it.

#### Syntax

```text
debug on
debug off
debug
```

A bare `debug` reports the current state without changing it.

#### Response

```text
info string debug on
```

Invalid input produces:

```text
info error: expected 'debug on' or 'debug off'
```

### `quit` and `exit`

Both commands terminate the protocol loop:

```text
quit
exit
```

## 4. Othello and Reversi Rules

The `Rule` option selects the rule set:

- **Othello**: if the side to move has no legal move but the opponent can move,
  the side passes. The game ends only when neither side can move.
- **Reversi**: if the side to move has no legal move, the game ends immediately;
  passes are not used.

## 5. Error and Informational Output

An unrecognized top-level command produces:

```text
info error: unknown command '<command>'
```

Output beginning with `info error:` indicates a protocol, parsing, or file
processing error. Output beginning with `info string` is engine-specific
informational output. A client that only needs the search result should wait
for `bestmove` after a `go` command.
