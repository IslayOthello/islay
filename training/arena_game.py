"""Independent scalar rules and frozen, model-independent arena openings."""
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from perft_study import START, scalar_moves, play


def actions(board):
    moves = scalar_moves(*board)
    return dict(moves) if moves else ({64: 0} if scalar_moves(board[1], board[0]) else {})


def advance(board, action):
    legal = actions(board)
    if type(action) is not int or action not in legal:
        raise ValueError("illegal arena action")
    return (board[1], board[0]) if action == 64 else play(*board, (action, legal[action]))


def token(action):
    return "pass" if action == 64 else chr(97 + action % 8) + str(1 + action // 8)


def parse_token(move):
    if move == "pass": return 64
    if len(move) != 2 or move[0] not in "abcdefgh" or move[1] not in "12345678":
        raise ValueError("invalid bestmove: " + move)
    return ord(move[0]) - 97 + 8 * (int(move[1]) - 1)


def fen(board):
    return "".join("X" if board[0] >> sq & 1 else "O" if board[1] >> sq & 1 else "-" for sq in range(64))


def canonical(board):
    variants = []
    for symmetry in range(8):
        result = [0, 0]
        for side, bits in enumerate(board):
            for sq in range(64):
                if not bits >> sq & 1: continue
                x, y = sq % 8, sq // 8
                if symmetry & 1: x = 7 - x
                if symmetry & 2: y = 7 - y
                if symmetry & 4: x, y = y, x
                result[side] |= 1 << (8 * y + x)
        variants.append(tuple(result))
    return min(variants)


def openings(count, seed, plies):
    # Independent random playouts with replacement; never select using model scores.
    rng = random.Random(seed)
    result = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 100: raise ValueError("cannot generate nonterminal openings")
        board, moves = START, []
        for _ in range(plies):
            legal = actions(board)
            if not legal: break
            action = rng.choice(list(legal))
            moves.append(action)
            board = advance(board, action)
        if len(moves) == plies and actions(board):
            result.append({"moves": moves, "board": list(board), "side": plies % 2})
    return result


def verify_game(opening, game):
    board = START
    for action in opening["moves"]: board = advance(board, action)
    if list(board) != opening["board"] or opening["side"] != len(opening["moves"]) % 2:
        raise ValueError("opening replay mismatch")
    if type(game["candidate_color"]) is not int or game["candidate_color"] not in (0, 1):
        raise ValueError("invalid color")
    if not 1 <= len(game["moves"]) <= 128: raise ValueError("invalid game length")
    for row in game["moves"]: board = advance(board, row["action"])
    if actions(board): raise ValueError("unfinished arena game")
    side = (opening["side"] + len(game["moves"])) % 2
    margin = board[0].bit_count() - board[1].bit_count()
    black_margin = margin if side == 0 else -margin
    candidate_margin = black_margin if game["candidate_color"] == 0 else -black_margin
    score = (1 + (candidate_margin > 0) - (candidate_margin < 0)) / 2
    if game["black_margin"] != black_margin or game["score"] != score:
        raise ValueError("arena result does not match terminal discs")
    return score
