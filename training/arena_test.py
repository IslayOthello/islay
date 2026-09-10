"""Arena rules, paired statistics, real UCI games, fixed-plan resume and corruption gates."""
import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import arena
from arena_engine import Engine
from arena_game import START, actions, advance, canonical, openings, verify_game
from arena_stats import summarize


def rejected(function):
    try: function()
    except (ValueError, RuntimeError, TimeoutError): pass
    else: raise AssertionError("invalid arena input accepted")


def trajectory(root):
    result = []
    for path in sorted(root.glob("pair-*/pair.json")):
        pair = json.loads(path.read_text())
        result.append([(g["candidate_color"], g["score"], g["black_margin"],
                        [(r["action"], r["nodes"], r["evaluations"]) for r in g["moves"]]) for g in pair["games"]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert actions((2, 1)) == {64: 0}
    assert advance((2, 1), 64) == (1, 2)
    assert advance((1, 2), 2) == (0, 7)
    assert not actions((0, 7)) and not actions((0xffffffff, 0xffffffff00000000))
    rejected(lambda: advance(START, 64))
    rejected(lambda: advance((0, 7), 64))
    corpus = openings(200, 20260910, 12)
    assert corpus == openings(200, 20260910, 12)
    assert len({canonical(tuple(o["board"])) for o in corpus}) >= 180
    for opening in corpus:
        board = START
        for action in opening["moves"]: board = advance(board, action)
        assert list(board) == opening["board"] and actions(board)
    rng, outcomes, passes = random.Random(37), set(), 0
    for opening in corpus:
        board, moves = tuple(opening["board"]), []
        while actions(board):
            action = rng.choice(list(actions(board)))
            passes += action == 64
            moves.append({"action": action})
            board = advance(board, action)
        side = (opening["side"] + len(moves)) % 2
        margin = (board[0].bit_count() - board[1].bit_count()) * (1 if side == 0 else -1)
        score = (1 + (margin > 0) - (margin < 0)) / 2
        for color in (0, 1):
            game = {"candidate_color": color, "moves": moves, "black_margin": margin,
                    "score": score if color == 0 else 1 - score}
            assert verify_game(opening, game) == game["score"]
        outcomes.add(score)
    assert outcomes == {0, 0.5, 1} and passes > 0

    def pairs(scores): return [{"games": [{"score": a}, {"score": b}]} for a, b in scores]
    balanced = summarize(pairs([(1, 0)] * 200), 200, 1, 200)
    assert balanced["elo"] == 0 and balanced["paired_normal_los"] == 0.5
    assert balanced["pentanomial"] == [0, 0, 200, 0, 0] and balanced["decision"] == "retain_champion"
    wins = summarize(pairs([(1, 1)] * 200), 200, 1, 200)
    assert wins["decision"] == "candidate_eligible" and wins["elo"] == "+inf"
    assert wins["paired_normal_los"] is None  # No invented certainty for zero observed variance.
    assert summarize(pairs([(1, 1)] * 20), 20, 1, 20)["decision"] == "retain_champion"
    assert summarize(pairs([(1, 1)] * 200), 201, 1, 200)["decision"] == "retain_champion"
    assert summarize(pairs([(1, 1)] * 200), 200, 1, 1)["decision"] == "retain_champion"
    assert summarize(pairs([(1, 1)] * 127 + [(0, 0)] * 73), 200, 1, 200)["decision"] == "retain_champion"
    assert summarize(pairs([(1, 1)] * 128 + [(0, 0)] * 72), 200, 1, 200)["decision"] == "candidate_eligible"
    mixed = [(1, 1), (1, 0.5), (0.5, 0.5), (0, 0.5), (0, 0)] * 50
    a = summarize(pairs(mixed), 250, 9, 250)
    b = summarize(pairs([(1 - x, 1 - y) for x, y in mixed]), 250, 9, 250)
    assert a["score"] == 1 - b["score"] and a["wdl"]["wins"] == b["wdl"]["losses"]
    assert a["paired_bootstrap_score_ci95"][0] < 0.5 < a["paired_bootstrap_score_ci95"][1]

    common = [sys.executable, str(Path(__file__).with_name("arena.py")), "--engine", str(args.engine),
              "--candidate", str(args.model), "--champion", str(args.model), "--pairs", "4", "--budget", "4"]
    def launch(root, *extra, success=True):
        r = subprocess.run([*common, "--output", str(root), *extra], text=True, capture_output=True, timeout=120)
        assert (r.returncode == 0) == success, r.stdout + r.stderr
        return r
    split, whole = args.output / "split", args.output / "whole"
    launch(split, "--max-new-pairs", "2")
    assert not (split / "final").exists()
    committed = {p: p.read_bytes() for p in split.glob("pair-*/*")}
    (split / ".pending-test").mkdir()
    (split / ".pending-test" / "pair.json").write_text("partial")
    launch(split, "--resume")
    assert all(p.read_bytes() == data for p, data in committed.items())
    launch(whole)
    assert trajectory(split) == trajectory(whole)
    result = json.loads((split / "final" / "report.json").read_text())
    assert result["complete"] and result["score"] == 0.5 and result["decision"] == "retain_champion"
    for pair in trajectory(split):
        assert pair[0][3] == pair[1][3] and pair[0][1] + pair[1][1] == 1
    launch(split, "--resume")
    launch(split, "--resume", "--pairs", "5", success=False)
    launch(split, "--resume", "--budget", "5", success=False)

    meta = json.loads((split / "assets" / "manifest.json").read_text())
    records = arena.load_pairs(split, meta)
    invalid = copy.deepcopy(records[0])
    invalid["games"][1]["candidate_color"] = invalid["games"][0]["candidate_color"]
    rejected(lambda: arena.verify_pair(invalid, meta, 0))
    invalid = copy.deepcopy(records[0])
    invalid["games"][0]["score"] = 0.25
    rejected(lambda: arena.verify_pair(invalid, meta, 0))
    invalid = copy.deepcopy(records[0])
    invalid["games"][0]["moves"][0]["nodes"] = 3
    rejected(lambda: arena.verify_pair(invalid, meta, 0))
    invalid["games"][0]["moves"][0]["nodes"] = 4
    invalid["games"][0]["moves"][0]["evaluator"] = "uniform"
    rejected(lambda: arena.verify_pair(invalid, meta, 0))
    invalid = copy.deepcopy(records[0])
    invalid["games"][0]["moves"][0]["elapsed_us"] = 1e20
    rejected(lambda: arena.verify_pair(invalid, meta, 0))
    path = split / "pair-000000" / "pair.json"
    path.write_text("corrupt")
    rejected(lambda: arena.load_pairs(split, meta))
    launch(split, "--resume", success=False)

    engine = Engine(args.engine)
    try:
        engine.configure(args.model.resolve(), 1)
        cfg = {"mode": "nodes", "budget": 4, "timeout_ms": 30000, "grace_ms": 250}
        assert engine.search((2, 1), cfg)["action"] == 64
        assert engine.search((1, 2), cfg)["action"] == 2
        rejected(lambda: engine.search(START, {**cfg, "budget": 1000000000}))  # Memory cap, not a valid result.
    finally: engine.close(abort=True)

    timed = args.output / "timed"
    launch(timed, "--pairs", "2", "--mode", "movetime", "--budget", "2")
    timing = json.loads((timed / "final" / "report.json").read_text())
    assert timing["mode"] == "movetime" and timing["decision"] == "retain_champion"
    print(json.dumps({"result": "ARENA TESTS PASSED", "resume": "identical node-mode moves/results/evaluations",
                      "self_match_score": result["score"], "unique_openings_200": len({canonical(tuple(o["board"])) for o in corpus}),
                      "time_mode_pairs": timing["pairs"]}))


if __name__ == "__main__":
    main()
