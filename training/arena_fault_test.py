"""Fault injection for bounded UCI reads and fail-closed arena publication."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import arena
from arena_engine import Engine
from arena_game import START
from arena_test import rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    template = '''import os, sys, time
mode = MODE
for command in sys.stdin:
    command = command.strip()
    if command == "uci": print("uciok", flush=True)
    elif command == "isready": print("readyok", flush=True)
    elif command == "quit": break
    elif command.startswith("go "):
        if mode == "crash": os._exit(1)
        if mode == "hang": time.sleep(60)
        if mode == "oversized": print("x" * 100000, flush=True); continue
        if mode == "error": print("info string search error: injected", flush=True); continue
        print("info string evaluator onnx-cpu b8c64-v1 checkpoint " + "0" * 64 + " training_steps 1")
        print("info nodes 4 nps 1000 time 4 pv d3")
        print("info string search " + ("memory" if mode == "memory" else "nodes") + " value 0.0 evaluations 5")
        print("bestmove " + ("a1" if mode == "illegal" else "d3"))
        if mode == "duplicate": print("bestmove d3")
        sys.stdout.flush()
'''
    cfg = {"mode": "nodes", "budget": 4, "timeout_ms": 100, "grace_ms": 250}
    for mode in ("crash", "hang", "oversized", "error", "memory", "illegal", "duplicate"):
        script = args.output / (mode + ".py")
        script.write_text("#!" + sys.executable + "\n" + template.replace("MODE", repr(mode)))
        script.chmod(0o700)
        engine = Engine(script)
        try: rejected(lambda: engine.search(START, cfg))
        finally: engine.close(abort=True)

    config = argparse.Namespace(engine=args.engine, candidate=args.model, champion=args.model,
                                output=args.output / "failed-run", pairs=2, seed=1, opening_plies=12,
                                mode="nodes", budget=4, tree_mib=1, timeout_ms=30000, grace_ms=250,
                                threshold=0.55, alpha=0.05, max_mib=256, max_new_pairs=None, resume=False)
    with patch("arena.Engine", side_effect=RuntimeError("injected engine initialization failure")):
        rejected(lambda: arena.run(config))
    failure = json.loads((config.output / "failure.json").read_text())
    assert failure["committed_pairs"] == 0 and not list(config.output.glob("pair-*"))
    assert not (config.output / "final").exists()
    config.resume = True
    rejected(lambda: arena.run(config))
    config.resume, config.output, config.max_mib = False, args.output / "capped", 1
    rejected(lambda: arena.run(config))
    print(json.dumps({"result": "ARENA FAULT TESTS PASSED", "protocol_faults": 7,
                      "failed_match": "recorded, no selection, no silent retry"}))


if __name__ == "__main__":
    main()
