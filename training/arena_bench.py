"""Alternating paired arena throughput: persistent sessions vs reload both models each game."""
import argparse
import json
from pathlib import Path
import statistics
import time

from arena import play_game
from arena_engine import Engine
from arena_game import openings
from replay import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "candidate", "champion", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 3 or not 1 <= args.budget <= 1000000: parser.error("invalid benchmark configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"mode": "nodes", "budget": args.budget, "timeout_ms": 30000, "grace_ms": 250}
    corpus = [openings(1, 20260910 + i, plies)[0] for i, plies in enumerate((12, 24, 40))]

    def create():
        engines = {}
        try:
            for owner in ("candidate", "champion"):
                engines[owner] = Engine(args.engine)
                engines[owner].configure(getattr(args, owner).resolve(), 8)
            return engines
        except BaseException:
            for engine in engines.values(): engine.close(abort=True)
            raise

    def measure(persistent):
        engines = create() if persistent else {}
        signatures = []
        started = time.perf_counter_ns()
        try:
            for opening in corpus:
                for color in (0, 1):
                    if not persistent: engines = create()
                    game = play_game(engines, opening, color, config)
                    signatures.append([game["score"], game["black_margin"], [r["action"] for r in game["moves"]]])
                    if not persistent:
                        for engine in engines.values(): engine.close()
                        engines = {}
            elapsed = (time.perf_counter_ns() - started) / 1e6
        finally:
            for engine in engines.values(): engine.close(abort=True)
        return {"milliseconds": elapsed, "games_per_hour": 6 * 3600000 / elapsed}, signatures

    # Untimed full workload warms both variants, including inference and filesystem cache.
    _, expected = measure(True)
    _, other = measure(False)
    assert expected == other
    rounds = []
    for i in range(args.rounds):
        row = {}
        for persistent in ((True, False) if i % 2 == 0 else (False, True)):
            timing, signature = measure(persistent)
            assert signature == expected, "session lifetime changed game result"
            row["persistent" if persistent else "reload_per_game"] = timing
        rounds.append(row)
        print(f"arena paired round {i + 1} complete", flush=True)
    result = {"engine_sha256": sha256(args.engine), "candidate_sha256": sha256(args.candidate),
              "champion_sha256": sha256(args.champion), "config": config, "openings": corpus, "rounds": rounds,
              "median_paired_speedup": statistics.median(r["reload_per_game"]["milliseconds"] /
                                                         r["persistent"]["milliseconds"] for r in rounds)}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
