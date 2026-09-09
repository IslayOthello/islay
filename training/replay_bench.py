"""Alternating paired self-play/cache and scalar/vectorized replay benchmarks."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from replay import Replay, decode, sha256


def scalar_decode(records):
    boards, legal, policy, outcomes, priors, actions, games, plies = [], [], [], [], [], [], [], []
    for r in records:
        boards.append([[float((int(r[name]) >> a) & 1) for a in range(64)] for name in ("player", "opponent")])
        legal.append([bool(int(r["legal"]) >> a & 1) for a in range(64)] + [bool(r["pass"])])
        visits = r["visits"].astype(np.float32)
        pi = visits / visits.sum()
        if not r["temperature"]:
            pi[:] = 0
            pi[visits.argmax()] = 1
        policy.append(pi)
        outcomes.append([float(r["outcome"])])
        priors.append(r["priors"])
        actions.append(int(r["action"]))
        games.append(r["game_id"])
        plies.append(r["ply"])
    return {"board": np.asarray(boards, np.float32).reshape(-1, 2, 8, 8), "legal": np.asarray(legal, bool),
            "policy": np.asarray(policy, np.float32), "outcome": np.asarray(outcomes, np.float32),
            "priors": np.asarray(priors, np.float32), "action": np.asarray(actions, np.int64),
            "game_id": np.asarray(games, np.uint64), "ply": np.asarray(plies, np.uint16)}


def timed(function):
    start = time.perf_counter_ns()
    result = function()
    return (time.perf_counter_ns() - start) / 1000, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=32)
    args = parser.parse_args()
    if args.rounds < 3 or args.games < 1 or args.simulations < 1: parser.error("invalid benchmark sizes")
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for r in range(args.rounds):
        row, payloads = {}, []
        for cache in ((0, 16) if r % 2 == 0 else (16, 0)):
            command = [str(args.engine), "--model", str(args.model), "--games", str(args.games),
                       "--start", str(r * args.games), "--simulations", str(args.simulations),
                       "--cache-mib", str(cache)]
            wall, result = timed(lambda: subprocess.run(command, check=True, capture_output=True, timeout=600))
            metrics = json.loads(result.stderr)
            metrics["wall_us"] = wall
            row[str(cache)] = metrics
            payloads.append(result.stdout)
        assert payloads[0] == payloads[1], "cache changed self-play trajectory"
        row["speedup_percent"] = (row["0"]["wall_us"] / row["16"]["wall_us"] - 1) * 100
        pairs.append(row)
        print(f"paired self-play round {r + 1}: {row['speedup_percent']:.2f}%", flush=True)
    replay = Replay(args.replay)
    rng = np.random.default_rng(20260909)
    batches = []
    for size in (128, 1024):
        scalar, vector, augmented = [], [], []
        for repeat in range(33):
            indices = rng.choice(replay.indices["train"], size=size)
            records = replay.gather(indices)
            outputs = {}
            for name, function in (("scalar", scalar_decode), ("vector", decode)) if repeat % 2 else (
                    ("vector", decode), ("scalar", scalar_decode)):
                elapsed, result = timed(lambda: function(records))
                outputs[name] = result
                if repeat >= 2: (scalar if name == "scalar" else vector).append(elapsed)
            for key in outputs["scalar"]:
                np.testing.assert_array_equal(outputs["scalar"][key], outputs["vector"][key])
            elapsed, _ = timed(lambda: replay.sample(size, rng, augment=True))
            if repeat >= 2: augmented.append(elapsed)
        row = {"batch": size}
        for name, timings in (("scalar", scalar), ("vector", vector), ("sample_augment", augmented)):
            row[name] = {"median_us": statistics.median(timings), "p95_us": sorted(timings)[28],
                         "samples_per_second": size * 1e6 / statistics.median(timings)}
        row["decode_speedup"] = statistics.median(scalar) / statistics.median(vector)
        batches.append(row)
    summary = {"engine_sha256": sha256(args.engine), "model_sha256": sha256(args.model),
               "games_per_pair": args.games, "simulations": args.simulations, "pairs": pairs,
               "median_cache_speedup_percent": statistics.median(p["speedup_percent"] for p in pairs),
               "replay_batches": batches}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
