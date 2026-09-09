#!/usr/bin/env python3
"""Paired, single-process-at-a-time PUCT benchmark on 18 legal positions.

Pass two separately built mcts_bench binaries; see MCTS_BENCHMARK.md.
The evaluator is synthetic, not a neural network or playing-strength test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
from perft_study import START, scalar_moves, play


def corpus():
    rng = random.Random(20260909)
    positions = [{"id": "start", "p": START[0], "o": START[1]}]
    for game in range(4):
        p, o = START
        for ply in range(53):
            if ply in (8, 20, 40, 52):
                positions.append({"id": f"g{game}-p{ply}", "p": p, "o": o})
            moves = scalar_moves(p, o)
            if not moves:
                if not scalar_moves(o, p):
                    raise RuntimeError("corpus game ended before all snapshots")
                p, o = o, p
            else:
                p, o = play(p, o, rng.choice(moves))
    positions.append({"id": "forced-pass", "p": 2455651727219126272, "o": 576743339727456007})
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--milliseconds", type=float, default=150)
    parser.add_argument("--simulations", type=int, default=4096)
    args = parser.parse_args()
    if args.rounds < 2 or args.milliseconds <= 0 or args.simulations < 1:
        parser.error("need rounds >= 2, milliseconds > 0 and simulations >= 1")
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    positions = corpus()
    metadata = {"corpus": positions, "rounds": args.rounds, "milliseconds": args.milliseconds,
                "simulations": args.simulations, "modes": ["uniform", "synthetic"],
                "binaries": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for k, p in binaries.items()}}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for binary in binaries.values():
        subprocess.run([str(binary), "--verify"], check=True)
    rng = random.Random(9012026)
    ratios = {mode: [] for mode in metadata["modes"]}
    with (args.output / "results.jsonl").open("x") as output:
        for round_id in range(args.rounds):
            workloads = [(position, mode) for position in positions for mode in metadata["modes"]]
            rng.shuffle(workloads)
            gains = {mode: [] for mode in metadata["modes"]}
            for position, mode in workloads:
                order = ["baseline", "candidate"]
                if round_id % 2:
                    order.reverse()
                pair = {}
                for name in order:
                    command = [str(binaries[name]), str(position["p"]), str(position["o"]),
                               str(args.simulations), str(args.milliseconds), mode]
                    result = json.loads(subprocess.check_output(command, text=True))
                    pair[name] = result
                    output.write(json.dumps({"round": round_id, "position": position["id"],
                                             "mode": mode, "variant": name, **result}) + "\n")
                    output.flush()
                if pair["baseline"]["signature"] != pair["candidate"]["signature"]:
                    raise RuntimeError(f"tree mismatch: {position['id']} {mode}")
                gains[mode].append(pair["candidate"]["nps"] / pair["baseline"]["nps"])
            for mode in metadata["modes"]:
                ratios[mode].extend(gains[mode])
            print(f"round {round_id + 1}: " + ", ".join(
                f"{mode} {100 * (statistics.median(gains[mode]) - 1):+.2f}%"
                for mode in metadata["modes"]), flush=True)
    summary = {mode: {"paired_median_gain_percent": 100 * (statistics.median(values) - 1),
                      "pairs": len(values)} for mode, values in ratios.items()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
