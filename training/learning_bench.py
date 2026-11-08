"""Alternating end-to-end cohort calibration at learning budgets; includes startup and replay I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

from learning_run import python_command
from replay import manifest, sha256, shards
from selfplay import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "verifier", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("ort-cpu", "torch-cpu", "torch-mps"), default="torch-mps")
    parser.add_argument("--workers", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--simulations", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if (args.rounds < 3 or args.games < 1 or len(set(args.workers)) != len(args.workers)
            or any(not 1 <= n <= 128 or args.games % n for n in args.workers)
            or any(not 1 <= n <= 1000000 for n in args.simulations)):
        parser.error("need >=3 rounds, unique widths and whole-cohort games")
    args.output.mkdir(parents=True, exist_ok=False)
    variants = [(budget, width) for budget in args.simulations for width in args.workers]
    rows, identities = [], {}
    for index in range(args.rounds):
        row = {}
        for budget, width in (variants if index % 2 == 0 else list(reversed(variants))):
            key = f"sim-{budget}-workers-{width}"
            output = args.output / f"round-{index}-{key}"
            command = python_command("batched_selfplay.py", "--worker", args.worker.resolve(),
                "--verifier", args.verifier.resolve(), "--model", args.model.resolve(),
                "--checkpoint", args.checkpoint.resolve(), "--backend", args.backend, "--workers", width,
                "--games", args.games, "--simulations", budget, "--seed", 20260910, "--output", output)
            log = output.with_suffix(".log")
            if platform.system() == "Darwin": command = ["/usr/bin/time", "-l", *command]
            with log.open("xb") as stream:
                started = time.perf_counter_ns()
                subprocess.run([str(v) for v in command], stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
                seconds = (time.perf_counter_ns() - started) / 1e9
            entries = shards(output, manifest(output))
            if sum(info["games"] for _, info in entries) != args.games: raise ValueError("incomplete benchmark")
            fingerprint = hashlib.sha256()
            for path, _ in entries: fingerprint.update(path.read_bytes())
            identity = fingerprint.hexdigest()
            if key in identities and identities[key] != identity: raise ValueError("changed replay across rounds")
            identities[key] = identity
            row[key] = {"seconds": seconds, "games_per_hour": args.games * 3600 / seconds,
                        "samples": sum(info["samples"] for _, info in entries),
                        "dataset_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
            if platform.system() == "Darwin":
                match = re.search(r"(\d+)\s+maximum resident set size", log.read_text())
                if not match: raise ValueError("missing macOS time RSS measurement")
                row[key]["owner_peak_rss_mib"] = int(match[1]) / 1048576
            print(json.dumps({"round": index, "variant": key, **row[key]}), flush=True)
        rows.append(row)
    base = args.workers[0]
    result = {"schema": "islay-learning-calibration-v1", "backend": args.backend, "games": args.games,
              "rounds": rows, "replay_sha256": identities,
              "inputs": {n: sha256(getattr(args, n)) for n in ("worker", "verifier", "model", "checkpoint")},
              "runtime": manifest(output)["runtime"], "platform": platform.platform(),
              "code": {n: sha256(Path(__file__).with_name(n)) for n in
                       ("learning_bench.py", "batched_selfplay.py", "batch_inference.py", "batch_workers.py")},
              "scope": "full child CLI wall time: imports, load/parity, workers, verification, fsync and exit; no strength claim",
              "median_games_per_hour": {key: statistics.median(row[key]["games_per_hour"] for row in rows)
                                         for key in identities},
              "median_paired_speedup": {f"sim-{budget}-workers-{width}": statistics.median(
                  row[f"sim-{budget}-workers-{base}"]["seconds"] / row[f"sim-{budget}-workers-{width}"]["seconds"]
                  for row in rows) for budget in args.simulations for width in args.workers}}
    write_json(args.output / "results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
