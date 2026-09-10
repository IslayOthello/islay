"""Alternating paired full self-play workloads; no strength inference from throughput."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from batch_inference import Backend
from batch_workers import Workers
from replay import RECORD_BYTES, sha256
from selfplay import FRAME, config_args, verify_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "verifier", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.games % 32 or args.games < 32 or args.rounds < 3: parser.error("need whole 32-game cohorts, >=3 rounds")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"seed": 20260909, "simulations": args.simulations, "tree_mib": 1, "cache_mib": 1,
              "epsilon": 0.25, "alpha": 0.3, "temperature_plies": 16, "timeout_ms": 30000}
    variants = [("native", 1), ("ort-cpu", 8), ("torch-mps", 8), ("torch-mps", 16), ("torch-mps", 32)]
    backends = {(kind, width): Backend(args.checkpoint, args.model, kind, width)
                for kind, width in variants if kind != "native"}
    signatures, rounds = {}, []
    for round_index in range(args.rounds + 1):
        row = {}
        for kind, width in (variants if round_index % 2 == 0 else list(reversed(variants))):
            name = f"{kind}-{width}"
            frames = []
            started = time.perf_counter_ns()
            if kind == "native":
                process = subprocess.run([str(args.verifier), "--model", str(args.model), "--games", str(args.games),
                                          "--cache-mib", "1", *config_args(config)], capture_output=True, timeout=300, check=True)
                elapsed = (time.perf_counter_ns() - started) / 1000
                position = 0
                for expected in range(args.games):
                    magic, game, _, length, flags = FRAME.unpack_from(process.stdout, position)
                    assert magic == b"ISLAGM01" and game == expected and not flags
                    position += FRAME.size
                    frames.append(process.stdout[position:position + length * RECORD_BYTES])
                    position += length * RECORD_BYTES
                assert position == len(process.stdout)
                metrics = json.loads(process.stderr)
            else:
                with Workers(args.worker.resolve(), backends[kind, width], {**config, "workers": width}) as workers:
                    for start in range(0, args.games, width): frames.extend(workers.cohort(start))
                    metrics = dict(workers.metrics)
                elapsed = (time.perf_counter_ns() - started) / 1000
            data = b"".join(frames)
            identity = hashlib.sha256(data).hexdigest()
            if name in signatures: assert signatures[name] == identity, name + " changed trajectory between rounds"
            signatures[name] = identity
            path = args.output / f"{name}-round-{round_index}.bin"
            path.write_bytes(data)
            verify_shard(args.verifier.resolve(), path, config, 0, args.games)
            row[name] = {"seconds": elapsed / 1e6, "games_per_hour": args.games * 3600e6 / elapsed,
                         "samples": len(data) // RECORD_BYTES, "metrics": metrics}
            print(f"round {round_index} {name}: {row[name]['games_per_hour']:.1f} games/hour", flush=True)
        if round_index: rounds.append(row)
    result = {"config": config, "games": args.games, "rounds": rounds, "replay_hashes": signatures,
              "engine_sha256": sha256(args.verifier), "worker_sha256": sha256(args.worker),
              "model_sha256": sha256(args.model), "checkpoint_sha256": sha256(args.checkpoint),
              "median_paired_speedups": {name: statistics.median(row["native-1"]["seconds"] / row[name]["seconds"]
                                          for row in rounds) for name in signatures}}
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median_paired_speedups"], indent=2))


if __name__ == "__main__": main()
