"""Shared-model batched self-play, with fixed cohorts and P4-compatible atomic replay shards."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from batch_inference import Backend
from batch_workers import SLOT_BYTES, Workers
from replay import RECORD_BYTES, SCHEMA, manifest, sha256, shards
from selfplay import sync_directory, verify_shard, write_json


def run(args):
    config = {k: getattr(args, k) for k in ("workers", "backend", "seed", "simulations", "tree_mib", "cache_mib",
                                           "epsilon", "alpha", "temperature_plies", "timeout_ms")}
    if (sys.byteorder != "little" or not 1 <= args.workers <= 128 or not 0 < args.games <= 1000000000
            or args.games % args.workers or not 1 <= args.simulations <= 1000000
            or not 0 <= args.seed < 1 << 64 or not 1 <= args.tree_mib <= 4096 or not 0 <= args.cache_mib <= 4096
            or not 0 <= args.epsilon <= 1 or not 0.01 <= args.alpha <= 100
            or not 0 <= args.temperature_plies <= 128 or not 100 <= args.timeout_ms <= 120000 or args.max_mib < 1
            or args.workers * (args.tree_mib + args.cache_mib) + args.workers * SLOT_BYTES / 1048576 > args.worker_mib):
        raise ValueError("invalid batch configuration; games must be whole cohorts and worker memory bounded")
    for path, limit in ((args.model, 64), (args.checkpoint, 32)):
        if not path.is_file() or not 0 < path.stat().st_size <= limit * 1048576: raise ValueError("invalid model asset size")
    source_hashes = {path: sha256(path) for path in (args.model, args.checkpoint, args.worker, args.verifier)}
    if args.model.stat().st_size + args.checkpoint.stat().st_size + 65536 > args.max_mib * 1048576:
        raise ValueError("model snapshots exceed storage cap")
    identity = json.loads(subprocess.check_output([str(args.worker.resolve()), "--identity"], text=True, timeout=20))
    if identity != {"protocol": "islay-shm-v1", "slot_bytes": SLOT_BYTES}: raise ValueError("worker protocol mismatch")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Validate model/ONNX identity and numeric parity before generating any training data.
        backend = Backend(args.checkpoint, args.model, args.backend, args.workers)
        if any(sha256(path) != expected for path, expected in source_hashes.items()):
            raise ValueError("generation inputs changed while loading model")
        directory = Path(__file__).parent
        meta = {"schema": SCHEMA, "record_bytes": RECORD_BYTES, "rule": "Othello", "encoding": "relative-2x8x8-a1-v1",
                "architecture": "b8c64-v1", "policy": "visits-tau1-prefix-tau0-lowest-v1",
                "rng": "splitmix64-boxmuller-marsaglia-v1", "split": "mix64-game-seed-mod10-v1", "c_puct": 1.5,
                **config, "engine_sha256": source_hashes[args.worker], "verifier_sha256": source_hashes[args.verifier],
                "model_sha256": source_hashes[args.model], "checkpoint_sha256": source_hashes[args.checkpoint],
                "runtime": backend.identity, "platform": platform.platform(), "python": sys.version,
                "batch_policy": "fixed-padded-cohort-v1-cache-reset-per-game",
                "code": {name: sha256(directory / name) for name in ("batched_selfplay.py", "batch_workers.py",
                         "batch_inference.py", "shm_region.py", "model.py", "export.py", "selfplay.py", "replay.py")}}
        dataset = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        if (root / "manifest.json").exists():
            if manifest(root) != meta: raise ValueError("resume requires identical model, backend, cohort width, config and code")
        else:
            if any(p.name != ".lock" for p in root.iterdir()): raise ValueError("incomplete initialization; use a fresh directory")
            reserve = args.model.stat().st_size + args.checkpoint.stat().st_size + 65536
            if reserve > args.max_mib * 1048576: raise ValueError("model snapshots exceed storage cap")
            with tempfile.TemporaryDirectory(prefix=".init-", dir=root) as temporary:
                staging = Path(temporary)
                for source, name in ((args.model, "model.onnx"), (args.checkpoint, "checkpoint.pt")):
                    shutil.copyfile(source, staging / name)
                    if sha256(staging / name) != source_hashes[source]: raise ValueError("model changed during snapshot")
                    with (staging / name).open("rb") as file: os.fsync(file.fileno())
                write_json(staging / "manifest.json", meta)
                for name in ("model.onnx", "checkpoint.pt", "manifest.json"):
                    os.rename(staging / name, root / name)
                sync_directory(root)

        def check_assets():
            for source, expected in source_hashes.items():
                if sha256(source) != expected: raise ValueError("generation input changed during run")
            if sha256(root / "model.onnx") != meta["model_sha256"] or sha256(root / "checkpoint.pt") != meta["checkpoint_sha256"]:
                raise ValueError("modified replay model snapshot")

        check_assets()
        existing = shards(root, meta)
        completed = sum(info["games"] for _, info in existing)
        for path, info in existing:
            if info["games"] != args.workers or info["start"] % args.workers: raise ValueError("non-cohort replay shard")
            verify_shard(args.verifier.resolve(), path, meta, info["start"], info["games"])
        if completed >= args.games: return {"new_games": 0, "total_games": completed, "startup": backend.startup}
        used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        started = time.perf_counter()
        initial = completed
        with Workers(args.worker.resolve(), backend, config) as workers:
            while completed < args.games:
                reserve = args.workers * 128 * RECORD_BYTES + 65536
                if used + reserve > args.max_mib * 1048576: raise ValueError("replay byte cap reached; nothing evicted")
                frames = workers.cohort(completed)
                with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
                    staging = Path(temporary)
                    with (staging / "samples.bin").open("xb") as file:
                        for frame in frames: file.write(frame)
                        file.flush()
                        os.fsync(file.fileno())
                    lengths = [len(frame) // RECORD_BYTES for frame in frames]
                    checked = verify_shard(args.verifier.resolve(), staging / "samples.bin", meta, completed, args.workers)
                    if checked["samples"] != sum(lengths): raise ValueError("shared frame sample mismatch")
                    check_assets()
                    info = {"dataset": dataset, "start": completed, "games": args.workers, "samples": sum(lengths),
                            "lengths": lengths, "sha256": sha256(staging / "samples.bin")}
                    write_json(staging / "index.json", info)
                    sync_directory(staging)
                    destination = root / f"shard-{completed:012d}"
                    if destination.exists(): raise ValueError("refusing to overwrite a committed cohort")
                    size = sum(p.stat().st_size for p in staging.iterdir())
                    if size > reserve: raise ValueError("cohort exceeds storage reserve")
                    os.rename(staging, destination)
                    sync_directory(root)
                    used += size
                completed += args.workers
                print(json.dumps({"committed_games": completed, "target_games": args.games}), flush=True)
            metrics = dict(workers.metrics)
        metrics.update(new_games=completed - initial, total_games=completed, dataset_bytes=used,
                       seconds=time.perf_counter() - started, startup=backend.startup)
        metrics["games_per_hour"] = metrics["new_games"] * 3600 / metrics["seconds"]
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker", "verifier", "model", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("ort-cpu", "torch-cpu", "torch-mps"), default="ort-cpu")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games", type=int, default=128, help="total whole-cohort target, including committed games")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--tree-mib", type=int, default=1)
    parser.add_argument("--cache-mib", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--temperature-plies", type=int, default=16)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--worker-mib", type=int, default=512, help="sum of tree/cache caps and shm, excludes model runtime/OS")
    parser.add_argument("--max-mib", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__": main()
